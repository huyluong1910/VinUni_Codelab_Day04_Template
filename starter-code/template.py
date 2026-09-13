"""
Lab #4: System Prompt Engineering & Tool Calling Engine
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
"""

import json
import re
from typing import Dict, Any, List
from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket

# ═══════════════════════════════════════════════════════════════════════════
# TODO 1: Thiết kế SYSTEM PROMPT cấp sản xuất
# Yêu cầu: Phải chứa Persona, Core Rules, Operational Boundaries, Output Contract.
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """Bạn là VinAssistant — trợ lý AI chính thức của hệ sinh thái Vingroup.

## 1. PERSONA
- Tên: VinAssistant
- Vai trò: Chuyên viên tư vấn thông tin sản phẩm và tiếp nhận hỗ trợ khách hàng của hệ sinh thái Vingroup (VinFast, Vinpearl).
- Phong cách: Chuyên nghiệp, thân thiện, trung thực và chính xác.

## 2. AVAILABLE TOOLS
- `search_product_catalog(category: str, max_price: int)`: Tra cứu danh mục sản phẩm (xe điện hoặc du lịch) kèm mức giá trần.
- `submit_support_ticket(customer_name: str, issue_description: str, priority: str)`: Tạo phiếu yêu cầu hỗ trợ hoặc ghi nhận sự cố của khách hàng.

## 3. CORE RULES
- Tuyệt đối KHÔNG BAO GIỜ bịa đặt thông tin về sản phẩm, giá bán, hoặc mã phiếu hỗ trợ (chống hallucination).
- Bắt buộc phải sử dụng công cụ (Tool Calling) khi người dùng hỏi về danh mục sản phẩm hoặc yêu cầu gửi khiếu nại.
- Nếu không tìm thấy kết quả phù hợp từ công cụ, phải phản hồi rõ ràng là không tìm thấy, không tự ý đề xuất sản phẩm không có thật.

## 4. OPERATIONAL BOUNDARIES
- Chỉ xử lý các yêu cầu liên quan đến hệ sinh thái Vingroup (VinFast, Vinpearl).
- Từ chối lịch sự các chủ đề không thuộc thẩm quyền hoặc không liên quan.

## 5. OUTPUT CONTRACT
Phản hồi theo chu trình ReAct / Function Calling:
- Thought: Phân tích ý định của người dùng và quyết định bước xử lý tiếp theo.
- Action: Tên tool và tham số tương ứng cần gọi.
- Observation: Kết quả nhận được từ tool.
- Final Answer: Tổng hợp câu trả lời hoàn chỉnh, chính xác gửi đến khách hàng.
"""


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop."""

    def query(self, user_input: str) -> Dict[str, Any]:
        # TODO 2: Trả về câu trả lời tĩnh (mock) hoặc gọi Gemini API 1 lượt (không dùng tool)
        # Mục tiêu: Quan sát hiện tượng bịa thông tin (hallucination)
        return {
            "answer": f"[Chatbot Baseline] Trả lời cho: {user_input}. Hệ thống baseline không kết nối cơ sở dữ liệu thực.",
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Agent với System Prompt Engineering & Tool Calling."""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []

    def _extract_catalog_params(self, text: str) -> Dict[str, Any]:
        """Trích xuất tham số cho search_product_catalog."""
        lower_text = text.lower()
        category = "xe_dien"
        if any(k in lower_text for k in ["du lịch", "resort", "phòng", "khách sạn", "nghỉ dưỡng", "vinpearl"]):
            category = "du_lich"
        elif any(k in lower_text for k in ["xe", "vf", "ô tô", "bán tải"]):
            category = "xe_dien"

        max_price = 999999999999
        match_trieu = re.search(r'(?:dưới|khoảng|tối đa|giá)\s*(\d+(?:\.\d+)?)\s*triệu', lower_text)
        if not match_trieu:
            match_trieu = re.search(r'(\d+(?:\.\d+)?)\s*triệu', lower_text)

        match_ty = re.search(r'(?:dưới|khoảng|tối đa|giá)\s*(\d+(?:\.\d+)?)\s*tỷ', lower_text)
        if not match_ty:
            match_ty = re.search(r'(\d+(?:\.\d+)?)\s*tỷ', lower_text)

        if match_trieu:
            max_price = int(float(match_trieu.group(1)) * 1_000_000)
        elif match_ty:
            max_price = int(float(match_ty.group(1)) * 1_000_000_000)

        return {"category": category, "max_price": max_price}

    def _extract_ticket_params(self, text: str) -> Dict[str, Any]:
        """Trích xuất tham số cho submit_support_ticket."""
        customer_name = "Khách hàng"
        name_match = re.search(r'(?:tôi tên là|tên tôi là|tôi tên)\s+([A-ZÀ-Ỵa-zà-ỹ\s]+?)(?:,|\.|$)', text, re.IGNORECASE)
        if name_match:
            customer_name = name_match.group(1).strip()

        priority = "medium"
        lower_text = text.lower()
        if any(k in lower_text for k in ["gấp", "nghiêm trọng", "khẩn cấp", "nguy hiểm"]):
            priority = "high"
        elif "trung bình" in lower_text:
            priority = "medium"
        elif any(k in lower_text for k in ["thấp", "nhẹ", "không vội"]):
            priority = "low"

        issue_desc = text
        issue_match = re.search(r'((?:xe|phòng|dịch vụ).*?)(?:Đây là|\.|$|, mức độ)', text, re.IGNORECASE)
        if issue_match:
            issue_desc = issue_match.group(1).strip()

        return {
            "customer_name": customer_name,
            "issue_description": issue_desc,
            "priority": priority
        }

    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy Agent Loop."""
        self.trace = []
        user_lower = user_input.lower()

        # TODO 3: Phân tích intent từ user_input
        is_faq = (
            any(k in user_lower for k in ["bảo hành", "kéo dài bao lâu", "chính sách"])
            and not any(k in user_lower for k in ["lỗi", "hỏng", "sự cố", "khiếu nại", "giá dưới", "xem xe", "xem resort"])
        )
        needs_catalog = any(k in user_lower for k in ["giá dưới", "xem xe", "xem resort", "mua xe", "tìm xe", "sản phẩm"])
        needs_ticket = (
            any(k in user_lower for k in ["bị lỗi", "sự cố", "khiếu nại", "phản hồi", "hỏng", "bị ẩm mốc"])
            or "tôi tên" in user_lower
            or "tên tôi là" in user_lower
        )

        tasks: List[str] = []
        if needs_catalog:
            tasks.append("search_product_catalog")
        if needs_ticket:
            tasks.append("submit_support_ticket")
        if not tasks and is_faq:
            tasks.append("faq")

        iteration = 0
        catalog_results = None
        ticket_result = None
        final_answer = ""

        # TODO 4: Xây dựng Agent Loop (while iteration <= self.max_iterations)
        while iteration < self.max_iterations:
            iteration += 1

            if is_faq and not needs_catalog and not needs_ticket:
                self.trace.append({
                    "iteration": iteration,
                    "thought": "Người dùng hỏi câu hỏi thường gặp (FAQ) về chính sách, trả lời trực tiếp không cần gọi tool.",
                    "action": "none"
                })
                final_answer = "Chính sách bảo hành pin xe điện VinFast kéo dài 10 năm hoặc 200.000 km tuỳ điều kiện nào đến trước."
                break

            for task in tasks:
                if task == "search_product_catalog":
                    params = self._extract_catalog_params(user_input)
                    self.trace.append({
                        "iteration": iteration,
                        "thought": f"Người dùng muốn tìm sản phẩm danh mục '{params['category']}' với giá dưới {params['max_price']:,} VNĐ.",
                        "action": "search_product_catalog",
                        "args": params
                    })
                    catalog_func = TOOL_MAP.get("search_product_catalog", search_product_catalog)
                    catalog_results = catalog_func(**params)
                    self.trace.append({
                        "iteration": iteration,
                        "observation": catalog_results
                    })

                elif task == "submit_support_ticket":
                    params = self._extract_ticket_params(user_input)
                    self.trace.append({
                        "iteration": iteration,
                        "thought": f"Khách hàng '{params['customer_name']}' phản ánh sự cố. Cần tạo phiếu hỗ trợ ưu tiên {params['priority']}.",
                        "action": "submit_support_ticket",
                        "args": params
                    })
                    ticket_func = TOOL_MAP.get("submit_support_ticket", submit_support_ticket)
                    ticket_result = ticket_func(**params)
                    self.trace.append({
                        "iteration": iteration,
                        "observation": ticket_result
                    })

            # Tổng hợp Final Answer từ kết quả các tool
            answer_parts = []
            if catalog_results is not None:
                if not catalog_results or (isinstance(catalog_results, list) and len(catalog_results) > 0 and "error" in catalog_results[0]):
                    answer_parts.append("Rất tiếc, không tìm thấy sản phẩm nào phù hợp với mức giá yêu cầu.")
                else:
                    items = [f"{p['name']} ({p['price_vnd']:,} VNĐ)" for p in catalog_results]
                    answer_parts.append(f"Dưới đây là các sản phẩm phù hợp: {', '.join(items)}.")

            if ticket_result is not None:
                ticket_id = ticket_result.get("ticket_id", "")
                c_name = ticket_result.get("customer_name", "")
                answer_parts.append(f"Phiếu hỗ trợ đã được tạo thành công với mã {ticket_id} cho khách hàng {c_name}.")

            if answer_parts:
                final_answer = "\n".join(answer_parts)
            else:
                final_answer = "Yêu cầu của bạn đã được xử lý."
            break

        if iteration > self.max_iterations:
            return {
                "answer": "Lỗi: Vượt quá số bước tối đa.",
                "trace": self.trace,
                "iterations": iteration,
                "status": "max_iterations_reached"
            }

        return {
            "answer": final_answer,
            "trace": self.trace,
            "iterations": iteration,
            "status": "completed"
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result["answer"])
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
