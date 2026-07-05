; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

define void @okay(ptr %p, i32 %x) {
	call void asm "addl $1, $0", "=*rm,r"(ptr elementtype(i32) %p, i32 %x)
  ret void
}

; CHECK: Operand for indirect constraint must have pointer type
; CHECK-NEXT: call void asm "addl $1, $0", "=*rm,r"(i32 %p, i32 %x)
define void @not_pointer_arg(i32 %p, i32 %x) {
	call void asm "addl $1, $0", "=*rm,r"(i32 %p, i32 %x)
  ret void
}

; CHECK: Elementtype attribute can only be applied for indirect constraints
; CHECK-NEXT: call void asm "addl $1, $0", "=*rm,r"(ptr elementtype(i32) %p, ptr elementtype(i32) %x)
define void @not_indirect(ptr %p, ptr %x) {
	call void asm "addl $1, $0", "=*rm,r"(ptr elementtype(i32) %p, ptr elementtype(i32) %x)
  ret void
}

; CHECK: Operand for indirect constraint must have elementtype attribute
; CHECK-NEXT: call void asm "addl $1, $0", "=*rm,r"(ptr %p, i32 %x)
define void @missing_elementtype(ptr %p, i32 %x) {
	call void asm "addl $1, $0", "=*rm,r"(ptr %p, i32 %x)
  ret void
}

; CHECK: Operand for indirect constraint must have pointer type
; CHECK-NEXT: invoke void asm "addl $1, $0", "=*rm,r"(i32 %p, i32 %x)
define void @not_pointer_arg_invoke(i32 %p, i32 %x) personality ptr null {
	invoke void asm "addl $1, $0", "=*rm,r"(i32 %p, i32 %x)
      to label %cont unwind label %lpad

lpad:
  %lp = landingpad i32
      cleanup
  ret void

cont:
  ret void
}

; CHECK: Operand for indirect constraint must have pointer type
; CHECK-NEXT: callbr void asm "addl $1, $0", "=*rm,r"(i32 %p, i32 %x)
define void @not_pointer_arg_callbr(i32 %p, i32 %x) {
	callbr void asm "addl $1, $0", "=*rm,r"(i32 %p, i32 %x)
      to label %cont []

cont:
  ret void
}
