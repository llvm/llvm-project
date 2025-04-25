; RUN: llc < %s | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

; The callee address computation should get folded into the call.
; CHECK-LABEL: f:
; CHECK-NOT: mov
; CHECK: jmpq *(%rdi,%rsi,8)

define void @f(ptr %table, i64 %idx) {
entry:
  %arrayidx = getelementptr inbounds ptr, ptr %table, i64 %idx
  %funcptr = load ptr, ptr %arrayidx, align 8
  tail call void %funcptr(ptr %table, i64 %idx)
  ret void
}
