; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu | FileCheck %s

; The callee address computation should get folded into the call.
; CHECK-LABEL: f:
; CHECK-NOT: mov
; CHECK: jmpq *(%rdi,%rsi,8)
define void @f(ptr %table, i64 %idx, i64 %aux1, i64 %aux2, i64 %aux3) {
entry:
  %arrayidx = getelementptr inbounds ptr, ptr %table, i64 %idx
  %funcptr = load ptr, ptr %arrayidx, align 8
  tail call void %funcptr(ptr %table, i64 %idx, i64 %aux1, i64 %aux2, i64 %aux3)
  ret void
}
