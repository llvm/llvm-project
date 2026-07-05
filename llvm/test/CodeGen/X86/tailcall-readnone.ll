; RUN: llc -mtriple=x86_64-unknown-linux-gnu -o - %s | FileCheck %s

define void @f(ptr %p) unnamed_addr {
entry:
  %v = tail call ptr @g()
  store ptr %v, ptr %p, align 8
  ret void
}
; CHECK-LABEL: f:
; CHECK: callq g
; CHECK: movq    %rax, (%rbx)

declare ptr @g() #2

attributes #2 = { nounwind readnone }
