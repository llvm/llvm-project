; RUN: llc < %s -march=x86-64 | FileCheck %s

; CHECK-NOT: imul

define i64 @t1(i64 %a) nounwind readnone {
entry:
  %0 = mul i64 %a, 81
; CHECK: lea
; CHECK: lea
  ret i64 %0
}

define i64 @t2(i64 %a) nounwind readnone {
entry:
  %0 = mul i64 %a, 40
; CHECK: shl
; CHECK: lea
  ret i64 %0
}
