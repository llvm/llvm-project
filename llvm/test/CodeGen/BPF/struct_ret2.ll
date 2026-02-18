; RUN: llc -mtriple=bpf < %s | FileCheck %s

; Function Attrs: nounwind uwtable
define { i64, i32 } @foo(i32 %a, i32 %b, i32 %c) #0 {
; CHECK-LABEL: foo:
; CHECK: w4 = 1
; CHECK-NEXT: w5 = 2
entry:
  %call = tail call { i64, i32 } @bar(i32 %a, i32 %b, i32 %c, i32 1, i32 2) #3
  ret { i64, i32 } %call
}

declare { i64, i32 } @bar(i32, i32, i32, i32, i32) #1
