; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=ir-passes --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefixes=CHECK-FINAL --input-file=%t %s

; CHECK-INTERESTINGNESS-LABEL: @f1
; CHECK-INTERESTINGNESS: add
; CHECK-INTERESTINGNESS-LABEL: @f2
; CHECK-INTERESTINGNESS: add
; CHECK-INTERESTINGNESS: add

; CHECK-FINAL-LABEL: @f1
; CHECK-FINAL: add i32 %a, 10
; CHECK-FINAL-LABEL: @f2
; CHECK-FINAL: add i32 %a, 5
; CHECK-FINAL: add i32 %b, 5

define i32 @f1(i32 %a) {
  %b = add i32 %a, 5
  %c = add i32 %b, 5
  ret i32 %c
}

define i32 @f2(i32 %a) {
  %b = add i32 %a, 5
  %c = add i32 %b, 5
  ret i32 %c
}

declare void @f3()