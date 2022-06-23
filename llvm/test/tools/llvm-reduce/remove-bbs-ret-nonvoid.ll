; Test that llvm-reduce inserts valid return instructions for functions with
; on-void return types.
;
; RUN: llvm-reduce --delta-passes=basic-blocks --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: cat %t | FileCheck %s

; CHECK-INTERESTINGNESS: interesting:
; CHECK-INTERESTINGNESS: interesting2:

define i32 @main(i1 %c) {
; CHECK-LABEL: define i32 @main(i1 %c) {
; CHECK-LABEL: interesting:
; CHECK-NEXT:    br label %interesting2

; CHECK-LABEL: interesting2:
; CHECK-NEXT:    ret i32 0

interesting:
  br label %interesting2

interesting2:
  br i1 true, label %uninteresting1, label %uninteresting

uninteresting:
  br label %uninteresting1

uninteresting1:
  ret i32 10
}
