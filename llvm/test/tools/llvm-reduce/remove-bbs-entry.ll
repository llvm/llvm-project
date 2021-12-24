; Test that llvm-reduce correctly removes the entry block of functions for
; linkages other than external and weak.
;
; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=basic-blocks --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: cat %t | FileCheck %s

; CHECK-INTERESTINGNESS: interesting1:

; CHECK-NOT: uninteresting
define linkonce_odr i32 @foo() {
uninteresting:
  ret i32 0
}

define i32 @main(i1 %c) {
interesting1:
  ret i32 0
}
