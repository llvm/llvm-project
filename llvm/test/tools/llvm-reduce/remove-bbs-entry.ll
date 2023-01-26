; Test that llvm-reduce does not remove the entry block of functions.
;
; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=basic-blocks --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck %s < %t

; CHECK-INTERESTINGNESS: foo

; CHECK: add i32
define i32 @foo() {
uninteresting:
  %a = add i32 0, 0
  ret i32 0
}

