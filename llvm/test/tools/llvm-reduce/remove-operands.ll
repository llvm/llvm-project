; Test that llvm-reduce can reduce operands to their default values.
;
; RUN: llvm-reduce --delta-passes=operands --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: cat %t | FileCheck %s

; CHECK-INTERESTINGNESS: ret i32

; CHECK-LABEL: define i32 @main() {
define i32 @main() {

; CHECK-LABEL: lb1:
; CHECK-NEXT: br label %lb2
lb1:
  br label %lb2

; CHECK-LABEL: lb2:
; CHECK-NEXT: ret i32 undef
lb2:
  ret i32 10
}
