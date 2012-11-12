; Test that linking two files with the same definition causes an error and
; that error is printed out.
; RUN: llvm-as %s -o %t.one.bc
; RUN: llvm-as %s -o %t.two.bc
; RUN: not llvm-link %t.one.bc %t.two.bc -o %t.bc 2>&1 | FileCheck %s

; CHECK: symbol multiply defined
define i32 @bar() {
  ret i32 0
}
