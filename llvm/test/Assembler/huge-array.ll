; RUN: llvm-as --opaque-pointers=0 < %s | llvm-dis --opaque-pointers=0 | FileCheck %s
; RUN: verify-uselistorder --opaque-pointers=0 %s

; CHECK: define [18446744073709551615 x i8]* @foo() {
; CHECK: ret [18446744073709551615 x i8]* null
define [18446744073709551615 x i8]* @foo() {
  ret [18446744073709551615 x i8]* null
}
