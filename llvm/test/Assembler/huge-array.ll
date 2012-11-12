; RUN: llvm-as < %s | llvm-dis | FileCheck %s

; CHECK: define [18446744073709551615 x i8]* @foo() {
; CHECK: ret [18446744073709551615 x i8]* null
define [18446744073709551615 x i8]* @foo() {
  ret [18446744073709551615 x i8]* null
}
