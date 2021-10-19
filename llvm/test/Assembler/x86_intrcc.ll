; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; Make sure no arguments is accepted
; CHECK: define x86_intrcc void @no_args() {
define x86_intrcc void @no_args() {
  ret void
}

; CHECK: define x86_intrcc void @byval_arg(i32* byval(i32) %0) {
define x86_intrcc void @byval_arg(i32* byval(i32)) {
  ret void
}
