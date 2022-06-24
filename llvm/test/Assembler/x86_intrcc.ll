; RUN: llvm-as --opaque-pointers=0 < %s | llvm-dis --opaque-pointers=0 | FileCheck %s
; RUN: verify-uselistorder --opaque-pointers=0 %s

; Make sure no arguments is accepted
; CHECK: define x86_intrcc void @no_args() {
define x86_intrcc void @no_args() {
  ret void
}

; CHECK: define x86_intrcc void @byval_arg(i32* byval(i32) %0) {
define x86_intrcc void @byval_arg(i32* byval(i32)) {
  ret void
}
