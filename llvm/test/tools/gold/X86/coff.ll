; RUN: llvm-as %s -o %t.o
; RUN: %gold -plugin %llvmshlibdir/LLVMgold.so -plugin-opt=emit-llvm \
; RUN:    -shared %t.o -o %t2.o
; RUN: llvm-dis %t2.o -o - | FileCheck %s


target datalayout = "m:w"

; CHECK: define void @f() {
define void @f() {
  ret void
}

; CHECK: define internal void @g() {
define hidden void @g() {
  ret void
}

; CHECK: define internal void @h() {
define linkonce_odr void @h() local_unnamed_addr {
  ret void
}
