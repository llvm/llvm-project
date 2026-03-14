; RUN: llvm-as %s -o %t.o
; RUN: %gold -plugin %llvmshlibdir/LLVMgold%shlibext -plugin-opt=emit-llvm \
; RUN:    -shared %t.o -o %t2.o
; RUN: llvm-dis %t2.o -o - | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
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
