; RUN: llvm-as %s -o %t.o
; RUN: llvm-as %p/Inputs/available-externally.ll -o %t2.o

; RUN: %gold -plugin %llvmshlibdir/LLVMgold.so \
; RUN:    --plugin-opt=emit-llvm \
; RUN:    -shared %t.o %t2.o -o %t3.o
; RUN: llvm-dis %t3.o -o - | FileCheck %s

; RUN: %gold -plugin %llvmshlibdir/LLVMgold.so \
; RUN:    --plugin-opt=emit-llvm \
; RUN:    -shared %t2.o %t.o -o %t3.o
; RUN: llvm-dis %t3.o -o - | FileCheck %s

define void @foo() {
  call void @bar()
  call void @zed()
  ret void
}
define available_externally void @bar() {
  ret void
}
define available_externally void @zed() {
  ret void
}

; CHECK-DAG: define available_externally void @bar() {
; CHECK-DAG: define void @zed() {
