; RUN: llvm-as %s -o %t.o
; RUN: llvm-as %p/Inputs/linkonce-weak.ll -o %t2.o

; RUN: %gold -plugin %llvmshlibdir/LLVMgold.so \
; RUN:    --plugin-opt=emit-llvm \
; RUN:    -shared %t.o %t2.o -o %t3.o
; RUN: llvm-dis %t3.o -o - | FileCheck %s

; RUN: %gold -plugin %llvmshlibdir/LLVMgold.so \
; RUN:    --plugin-opt=emit-llvm \
; RUN:    -shared %t2.o %t.o -o %t3.o
; RUN: llvm-dis %t3.o -o - | FileCheck %s

define linkonce_odr void @f() {
  ret void
}

; Test that we get a weak_odr regardless of the order of the files
; CHECK: define weak_odr void @f() {
