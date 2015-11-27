; RUN: llvm-as %s -o %t.o
; RUN: %gold -plugin %llvmshlibdir/LLVMgold.so \
; RUN:    --plugin-opt=emit-llvm \
; RUN:    -shared %t.o -o %t2.o
; RUN: llvm-dis %t2.o -o - | FileCheck %s

@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @foo, i8* null }]

define internal void @foo() {
  ret void
}

; CHECK: @llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @foo, i8* null }]
