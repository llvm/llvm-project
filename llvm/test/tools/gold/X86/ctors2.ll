; RUN: llvm-as %s -o %t.o
; RUN: llvm-as %p/Inputs/ctors2.ll -o %t2.o
; RUN: %gold -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    --plugin-opt=emit-llvm \
; RUN:    -shared %t.o %t2.o -o %t3.o
; RUN: llvm-dis %t3.o -o - | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @foo, ptr null }]

define void @foo() {
  ret void
}

; CHECK: @llvm.global_ctors = appending global [2 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @foo, ptr null }, { i32, ptr, ptr } { i32 65535, ptr @bar, ptr null }]
