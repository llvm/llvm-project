; RUN: llvm-as %s -o %t.o
; RUN: llvm-as %p/Inputs/weak.ll -o %t2.o

; RUN: %gold -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    --plugin-opt=emit-llvm \
; RUN:    -shared %t.o %t2.o -o %t3.o
; RUN: llvm-dis %t3.o -o - | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@a = weak global i32 42
@b = global i32* @a

; Test that @b and @c end up pointing to the same variable.

; CHECK: @a = weak global i32 42
; CHECK: @b = global i32* @a{{$}}
; CHECK: @c = global i32* @a{{$}}
