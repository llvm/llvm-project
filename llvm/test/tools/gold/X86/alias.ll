; RUN: llvm-as %s -o %t.o
; RUN: llvm-as %p/Inputs/alias-1.ll -o %t2.o
; RUN: %gold -shared -o %t3.o -plugin %llvmshlibdir/LLVMgold%shlibext %t2.o %t.o \
; RUN:  -plugin-opt=emit-llvm
; RUN: llvm-dis < %t3.o -o - | FileCheck %s

; CHECK-NOT: alias
; CHECK: @a = global i32 42
; CHECK-NEXT: @b = global i32 1
; CHECK-NOT: alias

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@a = weak alias i32, ptr @b
@b = global i32 1
