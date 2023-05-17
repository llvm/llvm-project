; RUN: llvm-as %s -o %t.o

; RUN: not %gold -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    --plugin-opt=emit-llvm \
; RUN:    -shared %t.o -o %t2.o 2>&1 | FileCheck %s

; CHECK: Unable to determine comdat of alias!

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@g1 = global i32 1
@g2 = global i32 2

@a = alias i32, inttoptr(i32 sub (i32 ptrtoint (ptr @g1 to i32),
                                  i32 ptrtoint (ptr @g2 to i32)) to ptr)
