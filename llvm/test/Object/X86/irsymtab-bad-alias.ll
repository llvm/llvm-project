; Check that we do not create an irsymtab for modules with malformed IR.

; RUN: opt -o %t %s
; RUN: llvm-bcanalyzer -dump %t | FileCheck %s

; CHECK-NOT: <SYMTAB_BLOCK

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@g1 = global i32 1
@g2 = global i32 2

@a = alias i32, inttoptr(i32 sub (i32 ptrtoint (i32* @g1 to i32),
                                  i32 ptrtoint (i32* @g2 to i32)) to i32*)
