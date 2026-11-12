; Test that inttoptr, add and ptrtoint don't cause problems in alias summaries.
; RUN: opt -module-summary %s -o - | llvm-dis | FileCheck %s

; CHECK: ^1 = gv: (name: "a", {{.*}} aliasee: ^2
; CHECK: ^2 = gv: (name: "b",

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@a = alias i32, i32* inttoptr (i64 add (i64 ptrtoint (i32* @b to i64), i64 1297036692682702848) to i32*)
@b = global i32 1
