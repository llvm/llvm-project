; RUN: llvm-link %s -S -o - | FileCheck %s
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9"

; CHECK-DAG: @gv0 = constant ptr @alias
; CHECK-DAG: @gv1 = constant i64 ptrtoint (ptr @gv1 to i64)
; CHECK-DAG: @alias = alias i64, ptr @gv1

@gv0 = constant ptr @alias
@gv1 = constant i64 ptrtoint (ptr @gv1 to i64)

@alias = alias i64, ptr @gv1
