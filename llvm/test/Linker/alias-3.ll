; RUN: llvm-link %s -S -o - | FileCheck %s
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9"

; CHECK-DAG: @A = internal constant i8 1
; CHECK-DAG: @B = alias i8, ptr @A
; CHECK-DAG: @C = global [2 x ptr] [ptr @A, ptr @B]

@A = internal constant i8 1
@B = alias i8, ptr @A
@C = global [2 x ptr] [ptr @A, ptr @B]


