; This testcase consists of alias relations which should be completely
; resolvable by basicaa.

; RUN: opt < %s -aa-pipeline=basic-aa -passes=aa-eval -print-may-aliases -disable-output 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

%T = type { i32, [10 x i8] }

; CHECK:     Function: test
; CHECK-NOT:   MayAlias:

define void @test(ptr %P) {
  %C = getelementptr %T, ptr %P, i64 0, i32 1
  %D = getelementptr %T, ptr %P, i64 0, i32 1, i64 0
  %E = getelementptr %T, ptr %P, i64 0, i32 1, i64 5
  ret void
}
