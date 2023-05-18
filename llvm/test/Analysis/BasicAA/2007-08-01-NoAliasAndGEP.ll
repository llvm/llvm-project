; RUN: opt < %s -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; CHECK: Function: foo
; CHECK:   MayAlias: i32* %Ipointer, i32* %Jpointer
; CHECK: 9 no alias responses
; CHECK: 6 may alias responses

define void @foo(ptr noalias %p, ptr noalias %q, i32 %i, i32 %j) {
  %Ipointer = getelementptr i32, ptr %p, i32 %i
  %qi = getelementptr i32, ptr %q, i32 %i
  %Jpointer = getelementptr i32, ptr %p, i32 %j
  %qj = getelementptr i32, ptr %q, i32 %j
  store i32 0, ptr %p
  store i32 0, ptr %Ipointer
  store i32 0, ptr %Jpointer
  store i32 0, ptr %q
  store i32 0, ptr %qi
  store i32 0, ptr %qj
  ret void
}
