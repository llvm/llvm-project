; RUN: opt < %s -basicaa -aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; CHECK: Function: foo
; CHECK:   MayAlias: i32* %Ipointer, i32* %Jpointer
; CHECK: 9 no alias responses
; CHECK: 6 may alias responses

define void @foo(i32* noalias %p, i32* noalias %q, i32 %i, i32 %j) {
  %Ipointer = getelementptr i32, i32* %p, i32 %i
  %qi = getelementptr i32, i32* %q, i32 %i
  %Jpointer = getelementptr i32, i32* %p, i32 %j
  %qj = getelementptr i32, i32* %q, i32 %j
  store i32 0, i32* %p
  store i32 0, i32* %Ipointer
  store i32 0, i32* %Jpointer
  store i32 0, i32* %q
  store i32 0, i32* %qi
  store i32 0, i32* %qj
  ret void
}
