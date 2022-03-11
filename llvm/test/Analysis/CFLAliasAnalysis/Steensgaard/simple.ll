; This testcase consists of alias relations which should be completely
; resolvable by cfl-steens-aa (derived from BasicAA/2003-11-04-SimpleCases.ll).

; RUN: opt < %s -aa-pipeline=cfl-steens-aa -passes=aa-eval -print-may-aliases -disable-output 2>&1 | FileCheck %s

%T = type { i32, [10 x i8] }

; CHECK:     Function: test
; CHECK-NOT:   May:

define void @test(%T* %P) {
  %A = getelementptr %T, %T* %P, i64 0
  %B = getelementptr %T, %T* %P, i64 0, i32 0
  %C = getelementptr %T, %T* %P, i64 0, i32 1
  %D = getelementptr %T, %T* %P, i64 0, i32 1, i64 0
  %E = getelementptr %T, %T* %P, i64 0, i32 1, i64 5
  ret void
}
