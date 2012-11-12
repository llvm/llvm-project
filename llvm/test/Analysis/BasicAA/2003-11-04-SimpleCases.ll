; This testcase consists of alias relations which should be completely
; resolvable by basicaa.

; RUN: opt < %s -basicaa -aa-eval -print-may-aliases -disable-output 2>&1 | FileCheck %s

%T = type { i32, [10 x i8] }

; CHECK:     Function: test
; CHECK-NOT:   May:

define void @test(%T* %P) {
  %A = getelementptr %T* %P, i64 0
  %B = getelementptr %T* %P, i64 0, i32 0
  %C = getelementptr %T* %P, i64 0, i32 1
  %D = getelementptr %T* %P, i64 0, i32 1, i64 0
  %E = getelementptr %T* %P, i64 0, i32 1, i64 5
  ret void
}
