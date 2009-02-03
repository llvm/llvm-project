; This testcase consists of alias relations which should be completely
; resolvable by basicaa.

; RUN: llvm-as < %s | opt -aa-eval -print-may-aliases -disable-output \
; RUN: |& not grep May:

%T = type { i32, [10 x i8] }

define void @test(%T* %P) {
  %A = getelementptr %T* %P, i64 0
  %B = getelementptr %T* %P, i64 0, i32 0
  %C = getelementptr %T* %P, i64 0, i32 1
  %D = getelementptr %T* %P, i64 0, i32 1, i64 0
  %E = getelementptr %T* %P, i64 0, i32 1, i64 5
  ret void
}
