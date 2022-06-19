; This testcase ensures that CFL AA gives conservative answers on variables
; that involve arguments.

; RUN: opt < %s -aa-pipeline=cfl-steens-aa -passes=aa-eval -print-may-aliases -disable-output 2>&1 | FileCheck %s

; CHECK:     Function: test

define void @test(i1 %c, i32* %arg1, i32* %arg2) {
  ; CHECK: 6 Total Alias Queries Performed
  ; CHECK: 3 no alias responses
  %a = alloca i32, align 4
  %b = select i1 %c, i32* %arg1, i32* %arg2
  load i32, i32* %a
  load i32, i32* %b
  load i32, i32* %arg1
  load i32, i32* %arg2

  ret void
}
