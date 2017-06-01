; This testcase ensures that CFL AA gives conservative answers on variables
; that involve arguments.
; (Everything should alias everything, because args can alias globals, so the
; aliasing sets should of args+alloca+global should be combined)

; RUN: opt < %s -disable-basicaa -cfl-steens-aa -aa-eval -print-may-aliases -disable-output 2>&1 | FileCheck %s

; CHECK:     Function: test

@g = external global i32

define void @test(i1 %c, i32* %arg1, i32* %arg2) {
  ; CHECK: 15 Total Alias Queries Performed
  ; CHECK: 0 no alias responses
  %A = alloca i32, align 4
  %B = select i1 %c, i32* %arg1, i32* %arg2
  %C = select i1 %c, i32* @g, i32* %A

  ret void
}
