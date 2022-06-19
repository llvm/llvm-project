; This testcase ensures that CFL AA gives conservative answers on variables
; that involve arguments.
; (Everything should alias everything, because args can alias globals, so the
; aliasing sets should of args+alloca+global should be combined)

; RUN: opt < %s -aa-pipeline=cfl-steens-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

; CHECK:     Function: test

@g = external global i32

; CHECK: MayAlias:	i32* %arg1, i32* %arg2
; CHECK: NoAlias:	i32* %A, i32* %arg1
; CHECK: NoAlias:	i32* %A, i32* %arg2
; CHECK: MayAlias:	i32* %arg1, i32* @g
; CHECK: MayAlias:	i32* %arg2, i32* @g
; CHECK: MayAlias:	i32* %A, i32* @g
define void @test(i1 %c, i32* %arg1, i32* %arg2) {
  %A = alloca i32
  load i32, i32* %arg1
  load i32, i32* %arg2
  load i32, i32* %A
  load i32, i32* @g

  ret void
}
