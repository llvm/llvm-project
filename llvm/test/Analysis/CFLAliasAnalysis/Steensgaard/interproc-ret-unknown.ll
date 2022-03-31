; This testcase ensures that CFL AA answers queries soundly when callee tries 
; to return an unknown pointer

; RUN: opt < %s -aa-pipeline=cfl-steens-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

@g = external global i32
define i32* @return_unknown_callee(i32* %arg1, i32* %arg2) {
	ret i32* @g
}
; CHECK-LABEL: Function: test_return_unknown
; CHECK: NoAlias: i32* %a, i32* %b
; CHECK: MayAlias: i32* %c, i32* %x
; CHECK: NoAlias: i32* %a, i32* %c
; CHECK: NoAlias: i32* %b, i32* %c
define void @test_return_unknown(i32* %x) {
  %a = alloca i32, align 4
  %b = alloca i32, align 4

  load i32, i32* %x
  load i32, i32* %a
  load i32, i32* %b
  %c = call i32* @return_unknown_callee(i32* %a, i32* %b)
  load i32, i32* %c

  ret void
}

@g2 = external global i32*
define i32** @return_unknown_callee2() {
	ret i32** @g2
}
; CHECK-LABEL: Function: test_return_unknown2
; CHECK: MayAlias: i32** %a, i32* %x
; CHECK: MayAlias: i32* %b, i32* %x
; CHECK: MayAlias: i32** %a, i32* %b
define void @test_return_unknown2(i32* %x) {
  load i32, i32* %x
  %a = call i32** @return_unknown_callee2()
  %b = load i32*, i32** %a
  load i32, i32* %b

  ret void
}
