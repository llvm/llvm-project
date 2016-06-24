; This testcase ensures that CFL AA answers queries soundly when callee tries 
; to mutate the memory pointed to by its parameters

; RUN: opt < %s -disable-basicaa -cfl-aa -aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s
; RUN: opt < %s -aa-pipeline=cfl-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

@g = external global i32

define void @store_arg_unknown_callee(i32** %arg1) {
	store i32* @g, i32** %arg1
	ret void
}
; CHECK-LABEL: Function: test_store_arg_unknown
; CHECK: NoAlias: i32* %x, i32** %p
; CHECK: NoAlias: i32* %a, i32** %p
; CHECK: NoAlias: i32* %b, i32** %p
; CHECK: MayAlias: i32* %lp, i32* %x
; CHECK: MayAlias: i32* %a, i32* %lp
; CHECK: NoAlias: i32* %b, i32* %lp
; CHECK: NoAlias: i32* %lp, i32** %p
define void @test_store_arg_unknown(i32* %x) {
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  %p = alloca i32*, align 8

  store i32* %a, i32** %p
  call void @store_arg_unknown_callee(i32** %p)

  %lp = load i32*, i32** %p

  ret void
}