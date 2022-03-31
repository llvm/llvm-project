; This testcase ensures that CFL AA answers queries soundly when callee tries 
; to mutate the memory pointed to by its parameters

; RUN: opt < %s -aa-pipeline=cfl-steens-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

define void @store_arg_callee(i32** %arg1, i32* %arg2) {
	store i32* %arg2, i32** %arg1
	ret void
}
; CHECK-LABEL: Function: test_store_arg
; CHECK: NoAlias: i32* %a, i32** %p
; CHECK: NoAlias: i32* %b, i32** %p
; CHECK: MayAlias: i32* %a, i32* %lp
; CHECK: MayAlias: i32* %b, i32* %lp
; CHECK: MayAlias: i32* %b, i32* %lq
; CHECK: MayAlias: i32* %lp, i32* %lq

; We could've proven the following facts if the analysis were inclusion-based:
; NoAlias: i32* %a, i32* %b
; NoAlias: i32* %a, i32* %lq
define void @test_store_arg() {
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  %p = alloca i32*, align 8
  %q = alloca i32*, align 8

  load i32, i32* %a
  load i32, i32* %b
  store i32* %a, i32** %p
  store i32* %b, i32** %q
  call void @store_arg_callee(i32** %p, i32* %b)

  %lp = load i32*, i32** %p
  %lq = load i32*, i32** %q
  load i32, i32* %lp
  load i32, i32* %lq

  ret void
}
