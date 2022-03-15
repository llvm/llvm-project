; This testcase ensures that CFL AA answers queries soundly when callee tries 
; to mutate the memory pointed to by its parameters

; RUN: opt < %s -aa-pipeline=cfl-steens-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

declare noalias i8* @malloc(i64)

define void @store_arg_multilevel_callee(i32*** %arg1, i32* %arg2) {
  %ptr = call noalias i8* @malloc(i64 8)
  %ptr_cast = bitcast i8* %ptr to i32**
	store i32* %arg2, i32** %ptr_cast
  store i32** %ptr_cast, i32*** %arg1
	ret void
}
; CHECK-LABEL: Function: test_store_arg_multilevel
; CHECK: NoAlias: i32* %a, i32** %lpp
; CHECK: NoAlias: i32* %b, i32** %lpp
; CHECK: MayAlias: i32** %lpp, i32** %p
; CHECK: MayAlias: i32* %a, i32* %lpp_deref
; CHECK: MayAlias: i32* %b, i32* %lpp_deref
; CHECK: NoAlias: i32* %lpp_deref, i32** %p
; CHECK: NoAlias: i32* %lpp_deref, i32*** %pp
; CHECK: NoAlias: i32** %lpp, i32* %lpp_deref
; CHECK: MayAlias: i32* %a, i32* %lp
; CHECK: NoAlias: i32* %lp, i32*** %pp
; CHECK: NoAlias: i32* %lp, i32** %lpp
; CHECK: MayAlias: i32* %lp, i32* %lpp_deref

; We could've proven the following facts if the analysis were inclusion-based:
; NoAlias: i32* %a, i32* %b
define void @test_store_arg_multilevel() {
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  %p = alloca i32*, align 8
  %pp = alloca i32**, align 8

  load i32, i32* %a
  load i32, i32* %b
  store i32* %a, i32** %p
  store i32** %p, i32*** %pp
  call void @store_arg_multilevel_callee(i32*** %pp, i32* %b)

  %lpp = load i32**, i32*** %pp
  %lpp_deref = load i32*, i32** %lpp
  %lp = load i32*, i32** %p
  load i32, i32* %lpp_deref
  load i32, i32* %lp

  ret void
}

