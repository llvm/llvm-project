; This testcase ensures that CFL AA answers queries soundly when callee tries 
; to return the multi-level dereference of one of its parameters

; RUN: opt < %s -aa-pipeline=cfl-anders-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

define i32* @return_deref_arg_multilevel_callee(i32*** %arg1) {
	%deref = load i32**, i32*** %arg1
	%deref2 = load i32*, i32** %deref
	ret i32* %deref2
}
; CHECK-LABEL: Function: test_return_deref_arg_multilevel
; CHECK: NoAlias: i32* %a, i32* %b
; CHECK: MayAlias: i32* %a, i32* %c
; CHECK: NoAlias: i32* %b, i32* %c
; CHECK: NoAlias: i32* %c, i32** %p
; CHECK: NoAlias: i32* %c, i32*** %pp
; CHECK: MayAlias: i32** %lpp, i32** %p
; CHECK: NoAlias: i32** %lpp, i32*** %pp
; CHECK: NoAlias: i32* %c, i32** %lpp
; CHECK: MayAlias: i32* %a, i32* %lpp_deref
; CHECK: NoAlias: i32* %b, i32* %lpp_deref
; CHECK: NoAlias: i32* %lpp_deref, i32*** %pp
; CHECK: MayAlias: i32* %a, i32* %lp
; CHECK: NoAlias: i32* %b, i32* %lp
; CHECK: NoAlias: i32* %lp, i32** %p
; CHECK: NoAlias: i32* %lp, i32*** %pp
; CHECK: MayAlias: i32* %c, i32* %lp
; CHECK: NoAlias: i32* %lp, i32** %lpp
; CHECK: MayAlias: i32* %lp, i32* %lpp_deref

; Temporarily disable modref checks
; Just Ref: Ptr: i32** %p <-> %c = call i32* @return_deref_arg_multilevel_callee(i32*** %pp)
; Just Ref: Ptr: i32*** %pp <-> %c = call i32* @return_deref_arg_multilevel_callee(i32*** %pp)
; Just Ref: Ptr: i32** %lpp <-> %c = call i32* @return_deref_arg_multilevel_callee(i32*** %pp)

define void @test_return_deref_arg_multilevel() {
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  %p = alloca i32*, align 8
  %pp = alloca i32**, align 8

  store i32* %a, i32** %p
  store i32** %p, i32*** %pp
  %c = call i32* @return_deref_arg_multilevel_callee(i32*** %pp)

  %lpp = load i32**, i32*** %pp
  %lpp_deref = load i32*, i32** %lpp
  %lp = load i32*, i32** %p

  ret void
}