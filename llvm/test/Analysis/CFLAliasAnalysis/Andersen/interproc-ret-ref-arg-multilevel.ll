; This testcase ensures that CFL AA answers queries soundly when callee tries 
; to return the multi-level reference of one of its parameters

; RUN: opt < %s -disable-basic-aa -cfl-anders-aa -aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s
; RUN: opt < %s -aa-pipeline=cfl-anders-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

declare noalias i8* @malloc(i64)

define i32*** @return_ref_arg_multilevel_callee(i32* %arg1) {
	%ptr = call noalias i8* @malloc(i64 8)
	%ptr_cast = bitcast i8* %ptr to i32***
  %ptr2 = call noalias i8* @malloc(i64 8)
  %ptr_cast2 = bitcast i8* %ptr2 to i32**
	store i32* %arg1, i32** %ptr_cast2
  store i32** %ptr_cast2, i32*** %ptr_cast
	ret i32*** %ptr_cast
}
; CHECK-LABEL: Function: test_return_ref_arg_multilevel
; CHECK: NoAlias: i32* %a, i32*** %b
; CHECK: NoAlias: i32** %p, i32*** %b
; CHECK: NoAlias: i32*** %b, i32*** %pp
; CHECK: NoAlias: i32* %a, i32** %lb
; CHECK: NoAlias: i32** %lb, i32** %p
; CHECK: NoAlias: i32** %lb, i32*** %pp
; CHECK: NoAlias: i32** %lb, i32*** %b
; CHECK: MayAlias: i32* %a, i32* %lb_deref
; CHECK: NoAlias: i32* %lb_deref, i32** %lpp
; CHECK: MayAlias: i32* %lb_deref, i32* %lpp_deref
; CHECK: NoAlias: i32* %lpp_deref, i32** %lpp
; CHECK: MayAlias: i32* %lb_deref, i32* %lp
; CHECK: NoAlias: i32* %lp, i32** %lpp
; CHECK: MayAlias: i32* %lp, i32* %lpp_deref

; Temporarily disable modref checks
; Just Mod: Ptr: i32*** %b <-> %b = call i32*** @return_ref_arg_multilevel_callee(i32* %a)
; Just Mod: Ptr: i32** %lb <-> %b = call i32*** @return_ref_arg_multilevel_callee(i32* %a)
define void @test_return_ref_arg_multilevel() {
  %a = alloca i32, align 4
  %p = alloca i32*, align 8
  %pp = alloca i32**, align 8

  store i32* %a, i32** %p
  store i32** %p, i32*** %pp
  %b = call i32*** @return_ref_arg_multilevel_callee(i32* %a)

  %lb = load i32**, i32*** %b
  %lb_deref = load i32*, i32** %lb
  %lpp = load i32**, i32*** %pp
  %lpp_deref = load i32*, i32** %lpp
  %lp = load i32*, i32** %p

  ret void
}