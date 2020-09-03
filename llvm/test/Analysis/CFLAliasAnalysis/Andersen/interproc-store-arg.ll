; This testcase ensures that CFL AA answers queries soundly when callee tries 
; to mutate the memory pointed to by its parameters

; RUN: opt < %s -disable-basic-aa -cfl-anders-aa -aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s
; RUN: opt < %s -aa-pipeline=cfl-anders-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

define void @store_arg_callee(i32** %arg1, i32* %arg2) {
	store i32* %arg2, i32** %arg1
	ret void
}
; CHECK-LABEL: Function: test_store_arg
; CHECK: NoAlias: i32* %a, i32* %b
; CHECK: NoAlias: i32* %a, i32** %p
; CHECK: NoAlias: i32* %b, i32** %p
; CHECK: MayAlias: i32* %a, i32* %lp
; CHECK: MayAlias: i32* %b, i32* %lp
; CHECK: NoAlias: i32* %a, i32* %lq
; CHECK: MayAlias: i32* %b, i32* %lq
; CHECK: MayAlias: i32* %lp, i32* %lq

; Temporarily disable modref checks
; NoModRef: Ptr: i32* %a <-> call void @store_arg_callee(i32** %p, i32* %b)
; Just Ref: Ptr: i32* %b <-> call void @store_arg_callee(i32** %p, i32* %b)
; Just Mod: Ptr: i32** %p  <-> call void @store_arg_callee(i32** %p, i32* %b)
; NoModRef: Ptr: i32** %q  <-> call void @store_arg_callee(i32** %p, i32* %b)
define void @test_store_arg() {
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  %p = alloca i32*, align 8
  %q = alloca i32*, align 8

  store i32* %a, i32** %p
  store i32* %b, i32** %q
  call void @store_arg_callee(i32** %p, i32* %b)

  %lp = load i32*, i32** %p
  %lq = load i32*, i32** %q

  ret void
}