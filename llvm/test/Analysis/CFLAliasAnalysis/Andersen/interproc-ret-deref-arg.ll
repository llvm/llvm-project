; This testcase ensures that CFL AA answers queries soundly when callee tries 
; to return the dereference of one of its parameters

; RUN: opt < %s -aa-pipeline=cfl-anders-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

define i32* @return_deref_arg_callee(i32** %arg1) {
  %deref = load i32*, i32** %arg1
  ret i32* %deref
}
; CHECK-LABEL: Function: test_return_deref_arg
; CHECK: NoAlias: i32* %a, i32* %b
; CHECK: MayAlias: i32* %a, i32* %c
; CHECK: NoAlias: i32* %b, i32* %c
; CHECK: MayAlias: i32* %a, i32* %lp
; CHECK: NoAlias: i32* %b, i32* %lp
; CHECK: NoAlias: i32* %lp, i32** %p
; CHECK: MayAlias: i32* %c, i32* %lp

; Temporarily disable modref checks
; NoModRef: Ptr: i32* %a <-> %c = call i32* @return_deref_arg_callee(i32** %p)
; NoModRef: Ptr: i32* %b <-> %c = call i32* @return_deref_arg_callee(i32** %p)
; Just Ref: Ptr: i32** %p <-> %c = call i32* @return_deref_arg_callee(i32** %p)
; NoModRef: Ptr: i32* %c <-> %c = call i32* @return_deref_arg_callee(i32** %p)
; NoModRef: Ptr: i32* %lp <-> %c = call i32* @return_deref_arg_callee(i32** %p)
define void @test_return_deref_arg() {
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  %p = alloca i32*, align 8

  store i32* %a, i32** %p
  %c = call i32* @return_deref_arg_callee(i32** %p)

  %lp = load i32*, i32** %p

  ret void
}