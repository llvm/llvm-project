; This testcase ensures that CFL AA answers queries soundly when callee tries 
; to escape the memory pointed to by its parameters

; RUN: opt < %s -disable-basicaa -cfl-steens-aa -aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s
; RUN: opt < %s -aa-pipeline=cfl-steens-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

declare void @opaque(i32*)
define void @escape_arg_deref(i32** %arg) {
	%arg_deref = load i32*, i32** %arg
	call void @opaque(i32* %arg_deref)
	ret void
}
; CHECK-LABEL: Function: test_arg_deref_escape
; CHECK: NoAlias: i32* %a, i32** %x
; CHECK: NoAlias: i32* %b, i32** %x
; CHECK: NoAlias: i32* %a, i32* %b
; CHECK: NoAlias: i32** %p, i32** %x
; CHECK: NoAlias: i32* %a, i32** %p
; CHECK: NoAlias: i32* %b, i32** %p
; CHECK: MayAlias: i32* %a, i32* %c
; CHECK: NoAlias: i32* %b, i32* %c
; CHECK: NoAlias: i32* %c, i32** %p
define void @test_arg_deref_escape(i32** %x) {
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  %p = alloca i32*, align 4
  
  store i32* %a, i32** %p
  call void @escape_arg_deref(i32** %p)
  %c = load i32*, i32** %x

  ret void
}