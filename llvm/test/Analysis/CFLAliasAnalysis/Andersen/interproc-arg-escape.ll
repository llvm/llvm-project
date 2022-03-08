; This testcase ensures that CFL AA answers queries soundly when callee tries 
; to escape its parameters

; RUN: opt < %s -aa-pipeline=cfl-anders-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

declare void @opaque(i32*)
define void @escape_arg(i32* %arg) {
	call void @opaque(i32* %arg)
	ret void
}
; CHECK-LABEL: Function: test_arg_escape
; CHECK: NoAlias: i32* %a, i32** %x
; CHECK: NoAlias: i32* %b, i32** %x
; CHECK: NoAlias: i32* %a, i32* %b
; CHECK: NoAlias: i32* %c, i32** %x
; CHECK: NoAlias: i32* %a, i32* %c
; CHECK: NoAlias: i32* %b, i32* %c
; CHECK: MayAlias: i32* %a, i32* %d
; CHECK: MayAlias: i32* %b, i32* %d
; CHECK: NoAlias: i32* %c, i32* %d
define void @test_arg_escape(i32** %x) {
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  %c = alloca i32, align 4
  call void @escape_arg(i32* %a)
  call void @escape_arg(i32* %b)
  %d = load i32*, i32** %x

  ret void
}