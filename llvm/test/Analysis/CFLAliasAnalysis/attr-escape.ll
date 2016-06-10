; This testcase ensures that CFL AA handles escaped values no more conservative than it should

; RUN: opt < %s -disable-basicaa -cfl-aa -aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s
; RUN: opt < %s -aa-pipeline=cfl-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

; CHECK-LABEL: Function: test_local
; CHECK: NoAlias: i32* %a, i32* %b
; CHECK: MayAlias: i32* %a, i32* %aAlias
; CHECK: NoAlias: i32* %aAlias, i32* %b
define void @test_local() {
	%a = alloca i32, align 4
	%b = alloca i32, align 4
	%aint = ptrtoint i32* %a to i64
	%aAlias = inttoptr i64 %aint to i32*
	ret void
}

; CHECK-LABEL: Function: test_global_param
; CHECK: NoAlias: i32* %a, i32** %x
; CHECK: MayAlias: i32* %a, i32* %xload
; CHECK: MayAlias: i32* %a, i32* %gload
; CHECK: MayAlias: i32* %gload, i32* %xload
; CHECK: MayAlias: i32** %x, i32** @ext_global
; CHECK: NoAlias: i32* %a, i32** @ext_global
@ext_global = external global i32*
define void @test_global_param(i32** %x) {
	%a = alloca i32, align 4
	%aint = ptrtoint i32* %a to i64
	%xload = load i32*, i32** %x
	%gload = load i32*, i32** @ext_global
	ret void
}
