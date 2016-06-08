; This testcase ensures that CFL AA handles escaped values no more conservative than it should

; RUN: opt < %s -disable-basicaa -cfl-aa -aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s
; RUN: opt < %s -aa-pipeline=cfl-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

; CHECK: Function: escape_ptrtoint
; CHECK: NoAlias: i32* %a, i32* %x
; CHECK: NoAlias: i32* %b, i32* %x
; CHECK: NoAlias: i32* %a, i32* %b
; CHECK: MayAlias: i32* %a, i32* %aAlias
; CHECK: NoAlias: i32* %aAlias, i32* %b
define void @escape_ptrtoint(i32* %x) {
	%a = alloca i32, align 4
	%b = alloca i32, align 4
	%aint = ptrtoint i32* %a to i64
	%aAlias = inttoptr i64 %aint to i32*
	ret void
}
