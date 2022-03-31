; This testcase ensures that CFL AA answers queries soundly when callee tries 
; to return an escaped pointer

; RUN: opt < %s -aa-pipeline=cfl-steens-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

declare noalias i8* @malloc(i64)
declare void @opaque(i32*)

define i32* @return_escaped_callee() {
	%ptr = call noalias i8* @malloc(i64 8)
	%ptr_cast = bitcast i8* %ptr to i32*
	call void @opaque(i32* %ptr_cast)
	ret i32* %ptr_cast
}
; CHECK-LABEL: Function: test_return_escape
; CHECK-DAG: NoAlias: i32* %a, i32** %x
; CHECK-DAG: NoAlias: i32* %b, i32** %x
; CHECK-DAG: NoAlias: i32* %a, i32* %b
; CHECK-DAG: NoAlias: i32* %c, i32** %x
; CHECK-DAG: NoAlias: i32* %a, i32* %c
; CHECK-DAG: NoAlias: i32* %b, i32* %c
; CHECK-DAG: NoAlias: i32* %a, i32* %d
; CHECK-DAG: MayAlias: i32* %b, i32* %d
; CHECK-DAG: MayAlias: i32* %c, i32* %d
define void @test_return_escape(i32** %x) {
  %a = alloca i32, align 4
  %b = call i32* @return_escaped_callee()
  %c = call i32* @return_escaped_callee()
  load i32, i32* %a
  load i32, i32* %b
  load i32, i32* %c
  %d = load i32*, i32** %x
  load i32, i32* %d

  ret void
}
