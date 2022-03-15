; This testcase ensures that CFL AA handles malloc and free in a sound and precise manner

; RUN: opt < %s -aa-pipeline=cfl-steens-aa -passes=aa-eval -print-no-aliases -disable-output 2>&1 | FileCheck %s

declare noalias i8* @malloc(i64)
declare noalias i8* @calloc(i64, i64)
declare void @free(i8* nocapture)

; CHECK: Function: test_malloc
; CHECK: NoAlias: i8* %p, i8* %q
define void @test_malloc(i8* %p) {
	%q = call i8* @malloc(i64 4)
  load i8, i8* %p
  load i8, i8* %q
	ret void
}

; CHECK: Function: test_calloc
; CHECK: NoAlias: i8* %p, i8* %q
define void @test_calloc(i8* %p) {
	%q = call i8* @calloc(i64 2, i64 4)
  load i8, i8* %p
  load i8, i8* %q
	ret void
}

; CHECK: Function: test_free
; CHECK: NoAlias: i8* %p, i8* %q
define void @test_free(i8* %p) {
	%q = alloca i8, align 4
	call void @free(i8* %q)
  load i8, i8* %p
  load i8, i8* %q
	ret void
}
