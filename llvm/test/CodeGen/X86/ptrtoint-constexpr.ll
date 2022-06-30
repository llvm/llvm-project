; RUN: llc < %s -mtriple=i386-linux | FileCheck %s
	%union.x = type { i64 }

; CHECK:	.globl r
; CHECK: r:
; CHECK: .quad	r&4294967295

@r = global %union.x { i64 ptrtoint (ptr @r to i64) }, align 4

; CHECK:	.globl x
; CHECK: x:
; CHECK: .quad	3

@x = global i64 mul (i64 3, i64 ptrtoint (ptr getelementptr (i2, ptr null, i64 1) to i64))
