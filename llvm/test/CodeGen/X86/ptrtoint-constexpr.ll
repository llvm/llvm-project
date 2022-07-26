; RUN: llc < %s -mtriple=i386-linux | FileCheck %s
	%union.x = type { i32 }

; CHECK:	.globl r
; CHECK: r:
; CHECK: .long	r

@r = global %union.x { i32 ptrtoint (ptr @r to i32) }, align 4

; CHECK:	.globl x
; CHECK: x:
; CHECK: .quad	3

@x = global i64 mul (i64 3, i64 ptrtoint (ptr getelementptr (i2, ptr null, i64 1) to i64))
