; RUN: llc < %s -opaque-pointers -mtriple=i386-linux | FileCheck %s
	%union.x = type { i32 }

; CHECK:	.globl r
; CHECK: r:
; CHECK: .long	r

@r = global %union.x { i32 ptrtoint (ptr @r to i32) }, align 4

; CHECK:	.globl x
; CHECK: x:
; CHECK: .quad	3

@x = global i64 mul (i64 3, i64 ptrtoint (i2* getelementptr (i2, i2* null, i64 1) to i64))
