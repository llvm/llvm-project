; Test that appending linkage works correctly.

; RUN: echo "@X = appending global [1 x i32] [i32 8] " | \
; RUN:   llvm-as > %t.2.bc
; RUN: llvm-as < %s > %t.1.bc
; RUN: llvm-link %t.1.bc %t.2.bc -S | FileCheck %s
; CHECK: [i32 7, i32 4, i32 8]

@X = appending global [2 x i32] [ i32 7, i32 4 ]		; <[2 x i32]*> [#uses=2]
@Y = global i32* getelementptr ([2 x i32], [2 x i32]* @X, i64 0, i64 0)		; <i32**> [#uses=0]

define void @foo(i64 %V) {
	%Y = getelementptr [2 x i32], [2 x i32]* @X, i64 0, i64 %V		; <i32*> [#uses=0]
	ret void
}

