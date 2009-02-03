; RUN: llvm-as < %s | llc
; PR1382

target datalayout = "E-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "powerpc-apple-darwin8.8.0"
@x = global [2 x i32] [ i32 1, i32 2 ]		; <[2 x i32]*> [#uses=1]

define void @foo() {
entry:
	tail call void asm sideeffect "$0 $1", "s,i"( i8* bitcast (i32* getelementptr ([2 x i32]* @x, i32 0, i32 1) to i8*), i8* bitcast (i32* getelementptr ([2 x i32]* @x, i32 0, i32 1) to i8*) )
	ret void
}
