; RUN: llvm-as < %s | llc | not grep shrl

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i686-apple-darwin8"
@array = weak global [4 x i32] zeroinitializer		; <[4 x i32]*> [#uses=1]

define i32 @foo(i32 %x) {
entry:
	%tmp2 = lshr i32 %x, 2		; <i32> [#uses=1]
	%tmp3 = and i32 %tmp2, 3		; <i32> [#uses=1]
	%tmp4 = getelementptr [4 x i32]* @array, i32 0, i32 %tmp3		; <i32*> [#uses=1]
	%tmp5 = load i32* %tmp4, align 4		; <i32> [#uses=1]
	ret i32 %tmp5
}

