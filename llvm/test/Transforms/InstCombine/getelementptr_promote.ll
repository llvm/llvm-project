; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep getelementptr | grep {, i64}

; Instcombine should promote the getelementptr index up to the target's
; pointer size, making the conversion explicit, which helps expose it to
; other optimizations.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-apple-darwin8"

define i64 @test(i64* %first, i32 %count) nounwind {
entry:
	%first_addr = alloca i64*		; <i64**> [#uses=2]
	%count_addr = alloca i32		; <i32*> [#uses=2]
	%retval = alloca i64		; <i64*> [#uses=2]
	%n = alloca i32		; <i32*> [#uses=5]
	%result = alloca i64		; <i64*> [#uses=4]
	%0 = alloca i64		; <i64*> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store i64* %first, i64** %first_addr
	store i32 %count, i32* %count_addr
	store i64 0, i64* %result, align 8
	store i32 0, i32* %n, align 4
	br label %bb1

bb:		; preds = %bb1
	%1 = load i64** %first_addr, align 8		; <i64*> [#uses=1]
	%2 = load i32* %n, align 4		; <i32> [#uses=1]
	%3 = bitcast i32 %2 to i32		; <i64> [#uses=1]
	%4 = getelementptr i64* %1, i32 %3		; <i64*> [#uses=1]
	%5 = load i64* %4, align 8		; <i64> [#uses=1]
	%6 = lshr i64 %5, 4		; <i64> [#uses=1]
	%7 = load i64* %result, align 8		; <i64> [#uses=1]
	%8 = add i64 %6, %7		; <i64> [#uses=1]
	store i64 %8, i64* %result, align 8
	%9 = load i32* %n, align 4		; <i32> [#uses=1]
	%10 = add i32 %9, 1		; <i32> [#uses=1]
	store i32 %10, i32* %n, align 4
	br label %bb1

bb1:		; preds = %bb, %entry
	%11 = load i32* %n, align 4		; <i32> [#uses=1]
	%12 = load i32* %count_addr, align 4		; <i32> [#uses=1]
	%13 = icmp slt i32 %11, %12		; <i1> [#uses=1]
	%14 = zext i1 %13 to i8		; <i8> [#uses=1]
	%toBool = icmp ne i8 %14, 0		; <i1> [#uses=1]
	br i1 %toBool, label %bb, label %bb2

bb2:		; preds = %bb1
	%15 = load i64* %result, align 8		; <i64> [#uses=1]
	store i64 %15, i64* %0, align 8
	%16 = load i64* %0, align 8		; <i64> [#uses=1]
	store i64 %16, i64* %retval, align 8
	br label %return

return:		; preds = %bb2
	%retval3 = load i64* %retval		; <i64> [#uses=1]
	ret i64 %retval3
}
