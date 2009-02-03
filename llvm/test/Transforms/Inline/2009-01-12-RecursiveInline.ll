; RUN: llvm-as < %s | opt -inline | llvm-dis | grep {call.*fib} | count 4
; First call to fib from fib is inlined, producing 2 instead of 1, total 3.
; Second call to fib from fib is not inlined because new body of fib exceeds
; inlining limit of 200.  Plus call in main = 4 total.

; ModuleID = '<stdin>'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin9.6"
@"\01LC" = internal constant [5 x i8] c"%ld\0A\00"		; <[5 x i8]*> [#uses=1]

define i32 @fib(i32 %n) nounwind {
entry:
	%n_addr = alloca i32		; <i32*> [#uses=4]
	%retval = alloca i32		; <i32*> [#uses=2]
	%0 = alloca i32		; <i32*> [#uses=3]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store i32 %n, i32* %n_addr
	%1 = load i32* %n_addr, align 4		; <i32> [#uses=1]
	%2 = icmp ule i32 %1, 1		; <i1> [#uses=1]
	br i1 %2, label %bb, label %bb1

bb:		; preds = %entry
	store i32 1, i32* %0, align 4
	br label %bb2

bb1:		; preds = %entry
	%3 = load i32* %n_addr, align 4		; <i32> [#uses=1]
	%4 = sub i32 %3, 2		; <i32> [#uses=1]
	%5 = call i32 @fib(i32 %4) nounwind		; <i32> [#uses=1]
	%6 = load i32* %n_addr, align 4		; <i32> [#uses=1]
	%7 = sub i32 %6, 1		; <i32> [#uses=1]
	%8 = call i32 @fib(i32 %7) nounwind		; <i32> [#uses=1]
	%9 = add i32 %5, %8		; <i32> [#uses=1]
	store i32 %9, i32* %0, align 4
	br label %bb2

bb2:		; preds = %bb1, %bb
	%10 = load i32* %0, align 4		; <i32> [#uses=1]
	store i32 %10, i32* %retval, align 4
	br label %return

return:		; preds = %bb2
	%retval3 = load i32* %retval		; <i32> [#uses=1]
	ret i32 %retval3
}

define i32 @main(i32 %argc, i8** %argv) nounwind {
entry:
	%argc_addr = alloca i32		; <i32*> [#uses=2]
	%argv_addr = alloca i8**		; <i8***> [#uses=2]
	%retval = alloca i32		; <i32*> [#uses=2]
	%N = alloca i32		; <i32*> [#uses=2]
	%0 = alloca i32		; <i32*> [#uses=2]
	%iftmp.0 = alloca i32		; <i32*> [#uses=3]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store i32 %argc, i32* %argc_addr
	store i8** %argv, i8*** %argv_addr
	%1 = load i32* %argc_addr, align 4		; <i32> [#uses=1]
	%2 = icmp eq i32 %1, 2		; <i1> [#uses=1]
	br i1 %2, label %bb, label %bb1

bb:		; preds = %entry
	%3 = load i8*** %argv_addr, align 4		; <i8**> [#uses=1]
	%4 = getelementptr i8** %3, i32 1		; <i8**> [#uses=1]
	%5 = load i8** %4, align 4		; <i8*> [#uses=1]
	%6 = call i32 @atoi(i8* %5) nounwind		; <i32> [#uses=1]
	store i32 %6, i32* %iftmp.0, align 4
	br label %bb2

bb1:		; preds = %entry
	store i32 43, i32* %iftmp.0, align 4
	br label %bb2

bb2:		; preds = %bb1, %bb
	%7 = load i32* %iftmp.0, align 4		; <i32> [#uses=1]
	store i32 %7, i32* %N, align 4
	%8 = load i32* %N, align 4		; <i32> [#uses=1]
	%9 = call i32 @fib(i32 %8) nounwind		; <i32> [#uses=1]
	%10 = call i32 (i8*, ...)* @printf(i8* getelementptr ([5 x i8]* @"\01LC", i32 0, i32 0), i32 %9) nounwind		; <i32> [#uses=0]
	store i32 0, i32* %0, align 4
	%11 = load i32* %0, align 4		; <i32> [#uses=1]
	store i32 %11, i32* %retval, align 4
	br label %return

return:		; preds = %bb2
	%retval3 = load i32* %retval		; <i32> [#uses=1]
	ret i32 %retval3
}

declare i32 @atoi(i8*)

declare i32 @printf(i8*, ...) nounwind
