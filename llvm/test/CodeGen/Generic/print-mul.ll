; RUN: llc < %s

@a_str = internal constant [8 x i8] c"a = %d\0A\00"		; <[8 x i8]*> [#uses=1]
@b_str = internal constant [8 x i8] c"b = %d\0A\00"		; <[8 x i8]*> [#uses=1]
@a_mul_str = internal constant [13 x i8] c"a * %d = %d\0A\00"		; <[13 x i8]*> [#uses=1]
@A = global i32 2		; <i32*> [#uses=1]
@B = global i32 5		; <i32*> [#uses=1]

declare i32 @printf(i8*, ...)

define i32 @main() {
entry:
	%a = load i32, i32* @A		; <i32> [#uses=2]
	%b = load i32, i32* @B		; <i32> [#uses=1]
	%a_s = getelementptr [8 x i8], [8 x i8]* @a_str, i64 0, i64 0		; <i8*> [#uses=1]
	%b_s = getelementptr [8 x i8], [8 x i8]* @b_str, i64 0, i64 0		; <i8*> [#uses=1]
	%a_mul_s = getelementptr [13 x i8], [13 x i8]* @a_mul_str, i64 0, i64 0		; <i8*> [#uses=1]
	call i32 (i8*, ...) @printf( i8* %a_s, i32 %a )		; <i32>:0 [#uses=0]
	call i32 (i8*, ...) @printf( i8* %b_s, i32 %b )		; <i32>:1 [#uses=0]
	br label %shl_test

shl_test:		; preds = %shl_test, %entry
	%s = phi i32 [ 0, %entry ], [ %s_inc, %shl_test ]		; <i32> [#uses=4]
	%result = mul i32 %a, %s		; <i32> [#uses=1]
	call i32 (i8*, ...) @printf( i8* %a_mul_s, i32 %s, i32 %result )		; <i32>:2 [#uses=0]
	%s_inc = add i32 %s, 1		; <i32> [#uses=1]
	%done = icmp eq i32 %s, 256		; <i1> [#uses=1]
	br i1 %done, label %fini, label %shl_test

fini:		; preds = %shl_test
	ret i32 0
}
