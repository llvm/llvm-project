; RUN: opt < %s -instcombine -S | grep "= or i32 %x, -5"

@.str = internal constant [5 x i8] c"foo\0A\00"		; <[5 x i8]*> [#uses=1]
@.str1 = internal constant [5 x i8] c"bar\0A\00"		; <[5 x i8]*> [#uses=1]

define i32 @main() nounwind  {
entry:
	%x = call i32 @func_11( ) nounwind 		; <i32> [#uses=1]
	%tmp3 = or i32 %x, -5		; <i32> [#uses=1]
	%tmp5 = urem i32 251, %tmp3		; <i32> [#uses=1]
	%tmp6 = icmp ne i32 %tmp5, 0		; <i1> [#uses=1]
	%tmp67 = zext i1 %tmp6 to i32		; <i32> [#uses=1]
	%tmp9 = urem i32 %tmp67, 95		; <i32> [#uses=1]
	%tmp10 = and i32 %tmp9, 1		; <i32> [#uses=1]
	%tmp12 = icmp eq i32 %tmp10, 0		; <i1> [#uses=1]
	br i1 %tmp12, label %bb14, label %bb

bb:		; preds = %entry
	br label %bb15

bb14:		; preds = %entry
	br label %bb15

bb15:		; preds = %bb14, %bb
	%iftmp.0.0 = phi i8* [ getelementptr ([5 x i8]* @.str1, i32 0, i32 0), %bb14 ], [ getelementptr ([5 x i8]* @.str, i32 0, i32 0), %bb ]		; <i8*> [#uses=1]
	%tmp17 = call i32 (i8*, ...)* @printf( i8* %iftmp.0.0 ) nounwind 		; <i32> [#uses=0]
	ret i32 0
}

declare i32 @func_11()

declare i32 @printf(i8*, ...) nounwind 
