; PR1015
; RUN: llvm-as < %s | opt -indvars | llvm-dis | not grep {ret i32 0}

target datalayout = "e-p:32:32"
target triple = "i686-apple-darwin8"
@foo = internal constant [5 x i8] c"\00abc\00"		; <[5 x i8]*> [#uses=1]
@str = internal constant [4 x i8] c"%d\0A\00"		; <[4 x i8]*> [#uses=1]


define i32 @test(i32 %J) {
entry:
	br label %bb2

bb:		; preds = %cond_next, %cond_true
	%tmp1 = add i32 %i.0, 1		; <i32> [#uses=1]
	br label %bb2

bb2:		; preds = %bb, %entry
	%i.0 = phi i32 [ 0, %entry ], [ %tmp1, %bb ]		; <i32> [#uses=4]
	%tmp = icmp eq i32 %i.0, 0		; <i1> [#uses=1]
	br i1 %tmp, label %cond_true, label %cond_next

cond_true:		; preds = %bb2
	br label %bb

cond_next:		; preds = %bb2
	%tmp2 = getelementptr [5 x i8]* @foo, i32 0, i32 %i.0		; <i8*> [#uses=1]
	%tmp3 = load i8* %tmp2		; <i8> [#uses=1]
	%tmp5 = icmp eq i8 %tmp3, 0		; <i1> [#uses=1]
	br i1 %tmp5, label %bb6, label %bb

bb6:		; preds = %cond_next
	br label %return

return:		; preds = %bb6
	ret i32 %i.0
}

