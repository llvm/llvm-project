; This testcase comes from this C fragment:
;
; void test(unsigned Num, int *Array) {
;  unsigned i, j, k;
;
;  for (i = 0; i != Num; ++i)
;    for (j = 0; j != Num; ++j)
;      for (k = 0; k != Num; ++k)
;        printf("%d\n", i+k+j);    /* Reassociate to (i+j)+k */
;}
;
; In this case, we want to reassociate the specified expr so that i+j can be
; hoisted out of the inner most loop.
;
; RUN: opt < %s -reassociate -S | grep 115 | not grep 117
; END.
@.LC0 = internal global [4 x i8] c"%d\0A\00"		; <[4 x i8]*> [#uses=1]

declare i32 @printf(i8*, ...)

define void @test(i32 %Num, i32* %Array) {
bb0:
	%cond221 = icmp eq i32 0, %Num		; <i1> [#uses=3]
	br i1 %cond221, label %bb7, label %bb2
bb2:		; preds = %bb6, %bb0
	%reg115 = phi i32 [ %reg120, %bb6 ], [ 0, %bb0 ]		; <i32> [#uses=2]
	br i1 %cond221, label %bb6, label %bb3
bb3:		; preds = %bb5, %bb2
	%reg116 = phi i32 [ %reg119, %bb5 ], [ 0, %bb2 ]		; <i32> [#uses=2]
	br i1 %cond221, label %bb5, label %bb4
bb4:		; preds = %bb4, %bb3
	%reg117 = phi i32 [ %reg118, %bb4 ], [ 0, %bb3 ]		; <i32> [#uses=2]
	%reg113 = add i32 %reg115, %reg117		; <i32> [#uses=1]
	%reg114 = add i32 %reg113, %reg116		; <i32> [#uses=1]
	%cast227 = getelementptr [4 x i8]* @.LC0, i64 0, i64 0		; <i8*> [#uses=1]
	call i32 (i8*, ...)* @printf( i8* %cast227, i32 %reg114 )		; <i32>:0 [#uses=0]
	%reg118 = add i32 %reg117, 1		; <i32> [#uses=2]
	%cond224 = icmp ne i32 %reg118, %Num		; <i1> [#uses=1]
	br i1 %cond224, label %bb4, label %bb5
bb5:		; preds = %bb4, %bb3
	%reg119 = add i32 %reg116, 1		; <i32> [#uses=2]
	%cond225 = icmp ne i32 %reg119, %Num		; <i1> [#uses=1]
	br i1 %cond225, label %bb3, label %bb6
bb6:		; preds = %bb5, %bb2
	%reg120 = add i32 %reg115, 1		; <i32> [#uses=2]
	%cond226 = icmp ne i32 %reg120, %Num		; <i1> [#uses=1]
	br i1 %cond226, label %bb2, label %bb7
bb7:		; preds = %bb6, %bb0
	ret void
}
