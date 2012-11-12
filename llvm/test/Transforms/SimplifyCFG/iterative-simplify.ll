; RUN: opt < %s -simplifycfg -S | not grep bb17
; PR1786

define i32 @main() {
entry:
	%retval = alloca i32, align 4		; <i32*> [#uses=1]
	%i = alloca i32, align 4		; <i32*> [#uses=7]
	%z = alloca i32, align 4		; <i32*> [#uses=4]
	%z16 = alloca i32, align 4		; <i32*> [#uses=4]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store i32 0, i32* %i
	%toBool = icmp ne i8 1, 0		; <i1> [#uses=1]
	br i1 %toBool, label %cond_true, label %cond_false

cond_true:		; preds = %entry
	store i32 0, i32* %z
	br label %bb

bb:		; preds = %cond_next, %cond_true
	%tmp = load i32* %z		; <i32> [#uses=1]
	%tmp1 = sub i32 %tmp, 16384		; <i32> [#uses=1]
	store i32 %tmp1, i32* %z
	%tmp2 = load i32* %i		; <i32> [#uses=1]
	%tmp3 = add i32 %tmp2, 1		; <i32> [#uses=1]
	store i32 %tmp3, i32* %i
	%tmp4 = load i32* %i		; <i32> [#uses=1]
	%tmp5 = icmp sgt i32 %tmp4, 262144		; <i1> [#uses=1]
	%tmp56 = zext i1 %tmp5 to i8		; <i8> [#uses=1]
	%toBool7 = icmp ne i8 %tmp56, 0		; <i1> [#uses=1]
	br i1 %toBool7, label %cond_true8, label %cond_next

cond_true8:		; preds = %bb
	call void @abort( )
	unreachable

cond_next:		; preds = %bb
	%tmp9 = load i32* %z		; <i32> [#uses=1]
	%tmp10 = icmp ne i32 %tmp9, 0		; <i1> [#uses=1]
	%tmp1011 = zext i1 %tmp10 to i8		; <i8> [#uses=1]
	%toBool12 = icmp ne i8 %tmp1011, 0		; <i1> [#uses=1]
	br i1 %toBool12, label %bb, label %bb13

bb13:		; preds = %cond_next
	call void @exit( i32 0 )
	unreachable

cond_false:		; preds = %entry
	%toBool14 = icmp ne i8 1, 0		; <i1> [#uses=1]
	br i1 %toBool14, label %cond_true15, label %cond_false33

cond_true15:		; preds = %cond_false
	store i32 0, i32* %z16
	br label %bb17

bb17:		; preds = %cond_next27, %cond_true15
	%tmp18 = load i32* %z16		; <i32> [#uses=1]
	%tmp19 = sub i32 %tmp18, 16384		; <i32> [#uses=1]
	store i32 %tmp19, i32* %z16
	%tmp20 = load i32* %i		; <i32> [#uses=1]
	%tmp21 = add i32 %tmp20, 1		; <i32> [#uses=1]
	store i32 %tmp21, i32* %i
	%tmp22 = load i32* %i		; <i32> [#uses=1]
	%tmp23 = icmp sgt i32 %tmp22, 262144		; <i1> [#uses=1]
	%tmp2324 = zext i1 %tmp23 to i8		; <i8> [#uses=1]
	%toBool25 = icmp ne i8 %tmp2324, 0		; <i1> [#uses=1]
	br i1 %toBool25, label %cond_true26, label %cond_next27

cond_true26:		; preds = %bb17
	call void @abort( )
	unreachable

cond_next27:		; preds = %bb17
	%tmp28 = load i32* %z16		; <i32> [#uses=1]
	%tmp29 = icmp ne i32 %tmp28, 0		; <i1> [#uses=1]
	%tmp2930 = zext i1 %tmp29 to i8		; <i8> [#uses=1]
	%toBool31 = icmp ne i8 %tmp2930, 0		; <i1> [#uses=1]
	br i1 %toBool31, label %bb17, label %bb32

bb32:		; preds = %cond_next27
	call void @exit( i32 0 )
	unreachable

cond_false33:		; preds = %cond_false
	call void @exit( i32 0 )
	unreachable

cond_next34:		; No predecessors!
	br label %cond_next35

cond_next35:		; preds = %cond_next34
	br label %return

return:		; preds = %cond_next35
	%retval36 = load i32* %retval		; <i32> [#uses=1]
	ret i32 %retval36
}

declare void @abort()

declare void @exit(i32)
