; RUN: llc < %s -mtriple=i686-- | grep shrl
; Bug in FindModifiedNodeSlot cause tmp14 load to become a zextload and shr 31
; is then optimized away.
@tree_code_type = external global [0 x i32]		; <[0 x i32]*> [#uses=1]

define void @copy_if_shared_r() {
	%tmp = load i32, i32* null		; <i32> [#uses=1]
	%tmp56 = and i32 %tmp, 255		; <i32> [#uses=1]
	%gep.upgrd.1 = zext i32 %tmp56 to i64		; <i64> [#uses=1]
	%tmp8 = getelementptr [0 x i32], [0 x i32]* @tree_code_type, i32 0, i64 %gep.upgrd.1	; <i32*> [#uses=1]
	%tmp9 = load i32, i32* %tmp8		; <i32> [#uses=1]
	%tmp10 = add i32 %tmp9, -1		; <i32> [#uses=1]
	%tmp.upgrd.2 = icmp ugt i32 %tmp10, 2		; <i1> [#uses=1]
	%tmp14 = load i32, i32* null		; <i32> [#uses=1]
	%tmp15 = lshr i32 %tmp14, 31		; <i32> [#uses=1]
	%tmp15.upgrd.3 = trunc i32 %tmp15 to i8		; <i8> [#uses=1]
	%tmp16 = icmp ne i8 %tmp15.upgrd.3, 0		; <i1> [#uses=1]
	br i1 %tmp.upgrd.2, label %cond_false25, label %cond_true
cond_true:		; preds = %0
	br i1 %tmp16, label %cond_true17, label %cond_false
cond_true17:		; preds = %cond_true
	ret void
cond_false:		; preds = %cond_true
	ret void
cond_false25:		; preds = %0
	ret void
}

