; RUN: llc < %s -mtriple=i686-apple-darwin -relocation-model=static | FileCheck %s -check-prefix=STATIC
; RUN: llc < %s -mtriple=i686-apple-darwin -relocation-model=pic | FileCheck %s -check-prefix=PIC

; By starting the IV at -64 instead of 0, a cmp is eliminated,
; as the flags from the add can be used directly.

; STATIC: movl    $-64, [[ECX:%e..]]

; STATIC: movl    [[EAX:%e..]], _state+76([[ECX]])
; STATIC: addl    $16, [[ECX]]
; STATIC: jne

; In PIC mode the symbol can't be folded, so the change-compare-stride
; trick applies.

; PIC: cmpl $64

@state = external global [0 x i32]		; <[0 x i32]*> [#uses=4]
@S = external global [0 x i32]		; <[0 x i32]*> [#uses=4]

define i32 @foo() nounwind {
entry:
	br label %bb

bb:		; preds = %bb, %entry
	%indvar = phi i32 [ 0, %entry ], [ %indvar.next, %bb ]		; <i32> [#uses=2]
	%t.063.0 = phi i32 [ 0, %entry ], [ %tmp47, %bb ]		; <i32> [#uses=1]
	%j.065.0 = shl i32 %indvar, 2		; <i32> [#uses=4]
	%tmp3 = getelementptr [0 x i32], [0 x i32]* @state, i32 0, i32 %j.065.0		; <i32*> [#uses=2]
	%tmp4 = load i32, i32* %tmp3, align 4		; <i32> [#uses=1]
	%tmp6 = getelementptr [0 x i32], [0 x i32]* @S, i32 0, i32 %t.063.0		; <i32*> [#uses=1]
	%tmp7 = load i32, i32* %tmp6, align 4		; <i32> [#uses=1]
	%tmp8 = xor i32 %tmp7, %tmp4		; <i32> [#uses=2]
	store i32 %tmp8, i32* %tmp3, align 4
	%tmp1378 = or i32 %j.065.0, 1		; <i32> [#uses=1]
	%tmp16 = getelementptr [0 x i32], [0 x i32]* @state, i32 0, i32 %tmp1378		; <i32*> [#uses=2]
	%tmp17 = load i32, i32* %tmp16, align 4		; <i32> [#uses=1]
	%tmp19 = getelementptr [0 x i32], [0 x i32]* @S, i32 0, i32 %tmp8		; <i32*> [#uses=1]
	%tmp20 = load i32, i32* %tmp19, align 4		; <i32> [#uses=1]
	%tmp21 = xor i32 %tmp20, %tmp17		; <i32> [#uses=2]
	store i32 %tmp21, i32* %tmp16, align 4
	%tmp2680 = or i32 %j.065.0, 2		; <i32> [#uses=1]
	%tmp29 = getelementptr [0 x i32], [0 x i32]* @state, i32 0, i32 %tmp2680		; <i32*> [#uses=2]
	%tmp30 = load i32, i32* %tmp29, align 4		; <i32> [#uses=1]
	%tmp32 = getelementptr [0 x i32], [0 x i32]* @S, i32 0, i32 %tmp21		; <i32*> [#uses=1]
	%tmp33 = load i32, i32* %tmp32, align 4		; <i32> [#uses=1]
	%tmp34 = xor i32 %tmp33, %tmp30		; <i32> [#uses=2]
	store i32 %tmp34, i32* %tmp29, align 4
	%tmp3982 = or i32 %j.065.0, 3		; <i32> [#uses=1]
	%tmp42 = getelementptr [0 x i32], [0 x i32]* @state, i32 0, i32 %tmp3982		; <i32*> [#uses=2]
	%tmp43 = load i32, i32* %tmp42, align 4		; <i32> [#uses=1]
	%tmp45 = getelementptr [0 x i32], [0 x i32]* @S, i32 0, i32 %tmp34		; <i32*> [#uses=1]
	%tmp46 = load i32, i32* %tmp45, align 4		; <i32> [#uses=1]
	%tmp47 = xor i32 %tmp46, %tmp43		; <i32> [#uses=3]
	store i32 %tmp47, i32* %tmp42, align 4
	%indvar.next = add i32 %indvar, 1		; <i32> [#uses=2]
	%exitcond = icmp eq i32 %indvar.next, 4		; <i1> [#uses=1]
	br i1 %exitcond, label %bb57, label %bb

bb57:		; preds = %bb
	%tmp59 = and i32 %tmp47, 255		; <i32> [#uses=1]
	ret i32 %tmp59
}
