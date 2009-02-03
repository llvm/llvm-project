; RUN: llvm-as < %s | opt -gvnpre | llvm-dis | not grep {%z3 =}
; RUN: llvm-as < %s | opt -gvnpre | llvm-dis | not grep {%z9 =}

define i32 @main() {
block1:
	%z1 = bitcast i32 0 to i32		; <i32> [#uses=5]
	br label %block2

block2:		; preds = %block6, %block1
	%z2 = phi i32 [ %z1, %block1 ], [ %z3, %block6 ]		; <i32> [#uses=3]
	%z3 = add i32 %z2, 1		; <i32> [#uses=5]
	br i1 false, label %block3, label %block7

block3:		; preds = %block2
	br i1 true, label %block4, label %block5

block4:		; preds = %block3
	%z4 = add i32 %z2, %z3		; <i32> [#uses=1]
	%z5 = bitcast i32 %z4 to i32		; <i32> [#uses=1]
	%z6 = add i32 %z1, %z5		; <i32> [#uses=0]
	br label %block6

block5:		; preds = %block3
	%z7 = add i32 %z3, 1		; <i32> [#uses=1]
	br label %block6

block6:		; preds = %block5, %block4
	%z8 = phi i32 [ %z1, %block4 ], [ %z7, %block5 ]		; <i32> [#uses=1]
	%z9 = add i32 %z2, %z3		; <i32> [#uses=2]
	%z10 = add i32 %z9, %z8		; <i32> [#uses=0]
	%z11 = bitcast i32 12 to i32		; <i32> [#uses=1]
	%z12 = add i32 %z9, %z11		; <i32> [#uses=1]
	%z13 = add i32 %z12, %z3		; <i32> [#uses=0]
	br label %block2

block7:		; preds = %block2
	ret i32 %z1
}
