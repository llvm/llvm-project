; RUN: llvm-as < %s | llc -march=x86 -stats |& \
; RUN:   grep {1 .*folded into instructions}
; RUN: llvm-as < %s | llc -march=x86 | grep cmp | count 4

	%struct.quad_struct = type { i32, i32, %struct.quad_struct*, %struct.quad_struct*, %struct.quad_struct*, %struct.quad_struct*, %struct.quad_struct* }

define fastcc i32 @perimeter(%struct.quad_struct* %tree, i32 %size) {
entry:
	%tree.idx7.val = load %struct.quad_struct** null		; <%struct.quad_struct*> [#uses=1]
	%tmp8.i51 = icmp eq %struct.quad_struct* %tree.idx7.val, null		; <i1> [#uses=2]
	br i1 %tmp8.i51, label %cond_next, label %cond_next.i52

cond_next.i52:		; preds = %entry
	ret i32 0

cond_next:		; preds = %entry
	%tmp59 = load i32* null, align 4		; <i32> [#uses=1]
	%tmp70 = icmp eq i32 %tmp59, 2		; <i1> [#uses=1]
	br i1 %tmp70, label %cond_true.i35, label %bb80

cond_true.i35:		; preds = %cond_next
	%tmp14.i.i37 = load %struct.quad_struct** null, align 4		; <%struct.quad_struct*> [#uses=1]
	%tmp3.i160 = load i32* null, align 4		; <i32> [#uses=1]
	%tmp4.i161 = icmp eq i32 %tmp3.i160, 2		; <i1> [#uses=1]
	br i1 %tmp4.i161, label %cond_true.i163, label %cond_false.i178

cond_true.i163:		; preds = %cond_true.i35
	%tmp7.i162 = sdiv i32 %size, 4		; <i32> [#uses=2]
	%tmp13.i168 = tail call fastcc i32 @sum_adjacent( %struct.quad_struct* null, i32 3, i32 2, i32 %tmp7.i162 )		; <i32> [#uses=1]
	%tmp18.i11.i170 = getelementptr %struct.quad_struct* %tmp14.i.i37, i32 0, i32 4		; <%struct.quad_struct**> [#uses=1]
	%tmp19.i12.i171 = load %struct.quad_struct** %tmp18.i11.i170, align 4		; <%struct.quad_struct*> [#uses=1]
	%tmp21.i173 = tail call fastcc i32 @sum_adjacent( %struct.quad_struct* %tmp19.i12.i171, i32 3, i32 2, i32 %tmp7.i162 )		; <i32> [#uses=1]
	%tmp22.i174 = add i32 %tmp21.i173, %tmp13.i168		; <i32> [#uses=1]
	br i1 %tmp4.i161, label %cond_true.i141, label %cond_false.i156

cond_false.i178:		; preds = %cond_true.i35
	ret i32 0

cond_true.i141:		; preds = %cond_true.i163
	%tmp7.i140 = sdiv i32 %size, 4		; <i32> [#uses=1]
	%tmp21.i151 = tail call fastcc i32 @sum_adjacent( %struct.quad_struct* null, i32 3, i32 2, i32 %tmp7.i140 )		; <i32> [#uses=0]
	ret i32 0

cond_false.i156:		; preds = %cond_true.i163
	%tmp22.i44 = add i32 0, %tmp22.i174		; <i32> [#uses=0]
	br i1 %tmp8.i51, label %bb22.i, label %cond_next.i

bb80:		; preds = %cond_next
	ret i32 0

cond_next.i:		; preds = %cond_false.i156
	ret i32 0

bb22.i:		; preds = %cond_false.i156
	ret i32 0
}

declare fastcc i32 @sum_adjacent(%struct.quad_struct*, i32, i32, i32)
