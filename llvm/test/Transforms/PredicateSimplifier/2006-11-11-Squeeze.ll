; RUN: llvm-as < %s | opt -predsimplify -disable-output

	%struct.cube_struct = type { i32, i32, i32, i32*, i32*, i32*, i32*, i32*, i32*, i32*, i32**, i32**, i32*, i32*, i32, i32, i32*, i32, i32 }
@cube = external global %struct.cube_struct		; <%struct.cube_struct*> [#uses=2]

define fastcc void @cube_setup() {
entry:
	%tmp = load i32* getelementptr (%struct.cube_struct* @cube, i32 0, i32 2)	; <i32> [#uses=2]
	%tmp.upgrd.1 = icmp slt i32 %tmp, 0		; <i1> [#uses=1]
	br i1 %tmp.upgrd.1, label %bb, label %cond_next
cond_next:		; preds = %entry
	%tmp2 = load i32* getelementptr (%struct.cube_struct* @cube, i32 0, i32 1)	; <i32> [#uses=2]
	%tmp5 = icmp slt i32 %tmp2, %tmp		; <i1> [#uses=1]
	br i1 %tmp5, label %bb, label %bb6
bb:		; preds = %cond_next, %entry
	unreachable
bb6:		; preds = %cond_next
	%tmp98124 = icmp sgt i32 %tmp2, 0		; <i1> [#uses=1]
	br i1 %tmp98124, label %bb42, label %bb99
bb42:		; preds = %bb6
	ret void
bb99:		; preds = %bb6
	ret void
}

