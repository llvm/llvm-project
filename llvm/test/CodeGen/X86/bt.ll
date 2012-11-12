; RUN: llc < %s -march=x86 | grep btl | count 28
; RUN: llc < %s -march=x86 -mcpu=pentium4 | grep btl | not grep esp
; RUN: llc < %s -march=x86 -mcpu=penryn   | grep btl | not grep esp
; PR3253

; The register+memory form of the BT instruction should be usable on
; pentium4, however it is currently disabled due to the register+memory
; form having different semantics than the register+register form.

; Test these patterns:
;    (X & (1 << N))  != 0  -->  BT(X, N).
;    ((X >>u N) & 1) != 0  -->  BT(X, N).
; as well as several variations:
;    - The second form can use an arithmetic shift.
;    - Either form can use == instead of !=.
;    - Either form can compare with an operand of the &
;      instead of with 0.
;    - The comparison can be commuted (only cases where neither
;      operand is constant are included).
;    - The and can be commuted.

define void @test2(i32 %x, i32 %n) nounwind {
entry:
	%tmp29 = lshr i32 %x, %n		; <i32> [#uses=1]
	%tmp3 = and i32 %tmp29, 1		; <i32> [#uses=1]
	%tmp4 = icmp eq i32 %tmp3, 0		; <i1> [#uses=1]
	br i1 %tmp4, label %bb, label %UnifiedReturnBlock

bb:		; preds = %entry
	call void @foo()
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

define void @test2b(i32 %x, i32 %n) nounwind {
entry:
	%tmp29 = lshr i32 %x, %n		; <i32> [#uses=1]
	%tmp3 = and i32 1, %tmp29
	%tmp4 = icmp eq i32 %tmp3, 0		; <i1> [#uses=1]
	br i1 %tmp4, label %bb, label %UnifiedReturnBlock

bb:		; preds = %entry
	call void @foo()
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

define void @atest2(i32 %x, i32 %n) nounwind {
entry:
	%tmp29 = ashr i32 %x, %n		; <i32> [#uses=1]
	%tmp3 = and i32 %tmp29, 1		; <i32> [#uses=1]
	%tmp4 = icmp eq i32 %tmp3, 0		; <i1> [#uses=1]
	br i1 %tmp4, label %bb, label %UnifiedReturnBlock

bb:		; preds = %entry
	call void @foo()
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

define void @atest2b(i32 %x, i32 %n) nounwind {
entry:
	%tmp29 = ashr i32 %x, %n		; <i32> [#uses=1]
	%tmp3 = and i32 1, %tmp29
	%tmp4 = icmp eq i32 %tmp3, 0		; <i1> [#uses=1]
	br i1 %tmp4, label %bb, label %UnifiedReturnBlock

bb:		; preds = %entry
	call void @foo()
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

define void @test3(i32 %x, i32 %n) nounwind {
entry:
	%tmp29 = shl i32 1, %n		; <i32> [#uses=1]
	%tmp3 = and i32 %tmp29, %x		; <i32> [#uses=1]
	%tmp4 = icmp eq i32 %tmp3, 0		; <i1> [#uses=1]
	br i1 %tmp4, label %bb, label %UnifiedReturnBlock

bb:		; preds = %entry
	call void @foo()
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

define void @test3b(i32 %x, i32 %n) nounwind {
entry:
	%tmp29 = shl i32 1, %n		; <i32> [#uses=1]
	%tmp3 = and i32 %x, %tmp29
	%tmp4 = icmp eq i32 %tmp3, 0		; <i1> [#uses=1]
	br i1 %tmp4, label %bb, label %UnifiedReturnBlock

bb:		; preds = %entry
	call void @foo()
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

define void @testne2(i32 %x, i32 %n) nounwind {
entry:
	%tmp29 = lshr i32 %x, %n		; <i32> [#uses=1]
	%tmp3 = and i32 %tmp29, 1		; <i32> [#uses=1]
	%tmp4 = icmp ne i32 %tmp3, 0		; <i1> [#uses=1]
	br i1 %tmp4, label %bb, label %UnifiedReturnBlock

bb:		; preds = %entry
	call void @foo()
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

define void @testne2b(i32 %x, i32 %n) nounwind {
entry:
	%tmp29 = lshr i32 %x, %n		; <i32> [#uses=1]
	%tmp3 = and i32 1, %tmp29
	%tmp4 = icmp ne i32 %tmp3, 0		; <i1> [#uses=1]
	br i1 %tmp4, label %bb, label %UnifiedReturnBlock

bb:		; preds = %entry
	call void @foo()
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

define void @atestne2(i32 %x, i32 %n) nounwind {
entry:
	%tmp29 = ashr i32 %x, %n		; <i32> [#uses=1]
	%tmp3 = and i32 %tmp29, 1		; <i32> [#uses=1]
	%tmp4 = icmp ne i32 %tmp3, 0		; <i1> [#uses=1]
	br i1 %tmp4, label %bb, label %UnifiedReturnBlock

bb:		; preds = %entry
	call void @foo()
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

define void @atestne2b(i32 %x, i32 %n) nounwind {
entry:
	%tmp29 = ashr i32 %x, %n		; <i32> [#uses=1]
	%tmp3 = and i32 1, %tmp29
	%tmp4 = icmp ne i32 %tmp3, 0		; <i1> [#uses=1]
	br i1 %tmp4, label %bb, label %UnifiedReturnBlock

bb:		; preds = %entry
	call void @foo()
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

define void @testne3(i32 %x, i32 %n) nounwind {
entry:
	%tmp29 = shl i32 1, %n		; <i32> [#uses=1]
	%tmp3 = and i32 %tmp29, %x		; <i32> [#uses=1]
	%tmp4 = icmp ne i32 %tmp3, 0		; <i1> [#uses=1]
	br i1 %tmp4, label %bb, label %UnifiedReturnBlock

bb:		; preds = %entry
	call void @foo()
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

define void @testne3b(i32 %x, i32 %n) nounwind {
entry:
	%tmp29 = shl i32 1, %n		; <i32> [#uses=1]
	%tmp3 = and i32 %x, %tmp29
	%tmp4 = icmp ne i32 %tmp3, 0		; <i1> [#uses=1]
	br i1 %tmp4, label %bb, label %UnifiedReturnBlock

bb:		; preds = %entry
	call void @foo()
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

define void @query2(i32 %x, i32 %n) nounwind {
entry:
	%tmp29 = lshr i32 %x, %n		; <i32> [#uses=1]
	%tmp3 = and i32 %tmp29, 1		; <i32> [#uses=1]
	%tmp4 = icmp eq i32 %tmp3, 1		; <i1> [#uses=1]
	br i1 %tmp4, label %bb, label %UnifiedReturnBlock

bb:		; preds = %entry
	call void @foo()
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

define void @query2b(i32 %x, i32 %n) nounwind {
entry:
	%tmp29 = lshr i32 %x, %n		; <i32> [#uses=1]
	%tmp3 = and i32 1, %tmp29
	%tmp4 = icmp eq i32 %tmp3, 1		; <i1> [#uses=1]
	br i1 %tmp4, label %bb, label %UnifiedReturnBlock

bb:		; preds = %entry
	call void @foo()
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

define void @aquery2(i32 %x, i32 %n) nounwind {
entry:
	%tmp29 = ashr i32 %x, %n		; <i32> [#uses=1]
	%tmp3 = and i32 %tmp29, 1		; <i32> [#uses=1]
	%tmp4 = icmp eq i32 %tmp3, 1		; <i1> [#uses=1]
	br i1 %tmp4, label %bb, label %UnifiedReturnBlock

bb:		; preds = %entry
	call void @foo()
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

define void @aquery2b(i32 %x, i32 %n) nounwind {
entry:
	%tmp29 = ashr i32 %x, %n		; <i32> [#uses=1]
	%tmp3 = and i32 1, %tmp29
	%tmp4 = icmp eq i32 %tmp3, 1		; <i1> [#uses=1]
	br i1 %tmp4, label %bb, label %UnifiedReturnBlock

bb:		; preds = %entry
	call void @foo()
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

define void @query3(i32 %x, i32 %n) nounwind {
entry:
	%tmp29 = shl i32 1, %n		; <i32> [#uses=1]
	%tmp3 = and i32 %tmp29, %x		; <i32> [#uses=1]
	%tmp4 = icmp eq i32 %tmp3, %tmp29		; <i1> [#uses=1]
	br i1 %tmp4, label %bb, label %UnifiedReturnBlock

bb:		; preds = %entry
	call void @foo()
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

define void @query3b(i32 %x, i32 %n) nounwind {
entry:
	%tmp29 = shl i32 1, %n		; <i32> [#uses=1]
	%tmp3 = and i32 %x, %tmp29
	%tmp4 = icmp eq i32 %tmp3, %tmp29		; <i1> [#uses=1]
	br i1 %tmp4, label %bb, label %UnifiedReturnBlock

bb:		; preds = %entry
	call void @foo()
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

define void @query3x(i32 %x, i32 %n) nounwind {
entry:
	%tmp29 = shl i32 1, %n		; <i32> [#uses=1]
	%tmp3 = and i32 %tmp29, %x		; <i32> [#uses=1]
	%tmp4 = icmp eq i32 %tmp29, %tmp3		; <i1> [#uses=1]
	br i1 %tmp4, label %bb, label %UnifiedReturnBlock

bb:		; preds = %entry
	call void @foo()
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

define void @query3bx(i32 %x, i32 %n) nounwind {
entry:
	%tmp29 = shl i32 1, %n		; <i32> [#uses=1]
	%tmp3 = and i32 %x, %tmp29
	%tmp4 = icmp eq i32 %tmp29, %tmp3		; <i1> [#uses=1]
	br i1 %tmp4, label %bb, label %UnifiedReturnBlock

bb:		; preds = %entry
	call void @foo()
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

define void @queryne2(i32 %x, i32 %n) nounwind {
entry:
	%tmp29 = lshr i32 %x, %n		; <i32> [#uses=1]
	%tmp3 = and i32 %tmp29, 1		; <i32> [#uses=1]
	%tmp4 = icmp ne i32 %tmp3, 1		; <i1> [#uses=1]
	br i1 %tmp4, label %bb, label %UnifiedReturnBlock

bb:		; preds = %entry
	call void @foo()
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

define void @queryne2b(i32 %x, i32 %n) nounwind {
entry:
	%tmp29 = lshr i32 %x, %n		; <i32> [#uses=1]
	%tmp3 = and i32 1, %tmp29
	%tmp4 = icmp ne i32 %tmp3, 1		; <i1> [#uses=1]
	br i1 %tmp4, label %bb, label %UnifiedReturnBlock

bb:		; preds = %entry
	call void @foo()
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

define void @aqueryne2(i32 %x, i32 %n) nounwind {
entry:
	%tmp29 = ashr i32 %x, %n		; <i32> [#uses=1]
	%tmp3 = and i32 %tmp29, 1		; <i32> [#uses=1]
	%tmp4 = icmp ne i32 %tmp3, 1		; <i1> [#uses=1]
	br i1 %tmp4, label %bb, label %UnifiedReturnBlock

bb:		; preds = %entry
	call void @foo()
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

define void @aqueryne2b(i32 %x, i32 %n) nounwind {
entry:
	%tmp29 = ashr i32 %x, %n		; <i32> [#uses=1]
	%tmp3 = and i32 1, %tmp29
	%tmp4 = icmp ne i32 %tmp3, 1		; <i1> [#uses=1]
	br i1 %tmp4, label %bb, label %UnifiedReturnBlock

bb:		; preds = %entry
	call void @foo()
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

define void @queryne3(i32 %x, i32 %n) nounwind {
entry:
	%tmp29 = shl i32 1, %n		; <i32> [#uses=1]
	%tmp3 = and i32 %tmp29, %x		; <i32> [#uses=1]
	%tmp4 = icmp ne i32 %tmp3, %tmp29		; <i1> [#uses=1]
	br i1 %tmp4, label %bb, label %UnifiedReturnBlock

bb:		; preds = %entry
	call void @foo()
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

define void @queryne3b(i32 %x, i32 %n) nounwind {
entry:
	%tmp29 = shl i32 1, %n		; <i32> [#uses=1]
	%tmp3 = and i32 %x, %tmp29
	%tmp4 = icmp ne i32 %tmp3, %tmp29		; <i1> [#uses=1]
	br i1 %tmp4, label %bb, label %UnifiedReturnBlock

bb:		; preds = %entry
	call void @foo()
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

define void @queryne3x(i32 %x, i32 %n) nounwind {
entry:
	%tmp29 = shl i32 1, %n		; <i32> [#uses=1]
	%tmp3 = and i32 %tmp29, %x		; <i32> [#uses=1]
	%tmp4 = icmp ne i32 %tmp29, %tmp3		; <i1> [#uses=1]
	br i1 %tmp4, label %bb, label %UnifiedReturnBlock

bb:		; preds = %entry
	call void @foo()
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

define void @queryne3bx(i32 %x, i32 %n) nounwind {
entry:
	%tmp29 = shl i32 1, %n		; <i32> [#uses=1]
	%tmp3 = and i32 %x, %tmp29
	%tmp4 = icmp ne i32 %tmp29, %tmp3		; <i1> [#uses=1]
	br i1 %tmp4, label %bb, label %UnifiedReturnBlock

bb:		; preds = %entry
	call void @foo()
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

declare void @foo()
