; RUN: opt < %s -inline -S | FileCheck %s
; RUN: opt < %s -passes='cgscc(inline)' -S | FileCheck %s
; RUN: opt < %s -passes='module-inline' -S | FileCheck %s
; Test that bar and bar2 are both inlined throughout and removed.
@A = weak global i32 0		; <i32*> [#uses=1]
@B = weak global i32 0		; <i32*> [#uses=1]
@C = weak global i32 0		; <i32*> [#uses=1]

define fastcc void @foo(i32 %X) {
entry:
; CHECK-LABEL: @foo(
	%ALL = alloca i32, align 4		; <i32*> [#uses=1]
	%tmp1 = and i32 %X, 1		; <i32> [#uses=1]
	%tmp1.upgrd.1 = icmp eq i32 %tmp1, 0		; <i1> [#uses=1]
	br i1 %tmp1.upgrd.1, label %cond_next, label %cond_true

cond_true:		; preds = %entry
	store i32 1, i32* @A
	br label %cond_next

cond_next:		; preds = %cond_true, %entry
	%tmp4 = and i32 %X, 2		; <i32> [#uses=1]
	%tmp4.upgrd.2 = icmp eq i32 %tmp4, 0		; <i1> [#uses=1]
	br i1 %tmp4.upgrd.2, label %cond_next7, label %cond_true5

cond_true5:		; preds = %cond_next
	store i32 1, i32* @B
	br label %cond_next7

cond_next7:		; preds = %cond_true5, %cond_next
	%tmp10 = and i32 %X, 4		; <i32> [#uses=1]
	%tmp10.upgrd.3 = icmp eq i32 %tmp10, 0		; <i1> [#uses=1]
	br i1 %tmp10.upgrd.3, label %cond_next13, label %cond_true11

cond_true11:		; preds = %cond_next7
	store i32 1, i32* @C
	br label %cond_next13

cond_next13:		; preds = %cond_true11, %cond_next7
	%tmp16 = and i32 %X, 8		; <i32> [#uses=1]
	%tmp16.upgrd.4 = icmp eq i32 %tmp16, 0		; <i1> [#uses=1]
	br i1 %tmp16.upgrd.4, label %UnifiedReturnBlock, label %cond_true17

cond_true17:		; preds = %cond_next13
	call void @ext( i32* %ALL )
	ret void

UnifiedReturnBlock:		; preds = %cond_next13
	ret void
}

; CHECK-NOT: @bar(
define internal fastcc void @bar(i32 %X) {
entry:
	%ALL = alloca i32, align 4		; <i32*> [#uses=1]
	%tmp1 = and i32 %X, 1		; <i32> [#uses=1]
	%tmp1.upgrd.1 = icmp eq i32 %tmp1, 0		; <i1> [#uses=1]
	br i1 %tmp1.upgrd.1, label %cond_next, label %cond_true

cond_true:		; preds = %entry
	store i32 1, i32* @A
	br label %cond_next

cond_next:		; preds = %cond_true, %entry
	%tmp4 = and i32 %X, 2		; <i32> [#uses=1]
	%tmp4.upgrd.2 = icmp eq i32 %tmp4, 0		; <i1> [#uses=1]
	br i1 %tmp4.upgrd.2, label %cond_next7, label %cond_true5

cond_true5:		; preds = %cond_next
	store i32 1, i32* @B
	br label %cond_next7

cond_next7:		; preds = %cond_true5, %cond_next
	%tmp10 = and i32 %X, 4		; <i32> [#uses=1]
	%tmp10.upgrd.3 = icmp eq i32 %tmp10, 0		; <i1> [#uses=1]
	br i1 %tmp10.upgrd.3, label %cond_next13, label %cond_true11

cond_true11:		; preds = %cond_next7
	store i32 1, i32* @C
	br label %cond_next13

cond_next13:		; preds = %cond_true11, %cond_next7
	%tmp16 = and i32 %X, 8		; <i32> [#uses=1]
	%tmp16.upgrd.4 = icmp eq i32 %tmp16, 0		; <i1> [#uses=1]
	br i1 %tmp16.upgrd.4, label %UnifiedReturnBlock, label %cond_true17

cond_true17:		; preds = %cond_next13
	call void @foo( i32 %X )
	ret void

UnifiedReturnBlock:		; preds = %cond_next13
	ret void
}

define internal fastcc void @bar2(i32 %X) {
entry:
	call void @foo( i32 %X )
	ret void
}

declare void @ext(i32*)

define void @test(i32 %X) {
entry:
; CHECK: test
; CHECK-NOT: @bar(
	tail call fastcc void @bar( i32 %X )
	tail call fastcc void @bar( i32 %X )
	tail call fastcc void @bar2( i32 %X )
	tail call fastcc void @bar2( i32 %X )
	ret void
; CHECK: ret
}
