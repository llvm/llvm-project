; Test that the inliner doesn't leave around dead allocas, and that it folds
; uncond branches away after it is done specializing.

; RUN: opt < %s -passes=inline -S | FileCheck %s
; RUN: opt < %s -passes='cgscc(inline)' -S | FileCheck %s

@A = weak global i32 0		; <ptr> [#uses=1]
@B = weak global i32 0		; <ptr> [#uses=1]
@C = weak global i32 0		; <ptr> [#uses=1]

define internal fastcc void @foo(i32 %X) {
entry:
	%ALL = alloca i32, align 4		; <ptr> [#uses=1]
	%tmp1 = and i32 %X, 1		; <i32> [#uses=1]
	%tmp1.upgrd.1 = icmp eq i32 %tmp1, 0		; <i1> [#uses=1]
	br i1 %tmp1.upgrd.1, label %cond_next, label %cond_true

cond_true:		; preds = %entry
	store i32 1, ptr @A
	br label %cond_next

cond_next:		; preds = %cond_true, %entry
	%tmp4 = and i32 %X, 2		; <i32> [#uses=1]
	%tmp4.upgrd.2 = icmp eq i32 %tmp4, 0		; <i1> [#uses=1]
	br i1 %tmp4.upgrd.2, label %cond_next7, label %cond_true5

cond_true5:		; preds = %cond_next
	store i32 1, ptr @B
	br label %cond_next7

cond_next7:		; preds = %cond_true5, %cond_next
	%tmp10 = and i32 %X, 4		; <i32> [#uses=1]
	%tmp10.upgrd.3 = icmp eq i32 %tmp10, 0		; <i1> [#uses=1]
	br i1 %tmp10.upgrd.3, label %cond_next13, label %cond_true11

cond_true11:		; preds = %cond_next7
	store i32 1, ptr @C
	br label %cond_next13

cond_next13:		; preds = %cond_true11, %cond_next7
	%tmp16 = and i32 %X, 8		; <i32> [#uses=1]
	%tmp16.upgrd.4 = icmp eq i32 %tmp16, 0		; <i1> [#uses=1]
	br i1 %tmp16.upgrd.4, label %UnifiedReturnBlock, label %cond_true17

cond_true17:		; preds = %cond_next13
	call void @ext( ptr %ALL )
	ret void

UnifiedReturnBlock:		; preds = %cond_next13
	ret void
}

declare void @ext(ptr)

define void @test() {
; CHECK-LABEL: @test(
; CHECK-NOT: ret
;
; FIXME: This should be a CHECK-NOT, but currently we have a bug that causes us
; to not nuke unused allocas.
; CHECK: alloca
; CHECK-NOT: ret
;
; No branches should survive the inliner's cleanup.
; CHECK-NOT: br
; CHECK: ret void

entry:
	tail call fastcc void @foo( i32 1 )
	tail call fastcc void @foo( i32 2 )
	tail call fastcc void @foo( i32 3 )
	tail call fastcc void @foo( i32 8 )
	ret void
}

declare void @f(i32 %x)

define void @inner2(i32 %x, i32 %y, i32 %z, i1 %b) {
entry:
  %cmp1 = icmp ne i32 %x, 0
  br i1 %cmp1, label %then1, label %end1

then1:
  call void @f(i32 %x)
  br label %end1

end1:
  %x2 = and i32 %x, %z
  %cmp2 = icmp sgt i32 %x2, 1
  br i1 %cmp2, label %then2, label %end2

then2:
  call void @f(i32 %x2)
  br label %end2

end2:
  %y2 = or i32 %y, %z
  %cmp3 = icmp sgt i32 %y2, 0
  br i1 %cmp3, label %then3, label %end3

then3:
  call void @f(i32 %y2)
  br label %end3

end3:
  br i1 %b, label %end3.1, label %end3.2

end3.1:
  %x3.1 = or i32 %x, 10
  br label %end3.3

end3.2:
  %x3.2 = or i32 %x, 10
  br label %end3.3

end3.3:
  %x3.3 = phi i32 [ %x3.1, %end3.1 ], [ %x3.2, %end3.2 ]
  %cmp4 = icmp slt i32 %x3.3, 1
  br i1 %cmp4, label %then4, label %end4

then4:
  call void @f(i32 %x3.3)
  br label %end4

end4:
  ret void
}

define void @outer2(i32 %z, i1 %b) {
; Ensure that after inlining, none of the blocks with a call to @f actually
; make it through inlining.
; CHECK-LABEL: define void @outer2(
; CHECK-NOT: call
; CHECK: ret void

entry:
  call void @inner2(i32 0, i32 -1, i32 %z, i1 %b)
  ret void
}

define void @PR12470_inner(i16 signext %p1, i1 %arg) nounwind uwtable {
entry:
  br i1 %arg, label %cond.true, label %cond.false

cond.true:
  br label %cond.end

cond.false:
  %conv = sext i16 %p1 to i32
  br label %cond.end

cond.end:
  %cond = phi i32 [ undef, %cond.true ], [ 0, %cond.false ]
  %tobool = icmp eq i32 %cond, 0
  br i1 %tobool, label %if.end5, label %if.then

if.then:
  ret void

if.end5:
  ret void
}

define void @PR12470_outer(i1 %arg) {
; This previously crashed during inliner cleanup and folding inner return
; instructions. Check that we don't crash and we produce a function with a single
; return instruction due to merging the returns of the inlined function.
; CHECK-LABEL: define void @PR12470_outer(
; CHECK-NOT: call
; CHECK: ret void
; CHECK-NOT: ret void
; CHECK: }

entry:
  call void @PR12470_inner(i16 signext 1, i1 1)
  ret void
}

define void @crasher_inner(i1 %arg) nounwind uwtable {
entry:
  br i1 false, label %for.end28, label %for.body6

for.body6:
  br i1 %arg, label %for.body6, label %for.cond12.for.inc26_crit_edge

for.cond12.for.inc26_crit_edge:
  br label %for.body6.1

for.end28:
  ret void

for.body6.1:
  br i1 %arg, label %for.body6.1, label %for.cond12.for.inc26_crit_edge.1

for.cond12.for.inc26_crit_edge.1:
  br label %for.body6.2

for.body6.2:
  br i1 %arg, label %for.body6.2, label %for.cond12.for.inc26_crit_edge.2

for.cond12.for.inc26_crit_edge.2:
  br label %for.end28
}

define void @crasher_outer() {
; CHECK-LABEL: @crasher_outer(
; CHECK-NOT: call
; CHECK: ret void
; CHECK-NOT: ret
; CHECK: }
entry:
  tail call void @crasher_inner(i1 1)
  ret void
}
