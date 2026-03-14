; RUN: opt < %s -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

; CHECK-LABEL: Function: simple: 5 pointers, 0 call sites
; CHECK:         NoAlias:      float* %src1, float* %src2
; CHECK:         NoAlias:      float* %phi, float* %src1
; CHECK:         MayAlias:     float* %phi, float* %src2
; CHECK:         NoAlias:      float* %next, float* %src1
; CHECK:         MayAlias:     float* %next, float* %src2
; CHECK:         NoAlias:      float* %next, float* %phi
; CHECK:         NoAlias:      float* %g, float* %src1
; CHECK:         NoAlias:      float* %g, float* %src2
; CHECK:         NoAlias:      float* %g, float* %phi
; CHECK:         NoAlias:      float* %g, float* %next
define void @simple(ptr %src1, ptr noalias %src2, i32 %n) nounwind {
entry:
  load float, ptr %src1
  load float, ptr %src2
  br label %loop

loop:
  %phi = phi ptr [ %src2, %entry ], [ %next, %loop ]
  %idx = phi i32 [ 0, %entry ], [ %idxn, %loop ]
  %next = getelementptr inbounds float, ptr %phi, i32 1
  %g = getelementptr inbounds float, ptr %src1, i32 3
  %l = load float, ptr %phi
  load float, ptr %next
  %a = fadd float %l, 1.0
  store float %a, ptr %g
  %idxn = add nsw nuw i32 %idx, 1
  %cmp5 = icmp eq i32 %idxn, %n
  br i1 %cmp5, label %end, label %loop

end:
  ret void
}

; CHECK-LABEL: Function: notmust: 6 pointers, 0 call sites
; CHECK: MustAlias:	i8* %tab, [2 x i32]* %tab
; CHECK: PartialAlias (off -4):	i32* %arrayidx, [2 x i32]* %tab
; CHECK: NoAlias:	i32* %arrayidx, i8* %tab
; CHECK: MustAlias:	i32* %tab, [2 x i32]* %tab
; CHECK: MustAlias:	i32* %tab, i8* %tab
; CHECK: NoAlias:	i32* %arrayidx, i32* %tab
; CHECK: MayAlias:	i32* %incdec.ptr.i, [2 x i32]* %tab
; CHECK: NoAlias:	i32* %incdec.ptr.i, i8* %tab
; CHECK: MayAlias:	i32* %arrayidx, i32* %incdec.ptr.i
; CHECK: NoAlias:	i32* %incdec.ptr.i, i32* %tab
; CHECK: MayAlias:	i32* %p.addr.05.i, [2 x i32]* %tab
; CHECK: MayAlias:	i32* %p.addr.05.i, i8* %tab
; CHECK: MayAlias:	i32* %arrayidx, i32* %p.addr.05.i
; CHECK: MayAlias:	i32* %p.addr.05.i, i32* %tab
; CHECK: NoAlias:	i32* %incdec.ptr.i, i32* %p.addr.05.i
define i32 @notmust() nounwind {
entry:
  %tab = alloca [2 x i32], align 4
  %ignore1 = load [2 x i32], ptr %tab
  %ignore2 = load i8, ptr %tab
  %arrayidx = getelementptr inbounds [2 x i32], ptr %tab, i32 0, i32 1
  store i32 0, ptr %arrayidx, align 4
  store i32 0, ptr %tab, align 4
  %0 = add i32 1, 1
  %cmp4.i = icmp slt i32 %0, 2
  br i1 %cmp4.i, label %while.body.i, label %f.exit

while.body.i: ; preds = %while.body.i, %entry
  %1 = phi i32 [ 1, %while.body.i ], [ %0, %entry ]
  %foo.06.i = phi i32 [ %sub.i, %while.body.i ], [ 2, %entry ]
  %p.addr.05.i = phi ptr [ %incdec.ptr.i, %while.body.i ], [ %tab, %entry ]
  %sub.i = sub nsw i32 %foo.06.i, %1
  %incdec.ptr.i = getelementptr inbounds i32, ptr %p.addr.05.i, i32 1
  %ignore3 = load i32, ptr %incdec.ptr.i
  store i32 %sub.i, ptr %p.addr.05.i, align 4
  %cmp.i = icmp sgt i32 %sub.i, 1
  br i1 %cmp.i, label %while.body.i, label %f.exit

f.exit: ; preds = %entry, %while.body.i
  %2 = load i32, ptr %tab, align 4
  %cmp = icmp eq i32 %2, 2
  %3 = load i32, ptr %arrayidx, align 4
  %cmp4 = icmp eq i32 %3, 1
  %or.cond = and i1 %cmp, %cmp4
  br i1 %or.cond, label %if.end, label %if.then

if.then: ; preds = %f.exit
  unreachable

if.end: ; preds = %f.exit
  ret i32 0
}

; CHECK-LABEL: Function: reverse: 6 pointers, 0 call sites
; CHECK: MustAlias:	i8* %tab, [10 x i32]* %tab
; CHECK: MustAlias:	i32* %tab, [10 x i32]* %tab
; CHECK: MustAlias:	i32* %tab, i8* %tab
; CHECK: PartialAlias (off -36):	i32* %arrayidx1, [10 x i32]* %tab
; CHECK: NoAlias:	i32* %arrayidx1, i8* %tab
; CHECK: NoAlias:	i32* %arrayidx1, i32* %tab
; CHECK: MayAlias:	i32* %incdec.ptr.i, [10 x i32]* %tab
; CHECK: MayAlias:	i32* %incdec.ptr.i, i8* %tab
; CHECK: MayAlias:	i32* %incdec.ptr.i, i32* %tab
; CHECK: MayAlias:	i32* %arrayidx1, i32* %incdec.ptr.i
; CHECK: MayAlias:	i32* %p.addr.05.i, [10 x i32]* %tab
; CHECK: MayAlias:	i32* %p.addr.05.i, i8* %tab
; CHECK: MayAlias:	i32* %p.addr.05.i, i32* %tab
; CHECK: MayAlias:	i32* %arrayidx1, i32* %p.addr.05.i
; CHECK: NoAlias:	i32* %incdec.ptr.i, i32* %p.addr.05.i
define i32 @reverse() nounwind {
entry:
  %tab = alloca [10 x i32], align 4
  %ignore1 = load [10 x i32], ptr %tab
  %ignore2 = load i8, ptr %tab
  store i32 0, ptr %tab, align 4
  %arrayidx1 = getelementptr inbounds [10 x i32], ptr %tab, i32 0, i32 9
  store i32 0, ptr %arrayidx1, align 4
  %0 = add i32 1, 1
  %cmp4.i = icmp slt i32 %0, 2
  br i1 %cmp4.i, label %while.body.i, label %f.exit

while.body.i: ; preds = %while.body.i, %entry
  %1 = phi i32 [ 1, %while.body.i ], [ %0, %entry ]
  %foo.06.i = phi i32 [ %sub.i, %while.body.i ], [ 2, %entry ]
  %p.addr.05.i = phi ptr [ %incdec.ptr.i, %while.body.i ], [ %arrayidx1, %entry ]
  %sub.i = sub nsw i32 %foo.06.i, %1
  %incdec.ptr.i = getelementptr inbounds i32, ptr %p.addr.05.i, i32 -1
  %ignore3 = load i32, ptr %incdec.ptr.i
  store i32 %sub.i, ptr %p.addr.05.i, align 4
  %cmp.i = icmp sgt i32 %sub.i, 1
  br i1 %cmp.i, label %while.body.i, label %f.exit

f.exit: ; preds = %entry, %while.body.i
  %2 = load i32, ptr %arrayidx1, align 4
  %cmp = icmp eq i32 %2, 2
  %3 = load i32, ptr %tab, align 4
  %cmp4 = icmp eq i32 %3, 1
  %or.cond = and i1 %cmp, %cmp4
  br i1 %or.cond, label %if.end, label %if.then

if.then: ; preds = %f.exit
  unreachable

if.end: ; preds = %f.exit
  ret i32 0
}

; CHECK-LABEL: Function: negative: 5 pointers, 1 call sites
; CHECK: PartialAlias (off -4):	i16* %_tmp1, [3 x i16]* %int_arr.10
; CHECK: MayAlias:	[3 x i16]* %int_arr.10, i16* %ls1.9.0
; CHECK: MayAlias:	i16* %_tmp1, i16* %ls1.9.0
; CHECK: MayAlias:	i16* %_tmp7, [3 x i16]* %int_arr.10
; CHECK: MayAlias:	i16* %_tmp1, i16* %_tmp7
; CHECK: NoAlias:	i16* %_tmp7, i16* %ls1.9.0
; CHECK: PartialAlias (off -2):	i16* %_tmp11, [3 x i16]* %int_arr.10
; CHECK: NoAlias:	i16* %_tmp1, i16* %_tmp11
; CHECK: MayAlias:	i16* %_tmp11, i16* %ls1.9.0
; CHECK: MayAlias:	i16* %_tmp11, i16* %_tmp7
; CHECK: NoModRef:  Ptr: [3 x i16]* %int_arr.10	<->  %_tmp16 = call i16 @call(i32 %_tmp13)
; CHECK: NoModRef:  Ptr: i16* %_tmp1	<->  %_tmp16 = call i16 @call(i32 %_tmp13)
; CHECK: Both ModRef:  Ptr: i16* %ls1.9.0	<->  %_tmp16 = call i16 @call(i32 %_tmp13)
; CHECK: Both ModRef:  Ptr: i16* %_tmp7	<->  %_tmp16 = call i16 @call(i32 %_tmp13)
; CHECK: NoModRef:  Ptr: i16* %_tmp11	<->  %_tmp16 = call i16 @call(i32 %_tmp13)
define i16 @negative(i16 %argc.5.par) {
  %int_arr.10 = alloca [3 x i16], align 1
  load [3 x i16], ptr %int_arr.10
  %_tmp1 = getelementptr inbounds [3 x i16], ptr %int_arr.10, i16 0, i16 2
  load i16, ptr %_tmp1
  br label %bb1

bb1:                                              ; preds = %bb1, %0
  %i.7.0 = phi i16 [ 2, %0 ], [ %_tmp5, %bb1 ]
  %ls1.9.0 = phi ptr [ %_tmp1, %0 ], [ %_tmp7, %bb1 ]
  store i16 %i.7.0, ptr %ls1.9.0, align 1
  %_tmp5 = add nsw i16 %i.7.0, -1
  %_tmp7 = getelementptr i16, ptr %ls1.9.0, i16 -1
  load i16, ptr %_tmp7
  %_tmp9 = icmp sgt i16 %i.7.0, 0
  br i1 %_tmp9, label %bb1, label %bb3

bb3:                                              ; preds = %bb1
  %_tmp11 = getelementptr inbounds [3 x i16], ptr %int_arr.10, i16 0, i16 1
  %_tmp12 = load i16, ptr %_tmp11, align 1
  %_tmp13 = sext i16 %_tmp12 to i32
  %_tmp16 = call i16 @call(i32 %_tmp13)
  %_tmp18.not = icmp eq i16 %_tmp12, 1
  br i1 %_tmp18.not, label %bb5, label %bb4

bb4:                                              ; preds = %bb3
  ret i16 1

bb5:                                              ; preds = %bb3, %bb4
  ret i16 0
}

; CHECK-LABEL: Function: dynamic_offset
; CHECK: NoAlias:  i8* %a, i8* %p.base
; CHECK: MayAlias: i8* %p, i8* %p.base
; CHECK: NoAlias:  i8* %a, i8* %p
; CHECK: MayAlias: i8* %p.base, i8* %p.next
; CHECK: NoAlias:  i8* %a, i8* %p.next
; CHECK: MayAlias: i8* %p, i8* %p.next
define void @dynamic_offset(i1 %c, ptr noalias %p.base) {
entry:
  %a = alloca i8
  load i8, ptr %p.base
  load i8, ptr %a
  br label %loop

loop:
  %p = phi ptr [ %p.base, %entry ], [ %p.next, %loop ]
  %offset = call i16 @call(i32 0)
  %p.next = getelementptr inbounds i8, ptr %p, i16 %offset
  load i8, ptr %p
  load i8, ptr %p.next
  br i1 %c, label %loop, label %exit

exit:
  ret void
}

; TODO: Currently yields an asymmetric result.
; CHECK-LABEL: Function: symmetry
; CHECK: MayAlias:  i32* %p, i32* %p.base
; CHECK: MayAlias:  i32* %p.base, i32* %p.next
; CHECK: NoAlias:   i32* %p, i32* %p.next
; CHECK: MayAlias:  i32* %p.base, i32* %result
; CHECK: NoAlias:   i32* %p, i32* %result
; CHECK: MustAlias: i32* %p.next, i32* %result
define ptr @symmetry(ptr %p.base, i1 %c) {
entry:
  load i32, ptr %p.base
  br label %loop

loop:
  %p = phi ptr [ %p.base, %entry ], [ %p.next, %loop ]
  %p.next = getelementptr inbounds i32, ptr %p, i32 1
  load i32, ptr %p
  load i32, ptr %p.next
  br i1 %c, label %loop, label %exit

exit:
  %result = phi ptr [ %p.next, %loop ]
  load i32, ptr %result
  ret ptr %result
}

; FIXME: %a and %p.inner do not alias.
; CHECK-LABEL: Function: nested_loop
; CHECK: NoAlias:  i8* %a, i8* %p.base
; CHECK: NoAlias:  i8* %a, i8* %p.outer
; CHECK: MayAlias: i8* %a, i8* %p.inner
; CHECK: NoAlias:  i8* %a, i8* %p.inner.next
; CHECK: NoAlias:  i8* %a, i8* %p.outer.next
define void @nested_loop(i1 %c, i1 %c2, ptr noalias %p.base) {
entry:
  %a = alloca i8
  load i8, ptr %p.base
  load i8, ptr %a
  br label %outer_loop

outer_loop:
  %p.outer = phi ptr [ %p.base, %entry ], [ %p.outer.next, %outer_loop_latch ]
  load i8, ptr %p.outer
  br label %inner_loop

inner_loop:
  %p.inner = phi ptr [ %p.outer, %outer_loop ], [ %p.inner.next, %inner_loop ]
  %p.inner.next = getelementptr inbounds i8, ptr %p.inner, i64 1
  load i8, ptr %p.inner
  load i8, ptr %p.inner.next
  br i1 %c, label %inner_loop, label %outer_loop_latch

outer_loop_latch:
  %p.outer.next = getelementptr inbounds i8, ptr %p.inner, i64 10
  load i8, ptr %p.outer.next
  br i1 %c2, label %outer_loop, label %exit

exit:
  ret void
}

; Same as the previous test case, but avoiding phi of phi.
; CHECK-LABEL: Function: nested_loop2
; CHECK: NoAlias:  i8* %a, i8* %p.base
; CHECK: NoAlias:  i8* %a, i8* %p.outer
; CHECK: NoAlias:  i8* %a, i8* %p.outer.next
; CHECK: MayAlias: i8* %a, i8* %p.inner
; CHECK: NoAlias:  i8* %a, i8* %p.inner.next
; TODO: (a, p.inner) could be NoAlias
define void @nested_loop2(i1 %c, i1 %c2, ptr noalias %p.base) {
entry:
  %a = alloca i8
  load i8, ptr %p.base
  load i8, ptr %a
  br label %outer_loop

outer_loop:
  %p.outer = phi ptr [ %p.base, %entry ], [ %p.outer.next, %outer_loop_latch ]
  %p.outer.next = getelementptr inbounds i8, ptr %p.outer, i64 10
  load i8, ptr %p.outer
  load i8, ptr %p.outer.next
  br label %inner_loop

inner_loop:
  %p.inner = phi ptr [ %p.outer.next, %outer_loop ], [ %p.inner.next, %inner_loop ]
  %p.inner.next = getelementptr inbounds i8, ptr %p.inner, i64 1
  load i8, ptr %p.inner
  load i8, ptr %p.inner.next
  br i1 %c, label %inner_loop, label %outer_loop_latch

outer_loop_latch:
  br i1 %c2, label %outer_loop, label %exit

exit:
  ret void
}

; CHECK-LABEL: Function: nested_loop3
; CHECK: NoAlias:	i8* %a, i8* %p.base
; CHECK: NoAlias:	i8* %a, i8* %p.outer
; CHECK: NoAlias:	i8* %a, i8* %p.outer.next
; CHECK: NoAlias:	i8* %a, i8* %p.inner
; CHECK: NoAlias:	i8* %a, i8* %p.inner.next
define void @nested_loop3(i1 %c, i1 %c2, ptr noalias %p.base) {
entry:
  %a = alloca i8
  load i8, ptr %p.base
  load i8, ptr %a
  br label %outer_loop

outer_loop:
  %p.outer = phi ptr [ %p.base, %entry ], [ %p.outer.next, %outer_loop_latch ]
  %p.outer.next = getelementptr inbounds i8, ptr %p.outer, i64 10
  load i8, ptr %p.outer
  load i8, ptr %p.outer.next
  br label %inner_loop

inner_loop:
  %p.inner = phi ptr [ %p.outer, %outer_loop ], [ %p.inner.next, %inner_loop ]
  %p.inner.next = getelementptr inbounds i8, ptr %p.inner, i64 1
  load i8, ptr %p.inner
  load i8, ptr %p.inner.next
  br i1 %c, label %inner_loop, label %outer_loop_latch

outer_loop_latch:
  br i1 %c2, label %outer_loop, label %exit

exit:
  ret void
}

; CHECK-LABEL: Function: sibling_loop
; CHECK: NoAlias:	i8* %a, i8* %p.base
; CHECK: NoAlias:	i8* %a, i8* %p1
; CHECK: NoAlias:	i8* %a, i8* %p1.next
; CHECK: MayAlias:	i8* %a, i8* %p2
; CHECK: NoAlias:	i8* %a, i8* %p2.next
; TODO: %p2 does not alias %a
define void @sibling_loop(i1 %c, i1 %c2, ptr noalias %p.base) {
entry:
  %a = alloca i8
  load i8, ptr %p.base
  load i8, ptr %a
  br label %loop1

loop1:
  %p1 = phi ptr [ %p.base, %entry ], [ %p1.next, %loop1 ]
  %p1.next = getelementptr inbounds i8, ptr %p1, i64 10
  load i8, ptr %p1
  load i8, ptr %p1.next
  br i1 %c, label %loop1, label %loop2

loop2:
  %p2 = phi ptr [ %p1.next, %loop1 ], [ %p2.next, %loop2 ]
  %p2.next = getelementptr inbounds i8, ptr %p2, i64 1
  load i8, ptr %p2
  load i8, ptr %p2.next
  br i1 %c2, label %loop2, label %exit

exit:
  ret void
}

; CHECK-LABEL: Function: sibling_loop2
; CHECK: NoAlias:	i8* %a, i8* %p.base
; CHECK: NoAlias:	i8* %a, i8* %p1
; CHECK: NoAlias:	i8* %a, i8* %p1.next
; CHECK: NoAlias:	i8* %a, i8* %p2
; CHECK: NoAlias:	i8* %a, i8* %p2.next
define void @sibling_loop2(i1 %c, i1 %c2, ptr noalias %p.base) {
entry:
  %a = alloca i8
  load i8, ptr %p.base
  load i8, ptr %a
  br label %loop1

loop1:
  %p1 = phi ptr [ %p.base, %entry ], [ %p1.next, %loop1 ]
  %p1.next = getelementptr inbounds i8, ptr %p1, i64 10
  load i8, ptr %p1
  load i8, ptr %p1.next
  br i1 %c, label %loop1, label %loop2

loop2:
  %p2 = phi ptr [ %p1, %loop1 ], [ %p2.next, %loop2 ]
  %p2.next = getelementptr inbounds i8, ptr %p2, i64 1
  load i8, ptr %p2
  load i8, ptr %p2.next
  br i1 %c2, label %loop2, label %exit

exit:
  ret void
}

; CHECK: MustAlias: i8* %a, i8* %phi
define void @phi_contains_self() {
entry:
  %a = alloca i32
  load i8, ptr %a
  br label %loop

loop:
  %phi = phi ptr [ %phi, %loop ], [ %a, %entry ]
  load i8, ptr %phi
  br label %loop
}

declare i16 @call(i32)
