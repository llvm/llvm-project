; RUN: opt < %s -basic-aa -aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s --check-prefixes=CHECK,NO-PHI-VALUES
; RUN: opt < %s -phi-values -basic-aa -aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s --check-prefixes=CHECK,PHI-VALUES

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
define void @simple(float *%src1, float * noalias %src2, i32 %n) nounwind {
entry:
  br label %loop

loop:
  %phi = phi float* [ %src2, %entry ], [ %next, %loop ]
  %idx = phi i32 [ 0, %entry ], [ %idxn, %loop ]
  %next = getelementptr inbounds float, float* %phi, i32 1
  %g = getelementptr inbounds float, float* %src1, i32 3
  %l = load float, float* %phi
  %a = fadd float %l, 1.0
  store float %a, float* %g
  %idxn = add nsw nuw i32 %idx, 1
  %cmp5 = icmp eq i32 %idxn, %n
  br i1 %cmp5, label %end, label %loop

end:
  ret void
}

; CHECK-LABEL: Function: notmust: 6 pointers, 0 call sites
; CHECK:        MustAlias:    [2 x i32]* %tab, i8* %0
; CHECK:        PartialAlias: [2 x i32]* %tab, i32* %arrayidx
; CHECK:        NoAlias:      i32* %arrayidx, i8* %0
; CHECK:        MustAlias:    [2 x i32]* %tab, i32* %arrayidx1
; CHECK:        MustAlias:    i32* %arrayidx1, i8* %0
; CHECK:        NoAlias:      i32* %arrayidx, i32* %arrayidx1
; CHECK:        MayAlias:     [2 x i32]* %tab, i32* %p.addr.05.i
; CHECK:        MayAlias:     i32* %p.addr.05.i, i8* %0
; CHECK:        MayAlias:     i32* %arrayidx, i32* %p.addr.05.i
; CHECK:        MayAlias:     i32* %arrayidx1, i32* %p.addr.05.i
; CHECK:        MayAlias:     [2 x i32]* %tab, i32* %incdec.ptr.i
; CHECK:        NoAlias:      i32* %incdec.ptr.i, i8* %0
; CHECK:        MayAlias:     i32* %arrayidx, i32* %incdec.ptr.i
; CHECK:        NoAlias:      i32* %arrayidx1, i32* %incdec.ptr.i
; CHECK:        NoAlias:      i32* %incdec.ptr.i, i32* %p.addr.05.i
define i32 @notmust() nounwind {
entry:
  %tab = alloca [2 x i32], align 4
  %0 = bitcast [2 x i32]* %tab to i8*
  %arrayidx = getelementptr inbounds [2 x i32], [2 x i32]* %tab, i32 0, i32 1
  store i32 0, i32* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds [2 x i32], [2 x i32]* %tab, i32 0, i32 0
  store i32 0, i32* %arrayidx1, align 4
  %1 = add i32 1, 1
  %cmp4.i = icmp slt i32 %1, 2
  br i1 %cmp4.i, label %while.body.i, label %f.exit

while.body.i: ; preds = %while.body.i, %entry
  %2 = phi i32 [ 1, %while.body.i ], [ %1, %entry ]
  %foo.06.i = phi i32 [ %sub.i, %while.body.i ], [ 2, %entry ]
  %p.addr.05.i = phi i32* [ %incdec.ptr.i, %while.body.i ], [ %arrayidx1, %entry ]
  %sub.i = sub nsw i32 %foo.06.i, %2
  %incdec.ptr.i = getelementptr inbounds i32, i32* %p.addr.05.i, i32 1
  store i32 %sub.i, i32* %p.addr.05.i, align 4
  %cmp.i = icmp sgt i32 %sub.i, 1
  br i1 %cmp.i, label %while.body.i, label %f.exit

f.exit: ; preds = %entry, %while.body.i
  %3 = load i32, i32* %arrayidx1, align 4
  %cmp = icmp eq i32 %3, 2
  %4 = load i32, i32* %arrayidx, align 4
  %cmp4 = icmp eq i32 %4, 1
  %or.cond = and i1 %cmp, %cmp4
  br i1 %or.cond, label %if.end, label %if.then

if.then: ; preds = %f.exit
  unreachable

if.end: ; preds = %f.exit
  ret i32 0
}

; CHECK-LABEL: Function: reverse: 6 pointers, 0 call sites
; CHECK:         MustAlias:    [10 x i32]* %tab, i8* %0
; CHECK:         MustAlias:    [10 x i32]* %tab, i32* %arrayidx
; CHECK:         MustAlias:    i32* %arrayidx, i8* %0
; CHECK:         PartialAlias: [10 x i32]* %tab, i32* %arrayidx1
; CHECK:         NoAlias:      i32* %arrayidx1, i8* %0
; CHECK:         NoAlias:      i32* %arrayidx, i32* %arrayidx1
; CHECK:         MayAlias:     [10 x i32]* %tab, i32* %p.addr.05.i
; CHECK:         MayAlias:     i32* %p.addr.05.i, i8* %0
; CHECK:         MayAlias:     i32* %arrayidx, i32* %p.addr.05.i
; CHECK:         MayAlias:     i32* %arrayidx1, i32* %p.addr.05.i
; CHECK:         MayAlias:     [10 x i32]* %tab, i32* %incdec.ptr.i
; CHECK:         MayAlias:     i32* %incdec.ptr.i, i8* %0
; CHECK:         MayAlias:     i32* %arrayidx, i32* %incdec.ptr.i
; CHECK:         MayAlias:     i32* %arrayidx1, i32* %incdec.ptr.i
; CHECK:         NoAlias:      i32* %incdec.ptr.i, i32* %p.addr.05.i
define i32 @reverse() nounwind {
entry:
  %tab = alloca [10 x i32], align 4
  %0 = bitcast [10 x i32]* %tab to i8*
  %arrayidx = getelementptr inbounds [10 x i32], [10 x i32]* %tab, i32 0, i32 0
  store i32 0, i32* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds [10 x i32], [10 x i32]* %tab, i32 0, i32 9
  store i32 0, i32* %arrayidx1, align 4
  %1 = add i32 1, 1
  %cmp4.i = icmp slt i32 %1, 2
  br i1 %cmp4.i, label %while.body.i, label %f.exit

while.body.i: ; preds = %while.body.i, %entry
  %2 = phi i32 [ 1, %while.body.i ], [ %1, %entry ]
  %foo.06.i = phi i32 [ %sub.i, %while.body.i ], [ 2, %entry ]
  %p.addr.05.i = phi i32* [ %incdec.ptr.i, %while.body.i ], [ %arrayidx1, %entry ]
  %sub.i = sub nsw i32 %foo.06.i, %2
  %incdec.ptr.i = getelementptr inbounds i32, i32* %p.addr.05.i, i32 -1
  store i32 %sub.i, i32* %p.addr.05.i, align 4
  %cmp.i = icmp sgt i32 %sub.i, 1
  br i1 %cmp.i, label %while.body.i, label %f.exit

f.exit: ; preds = %entry, %while.body.i
  %3 = load i32, i32* %arrayidx1, align 4
  %cmp = icmp eq i32 %3, 2
  %4 = load i32, i32* %arrayidx, align 4
  %cmp4 = icmp eq i32 %4, 1
  %or.cond = and i1 %cmp, %cmp4
  br i1 %or.cond, label %if.end, label %if.then

if.then: ; preds = %f.exit
  unreachable

if.end: ; preds = %f.exit
  ret i32 0
}

; CHECK-LABEL: Function: negative: 6 pointers, 1 call sites
; CHECK:         NoAlias:      [3 x i16]* %int_arr.10, i16** %argv.6.par
; CHECK:         NoAlias:      i16* %_tmp1, i16** %argv.6.par
; CHECK:         PartialAlias: [3 x i16]* %int_arr.10, i16* %_tmp1
; CHECK:         NoAlias:      i16* %ls1.9.0, i16** %argv.6.par
; CHECK:         MayAlias:     [3 x i16]* %int_arr.10, i16* %ls1.9.0
; CHECK:         MayAlias:     i16* %_tmp1, i16* %ls1.9.0
; CHECK:         NoAlias:      i16* %_tmp7, i16** %argv.6.par
; CHECK:         MayAlias:     [3 x i16]* %int_arr.10, i16* %_tmp7
; CHECK:         MayAlias:     i16* %_tmp1, i16* %_tmp7
; CHECK:         NoAlias:      i16* %_tmp7, i16* %ls1.9.0
; CHECK:         NoAlias:      i16* %_tmp11, i16** %argv.6.par
; CHECK:         PartialAlias: [3 x i16]* %int_arr.10, i16* %_tmp11
; CHECK:         NoAlias:      i16* %_tmp1, i16* %_tmp11
; CHECK:         MayAlias:     i16* %_tmp11, i16* %ls1.9.0
; CHECK:         MayAlias:     i16* %_tmp11, i16* %_tmp7
; CHECK:         Both ModRef:  Ptr: i16** %argv.6.par  <->  %_tmp16 = call i16 @call(i32 %_tmp13)
; CHECK:         NoModRef:  Ptr: [3 x i16]* %int_arr.10        <->  %_tmp16 = call i16 @call(i32 %_tmp13)
; CHECK:         NoModRef:  Ptr: i16* %_tmp1   <->  %_tmp16 = call i16 @call(i32 %_tmp13)
; CHECK:         Both ModRef:  Ptr: i16* %ls1.9.0      <->  %_tmp16 = call i16 @call(i32 %_tmp13)
; CHECK:         Both ModRef:  Ptr: i16* %_tmp7        <->  %_tmp16 = call i16 @call(i32 %_tmp13)
; CHECK:         NoModRef:  Ptr: i16* %_tmp11  <->  %_tmp16 = call i16 @call(i32 %_tmp13)
define i16 @negative(i16 %argc.5.par, i16** nocapture readnone %argv.6.par) {
  %int_arr.10 = alloca [3 x i16], align 1
  %_tmp1 = getelementptr inbounds [3 x i16], [3 x i16]* %int_arr.10, i16 0, i16 2
  br label %bb1

bb1:                                              ; preds = %bb1, %0
  %i.7.0 = phi i16 [ 2, %0 ], [ %_tmp5, %bb1 ]
  %ls1.9.0 = phi i16* [ %_tmp1, %0 ], [ %_tmp7, %bb1 ]
  store i16 %i.7.0, i16* %ls1.9.0, align 1
  %_tmp5 = add nsw i16 %i.7.0, -1
  %_tmp7 = getelementptr i16, i16* %ls1.9.0, i16 -1
  %_tmp9 = icmp sgt i16 %i.7.0, 0
  br i1 %_tmp9, label %bb1, label %bb3

bb3:                                              ; preds = %bb1
  %_tmp11 = getelementptr inbounds [3 x i16], [3 x i16]* %int_arr.10, i16 0, i16 1
  %_tmp12 = load i16, i16* %_tmp11, align 1
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
define void @dynamic_offset(i1 %c, i8* noalias %p.base) {
entry:
  %a = alloca i8
  br label %loop

loop:
  %p = phi i8* [ %p.base, %entry ], [ %p.next, %loop ]
  %offset = call i16 @call(i32 0)
  %p.next = getelementptr inbounds i8, i8* %p, i16 %offset
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
define i32* @symmetry(i32* %p.base, i1 %c) {
entry:
  br label %loop

loop:
  %p = phi i32* [ %p.base, %entry ], [ %p.next, %loop ]
  %p.next = getelementptr inbounds i32, i32* %p, i32 1
  br i1 %c, label %loop, label %exit

exit:
  %result = phi i32* [ %p.next, %loop ]
  ret i32* %result
}

; CHECK-LABEL: Function: nested_loop
; CHECK: NoAlias:  i8* %a, i8* %p.base
; CHECK: NoAlias:  i8* %a, i8* %p.outer
; CHECK: NoAlias:  i8* %a, i8* %p.outer.next
; NO-PHI-VALUES: MayAlias: i8* %a, i8* %p.inner
; PHI-VALUES: NoAlias: i8* %a, i8* %p.inner
; CHECK: NoAlias:  i8* %a, i8* %p.inner.next
define void @nested_loop(i1 %c, i1 %c2, i8* noalias %p.base) {
entry:
  %a = alloca i8
  br label %outer_loop

outer_loop:
  %p.outer = phi i8* [ %p.base, %entry ], [ %p.outer.next, %outer_loop_latch ]
  br label %inner_loop

inner_loop:
  %p.inner = phi i8* [ %p.outer, %outer_loop ], [ %p.inner.next, %inner_loop ]
  %p.inner.next = getelementptr inbounds i8, i8* %p.inner, i64 1
  br i1 %c, label %inner_loop, label %outer_loop_latch

outer_loop_latch:
  %p.outer.next = getelementptr inbounds i8, i8* %p.inner, i64 10
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
define void @nested_loop2(i1 %c, i1 %c2, i8* noalias %p.base) {
entry:
  %a = alloca i8
  br label %outer_loop

outer_loop:
  %p.outer = phi i8* [ %p.base, %entry ], [ %p.outer.next, %outer_loop_latch ]
  %p.outer.next = getelementptr inbounds i8, i8* %p.outer, i64 10
  br label %inner_loop

inner_loop:
  %p.inner = phi i8* [ %p.outer.next, %outer_loop ], [ %p.inner.next, %inner_loop ]
  %p.inner.next = getelementptr inbounds i8, i8* %p.inner, i64 1
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
; NO-PHI-VALUES: NoAlias:	i8* %a, i8* %p.inner
; PHI-VALUES: MayAlias:	i8* %a, i8* %p.inner
; CHECK: NoAlias:	i8* %a, i8* %p.inner.next
define void @nested_loop3(i1 %c, i1 %c2, i8* noalias %p.base) {
entry:
  %a = alloca i8
  br label %outer_loop

outer_loop:
  %p.outer = phi i8* [ %p.base, %entry ], [ %p.outer.next, %outer_loop_latch ]
  %p.outer.next = getelementptr inbounds i8, i8* %p.outer, i64 10
  br label %inner_loop

inner_loop:
  %p.inner = phi i8* [ %p.outer, %outer_loop ], [ %p.inner.next, %inner_loop ]
  %p.inner.next = getelementptr inbounds i8, i8* %p.inner, i64 1
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
define void @sibling_loop(i1 %c, i1 %c2, i8* noalias %p.base) {
entry:
  %a = alloca i8
  br label %loop1

loop1:
  %p1 = phi i8* [ %p.base, %entry ], [ %p1.next, %loop1 ]
  %p1.next = getelementptr inbounds i8, i8* %p1, i64 10
  br i1 %c, label %loop1, label %loop2

loop2:
  %p2 = phi i8* [ %p1.next, %loop1 ], [ %p2.next, %loop2 ]
  %p2.next = getelementptr inbounds i8, i8* %p2, i64 1
  br i1 %c2, label %loop2, label %exit

exit:
  ret void
}

; CHECK-LABEL: Function: sibling_loop2
; CHECK: NoAlias:	i8* %a, i8* %p.base
; CHECK: NoAlias:	i8* %a, i8* %p1
; CHECK: NoAlias:	i8* %a, i8* %p1.next
; NO-PHI-VALUES: NoAlias:	i8* %a, i8* %p2
; PHI-VALUES: MayAlias:	i8* %a, i8* %p2
; CHECK: NoAlias:	i8* %a, i8* %p2.next
define void @sibling_loop2(i1 %c, i1 %c2, i8* noalias %p.base) {
entry:
  %a = alloca i8
  br label %loop1

loop1:
  %p1 = phi i8* [ %p.base, %entry ], [ %p1.next, %loop1 ]
  %p1.next = getelementptr inbounds i8, i8* %p1, i64 10
  br i1 %c, label %loop1, label %loop2

loop2:
  %p2 = phi i8* [ %p1, %loop1 ], [ %p2.next, %loop2 ]
  %p2.next = getelementptr inbounds i8, i8* %p2, i64 1
  br i1 %c2, label %loop2, label %exit

exit:
  ret void
}

declare i16 @call(i32)
