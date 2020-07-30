; RUN: opt < %s -basic-aa -aa-eval -print-all-alias-modref-info -basic-aa-recphi -disable-output 2>&1 | FileCheck %s

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
