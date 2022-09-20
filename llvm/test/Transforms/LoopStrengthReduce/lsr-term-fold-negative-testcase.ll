; REQUIRES: asserts
; RUN: opt < %s -passes="loop-reduce" -S -debug -lsr-term-fold 2>&1 | FileCheck %s

target datalayout = "e-p:32:32:32-n32"

define i32 @loop_variant(ptr %ar, i32 %n, i32 %m) {
; CHECK: Cannot fold on backedge that is loop variant
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.cond, %entry
  %n.addr.0 = phi i32 [ %n, %entry ], [ %mul, %for.cond ]
  %cmp = icmp slt i32 %n.addr.0, %m
  %mul = shl nsw i32 %n.addr.0, 1
  br i1 %cmp, label %for.cond, label %for.end

for.end:                                          ; preds = %for.cond
  ret i32 %n.addr.0
}

define i32 @nested_loop(ptr %ar, i32 %n, i32 %m, i32 %o) {
; CHECK: Cannot fold on backedge that is loop variant
; CHECK: Cannot fold on non-innermost loop
entry:
  %cmp15 = icmp sgt i32 %o, 0
  br i1 %cmp15, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup3, %entry
  %cnt.0.lcssa = phi i32 [ 0, %entry ], [ %cnt.1.lcssa, %for.cond.cleanup3 ]
  ret i32 %cnt.0.lcssa

for.body:                                         ; preds = %entry, %for.cond.cleanup3
  %i.017 = phi i32 [ %inc6, %for.cond.cleanup3 ], [ 0, %entry ]
  %cnt.016 = phi i32 [ %cnt.1.lcssa, %for.cond.cleanup3 ], [ 0, %entry ]
  %sub = sub nsw i32 %n, %i.017
  %cmp212 = icmp slt i32 %sub, %m
  br i1 %cmp212, label %for.body4, label %for.cond.cleanup3

for.cond.cleanup3:                                ; preds = %for.body4, %for.body
  %cnt.1.lcssa = phi i32 [ %cnt.016, %for.body ], [ %inc, %for.body4 ]
  %inc6 = add nuw nsw i32 %i.017, 1
  %cmp = icmp slt i32 %inc6, %o
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.body4:                                        ; preds = %for.body, %for.body4
  %j.014 = phi i32 [ %mul, %for.body4 ], [ %sub, %for.body ]
  %cnt.113 = phi i32 [ %inc, %for.body4 ], [ %cnt.016, %for.body ]
  %inc = add nsw i32 %cnt.113, 1
  %mul = shl nsw i32 %j.014, 1
  %cmp2 = icmp slt i32 %mul, %m
  br i1 %cmp2, label %for.body4, label %for.cond.cleanup3
}

; The terminating condition folding transformation cannot find the ptr IV
; because it checks if the value comes from the LoopPreheader. %mark is from
; the function argument, so it is not qualified for the transformation.
define void @no_iv_to_help(ptr %mark, i32 signext %length) {
; CHECK: Cannot find other AddRec IV to help folding
entry:
  %tobool.not3 = icmp eq i32 %length, 0
  br i1 %tobool.not3, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %entry, %for.body
  %i.05 = phi i32 [ %dec, %for.body ], [ %length, %entry ]
  %dst.04 = phi ptr [ %incdec.ptr, %for.body ], [ %mark, %entry ]
  %0 = load ptr, ptr %dst.04, align 8
  call ptr @foo(ptr %0)
  %incdec.ptr = getelementptr inbounds ptr, ptr %dst.04, i64 1
  %dec = add nsw i32 %i.05, -1
  %tobool.not = icmp eq i32 %dec, 0
  br i1 %tobool.not, label %for.cond.cleanup, label %for.body
}

declare void @foo(ptr)

define void @NonAddRecIV(ptr %a) {
; CHECK: SCEV of phi '  %lsr.iv = phi i32 [ %lsr.iv.next, %for.body ], [ 1, %entry ]'
; CHECK: is not an affine add recursion, not qualified for the terminating condition folding.
entry:
  %uglygep = getelementptr i8, ptr %a, i32 84
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %lsr.iv1 = phi ptr [ %uglygep2, %for.body ], [ %uglygep, %entry ]
  %lsr.iv = phi i32 [ %lsr.iv.next, %for.body ], [ 1, %entry ]
  store i32 1, ptr %lsr.iv1, align 4
  %lsr.iv.next = mul nsw i32 %lsr.iv, 2
  %uglygep2 = getelementptr i8, ptr %lsr.iv1, i64 4
  %exitcond.not = icmp eq i32 %lsr.iv.next, 65536
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

@fp_inc = common global float 0.000000e+00, align 4

define void @NonSCEVableIV(float %init, float* %A, i32 %N) {
; CHECK: IV of phi '  %x.05 = phi float [ %init, %entry ], [ %add, %for.body ]'
; CHECK: is not SCEV-able, not qualified for the terminating condition folding.
entry:
  %0 = load float, float* @fp_inc, align 4
  br label %for.body

for.body:                                         ; preds = %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %x.05 = phi float [ %init, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds float, float* %A, i64 %indvars.iv
  store float %x.05, float* %arrayidx, align 4
  %add = fsub float %x.05, %0
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %N
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.end
  ret void
}

define void @NonIcmpEqNe(ptr %a) {
; CHECK: Cannot fold on branching condition that is not an ICmpInst::eq / ICmpInst::ne
entry:
  %uglygep = getelementptr i8, ptr %a, i64 84
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %lsr.iv1 = phi ptr [ %uglygep2, %for.body ], [ %uglygep, %entry ]
  %lsr.iv = phi i64 [ %lsr.iv.next, %for.body ], [ 379, %entry ]
  store i32 1, ptr %lsr.iv1, align 4
  %lsr.iv.next = add nsw i64 %lsr.iv, -1
  %uglygep2 = getelementptr i8, ptr %lsr.iv1, i64 4
  %exitcond.not = icmp sle i64 %lsr.iv.next, 0
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

define void @TermCondMoreThanOneUse(ptr %a) {
;CHECK: Cannot replace terminating condition with more than one use
entry:
  %uglygep = getelementptr i8, ptr %a, i64 84
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %lsr.iv1 = phi ptr [ %uglygep2, %for.body ], [ %uglygep, %entry ]
  %lsr.iv = phi i64 [ %lsr.iv.next, %for.body ], [ 379, %entry ]
  store i32 1, ptr %lsr.iv1, align 4
  %lsr.iv.next = add nsw i64 %lsr.iv, -1
  %uglygep2 = getelementptr i8, ptr %lsr.iv1, i64 4
  %exitcond.not = icmp eq i64 %lsr.iv.next, 0
  %dummy = select i1 %exitcond.not, i8 0, i8 1
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}
