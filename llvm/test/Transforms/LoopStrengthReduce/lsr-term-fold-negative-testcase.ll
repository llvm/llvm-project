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

; The test case is reduced from FFmpeg/libavfilter/ebur128.c
; Testing check if terminating value is safe to expand
%struct.FFEBUR128State = type { i32, ptr, i64, i64 }

@histogram_energy_boundaries = global [1001 x double] zeroinitializer, align 8

define void @ebur128_calc_gating_block(ptr %st, ptr %optional_output) {
; CHECK: Is not safe to expand terminating value for phi node  %i.026 = phi i64 [ 0, %for.body7.lr.ph ], [ %inc, %for.body7 ]
entry:
  %0 = load i32, ptr %st, align 8
  %conv = zext i32 %0 to i64
  %cmp28.not = icmp eq i32 %0, 0
  br i1 %cmp28.not, label %for.end13, label %for.cond2.preheader.lr.ph

for.cond2.preheader.lr.ph:                        ; preds = %entry
  %audio_data_index = getelementptr inbounds %struct.FFEBUR128State, ptr %st, i64 0, i32 3
  %1 = load i64, ptr %audio_data_index, align 8
  %div = udiv i64 %1, %conv
  %cmp525.not = icmp ult i64 %1, %conv
  %audio_data = getelementptr inbounds %struct.FFEBUR128State, ptr %st, i64 0, i32 1
  %umax = tail call i64 @llvm.umax.i64(i64 %div, i64 1)
  br label %for.cond2.preheader

for.cond2.preheader:                              ; preds = %for.cond2.preheader.lr.ph, %for.inc11
  %channel_sum.030 = phi double [ 0.000000e+00, %for.cond2.preheader.lr.ph ], [ %channel_sum.1.lcssa, %for.inc11 ]
  %c.029 = phi i64 [ 0, %for.cond2.preheader.lr.ph ], [ %inc12, %for.inc11 ]
  br i1 %cmp525.not, label %for.inc11, label %for.body7.lr.ph

for.body7.lr.ph:                                  ; preds = %for.cond2.preheader
  %2 = load ptr, ptr %audio_data, align 8
  br label %for.body7

for.body7:                                        ; preds = %for.body7.lr.ph, %for.body7
  %channel_sum.127 = phi double [ %channel_sum.030, %for.body7.lr.ph ], [ %add10, %for.body7 ]
  %i.026 = phi i64 [ 0, %for.body7.lr.ph ], [ %inc, %for.body7 ]
  %mul = mul i64 %i.026, %conv
  %add = add i64 %mul, %c.029
  %arrayidx = getelementptr inbounds double, ptr %2, i64 %add
  %3 = load double, ptr %arrayidx, align 8
  %add10 = fadd double %channel_sum.127, %3
  %inc = add nuw i64 %i.026, 1
  %exitcond.not = icmp eq i64 %inc, %umax
  br i1 %exitcond.not, label %for.inc11, label %for.body7

for.inc11:                                        ; preds = %for.body7, %for.cond2.preheader
  %channel_sum.1.lcssa = phi double [ %channel_sum.030, %for.cond2.preheader ], [ %add10, %for.body7 ]
  %inc12 = add nuw nsw i64 %c.029, 1
  %exitcond32.not = icmp eq i64 %inc12, %conv
  br i1 %exitcond32.not, label %for.end13, label %for.cond2.preheader

for.end13:                                        ; preds = %for.inc11, %entry
  %channel_sum.0.lcssa = phi double [ 0.000000e+00, %entry ], [ %channel_sum.1.lcssa, %for.inc11 ]
  %add14 = fadd double %channel_sum.0.lcssa, 0.000000e+00
  store double %add14, ptr %optional_output, align 8
  ret void
}

declare i64 @llvm.umax.i64(i64, i64)

%struct.PAKT_INFO = type { i32, i32, i32, [0 x i32] }

define i64 @alac_seek(ptr %0) {
; CHECK: Is not safe to expand terminating value for phi node  %indvars.iv.i = phi i64 [ 0, %entry ], [ %indvars.iv.next.i, %for.body.i ]
entry:
  %div = udiv i64 1, 0
  br label %for.body.i

for.body.i:                                       ; preds = %for.body.i, %entry
  %indvars.iv.i = phi i64 [ 0, %entry ], [ %indvars.iv.next.i, %for.body.i ]
  %arrayidx.i = getelementptr %struct.PAKT_INFO, ptr %0, i64 0, i32 3, i64 %indvars.iv.i
  %1 = load i32, ptr %arrayidx.i, align 4
  %indvars.iv.next.i = add i64 %indvars.iv.i, 1
  %exitcond.not.i = icmp eq i64 %indvars.iv.i, %div
  br i1 %exitcond.not.i, label %alac_pakt_block_offset.exit, label %for.body.i

alac_pakt_block_offset.exit:                      ; preds = %for.body.i
  ret i64 0
}
