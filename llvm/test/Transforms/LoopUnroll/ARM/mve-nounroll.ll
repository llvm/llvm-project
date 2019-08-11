; RUN: opt -mtriple=thumbv8.1m.main -mattr=+mve.fp -loop-unroll -S < %s -o - | FileCheck %s

; CHECK-LABEL: @loopfn
; CHECK: vector.body:
; CHECK:   br i1 %7, label %middle.block, label %vector.body, !llvm.loop !0
; CHECK: middle.block:
; CHECK:   br i1 %cmp.n, label %for.cond.cleanup, label %for.body.preheader13
; CHECK: for.body:
; CHECK:   br i1 %exitcond, label %for.cond.cleanup.loopexit, label %for.body, !llvm.loop !2

define void @loopfn(float* %s1, float* %s2, float* %d, i32 %n) {
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %min.iters.check = icmp ult i32 %n, 4
  br i1 %min.iters.check, label %for.body.preheader13, label %vector.ph

for.body.preheader13:                             ; preds = %middle.block, %for.body.preheader
  %i.011.ph = phi i32 [ 0, %for.body.preheader ], [ %n.vec, %middle.block ]
  br label %for.body

vector.ph:                                        ; preds = %for.body.preheader
  %n.vec = and i32 %n, -4
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i32 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %0 = getelementptr inbounds float, float* %s1, i32 %index
  %1 = bitcast float* %0 to <4 x float>*
  %wide.load = load <4 x float>, <4 x float>* %1, align 4
  %2 = getelementptr inbounds float, float* %s2, i32 %index
  %3 = bitcast float* %2 to <4 x float>*
  %wide.load12 = load <4 x float>, <4 x float>* %3, align 4
  %4 = fadd fast <4 x float> %wide.load12, %wide.load
  %5 = getelementptr inbounds float, float* %d, i32 %index
  %6 = bitcast float* %5 to <4 x float>*
  store <4 x float> %4, <4 x float>* %6, align 4
  %index.next = add i32 %index, 4
  %7 = icmp eq i32 %index.next, %n.vec
  br i1 %7, label %middle.block, label %vector.body, !llvm.loop !0

middle.block:                                     ; preds = %vector.body
  %cmp.n = icmp eq i32 %n.vec, %n
  br i1 %cmp.n, label %for.cond.cleanup, label %for.body.preheader13

for.cond.cleanup.loopexit:                        ; preds = %for.body
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %middle.block, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader13, %for.body
  %i.011 = phi i32 [ %add3, %for.body ], [ %i.011.ph, %for.body.preheader13 ]
  %arrayidx = getelementptr inbounds float, float* %s1, i32 %i.011
  %8 = load float, float* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds float, float* %s2, i32 %i.011
  %9 = load float, float* %arrayidx1, align 4
  %add = fadd fast float %9, %8
  %arrayidx2 = getelementptr inbounds float, float* %d, i32 %i.011
  store float %add, float* %arrayidx2, align 4
  %add3 = add nuw nsw i32 %i.011, 1
  %exitcond = icmp eq i32 %add3, %n
  br i1 %exitcond, label %for.cond.cleanup.loopexit, label %for.body, !llvm.loop !2
}



; CHECK-LABEL: @nested
; CHECK: for.outer:
; CHECK:   br label %vector.body
; CHECK: vector.body:
; CHECK:   br i1 %8, label %for.latch, label %vector.body, !llvm.loop !0
; CHECK: for.latch:
; CHECK:   br i1 %exitcond34, label %for.cond.cleanup.loopexit, label %for.outer

define void @nested(float* %s1, float* %s2, float* %d, i32 %n) {
entry:
  %cmp31 = icmp eq i32 %n, 0
  br i1 %cmp31, label %for.cond.cleanup, label %for.outer.preheader

for.outer.preheader:                 ; preds = %entry
  %min.iters.check = icmp ult i32 %n, 4
  %n.vec = and i32 %n, -4
  %cmp.n = icmp eq i32 %n.vec, %n
  br label %for.outer

for.outer:                           ; preds = %for.outer.preheader, %for.cond1.for.cond.cleanup3_crit_edge.us
  %j.032.us = phi i32 [ %inc.us, %for.latch ], [ 0, %for.outer.preheader ]
  %mul.us = mul i32 %j.032.us, %n
  br label %vector.body

vector.body:                                      ; preds = %for.outer, %vector.body
  %index = phi i32 [ %index.next, %vector.body ], [ 0, %for.outer ]
  %0 = add i32 %index, %mul.us
  %1 = getelementptr inbounds float, float* %s1, i32 %0
  %2 = bitcast float* %1 to <4 x float>*
  %wide.load = load <4 x float>, <4 x float>* %2, align 4
  %3 = getelementptr inbounds float, float* %s2, i32 %0
  %4 = bitcast float* %3 to <4 x float>*
  %wide.load35 = load <4 x float>, <4 x float>* %4, align 4
  %5 = fadd fast <4 x float> %wide.load35, %wide.load
  %6 = getelementptr inbounds float, float* %d, i32 %0
  %7 = bitcast float* %6 to <4 x float>*
  store <4 x float> %5, <4 x float>* %7, align 4
  %index.next = add i32 %index, 4
  %8 = icmp eq i32 %index.next, %n.vec
  br i1 %8, label %for.latch, label %vector.body, !llvm.loop !0

for.latch:                           ; preds = %vector.body, %for.outer
  %i.030.us.ph = phi i32 [ %n.vec, %vector.body ]
  %inc.us = add nuw i32 %j.032.us, 1
  %exitcond34 = icmp eq i32 %inc.us, %n
  br i1 %exitcond34, label %for.cond.cleanup.loopexit, label %for.outer

for.cond.cleanup.loopexit:
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond1.for.cond.cleanup3_crit_edge.us, %entry
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.isvectorized", i32 1}
!2 = distinct !{!2, !3, !1}
!3 = !{!"llvm.loop.unroll.runtime.disable"}
