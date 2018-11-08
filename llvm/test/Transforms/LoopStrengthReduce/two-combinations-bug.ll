; RUN: opt < %s -loop-reduce -S | FileCheck %s

; This test is adapted from the n-body test of the LLVM test-suite: A bug in
; r345114 caused LSR to generate incorrect code. The test verifies that the
; induction variable generated for the inner loop depends on the induction
; variable of the outer loop.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.planet.0.3.6.11.12.15.16.17.24.25.26.33.44 = type { double, double, double, double, double, double, double }

; Function Attrs: nounwind uwtable
define dso_local void @advance(i32 %nbodies, %struct.planet.0.3.6.11.12.15.16.17.24.25.26.33.44* nocapture %bodies) local_unnamed_addr #0 {
; CHECK-LABEL: @advance(
; CHECK:  for.cond.loopexit:
; CHECK:    [[LSR_IV_NEXT:%.*]] = add i64 [[LSR_IV:%.*]], -1
; CHECK:    br label %for.body
; CHECK:  for.body:
; CHECK:    [[LSR_IV]] = phi i64 [ [[LSR_IV_NEXT]]
; CHECK:    br label %for.body3
; CHECK:  for.body3:
; CHECK:    [[LSR_IV1:%.*]] = phi i64 [ [[LSR_IV_NEXT2:%.*]], %for.body3 ], [ [[LSR_IV]], %for.body ]
; CHECK:    [[LSR_IV_NEXT2]] = add i64 [[LSR_IV1]], -1
; CHECK:    [[EXITCOND:%.*]] = icmp eq i64 [[LSR_IV_NEXT2]], 0
; CHECK:    br i1 [[EXITCOND]], label %for.cond.loopexit, label %for.body3
;
entry:
  %wide.trip.count = zext i32 %nbodies to i64
  br label %for.body

for.cond.loopexit:                                ; preds = %for.body3
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.body

for.body:                                         ; preds = %for.cond.loopexit, %entry
  %indvars.iv = phi i64 [ 1, %entry ], [ %indvars.iv.next, %for.cond.loopexit ]
  br label %for.body3

for.body3:                                        ; preds = %for.body3, %for.body
  %indvars.iv98 = phi i64 [ %indvars.iv, %for.body ], [ %indvars.iv.next99, %for.body3 ]
  %z9 = getelementptr inbounds %struct.planet.0.3.6.11.12.15.16.17.24.25.26.33.44, %struct.planet.0.3.6.11.12.15.16.17.24.25.26.33.44* %bodies, i64 %indvars.iv98, i32 2
  %tmp = load double, double* %z9, align 8, !tbaa !0
  %indvars.iv.next99 = add nuw nsw i64 %indvars.iv98, 1
  %exitcond = icmp eq i64 %indvars.iv.next99, %wide.trip.count
  br i1 %exitcond, label %for.cond.loopexit, label %for.body3
}

attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!0 = !{!1, !2, i64 16}
!1 = !{!"planet", !2, i64 0, !2, i64 8, !2, i64 16, !2, i64 24, !2, i64 32, !2, i64 40, !2, i64 48}
!2 = !{!"double", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
