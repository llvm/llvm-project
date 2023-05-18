; RUN: opt -S -passes='print<access-info>' -pass-remarks-analysis=loop-vectorize -disable-output < %s 2>&1 | FileCheck %s --check-prefix=ANALYSIS

; Test that LoopVectorize don't report 'Use #pragma loop distribute(enable) to allow loop distribution'
; when we already add #pragma clang loop distribute(enable).
;
; Testcase derived from the following C:
;
; #define M  100
; void foo (int *restrict y, int *restrict x, int *restrict indices, int n)
; {
;   int k = 3;
;   #pragma clang loop distribute(enable)
;   for (int i = 0; i < n; i++) {
;     y[i + k * M] = y[i + k* M] + 1;
;     y[i + k * (M+1)] = indices[i] + 2;
;   }
; }

define void @foo(ptr noalias nocapture noundef %y, ptr noalias nocapture noundef readnone %x, ptr noalias nocapture noundef readonly %indices, i32 noundef %n) {
; ANALYSIS: Report: unsafe dependent memory operations in loop.
; ANALYSIS: Backward loop carried data dependence that prevents store-to-load forwarding.
entry:
  %cmp22 = icmp sgt i32 %n, 0
  br i1 %cmp22, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup.loopexit:                        ; preds = %for.body
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  ret void

for.body:                                         ; preds = %for.body, %for.body.preheader
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %0 = add nuw nsw i64 %indvars.iv, 300
  %arrayidx = getelementptr inbounds i32, ptr %y, i64 %0
  %1 = load i32, ptr %arrayidx, align 4
  %add1 = add nsw i32 %1, 1
  store i32 %add1, ptr %arrayidx, align 4
  %arrayidx7 = getelementptr inbounds i32, ptr %indices, i64 %indvars.iv
  %2 = load i32, ptr %arrayidx7, align 4
  %add8 = add nsw i32 %2, 2
  %3 = add nuw nsw i64 %indvars.iv, 303
  %arrayidx12 = getelementptr inbounds i32, ptr %y, i64 %3
  store i32 %add8, ptr %arrayidx12, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup.loopexit, label %for.body, !llvm.loop !0
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.distribute.enable", i1 true}
