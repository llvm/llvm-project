; RUN: opt < %s -passes=loop-vectorize -enable-vplan-native-path -pass-remarks-analysis=loop-vectorize -debug-only=loop-vectorize -S 2>&1 | FileCheck %s
; REQUIRES: asserts

; Verify that LV bails out on explicit vectorization outer loops that contain
; divergent inner loops.

; Root C/C++ source code for all the test cases
; void foo(int *a, int *b, int N, int M)
; {
;   int i, j;
; #pragma clang loop vectorize(enable) vectorize_width(8)
;   for (i = 0; i < N; i++) {
;     // Tested inner loop. It will be replaced per test.
;     for (j = 0; j < M; j++) {
;       a[i*M+j] = b[i*M+j] * b[i*M+j];
;     }
;   }
; }

; Case 1 (for (j = i; j < M; j++)): Inner loop with divergent IV start.

; CHECK-LABEL: iv_start
; CHECK: LV: Not vectorizing: Outer loop contains divergent loops.
; CHECK: LV: Not vectorizing: Unsupported outer loop.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @iv_start(i32* nocapture %a, i32* nocapture readonly %b, i32 %N, i32 %M) local_unnamed_addr {
entry:
  %cmp33 = icmp sgt i32 %N, 0
  br i1 %cmp33, label %outer.ph, label %for.end15

outer.ph:                                   ; preds = %entry
  %0 = sext i32 %M to i64
  %wide.trip.count = zext i32 %M to i64
  %wide.trip.count41 = zext i32 %N to i64
  br label %outer.body

outer.body:                                 ; preds = %outer.inc, %outer.ph
  %indvars.iv38 = phi i64 [ 0, %outer.ph ], [ %indvars.iv.next39, %outer.inc ]
  %cmp231 = icmp slt i64 %indvars.iv38, %0
  br i1 %cmp231, label %inner.ph, label %outer.inc

inner.ph:                                   ; preds = %outer.body
  %1 = mul nsw i64 %indvars.iv38, %0
  br label %inner.body

inner.body:                                 ; preds = %inner.body, %inner.ph
  %indvars.iv35 = phi i64 [ %indvars.iv38, %inner.ph ], [ %indvars.iv.next36, %inner.body ]
  %2 = add nsw i64 %indvars.iv35, %1
  %arrayidx = getelementptr inbounds i32, i32* %b, i64 %2
  %3 = load i32, i32* %arrayidx, align 4, !tbaa !2
  %mul8 = mul nsw i32 %3, %3
  %arrayidx12 = getelementptr inbounds i32, i32* %a, i64 %2
  store i32 %mul8, i32* %arrayidx12, align 4, !tbaa !2
  %indvars.iv.next36 = add nuw nsw i64 %indvars.iv35, 1
  %exitcond = icmp eq i64 %indvars.iv.next36, %wide.trip.count
  br i1 %exitcond, label %outer.inc, label %inner.body

outer.inc:                                  ; preds = %inner.body, %outer.body
  %indvars.iv.next39 = add nuw nsw i64 %indvars.iv38, 1
  %exitcond42 = icmp eq i64 %indvars.iv.next39, %wide.trip.count41
  br i1 %exitcond42, label %for.end15, label %outer.body, !llvm.loop !6

for.end15:                                  ; preds = %outer.inc, %entry
  ret void
}


; Case 2 (for (j = 0; j < i; j++)): Inner loop with divergent upper-bound.

; CHECK-LABEL: loop_ub
; CHECK: LV: Not vectorizing: Outer loop contains divergent loops.
; CHECK: LV: Not vectorizing: Unsupported outer loop.

define void @loop_ub(i32* nocapture %a, i32* nocapture readonly %b, i32 %N, i32 %M) local_unnamed_addr {
entry:
  %cmp32 = icmp sgt i32 %N, 0
  br i1 %cmp32, label %outer.ph, label %for.end15

outer.ph:                                   ; preds = %entry
  %0 = sext i32 %M to i64
  %wide.trip.count41 = zext i32 %N to i64
  br label %outer.body

outer.body:                                 ; preds = %outer.inc, %outer.ph
  %indvars.iv38 = phi i64 [ 0, %outer.ph ], [ %indvars.iv.next39, %outer.inc ]
  %cmp230 = icmp eq i64 %indvars.iv38, 0
  br i1 %cmp230, label %outer.inc, label %inner.ph

inner.ph:                                   ; preds = %outer.body
  %1 = mul nsw i64 %indvars.iv38, %0
  br label %inner.body

inner.body:                                 ; preds = %inner.body, %inner.ph
  %indvars.iv = phi i64 [ 0, %inner.ph ], [ %indvars.iv.next, %inner.body ]
  %2 = add nsw i64 %indvars.iv, %1
  %arrayidx = getelementptr inbounds i32, i32* %b, i64 %2
  %3 = load i32, i32* %arrayidx, align 4, !tbaa !2
  %mul8 = mul nsw i32 %3, %3
  %arrayidx12 = getelementptr inbounds i32, i32* %a, i64 %2
  store i32 %mul8, i32* %arrayidx12, align 4, !tbaa !2
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %indvars.iv38
  br i1 %exitcond, label %outer.inc, label %inner.body

outer.inc:                                  ; preds = %inner.body, %outer.body
  %indvars.iv.next39 = add nuw nsw i64 %indvars.iv38, 1
  %exitcond42 = icmp eq i64 %indvars.iv.next39, %wide.trip.count41
  br i1 %exitcond42, label %for.end15, label %outer.body, !llvm.loop !6

for.end15:                                  ; preds = %outer.inc, %entry
  ret void
}

; Case 3 (for (j = 0; j < M; j+=i)): Inner loop with divergent step.

; CHECK-LABEL: iv_step
; CHECK: LV: Not vectorizing: Outer loop contains divergent loops.
; CHECK: LV: Not vectorizing: Unsupported outer loop.

define void @iv_step(i32* nocapture %a, i32* nocapture readonly %b, i32 %N, i32 %M) local_unnamed_addr {
entry:
  %cmp33 = icmp sgt i32 %N, 0
  br i1 %cmp33, label %outer.ph, label %for.end15

outer.ph:                                   ; preds = %entry
  %cmp231 = icmp sgt i32 %M, 0
  %0 = sext i32 %M to i64
  %wide.trip.count = zext i32 %N to i64
  br label %outer.body

outer.body:                                 ; preds = %for.inc14, %outer.ph
  %indvars.iv39 = phi i64 [ 0, %outer.ph ], [ %indvars.iv.next40, %for.inc14 ]
  br i1 %cmp231, label %inner.ph, label %for.inc14

inner.ph:                                   ; preds = %outer.body
  %1 = mul nsw i64 %indvars.iv39, %0
  br label %inner.body

inner.body:                                 ; preds = %inner.ph, %inner.body
  %indvars.iv36 = phi i64 [ 0, %inner.ph ], [ %indvars.iv.next37, %inner.body ]
  %2 = add nsw i64 %indvars.iv36, %1
  %arrayidx = getelementptr inbounds i32, i32* %b, i64 %2
  %3 = load i32, i32* %arrayidx, align 4, !tbaa !2
  %mul8 = mul nsw i32 %3, %3
  %arrayidx12 = getelementptr inbounds i32, i32* %a, i64 %2
  store i32 %mul8, i32* %arrayidx12, align 4, !tbaa !2
  %indvars.iv.next37 = add nuw nsw i64 %indvars.iv36, %indvars.iv39
  %cmp2 = icmp slt i64 %indvars.iv.next37, %0
  br i1 %cmp2, label %inner.body, label %for.inc14

for.inc14:                                 ; preds = %inner.body, %outer.body
  %indvars.iv.next40 = add nuw nsw i64 %indvars.iv39, 1
  %exitcond = icmp eq i64 %indvars.iv.next40, %wide.trip.count
  br i1 %exitcond, label %for.end15, label %outer.body, !llvm.loop !6

for.end15:                                 ; preds = %for.inc14, %entry
  ret void
}

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 6.0.0"}
!2 = !{!3, !3, i64 0}
!3 = !{!"int", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = distinct !{!6, !7, !8}
!7 = !{!"llvm.loop.vectorize.width", i32 8}
!8 = !{!"llvm.loop.vectorize.enable", i1 true}
