; RUN: opt < %s -loop-vectorize -enable-vplan-native-path -debug-only=loop-vectorize -S 2>&1 | FileCheck %s
; REQUIRES: asserts

; Verify that outer loops annotated only with the expected explicit
; vectorization hints are collected for vectorization instead of inner loops.

; Root C/C++ source code for all the test cases
; void foo(int *a, int *b, int N, int M)
; {
;   int i, j;
; #pragma clang loop vectorize(enable)
;   for (i = 0; i < N; i++) {
;     for (j = 0; j < M; j++) {
;       a[i*M+j] = b[i*M+j] * b[i*M+j];
;     }
;   }
; }

; Case 1: Annotated outer loop WITH vector width information must be collected.

; CHECK-LABEL: vector_width
; CHECK: LV: Loop hints: force=enabled width=4 unroll=0
; CHECK: LV: We can vectorize this outer loop!
; CHECK: LV: Using user VF 4 to build VPlans.
; CHECK-NOT: LV: Loop hints: force=?
; CHECK-NOT: LV: Found a loop: inner.body

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @vector_width(i32* nocapture %a, i32* nocapture readonly %b, i32 %N, i32 %M) local_unnamed_addr {
entry:
  %cmp32 = icmp sgt i32 %N, 0
  br i1 %cmp32, label %outer.ph, label %for.end15

outer.ph:                                   ; preds = %entry
  %cmp230 = icmp sgt i32 %M, 0
  %0 = sext i32 %M to i64
  %wide.trip.count = zext i32 %M to i64
  %wide.trip.count38 = zext i32 %N to i64
  br label %outer.body

outer.body:                                 ; preds = %outer.inc, %outer.ph
  %indvars.iv35 = phi i64 [ 0, %outer.ph ], [ %indvars.iv.next36, %outer.inc ]
  br i1 %cmp230, label %inner.ph, label %outer.inc

inner.ph:                                   ; preds = %outer.body
  %1 = mul nsw i64 %indvars.iv35, %0
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
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %outer.inc, label %inner.body

outer.inc:                                        ; preds = %inner.body, %outer.body
  %indvars.iv.next36 = add nuw nsw i64 %indvars.iv35, 1
  %exitcond39 = icmp eq i64 %indvars.iv.next36, %wide.trip.count38
  br i1 %exitcond39, label %for.end15, label %outer.body, !llvm.loop !6

for.end15:                                        ; preds = %outer.inc, %entry
  ret void
}

; Case 2: Annotated outer loop WITHOUT vector width information must be collected.

; CHECK-LABEL: case2
; CHECK: LV: Loop hints: force=enabled width=0 unroll=0
; CHECK: LV: We can vectorize this outer loop!
; CHECK: LV: Using VF 1 to build VPlans.

define void @case2(i32* nocapture %a, i32* nocapture readonly %b, i32 %N, i32 %M) local_unnamed_addr {
entry:
  %cmp32 = icmp sgt i32 %N, 0
  br i1 %cmp32, label %outer.ph, label %for.end15

outer.ph:                                          ; preds = %entry
  %cmp230 = icmp sgt i32 %M, 0
  %0 = sext i32 %M to i64
  %wide.trip.count = zext i32 %M to i64
  %wide.trip.count38 = zext i32 %N to i64
  br label %outer.body

outer.body:                                        ; preds = %outer.inc, %outer.ph
  %indvars.iv35 = phi i64 [ 0, %outer.ph ], [ %indvars.iv.next36, %outer.inc ]
  br i1 %cmp230, label %inner.ph, label %outer.inc

inner.ph:                                  ; preds = %outer.body
  %1 = mul nsw i64 %indvars.iv35, %0
  br label %inner.body

inner.body:                                        ; preds = %inner.body, %inner.ph
  %indvars.iv = phi i64 [ 0, %inner.ph ], [ %indvars.iv.next, %inner.body ]
  %2 = add nsw i64 %indvars.iv, %1
  %arrayidx = getelementptr inbounds i32, i32* %b, i64 %2
  %3 = load i32, i32* %arrayidx, align 4, !tbaa !2
  %mul8 = mul nsw i32 %3, %3
  %arrayidx12 = getelementptr inbounds i32, i32* %a, i64 %2
  store i32 %mul8, i32* %arrayidx12, align 4, !tbaa !2
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %outer.inc, label %inner.body

outer.inc:                                        ; preds = %inner.body, %outer.body
  %indvars.iv.next36 = add nuw nsw i64 %indvars.iv35, 1
  %exitcond39 = icmp eq i64 %indvars.iv.next36, %wide.trip.count38
  br i1 %exitcond39, label %for.end15, label %outer.body, !llvm.loop !9

for.end15:                                        ; preds = %outer.inc, %entry
  ret void
}

; Case 3: Annotated outer loop WITH vector width and interleave information
; doesn't have to be collected.

; CHECK-LABEL: case3
; CHECK-NOT: LV: Loop hints: force=enabled
; CHECK-NOT: LV: We can vectorize this outer loop!
; CHECK: LV: Loop hints: force=?
; CHECK: LV: Found a loop: inner.body

define void @case3(i32* nocapture %a, i32* nocapture readonly %b, i32 %N, i32 %M) local_unnamed_addr {
entry:
  %cmp32 = icmp sgt i32 %N, 0
  br i1 %cmp32, label %outer.ph, label %for.end15

outer.ph:                                         ; preds = %entry
  %cmp230 = icmp sgt i32 %M, 0
  %0 = sext i32 %M to i64
  %wide.trip.count = zext i32 %M to i64
  %wide.trip.count38 = zext i32 %N to i64
  br label %outer.body

outer.body:                                       ; preds = %outer.inc, %outer.ph
  %indvars.iv35 = phi i64 [ 0, %outer.ph ], [ %indvars.iv.next36, %outer.inc ]
  br i1 %cmp230, label %inner.ph, label %outer.inc

inner.ph:                                         ; preds = %outer.body
  %1 = mul nsw i64 %indvars.iv35, %0
  br label %inner.body

inner.body:                                       ; preds = %inner.body, %inner.ph
  %indvars.iv = phi i64 [ 0, %inner.ph ], [ %indvars.iv.next, %inner.body ]
  %2 = add nsw i64 %indvars.iv, %1
  %arrayidx = getelementptr inbounds i32, i32* %b, i64 %2
  %3 = load i32, i32* %arrayidx, align 4, !tbaa !2
  %mul8 = mul nsw i32 %3, %3
  %arrayidx12 = getelementptr inbounds i32, i32* %a, i64 %2
  store i32 %mul8, i32* %arrayidx12, align 4, !tbaa !2
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %outer.inc, label %inner.body

outer.inc:                                        ; preds = %inner.body, %outer.body
  %indvars.iv.next36 = add nuw nsw i64 %indvars.iv35, 1
  %exitcond39 = icmp eq i64 %indvars.iv.next36, %wide.trip.count38
  br i1 %exitcond39, label %for.end15, label %outer.body, !llvm.loop !11

for.end15:                                        ; preds = %outer.inc, %entry
  ret void
}

; Case 4: Outer loop without any explicit vectorization annotation doesn't have
; to be collected.

; CHECK-LABEL: case4
; CHECK-NOT: LV: Loop hints: force=enabled
; CHECK-NOT: LV: We can vectorize this outer loop!
; CHECK: LV: Loop hints: force=?
; CHECK: LV: Found a loop: inner.body

define void @case4(i32* nocapture %a, i32* nocapture readonly %b, i32 %N, i32 %M) local_unnamed_addr {
entry:
  %cmp32 = icmp sgt i32 %N, 0
  br i1 %cmp32, label %outer.ph, label %for.end15

outer.ph:                                         ; preds = %entry
  %cmp230 = icmp sgt i32 %M, 0
  %0 = sext i32 %M to i64
  %wide.trip.count = zext i32 %M to i64
  %wide.trip.count38 = zext i32 %N to i64
  br label %outer.body

outer.body:                                       ; preds = %outer.inc, %outer.ph
  %indvars.iv35 = phi i64 [ 0, %outer.ph ], [ %indvars.iv.next36, %outer.inc ]
  br i1 %cmp230, label %inner.ph, label %outer.inc

inner.ph:                                  ; preds = %outer.body
  %1 = mul nsw i64 %indvars.iv35, %0
  br label %inner.body

inner.body:                                        ; preds = %inner.body, %inner.ph
  %indvars.iv = phi i64 [ 0, %inner.ph ], [ %indvars.iv.next, %inner.body ]
  %2 = add nsw i64 %indvars.iv, %1
  %arrayidx = getelementptr inbounds i32, i32* %b, i64 %2
  %3 = load i32, i32* %arrayidx, align 4, !tbaa !2
  %mul8 = mul nsw i32 %3, %3
  %arrayidx12 = getelementptr inbounds i32, i32* %a, i64 %2
  store i32 %mul8, i32* %arrayidx12, align 4, !tbaa !2
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %outer.inc, label %inner.body

outer.inc:                                        ; preds = %inner.body, %outer.body
  %indvars.iv.next36 = add nuw nsw i64 %indvars.iv35, 1
  %exitcond39 = icmp eq i64 %indvars.iv.next36, %wide.trip.count38
  br i1 %exitcond39, label %for.end15, label %outer.body

for.end15:                                        ; preds = %outer.inc, %entry
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
; Case 1
!6 = distinct !{!6, !7, !8}
!7 = !{!"llvm.loop.vectorize.width", i32 4}
!8 = !{!"llvm.loop.vectorize.enable", i1 true}
; Case 2
!9 = distinct !{!9, !8}
; Case 3
!10 = !{!"llvm.loop.interleave.count", i32 2}
!11 = distinct !{!11, !7, !10, !8}
