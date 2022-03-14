; RUN: opt %loadPolly -polly-print-ast -disable-basic-aa -disable-output < %s | FileCheck %s --check-prefix=NOAA
; RUN: opt %loadPolly -polly-print-ast -disable-output < %s | FileCheck %s --check-prefix=BASI
; RUN: opt %loadPolly -polly-print-ast -disable-basic-aa -tbaa -disable-output < %s | FileCheck %s --check-prefix=TBAA
; RUN: opt %loadPolly -polly-print-ast -disable-basic-aa -scev-aa -disable-output < %s | FileCheck %s --check-prefix=SCEV
; RUN: opt %loadPolly -polly-print-ast -disable-basic-aa -globals-aa -disable-output < %s | FileCheck %s --check-prefix=GLOB
;
;    int A[1024];
;    float B[1024];
;
;    void jd(int N) {
;      for (int i = 0; i < N; i++)
;        A[i] = B[i];
;    }
;
; NOAA: if (1 && 0 == N <= 0 && (&MemRef_B[N] <= &MemRef_A[0] || &MemRef_A[N] <= &MemRef_B[0]))
; BASI: if (1 && 0 == N <= 0)
; TBAA: if (1 && 0 == N <= 0)
; SCEV: if (1 && 0 == N <= 0 && (&MemRef_B[N] <= &MemRef_A[0] || &MemRef_A[N] <= &MemRef_B[0]))
; GLOB: if (1 && 0 == N <= 0 && (&MemRef_B[N] <= &MemRef_A[0] || &MemRef_A[N] <= &MemRef_B[0]))
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@A = common global [1024 x i32] zeroinitializer, align 16
@B = common global [1024 x float] zeroinitializer, align 16

define void @jd(i32 %N) {
entry:
  %cmp6 = icmp sgt i32 %N, 0
  br i1 %cmp6, label %for.body.preheader, label %for.end

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds  [1024 x float],  [1024 x float]* @B, i64 0, i64 %indvars.iv
  %tmp = load float, float* %arrayidx, align 4, !tbaa !1
  %conv = fptosi float %tmp to i32
  %arrayidx2 = getelementptr inbounds [1024 x i32], [1024 x i32]* @A, i64 0, i64 %indvars.iv
  store i32 %conv, i32* %arrayidx2, align 4, !tbaa !5
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv1 = trunc i64 %indvars.iv.next to i32
  %exitcond2 = icmp eq i32 %lftr.wideiv1, %N
  br i1 %exitcond2, label %for.end.loopexit, label %for.body

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret void
}

!0 = !{!""}
!1 = !{!2, !2, i64 0}
!2 = !{!"float", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
!5 = !{!6, !6, i64 0}
!6 = !{!"int", !3, i64 0}
