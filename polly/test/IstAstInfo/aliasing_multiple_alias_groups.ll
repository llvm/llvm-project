; RUN: opt %loadNPMPolly '-passes=print<polly-ast>' -aa-pipeline=      -disable-output < %s | FileCheck %s --check-prefix=NOAA
; RUN: opt %loadNPMPolly '-passes=print<polly-ast>' -aa-pipeline=tbaa -disable-output < %s | FileCheck %s --check-prefix=TBAA
;
;    void jd(int *Int0, int *Int1, float *Float0, float *Float1) {
;      for (int i = 0; i < 1024; i++) {
;        Int0[i] = Int1[i];
;        Float0[i] = Float1[i];
;      }
;    }
;
; NOAA:      if (1 && (
; NOAA-DAG:    &MemRef_Int0[1024] <= &MemRef_Float0[0] || &MemRef_Float0[1024] <= &MemRef_Int0[0]
; NOAA-DAG:    &MemRef_Int1[1024] <= &MemRef_Float0[0] || &MemRef_Float0[1024] <= &MemRef_Int1[0]
; NOAA-DAG:    &MemRef_Float1[1024] <= &MemRef_Float0[0] || &MemRef_Float0[1024] <= &MemRef_Float1[0]
; NOAA-DAG:    &MemRef_Int1[1024] <= &MemRef_Int0[0] || &MemRef_Int0[1024] <= &MemRef_Int1[0]
; NOAA-DAG:    &MemRef_Float1[1024] <= &MemRef_Int0[0] || &MemRef_Int0[1024] <= &MemRef_Float1[0]
; NOAA:      ))
;
; TBAA:      if (1 && (
; TBAA-DAG:    &MemRef_Int1[1024] <= &MemRef_Int0[0] || &MemRef_Int0[1024] <= &MemRef_Int1[0]
; TBAA-DAG:    &MemRef_Float1[1024] <= &MemRef_Float0[0] || &MemRef_Float0[1024] <= &MemRef_Float1[0]
; TBAA:      ))
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @jd(ptr nocapture %Int0, ptr nocapture readonly %Int1, ptr nocapture %Float0, ptr nocapture readonly %Float1) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %Int1, i64 %indvars.iv
  %tmp = load i32, ptr %arrayidx, align 4, !tbaa !0
  %arrayidx2 = getelementptr inbounds i32, ptr %Int0, i64 %indvars.iv
  store i32 %tmp, ptr %arrayidx2, align 4, !tbaa !0
  %arrayidx4 = getelementptr inbounds float, ptr %Float1, i64 %indvars.iv
  %tmp1 = load float, ptr %arrayidx4, align 4, !tbaa !4
  %arrayidx6 = getelementptr inbounds float, ptr %Float0, i64 %indvars.iv
  store float %tmp1, ptr %arrayidx6, align 4, !tbaa !4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

!0 = !{!1, !1, i64 0}
!1 = !{!"int", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
!4 = !{!5, !5, i64 0}
!5 = !{!"float", !2, i64 0}
