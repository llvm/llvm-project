; RUN: opt %loadPolly -polly-allow-nonaffine -polly-process-unprofitable -polly-print-scops -disable-output < %s | FileCheck %s
;
; CHECK:        Domain :=
; CHECK-NEXT:       [srcHeight] -> { Stmt_for_cond6_preheader_us[i0] : 0 <= i0 <= -3 + srcHeight };
; CHECK-NEXT:   Schedule :=
; CHECK-NEXT:       [srcHeight] -> { Stmt_for_cond6_preheader_us[i0] -> [i0] };
; CHECK-NEXT:   ReadAccess :=	[Reduction Type: +] [Scalar: 0]
; CHECK-NEXT:       [srcHeight] -> { Stmt_for_cond6_preheader_us[i0] -> MemRef_src[o0] };
; CHECK-NEXT:   MayWriteAccess :=	[Reduction Type: +] [Scalar: 0]
; CHECK-NEXT:       [srcHeight] -> { Stmt_for_cond6_preheader_us[i0] -> MemRef_src[o0] };

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"

define void @test_case(i8* noalias nocapture readonly %src, i32 %srcHeight, i32 %srcStride) local_unnamed_addr {
entry:
  %extended = zext i32 %srcStride to i64
  %sub = add i32 %srcHeight, -1
  br label %for.cond6.preheader.us

for.cond6.preheader.us:                           ; preds = %for.cond6.preheader.us, %entry
  %srcPtr.075.us.pn = phi i8* [ %srcPtr.075.us, %for.cond6.preheader.us ], [ %src, %entry ]
  %y.072.us = phi i32 [ %inc37.us, %for.cond6.preheader.us ], [ 1, %entry ]
  %srcPtr.075.us = getelementptr inbounds i8, i8* %srcPtr.075.us.pn, i64 %extended

  %0 = load i8, i8* %srcPtr.075.us, align 1, !tbaa !0
  %1 = add i8 %0, 1
  store i8 %1, i8* %srcPtr.075.us, align 1, !tbaa !0

  %inc37.us = add nuw i32 %y.072.us, 1
  %exitcond78 = icmp eq i32 %inc37.us, %sub
  br i1 %exitcond78, label %for.cond.cleanup.loopexit, label %for.cond6.preheader.us

for.cond.cleanup.loopexit:                        ; preds = %for.cond6.preheader.us
  ret void
}

!0 = !{!1, !1, i64 0}
!1 = !{!"omnipotent char", !2, i64 0}
!2 = !{!"Simple C++ TBAA"}
!3 = !{!4, !4, i64 0}
!4 = !{!"float", !1, i64 0}
