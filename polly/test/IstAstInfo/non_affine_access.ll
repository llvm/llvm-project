; RUN: opt %loadPolly -polly-print-ast -polly-ast-print-accesses -polly-allow-nonaffine -disable-output < %s | FileCheck %s
;
;    void non_affine_access(float A[]) {
;      for (long i = 0; i < 1024; i++)
;        A[i * i] = 1;
;    }

; CHECK: for (int c0 = 0; c0 <= 1023; c0 += 1)
; CHECK:   Stmt_bb3(
; CHECK:     /* write */  MemRef_A[*]
; CHECK:   );

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

define void @non_affine_access(ptr %A) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb6, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp7, %bb6 ]
  %exitcond = icmp ne i64 %i.0, 1024
  br i1 %exitcond, label %bb3, label %bb2

bb2:                                              ; preds = %bb1
  br label %bb8

bb3:                                              ; preds = %bb1
  %prod = mul i64 %i.0, %i.0
  %tmp5 = getelementptr inbounds float, ptr %A, i64 %prod
  store float 1.000000e+00, ptr %tmp5, align 4, !tbaa !5
  br label %bb6

bb6:                                              ; preds = %bb3
  %tmp7 = add nuw nsw i64 %i.0, 1
  br label %bb1

bb8:                                              ; preds = %bb2
  ret void
}

!llvm.ident = !{!0}

!0 = !{!"Ubuntu clang version 3.7.1-3ubuntu4 (tags/RELEASE_371/final) (based on LLVM 3.7.1)"}
!1 = !{!2, !2, i64 0}
!2 = !{!"long", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
!5 = !{!6, !6, i64 0}
!6 = !{!"float", !3, i64 0}
