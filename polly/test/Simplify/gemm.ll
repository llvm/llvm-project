; RUN: opt %loadPolly -polly-import-jscop \
; RUN:   -polly-import-jscop-postfix=transformed -polly-simplify -analyze < %s \
; RUN:   | FileCheck %s
;
;    void gemm(float A[][1024], float B[][1024], float C[][1024]) {
;      for (long i = 0; i < 1024; i++)
;        for (long j = 0; j < 1024; j++) {
;          float tmp = C[i][j];
;          for (long k = 0; k < 1024; k++)
;            tmp += A[i][k] * B[k][j];
;          C[i][j] = tmp;
;        }
;    }

; CHECK:      After accesses {
; CHECK-NEXT:     Stmt_bb13
; CHECK-NEXT:             MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 { Stmt_bb13[i0, i1, i2] -> MemRef_tmp_0__phi[] };
; CHECK-NEXT:            new: { Stmt_bb13[i0, i1, i2] -> MemRef_C[i0, i1] };
; CHECK-NEXT:             ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 { Stmt_bb13[i0, i1, i2] -> MemRef_A[i0, i2] };
; CHECK-NEXT:             ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 { Stmt_bb13[i0, i1, i2] -> MemRef_B[i2, i1] };
; CHECK-NEXT:             ReadAccess :=	[Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 { Stmt_bb13[i0, i1, i2] -> MemRef_tmp_0[] };
; CHECK-NEXT:            new: { Stmt_bb13[i0, i1, i2] -> MemRef_C[i0, i1] };
; CHECK-NEXT: }

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

define void @gemm([1024 x float]* %A, [1024 x float]* %B, [1024 x float]* %C) {
bb:
  br label %bb3

bb3:                                              ; preds = %bb26, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp27, %bb26 ]
  %exitcond2 = icmp ne i64 %i.0, 1024
  br i1 %exitcond2, label %bb5, label %bb4

bb4:                                              ; preds = %bb3
  br label %bb28

bb5:                                              ; preds = %bb3
  br label %bb6

bb6:                                              ; preds = %bb23, %bb5
  %j.0 = phi i64 [ 0, %bb5 ], [ %tmp24, %bb23 ]
  %exitcond1 = icmp ne i64 %j.0, 1024
  br i1 %exitcond1, label %bb8, label %bb7

bb7:                                              ; preds = %bb6
  br label %bb25

bb8:                                              ; preds = %bb6
  %tmp = getelementptr inbounds [1024 x float], [1024 x float]* %C, i64 %i.0, i64 %j.0
  %tmp9 = load float, float* %tmp, align 4, !tbaa !1
  br label %bb10

bb10:                                             ; preds = %bb13, %bb8
  %tmp.0 = phi float [ %tmp9, %bb8 ], [ %tmp19, %bb13 ]
  %k.0 = phi i64 [ 0, %bb8 ], [ %tmp20, %bb13 ]
  %exitcond = icmp ne i64 %k.0, 1024
  br i1 %exitcond, label %bb12, label %bb11

bb11:                                             ; preds = %bb10
  %tmp.0.lcssa = phi float [ %tmp.0, %bb10 ]
  br label %bb21

bb12:                                             ; preds = %bb10
  br label %bb13

bb13:                                             ; preds = %bb12
  %tmp14 = getelementptr inbounds [1024 x float], [1024 x float]* %A, i64 %i.0, i64 %k.0
  %tmp15 = load float, float* %tmp14, align 4, !tbaa !1
  %tmp16 = getelementptr inbounds [1024 x float], [1024 x float]* %B, i64 %k.0, i64 %j.0
  %tmp17 = load float, float* %tmp16, align 4, !tbaa !1
  %tmp18 = fmul float %tmp15, %tmp17
  %tmp19 = fadd float %tmp.0, %tmp18
  %tmp20 = add nuw nsw i64 %k.0, 1
  br label %bb10

bb21:                                             ; preds = %bb11
  %tmp22 = getelementptr inbounds [1024 x float], [1024 x float]* %C, i64 %i.0, i64 %j.0
  store float %tmp.0.lcssa, float* %tmp22, align 4, !tbaa !1
  br label %bb23

bb23:                                             ; preds = %bb21
  %tmp24 = add nuw nsw i64 %j.0, 1
  br label %bb6

bb25:                                             ; preds = %bb7
  br label %bb26

bb26:                                             ; preds = %bb25
  %tmp27 = add nuw nsw i64 %i.0, 1
  br label %bb3

bb28:                                             ; preds = %bb4
  ret void
}

declare void @llvm.lifetime.start(i64, i8* nocapture)

declare void @llvm.lifetime.end(i64, i8* nocapture)


!llvm.ident = !{!0}

!0 = !{!"Ubuntu clang version 3.7.1-3ubuntu4 (tags/RELEASE_371/final) (based on LLVM 3.7.1)"}
!1 = !{!2, !2, i64 0}
!2 = !{!"float", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
