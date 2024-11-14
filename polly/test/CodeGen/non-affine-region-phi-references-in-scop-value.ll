; RUN: opt %loadNPMPolly -passes=polly-codegen -polly-allow-nonaffine-loops \
; RUN: -S < %s | FileCheck %s

; This test verifies that values defined in another scop statement and used by
; PHI-nodes in non-affine regions are code generated correctly.

; CHECK: polly.stmt.bb3.entry:
; CHECK-NEXT:   %j.0.phiops.reload = load i64, ptr %j.0.phiops
; CHECK-NEXT:   br label %polly.stmt.bb3

; CHECK: polly.stmt.bb3:
; CHECK-NEXT:   %polly.subregion.iv = phi i32 [ %polly.subregion.iv.inc, %polly.stmt.bb9 ], [ 0, %polly.stmt.bb3.entry ]
; CHECK-NEXT:   %polly.j.0 = phi i64 [ %j.0.phiops.reload, %polly.stmt.bb3.entry ], [ %p_tmp10, %polly.stmt.bb9 ]

;    void foo(long A[], float B[], long *x) {
;      for (long i = 0; i < 1024; i++)
;        for (long j = *x; j < i * i; j++)
;          B[42]++;
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @foo(ptr %A, ptr %B, ptr %xptr) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb12, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp13, %bb12 ]
  %exitcond = icmp ne i64 %i.0, 1024
  br i1 %exitcond, label %bb2, label %bb14

bb2:                                              ; preds = %bb1
  %x = load i64, ptr %xptr
  br label %bb3

bb3:                                              ; preds = %bb9, %bb2
  %j.0 = phi i64 [ %x, %bb2 ], [ %tmp10, %bb9 ]
  %tmp = mul nsw i64 %i.0, %i.0
  %tmp4 = icmp slt i64 %j.0, %tmp
  br i1 %tmp4, label %bb5, label %bb11

bb5:                                              ; preds = %bb3
  %tmp6 = getelementptr inbounds float, ptr %B, i64 42
  %tmp7 = load float, ptr %tmp6, align 4
  %tmp8 = fadd float %tmp7, 1.000000e+00
  store float %tmp8, ptr %tmp6, align 4
  br label %bb9

bb9:                                              ; preds = %bb5
  %tmp10 = add nuw nsw i64 %j.0, 1
  br label %bb3

bb11:                                             ; preds = %bb3
  br label %bb12

bb12:                                             ; preds = %bb11
  %tmp13 = add nuw nsw i64 %i.0, 1
  br label %bb1

bb14:                                             ; preds = %bb1
  ret void
}
