; RUN: opt %loadNPMPolly '-passes=polly-import-jscop,polly-codegen' \
; RUN: -polly-import-jscop-postfix=transformed \
; RUN: -verify-dom-info -polly-allow-nonaffine-loops \
; RUN: -S < %s | FileCheck %s

; This test verifies that partial writes within non-affine loops are code
; generated correctly.

; CHECK:polly.stmt.bb3:
; CHECK-NEXT:  %polly.subregion.iv = phi i32 [ %polly.subregion.iv.inc, %polly.stmt.bb5.cont ], [ 0, %polly.stmt.bb3.entry ]
; CHECK-NEXT:  %polly.j.0 = phi i64 [ %j.0.phiops.reload, %polly.stmt.bb3.entry ], [ %p_tmp10, %polly.stmt.bb5.cont ]
; CHECK-NEXT:  %8 = zext nneg i64 %polly.indvar to i65
; CHECK-NEXT:  %9 = add nsw i64 %polly.indvar, -1
; CHECK-NEXT:  %10 = zext i64 %9 to i65
; CHECK-NEXT:  %11 = mul i65 %8, %10
; CHECK-NEXT:  %12 = lshr i65 %11, 1
; CHECK-NEXT:  %13 = trunc i65 %12 to i64
; CHECK-NEXT:  %14 = shl i64 %13, 1
; CHECK-NEXT:  %15 = add i64 %polly.indvar, %14
; CHECK-NEXT:  %p_tmp4 = icmp slt i64 %polly.j.0, %15
; CHECK-NEXT:  %polly.subregion.iv.inc = add i32 %polly.subregion.iv, 1
; CHECK-NEXT:  br i1 %p_tmp4, label %polly.stmt.bb5, label %polly.stmt.bb11.exit

; CHECK:polly.stmt.bb5:
; CHECK-NEXT:  %p_tmp6 = getelementptr inbounds float, ptr %B, i64 42
; CHECK-NEXT:  %tmp7_p_scalar_ = load float, ptr %p_tmp6
; CHECK-NEXT:  %p_tmp8 = fadd float %tmp7_p_scalar_, 1.000000e+00
; CHECK-NEXT:  %16 = icmp sle i64 %polly.indvar, 9
; CHECK-NEXT:  %polly.Stmt_bb3__TO__bb11_MayWrite2.cond = icmp ne i1 %16, false
; CHECK-NEXT:  br i1 %polly.Stmt_bb3__TO__bb11_MayWrite2.cond, label %polly.stmt.bb5.Stmt_bb3__TO__bb11_MayWrite2.partial, label %polly.stmt.bb5.cont

; CHECK:polly.stmt.bb5.Stmt_bb3__TO__bb11_MayWrite2.partial: ; preds = %polly.stmt.bb5
; CHECK-NEXT:  %polly.access.B3 = getelementptr float, ptr %B, i64 42
; CHECK-NEXT:  store float %p_tmp8, ptr %polly.access.B3
; CHECK-NEXT:  br label %polly.stmt.bb5.cont

; CHECK:polly.stmt.bb5.cont:
; CHECK-NEXT:  %p_tmp10 = add nuw nsw i64 %polly.j.0, 1
; CHECK-NEXT:  br label %polly.stmt.bb3



;    void foo(long A[], float B[], long *x) {
;      for (long i = 0; i < 1024; i++)
;        for (long j = *x; j < i * i; j++)
;          B[42]++;
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @partial_write_in_region_with_loop(ptr %A, ptr %B, ptr %xptr) {
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
  %j.0 = phi i64 [ %x, %bb2 ], [ %tmp10, %bb5 ]
  %tmp = mul nsw i64 %i.0, %i.0
  %tmp4 = icmp slt i64 %j.0, %tmp
  br i1 %tmp4, label %bb5, label %bb11

bb5:                                              ; preds = %bb3
  %tmp6 = getelementptr inbounds float, ptr %B, i64 42
  %tmp7 = load float, ptr %tmp6, align 4
  %tmp8 = fadd float %tmp7, 1.000000e+00
  store float %tmp8, ptr %tmp6, align 4
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
