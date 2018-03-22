; RUN: opt < %s -tapir2target -tapir-target=cilk -debug-abi-calls -simplifycfg -instcombine -S | FileCheck %s

source_filename = "c2islModule"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind
define void @kernel_anon([24 x [21 x [33 x float]]]* noalias nocapture nonnull readonly %A, [24 x float]* noalias nocapture nonnull readonly %B, [24 x float]* noalias nocapture nonnull readonly %C, [24 x float]* noalias nocapture nonnull readonly %D, [24 x float]* noalias nocapture nonnull %O1) local_unnamed_addr #0 {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  br label %loop_body

loop_body:                                        ; preds = %loop_latch, %entry
  %c06 = phi i64 [ 0, %entry ], [ %0, %loop_latch ]
  detach within %syncreg, label %det.achd, label %loop_latch
; CHECK-LABEL: define void @kernel_anon(
; CHECK-LABEL: loop_body.split: ; preds = %loop_body
; CHECK-NEXT: call fastcc void @kernel_anon_det.achd.cilk(
; CHECK-DAG: [24 x float]* %O1
; CHECK-DAG: i64 %c06
; CHECK-DAG: [24 x float]* %B
; CHECK-DAG: [24 x [21 x [33 x float]]]* %A
; CHECK-NEXT: {{br label %loop_latch|br label %loop_body.split.split}}

loop_latch:                                       ; preds = %synced21, %loop_body
  %0 = add nuw nsw i64 %c06, 1
  %exitcond10 = icmp eq i64 %0, 40
  br i1 %exitcond10, label %loop_exit, label %loop_body

loop_exit:                                        ; preds = %loop_latch
  sync within %syncreg, label %synced22

det.achd:                                         ; preds = %loop_body
  %syncreg5 = tail call token @llvm.syncregion.start()
  br label %loop_body2

loop_body2:                                       ; preds = %loop_latch3, %det.achd
  %c14 = phi i64 [ 0, %det.achd ], [ %1, %loop_latch3 ]
  detach within %syncreg5, label %block_exit, label %loop_latch3
; CHECK-LABEL: define internal fastcc void @kernel_anon_det.achd.cilk(
; CHECK-LABEL: loop_body2.cilk.split: ; preds = %loop_body2.cilk
; CHECK-NEXT: call fastcc void @kernel_anon_det.achd.cilk_block_exit.cilk.cilk(
; CHECK-DAG: [24 x float]* %O1.cilk
; CHECK-DAG: i64 %c06.cilk
; CHECK-DAG: i64 %c14.cilk
; CHECK-DAG: [24 x float]* %B.cilk
; CHECK-DAG: [24 x [21 x [33 x float]]]* %A.cilk
; CHECK-NEXT: {{br label %loop_latch3.cilk|br label %loop_body2.cilk.split.split}}

loop_latch3:                                      ; preds = %block_exit20, %loop_body2
  %1 = add nuw nsw i64 %c14, 1
  %exitcond9 = icmp eq i64 %1, 24
  br i1 %exitcond9, label %loop_exit4, label %loop_body2

loop_exit4:                                       ; preds = %loop_latch3
  sync within %syncreg5, label %synced21

block_exit:                                       ; preds = %loop_body2
  %2 = getelementptr inbounds [24 x float], [24 x float]* %O1, i64 %c06, i64 %c14
  store float 0.000000e+00, float* %2, align 4
  %syncreg11 = tail call token @llvm.syncregion.start()
  %3 = getelementptr inbounds [24 x float], [24 x float]* %B, i64 %c06, i64 %c14
  %4 = load float, float* %3, align 4
  br label %loop_body8

loop_body8:                                       ; preds = %loop_latch9, %block_exit
  %c22 = phi i64 [ 0, %block_exit ], [ %5, %loop_latch9 ]
  detach within %syncreg11, label %det.achd12, label %loop_latch9
; CHECK-LABEL: define internal fastcc void @kernel_anon_det.achd.cilk_block_exit.cilk.cilk(
; CHECK-LABEL: loop_body8.cilk.cilk.split: ; preds = %loop_body8.cilk.cilk
; CHECK-NEXT: call fastcc void @kernel_anon_det.achd.cilk_block_exit.cilk.cilk_det.achd12.cilk.cilk.cilk(
; CHECK-DAG: [24 x [21 x [33 x float]]]* %A.cilk.cilk
; CHECK-DAG: i64 %c06.cilk.cilk
; CHECK-DAG: i64 %c14.cilk.cilk
; CHECK-DAG: i64 %c22.cilk.cilk
; CHECK-DAG: float %2
; CHECK-DAG: float* nonnull %0
; CHECK-NEXT: {{br label %loop_latch9.cilk.cilk|br label %loop_body8.cilk.cilk.split.split}}

loop_latch9:                                      ; preds = %synced, %loop_body8
  %5 = add nuw nsw i64 %c22, 1
  %exitcond8 = icmp eq i64 %5, 21
  br i1 %exitcond8, label %loop_exit10, label %loop_body8

loop_exit10:                                      ; preds = %loop_latch9
  sync within %syncreg11, label %block_exit20

det.achd12:                                       ; preds = %loop_body8
  %syncreg17 = tail call token @llvm.syncregion.start()
  br label %loop_body14

loop_body14:                                      ; preds = %loop_latch15, %det.achd12
  %c31 = phi i64 [ 0, %det.achd12 ], [ %6, %loop_latch15 ]
  detach within %syncreg17, label %det.achd18, label %loop_latch15
; CHECK-LABEL: define internal fastcc void @kernel_anon_det.achd.cilk_block_exit.cilk.cilk_det.achd12.cilk.cilk.cilk(
; CHECK: loop_body14.cilk.cilk.cilk.split:
; CHECK-NEXT: call fastcc void @kernel_anon_det.achd.cilk_block_exit.cilk.cilk_det.achd12.cilk.cilk.cilk_det.achd18.cilk.cilk.cilk.cilk(
; CHECK-DAG: [24 x [21 x [33 x float]]]* %A.cilk.cilk.cilk
; CHECK-DAG: i64 %c06.cilk.cilk.cilk
; CHECK-DAG: i64 %c14.cilk.cilk.cilk
; CHECK-DAG: i64 %c22.cilk.cilk.cilk
; CHECK-DAG: i64 %c31.cilk.cilk.cilk
; CHECK-DAG: float %.cilk
; CHECK-DAG: float* %.cilk1
; CHECK-NEXT: {{br label %loop_latch15.cilk.cilk.cilk|loop_body14.cilk.cilk.cilk.split.split}}

loop_latch15:                                     ; preds = %det.achd18, %loop_body14
  %6 = add nuw nsw i64 %c31, 1
  %exitcond = icmp eq i64 %6, 33
  br i1 %exitcond, label %loop_exit16, label %loop_body14

loop_exit16:                                      ; preds = %loop_latch15
  sync within %syncreg17, label %synced

det.achd18:                                       ; preds = %loop_body14
  %7 = getelementptr inbounds [24 x [21 x [33 x float]]], [24 x [21 x [33 x float]]]* %A, i64 %c06, i64 %c14, i64 %c22, i64 %c31
  %8 = load float, float* %7, align 4
  %9 = fmul float %4, %8
  %10 = load float, float* %2, align 4
  %11 = fadd float %10, %9
  store float %11, float* %2, align 4
  reattach within %syncreg17, label %loop_latch15
; CHECK-LABEL: define internal fastcc void @kernel_anon_det.achd.cilk_block_exit.cilk.cilk_det.achd12.cilk.cilk.cilk_det.achd18.cilk.cilk.cilk.cilk(
; CHECK: getelementptr
; CHECK-NEXT: load
; CHECK-NEXT: fmul
; CHECK-NEXT: load
; CHECK-NEXT: fadd
; CHECK-NEXT: store
; CHECK: ret void

synced:                                           ; preds = %loop_exit16
  reattach within %syncreg11, label %loop_latch9

block_exit20:                                     ; preds = %loop_exit10
  reattach within %syncreg5, label %loop_latch3

synced21:                                         ; preds = %loop_exit4
  reattach within %syncreg, label %loop_latch

synced22:                                         ; preds = %loop_exit
  ret void
}

; Function Attrs: argmemonly nounwind
declare token @llvm.syncregion.start() #1

attributes #0 = { nounwind }
attributes #1 = { argmemonly nounwind }
