; Check that the loop with a floating-point reduction is vectorized
; due to llvm.loop.vectorize.reassociation.enable metadata.
; RUN: opt -passes=loop-vectorize -S < %s 2>&1 | FileCheck %s

source_filename = "FIRModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite)
define void @test_(ptr captures(none) %0, ptr readonly captures(none) %1) local_unnamed_addr #0 {
; CHECK-LABEL: define void @test_(
; CHECK-NEXT:    fadd contract <4 x float> {{.*}}
; CHECK-NEXT:    call contract float @llvm.vector.reduce.fadd.v4f32(float -0.000000e+00, <4 x float> {{.*}})
;
  %invariant.gep = getelementptr i8, ptr %1, i64 -4
  %.promoted = load float, ptr %0, align 4
  br label %3

3:                                                ; preds = %2, %3
  %indvars.iv = phi i64 [ 1, %2 ], [ %indvars.iv.next, %3 ]
  %4 = phi float [ %.promoted, %2 ], [ %6, %3 ]
  %gep = getelementptr float, ptr %invariant.gep, i64 %indvars.iv
  %5 = load float, ptr %gep, align 4
  %6 = fadd contract float %4, %5
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1001
  br i1 %exitcond.not, label %7, label %3, !llvm.loop !2

7:                                                ; preds = %3
  %.lcssa = phi float [ %6, %3 ]
  store float %.lcssa, ptr %0, align 4
  ret void
}

attributes #0 = { nofree norecurse nosync nounwind memory(argmem: readwrite) "target-cpu"="x86-64" }

!llvm.ident = !{!0}
!llvm.module.flags = !{!1}

!0 = !{!"flang version 21.0.0"}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !{!2, !3}
!3 = !{!"llvm.loop.vectorize.reassociation.enable", i1 true}

; CHECK-NOT: llvm.loop.vectorize.reassociation.enable
; CHECK: [[META3]] = !{!"llvm.loop.isvectorized", i32 1}
; CHECK: [[META4]] = !{!"llvm.loop.unroll.runtime.disable"}
