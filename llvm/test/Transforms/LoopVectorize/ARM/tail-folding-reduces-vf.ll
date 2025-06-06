; RUN: opt < %s -mattr=+mve,+mve.fp -passes=loop-vectorize -tail-predication=disabled -S | FileCheck %s --check-prefixes=DEFAULT
; RUN: opt < %s -mattr=+mve,+mve.fp -passes=loop-vectorize -prefer-predicate-over-epilogue=predicate-else-scalar-epilogue -S | FileCheck %s --check-prefixes=TAILPRED

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv8.1m.main-arm-none-eabi"

; When TP is disabled, this test can vectorize with a VF of 16.
; When TP is enabled, this test should vectorize with a VF of 8.
;
; DEFAULT: load <16 x i8>, ptr
; DEFAULT: sext <16 x i8> %{{.*}} to <16 x i16>
; DEFAULT: add <16 x i16>
; DEFAULT-NOT: llvm.masked.load
; DEFAULT-NOT: llvm.masked.store
;
; TAILPRED: llvm.masked.load.v8i8.p0
; TAILPRED: sext <8 x i8> %{{.*}} to <8 x i16>
; TAILPRED: add <8 x i16>
; TAILPRED: call void @llvm.masked.store.v8i8.p0
; TAILPRED-NOT: load <16 x i8>, ptr

define i32 @tp_reduces_vf(ptr nocapture %0, i32 %1, ptr %input) {
  %3 = load ptr, ptr %input, align 8
  %4 = sext i32 %1 to i64
  %5 = icmp eq i32 %1, 0
  br i1 %5, label %._crit_edge, label %.preheader47.preheader

.preheader47.preheader:
  br label %.preheader47

.preheader47:
  %.050 = phi i64 [ %54, %53 ], [ 0, %.preheader47.preheader ]
  br label %.preheader

._crit_edge.loopexit:
  br label %._crit_edge

._crit_edge:
  ret i32 0

.preheader:
  %indvars.iv51 = phi i32 [ 1, %.preheader47 ], [ %indvars.iv.next52, %52 ]
  %6 = mul nuw nsw i32 %indvars.iv51, 320
  br label %7

7:
  %indvars.iv = phi i32 [ 1, %.preheader ], [ %indvars.iv.next, %7 ]
  %8 = add nuw nsw i32 %6, %indvars.iv
  %17 = add nsw i32 %8, -319
  %18 = getelementptr inbounds i8, ptr %3, i32 %17
  %19 = load i8, ptr %18, align 1
  %20 = sext i8 %19 to i32
  %25 = getelementptr inbounds i8, ptr %3, i32 %8
  %26 = load i8, ptr %25, align 1
  %27 = sext i8 %26 to i32
  %28 = mul nsw i32 %27, 255
  %29 = add nuw nsw i32 %8, 1
  %30 = getelementptr inbounds i8, ptr %3, i32 %29
  %31 = load i8, ptr %30, align 1
  %32 = sext i8 %31 to i32
  %33 = add nuw nsw i32 %8, 320
  %38 = getelementptr inbounds i8, ptr %3, i32 %33
  %39 = load i8, ptr %38, align 1
  %40 = sext i8 %39 to i32
  %reass.add = add nsw i32 %20, %20
  %reass.add44 = add nsw i32 %reass.add, %20
  %reass.add45 = add nsw i32 %reass.add44, %20
  %45 = add nsw i32 %reass.add45, %32
  %46 = add nsw i32 %45, %32
  %47 = add nsw i32 %46, %40
  %reass.add46 = add nsw i32 %47, %40
  %reass.mul = mul nsw i32 %reass.add46, -28
  %48 = add nsw i32 %reass.mul, %28
  %49 = lshr i32 %48, 8
  %50 = trunc i32 %49 to i8
  %51 = getelementptr inbounds i8, ptr %0, i32 %8
  store i8 %50, ptr %51, align 1
  %indvars.iv.next = add nuw nsw i32 %indvars.iv, 1
  %exitcond = icmp eq i32 %indvars.iv.next, 319
  br i1 %exitcond, label %52, label %7

52:
  %indvars.iv.next52 = add nuw nsw i32 %indvars.iv51, 1
  %exitcond53 = icmp eq i32 %indvars.iv.next52, 239
  br i1 %exitcond53, label %53, label %.preheader

53:
  %54 = add nuw i64 %.050, 1
  %55 = icmp ult i64 %54, %4
  br i1 %55, label %.preheader47, label %._crit_edge.loopexit
}
