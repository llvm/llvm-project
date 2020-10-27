; RUN: opt < %s -mattr=+mve,+mve.fp -loop-vectorize -S | FileCheck %s --check-prefixes=DEFAULT
; RUN: opt < %s -mattr=+mve,+mve.fp -loop-vectorize -prefer-predicate-over-epilog -S | FileCheck %s --check-prefixes=TAILPRED

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv8.1m.main-arm-none-eabi"

; When TP is disabled, this test can vectorize with a VF of 16.
; When TP is enabled, this test should vectorize with a VF of 8.
;
; DEFAULT: load <16 x i8>, <16 x i8>*
; DEFAULT: sext <16 x i8> %{{.*}} to <16 x i16>
; DEFAULT: add <16 x i16>
; DEFAULT-NOT: llvm.masked.load
; DEFAULT-NOT: llvm.masked.store
;
; TAILPRED: llvm.masked.load.v8i8.p0v8i8
; TAILPRED: sext <8 x i8> %{{.*}} to <8 x i16>
; TAILPRED: add <8 x i16>
; TAILPRED: call void @llvm.masked.store.v8i8.p0v8i8
; TAILPRED-NOT: load <16 x i8>, <16 x i8>*

define i32 @tp_reduces_vf(i8* nocapture %0, i32 %1, i8** %input) {
  %3 = load i8*, i8** %input, align 8
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
  %9 = add nsw i32 %8, -320
  %10 = add nsw i32 %8, -321
  %11 = getelementptr inbounds i8, i8* %3, i32 %10
  %12 = load i8, i8* %11, align 1
  %13 = sext i8 %12 to i32
  %14 = getelementptr inbounds i8, i8* %3, i32 %9
  %15 = load i8, i8* %14, align 1
  %16 = sext i8 %15 to i32
  %17 = add nsw i32 %8, -319
  %18 = getelementptr inbounds i8, i8* %3, i32 %17
  %19 = load i8, i8* %18, align 1
  %20 = sext i8 %19 to i32
  %21 = add nsw i32 %8, -1
  %22 = getelementptr inbounds i8, i8* %3, i32 %21
  %23 = load i8, i8* %22, align 1
  %24 = sext i8 %23 to i32
  %25 = getelementptr inbounds i8, i8* %3, i32 %8
  %26 = load i8, i8* %25, align 1
  %27 = sext i8 %26 to i32
  %28 = mul nsw i32 %27, 255
  %29 = add nuw nsw i32 %8, 1
  %30 = getelementptr inbounds i8, i8* %3, i32 %29
  %31 = load i8, i8* %30, align 1
  %32 = sext i8 %31 to i32
  %33 = add nuw nsw i32 %8, 320
  %34 = add nuw nsw i32 %8, 319
  %35 = getelementptr inbounds i8, i8* %3, i32 %34
  %36 = load i8, i8* %35, align 1
  %37 = sext i8 %36 to i32
  %38 = getelementptr inbounds i8, i8* %3, i32 %33
  %39 = load i8, i8* %38, align 1
  %40 = sext i8 %39 to i32
  %41 = add nuw nsw i32 %8, 321
  %42 = getelementptr inbounds i8, i8* %3, i32 %41
  %43 = load i8, i8* %42, align 1
  %44 = sext i8 %43 to i32
  %reass.add = add nsw i32 %16, %13
  %reass.add44 = add nsw i32 %reass.add, %20
  %reass.add45 = add nsw i32 %reass.add44, %24
  %45 = add nsw i32 %reass.add45, %32
  %46 = add nsw i32 %45, %37
  %47 = add nsw i32 %46, %40
  %reass.add46 = add nsw i32 %47, %44
  %reass.mul = mul nsw i32 %reass.add46, -28
  %48 = add nsw i32 %reass.mul, %28
  %49 = lshr i32 %48, 8
  %50 = trunc i32 %49 to i8
  %51 = getelementptr inbounds i8, i8* %0, i32 %8
  store i8 %50, i8* %51, align 1
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
