; RUN: opt -S -passes=vector-combine < %s | FileCheck %s

target datalayout = "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64"
target triple = "dxil-unknown-shadermodel6.7-library"

; Make sure vec combine min vec size is 1 instead of 4 for float.
; CHECK:@foo()
; CHECK-NEXT:%[[LD:[0-9]+]] = load <1 x float>, ptr @a, align 8
; CHECK-NEXT:%insert = shufflevector <1 x float> %[[LD]], <1 x float> poison, <2 x i32> <i32 0, i32 undef>
; CHECK-NEXT:%shuffle = shufflevector <2 x float> %insert, <2 x float> poison, <2 x i32> zeroinitializer
; CHECK-NEXT:ret <2 x float> %shuffle

@a = external local_unnamed_addr constant float

; Function Attrs: mustprogress nofree norecurse nosync nounwind readnone willreturn
define noundef <2 x float> @foo() local_unnamed_addr {
  %1 = load float, ptr @a, align 8
  %insert = insertelement <2 x float> poison, float %1, i64 0
  %shuffle = shufflevector <2 x float> %insert, <2 x float> poison, <2 x i32> zeroinitializer
  ret <2 x float> %shuffle
}
