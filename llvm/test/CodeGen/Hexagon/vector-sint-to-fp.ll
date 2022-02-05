; RUN: llc -march=hexagon < %s | FileCheck %s

; Test that code is generated for the vector sint_to_fp node. The compiler
; asserts with a cannot select message if the node is not expanded. When
; expanded, the generated code is very inefficient, so iwe need to find a more
; efficient code sequence to generate.

; CHECK: convert_w2sf
; CHECK: call floorf

target triple = "hexagon"

define dllexport void @f0() #0 {
b0:
  br label %b1

b1:                                               ; preds = %b1, %b0
  %v0 = phi i32 [ 0, %b0 ], [ %v17, %b1 ]
  %v1 = mul nsw i32 %v0, 2
  %v2 = add nsw i32 undef, %v1
  %v3 = insertelement <64 x i32> undef, i32 %v2, i32 0
  %v4 = shufflevector <64 x i32> %v3, <64 x i32> undef, <64 x i32> zeroinitializer
  %v5 = add nsw <64 x i32> %v4, <i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1>
  %v6 = sitofp <64 x i32> %v5 to <64 x float>
  %v7 = fmul <64 x float> %v6, <float 0x3FCFF00800000000, float 0x3FCFF00800000000, float 0x3FCFF00800000000, float 0x3FCFF00800000000, float 0x3FCFF00800000000, float 0x3FCFF00800000000, float 0x3FCFF00800000000, float 0x3FCFF00800000000, float 0x3FCFF00800000000, float 0x3FCFF00800000000, float 0x3FCFF00800000000, float 0x3FCFF00800000000, float 0x3FCFF00800000000, float 0x3FCFF00800000000, float 0x3FCFF00800000000, float 0x3FCFF00800000000, float 0x3FCFF00800000000, float 0x3FCFF00800000000, float 0x3FCFF00800000000, float 0x3FCFF00800000000, float 0x3FCFF00800000000, float 0x3FCFF00800000000, float 0x3FCFF00800000000, float 0x3FCFF00800000000, float 0x3FCFF00800000000, float 0x3FCFF00800000000, float 0x3FCFF00800000000, float 0x3FCFF00800000000, float 0x3FCFF00800000000, float 0x3FCFF00800000000, float 0x3FCFF00800000000, float 0x3FCFF00800000000, float 0x3FCFF00800000000, float 0x3FCFF00800000000, float 0x3FCFF00800000000, float 0x3FCFF00800000000, float 0x3FCFF00800000000, float 0x3FCFF00800000000, float 0x3FCFF00800000000, float 0x3FCFF00800000000, float 0x3FCFF00800000000, float 0x3FCFF00800000000, float 0x3FCFF00800000000, float 0x3FCFF00800000000, float 0x3FCFF00800000000, float 0x3FCFF00800000000, float 0x3FCFF00800000000, float 0x3FCFF00800000000, float 0x3FCFF00800000000, float 0x3FCFF00800000000, float 0x3FCFF00800000000, float 0x3FCFF00800000000, float 0x3FCFF00800000000, float 0x3FCFF00800000000, float 0x3FCFF00800000000, float 0x3FCFF00800000000, float 0x3FCFF00800000000, float 0x3FCFF00800000000, float 0x3FCFF00800000000, float 0x3FCFF00800000000, float 0x3FCFF00800000000, float 0x3FCFF00800000000, float 0x3FCFF00800000000, float 0x3FCFF00800000000>
  %v8 = fsub <64 x float> %v7, zeroinitializer
  %v9 = call <64 x float> @llvm.floor.v64f32(<64 x float> %v8)
  %v10 = fsub <64 x float> zeroinitializer, %v9
  %v11 = fptrunc <64 x float> %v10 to <64 x half>
  %v12 = call <64 x half> @llvm.fmuladd.v64f16(<64 x half> %v11, <64 x half> zeroinitializer, <64 x half> zeroinitializer)
  %v13 = fsub <64 x half> %v12, zeroinitializer
  %v14 = call <64 x half> @llvm.fmuladd.v64f16(<64 x half> zeroinitializer, <64 x half> %v13, <64 x half> zeroinitializer)
  %v15 = shufflevector <64 x half> %v14, <64 x half> undef, <128 x i32> <i32 0, i32 undef, i32 2, i32 undef, i32 4, i32 undef, i32 6, i32 undef, i32 8, i32 undef, i32 10, i32 undef, i32 12, i32 undef, i32 14, i32 undef, i32 16, i32 undef, i32 18, i32 undef, i32 20, i32 undef, i32 22, i32 undef, i32 24, i32 undef, i32 26, i32 undef, i32 28, i32 undef, i32 30, i32 undef, i32 32, i32 undef, i32 34, i32 undef, i32 36, i32 undef, i32 38, i32 undef, i32 40, i32 undef, i32 42, i32 undef, i32 44, i32 undef, i32 46, i32 undef, i32 48, i32 undef, i32 50, i32 undef, i32 52, i32 undef, i32 54, i32 undef, i32 56, i32 undef, i32 58, i32 undef, i32 60, i32 undef, i32 62, i32 undef, i32 1, i32 undef, i32 3, i32 undef, i32 5, i32 undef, i32 7, i32 undef, i32 9, i32 undef, i32 11, i32 undef, i32 13, i32 undef, i32 15, i32 undef, i32 17, i32 undef, i32 19, i32 undef, i32 21, i32 undef, i32 23, i32 undef, i32 25, i32 undef, i32 27, i32 undef, i32 29, i32 undef, i32 31, i32 undef, i32 33, i32 undef, i32 35, i32 undef, i32 37, i32 undef, i32 39, i32 undef, i32 41, i32 undef, i32 43, i32 undef, i32 45, i32 undef, i32 47, i32 undef, i32 49, i32 undef, i32 51, i32 undef, i32 53, i32 undef, i32 55, i32 undef, i32 57, i32 undef, i32 59, i32 undef, i32 61, i32 undef, i32 63, i32 undef>
  %v16 = shufflevector <128 x half> %v15, <128 x half> undef, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  call void @llvm.masked.store.v64f16.p0v64f16(<64 x half> %v16, <64 x half>* undef, i32 64, <64 x i1> <i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false>)
  %v17 = add nsw i32 %v0, 1
  br label %b1
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare <64 x float> @llvm.floor.v64f32(<64 x float>) #1

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare <64 x half> @llvm.fmuladd.v64f16(<64 x half>, <64 x half>, <64 x half>) #1

; Function Attrs: argmemonly nofree nosync nounwind willreturn writeonly
declare void @llvm.masked.store.v64f16.p0v64f16(<64 x half>, <64 x half>*, i32 immarg, <64 x i1>) #2

attributes #0 = { "target-features"="+hvxv69,+hvx-length128b,+hvx-qfloat" }
attributes #1 = { nofree nosync nounwind readnone speculatable willreturn }
attributes #2 = { argmemonly nofree nosync nounwind willreturn writeonly }
