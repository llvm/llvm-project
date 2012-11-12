; RUN: llc < %s -march=arm -mcpu=cortex-a8 | FileCheck %s
; Radar 7872877

define void @test(float* %fltp, i32 %packedValue, float* %table) nounwind {
entry:
  %0 = load float* %fltp
  %1 = insertelement <4 x float> undef, float %0, i32 0
  %2 = shufflevector <4 x float> %1, <4 x float> undef, <4 x i32> zeroinitializer
  %3 = shl i32 %packedValue, 16
  %4 = ashr i32 %3, 30
  %.sum = add i32 %4, 4
  %5 = getelementptr inbounds float* %table, i32 %.sum
;CHECK: vldr s
  %6 = load float* %5, align 4
  %tmp11 = insertelement <4 x float> undef, float %6, i32 0
  %7 = shl i32 %packedValue, 18
  %8 = ashr i32 %7, 30
  %.sum12 = add i32 %8, 4
  %9 = getelementptr inbounds float* %table, i32 %.sum12
;CHECK: vldr s
  %10 = load float* %9, align 4
  %tmp9 = insertelement <4 x float> %tmp11, float %10, i32 1
  %11 = shl i32 %packedValue, 20
  %12 = ashr i32 %11, 30
  %.sum13 = add i32 %12, 4
  %13 = getelementptr inbounds float* %table, i32 %.sum13
;CHECK: vldr s
  %14 = load float* %13, align 4
  %tmp7 = insertelement <4 x float> %tmp9, float %14, i32 2
  %15 = shl i32 %packedValue, 22
  %16 = ashr i32 %15, 30
  %.sum14 = add i32 %16, 4
  %17 = getelementptr inbounds float* %table, i32 %.sum14
;CHECK: vldr s
  %18 = load float* %17, align 4
  %tmp5 = insertelement <4 x float> %tmp7, float %18, i32 3
  %19 = fmul <4 x float> %tmp5, %2
  %20 = bitcast float* %fltp to i8*
  tail call void @llvm.arm.neon.vst1.v4f32(i8* %20, <4 x float> %19, i32 1)
  ret void
}

declare void @llvm.arm.neon.vst1.v4f32(i8*, <4 x float>, i32) nounwind
