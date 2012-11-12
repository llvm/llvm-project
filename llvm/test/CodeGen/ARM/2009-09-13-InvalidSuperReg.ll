; RUN: llc < %s -march=arm -mattr=+neon -mcpu=cortex-a9

define arm_aapcs_vfpcc <4 x float> @foo(i8* nocapture %pBuffer, i32 %numItems) nounwind {
  %1 = ptrtoint i8* %pBuffer to i32

  %lsr.iv2641 = inttoptr i32 %1 to float*
  %tmp29 = add i32 %1, 4
  %tmp2930 = inttoptr i32 %tmp29 to float*
  %tmp31 = add i32 %1, 8
  %tmp3132 = inttoptr i32 %tmp31 to float*
  %tmp33 = add i32 %1, 12
  %tmp3334 = inttoptr i32 %tmp33 to float*
  %tmp35 = add i32 %1, 16
  %tmp3536 = inttoptr i32 %tmp35 to float*
  %tmp37 = add i32 %1, 20
  %tmp3738 = inttoptr i32 %tmp37 to float*
  %tmp39 = add i32 %1, 24
  %tmp3940 = inttoptr i32 %tmp39 to float*
  %2 = load float* %lsr.iv2641, align 4
  %3 = load float* %tmp2930, align 4
  %4 = load float* %tmp3132, align 4
  %5 = load float* %tmp3334, align 4
  %6 = load float* %tmp3536, align 4
  %7 = load float* %tmp3738, align 4
  %8 = load float* %tmp3940, align 4
  %9 = insertelement <4 x float> undef, float %6, i32 0
  %10 = shufflevector <4 x float> %9, <4 x float> undef, <4 x i32> zeroinitializer
  %11 = insertelement <4 x float> %10, float %7, i32 1
  %12 = insertelement <4 x float> %11, float %8, i32 2
  %13 = insertelement <4 x float> undef, float %2, i32 0
  %14 = shufflevector <4 x float> %13, <4 x float> undef, <4 x i32> zeroinitializer
  %15 = insertelement <4 x float> %14, float %3, i32 1
  %16 = insertelement <4 x float> %15, float %4, i32 2
  %17 = insertelement <4 x float> %16, float %5, i32 3
  %18 = fsub <4 x float> zeroinitializer, %12
  %19 = shufflevector <4 x float> %18, <4 x float> undef, <4 x i32> zeroinitializer
  %20 = shufflevector <4 x float> %17, <4 x float> undef, <2 x i32> <i32 0, i32 1>
  %21 = shufflevector <2 x float> %20, <2 x float> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1>

  ret <4 x float> %21
}
