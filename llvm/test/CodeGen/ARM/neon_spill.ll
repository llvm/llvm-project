; RUN: llc < %s -verify-machineinstrs
; RUN: llc < %s -verify-machineinstrs -O0
; PR12177
;
; This test case spills a QQQQ register.
;
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:64:128-a0:0:64-n32-S64"
target triple = "armv7-none-linux-gnueabi"

%0 = type { %1*, i32, i32, i32, i8 }
%1 = type { i32 (...)** }
%2 = type { i8*, i8*, i8*, i32 }
%3 = type { %4 }
%4 = type { i32 (...)**, %2, %4*, i8, i8 }

declare arm_aapcs_vfpcc void @func1(%0*, float* nocapture, float* nocapture, %2*) nounwind

declare arm_aapcs_vfpcc %0** @func2()

declare arm_aapcs_vfpcc %2* @func3(%2*, %2*, i32)

declare arm_aapcs_vfpcc %2** @func4()

define arm_aapcs_vfpcc void @foo(%3* nocapture) nounwind align 2 {
  call void @llvm.arm.neon.vst4.v4i32(i8* undef, <4 x i32> <i32 0, i32 1065353216, i32 1073741824, i32 1077936128>, <4 x i32> <i32 1082130432, i32 1084227584, i32 1086324736, i32 1088421888>, <4 x i32> <i32 1090519040, i32 1091567616, i32 1092616192, i32 1093664768>, <4 x i32> <i32 1094713344, i32 1095761920, i32 1096810496, i32 1097859072>, i32 16) nounwind
  %2 = call arm_aapcs_vfpcc  %0** @func2() nounwind
  %3 = load %0** %2, align 4, !tbaa !0
  store float 0.000000e+00, float* undef, align 4
  %4 = call arm_aapcs_vfpcc  %2* @func3(%2* undef, %2* undef, i32 2956) nounwind
  call arm_aapcs_vfpcc  void @func1(%0* %3, float* undef, float* undef, %2* undef)
  %5 = call arm_aapcs_vfpcc  %0** @func2() nounwind
  store float 1.000000e+00, float* undef, align 4
  call arm_aapcs_vfpcc  void @func1(%0* undef, float* undef, float* undef, %2* undef)
  store float 1.500000e+01, float* undef, align 4
  %6 = call arm_aapcs_vfpcc  %2** @func4() nounwind
  %7 = call arm_aapcs_vfpcc  %2* @func3(%2* undef, %2* undef, i32 2971) nounwind
  %8 = fadd float undef, -1.000000e+05
  store float %8, float* undef, align 16, !tbaa !3
  %9 = call arm_aapcs_vfpcc  i32 @rand() nounwind
  %10 = fmul float undef, 2.000000e+05
  %11 = fadd float %10, -1.000000e+05
  store float %11, float* undef, align 4, !tbaa !3
  call void @llvm.arm.neon.vst4.v4i32(i8* undef, <4 x i32> <i32 0, i32 1065353216, i32 1073741824, i32 1077936128>, <4 x i32> <i32 1082130432, i32 1084227584, i32 1086324736, i32 1088421888>, <4 x i32> <i32 1090519040, i32 1091567616, i32 1092616192, i32 1093664768>, <4 x i32> <i32 1094713344, i32 1095761920, i32 1096810496, i32 1097859072>, i32 16) nounwind
  ret void
}

declare void @llvm.arm.neon.vst4.v4i32(i8*, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, i32) nounwind

declare arm_aapcs_vfpcc i32 @rand()

!0 = metadata !{metadata !"any pointer", metadata !1}
!1 = metadata !{metadata !"omnipotent char", metadata !2}
!2 = metadata !{metadata !"Simple C/C++ TBAA", null}
!3 = metadata !{metadata !"float", metadata !1}
