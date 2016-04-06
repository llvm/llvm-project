;RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s

;CHECK-NOT: MOV

define amdgpu_vs void @test(<4 x float> inreg %reg0) {
  %1 = extractelement <4 x float> %reg0, i32 0
  %2 = extractelement <4 x float> %reg0, i32 1
  %3 = extractelement <4 x float> %reg0, i32 2
  %4 = extractelement <4 x float> %reg0, i32 3
  %5 = fmul float %1, 3.0
  %6 = fmul float %2, 3.0
  %7 = fmul float %3, 3.0
  %8 = fmul float %4, 3.0
  %9 = insertelement <4 x float> undef, float %5, i32 0
  %10 = insertelement <4 x float> %9, float %6, i32 1
  %11 = insertelement <4 x float> undef, float %7, i32 0
  %12 = insertelement <4 x float> %11, float %5, i32 1
  %13 = insertelement <4 x float> undef, float %8, i32 0
  %14 = call <4 x float> @llvm.r600.tex(<4 x float> %10, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %15 = call <4 x float> @llvm.r600.tex(<4 x float> %12, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %16 = call <4 x float> @llvm.r600.tex(<4 x float> %13, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %17 = fadd <4 x float> %14, %15
  %18 = fadd <4 x float> %17, %16
  call void @llvm.R600.store.swizzle(<4 x float> %18, i32 0, i32 0)
  ret void
}

declare <4 x float> @llvm.r600.tex(<4 x float>, i32, i32, i32, i32, i32, i32, i32, i32, i32) readnone
declare void @llvm.R600.store.swizzle(<4 x float>, i32, i32)
