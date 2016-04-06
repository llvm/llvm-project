;RUN: llc < %s -march=r600 -mcpu=cayman

define amdgpu_ps void @main(<4 x float> inreg, <4 x float> inreg) {
main_body:
  %2 = extractelement <4 x float> %0, i32 0
  %3 = extractelement <4 x float> %0, i32 1
  %4 = extractelement <4 x float> %0, i32 2
  %5 = extractelement <4 x float> %0, i32 3
  %6 = insertelement <4 x float> undef, float %2, i32 0
  %7 = insertelement <4 x float> %6, float %3, i32 1
  %8 = insertelement <4 x float> %7, float %4, i32 2
  %9 = insertelement <4 x float> %8, float %5, i32 3
  %10 = call <4 x float> @llvm.AMDGPU.cube(<4 x float> %9)
  %11 = extractelement <4 x float> %10, i32 0
  %12 = extractelement <4 x float> %10, i32 1
  %13 = extractelement <4 x float> %10, i32 2
  %14 = extractelement <4 x float> %10, i32 3
  %15 = call float @fabs(float %13)
  %16 = fdiv float 1.000000e+00, %15
  %17 = fmul float %11, %16
  %18 = fadd float %17, 1.500000e+00
  %19 = fmul float %12, %16
  %20 = fadd float %19, 1.500000e+00
  %21 = insertelement <4 x float> undef, float %20, i32 0
  %22 = insertelement <4 x float> %21, float %18, i32 1
  %23 = insertelement <4 x float> %22, float %14, i32 2
  %24 = insertelement <4 x float> %23, float %5, i32 3
  %25 = extractelement <4 x float> %24, i32 0
  %26 = extractelement <4 x float> %24, i32 1
  %27 = extractelement <4 x float> %24, i32 2
  %28 = extractelement <4 x float> %24, i32 3
  %29 = insertelement <4 x float> undef, float %25, i32 0
  %30 = insertelement <4 x float> %29, float %26, i32 1
  %31 = insertelement <4 x float> %30, float %27, i32 2
  %32 = insertelement <4 x float> %31, float %28, i32 3
  %33 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %32, i32 16, i32 0, i32 13)
  %34 = extractelement <4 x float> %33, i32 0
  %35 = insertelement <4 x float> undef, float %34, i32 0
  %36 = insertelement <4 x float> %35, float %34, i32 1
  %37 = insertelement <4 x float> %36, float %34, i32 2
  %38 = insertelement <4 x float> %37, float 1.000000e+00, i32 3
  call void @llvm.R600.store.swizzle(<4 x float> %38, i32 0, i32 0)
  ret void
}

; Function Attrs: readnone
declare <4 x float> @llvm.AMDGPU.cube(<4 x float>) #1

; Function Attrs: readnone
declare float @fabs(float) #1

; Function Attrs: readnone
declare <4 x float> @llvm.AMDGPU.tex(<4 x float>, i32, i32, i32) #1

declare void @llvm.R600.store.swizzle(<4 x float>, i32, i32)

attributes #1 = { readnone }
