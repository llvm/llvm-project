; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s

; CHECK-LABEL: {{^}}cube:
; CHECK: CUBE T{{[0-9]}}.X
; CHECK: CUBE T{{[0-9]}}.Y
; CHECK: CUBE T{{[0-9]}}.Z
; CHECK: CUBE * T{{[0-9]}}.W
define void @cube() #0 {
main_body:
  %0 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 9)
  %1 = extractelement <4 x float> %0, i32 3
  %2 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 9)
  %3 = extractelement <4 x float> %2, i32 0
  %4 = fdiv float %3, %1
  %5 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 9)
  %6 = extractelement <4 x float> %5, i32 1
  %7 = fdiv float %6, %1
  %8 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 9)
  %9 = extractelement <4 x float> %8, i32 2
  %10 = fdiv float %9, %1
  %11 = insertelement <4 x float> undef, float %4, i32 0
  %12 = insertelement <4 x float> %11, float %7, i32 1
  %13 = insertelement <4 x float> %12, float %10, i32 2
  %14 = insertelement <4 x float> %13, float 1.000000e+00, i32 3
  %15 = call <4 x float> @llvm.AMDGPU.cube(<4 x float> %14)
  %16 = extractelement <4 x float> %15, i32 0
  %17 = extractelement <4 x float> %15, i32 1
  %18 = extractelement <4 x float> %15, i32 2
  %19 = extractelement <4 x float> %15, i32 3
  %20 = call float @fabs(float %18)
  %21 = fdiv float 1.000000e+00, %20
  %22 = fmul float %16, %21
  %23 = fadd float %22, 1.500000e+00
  %24 = fmul float %17, %21
  %25 = fadd float %24, 1.500000e+00
  %26 = insertelement <4 x float> undef, float %25, i32 0
  %27 = insertelement <4 x float> %26, float %23, i32 1
  %28 = insertelement <4 x float> %27, float %19, i32 2
  %29 = insertelement <4 x float> %28, float %25, i32 3
  %30 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %29, i32 16, i32 0, i32 4)
  call void @llvm.R600.store.swizzle(<4 x float> %30, i32 0, i32 0)
  ret void
}

; Function Attrs: readnone
declare <4 x float> @llvm.AMDGPU.cube(<4 x float>) #1

; Function Attrs: readnone
declare float @fabs(float) #1

; Function Attrs: readnone
declare <4 x float> @llvm.AMDGPU.tex(<4 x float>, i32, i32, i32) #1

declare void @llvm.R600.store.swizzle(<4 x float>, i32, i32)

attributes #0 = { "ShaderType"="0" }
attributes #1 = { readnone }

