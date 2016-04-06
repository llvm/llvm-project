;RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s

; CHECK-LABEL: {{^}}main:
; CHECK: ADD *

define amdgpu_vs void @main(<4 x float> inreg %reg0, <4 x float> inreg %reg1, <4 x float> inreg %reg2) {
main_body:
  %0 = extractelement <4 x float> %reg1, i32 0
  %1 = extractelement <4 x float> %reg1, i32 1
  %2 = extractelement <4 x float> %reg1, i32 2
  %3 = extractelement <4 x float> %reg1, i32 3
  %4 = extractelement <4 x float> %reg2, i32 0
  %5 = fadd float %0, 2.0
  %6 = fadd float %1, 3.0
  %7 = fadd float %2, 4.0
  %8 = fadd float %3, 5.0
  %9 = bitcast float %4 to i32
  %10 = mul i32 %9, 6
  %11 = bitcast i32 %10 to float
  %12 = insertelement <4 x float> undef, float %5, i32 0
  %13 = insertelement <4 x float> %12, float %6, i32 1
  %14 = insertelement <4 x float> %13, float %7, i32 2
  %15 = insertelement <4 x float> %14, float %8, i32 3
  %16 = insertelement <4 x float> %15, float %11, i32 3

  %17 = call float @llvm.AMDGPU.dp4(<4 x float> %15,<4 x float> %16)
  %18 = insertelement <4 x float> undef, float %17, i32 0
  call void @llvm.R600.store.swizzle(<4 x float> %18, i32 0, i32 2)
  ret void
}

; CHECK-LABEL: {{^}}main2:
; CHECK-NOT: ADD *

define amdgpu_vs void @main2(<4 x float> inreg %reg0, <4 x float> inreg %reg1, <4 x float> inreg %reg2) {
main_body:
  %0 = extractelement <4 x float> %reg1, i32 0
  %1 = extractelement <4 x float> %reg1, i32 1
  %2 = extractelement <4 x float> %reg1, i32 2
  %3 = extractelement <4 x float> %reg1, i32 3
  %4 = extractelement <4 x float> %reg2, i32 0
  %5 = fadd float %0, 2.0
  %6 = fadd float %1, 3.0
  %7 = fadd float %2, 4.0
  %8 = fadd float %3, 2.0
  %9 = bitcast float %4 to i32
  %10 = mul i32 %9, 6
  %11 = bitcast i32 %10 to float
  %12 = insertelement <4 x float> undef, float %5, i32 0
  %13 = insertelement <4 x float> %12, float %6, i32 1
  %14 = insertelement <4 x float> %13, float %7, i32 2
  %15 = insertelement <4 x float> %14, float %8, i32 3
  %16 = insertelement <4 x float> %15, float %11, i32 3

  %17 = call float @llvm.AMDGPU.dp4(<4 x float> %15,<4 x float> %16)
  %18 = insertelement <4 x float> undef, float %17, i32 0
  call void @llvm.R600.store.swizzle(<4 x float> %18, i32 0, i32 2)
  ret void
}

; Function Attrs: readnone
declare float @llvm.AMDGPU.dp4(<4 x float>, <4 x float>) #1

declare void @llvm.R600.store.swizzle(<4 x float>, i32, i32)

attributes #1 = { readnone }
