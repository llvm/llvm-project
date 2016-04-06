; RUN: llc < %s -march=r600 -show-mc-encoding  -mcpu=rv710 | FileCheck %s

; CHECK: TEX 9 @6 ;  encoding: [0x06,0x00,0x00,0x00,0x00,0x04,0x88,0x80]

define amdgpu_vs void @test(<4 x float> inreg %reg0, <4 x float> inreg %reg1) {
   %1 = extractelement <4 x float> %reg1, i32 0
   %2 = extractelement <4 x float> %reg1, i32 1
   %3 = extractelement <4 x float> %reg1, i32 2
   %4 = extractelement <4 x float> %reg1, i32 3
   %5 = insertelement <4 x float> undef, float %1, i32 0
   %6 = insertelement <4 x float> %5, float %2, i32 1
   %7 = insertelement <4 x float> %6, float %3, i32 2
   %8 = insertelement <4 x float> %7, float %4, i32 3
   %9 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %8, i32 0, i32 0, i32 1)
   %10 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %8, i32 1, i32 0, i32 1)
   %11 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %8, i32 2, i32 0, i32 1)
   %12 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %8, i32 3, i32 0, i32 1)
   %13 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %8, i32 4, i32 0, i32 1)
   %14 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %8, i32 5, i32 0, i32 1)
   %15 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %8, i32 6, i32 0, i32 1)
   %16 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %8, i32 7, i32 0, i32 1)
   %17 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %8, i32 8, i32 0, i32 1)
   %18 = call <4 x float> @llvm.AMDGPU.tex(<4 x float> %8, i32 9, i32 0, i32 1)
   %19 = fadd <4 x float> %9, %10
   %20 = fadd <4 x float> %19, %11
   %21 = fadd <4 x float> %20, %12
   %22 = fadd <4 x float> %21, %13
   %23 = fadd <4 x float> %22, %14
   %24 = fadd <4 x float> %23, %15
   %25 = fadd <4 x float> %24, %16
   %26 = fadd <4 x float> %25, %17
   %27 = fadd <4 x float> %26, %18
   call void @llvm.R600.store.swizzle(<4 x float> %27, i32 0, i32 2)
   ret void
}

declare <4 x float> @llvm.AMDGPU.tex(<4 x float>, i32, i32, i32) readnone

declare void @llvm.R600.store.swizzle(<4 x float>, i32, i32)
