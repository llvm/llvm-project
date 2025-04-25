; RUN: llc < %s -mtriple=r600 -show-mc-encoding  -mcpu=rv710 | FileCheck %s

; CHECK: TEX 9 @6 ;  encoding: [0x06,0x00,0x00,0x00,0x00,0x04,0x88,0x80]
define amdgpu_vs void @test(<4 x float> inreg %reg0, <4 x float> inreg %reg1) {
bb:
  %tmp = extractelement <4 x float> %reg1, i32 0
  %tmp1 = extractelement <4 x float> %reg1, i32 1
  %tmp2 = extractelement <4 x float> %reg1, i32 2
  %tmp3 = extractelement <4 x float> %reg1, i32 3
  %tmp4 = insertelement <4 x float> undef, float %tmp, i32 0
  %tmp5 = insertelement <4 x float> %tmp4, float %tmp1, i32 1
  %tmp6 = insertelement <4 x float> %tmp5, float %tmp2, i32 2
  %tmp7 = insertelement <4 x float> %tmp6, float %tmp3, i32 3
  %tmp8 = shufflevector <4 x float> %tmp7, <4 x float> %tmp7, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %tmp9 = call <4 x float> @llvm.r600.tex(<4 x float> %tmp8, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1)
  %tmp10 = shufflevector <4 x float> %tmp7, <4 x float> %tmp7, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %tmp11 = call <4 x float> @llvm.r600.tex(<4 x float> %tmp10, i32 0, i32 0, i32 0, i32 1, i32 0, i32 1, i32 1, i32 1, i32 1)
  %tmp12 = shufflevector <4 x float> %tmp7, <4 x float> %tmp7, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %tmp13 = call <4 x float> @llvm.r600.tex(<4 x float> %tmp12, i32 0, i32 0, i32 0, i32 2, i32 0, i32 1, i32 1, i32 1, i32 1)
  %tmp14 = shufflevector <4 x float> %tmp7, <4 x float> %tmp7, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %tmp15 = call <4 x float> @llvm.r600.tex(<4 x float> %tmp14, i32 0, i32 0, i32 0, i32 3, i32 0, i32 1, i32 1, i32 1, i32 1)
  %tmp16 = shufflevector <4 x float> %tmp7, <4 x float> %tmp7, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %tmp17 = call <4 x float> @llvm.r600.tex(<4 x float> %tmp16, i32 0, i32 0, i32 0, i32 4, i32 0, i32 1, i32 1, i32 1, i32 1)
  %tmp18 = shufflevector <4 x float> %tmp7, <4 x float> %tmp7, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %tmp19 = call <4 x float> @llvm.r600.tex(<4 x float> %tmp18, i32 0, i32 0, i32 0, i32 5, i32 0, i32 1, i32 1, i32 1, i32 1)
  %tmp20 = shufflevector <4 x float> %tmp7, <4 x float> %tmp7, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %tmp21 = call <4 x float> @llvm.r600.tex(<4 x float> %tmp20, i32 0, i32 0, i32 0, i32 6, i32 0, i32 1, i32 1, i32 1, i32 1)
  %tmp22 = shufflevector <4 x float> %tmp7, <4 x float> %tmp7, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %tmp23 = call <4 x float> @llvm.r600.tex(<4 x float> %tmp22, i32 0, i32 0, i32 0, i32 7, i32 0, i32 1, i32 1, i32 1, i32 1)
  %tmp24 = shufflevector <4 x float> %tmp7, <4 x float> %tmp7, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %tmp25 = call <4 x float> @llvm.r600.tex(<4 x float> %tmp24, i32 0, i32 0, i32 0, i32 8, i32 0, i32 1, i32 1, i32 1, i32 1)
  %tmp26 = shufflevector <4 x float> %tmp7, <4 x float> %tmp7, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %tmp27 = call <4 x float> @llvm.r600.tex(<4 x float> %tmp26, i32 0, i32 0, i32 0, i32 9, i32 0, i32 1, i32 1, i32 1, i32 1)
  %tmp28 = fadd <4 x float> %tmp9, %tmp11
  %tmp29 = fadd <4 x float> %tmp28, %tmp13
  %tmp30 = fadd <4 x float> %tmp29, %tmp15
  %tmp31 = fadd <4 x float> %tmp30, %tmp17
  %tmp32 = fadd <4 x float> %tmp31, %tmp19
  %tmp33 = fadd <4 x float> %tmp32, %tmp21
  %tmp34 = fadd <4 x float> %tmp33, %tmp23
  %tmp35 = fadd <4 x float> %tmp34, %tmp25
  %tmp36 = fadd <4 x float> %tmp35, %tmp27
  call void @llvm.r600.store.swizzle(<4 x float> %tmp36, i32 0, i32 2)
  ret void
}

declare void @llvm.r600.store.swizzle(<4 x float>, i32, i32)

; Function Attrs: nounwind readnone
declare <4 x float> @llvm.r600.tex(<4 x float>, i32, i32, i32, i32, i32, i32, i32, i32, i32) #0

attributes #0 = { nounwind readnone }
