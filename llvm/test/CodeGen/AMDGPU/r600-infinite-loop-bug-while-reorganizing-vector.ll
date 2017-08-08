; RUN: llc -march=r600 -mcpu=cayman < %s

define amdgpu_ps void @main(<4 x float> inreg %arg, <4 x float> inreg %arg1) {
main_body:
  %tmp = extractelement <4 x float> %arg, i32 0
  %tmp2 = extractelement <4 x float> %arg, i32 1
  %tmp3 = extractelement <4 x float> %arg, i32 2
  %tmp4 = extractelement <4 x float> %arg, i32 3
  %tmp5 = insertelement <4 x float> undef, float %tmp, i32 0
  %tmp6 = insertelement <4 x float> %tmp5, float %tmp2, i32 1
  %tmp7 = insertelement <4 x float> %tmp6, float %tmp3, i32 2
  %tmp8 = insertelement <4 x float> %tmp7, float %tmp4, i32 3
  %tmp9 = call <4 x float> @llvm.r600.cube(<4 x float> %tmp8)
  %tmp10 = extractelement <4 x float> %tmp9, i32 0
  %tmp11 = extractelement <4 x float> %tmp9, i32 1
  %tmp12 = extractelement <4 x float> %tmp9, i32 2
  %tmp13 = extractelement <4 x float> %tmp9, i32 3
  %tmp14 = call float @fabs(float %tmp12)
  %tmp15 = fdiv float 1.000000e+00, %tmp14
  %tmp16 = fmul float %tmp10, %tmp15
  %tmp17 = fadd float %tmp16, 1.500000e+00
  %tmp18 = fmul float %tmp11, %tmp15
  %tmp19 = fadd float %tmp18, 1.500000e+00
  %tmp20 = insertelement <4 x float> undef, float %tmp19, i32 0
  %tmp21 = insertelement <4 x float> %tmp20, float %tmp17, i32 1
  %tmp22 = insertelement <4 x float> %tmp21, float %tmp13, i32 2
  %tmp23 = insertelement <4 x float> %tmp22, float %tmp4, i32 3
  %tmp24 = extractelement <4 x float> %tmp23, i32 0
  %tmp25 = extractelement <4 x float> %tmp23, i32 1
  %tmp26 = extractelement <4 x float> %tmp23, i32 2
  %tmp27 = extractelement <4 x float> %tmp23, i32 3
  %tmp28 = insertelement <4 x float> undef, float %tmp24, i32 0
  %tmp29 = insertelement <4 x float> %tmp28, float %tmp25, i32 1
  %tmp30 = insertelement <4 x float> %tmp29, float %tmp26, i32 2
  %tmp31 = insertelement <4 x float> %tmp30, float %tmp27, i32 3
  %tmp32 = shufflevector <4 x float> %tmp31, <4 x float> %tmp31, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %tmp33 = call <4 x float> @llvm.r600.texc(<4 x float> %tmp32, i32 0, i32 0, i32 0, i32 16, i32 0, i32 1, i32 1, i32 1, i32 1)
  %tmp34 = extractelement <4 x float> %tmp33, i32 0
  %tmp35 = insertelement <4 x float> undef, float %tmp34, i32 0
  %tmp36 = insertelement <4 x float> %tmp35, float %tmp34, i32 1
  %tmp37 = insertelement <4 x float> %tmp36, float %tmp34, i32 2
  %tmp38 = insertelement <4 x float> %tmp37, float 1.000000e+00, i32 3
  call void @llvm.r600.store.swizzle(<4 x float> %tmp38, i32 0, i32 0)
  ret void
}

; Function Attrs: readnone
declare <4 x float> @llvm.r600.cube(<4 x float>) #0

; Function Attrs: readnone
declare float @fabs(float) #0

declare void @llvm.r600.store.swizzle(<4 x float>, i32, i32)

; Function Attrs: readnone
declare <4 x float> @llvm.r600.texc(<4 x float>, i32, i32, i32, i32, i32, i32, i32, i32, i32) #0

attributes #0 = { nounwind readnone }
