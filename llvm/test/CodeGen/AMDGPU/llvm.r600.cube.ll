; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck %s

; CHECK-LABEL: {{^}}cube:
; CHECK: CUBE T{{[0-9]}}.X
; CHECK: CUBE T{{[0-9]}}.Y
; CHECK: CUBE T{{[0-9]}}.Z
; CHECK: CUBE * T{{[0-9]}}.W
define amdgpu_ps void @cube() {
main_body:
  %tmp = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 9)
  %tmp1 = extractelement <4 x float> %tmp, i32 3
  %tmp2 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 9)
  %tmp3 = extractelement <4 x float> %tmp2, i32 0
  %tmp4 = fdiv float %tmp3, %tmp1
  %tmp5 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 9)
  %tmp6 = extractelement <4 x float> %tmp5, i32 1
  %tmp7 = fdiv float %tmp6, %tmp1
  %tmp8 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 9)
  %tmp9 = extractelement <4 x float> %tmp8, i32 2
  %tmp10 = fdiv float %tmp9, %tmp1
  %tmp11 = insertelement <4 x float> undef, float %tmp4, i32 0
  %tmp12 = insertelement <4 x float> %tmp11, float %tmp7, i32 1
  %tmp13 = insertelement <4 x float> %tmp12, float %tmp10, i32 2
  %tmp14 = insertelement <4 x float> %tmp13, float 1.000000e+00, i32 3
  %tmp15 = call <4 x float> @llvm.r600.cube(<4 x float> %tmp14)
  %tmp16 = extractelement <4 x float> %tmp15, i32 0
  %tmp17 = extractelement <4 x float> %tmp15, i32 1
  %tmp18 = extractelement <4 x float> %tmp15, i32 2
  %tmp19 = extractelement <4 x float> %tmp15, i32 3
  %tmp20 = call float @llvm.fabs.f32(float %tmp18)
  %tmp21 = fdiv float 1.000000e+00, %tmp20
  %tmp22 = fmul float %tmp16, %tmp21
  %tmp23 = fadd float %tmp22, 1.500000e+00
  %tmp24 = fmul float %tmp17, %tmp21
  %tmp25 = fadd float %tmp24, 1.500000e+00
  %tmp26 = insertelement <4 x float> undef, float %tmp25, i32 0
  %tmp27 = insertelement <4 x float> %tmp26, float %tmp23, i32 1
  %tmp28 = insertelement <4 x float> %tmp27, float %tmp19, i32 2
  %tmp29 = insertelement <4 x float> %tmp28, float %tmp25, i32 3
  %tmp30 = shufflevector <4 x float> %tmp29, <4 x float> %tmp29, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %tmp31 = call <4 x float> @llvm.r600.tex(<4 x float> %tmp30, i32 0, i32 0, i32 0, i32 16, i32 0, i32 1, i32 1, i32 1, i32 1)
  call void @llvm.r600.store.swizzle(<4 x float> %tmp31, i32 0, i32 0)
  ret void
}

; Function Attrs: readnone
declare <4 x float> @llvm.r600.cube(<4 x float>) #0

; Function Attrs: nounwind readnone
declare float @llvm.fabs.f32(float) #0

declare void @llvm.r600.store.swizzle(<4 x float>, i32, i32)

; Function Attrs: readnone
declare <4 x float> @llvm.r600.tex(<4 x float>, i32, i32, i32, i32, i32, i32, i32, i32, i32) #0

attributes #0 = { nounwind readnone }
