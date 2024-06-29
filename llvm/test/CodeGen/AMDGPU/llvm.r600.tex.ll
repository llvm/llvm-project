;RUN: llc < %s -mtriple=r600 -mcpu=redwood | FileCheck %s

;CHECK: TEX_SAMPLE T{{[0-9]+\.XYZW, T[0-9]+\.XYZW}} RID:0 SID:0 CT:NNNN
;CHECK: TEX_SAMPLE T{{[0-9]+\.XYZW, T[0-9]+\.XYZW}} RID:0 SID:0 CT:NNNN
;CHECK: TEX_SAMPLE T{{[0-9]+\.XYZW, T[0-9]+\.XYZW}} RID:0 SID:0 CT:NNNN
;CHECK: TEX_SAMPLE T{{[0-9]+\.XYZW, T[0-9]+\.XYZW}} RID:0 SID:0 CT:NNNN
;CHECK: TEX_SAMPLE T{{[0-9]+\.XYZW, T[0-9]+\.XYZW}} RID:0 SID:0 CT:UUNN
;CHECK: TEX_SAMPLE_C T{{[0-9]+\.XYZW, T[0-9]+\.XYZZ}} RID:0 SID:0 CT:NNNN
;CHECK: TEX_SAMPLE_C T{{[0-9]+\.XYZW, T[0-9]+\.XYZZ}} RID:0 SID:0 CT:NNNN
;CHECK: TEX_SAMPLE_C T{{[0-9]+\.XYZW, T[0-9]+\.XYZZ}} RID:0 SID:0 CT:UUNN
;CHECK: TEX_SAMPLE T{{[0-9]+\.XYZW, T[0-9]+\.XYYW}} RID:0 SID:0 CT:NNUN
;CHECK: TEX_SAMPLE T{{[0-9]+\.XYZW, T[0-9]+\.XYZW}} RID:0 SID:0 CT:NNUN
;CHECK: TEX_SAMPLE_C T{{[0-9]+\.XYZW, T[0-9]+\.XYYZ}} RID:0 SID:0 CT:NNUN
;CHECK: TEX_SAMPLE_C T{{[0-9]+\.XYZW, T[0-9]+\.XYZW}} RID:0 SID:0 CT:NNUN
;CHECK: TEX_SAMPLE_C T{{[0-9]+\.XYZW, T[0-9]+\.XYZW}} RID:0 SID:0 CT:NNNN
;CHECK: TEX_SAMPLE T{{[0-9]+\.XYZW, T[0-9]+\.XYZW}} RID:0 SID:0 CT:NNNN
;CHECK: TEX_SAMPLE T{{[0-9]+\.XYZW, T[0-9]+\.XYZW}} RID:0 SID:0 CT:NNNN
;CHECK: TEX_SAMPLE T{{[0-9]+\.XYZW, T[0-9]+\.XYZW}} RID:0 SID:0 CT:NNUN

define amdgpu_kernel void @test(ptr addrspace(1) %out, ptr addrspace(1) %in) {
bb:
  %addr = load <4 x float>, ptr addrspace(1) %in
  %tmp = shufflevector <4 x float> %addr, <4 x float> %addr, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %tmp1 = call <4 x float> @llvm.r600.tex(<4 x float> %tmp, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1)
  %tmp2 = shufflevector <4 x float> %tmp1, <4 x float> %tmp1, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %tmp3 = call <4 x float> @llvm.r600.tex(<4 x float> %tmp2, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1)
  %tmp4 = shufflevector <4 x float> %tmp3, <4 x float> %tmp3, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %tmp5 = call <4 x float> @llvm.r600.tex(<4 x float> %tmp4, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1)
  %tmp6 = shufflevector <4 x float> %tmp5, <4 x float> %tmp5, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %tmp7 = call <4 x float> @llvm.r600.tex(<4 x float> %tmp6, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1)
  %tmp8 = shufflevector <4 x float> %tmp7, <4 x float> %tmp7, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %tmp9 = call <4 x float> @llvm.r600.tex(<4 x float> %tmp8, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1)
  %tmp10 = shufflevector <4 x float> %tmp9, <4 x float> %tmp9, <4 x i32> <i32 0, i32 1, i32 2, i32 2>
  %tmp11 = call <4 x float> @llvm.r600.texc(<4 x float> %tmp10, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1)
  %tmp12 = shufflevector <4 x float> %tmp11, <4 x float> %tmp11, <4 x i32> <i32 0, i32 1, i32 2, i32 2>
  %tmp13 = call <4 x float> @llvm.r600.texc(<4 x float> %tmp12, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1)
  %tmp14 = shufflevector <4 x float> %tmp13, <4 x float> %tmp13, <4 x i32> <i32 0, i32 1, i32 2, i32 2>
  %tmp15 = call <4 x float> @llvm.r600.texc(<4 x float> %tmp14, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1)
  %tmp16 = shufflevector <4 x float> %tmp15, <4 x float> %tmp15, <4 x i32> <i32 0, i32 1, i32 1, i32 3>
  %tmp17 = call <4 x float> @llvm.r600.tex(<4 x float> %tmp16, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 0, i32 1)
  %tmp18 = shufflevector <4 x float> %tmp17, <4 x float> %tmp17, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %tmp19 = call <4 x float> @llvm.r600.tex(<4 x float> %tmp18, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 0, i32 1)
  %tmp20 = shufflevector <4 x float> %tmp19, <4 x float> %tmp19, <4 x i32> <i32 0, i32 1, i32 1, i32 2>
  %tmp21 = call <4 x float> @llvm.r600.texc(<4 x float> %tmp20, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 0, i32 1)
  %tmp22 = shufflevector <4 x float> %tmp21, <4 x float> %tmp21, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %tmp23 = call <4 x float> @llvm.r600.texc(<4 x float> %tmp22, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 0, i32 1)
  %tmp24 = shufflevector <4 x float> %tmp23, <4 x float> %tmp23, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %tmp25 = call <4 x float> @llvm.r600.texc(<4 x float> %tmp24, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1)
  %tmp26 = shufflevector <4 x float> %tmp25, <4 x float> %tmp25, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %tmp27 = call <4 x float> @llvm.r600.tex(<4 x float> %tmp26, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1)
  %tmp28 = shufflevector <4 x float> %tmp27, <4 x float> %tmp27, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %tmp29 = call <4 x float> @llvm.r600.tex(<4 x float> %tmp28, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1)
  %tmp30 = shufflevector <4 x float> %tmp29, <4 x float> %tmp29, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %tmp31 = call <4 x float> @llvm.r600.tex(<4 x float> %tmp30, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 0, i32 1)
  store <4 x float> %tmp31, ptr addrspace(1) %out
  ret void
}

; Function Attrs: readnone
declare <4 x float> @llvm.r600.tex(<4 x float>, i32, i32, i32, i32, i32, i32, i32, i32, i32) #0

; Function Attrs: readnone
declare <4 x float> @llvm.r600.texc(<4 x float>, i32, i32, i32, i32, i32, i32, i32, i32, i32) #0

attributes #0 = { nounwind readnone }
