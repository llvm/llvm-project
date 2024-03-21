; RUN: llc -mtriple=r600 -mcpu=redwood < %s | FileCheck %s

; CHECK: {{^}}main1:
; CHECK: MOV * T{{[0-9]+\.[XYZW], KC0}}
define amdgpu_kernel void @main1() #0 {
main_body:
  %tmp = load <4 x float>, ptr addrspace(8) null
  %tmp7 = extractelement <4 x float> %tmp, i32 0
  %tmp8 = load <4 x float>, ptr addrspace(8) getelementptr ([1024 x <4 x float>], ptr addrspace(8) null, i64 0, i32 1)
  %tmp9 = extractelement <4 x float> %tmp8, i32 0
  %tmp10 = load <4 x float>, ptr addrspace(8) getelementptr ([1024 x <4 x float>], ptr addrspace(8) null, i64 0, i32 2)
  %tmp11 = extractelement <4 x float> %tmp10, i32 0
  %tmp12 = fcmp ogt float %tmp7, 0.000000e+00
  %tmp13 = select i1 %tmp12, float %tmp9, float %tmp11
  %tmp14 = load <4 x float>, ptr addrspace(8) null
  %tmp15 = extractelement <4 x float> %tmp14, i32 1
  %tmp16 = load <4 x float>, ptr addrspace(8) getelementptr ([1024 x <4 x float>], ptr addrspace(8) null, i64 0, i32 1)
  %tmp17 = extractelement <4 x float> %tmp16, i32 1
  %tmp18 = load <4 x float>, ptr addrspace(8) getelementptr ([1024 x <4 x float>], ptr addrspace(8) null, i64 0, i32 2)
  %tmp19 = extractelement <4 x float> %tmp18, i32 1
  %tmp20 = fcmp ogt float %tmp15, 0.000000e+00
  %tmp21 = select i1 %tmp20, float %tmp17, float %tmp19
  %tmp22 = load <4 x float>, ptr addrspace(8) null
  %tmp23 = extractelement <4 x float> %tmp22, i32 2
  %tmp24 = load <4 x float>, ptr addrspace(8) getelementptr ([1024 x <4 x float>], ptr addrspace(8) null, i64 0, i32 1)
  %tmp25 = extractelement <4 x float> %tmp24, i32 2
  %tmp26 = load <4 x float>, ptr addrspace(8) getelementptr ([1024 x <4 x float>], ptr addrspace(8) null, i64 0, i32 2)
  %tmp27 = extractelement <4 x float> %tmp26, i32 2
  %tmp28 = fcmp ogt float %tmp23, 0.000000e+00
  %tmp29 = select i1 %tmp28, float %tmp25, float %tmp27
  %tmp30 = load <4 x float>, ptr addrspace(8) null
  %tmp31 = extractelement <4 x float> %tmp30, i32 3
  %tmp32 = load <4 x float>, ptr addrspace(8) getelementptr ([1024 x <4 x float>], ptr addrspace(8) null, i64 0, i32 1)
  %tmp33 = extractelement <4 x float> %tmp32, i32 3
  %tmp34 = load <4 x float>, ptr addrspace(8) getelementptr ([1024 x <4 x float>], ptr addrspace(8) null, i64 0, i32 2)
  %tmp35 = extractelement <4 x float> %tmp34, i32 3
  %tmp36 = fcmp ogt float %tmp31, 0.000000e+00
  %tmp37 = select i1 %tmp36, float %tmp33, float %tmp35
  %max.0.i = call float @llvm.maxnum.f32(float %tmp13, float 0.000000e+00)
  %clamp.i = call float @llvm.minnum.f32(float %max.0.i, float 1.000000e+00)
  %max.0.i5 = call float @llvm.maxnum.f32(float %tmp21, float 0.000000e+00)
  %clamp.i6 = call float @llvm.minnum.f32(float %max.0.i5, float 1.000000e+00)
  %max.0.i3 = call float @llvm.maxnum.f32(float %tmp29, float 0.000000e+00)
  %clamp.i4 = call float @llvm.minnum.f32(float %max.0.i3, float 1.000000e+00)
  %max.0.i1 = call float @llvm.maxnum.f32(float %tmp37, float 0.000000e+00)
  %clamp.i2 = call float @llvm.minnum.f32(float %max.0.i1, float 1.000000e+00)
  %tmp38 = insertelement <4 x float> undef, float %clamp.i, i32 0
  %tmp39 = insertelement <4 x float> %tmp38, float %clamp.i6, i32 1
  %tmp40 = insertelement <4 x float> %tmp39, float %clamp.i4, i32 2
  %tmp41 = insertelement <4 x float> %tmp40, float %clamp.i2, i32 3
  call void @llvm.r600.store.swizzle(<4 x float> %tmp41, i32 0, i32 0)
  ret void
}

; CHECK: {{^}}main2:
; CHECK-NOT: MOV
define amdgpu_kernel void @main2() #0 {
main_body:
  %tmp = load <4 x float>, ptr addrspace(8) null
  %tmp7 = extractelement <4 x float> %tmp, i32 0
  %tmp8 = load <4 x float>, ptr addrspace(8) getelementptr ([1024 x <4 x float>], ptr addrspace(8) null, i64 0, i32 1)
  %tmp9 = extractelement <4 x float> %tmp8, i32 0
  %tmp10 = load <4 x float>, ptr addrspace(8) getelementptr ([1024 x <4 x float>], ptr addrspace(8) null, i64 0, i32 1)
  %tmp11 = extractelement <4 x float> %tmp10, i32 1
  %tmp12 = fcmp ogt float %tmp7, 0.000000e+00
  %tmp13 = select i1 %tmp12, float %tmp9, float %tmp11
  %tmp14 = load <4 x float>, ptr addrspace(8) null
  %tmp15 = extractelement <4 x float> %tmp14, i32 1
  %tmp16 = load <4 x float>, ptr addrspace(8) getelementptr ([1024 x <4 x float>], ptr addrspace(8) null, i64 0, i32 2)
  %tmp17 = extractelement <4 x float> %tmp16, i32 0
  %tmp18 = load <4 x float>, ptr addrspace(8) getelementptr ([1024 x <4 x float>], ptr addrspace(8) null, i64 0, i32 2)
  %tmp19 = extractelement <4 x float> %tmp18, i32 1
  %tmp20 = fcmp ogt float %tmp15, 0.000000e+00
  %tmp21 = select i1 %tmp20, float %tmp17, float %tmp19
  %tmp22 = load <4 x float>, ptr addrspace(8) null
  %tmp23 = extractelement <4 x float> %tmp22, i32 2
  %tmp24 = load <4 x float>, ptr addrspace(8) getelementptr ([1024 x <4 x float>], ptr addrspace(8) null, i64 0, i32 1)
  %tmp25 = extractelement <4 x float> %tmp24, i32 3
  %tmp26 = load <4 x float>, ptr addrspace(8) getelementptr ([1024 x <4 x float>], ptr addrspace(8) null, i64 0, i32 1)
  %tmp27 = extractelement <4 x float> %tmp26, i32 2
  %tmp28 = fcmp ogt float %tmp23, 0.000000e+00
  %tmp29 = select i1 %tmp28, float %tmp25, float %tmp27
  %tmp30 = load <4 x float>, ptr addrspace(8) null
  %tmp31 = extractelement <4 x float> %tmp30, i32 3
  %tmp32 = load <4 x float>, ptr addrspace(8) getelementptr ([1024 x <4 x float>], ptr addrspace(8) null, i64 0, i32 2)
  %tmp33 = extractelement <4 x float> %tmp32, i32 3
  %tmp34 = load <4 x float>, ptr addrspace(8) getelementptr ([1024 x <4 x float>], ptr addrspace(8) null, i64 0, i32 2)
  %tmp35 = extractelement <4 x float> %tmp34, i32 2
  %tmp36 = fcmp ogt float %tmp31, 0.000000e+00
  %tmp37 = select i1 %tmp36, float %tmp33, float %tmp35
  %max.0.i = call float @llvm.maxnum.f32(float %tmp13, float 0.000000e+00)
  %clamp.i = call float @llvm.minnum.f32(float %max.0.i, float 1.000000e+00)
  %max.0.i5 = call float @llvm.maxnum.f32(float %tmp21, float 0.000000e+00)
  %clamp.i6 = call float @llvm.minnum.f32(float %max.0.i5, float 1.000000e+00)
  %max.0.i3 = call float @llvm.maxnum.f32(float %tmp29, float 0.000000e+00)
  %clamp.i4 = call float @llvm.minnum.f32(float %max.0.i3, float 1.000000e+00)
  %max.0.i1 = call float @llvm.maxnum.f32(float %tmp37, float 0.000000e+00)
  %clamp.i2 = call float @llvm.minnum.f32(float %max.0.i1, float 1.000000e+00)
  %tmp38 = insertelement <4 x float> undef, float %clamp.i, i32 0
  %tmp39 = insertelement <4 x float> %tmp38, float %clamp.i6, i32 1
  %tmp40 = insertelement <4 x float> %tmp39, float %clamp.i4, i32 2
  %tmp41 = insertelement <4 x float> %tmp40, float %clamp.i2, i32 3
  call void @llvm.r600.store.swizzle(<4 x float> %tmp41, i32 0, i32 0)
  ret void
}

declare void @llvm.r600.store.swizzle(<4 x float>, i32, i32) #0
declare float @llvm.minnum.f32(float, float) #1
declare float @llvm.maxnum.f32(float, float) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
