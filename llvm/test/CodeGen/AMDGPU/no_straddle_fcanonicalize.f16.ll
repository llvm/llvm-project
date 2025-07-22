;RUN: llc --amdgpu-prevent-half-cache-line-straddling -mtriple=amdgcn -mcpu=gfx1100 -mattr=+real-true16,dumpcode --filetype=obj < %s | llvm-objdump --triple=amdgcn --mcpu=gfx1100 -d  - > %t.dis
;RUN: %python %p/has_cache_straddle.py %t.dis

declare half @llvm.fabs.f16(half) #0
declare half @llvm.canonicalize.f16(half) #0
declare <2 x half> @llvm.fabs.v2f16(<2 x half>) #0
declare <2 x half> @llvm.canonicalize.v2f16(<2 x half>) #0
declare <3 x half> @llvm.canonicalize.v3f16(<3 x half>) #0
declare <4 x half> @llvm.canonicalize.v4f16(<4 x half>) #0
declare <6 x half> @llvm.canonicalize.v6f16(<6 x half>) #0
declare <8 x half> @llvm.canonicalize.v8f16(<8 x half>) #0
declare <12 x half> @llvm.canonicalize.v12f16(<12 x half>) #0
declare <16 x half> @llvm.canonicalize.v16f16(<16 x half>) #0
declare <32 x half> @llvm.canonicalize.v32f16(<32 x half>) #0
declare <64 x half> @llvm.canonicalize.v64f16(<64 x half>) #0
declare i32 @llvm.amdgcn.workitem.id.x() #0

define amdgpu_kernel void @test_fold_canonicalize_undef_value_f16(ptr addrspace(1) %out) #1 {
  %canonicalized = call half @llvm.canonicalize.f16(half undef)
  store half %canonicalized, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @v_test_canonicalize_var_f16(ptr addrspace(1) %out) #1 {
  %val = load half, ptr addrspace(1) %out
  %canonicalized = call half @llvm.canonicalize.f16(half %val)
  store half %canonicalized, ptr addrspace(1) poison
  ret void
}

define amdgpu_kernel void @s_test_canonicalize_var_f16(ptr addrspace(1) %out, i16 zeroext %val.arg) #1 {
  %val = bitcast i16 %val.arg to half
  %canonicalized = call half @llvm.canonicalize.f16(half %val)
  store half %canonicalized, ptr addrspace(1) %out
  ret void
}

define <2 x half> @v_test_canonicalize_build_vector_v2f16(half %lo, half %hi) #1 {
  %ins0 = insertelement <2 x half> poison, half %lo, i32 0
  %ins1 = insertelement <2 x half> %ins0, half %hi, i32 1
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> %ins1)
  ret <2 x half> %canonicalized
}

define amdgpu_kernel void @v_test_canonicalize_fabs_var_f16(ptr addrspace(1) %out) #1 {
  %val = load half, ptr addrspace(1) %out
  %val.fabs = call half @llvm.fabs.f16(half %val)
  %canonicalized = call half @llvm.canonicalize.f16(half %val.fabs)
  store half %canonicalized, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @v_test_canonicalize_fneg_fabs_var_f16(ptr addrspace(1) %out) #1 {
  %val = load half, ptr addrspace(1) %out
  %val.fabs = call half @llvm.fabs.f16(half %val)
  %val.fabs.fneg = fneg half %val.fabs
  %canonicalized = call half @llvm.canonicalize.f16(half %val.fabs.fneg)
  store half %canonicalized, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @v_test_canonicalize_fneg_var_f16(ptr addrspace(1) %out) #1 {
  %val = load half, ptr addrspace(1) %out
  %val.fneg = fneg half %val
  %canonicalized = call half @llvm.canonicalize.f16(half %val.fneg)
  store half %canonicalized, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @v_test_no_denormals_canonicalize_fneg_var_f16(ptr addrspace(1) %out) #2 {
  %val = load half, ptr addrspace(1) %out
  %val.fneg = fneg half %val
  %canonicalized = call half @llvm.canonicalize.f16(half %val.fneg)
  store half %canonicalized, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @v_test_no_denormals_canonicalize_fneg_fabs_var_f16(ptr addrspace(1) %out) #2 {
  %val = load half, ptr addrspace(1) %out
  %val.fabs = call half @llvm.fabs.f16(half %val)
  %val.fabs.fneg = fneg half %val.fabs
  %canonicalized = call half @llvm.canonicalize.f16(half %val.fabs.fneg)
  store half %canonicalized, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @test_fold_canonicalize_p0_f16(ptr addrspace(1) %out) #1 {
  %canonicalized = call half @llvm.canonicalize.f16(half 0.0)
  store half %canonicalized, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @test_fold_canonicalize_n0_f16(ptr addrspace(1) %out) #1 {
  %canonicalized = call half @llvm.canonicalize.f16(half -0.0)
  store half %canonicalized, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @test_fold_canonicalize_p1_f16(ptr addrspace(1) %out) #1 {
  %canonicalized = call half @llvm.canonicalize.f16(half 1.0)
  store half %canonicalized, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @test_fold_canonicalize_n1_f16(ptr addrspace(1) %out) #1 {
  %canonicalized = call half @llvm.canonicalize.f16(half -1.0)
  store half %canonicalized, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @test_fold_canonicalize_literal_f16(ptr addrspace(1) %out) #1 {
  %canonicalized = call half @llvm.canonicalize.f16(half 16.0)
  store half %canonicalized, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @test_default_denormals_fold_canonicalize_denormal0_f16(ptr addrspace(1) %out) #1 {
  %canonicalized = call half @llvm.canonicalize.f16(half 0xH03FF)
  store half %canonicalized, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @test_denormals_fold_canonicalize_denormal0_f16(ptr addrspace(1) %out) #3 {
  %canonicalized = call half @llvm.canonicalize.f16(half 0xH03FF)
  store half %canonicalized, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @test_default_denormals_fold_canonicalize_denormal1_f16(ptr addrspace(1) %out) #1 {
  %canonicalized = call half @llvm.canonicalize.f16(half 0xH83FF)
  store half %canonicalized, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @test_denormals_fold_canonicalize_denormal1_f16(ptr addrspace(1) %out) #3 {
  %canonicalized = call half @llvm.canonicalize.f16(half 0xH83FF)
  store half %canonicalized, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @test_fold_canonicalize_qnan_f16(ptr addrspace(1) %out) #1 {
  %canonicalized = call half @llvm.canonicalize.f16(half 0xH7C00)
  store half %canonicalized, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @test_fold_canonicalize_qnan_value_neg1_f16(ptr addrspace(1) %out) #1 {
  %canonicalized = call half @llvm.canonicalize.f16(half bitcast (i16 -1 to half))
  store half %canonicalized, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @test_fold_canonicalize_qnan_value_neg2_f16(ptr addrspace(1) %out) #1 {
  %canonicalized = call half @llvm.canonicalize.f16(half bitcast (i16 -2 to half))
  store half %canonicalized, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @test_fold_canonicalize_snan0_value_f16(ptr addrspace(1) %out) #1 {
  %canonicalized = call half @llvm.canonicalize.f16(half 0xH7C01)
  store half %canonicalized, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @test_fold_canonicalize_snan1_value_f16(ptr addrspace(1) %out) #1 {
  %canonicalized = call half @llvm.canonicalize.f16(half 0xH7DFF)
  store half %canonicalized, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @test_fold_canonicalize_snan2_value_f16(ptr addrspace(1) %out) #1 {
  %canonicalized = call half @llvm.canonicalize.f16(half 0xHFDFF)
  store half %canonicalized, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @test_fold_canonicalize_snan3_value_f16(ptr addrspace(1) %out) #1 {
  %canonicalized = call half @llvm.canonicalize.f16(half 0xHFC01)
  store half %canonicalized, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @v_test_canonicalize_var_v2f16(ptr addrspace(1) %out) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr <2 x half>, ptr addrspace(1) %out, i32 %tid
  %val = load <2 x half>, ptr addrspace(1) %gep
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> %val)
  store <2 x half> %canonicalized, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @v_test_canonicalize_fabs_var_v2f16(ptr addrspace(1) %out) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr <2 x half>, ptr addrspace(1) %out, i32 %tid
  %val = load <2 x half>, ptr addrspace(1) %gep
  %val.fabs = call <2 x half> @llvm.fabs.v2f16(<2 x half> %val)
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> %val.fabs)
  store <2 x half> %canonicalized, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @v_test_canonicalize_fneg_fabs_var_v2f16(ptr addrspace(1) %out) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr <2 x half>, ptr addrspace(1) %out, i32 %tid
  %val = load <2 x half>, ptr addrspace(1) %gep
  %val.fabs = call <2 x half> @llvm.fabs.v2f16(<2 x half> %val)
  %val.fabs.fneg = fneg <2 x half> %val.fabs
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> %val.fabs.fneg)
  store <2 x half> %canonicalized, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @v_test_canonicalize_fneg_var_v2f16(ptr addrspace(1) %out) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr <2 x half>, ptr addrspace(1) %out, i32 %tid
  %val = load <2 x half>, ptr addrspace(1) %gep
  %fneg.val = fneg <2 x half> %val
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> %fneg.val)
  store <2 x half> %canonicalized, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @s_test_canonicalize_var_v2f16(ptr addrspace(1) %out, i32 zeroext %val.arg) #1 {
  %val = bitcast i32 %val.arg to <2 x half>
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> %val)
  store <2 x half> %canonicalized, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @test_fold_canonicalize_p0_v2f16(ptr addrspace(1) %out) #1 {
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> zeroinitializer)
  store <2 x half> %canonicalized, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @test_fold_canonicalize_n0_v2f16(ptr addrspace(1) %out) #1 {
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> <half -0.0, half -0.0>)
  store <2 x half> %canonicalized, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @test_fold_canonicalize_p1_v2f16(ptr addrspace(1) %out) #1 {
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> <half 1.0, half 1.0>)
  store <2 x half> %canonicalized, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @test_fold_canonicalize_n1_v2f16(ptr addrspace(1) %out) #1 {
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> <half -1.0, half -1.0>)
  store <2 x half> %canonicalized, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @test_fold_canonicalize_literal_v2f16(ptr addrspace(1) %out) #1 {
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> <half 16.0, half 16.0>)
  store <2 x half> %canonicalized, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @test_no_denormals_fold_canonicalize_denormal0_v2f16(ptr addrspace(1) %out) #1 {
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> <half 0xH03FF, half 0xH03FF>)
  store <2 x half> %canonicalized, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @test_denormals_fold_canonicalize_denormal0_v2f16(ptr addrspace(1) %out) #3 {
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> <half 0xH03FF, half 0xH03FF>)
  store <2 x half> %canonicalized, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @test_no_denormals_fold_canonicalize_denormal1_v2f16(ptr addrspace(1) %out) #1 {
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> <half 0xH83FF, half 0xH83FF>)
  store <2 x half> %canonicalized, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @test_denormals_fold_canonicalize_denormal1_v2f16(ptr addrspace(1) %out) #3 {
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> <half 0xH83FF, half 0xH83FF>)
  store <2 x half> %canonicalized, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @test_fold_canonicalize_qnan_v2f16(ptr addrspace(1) %out) #1 {
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> <half 0xH7C00, half 0xH7C00>)
  store <2 x half> %canonicalized, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @test_fold_canonicalize_qnan_value_neg1_v2f16(ptr addrspace(1) %out) #1 {
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> bitcast (i32 -1 to <2 x half>))
  store <2 x half> %canonicalized, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @test_fold_canonicalize_qnan_value_neg2_v2f16(ptr addrspace(1) %out) #1 {
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> <half bitcast (i16 -2 to half), half bitcast (i16 -2 to half)>)
  store <2 x half> %canonicalized, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @test_fold_canonicalize_snan0_value_v2f16(ptr addrspace(1) %out) #1 {
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> <half 0xH7C01, half 0xH7C01>)
  store <2 x half> %canonicalized, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @test_fold_canonicalize_snan1_value_v2f16(ptr addrspace(1) %out) #1 {
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> <half 0xH7DFF, half 0xH7DFF>)
  store <2 x half> %canonicalized, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @test_fold_canonicalize_snan2_value_v2f16(ptr addrspace(1) %out) #1 {
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> <half 0xHFDFF, half 0xHFDFF>)
  store <2 x half> %canonicalized, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @test_fold_canonicalize_snan3_value_v2f16(ptr addrspace(1) %out) #1 {
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> <half 0xHFC01, half 0xHFC01>)
  store <2 x half> %canonicalized, ptr addrspace(1) %out
  ret void
}

define <3 x half> @v_test_canonicalize_var_v3f16(<3 x half> %val) #1 {
  %canonicalized = call <3 x half> @llvm.canonicalize.v3f16(<3 x half> %val)
  ret <3 x half> %canonicalized
}

define <4 x half> @v_test_canonicalize_var_v4f16(<4 x half> %val) #1 {
  %canonicalized = call <4 x half> @llvm.canonicalize.v4f16(<4 x half> %val)
  ret <4 x half> %canonicalized
}

define amdgpu_kernel void @s_test_canonicalize_undef_v2f16(ptr addrspace(1) %out) #1 {
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> undef)
  store <2 x half> %canonicalized, ptr addrspace(1) %out
  ret void
}

define <2 x half> @v_test_canonicalize_reg_undef_v2f16(half %val) #1 {
  %vec = insertelement <2 x half> poison, half %val, i32 0
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> %vec)
  ret <2 x half> %canonicalized
}

define <2 x half> @v_test_canonicalize_undef_reg_v2f16(half %val) #1 {
  %vec = insertelement <2 x half> poison, half %val, i32 1
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> %vec)
  ret <2 x half> %canonicalized
}

define <2 x half> @v_test_canonicalize_undef_lo_imm_hi_v2f16() #1 {
  %vec = insertelement <2 x half> undef, half 1.0, i32 1
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> %vec)
  ret <2 x half> %canonicalized
}

define <2 x half> @v_test_canonicalize_imm_lo_undef_hi_v2f16() #1 {
  %vec = insertelement <2 x half> undef, half 1.0, i32 0
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> %vec)
  ret <2 x half> %canonicalized
}

define <2 x half> @v_test_canonicalize_undef_lo_k_hi_v2f16() #1 {
  %vec = insertelement <2 x half> undef, half 16.0, i32 1
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> %vec)
  ret <2 x half> %canonicalized
}

define <2 x half> @v_test_canonicalize_k_lo_undef_hi_v2f16() #1 {
  %vec = insertelement <2 x half> undef, half 16.0, i32 0
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> %vec)
  ret <2 x half> %canonicalized
}

define <2 x half> @v_test_canonicalize_reg_k_v2f16(half %val) #1 {
  %vec0 = insertelement <2 x half> poison, half %val, i32 0
  %vec1 = insertelement <2 x half> %vec0, half 2.0, i32 1
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> %vec1)
  ret <2 x half> %canonicalized
}

define <2 x half> @v_test_canonicalize_k_reg_v2f16(half %val) #1 {
  %vec0 = insertelement <2 x half> poison, half 2.0, i32 0
  %vec1 = insertelement <2 x half> %vec0, half %val, i32 1
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> %vec1)
  ret <2 x half> %canonicalized
}

define amdgpu_kernel void @s_test_canonicalize_undef_v4f16(ptr addrspace(1) %out) #1 {
  %canonicalized = call <4 x half> @llvm.canonicalize.v4f16(<4 x half> undef)
  store <4 x half> %canonicalized, ptr addrspace(1) %out
  ret void
}

define <4 x half> @v_test_canonicalize_reg_undef_undef_undef_v4f16(half %val) #1 {
  %vec = insertelement <4 x half> poison, half %val, i32 0
  %canonicalized = call <4 x half> @llvm.canonicalize.v4f16(<4 x half> %vec)
  ret <4 x half> %canonicalized
}

define <4 x half> @v_test_canonicalize_reg_reg_undef_undef_v4f16(half %val0, half %val1) #1 {
  %vec0 = insertelement <4 x half> poison, half %val0, i32 0
  %vec1 = insertelement <4 x half> %vec0, half %val1, i32 1
  %canonicalized = call <4 x half> @llvm.canonicalize.v4f16(<4 x half> %vec1)
  ret <4 x half> %canonicalized
}

define <4 x half> @v_test_canonicalize_reg_undef_reg_reg_v4f16(half %val0, half %val1, half %val2) #1 {
  %vec0 = insertelement <4 x half> poison, half %val0, i32 0
  %vec1 = insertelement <4 x half> %vec0, half %val1, i32 2
  %vec2 = insertelement <4 x half> %vec1, half %val2, i32 3
  %canonicalized = call <4 x half> @llvm.canonicalize.v4f16(<4 x half> %vec2)
  ret <4 x half> %canonicalized
}

define <6 x half> @v_test_canonicalize_var_v6f16(<6 x half> %val) #1 {
  %canonicalized = call <6 x half> @llvm.canonicalize.v6f16(<6 x half> %val)
  ret <6 x half> %canonicalized
}

define <8 x half> @v_test_canonicalize_var_v8f16(<8 x half> %val) #1 {
  %canonicalized = call <8 x half> @llvm.canonicalize.v8f16(<8 x half> %val)
  ret <8 x half> %canonicalized
}

define <12 x half> @v_test_canonicalize_var_v12f16(<12 x half> %val) #1 {
  %canonicalized = call <12 x half> @llvm.canonicalize.v12f16(<12 x half> %val)
  ret <12 x half> %canonicalized
}

define <16 x half> @v_test_canonicalize_var_v16f16(<16 x half> %val) #1 {
  %canonicalized = call <16 x half> @llvm.canonicalize.v16f16(<16 x half> %val)
  ret <16 x half> %canonicalized
}

define <32 x half> @v_test_canonicalize_var_v32f16(<32 x half> %val) #1 {
  %canonicalized = call <32 x half> @llvm.canonicalize.v32f16(<32 x half> %val)
  ret <32 x half> %canonicalized
}

define <64 x half> @v_test_canonicalize_var_v64f16(<64 x half> %val) #1 {
  %canonicalized = call <64 x half> @llvm.canonicalize.v64f16(<64 x half> %val)
  ret <64 x half> %canonicalized
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind "denormal-fp-math-f32"="preserve-sign,preserve-sign" }
attributes #2 = { nounwind "denormal-fp-math"="preserve-sign,preserve-sign" }
attributes #3 = { nounwind "denormal-fp-math-f32"="preserve-sign,preserve-sign" }
