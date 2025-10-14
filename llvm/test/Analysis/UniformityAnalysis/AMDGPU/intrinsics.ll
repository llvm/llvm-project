; RUN: opt -mtriple amdgcn-- -passes='print<uniformity>' -disable-output %s 2>&1 | FileCheck %s

; CHECK: DIVERGENT: %swizzle = call i32 @llvm.amdgcn.ds.swizzle(i32 %src, i32 100) #0
define amdgpu_kernel void @ds_swizzle(ptr addrspace(1) %out, i32 %src) #0 {
  %swizzle = call i32 @llvm.amdgcn.ds.swizzle(i32 %src, i32 100) #0
  store i32 %swizzle, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK: DIVERGENT: %v = call i32 @llvm.amdgcn.permlane16.i32(i32 %src0, i32 %src0, i32 %src1, i32 %src2, i1 false, i1 false) #0
define amdgpu_kernel void @v_permlane16_b32(ptr addrspace(1) %out, i32 %src0, i32 %src1, i32 %src2) #0 {
  %v = call i32 @llvm.amdgcn.permlane16.i32(i32 %src0, i32 %src0, i32 %src1, i32 %src2, i1 false, i1 false) #0
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; CHECK: DIVERGENT: %v = call i32 @llvm.amdgcn.permlanex16.i32(i32 %src0, i32 %src0, i32 %src1, i32 %src2, i1 false, i1 false) #0
define amdgpu_kernel void @v_permlanex16_b32(ptr addrspace(1) %out, i32 %src0, i32 %src1, i32 %src2) #0 {
  %v = call i32 @llvm.amdgcn.permlanex16.i32(i32 %src0, i32 %src0, i32 %src1, i32 %src2, i1 false, i1 false) #0
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; CHECK: DIVERGENT: %v = call i32 @llvm.amdgcn.permlane16.var(i32 %src0, i32 %src1, i32 %src2, i1 false, i1 false) #0
define amdgpu_kernel void @v_permlane16_var_b32(ptr addrspace(1) %out, i32 %src0, i32 %src1, i32 %src2) #0 {
  %v = call i32 @llvm.amdgcn.permlane16.var(i32 %src0, i32 %src1, i32 %src2, i1 false, i1 false) #0
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; CHECK: DIVERGENT: %v = call i32 @llvm.amdgcn.permlanex16.var(i32 %src0, i32 %src1, i32 %src2, i1 false, i1 false) #0
define amdgpu_kernel void @v_permlanex16_var_b32(ptr addrspace(1) %out, i32 %src0, i32 %src1, i32 %src2) #0 {
  %v = call i32 @llvm.amdgcn.permlanex16.var(i32 %src0, i32 %src1, i32 %src2, i1 false, i1 false) #0
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call i32 @llvm.amdgcn.update.dpp.i32(i32 %in1, i32 %in2, i32 1, i32 1, i32 1, i1 false) #0
define amdgpu_kernel void @update_dpp(ptr addrspace(1) %out, i32 %in1, i32 %in2) #0 {
  %tmp0 = call i32 @llvm.amdgcn.update.dpp.i32(i32 %in1, i32 %in2, i32 1, i32 1, i32 1, i1 false) #0
  store i32 %tmp0, ptr addrspace(1) %out
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %in, i32 1, i32 1, i32 1, i1 true) #0
define amdgpu_kernel void @mov_dpp(ptr addrspace(1) %out, i32 %in) #0 {
  %tmp0 = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %in, i32 1, i32 1, i32 1, i1 true) #0
  store i32 %tmp0, ptr addrspace(1) %out
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call i32 @llvm.amdgcn.mov.dpp8.i32(i32 %in, i32 1) #0
define amdgpu_kernel void @mov_dpp8(ptr addrspace(1) %out, i32 %in) #0 {
  %tmp0 = call i32 @llvm.amdgcn.mov.dpp8.i32(i32 %in, i32 1) #0
  store i32 %tmp0, ptr addrspace(1) %out
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call i32 @llvm.amdgcn.writelane.i32(i32 0, i32 1, i32 2)
define amdgpu_kernel void @writelane(ptr addrspace(1) %out) #0 {
  %tmp0 = call i32 @llvm.amdgcn.writelane.i32(i32 0, i32 1, i32 2)
  store i32 %tmp0, ptr addrspace(1) %out
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v16f16(<16 x half> %A, <16 x half> %B, <8 x float> %C)
define amdgpu_kernel void @wmma_f32_16x16x16_f16(<16 x half> %A, <16 x half> %B, <8 x float> %C, ptr addrspace(1) %out) {
  %tmp0 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v16f16(<16 x half> %A, <16 x half> %B, <8 x float> %C)
  store <8 x float> %tmp0, ptr addrspace(1) %out, align 32
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v16i16(<16 x i16> %A, <16 x i16> %B, <8 x float> %C)
define amdgpu_kernel void @wmma_f32_16x16x16_ibf16(<16 x i16> %A, <16 x i16> %B, <8 x float> %C, ptr addrspace(1) %out) {
  %tmp0 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v16i16(<16 x i16> %A, <16 x i16> %B, <8 x float> %C)
  store <8 x float> %tmp0, ptr addrspace(1) %out, align 32
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call <16 x half> @llvm.amdgcn.wmma.f16.16x16x16.f16.v16f16.v16f16(<16 x half> %A, <16 x half> %B, <16 x half> %C, i1 false)
define amdgpu_kernel void @wmma_f16_16x16x16_f16(<16 x half> %A, <16 x half> %B, <16 x half> %C, ptr addrspace(1) %out) {
bb:
  %tmp0 = call <16 x half> @llvm.amdgcn.wmma.f16.16x16x16.f16.v16f16.v16f16(<16 x half> %A, <16 x half> %B, <16 x half> %C, i1 false)
  store <16 x half> %tmp0, ptr addrspace(1) %out, align 32
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call <16 x i16> @llvm.amdgcn.wmma.bf16.16x16x16.bf16.v16i16.v16i16(<16 x i16> %A, <16 x i16> %B, <16 x i16> %C, i1 false)
define amdgpu_kernel void @wmma_f16_16x16x16_bf16(<16 x i16> %A, <16 x i16> %B, <16 x i16> %C, ptr addrspace(1) %out) {
bb:
  %tmp0 = call <16 x i16> @llvm.amdgcn.wmma.bf16.16x16x16.bf16.v16i16.v16i16(<16 x i16> %A, <16 x i16> %B, <16 x i16> %C, i1 false)
  store <16 x i16> %tmp0, ptr addrspace(1) %out, align 32
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call <8 x i32> @llvm.amdgcn.wmma.i32.16x16x16.iu8.v8i32.v4i32(i1 false, <4 x i32> %A, i1 false, <4 x i32> %B, <8 x i32> %C, i1 false)
define amdgpu_kernel void @wmma_i32_16x16x16_ui8(<4 x i32> %A, <4 x i32> %B, <8 x i32> %C, ptr addrspace(1) %out) {
bb:
  %tmp0 = call <8 x i32> @llvm.amdgcn.wmma.i32.16x16x16.iu8.v8i32.v4i32(i1 false, <4 x i32> %A, i1 false, <4 x i32> %B, <8 x i32> %C, i1 false)
  store <8 x i32> %tmp0, ptr addrspace(1) %out, align 32
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call <8 x i32> @llvm.amdgcn.wmma.i32.16x16x16.iu4.v8i32.v2i32(i1 false, <2 x i32> %A, i1 false, <2 x i32> %B, <8 x i32> %C, i1 false)
define amdgpu_kernel void @wmma_i32_16x16x16_ui4(<2 x i32> %A, <2 x i32> %B, <8 x i32> %C, ptr addrspace(1) %out) {
bb:
  %tmp0 = call <8 x i32> @llvm.amdgcn.wmma.i32.16x16x16.iu4.v8i32.v2i32(i1 false, <2 x i32> %A, i1 false, <2 x i32> %B, <8 x i32> %C, i1 false)
  store <8 x i32> %tmp0, ptr addrspace(1) %out, align 32
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call <8 x float> @llvm.amdgcn.swmmac.f32.16x16x32.f16.v8f32.v8f16.v16f16.i16(<8 x half> %A, <16 x half> %B, <8 x float> %C, i16 %Index)
define amdgpu_kernel void @swmmac_f32_16x16x32_f16(<8 x half> %A, <16 x half> %B, <8 x float> %C, i16 %Index, ptr addrspace(1) %out) {
bb:
  %tmp0 = call <8 x float> @llvm.amdgcn.swmmac.f32.16x16x32.f16.v8f32.v8f16.v16f16.i16(<8 x half> %A, <16 x half> %B, <8 x float> %C, i16 %Index)
    store <8 x float> %tmp0, ptr addrspace(1) %out, align 32
    ret void
}

; CHECK: DIVERGENT: %tmp0 = call <8 x float> @llvm.amdgcn.swmmac.f32.16x16x32.bf16.v8f32.v8i16.v16i16.i16(<8 x i16> %A, <16 x i16> %B, <8 x float> %C, i16 %Index)
define amdgpu_kernel void @swmmac_f32_16x16x32_bf16(<8 x i16> %A, <16 x i16> %B, <8 x float> %C, i16 %Index, ptr addrspace(1) %out) {
bb:
  %tmp0 = call <8 x float> @llvm.amdgcn.swmmac.f32.16x16x32.bf16.v8f32.v8i16.v16i16.i16(<8 x i16> %A, <16 x i16> %B, <8 x float> %C, i16 %Index)
    store <8 x float> %tmp0, ptr addrspace(1) %out, align 32
    ret void
}

; CHECK: DIVERGENT: %tmp0 = call <8 x half> @llvm.amdgcn.swmmac.f16.16x16x32.f16.v8f16.v8f16.v16f16.i16(<8 x half> %A, <16 x half> %B, <8 x half> %C, i16 %Index)
define amdgpu_kernel void @swmmac_f16_16x16x32_f16(<8 x half> %A, <16 x half> %B, <8 x half> %C, i16 %Index, ptr addrspace(1) %out) {
bb:
  %tmp0 = call <8 x half> @llvm.amdgcn.swmmac.f16.16x16x32.f16.v8f16.v8f16.v16f16.i16(<8 x half> %A, <16 x half> %B, <8 x half> %C, i16 %Index)
    store <8 x half> %tmp0, ptr addrspace(1) %out, align 32
    ret void
}

; CHECK: DIVERGENT: %tmp0 = call <8 x i16> @llvm.amdgcn.swmmac.bf16.16x16x32.bf16.v8i16.v8i16.v16i16.i16(<8 x i16> %A, <16 x i16> %B, <8 x i16> %C, i16 %Index)
define amdgpu_kernel void @swmmac_bf16_16x16x32_bf16(<8 x i16> %A, <16 x i16> %B, <8 x i16> %C, i16 %Index, ptr addrspace(1) %out) {
bb:
  %tmp0 = call <8 x i16> @llvm.amdgcn.swmmac.bf16.16x16x32.bf16.v8i16.v8i16.v16i16.i16(<8 x i16> %A, <16 x i16> %B, <8 x i16> %C, i16 %Index)
  store <8 x i16> %tmp0, ptr addrspace(1) %out, align 32
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call <8 x i32> @llvm.amdgcn.swmmac.i32.16x16x32.iu8.v8i32.v2i32.v4i32.i16(i1 false, <2 x i32> %A, i1 false, <4 x i32> %B, <8 x i32> %C, i16 %Index, i1 false)
define amdgpu_kernel void @swmmac_i32_16x16x32_iu8(<2 x i32> %A, <4 x i32> %B, <8 x i32> %C, i16 %Index, ptr addrspace(1) %out) {
bb:
  %tmp0 = call <8 x i32> @llvm.amdgcn.swmmac.i32.16x16x32.iu8.v8i32.v2i32.v4i32.i16(i1 false, <2 x i32> %A, i1 false, <4 x i32> %B, <8 x i32> %C, i16 %Index, i1 false)
  store <8 x i32> %tmp0, ptr addrspace(1) %out, align 32
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call <8 x i32> @llvm.amdgcn.swmmac.i32.16x16x32.iu4.v8i32.i32.v2i32.i16(i1 false, i32 %A, i1 false, <2 x i32> %B, <8 x i32> %C, i16 %Index, i1 false)
define amdgpu_kernel void @swmmac_i32_16x16x32_iu4(i32 %A, <2 x i32> %B, <8 x i32> %C, i16 %Index, ptr addrspace(1) %out) {
bb:
  %tmp0 = call <8 x i32> @llvm.amdgcn.swmmac.i32.16x16x32.iu4.v8i32.i32.v2i32.i16(i1 false, i32 %A, i1 false, <2 x i32> %B, <8 x i32> %C, i16 %Index, i1 false)
  store <8 x i32> %tmp0, ptr addrspace(1) %out, align 32
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call <8 x i32> @llvm.amdgcn.swmmac.i32.16x16x64.iu4.v8i32.v2i32.v4i32.i16(i1 false, <2 x i32> %A, i1 false, <4 x i32> %B, <8 x i32> %C, i16 %Index, i1 false)
define amdgpu_kernel void @swmmac_i32_16x16x64_iu4(<2 x i32> %A, <4 x i32> %B, <8 x i32> %C, i16 %Index, ptr addrspace(1) %out) {
bb:
  %tmp0 = call <8 x i32> @llvm.amdgcn.swmmac.i32.16x16x64.iu4.v8i32.v2i32.v4i32.i16(i1 false, <2 x i32> %A, i1 false, <4 x i32> %B, <8 x i32> %C, i16 %Index, i1 false)
  store <8 x i32> %tmp0, ptr addrspace(1) %out, align 32
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call <8 x float> @llvm.amdgcn.swmmac.f32.16x16x32.fp8.fp8.v8f32.v2i32.v4i32.i16(<2 x i32> %A, <4 x i32> %B, <8 x float> %C, i16 %Index)
define amdgpu_kernel void @swmmac_f32_16x16x32_fp8.fp8(<2 x i32> %A, <4 x i32> %B, <8 x float> %C, i16 %Index, ptr addrspace(1) %out) {
bb:
  %tmp0 = call <8 x float> @llvm.amdgcn.swmmac.f32.16x16x32.fp8.fp8.v8f32.v2i32.v4i32.i16(<2 x i32> %A, <4 x i32> %B, <8 x float> %C, i16 %Index)
  store <8 x float> %tmp0, ptr addrspace(1) %out, align 32
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call <8 x float> @llvm.amdgcn.swmmac.f32.16x16x32.fp8.bf8.v8f32.v2i32.v4i32.i16(<2 x i32> %A, <4 x i32> %B, <8 x float> %C, i16 %Index)
define amdgpu_kernel void @swmmac_f32_16x16x32_fp8.bf8(<2 x i32> %A, <4 x i32> %B, <8 x float> %C, i16 %Index, ptr addrspace(1) %out) {
bb:
  %tmp0 = call <8 x float> @llvm.amdgcn.swmmac.f32.16x16x32.fp8.bf8.v8f32.v2i32.v4i32.i16(<2 x i32> %A, <4 x i32> %B, <8 x float> %C, i16 %Index)
  store <8 x float> %tmp0, ptr addrspace(1) %out, align 32
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call <8 x float> @llvm.amdgcn.swmmac.f32.16x16x32.bf8.fp8.v8f32.v2i32.v4i32.i16(<2 x i32> %A, <4 x i32> %B, <8 x float> %C, i16 %Index)
define amdgpu_kernel void @swmmac_f32_16x16x32_bf8.fp8.v8f32.v2i32.v4i32.i16(<2 x i32> %A, <4 x i32> %B, <8 x float> %C, i16 %Index, ptr addrspace(1) %out) {
bb:
  %tmp0 = call <8 x float> @llvm.amdgcn.swmmac.f32.16x16x32.bf8.fp8(<2 x i32> %A, <4 x i32> %B, <8 x float> %C, i16 %Index)
  store <8 x float> %tmp0, ptr addrspace(1) %out, align 32
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call <8 x float> @llvm.amdgcn.swmmac.f32.16x16x32.bf8.bf8.v8f32.v2i32.v4i32.i16(<2 x i32> %A, <4 x i32> %B, <8 x float> %C, i16 %Index)
define amdgpu_kernel void @swmmac_f32_16x16x32_bf8.bf8.v8f32.v2i32.v4i32.i16(<2 x i32> %A, <4 x i32> %B, <8 x float> %C, i16 %Index, ptr addrspace(1) %out) {
bb:
  %tmp0 = call <8 x float> @llvm.amdgcn.swmmac.f32.16x16x32.bf8.bf8(<2 x i32> %A, <4 x i32> %B, <8 x float> %C, i16 %Index)
  store <8 x float> %tmp0, ptr addrspace(1) %out, align 32
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x4.f32.v8f32.v2f32(i1 false, <2 x float> %A, i1 false, <2 x float> %B, i16 0, <8 x float> %C, i1 false, i1 false)
define amdgpu_kernel void @wmma_f32_16x16x4_f32(<2 x float> %A, <2 x float> %B, <8 x float> %C, ptr addrspace(1) %out) {
  %tmp0 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x4.f32.v8f32.v2f32(i1 0, <2 x float> %A, i1 0, <2 x float> %B, i16 0, <8 x float> %C, i1 false, i1 false)
  store <8 x float> %tmp0, ptr addrspace(1) %out
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(i1 false, <16 x bfloat> %A, i1 false, <16 x bfloat> %B, i16 0, <8 x float> %C, i1 false, i1 false)
define amdgpu_kernel void @wmma_f32_16x16x32_bf16(<16 x bfloat> %A, <16 x bfloat> %B, <8 x float> %C, ptr addrspace(1) %out) {
  %tmp0 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(i1 0, <16 x bfloat> %A, i1 0, <16 x bfloat> %B, i16 0, <8 x float> %C, i1 false, i1 false)
  store <8 x float> %tmp0, ptr addrspace(1) %out
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.f16.v8f32.v16f16(i1 false, <16 x half> %A, i1 false, <16 x half> %B, i16 0, <8 x float> %C, i1 false, i1 false)
define amdgpu_kernel void @wmma_f32_16x16x32_f16(<16 x half> %A, <16 x half> %B, <8 x float> %C, ptr addrspace(1) %out) {
  %tmp0 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.f16.v8f32.v16f16(i1 0, <16 x half> %A, i1 0, <16 x half> %B, i16 0, <8 x float> %C, i1 false, i1 false)
  store <8 x float> %tmp0, ptr addrspace(1) %out
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call <8 x half> @llvm.amdgcn.wmma.f16.16x16x32.f16.v8f16.v16f16(i1 false, <16 x half> %A, i1 false, <16 x half> %B, i16 0, <8 x half> %C, i1 false, i1 false)
define amdgpu_kernel void @wmma_f16_16x16x32_f16(<16 x half> %A, <16 x half> %B, <8 x half> %C, ptr addrspace(1) %out) {
  %tmp0 = call <8 x half> @llvm.amdgcn.wmma.f16.16x16x32.f16.v8f16.v16f16(i1 0, <16 x half> %A, i1 0, <16 x half> %B, i16 0, <8 x half> %C, i1 false, i1 false)
  store <8 x half> %tmp0, ptr addrspace(1) %out
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call <8 x bfloat> @llvm.amdgcn.wmma.bf16.16x16x32.bf16.v8bf16.v16bf16(i1 false, <16 x bfloat> %A, i1 false, <16 x bfloat> %B, i16 0, <8 x bfloat> %C, i1 false, i1 false)
define amdgpu_kernel void @wmma_bf16_16x16x32_bf16(<16 x bfloat> %A, <16 x bfloat> %B, <8 x bfloat> %C, ptr addrspace(1) %out) {
  %tmp0 = call <8 x bfloat> @llvm.amdgcn.wmma.bf16.16x16x32.bf16.v8bf16.v16bf16(i1 0, <16 x bfloat> %A, i1 0, <16 x bfloat> %B, i16 0, <8 x bfloat> %C, i1 false, i1 false)
  store <8 x bfloat> %tmp0, ptr addrspace(1) %out
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call <8 x bfloat> @llvm.amdgcn.wmma.bf16f32.16x16x32.bf16.v8bf16.v16bf16.v8f32(i1 false, <16 x bfloat> %A, i1 false, <16 x bfloat> %B, i16 0, <8 x float> %C, i1 false, i1 false)
define amdgpu_kernel void @wmma_bf16f32_16x16x32_bf16(<16 x bfloat> %A, <16 x bfloat> %B, <8 x float> %C, ptr addrspace(1) %out) {
  %tmp0 = call <8 x bfloat> @llvm.amdgcn.wmma.bf16f32.16x16x32.bf16.v8bf16.v16bf16(i1 0, <16 x bfloat> %A, i1 0, <16 x bfloat> %B, i16 0, <8 x float> %C, i1 false, i1 false)
  store <8 x bfloat> %tmp0, ptr addrspace(1) %out
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x64.fp8.fp8.v8f32.v8i32(<8 x i32> %A, <8 x i32> %B, i16 0, <8 x float> %C, i1 false, i1 false)
define amdgpu_kernel void @wmma_f32_16x16x64_fp8_fp8(<8 x i32> %A, <8 x i32> %B, <8 x float> %C, ptr addrspace(1) %out) {
  %tmp0 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x64.fp8.fp8.v8f32.v8i32(<8 x i32> %A, <8 x i32> %B, i16 0, <8 x float> %C, i1 false, i1 false)
  store <8 x float> %tmp0, ptr addrspace(1) %out
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x64.fp8.bf8.v8f32.v8i32(<8 x i32> %A, <8 x i32> %B, i16 0, <8 x float> %C, i1 false, i1 false)
define amdgpu_kernel void @wmma_f32_16x16x64_fp8_bf8(<8 x i32> %A, <8 x i32> %B, <8 x float> %C, ptr addrspace(1) %out) {
  %tmp0 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x64.fp8.bf8.v8f32.v8i32(<8 x i32> %A, <8 x i32> %B, i16 0, <8 x float> %C, i1 false, i1 false)
  store <8 x float> %tmp0, ptr addrspace(1) %out
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x64.bf8.fp8.v8f32.v8i32(<8 x i32> %A, <8 x i32> %B, i16 0, <8 x float> %C, i1 false, i1 false)
define amdgpu_kernel void @wmma_f32_16x16x64_bf8_fp8(<8 x i32> %A, <8 x i32> %B, <8 x float> %C, ptr addrspace(1) %out) {
  %tmp0 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x64.bf8.fp8.v8f32.v8i32(<8 x i32> %A, <8 x i32> %B, i16 0, <8 x float> %C, i1 false, i1 false)
  store <8 x float> %tmp0, ptr addrspace(1) %out
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x64.bf8.bf8.v8f32.v8i32(<8 x i32> %A, <8 x i32> %B, i16 0, <8 x float> %C, i1 false, i1 false)
define amdgpu_kernel void @wmma_f32_16x16x64_bf8_bf8(<8 x i32> %A, <8 x i32> %B, <8 x float> %C, ptr addrspace(1) %out) {
  %tmp0 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x64.bf8.bf8.v8f32.v8i32(<8 x i32> %A, <8 x i32> %B, i16 0, <8 x float> %C, i1 false, i1 false)
  store <8 x float> %tmp0, ptr addrspace(1) %out
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call <8 x half> @llvm.amdgcn.wmma.f16.16x16x64.fp8.fp8.v8f16.v8i32(<8 x i32> %A, <8 x i32> %B, i16 0, <8 x half> %C, i1 false, i1 false)
define amdgpu_kernel void @wmma_f16_16x16x64_fp8_fp8(<8 x i32> %A, <8 x i32> %B, <8 x half> %C, ptr addrspace(1) %out) {
  %tmp0 = call <8 x half> @llvm.amdgcn.wmma.f16.16x16x64.fp8.fp8.v8f16.v8i32(<8 x i32> %A, <8 x i32> %B, i16 0, <8 x half> %C, i1 false, i1 false)
  store <8 x half> %tmp0, ptr addrspace(1) %out
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call <8 x half> @llvm.amdgcn.wmma.f16.16x16x64.fp8.bf8.v8f16.v8i32(<8 x i32> %A, <8 x i32> %B, i16 0, <8 x half> %C, i1 false, i1 false)
define amdgpu_kernel void @wmma_f16_16x16x64_fp8_bf8(<8 x i32> %A, <8 x i32> %B, <8 x half> %C, ptr addrspace(1) %out) {
  %tmp0 = call <8 x half> @llvm.amdgcn.wmma.f16.16x16x64.fp8.bf8.v8f16.v8i32(<8 x i32> %A, <8 x i32> %B, i16 0, <8 x half> %C, i1 false, i1 false)
  store <8 x half> %tmp0, ptr addrspace(1) %out
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call <8 x half> @llvm.amdgcn.wmma.f16.16x16x64.bf8.fp8.v8f16.v8i32(<8 x i32> %A, <8 x i32> %B, i16 0, <8 x half> %C, i1 false, i1 false)
define amdgpu_kernel void @wmma_f16_16x16x64_bf8_fp8(<8 x i32> %A, <8 x i32> %B, <8 x half> %C, ptr addrspace(1) %out) {
  %tmp0 = call <8 x half> @llvm.amdgcn.wmma.f16.16x16x64.bf8.fp8.v8f16.v8i32(<8 x i32> %A, <8 x i32> %B, i16 0, <8 x half> %C, i1 false, i1 false)
  store <8 x half> %tmp0, ptr addrspace(1) %out
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call <8 x half> @llvm.amdgcn.wmma.f16.16x16x64.bf8.bf8.v8f16.v8i32(<8 x i32> %A, <8 x i32> %B, i16 0, <8 x half> %C, i1 false, i1 false)
define amdgpu_kernel void @wmma_f16_16x16x64_bf8_bf8(<8 x i32> %A, <8 x i32> %B, <8 x half> %C, ptr addrspace(1) %out) {
  %tmp0 = call <8 x half> @llvm.amdgcn.wmma.f16.16x16x64.bf8.bf8.v8f16.v8i32(<8 x i32> %A, <8 x i32> %B, i16 0, <8 x half> %C, i1 false, i1 false)
  store <8 x half> %tmp0, ptr addrspace(1) %out
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call <8 x i32> @llvm.amdgcn.wmma.i32.16x16x64.iu8.v8i32.v8i32(i1 false, <8 x i32> %A, i1 false, <8 x i32> %B, <8 x i32> %C, i1 false, i1 false)
define amdgpu_kernel void @wmma_i32_16x16x64_iu8(<8 x i32> %A, <8 x i32> %B, <8 x i32> %C, ptr addrspace(1) %out) {
  %tmp0 = call <8 x i32> @llvm.amdgcn.wmma.i32.16x16x64.iu8.v8i32.v8i32(i1 0, <8 x i32> %A, i1 0, <8 x i32> %B, <8 x i32> %C, i1 false, i1 false)
  store <8 x i32> %tmp0, ptr addrspace(1) %out
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %A, i32 0, <16 x i32> %B, i16 0, <8 x float> %C)
define amdgpu_ps void @wmma_f32_16x16x128_f8f6f4(<16 x i32> %A, <16 x i32> %B, <8 x float> %C, ptr addrspace(1) %out) {
bb:
  %tmp0 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %A, i32 0, <16 x i32> %B, i16 0, <8 x float> %C)
  store <8 x float> %tmp0, ptr addrspace(1) %out
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %A, i32 0, <16 x i32> %B, i16 0, <8 x float> %C, i32 1, i32 0, i32 1, i32 1, i32 0, i32 1, i1 false, i1 false)
define amdgpu_ps void @wmma_scale_f32_16x16x128_f8f6f4(<16 x i32> %A, <16 x i32> %B, <8 x float> %C, ptr addrspace(1) %out) {
bb:
  %tmp0 = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %A, i32 0, <16 x i32> %B, i16 0, <8 x float> %C, i32 1, i32 0, i32 1, i32 1, i32 0, i32 1, i1 false, i1 false)
  store <8 x float> %tmp0, ptr addrspace(1) %out
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call <8 x float> @llvm.amdgcn.wmma.scale16.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %A, i32 0, <16 x i32> %B, i16 0, <8 x float> %C, i32 1, i32 0, i64 1, i32 1, i32 0, i64 1, i1 false, i1 false)
define amdgpu_ps void @wmma_scale16_f32_16x16x128_f8f6f4(<16 x i32> %A, <16 x i32> %B, <8 x float> %C, ptr addrspace(1) %out) {
bb:
  %tmp0 = call <8 x float> @llvm.amdgcn.wmma.scale16.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 0, <16 x i32> %A, i32 0, <16 x i32> %B, i16 0, <8 x float> %C, i32 1, i32 0, i64 1, i32 1, i32 0, i64 1, i1 false, i1 false)
  store <8 x float> %tmp0, ptr addrspace(1) %out
  ret void
}

; CHRCK: DIVERGENT:   %tmp0 = call <8 x float> @llvm.amdgcn.swmmac.f32.16x16x64.f16.v8f32.v16f16.v32f16.i16(i1 false, <16 x half> %A, i1 false, <32 x half> %B, <8 x float> %C, i16 %Index, i1 false, i1 false)
define amdgpu_ps void @swmmac_f32_16x16x64_f16(<16 x half> %A, <32 x half> %B, <8 x float> %C, i16 %Index, ptr addrspace(1) %out) {
  %tmp0 = call <8 x float> @llvm.amdgcn.swmmac.f32.16x16x64.f16.v8f32.v16f16.v32f16.i16(i1 0, <16 x half> %A, i1 0, <32 x half> %B, <8 x float> %C, i16 %Index, i1 false, i1 false)
  store <8 x float> %tmp0, ptr addrspace(1) %out
  ret void
}

; CHECK: DIVERGENT:   %tmp0 = call <8 x float> @llvm.amdgcn.swmmac.f32.16x16x64.bf16.v8f32.v16bf16.v32bf16.i16(i1 false, <16 x bfloat> %A, i1 false, <32 x bfloat> %B, <8 x float> %C, i16 %Index, i1 false, i1 false)
define amdgpu_ps void @swmmac_f32_16x16x64_bf16(<16 x bfloat> %A, <32 x bfloat> %B, <8 x float> %C, i16 %Index, ptr addrspace(1) %out) {
  %tmp0 = call <8 x float> @llvm.amdgcn.swmmac.f32.16x16x64.bf16.v8f32.v16bf16.v32bf16.i16(i1 0, <16 x bfloat> %A, i1 0, <32 x bfloat> %B, <8 x float> %C, i16 %Index, i1 false, i1 false)
  store <8 x float> %tmp0, ptr addrspace(1) %out
  ret void
}

; CHECK: DIVERGENT:   %tmp0 = call <8 x half> @llvm.amdgcn.swmmac.f16.16x16x64.f16.v8f16.v16f16.v32f16.i16(i1 false, <16 x half> %A, i1 false, <32 x half> %B, <8 x half> %C, i16 %Index, i1 false, i1 false)
define amdgpu_ps void @swmmac_f16_16x16x64_f16(<16 x half> %A, <32 x half> %B, <8 x half> %C, i16 %Index, ptr addrspace(1) %out) {
  %tmp0 = call <8 x half> @llvm.amdgcn.swmmac.f16.16x16x64.f16.v8f16.v16f16.v32f16.i16(i1 0, <16 x half> %A, i1 0, <32 x half> %B, <8 x half> %C, i16 %Index, i1 false, i1 false)
  store <8 x half> %tmp0, ptr addrspace(1) %out
  ret void
}

; CHECK: DIVERGENT:   %tmp0 = call <8 x bfloat> @llvm.amdgcn.swmmac.bf16.16x16x64.bf16.v8bf16.v16bf16.v32bf16.i16(i1 false, <16 x bfloat> %A, i1 false, <32 x bfloat> %B, <8 x bfloat> %C, i16 %Index, i1 false, i1 false)
define amdgpu_ps void @swmmac_bf16_16x16x64_bf16(<16 x bfloat> %A, <32 x bfloat> %B, <8 x bfloat> %C, i16 %Index, ptr addrspace(1) %out) {
  %tmp0 = call <8 x bfloat> @llvm.amdgcn.swmmac.bf16.16x16x64.bf16.v8bf16.v16bf16.v32bf16.i16(i1 0, <16 x bfloat> %A, i1 0, <32 x bfloat> %B, <8 x bfloat> %C, i16 %Index, i1 false, i1 false)
  store <8 x bfloat> %tmp0, ptr addrspace(1) %out
  ret void
}

; CHECK: DIVERGENT:   %tmp0 = call <8 x float> @llvm.amdgcn.swmmac.bf16f32.16x16x64.bf16.v8f32.v16bf16.v32bf16.i16(i1 false, <16 x bfloat> %A, i1 false, <32 x bfloat> %B, <8 x float> %C, i16 %Index, i1 false, i1 false)
define amdgpu_ps void @swmmac_bf16f32_16x16x64_bf16(<16 x bfloat> %A, <32 x bfloat> %B, <8 x float> %C, i16 %Index, ptr addrspace(1) %out) {
  %tmp0 = call <8 x float> @llvm.amdgcn.swmmac.bf16f32.16x16x64.bf16.v8f32.v16bf16.v32bf16.i16(i1 0, <16 x bfloat> %A, i1 0, <32 x bfloat> %B, <8 x float> %C, i16 %Index, i1 false, i1 false)
  store <8 x float> %tmp0, ptr addrspace(1) %out
  ret void
}

; CHECK: DIVERGENT:   %tmp0 = call <8 x float> @llvm.amdgcn.swmmac.f32.16x16x128.fp8.fp8.v8f32.v8i32.v16i32.i16(<8 x i32> %A, <16 x i32> %B, <8 x float> %C, i16 %Index, i1 false, i1 false)
define amdgpu_ps void @swmmac_f32_16x16x128_fp8_fp8(<8 x i32> %A, <16 x i32> %B, <8 x float> %C, i16 %Index, ptr addrspace(1) %out) {
  %tmp0 = call <8 x float> @llvm.amdgcn.swmmac.f32.16x16x128.fp8.fp8.v8f32.v8i32.v16i32.i16(<8 x i32> %A, <16 x i32> %B, <8 x float> %C, i16 %Index, i1 false, i1 false)
  store <8 x float> %tmp0, ptr addrspace(1) %out
  ret void
}

; CHECK: DIVERGENT:   %tmp0 = call <8 x float> @llvm.amdgcn.swmmac.f32.16x16x128.fp8.bf8.v8f32.v8i32.v16i32.i16(<8 x i32> %A, <16 x i32> %B, <8 x float> %C, i16 %Index, i1 false, i1 false)
define amdgpu_ps void @swmmac_f32_16x16x128_fp8_bf8(<8 x i32> %A, <16 x i32> %B, <8 x float> %C, i16 %Index, ptr addrspace(1) %out) {
  %tmp0 = call <8 x float> @llvm.amdgcn.swmmac.f32.16x16x128.fp8.bf8.v8f32.v8i32.v16i32.i16(<8 x i32> %A, <16 x i32> %B, <8 x float> %C, i16 %Index, i1 false, i1 false)
  store <8 x float> %tmp0, ptr addrspace(1) %out
  ret void
}

; CHECK: DIVERGENT:   %tmp0 = call <8 x float> @llvm.amdgcn.swmmac.f32.16x16x128.bf8.fp8.v8f32.v8i32.v16i32.i16(<8 x i32> %A, <16 x i32> %B, <8 x float> %C, i16 %Index, i1 false, i1 false)
define amdgpu_ps void @swmmac_f32_16x16x128_bf8_fp8(<8 x i32> %A, <16 x i32> %B, <8 x float> %C, i16 %Index, ptr addrspace(1) %out) {
  %tmp0 = call <8 x float> @llvm.amdgcn.swmmac.f32.16x16x128.bf8.fp8.v8f32.v8i32.v16i32.i16(<8 x i32> %A, <16 x i32> %B, <8 x float> %C, i16 %Index, i1 false, i1 false)
  store <8 x float> %tmp0, ptr addrspace(1) %out
  ret void
}

; CHECK: DIVERGENT:   %tmp0 = call <8 x float> @llvm.amdgcn.swmmac.f32.16x16x128.bf8.bf8.v8f32.v8i32.v16i32.i16(<8 x i32> %A, <16 x i32> %B, <8 x float> %C, i16 %Index, i1 false, i1 false)
define amdgpu_ps void @swmmac_f32_16x16x128_bf8_bf8(<8 x i32> %A, <16 x i32> %B, <8 x float> %C, i16 %Index, ptr addrspace(1) %out) {
  %tmp0 = call <8 x float> @llvm.amdgcn.swmmac.f32.16x16x128.bf8.bf8.v8f32.v8i32.v16i32.i16(<8 x i32> %A, <16 x i32> %B, <8 x float> %C, i16 %Index, i1 false, i1 false)
  store <8 x float> %tmp0, ptr addrspace(1) %out
  ret void
}

; CHECK: DIVERGENT:   %tmp0 = call <8 x half> @llvm.amdgcn.swmmac.f16.16x16x128.fp8.fp8.v8f16.v8i32.v16i32.i16(<8 x i32> %A, <16 x i32> %B, <8 x half> %C, i16 %Index, i1 false, i1 false)
define amdgpu_ps void @swmmac_f16_16x16x128_fp8_fp8(<8 x i32> %A, <16 x i32> %B, <8 x half> %C, i16 %Index, ptr addrspace(1) %out) {
  %tmp0 = call <8 x half> @llvm.amdgcn.swmmac.f16.16x16x128.fp8.fp8.v8f16.v8i32.v16i32.i16(<8 x i32> %A, <16 x i32> %B, <8 x half> %C, i16 %Index, i1 false, i1 false)
  store <8 x half> %tmp0, ptr addrspace(1) %out
  ret void
}

; CHECK: DIVERGENT:   %tmp0 = call <8 x half> @llvm.amdgcn.swmmac.f16.16x16x128.fp8.bf8.v8f16.v8i32.v16i32.i16(<8 x i32> %A, <16 x i32> %B, <8 x half> %C, i16 %Index, i1 false, i1 false)
define amdgpu_ps void @swmmac_f16_16x16x128_fp8_bf8(<8 x i32> %A, <16 x i32> %B, <8 x half> %C, i16 %Index, ptr addrspace(1) %out) {
  %tmp0 = call <8 x half> @llvm.amdgcn.swmmac.f16.16x16x128.fp8.bf8.v8f16.v8i32.v16i32.i16(<8 x i32> %A, <16 x i32> %B, <8 x half> %C, i16 %Index, i1 false, i1 false)
  store <8 x half> %tmp0, ptr addrspace(1) %out
  ret void
}

; CHECK: DIVERGENT:   %tmp0 = call <8 x half> @llvm.amdgcn.swmmac.f16.16x16x128.bf8.fp8.v8f16.v8i32.v16i32.i16(<8 x i32> %A, <16 x i32> %B, <8 x half> %C, i16 %Index, i1 false, i1 false)
define amdgpu_ps void @swmmac_f16_16x16x128_bf8_fp8(<8 x i32> %A, <16 x i32> %B, <8 x half> %C, i16 %Index, ptr addrspace(1) %out) {
  %tmp0 = call <8 x half> @llvm.amdgcn.swmmac.f16.16x16x128.bf8.fp8.v8f16.v8i32.v16i32.i16(<8 x i32> %A, <16 x i32> %B, <8 x half> %C, i16 %Index, i1 false, i1 false)
  store <8 x half> %tmp0, ptr addrspace(1) %out
  ret void
}

; CHECK: DIVERGENT:   %tmp0 = call <8 x half> @llvm.amdgcn.swmmac.f16.16x16x128.bf8.bf8.v8f16.v8i32.v16i32.i16(<8 x i32> %A, <16 x i32> %B, <8 x half> %C, i16 %Index, i1 false, i1 false)
define amdgpu_ps void @swmmac_f16_16x16x128_bf8_bf8(<8 x i32> %A, <16 x i32> %B, <8 x half> %C, i16 %Index, ptr addrspace(1) %out) {
  %tmp0 = call <8 x half> @llvm.amdgcn.swmmac.f16.16x16x128.bf8.bf8.v8f16.v8i32.v16i32.i16(<8 x i32> %A, <16 x i32> %B, <8 x half> %C, i16 %Index, i1 false, i1 false)
  store <8 x half> %tmp0, ptr addrspace(1) %out
  ret void
}

; CHECK: DIVERGENT:   %tmp0 = call <8 x i32> @llvm.amdgcn.swmmac.i32.16x16x128.iu8.v8i32.v8i32.v16i32.i16(i1 false, <8 x i32> %A, i1 false, <16 x i32> %B, <8 x i32> %C, i16 %Index, i1 false, i1 false)
define amdgpu_ps void @swmmac_i32_16x16x128_iu8(<8 x i32> %A, <16 x i32> %B, <8 x i32> %C, i16 %Index, ptr addrspace(1) %out) {
  %tmp0 = call <8 x i32> @llvm.amdgcn.swmmac.i32.16x16x128.iu8.v8i32.v8i32.v16i32.i16(i1 0, <8 x i32> %A, i1 0, <16 x i32> %B, <8 x i32> %C, i16 %Index, i1 false, i1 false)
  store <8 x i32> %tmp0, ptr addrspace(1) %out
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call <2 x i32> @llvm.amdgcn.global.load.tr.b64.v2i32(ptr addrspace(1) %addr)
define amdgpu_kernel void @global_load_tr_b64_v2i32(ptr addrspace(1) %addr, ptr addrspace(1) %out) {
bb:
  %tmp0 = call <2 x i32> @llvm.amdgcn.global.load.tr.b64.v2i32(ptr addrspace(1) %addr)
  store <2 x i32> %tmp0, ptr addrspace(1) %out, align 8
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call <8 x i16> @llvm.amdgcn.global.load.tr.b128.v8i16(ptr addrspace(1) %addr)
define amdgpu_kernel void @global_load_tr_b128_v8i16(ptr addrspace(1) %addr, ptr addrspace(1) %out) {
bb:
  %tmp0 = call <8 x i16> @llvm.amdgcn.global.load.tr.b128.v8i16(ptr addrspace(1) %addr)
  store <8 x i16> %tmp0, ptr addrspace(1) %out, align 16
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call <8 x half> @llvm.amdgcn.global.load.tr.b128.v8f16(ptr addrspace(1) %addr)
define amdgpu_kernel void @global_load_tr_b128_v8f16(ptr addrspace(1) %addr, ptr addrspace(1) %out) {
bb:
  %tmp0 = call <8 x half> @llvm.amdgcn.global.load.tr.b128.v8f16(ptr addrspace(1) %addr)
  store <8 x half> %tmp0, ptr addrspace(1) %out, align 16
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call <8 x bfloat> @llvm.amdgcn.global.load.tr.b128.v8bf16(ptr addrspace(1) %addr)
define amdgpu_kernel void @global_load_tr_b128_v8bf16(ptr addrspace(1) %addr, ptr addrspace(1) %out) {
bb:
  %tmp0 = call <8 x bfloat> @llvm.amdgcn.global.load.tr.b128.v8bf16(ptr addrspace(1) %addr)
  store <8 x bfloat> %tmp0, ptr addrspace(1) %out, align 16
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call i32 @llvm.amdgcn.global.load.tr.b64.i32(ptr addrspace(1) %addr)
define amdgpu_kernel void @global_load_tr_b64_i32(ptr addrspace(1) %addr, ptr addrspace(1) %out) {
bb:
  %tmp0 = call i32 @llvm.amdgcn.global.load.tr.b64.i32(ptr addrspace(1) %addr)
  store i32 %tmp0, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call <4 x i16> @llvm.amdgcn.global.load.tr.b128.v4i16(ptr addrspace(1) %addr)
define amdgpu_kernel void @global_load_tr_b128_v4i16(ptr addrspace(1) %addr, ptr addrspace(1) %out) {
bb:
  %tmp0 = call <4 x i16> @llvm.amdgcn.global.load.tr.b128.v4i16(ptr addrspace(1) %addr)
  store <4 x i16> %tmp0, ptr addrspace(1) %out, align 8
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call <4 x half> @llvm.amdgcn.global.load.tr.b128.v4f16(ptr addrspace(1) %addr)
define amdgpu_kernel void @global_load_tr_b128_v4f16(ptr addrspace(1) %addr, ptr addrspace(1) %out) {
bb:
  %tmp0 = call <4 x half> @llvm.amdgcn.global.load.tr.b128.v4f16(ptr addrspace(1) %addr)
  store <4 x half> %tmp0, ptr addrspace(1) %out, align 8
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call <4 x bfloat> @llvm.amdgcn.global.load.tr.b128.v4bf16(ptr addrspace(1) %addr)
define amdgpu_kernel void @global_load_tr_b128_v4bf16(ptr addrspace(1) %addr, ptr addrspace(1) %out) {
bb:
  %tmp0 = call <4 x bfloat> @llvm.amdgcn.global.load.tr.b128.v4bf16(ptr addrspace(1) %addr)
  store <4 x bfloat> %tmp0, ptr addrspace(1) %out, align 8
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call <2 x i32> @llvm.amdgcn.global.load.tr4.b64.v2i32(ptr addrspace(1) %addr)
define amdgpu_kernel void @global_load_tr4_b64_v2i32(ptr addrspace(1) %addr, ptr addrspace(1) %out) {
bb:
  %tmp0 = call <2 x i32> @llvm.amdgcn.global.load.tr4.b64.v2i32(ptr addrspace(1) %addr)
  store <2 x i32> %tmp0, ptr addrspace(1) %out, align 8
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call <3 x i32> @llvm.amdgcn.global.load.tr6.b96.v3i32(ptr addrspace(1) %addr)
define amdgpu_kernel void @global_load_tr6_b96_v3i32(ptr addrspace(1) %addr, ptr addrspace(1) %out) {
bb:
  %tmp0 = call <3 x i32> @llvm.amdgcn.global.load.tr6.b96.v3i32(ptr addrspace(1) %addr)
  store <3 x i32> %tmp0, ptr addrspace(1) %out, align 8
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call <2 x i32> @llvm.amdgcn.ds.load.tr8.b64.v2i32(ptr addrspace(3) %addr)
define amdgpu_kernel void @ds_load_tr8_b64_v2i32(ptr addrspace(3) %addr, ptr addrspace(1) %out) {
bb:
  %tmp0 = call <2 x i32> @llvm.amdgcn.ds.load.tr8.b64.v2i32(ptr addrspace(3) %addr)
  store <2 x i32> %tmp0, ptr addrspace(1) %out, align 8
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call <2 x i32> @llvm.amdgcn.ds.load.tr4.b64.v2i32(ptr addrspace(3) %addr)
define amdgpu_kernel void @ds_load_tr4_b64_v2i32(ptr addrspace(3) %addr, ptr addrspace(1) %out) {
bb:
  %tmp0 = call <2 x i32> @llvm.amdgcn.ds.load.tr4.b64.v2i32(ptr addrspace(3) %addr)
  store <2 x i32> %tmp0, ptr addrspace(1) %out, align 8
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call <3 x i32> @llvm.amdgcn.ds.load.tr6.b96.v3i32(ptr addrspace(3) %addr)
define amdgpu_kernel void @ds_load_tr6_b96_v3i32(ptr addrspace(3) %addr, ptr addrspace(1) %out) {
bb:
  %tmp0 = call <3 x i32> @llvm.amdgcn.ds.load.tr6.b96.v3i32(ptr addrspace(3) %addr)
  store <3 x i32> %tmp0, ptr addrspace(1) %out, align 8
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call <8 x i16> @llvm.amdgcn.ds.load.tr16.b128.v8i16(ptr addrspace(3) %addr)
define amdgpu_kernel void @ds_load_tr16_b128_v8i16(ptr addrspace(3) %addr, ptr addrspace(1) %out) {
bb:
  %tmp0 = call <8 x i16> @llvm.amdgcn.ds.load.tr16.b128.v8i16(ptr addrspace(3) %addr)
  store <8 x i16> %tmp0, ptr addrspace(1) %out, align 16
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call <8 x half> @llvm.amdgcn.ds.load.tr16.b128.v8f16(ptr addrspace(3) %addr)
define amdgpu_kernel void @ds_load_tr16_b128_v8f16(ptr addrspace(3) %addr, ptr addrspace(1) %out) {
bb:
  %tmp0 = call <8 x half> @llvm.amdgcn.ds.load.tr16.b128.v8f16(ptr addrspace(3) %addr)
  store <8 x half> %tmp0, ptr addrspace(1) %out, align 16
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %addr)
define amdgpu_kernel void @ds_load_tr16_b128_v8bf16(ptr addrspace(3) %addr, ptr addrspace(1) %out) {
bb:
  %tmp0 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %addr)
  store <8 x bfloat> %tmp0, ptr addrspace(1) %out, align 16
  ret void
}

declare <2 x i32> @llvm.amdgcn.ds.read.tr4.b64.v2i32(ptr addrspace(3))

; CHECK: DIVERGENT: %tmp0 = call <2 x i32> @llvm.amdgcn.ds.read.tr4.b64.v2i32(ptr addrspace(3) %gep)
define amdgpu_kernel void @ds_read_b64_tr4_v2i32(ptr addrspace(3) %addr, ptr addrspace(1) %out) {
bb:
  %gep = getelementptr i64, ptr addrspace(3) %addr, i32 4
  %tmp0 = call <2 x i32> @llvm.amdgcn.ds.read.tr4.b64.v2i32(ptr addrspace(3) %gep)
  store <2 x i32> %tmp0, ptr addrspace(1) %out, align 8
  ret void
}

declare <3 x i32> @llvm.amdgcn.ds.read.tr6.b96.v3i32(ptr addrspace(3))

; CHECK: DIVERGENT: %tmp0 = call <3 x i32> @llvm.amdgcn.ds.read.tr6.b96.v3i32(ptr addrspace(3) %gep)
define amdgpu_kernel void @ds_read_b96_tr6_v3i32(ptr addrspace(3) %addr, ptr addrspace(1) %out) {
bb:
  %gep = getelementptr i64, ptr addrspace(3) %addr, i32 4
  %tmp0 = call <3 x i32> @llvm.amdgcn.ds.read.tr6.b96.v3i32(ptr addrspace(3) %gep)
  store <3 x i32> %tmp0, ptr addrspace(1) %out, align 16
  ret void
}

declare <2 x i32> @llvm.amdgcn.ds.read.tr8.b64.v2i32(ptr addrspace(3))

; CHECK: DIVERGENT: %tmp0 = call <2 x i32> @llvm.amdgcn.ds.read.tr8.b64.v2i32(ptr addrspace(3) %gep)
define amdgpu_kernel void @ds_read_b64_tr8_v2i32(ptr addrspace(3) %addr, ptr addrspace(1) %out) {
bb:
  %gep = getelementptr i64, ptr addrspace(3) %addr, i32 4
  %tmp0 = call <2 x i32> @llvm.amdgcn.ds.read.tr8.b64.v2i32(ptr addrspace(3) %gep)
  store <2 x i32> %tmp0, ptr addrspace(1) %out, align 8
  ret void
}

declare <4 x i16> @llvm.amdgcn.ds.read.tr16.b64.v4i16(ptr addrspace(3))

; CHECK: DIVERGENT: %tmp0 = call <4 x i16> @llvm.amdgcn.ds.read.tr16.b64.v4i16(ptr addrspace(3) %gep)
define amdgpu_kernel void @ds_read_b64_tr_b16_v4i16(ptr addrspace(3) %addr, ptr addrspace(1) %out) {
bb:
  %gep = getelementptr i64, ptr addrspace(3) %addr, i16 4
  %tmp0 = call <4 x i16> @llvm.amdgcn.ds.read.tr16.b64.v4i16(ptr addrspace(3) %gep)
  store <4 x i16> %tmp0, ptr addrspace(1) %out, align 16
  ret void
}

declare <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3))

; CHECK: DIVERGENT: %tmp0 = call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) %gep)
define amdgpu_kernel void @ds_read_b64_tr_b16_v4f16(ptr addrspace(3) %addr, ptr addrspace(1) %out) {
bb:
  %gep = getelementptr i64, ptr addrspace(3) %addr, i32 4
  %tmp0 = call <4 x half> @llvm.amdgcn.ds.read.tr16.b64.v4f16(ptr addrspace(3) %gep)
  store <4 x half> %tmp0, ptr addrspace(1) %out, align 16
  ret void
}

declare <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3))

; CHECK: DIVERGENT: %tmp0 = call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) %gep)
define amdgpu_kernel void @ds_read_b64_tr_b16_v4bf16(ptr addrspace(3) %addr, ptr addrspace(1) %out) {
bb:
  %gep = getelementptr i64, ptr addrspace(3) %addr, i32 4
  %tmp0 = call <4 x bfloat> @llvm.amdgcn.ds.read.tr16.b64.v4bf16(ptr addrspace(3) %gep)
  store <4 x bfloat> %tmp0, ptr addrspace(1) %out, align 16
  ret void
}

declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half>, <8 x half>, <4 x float>, i32 immarg, i32 immarg, i32 immarg)
declare <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half>, <8 x half>, <16 x float>, i32 immarg, i32 immarg, i32 immarg)
declare <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat>, <8 x bfloat>, <16 x float>, i32 immarg, i32 immarg, i32 immarg)

; CHECK: DIVERGENT: %result = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %arg0, <8 x half> %arg1, <4 x float> %arg2, i32 immarg 0, i32 immarg 0, i32 immarg 0)
define amdgpu_kernel void @mfma_f32_16x16x32_f16(<8 x half> %arg0, <8 x half> %arg1, <4 x float> %arg2, ptr addrspace(1) %out) {
  %result = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %arg0, <8 x half> %arg1, <4 x float> %arg2, i32 immarg 0, i32 immarg 0, i32 immarg 0)
  store <4 x float> %result, ptr addrspace(1) %out
  ret void
}

; CHECK: DIVERGENT: %result = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %arg0, <8 x half> %arg1, <16 x float> %arg2, i32 immarg 0, i32 immarg 0, i32 immarg 0)
define amdgpu_kernel void @mfma_f32_32x32x16_f16(<8 x half> %arg0, <8 x half> %arg1, <16 x float> %arg2, ptr addrspace(1) %out) {
  %result = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> %arg0, <8 x half> %arg1, <16 x float> %arg2, i32 immarg 0, i32 immarg 0, i32 immarg 0)
  store <16 x float> %result, ptr addrspace(1) %out
  ret void
}

; CHECK: DIVERGENT:  %result = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> %arg0, <8 x bfloat> %arg1, <16 x float> %arg2, i32 immarg 0, i32 immarg 0, i32 immarg 0)
define amdgpu_kernel void @mfma_f32_32x32x16_bf16(<8 x bfloat> %arg0, <8 x bfloat> %arg1, <16 x float> %arg2, ptr addrspace(1) %out) {
  %result = call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> %arg0, <8 x bfloat> %arg1, <16 x float> %arg2, i32 immarg 0, i32 immarg 0, i32 immarg 0)
  store <16 x float> %result, ptr addrspace(1) %out
  ret void
}

declare <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32>, <8 x i32>, <4 x float>, i32 immarg, i32 immarg,
                                                                              i32 immarg, i32, i32 immarg, i32)

; CHECK: DIVERGENT:   %result = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %arg0, <8 x i32> %arg1, <4 x float> %arg2, i32 immarg 0, i32 immarg 0, i32 immarg 0, i32 %arg3, i32 immarg 0, i32 %arg4)
define amdgpu_kernel void @mfma_scale_f32_16x16x128_f8f6f4(<8 x i32> %arg0, <8 x i32> %arg1, <4 x float> %arg2, i32 %arg3, i32 %arg4, ptr addrspace(1) %out) {
  %result = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %arg0, <8 x i32> %arg1, <4 x float> %arg2, i32 immarg 0, i32 immarg 0, i32 immarg 0, i32 %arg3, i32 immarg 0, i32 %arg4)
  store <4 x float> %result, ptr addrspace(1) %out
  ret void
}

declare <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4(<8 x i32>, <8 x i32>, <16 x float>, i32 immarg, i32 immarg,
                                                                i32 immarg, i32, i32 immarg, i32)

; CHECK: DIVERGENT: %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 immarg 0, i32 immarg 0, i32 immarg 0, i32 %arg3, i32 immarg 0, i32 %arg4)
define amdgpu_kernel void @mfma_f32_scale_32x32x64_f8f6f4(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 %arg3, i32 %arg4, ptr addrspace(1) %out) {
  %result = call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32(<8 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 immarg 0, i32 immarg 0, i32 immarg 0, i32 %arg3, i32 immarg 0, i32 %arg4)
  store <16 x float> %result, ptr addrspace(1) %out
  ret void
}

declare <4 x i32> @llvm.amdgcn.mfma.i32.16x16x64.i8(<4 x i32>, <4 x i32>, <4 x i32>, i32 immarg, i32 immarg, i32 immarg)

; CHECK: DIVERGENT: %result = call <4 x i32> @llvm.amdgcn.mfma.i32.16x16x64.i8(<4 x i32> %arg0, <4 x i32> %arg1, <4 x i32> %arg2, i32 immarg 0, i32 immarg 0, i32 immarg 0)
define amdgpu_kernel void @mfma_i32_16x16x64_i8(<4 x i32> %arg0, <4 x i32> %arg1, <4 x i32> %arg2, ptr addrspace(1) %out) {
  %result = call <4 x i32> @llvm.amdgcn.mfma.i32.16x16x64.i8(<4 x i32> %arg0, <4 x i32> %arg1, <4 x i32> %arg2, i32 immarg 0, i32 immarg 0, i32 immarg 0)
  store <4 x i32> %result, ptr addrspace(1) %out
  ret void
}

declare <16 x i32> @llvm.amdgcn.mfma.i32.32x32x32.i8(<4 x i32>, <4 x i32>, <16 x i32>, i32 immarg, i32 immarg, i32 immarg)

; CHECK: DIVERGENT: %result = call <16 x i32> @llvm.amdgcn.mfma.i32.32x32x32.i8(<4 x i32> %arg0, <4 x i32> %arg1, <16 x i32> %arg2, i32 immarg 0, i32 immarg 0, i32 immarg 0)
define amdgpu_kernel void @mfma_i32_32x32x32_i8(<4 x i32> %arg0, <4 x i32> %arg1, <16 x i32> %arg2, ptr addrspace(1) %out) {
  %result = call <16 x i32> @llvm.amdgcn.mfma.i32.32x32x32.i8(<4 x i32> %arg0, <4 x i32> %arg1, <16 x i32> %arg2, i32 immarg 0, i32 immarg 0, i32 immarg 0)
  store <16 x i32> %result, ptr addrspace(1) %out
  ret void
}

declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat>, <8 x bfloat>, <4 x float>, i32 immarg, i32 immarg, i32 immarg)

; CHECK: DIVERGENT: %result = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> %arg0, <8 x bfloat> %arg1, <4 x float> %arg2, i32 immarg 0, i32 immarg 0, i32 immarg 0)
define amdgpu_kernel void @mfma_f32_16x16x32_bf16(<8 x bfloat> %arg0, <8 x bfloat> %arg1, <4 x float> %arg2, ptr addrspace(1) %out) {
  %result = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> %arg0, <8 x bfloat> %arg1, <4 x float> %arg2, i32 immarg 0, i32 immarg 0, i32 immarg 0)
  store <4 x float> %result, ptr addrspace(1) %out
  ret void
}

declare <4 x float> @llvm.amdgcn.smmfmac.f32.16x16x64.f16(<8 x half>, <16 x half>, <4 x float>, i32, i32 immarg, i32 immarg)

; CHECK: DIVERGENT:   %result = call <4 x float> @llvm.amdgcn.smfmac.f32.16x16x64.f16(<8 x half> %arg0, <16 x half> %arg1, <4 x float> %arg2, i32 %arg3, i32 immarg 0, i32 immarg 0)
define amdgpu_kernel void @smfmac_f32_16x16x64_f16(<8 x half> %arg0, <16 x half> %arg1, <4 x float> %arg2, i32 %arg3, ptr addrspace(1) %out) {
  %result = call <4 x float> @llvm.amdgcn.smfmac.f32.16x16x64.f16(<8 x half> %arg0, <16 x half> %arg1, <4 x float> %arg2, i32 %arg3, i32 immarg 0, i32 immarg 0)
  store <4 x float> %result, ptr addrspace(1) %out
  ret void
}

declare <16 x float> @llvm.amdgcn.smfmac.f32.32x32x32.f16(<8 x half>, <16 x half>, <16 x float>, i32, i32 immarg, i32 immarg)

; CHECK: DIVERGENT:   %result = call <16 x float> @llvm.amdgcn.smfmac.f32.32x32x32.f16(<8 x half> %arg0, <16 x half> %arg1, <16 x float> %arg2, i32 %arg3, i32 immarg 0, i32 immarg 0)
define amdgpu_kernel void @smfmac_f32_32x32x32_f16(<8 x half> %arg0, <16 x half> %arg1, <16 x float> %arg2, i32 %arg3, ptr addrspace(1) %out) {
  %result = call <16 x float> @llvm.amdgcn.smfmac.f32.32x32x32.f16(<8 x half> %arg0, <16 x half> %arg1, <16 x float> %arg2, i32 %arg3, i32 immarg 0, i32 immarg 0)
  store <16 x float> %result, ptr addrspace(1) %out
  ret void
}

declare <4 x float> @llvm.amdgcn.smmfmac.f32.16x16x64.bf16(<8 x bfloat>, <16 x bfloat>, <4 x float>, i32, i32 immarg, i32 immarg)

; CHECK: DIVERGENT:   %result = call <4 x float> @llvm.amdgcn.smfmac.f32.16x16x64.bf16(<8 x bfloat> %arg0, <16 x bfloat> %arg1, <4 x float> %arg2, i32 %arg3, i32 immarg 0, i32 immarg 0)
define amdgpu_kernel void @smfmac_f32_16x16x64_bf16(<8 x bfloat> %arg0, <16 x bfloat> %arg1, <4 x float> %arg2, i32 %arg3, ptr addrspace(1) %out) {
  %result = call <4 x float> @llvm.amdgcn.smfmac.f32.16x16x64.bf16(<8 x bfloat> %arg0, <16 x bfloat> %arg1, <4 x float> %arg2, i32 %arg3, i32 immarg 0, i32 immarg 0)
  store <4 x float> %result, ptr addrspace(1) %out
  ret void
}

declare <4 x i32> @llvm.amdgcn.smfmac.i32.16x16x128.i8(<4 x i32>, <8 x i32>, <4 x i32>, i32, i32, i32)

; CHECK: DIVERGENT:   %result = call <4 x i32> @llvm.amdgcn.smfmac.i32.16x16x128.i8(<4 x i32> %arg0, <8 x i32> %arg1, <4 x i32> %arg2, i32 %arg3, i32 1, i32 2)
define amdgpu_kernel void @smfmac_i32_16x16x128_i8(<4 x i32> %arg0, <8 x i32> %arg1, <4 x i32> %arg2, i32 %arg3, ptr addrspace(1) %out) {
  %result = call <4 x i32> @llvm.amdgcn.smfmac.i32.16x16x128.i8(<4 x i32> %arg0, <8 x i32> %arg1, <4 x i32> %arg2, i32 %arg3, i32 1, i32 2)
  store <4 x i32> %result, ptr addrspace(1) %out
  ret void
}

declare <4 x float> @llvm.amdgcn.smfmac.f32.16x16x128.bf8.bf8(<4 x i32>, <8 x i32>, <4 x float>, i32, i32, i32)

; CHECK: DIVERGENT:  %result = call <4 x float> @llvm.amdgcn.smfmac.f32.16x16x128.bf8.bf8(<4 x i32> %arg0, <8 x i32> %arg1, <4 x float> %arg2, i32 %arg3, i32 1, i32 2)
define amdgpu_kernel void @smfmac_f32_16x16x128_bf8_bf8(<4 x i32> %arg0, <8 x i32> %arg1, <4 x float> %arg2, i32 %arg3, ptr addrspace(1) %out) {
  %result = call <4 x float> @llvm.amdgcn.smfmac.f32.16x16x128.bf8.bf8(<4 x i32> %arg0, <8 x i32> %arg1, <4 x float> %arg2, i32 %arg3, i32 1, i32 2)
  store <4 x float> %result, ptr addrspace(1) %out
  ret void
}

declare <4 x float> @llvm.amdgcn.smfmac.f32.16x16x128.bf8.fp8(<4 x i32>, <8 x i32>, <4 x float>, i32, i32, i32)

; CHECK: DIVERGENT:  %result = call <4 x float> @llvm.amdgcn.smfmac.f32.16x16x128.bf8.fp8(<4 x i32> %arg0, <8 x i32> %arg1, <4 x float> %arg2, i32 %arg3, i32 1, i32 2)
define amdgpu_kernel void @smfmac_f32_16x16x128_bf8_fp8(<4 x i32> %arg0, <8 x i32> %arg1, <4 x float> %arg2, i32 %arg3, ptr addrspace(1) %out) {
  %result = call <4 x float> @llvm.amdgcn.smfmac.f32.16x16x128.bf8.fp8(<4 x i32> %arg0, <8 x i32> %arg1, <4 x float> %arg2, i32 %arg3, i32 1, i32 2)
  store <4 x float> %result, ptr addrspace(1) %out
  ret void
}

declare <4 x float> @llvm.amdgcn.smfmac.f32.16x16x128.fp8.bf8(<4 x i32>, <8 x i32>, <4 x float>, i32, i32, i32)

; CHECK: DIVERGENT:  %result = call <4 x float> @llvm.amdgcn.smfmac.f32.16x16x128.fp8.bf8(<4 x i32> %arg0, <8 x i32> %arg1, <4 x float> %arg2, i32 %arg3, i32 1, i32 2)
define amdgpu_kernel void @smfmac_f32_16x16x128_fp8_bf8(<4 x i32> %arg0, <8 x i32> %arg1, <4 x float> %arg2, i32 %arg3, ptr addrspace(1) %out) {
  %result = call <4 x float> @llvm.amdgcn.smfmac.f32.16x16x128.fp8.bf8(<4 x i32> %arg0, <8 x i32> %arg1, <4 x float> %arg2, i32 %arg3, i32 1, i32 2)
  store <4 x float> %result, ptr addrspace(1) %out
  ret void
}

declare <4 x float> @llvm.amdgcn.smfmac.f32.16x16x128.fp8.fp8(<4 x i32>, <8 x i32>, <4 x float>, i32, i32, i32)

; CHECK: DIVERGENT:  %result = call <4 x float> @llvm.amdgcn.smfmac.f32.16x16x128.fp8.fp8(<4 x i32> %arg0, <8 x i32> %arg1, <4 x float> %arg2, i32 %arg3, i32 1, i32 2)
define amdgpu_kernel void @smfmac_f32_16x16x128_fp8_fp8(<4 x i32> %arg0, <8 x i32> %arg1, <4 x float> %arg2, i32 %arg3, ptr addrspace(1) %out) {
  %result = call <4 x float> @llvm.amdgcn.smfmac.f32.16x16x128.fp8.fp8(<4 x i32> %arg0, <8 x i32> %arg1, <4 x float> %arg2, i32 %arg3, i32 1, i32 2)
  store <4 x float> %result, ptr addrspace(1) %out
  ret void
}

declare <16 x float> @llvm.amdgcn.smfmac.f32.32x32x64.bf8.bf8(<4 x i32>, <8 x i32>, <16 x float>, i32, i32, i32)

; CHECK: DIVERGENT:  %result = call <16 x float> @llvm.amdgcn.smfmac.f32.32x32x64.bf8.bf8(<4 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 %arg3, i32 1, i32 2)
define amdgpu_kernel void @smfmac_f32_32x32x64_bf8_bf8(<4 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 %arg3, ptr addrspace(1) %out) {
  %result = call <16 x float> @llvm.amdgcn.smfmac.f32.32x32x64.bf8.bf8(<4 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 %arg3, i32 1, i32 2)
  store <16 x float> %result, ptr addrspace(1) %out
  ret void
}

declare <16 x float> @llvm.amdgcn.smfmac.f32.32x32x64.bf8.fp8(<4 x i32>, <8 x i32>, <16 x float>, i32, i32, i32)

; CHECK: DIVERGENT:  %result = call <16 x float> @llvm.amdgcn.smfmac.f32.32x32x64.bf8.fp8(<4 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 %arg3, i32 1, i32 2)
define amdgpu_kernel void @smfmac_f32_32x32x64_bf8_fp8(<4 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 %arg3, ptr addrspace(1) %out) {
  %result = call <16 x float> @llvm.amdgcn.smfmac.f32.32x32x64.bf8.fp8(<4 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 %arg3, i32 1, i32 2)
  store <16 x float> %result, ptr addrspace(1) %out
  ret void
}

declare <16 x float> @llvm.amdgcn.smfmac.f32.32x32x64.fp8.bf8(<4 x i32>, <8 x i32>, <16 x float>, i32, i32, i32)

; CHECK: DIVERGENT:  %result = call <16 x float> @llvm.amdgcn.smfmac.f32.32x32x64.fp8.bf8(<4 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 %arg3, i32 1, i32 2)
define amdgpu_kernel void @smfmac_f32_32x32x64_fp8_bf8(<4 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 %arg3, ptr addrspace(1) %out) {
  %result = call <16 x float> @llvm.amdgcn.smfmac.f32.32x32x64.fp8.bf8(<4 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 %arg3, i32 1, i32 2)
  store <16 x float> %result, ptr addrspace(1) %out
  ret void
}

declare <16 x float> @llvm.amdgcn.smfmac.f32.32x32x64.fp8.fp8(<4 x i32>, <8 x i32>, <16 x float>, i32, i32, i32)

; CHECK: DIVERGENT:  %result = call <16 x float> @llvm.amdgcn.smfmac.f32.32x32x64.fp8.fp8(<4 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 %arg3, i32 1, i32 2)
define amdgpu_kernel void @smfmac_f32_32x32x64_fp8_fp8(<4 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 %arg3, ptr addrspace(1) %out) {
  %result = call <16 x float> @llvm.amdgcn.smfmac.f32.32x32x64.fp8.fp8(<4 x i32> %arg0, <8 x i32> %arg1, <16 x float> %arg2, i32 %arg3, i32 1, i32 2)
  store <16 x float> %result, ptr addrspace(1) %out
  ret void
}

; CHECK: DIVERGENT:   %v = call { i32, i32 } @llvm.amdgcn.permlane16.swap(i32 %src0, i32 %src1, i1 false, i1 false)
define amdgpu_kernel void @v_permlane16_swap(ptr addrspace(1) %out, i32 %src0, i32 %src1) #0 {
  %v = call { i32, i32 } @llvm.amdgcn.permlane16.swap(i32 %src0, i32 %src1, i1 false, i1 false)
  store { i32, i32 } %v, ptr addrspace(1) %out
  ret void
}

; CHECK: DIVERGENT:   %v = call { i32, i32 } @llvm.amdgcn.permlane32.swap(i32 %src0, i32 %src1, i1 false, i1 false)
define amdgpu_kernel void @v_permlane32_swap(ptr addrspace(1) %out, i32 %src0, i32 %src1) #0 {
  %v = call { i32, i32 } @llvm.amdgcn.permlane32.swap(i32 %src0, i32 %src1, i1 false, i1 false)
  store { i32, i32 } %v, ptr addrspace(1) %out
  ret void
}

; CHECK: DIVERGENT:  %result = call i32 @llvm.amdgcn.permlane.bcast(i32 %src0, i32 %src1, i32 %src2)
define amdgpu_kernel void @v_permlane_bcast_b32(ptr addrspace(1) %out, i32 %src0, i32 %src1, i32 %src2) {
  %result= call i32 @llvm.amdgcn.permlane.bcast(i32 %src0, i32 %src1, i32 %src2)
  store i32 %result, ptr addrspace(1) %out
  ret void
}

; CHECK: DIVERGENT:  %result = call i32 @llvm.amdgcn.permlane.up(i32 %src0, i32 %src1, i32 %src2)
define amdgpu_kernel void @v_permlane_up_b32(ptr addrspace(1) %out, i32 %src0, i32 %src1, i32 %src2) {
  %result= call i32 @llvm.amdgcn.permlane.up(i32 %src0, i32 %src1, i32 %src2)
  store i32 %result, ptr addrspace(1) %out
  ret void
}

; CHECK: DIVERGENT:  %result = call i32 @llvm.amdgcn.permlane.down(i32 %src0, i32 %src1, i32 %src2)
define amdgpu_kernel void @v_permlane_down_b32(ptr addrspace(1) %out, i32 %src0, i32 %src1, i32 %src2) {
  %result= call i32 @llvm.amdgcn.permlane.down(i32 %src0, i32 %src1, i32 %src2)
  store i32 %result, ptr addrspace(1) %out
  ret void
}

; CHECK: DIVERGENT:  %result = call i32 @llvm.amdgcn.permlane.xor(i32 %src0, i32 %src1, i32 %src2)
define amdgpu_kernel void @v_permlane_xor_b32(ptr addrspace(1) %out, i32 %src0, i32 %src1, i32 %src2) {
  %result= call i32 @llvm.amdgcn.permlane.xor(i32 %src0, i32 %src1, i32 %src2)
  store i32 %result, ptr addrspace(1) %out
  ret void
}

; CHECK: DIVERGENT:  %result = call i32 @llvm.amdgcn.permlane.idx.gen(i32 %src0, i32 %src1)
define amdgpu_kernel void @v_permlane_idx_gen_b32(ptr addrspace(1) %out, i32 %src0, i32 %src1) {
  %result= call i32 @llvm.amdgcn.permlane.idx.gen(i32 %src0, i32 %src1)
  store i32 %result, ptr addrspace(1) %out
  ret void
}

; CHECK: DIVERGENT:   %v = call i32 @llvm.amdgcn.dead.i32()
define amdgpu_cs_chain void @dead(ptr addrspace(1) %out) {
  %v = call i32 @llvm.amdgcn.dead.i32()
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; CHECK: DIVERGENT:   %v = call i32 (ptr, ...) @llvm.amdgcn.call.whole.wave.i32.p0(ptr @wwf, i32 15)
define amdgpu_cs void @call_whole_wave(ptr addrspace(1) %out) {
  %v = call i32(ptr, ...) @llvm.amdgcn.call.whole.wave.i32.p0(ptr @wwf, i32 15)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

declare amdgpu_gfx_whole_wave i32 @wwf(i1, i32) #0

declare i32 @llvm.amdgcn.ds.swizzle(i32, i32) #1
declare i32 @llvm.amdgcn.permlane16.i32(i32, i32, i32, i32, i1, i1) #1
declare i32 @llvm.amdgcn.permlanex16.i32(i32, i32, i32, i32, i1, i1) #1
declare i32 @llvm.amdgcn.permlane16.var(i32, i32, i32, i1, i1) #1
declare i32 @llvm.amdgcn.permlanex16.var(i32, i32, i32, i1, i1) #1
declare i32 @llvm.amdgcn.mov.dpp.i32(i32, i32, i32, i32, i1) #1
declare i32 @llvm.amdgcn.mov.dpp8.i32(i32, i32) #1
declare i32 @llvm.amdgcn.update.dpp.i32(i32, i32, i32, i32, i32, i1) #1
declare i32 @llvm.amdgcn.writelane.i32(i32, i32, i32) #1
declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v16f16(<16 x half>, <16 x half> , <8 x float>) #1
declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v16i16(<16 x i16>, <16 x i16> , <8 x float>) #1
declare <16 x half> @llvm.amdgcn.wmma.f16.16x16x16.f16.v16f16.v16f16(<16 x half>, <16 x half> , <16 x half>, i1 immarg) #1
declare <16 x i16> @llvm.amdgcn.wmma.bf16.16x16x16.bf16.v16i16.v16i16(<16 x i16>, <16 x i16> , <16 x i16>, i1 immarg) #1
declare <8 x i32> @llvm.amdgcn.wmma.i32.16x16x16.iu8.v8i32.v4i32(i1 immarg, <4 x i32>, i1 immarg, <4 x i32> , <8 x i32>, i1 immarg) #1
declare <8 x i32> @llvm.amdgcn.wmma.i32.16x16x16.iu4.v8i32.v2i32(i1 immarg, <2 x i32>, i1 immarg, <2 x i32> , <8 x i32>, i1 immarg) #1
declare <8 x float> @llvm.amdgcn.swmmac.f32.16x16x32.f16.v8f32.v8f16.v16f16.i16(<8 x half>, <16 x half>, <8 x float>, i16)
declare <8 x float> @llvm.amdgcn.swmmac.f32.16x16x32.bf16.v8f32.v8i16.v16i16.i16(<8 x i16>, <16 x i16>, <8 x float>, i16)
declare <8 x half> @llvm.amdgcn.swmmac.f16.16x16x32.f16.v8f16.v8f16.v16f16.i16(<8 x half>, <16 x half>, <8 x half>, i16)
declare <8 x i16> @llvm.amdgcn.swmmac.bf16.16x16x32.bf16.v8i16.v8i16.v16i16.i16(<8 x i16>, <16 x i16>, <8 x i16>, i16)
declare <8 x i32> @llvm.amdgcn.swmmac.i32.16x16x32.iu8.v8i32.v2i32.v4i32.i16(i1 immarg, <2 x i32>, i1 immarg, <4 x i32>, <8 x i32>, i16, i1)
declare <8 x i32> @llvm.amdgcn.swmmac.i32.16x16x32.iu4.v8i32.i32.v2i32.i16(i1 immarg, i32, i1 immarg, <2 x i32>, <8 x i32>, i16, i1)
declare <8 x i32> @llvm.amdgcn.swmmac.i32.16x16x64.iu4.v8i32.v2i32.v4i32.i16(i1 immarg, <2 x i32>, i1 immarg, <4 x i32>, <8 x i32>, i16, i1)
declare <8 x float> @llvm.amdgcn.swmmac.f32.16x16x32.fp8.fp8(<2 x i32>, <4 x i32>, <8 x float>, i16)
declare <8 x float> @llvm.amdgcn.swmmac.f32.16x16x32.fp8.bf8(<2 x i32>, <4 x i32>, <8 x float>, i16)
declare <8 x float> @llvm.amdgcn.swmmac.f32.16x16x32.bf8.fp8(<2 x i32>, <4 x i32>, <8 x float>, i16)
declare <8 x float> @llvm.amdgcn.swmmac.f32.16x16x32.bf8.bf8(<2 x i32>, <4 x i32>, <8 x float>, i16)
declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x4.f32.v8f32.v2f32(i1, <2 x float>, i1, <2 x float>, i16, <8 x float>, i1, i1)
declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(i1, <16 x bfloat>, i1, <16 x bfloat>, i16, <8 x float>, i1, i1)
declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.f16.v8f32.v16f16(i1, <16 x half>, i1, <16 x half>, i16, <8 x float>, i1, i1)
declare <8 x half> @llvm.amdgcn.wmma.f16.16x16x32.f16.v8f16.v16f16(i1, <16 x half>, i1, <16 x half>, i16, <8 x half>, i1, i1)
declare <8 x bfloat> @llvm.amdgcn.wmma.bf16.16x16x32.bf16.v8bf16.v16bf16(i1, <16 x bfloat>, i1, <16 x bfloat>, i16, <8 x bfloat>, i1, i1)
declare <8 x bfloat> @llvm.amdgcn.wmma.bf16f32.16x16x32.bf16.v8bf16.v16bf16(i1, <16 x bfloat>, i1, <16 x bfloat>, i16, <8 x float>, i1, i1)
declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x64.fp8.fp8.v8f32.v8i32(<8 x i32>, <8 x i32>, i16, <8 x float>, i1, i1)
declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x64.fp8.bf8.v8f32.v8i32(<8 x i32>, <8 x i32>, i16, <8 x float>, i1, i1)
declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x64.bf8.fp8.v8f32.v8i32(<8 x i32>, <8 x i32>, i16, <8 x float>, i1, i1)
declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x64.bf8.bf8.v8f32.v8i32(<8 x i32>, <8 x i32>, i16, <8 x float>, i1, i1)
declare <8 x half> @llvm.amdgcn.wmma.f16.16x16x64.fp8.fp8.v8f16.v8i32(<8 x i32>, <8 x i32>, i16, <8 x half>, i1, i1)
declare <8 x half> @llvm.amdgcn.wmma.f16.16x16x64.fp8.bf8.v8f16.v8i32(<8 x i32>, <8 x i32>, i16, <8 x half>, i1, i1)
declare <8 x half> @llvm.amdgcn.wmma.f16.16x16x64.bf8.fp8.v8f16.v8i32(<8 x i32>, <8 x i32>, i16, <8 x half>, i1, i1)
declare <8 x half> @llvm.amdgcn.wmma.f16.16x16x64.bf8.bf8.v8f16.v8i32(<8 x i32>, <8 x i32>, i16, <8 x half>, i1, i1)
declare <8 x i32> @llvm.amdgcn.wmma.i32.16x16x64.iu8.v8i32.v8i32(i1 immarg, <8 x i32>, i1 immarg, <8 x i32>, <8 x i32>, i1, i1)
declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32, <16 x i32>, i32, <16 x i32>, i16, <8 x float>)
declare <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32, <16 x i32>, i32, <16 x i32>, i16, <8 x float>, i32, i32, i32, i32, i32, i32, i1, i1)
declare <8 x float> @llvm.amdgcn.wmma.scale16.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32, <16 x i32>, i32, <16 x i32>, i16, <8 x float>, i32, i32, i64, i32, i32, i64, i1, i1)
declare <8 x float> @llvm.amdgcn.swmmac.f32.16x16x64.f16.v8f32.v16f16.v32f16.i16(i1, <16 x half>, i1, <32 x half>, <8 x float>, i16, i1, i1)
declare <8 x float> @llvm.amdgcn.swmmac.f32.16x16x64.bf16.v8f32.v16bf16.v32bf16.i16(i1, <16 x bfloat>, i1, <32 x bfloat>, <8 x float>, i16, i1, i1)
declare <8 x half> @llvm.amdgcn.swmmac.f16.16x16x64.f16.v8f16.v16f16.v32f16.i16(i1, <16 x half>, i1, <32 x half>, <8 x half>, i16, i1, i1)
declare <8 x bfloat> @llvm.amdgcn.swmmac.bf16.16x16x64.bf16.v8bf16.v16bf16.v32bf16.i16(i1, <16 x bfloat>, i1, <32 x bfloat>, <8 x bfloat>, i16, i1, i1)
declare <8 x float> @llvm.amdgcn.swmmac.bf16f32.16x16x64.bf16.v8f32.v16bf16.v32bf16.i16(i1, <16 x bfloat>, i1, <32 x bfloat>, <8 x float>, i16, i1, i1)
declare <8 x float> @llvm.amdgcn.swmmac.f32.16x16x128.fp8.fp8.v8f32.v8i32.v16i32.i16(<8 x i32>, <16 x i32>, <8 x float>, i16, i1, i1)
declare <8 x float> @llvm.amdgcn.swmmac.f32.16x16x128.fp8.bf8.v8f32.v8i32.v16i32.i16(<8 x i32>, <16 x i32>, <8 x float>, i16, i1, i1)
declare <8 x float> @llvm.amdgcn.swmmac.f32.16x16x128.bf8.fp8.v8f32.v8i32.v16i32.i16(<8 x i32>, <16 x i32>, <8 x float>, i16, i1, i1)
declare <8 x float> @llvm.amdgcn.swmmac.f32.16x16x128.bf8.bf8.v8f32.v8i32.v16i32.i16(<8 x i32>, <16 x i32>, <8 x float>, i16, i1, i1)
declare <8 x half> @llvm.amdgcn.swmmac.f16.16x16x128.fp8.fp8.v8f16.v8i32.v16i32.i16(<8 x i32>, <16 x i32>, <8 x half>, i16, i1, i1)
declare <8 x half> @llvm.amdgcn.swmmac.f16.16x16x128.fp8.bf8.v8f16.v8i32.v16i32.i16(<8 x i32>, <16 x i32>, <8 x half>, i16, i1, i1)
declare <8 x half> @llvm.amdgcn.swmmac.f16.16x16x128.bf8.fp8.v8f16.v8i32.v16i32.i16(<8 x i32>, <16 x i32>, <8 x half>, i16, i1, i1)
declare <8 x half> @llvm.amdgcn.swmmac.f16.16x16x128.bf8.bf8.v8f16.v8i32.v16i32.i16(<8 x i32>, <16 x i32>, <8 x half>, i16, i1, i1)
declare <8 x i32> @llvm.amdgcn.swmmac.i32.16x16x128.iu8.v8i32.v8i32.v16i32.i16(i1 immarg, <8 x i32>, i1 immarg, <16 x i32>, <8 x i32>, i16 %Index, i1, i1)

declare <2 x i32> @llvm.amdgcn.global.load.tr.b64.v2i32(ptr addrspace(1))
declare <8 x i16> @llvm.amdgcn.global.load.tr.b128.v8i16(ptr addrspace(1))
declare <8 x half> @llvm.amdgcn.global.load.tr.b128.v8f16(ptr addrspace(1))
declare <8 x bfloat> @llvm.amdgcn.global.load.tr.b128.v8bf16(ptr addrspace(1))
declare i32 @llvm.amdgcn.global.load.tr.b64.i32(ptr addrspace(1))
declare <4 x i16> @llvm.amdgcn.global.load.tr.b128.v4i16(ptr addrspace(1))
declare <4 x half> @llvm.amdgcn.global.load.tr.b128.v4f16(ptr addrspace(1))
declare <4 x bfloat> @llvm.amdgcn.global.load.tr.b128.v4bf16(ptr addrspace(1))
declare <2 x i32> @llvm.amdgcn.global.load.tr4.b64.v2i32(ptr addrspace(1))
declare <3 x i32> @llvm.amdgcn.global.load.tr6.b96.v3i32(ptr addrspace(1))
declare <2 x i32> @llvm.amdgcn.ds.load.tr8.b64.v2i32(ptr addrspace(3))
declare <2 x i32> @llvm.amdgcn.ds.load.tr4.b64.v2i32(ptr addrspace(3))
declare <3 x i32> @llvm.amdgcn.ds.load.tr6.b96.v3i32(ptr addrspace(3))
declare <8 x i16> @llvm.amdgcn.ds.load.tr16.b128.v8i16(ptr addrspace(3))
declare <8 x half> @llvm.amdgcn.ds.load.tr16.b128.v8f16(ptr addrspace(3))
declare <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3))

declare i32 @llvm.amdgcn.dead.i32()
declare i32 @llvm.amdgcn.call.whole.wave.i32.p0(ptr, ...) #0

attributes #0 = { nounwind convergent }
attributes #1 = { nounwind readnone convergent }
