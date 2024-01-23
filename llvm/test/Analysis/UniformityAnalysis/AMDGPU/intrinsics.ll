; RUN: opt -mtriple amdgcn-- -passes='print<uniformity>' -disable-output %s 2>&1 | FileCheck %s

; CHECK: DIVERGENT: %swizzle = call i32 @llvm.amdgcn.ds.swizzle(i32 %src, i32 100) #0
define amdgpu_kernel void @ds_swizzle(ptr addrspace(1) %out, i32 %src) #0 {
  %swizzle = call i32 @llvm.amdgcn.ds.swizzle(i32 %src, i32 100) #0
  store i32 %swizzle, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK: DIVERGENT: %v = call i32 @llvm.amdgcn.permlane16(i32 %src0, i32 %src0, i32 %src1, i32 %src2, i1 false, i1 false) #0
define amdgpu_kernel void @v_permlane16_b32(ptr addrspace(1) %out, i32 %src0, i32 %src1, i32 %src2) #0 {
  %v = call i32 @llvm.amdgcn.permlane16(i32 %src0, i32 %src0, i32 %src1, i32 %src2, i1 false, i1 false) #0
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; CHECK: DIVERGENT: %v = call i32 @llvm.amdgcn.permlanex16(i32 %src0, i32 %src0, i32 %src1, i32 %src2, i1 false, i1 false) #0
define amdgpu_kernel void @v_permlanex16_b32(ptr addrspace(1) %out, i32 %src0, i32 %src1, i32 %src2) #0 {
  %v = call i32 @llvm.amdgcn.permlanex16(i32 %src0, i32 %src0, i32 %src1, i32 %src2, i1 false, i1 false) #0
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

; CHECK: DIVERGENT: %tmp0 = call i32 @llvm.amdgcn.writelane(i32 0, i32 1, i32 2)
define amdgpu_kernel void @writelane(ptr addrspace(1) %out) #0 {
  %tmp0 = call i32 @llvm.amdgcn.writelane(i32 0, i32 1, i32 2)
  store i32 %tmp0, ptr addrspace(1) %out
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32(<16 x half> %A, <16 x half> %B, <8 x float> %C)
define amdgpu_kernel void @wmma_f32_16x16x16_f16(<16 x half> %A, <16 x half> %B, <8 x float> %C, ptr addrspace(1) %out) {
  %tmp0 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32(<16 x half> %A, <16 x half> %B, <8 x float> %C)
  store <8 x float> %tmp0, ptr addrspace(1) %out, align 32
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32(<16 x i16> %A, <16 x i16> %B, <8 x float> %C)
define amdgpu_kernel void @wmma_f32_16x16x16_ibf16(<16 x i16> %A, <16 x i16> %B, <8 x float> %C, ptr addrspace(1) %out) {
  %tmp0 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32(<16 x i16> %A, <16 x i16> %B, <8 x float> %C)
  store <8 x float> %tmp0, ptr addrspace(1) %out, align 32
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call <16 x half> @llvm.amdgcn.wmma.f16.16x16x16.f16.v16f16(<16 x half> %A, <16 x half> %B, <16 x half> %C, i1 false)
define amdgpu_kernel void @wmma_f16_16x16x16_f16(<16 x half> %A, <16 x half> %B, <16 x half> %C, ptr addrspace(1) %out) {
bb:
  %tmp0 = call <16 x half> @llvm.amdgcn.wmma.f16.16x16x16.f16.v16f16(<16 x half> %A, <16 x half> %B, <16 x half> %C, i1 false)
  store <16 x half> %tmp0, ptr addrspace(1) %out, align 32
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call <16 x i16> @llvm.amdgcn.wmma.bf16.16x16x16.bf16.v16i16(<16 x i16> %A, <16 x i16> %B, <16 x i16> %C, i1 false)
define amdgpu_kernel void @wmma_f16_16x16x16_bf16(<16 x i16> %A, <16 x i16> %B, <16 x i16> %C, ptr addrspace(1) %out) {
bb:
  %tmp0 = call <16 x i16> @llvm.amdgcn.wmma.bf16.16x16x16.bf16.v16i16(<16 x i16> %A, <16 x i16> %B, <16 x i16> %C, i1 false)
  store <16 x i16> %tmp0, ptr addrspace(1) %out, align 32
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call <8 x i32> @llvm.amdgcn.wmma.i32.16x16x16.iu8.v8i32(i1 false, <4 x i32> %A, i1 false, <4 x i32> %B, <8 x i32> %C, i1 false)
define amdgpu_kernel void @wmma_i32_16x16x16_ui8(<4 x i32> %A, <4 x i32> %B, <8 x i32> %C, ptr addrspace(1) %out) {
bb:
  %tmp0 = call <8 x i32> @llvm.amdgcn.wmma.i32.16x16x16.iu8.v8i32(i1 false, <4 x i32> %A, i1 false, <4 x i32> %B, <8 x i32> %C, i1 false)
  store <8 x i32> %tmp0, ptr addrspace(1) %out, align 32
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call <8 x i32> @llvm.amdgcn.wmma.i32.16x16x16.iu4.v8i32(i1 false, <2 x i32> %A, i1 false, <2 x i32> %B, <8 x i32> %C, i1 false)
define amdgpu_kernel void @wmma_i32_16x16x16_ui4(<2 x i32> %A, <2 x i32> %B, <8 x i32> %C, ptr addrspace(1) %out) {
bb:
  %tmp0 = call <8 x i32> @llvm.amdgcn.wmma.i32.16x16x16.iu4.v8i32(i1 false, <2 x i32> %A, i1 false, <2 x i32> %B, <8 x i32> %C, i1 false)
  store <8 x i32> %tmp0, ptr addrspace(1) %out, align 32
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call <2 x i32> @llvm.amdgcn.global.load.tr.v2i32(ptr addrspace(1) %gep)
define amdgpu_kernel void @global_load_tr_b64_v2i32(ptr addrspace(1) %addr, ptr addrspace(1) %out) {
bb:
  %gep = getelementptr i64, ptr addrspace(1) %addr, i32 4
  %tmp0 = call <2 x i32> @llvm.amdgcn.global.load.tr.v2i32(ptr addrspace(1) %gep)
  store <2 x i32> %tmp0, ptr addrspace(1) %out, align 8
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call <8 x i16> @llvm.amdgcn.global.load.tr.v8i16(ptr addrspace(1) %gep)
define amdgpu_kernel void @global_load_tr_b128_v8i16(ptr addrspace(1) %addr, ptr addrspace(1) %out) {
bb:
  %gep = getelementptr i64, ptr addrspace(1) %addr, i32 4
  %tmp0 = call <8 x i16> @llvm.amdgcn.global.load.tr.v8i16(ptr addrspace(1) %gep)
  store <8 x i16> %tmp0, ptr addrspace(1) %out, align 16
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call <8 x half> @llvm.amdgcn.global.load.tr.v8f16(ptr addrspace(1) %gep)
define amdgpu_kernel void @global_load_tr_b128_v8f16(ptr addrspace(1) %addr, ptr addrspace(1) %out) {
bb:
  %gep = getelementptr i64, ptr addrspace(1) %addr, i32 4
  %tmp0 = call <8 x half> @llvm.amdgcn.global.load.tr.v8f16(ptr addrspace(1) %gep)
  store <8 x half> %tmp0, ptr addrspace(1) %out, align 16
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call <8 x bfloat> @llvm.amdgcn.global.load.tr.v8bf16(ptr addrspace(1) %gep)
define amdgpu_kernel void @global_load_tr_b128_v8bf16(ptr addrspace(1) %addr, ptr addrspace(1) %out) {
bb:
  %gep = getelementptr i64, ptr addrspace(1) %addr, i32 4
  %tmp0 = call <8 x bfloat> @llvm.amdgcn.global.load.tr.v8bf16(ptr addrspace(1) %gep)
  store <8 x bfloat> %tmp0, ptr addrspace(1) %out, align 16
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call i32 @llvm.amdgcn.global.load.tr.i32(ptr addrspace(1) %gep)
define amdgpu_kernel void @global_load_tr_b64_i32(ptr addrspace(1) %addr, ptr addrspace(1) %out) {
bb:
  %gep = getelementptr i64, ptr addrspace(1) %addr, i32 4
  %tmp0 = call i32 @llvm.amdgcn.global.load.tr.i32(ptr addrspace(1) %gep)
  store i32 %tmp0, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call <4 x i16> @llvm.amdgcn.global.load.tr.v4i16(ptr addrspace(1) %gep)
define amdgpu_kernel void @global_load_tr_b128_v4i16_(ptr addrspace(1) %addr, ptr addrspace(1) %out) {
bb:
  %gep = getelementptr i64, ptr addrspace(1) %addr, i32 4
  %tmp0 = call <4 x i16> @llvm.amdgcn.global.load.tr.v4i16(ptr addrspace(1) %gep)
  store <4 x i16> %tmp0, ptr addrspace(1) %out, align 8
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call <4 x half> @llvm.amdgcn.global.load.tr.v4f16(ptr addrspace(1) %gep)
define amdgpu_kernel void @global_load_tr_b128_v4f16(ptr addrspace(1) %addr, ptr addrspace(1) %out) {
bb:
  %gep = getelementptr i64, ptr addrspace(1) %addr, i32 4
  %tmp0 = call <4 x half> @llvm.amdgcn.global.load.tr.v4f16(ptr addrspace(1) %gep)
  store <4 x half> %tmp0, ptr addrspace(1) %out, align 8
  ret void
}

; CHECK: DIVERGENT: %tmp0 = call <4 x bfloat> @llvm.amdgcn.global.load.tr.v4bf16(ptr addrspace(1) %gep)
define amdgpu_kernel void @global_load_tr_b128_v4bf16(ptr addrspace(1) %addr, ptr addrspace(1) %out) {
bb:
  %gep = getelementptr i64, ptr addrspace(1) %addr, i32 4
  %tmp0 = call <4 x bfloat> @llvm.amdgcn.global.load.tr.v4bf16(ptr addrspace(1) %gep)
  store <4 x bfloat> %tmp0, ptr addrspace(1) %out, align 8
  ret void
}

declare i32 @llvm.amdgcn.ds.swizzle(i32, i32) #1
declare i32 @llvm.amdgcn.permlane16(i32, i32, i32, i32, i1, i1) #1
declare i32 @llvm.amdgcn.permlanex16(i32, i32, i32, i32, i1, i1) #1
declare i32 @llvm.amdgcn.permlane16.var(i32, i32, i32, i1, i1) #1
declare i32 @llvm.amdgcn.permlanex16.var(i32, i32, i32, i1, i1) #1
declare i32 @llvm.amdgcn.mov.dpp.i32(i32, i32, i32, i32, i1) #1
declare i32 @llvm.amdgcn.mov.dpp8.i32(i32, i32) #1
declare i32 @llvm.amdgcn.update.dpp.i32(i32, i32, i32, i32, i32, i1) #1
declare i32 @llvm.amdgcn.writelane(i32, i32, i32) #1
declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32(<16 x half>, <16 x half> , <8 x float>) #1
declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32(<16 x i16>, <16 x i16> , <8 x float>) #1
declare <16 x half> @llvm.amdgcn.wmma.f16.16x16x16.f16.v16f16(<16 x half>, <16 x half> , <16 x half>, i1 immarg) #1
declare <16 x i16> @llvm.amdgcn.wmma.bf16.16x16x16.bf16.v16i16(<16 x i16>, <16 x i16> , <16 x i16>, i1 immarg) #1
declare <8 x i32> @llvm.amdgcn.wmma.i32.16x16x16.iu8.v8i32(i1 immarg, <4 x i32>, i1 immarg, <4 x i32> , <8 x i32>, i1 immarg) #1
declare <8 x i32> @llvm.amdgcn.wmma.i32.16x16x16.iu4.v8i32(i1 immarg, <2 x i32>, i1 immarg, <2 x i32> , <8 x i32>, i1 immarg) #1

declare <2 x i32> @llvm.amdgcn.global.load.tr.v2i32(ptr addrspace(1))
declare <8 x i16> @llvm.amdgcn.global.load.tr.v8i16(ptr addrspace(1))
declare <8 x half> @llvm.amdgcn.global.load.tr.v8f16(ptr addrspace(1))
declare <8 x bfloat> @llvm.amdgcn.global.load.tr.v8bf16(ptr addrspace(1))
declare i32 @llvm.amdgcn.global.load.tr.i32(ptr addrspace(1))
declare <4 x i16> @llvm.amdgcn.global.load.tr.v4i16(ptr addrspace(1))
declare <4 x half> @llvm.amdgcn.global.load.tr.v4f16(ptr addrspace(1))
declare <4 x bfloat> @llvm.amdgcn.global.load.tr.v4bf16(ptr addrspace(1))

attributes #0 = { nounwind convergent }
attributes #1 = { nounwind readnone convergent }
