; RUN: not llc -mtriple=amdgpu9.0a -filetype=null %s 2>&1 | FileCheck %s

; llvm.amdgcn.image.atomic.<op> only supports 32-bit, 64-bit, and packed
; f16/bf16 data.

; CHECK: error: {{.*}}unsupported image atomic data type
define amdgpu_kernel void @swap_v3i32(<8 x i32> inreg %rsrc, i32 %s, <3 x i32> %data, ptr addrspace(1) %out) {
  %v = call <3 x i32> @llvm.amdgcn.image.atomic.swap.1d.v3i32.i32(<3 x i32> %data, i32 %s, <8 x i32> %rsrc, i32 0, i32 0)
  store <3 x i32> %v, ptr addrspace(1) %out
  ret void
}

; CHECK: error: {{.*}}unsupported image atomic data type
define amdgpu_kernel void @swap_v3i16(<8 x i32> inreg %rsrc, i32 %s, <3 x i16> %data, ptr addrspace(1) %out) {
  %v = call <3 x i16> @llvm.amdgcn.image.atomic.swap.1d.v3i16.i32(<3 x i16> %data, i32 %s, <8 x i32> %rsrc, i32 0, i32 0)
  store <3 x i16> %v, ptr addrspace(1) %out
  ret void
}

; CHECK: error: {{.*}}unsupported image atomic data type
define amdgpu_kernel void @swap_bf16(<8 x i32> inreg %rsrc, i32 %s, bfloat %data, ptr addrspace(1) %out) {
  %v = call bfloat @llvm.amdgcn.image.atomic.swap.1d.bf16.i32(bfloat %data, i32 %s, <8 x i32> %rsrc, i32 0, i32 0)
  store bfloat %v, ptr addrspace(1) %out
  ret void
}

; CHECK: error: {{.*}}unsupported image atomic data type
define amdgpu_kernel void @swap_i128(<8 x i32> inreg %rsrc, i32 %s, i128 %data, ptr addrspace(1) %out) {
  %v = call i128 @llvm.amdgcn.image.atomic.swap.1d.i128.i32(i128 %data, i32 %s, <8 x i32> %rsrc, i32 0, i32 0)
  store i128 %v, ptr addrspace(1) %out
  ret void
}

; CHECK: error: {{.*}}unsupported image atomic data type
define amdgpu_kernel void @cmpswap_i128(<8 x i32> inreg %rsrc, i32 %s, i128 %cmp, i128 %data, ptr addrspace(1) %out) {
  %v = call i128 @llvm.amdgcn.image.atomic.cmpswap.1d.i128.i32(i128 %cmp, i128 %data, i32 %s, <8 x i32> %rsrc, i32 0, i32 0)
  store i128 %v, ptr addrspace(1) %out
  ret void
}

declare <3 x i32> @llvm.amdgcn.image.atomic.swap.1d.v3i32.i32(<3 x i32>, i32, <8 x i32>, i32, i32)
declare <3 x i16> @llvm.amdgcn.image.atomic.swap.1d.v3i16.i32(<3 x i16>, i32, <8 x i32>, i32, i32)
declare bfloat @llvm.amdgcn.image.atomic.swap.1d.bf16.i32(bfloat, i32, <8 x i32>, i32, i32)
declare i128 @llvm.amdgcn.image.atomic.swap.1d.i128.i32(i128, i32, <8 x i32>, i32, i32)
declare i128 @llvm.amdgcn.image.atomic.cmpswap.1d.i128.i32(i128, i128, i32, <8 x i32>, i32, i32)
