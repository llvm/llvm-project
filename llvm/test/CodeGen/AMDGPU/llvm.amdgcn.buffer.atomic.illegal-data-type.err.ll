; RUN: not llc -mtriple=amdgpu9.0a -O0 -filetype=null %s 2>&1 | FileCheck %s

; llvm.amdgcn.{raw,struct}.ptr.buffer.atomic.<op> only supports 32-bit and
; 64-bit data.

; CHECK: error: {{.*}}unsupported buffer atomic data type
define amdgpu_kernel void @raw_add_v3i16(ptr addrspace(8) %rsrc, <3 x i16> %data, ptr addrspace(1) %out) {
  %v = call <3 x i16> @llvm.amdgcn.raw.ptr.buffer.atomic.add.v3i16(<3 x i16> %data, ptr addrspace(8) %rsrc, i32 0, i32 0, i32 0)
  store <3 x i16> %v, ptr addrspace(1) %out
  ret void
}

; CHECK: error: {{.*}}unsupported buffer atomic data type
define amdgpu_kernel void @raw_add_i128(ptr addrspace(8) %rsrc, i128 %data, ptr addrspace(1) %out) {
  %v = call i128 @llvm.amdgcn.raw.ptr.buffer.atomic.add.i128(i128 %data, ptr addrspace(8) %rsrc, i32 0, i32 0, i32 0)
  store i128 %v, ptr addrspace(1) %out
  ret void
}

; CHECK: error: {{.*}}unsupported buffer atomic data type
define amdgpu_kernel void @raw_fadd_bf16(ptr addrspace(8) %rsrc, bfloat %data, ptr addrspace(1) %out) {
  %v = call bfloat @llvm.amdgcn.raw.ptr.buffer.atomic.fadd.bf16(bfloat %data, ptr addrspace(8) %rsrc, i32 0, i32 0, i32 0)
  store bfloat %v, ptr addrspace(1) %out
  ret void
}

; CHECK: error: {{.*}}unsupported buffer atomic data type
define amdgpu_kernel void @struct_add_v3i16(ptr addrspace(8) %rsrc, <3 x i16> %data, i32 %idx, ptr addrspace(1) %out) {
  %v = call <3 x i16> @llvm.amdgcn.struct.ptr.buffer.atomic.add.v3i16(<3 x i16> %data, ptr addrspace(8) %rsrc, i32 %idx, i32 0, i32 0, i32 0)
  store <3 x i16> %v, ptr addrspace(1) %out
  ret void
}

; CHECK: error: {{.*}}unsupported buffer atomic data type
define amdgpu_kernel void @raw_cmpswap_i128(ptr addrspace(8) %rsrc, i128 %cmp, i128 %data, ptr addrspace(1) %out) {
  %v = call i128 @llvm.amdgcn.raw.ptr.buffer.atomic.cmpswap.i128(i128 %cmp, i128 %data, ptr addrspace(8) %rsrc, i32 0, i32 0, i32 0)
  store i128 %v, ptr addrspace(1) %out
  ret void
}

; CHECK: error: {{.*}}unsupported buffer atomic data type
define amdgpu_kernel void @struct_cmpswap_i128(ptr addrspace(8) %rsrc, i128 %cmp, i128 %data, i32 %idx, ptr addrspace(1) %out) {
  %v = call i128 @llvm.amdgcn.struct.ptr.buffer.atomic.cmpswap.i128(i128 %cmp, i128 %data, ptr addrspace(8) %rsrc, i32 %idx, i32 0, i32 0, i32 0)
  store i128 %v, ptr addrspace(1) %out
  ret void
}

declare <3 x i16> @llvm.amdgcn.raw.ptr.buffer.atomic.add.v3i16(<3 x i16>, ptr addrspace(8), i32, i32, i32 immarg)
declare i128 @llvm.amdgcn.raw.ptr.buffer.atomic.add.i128(i128, ptr addrspace(8), i32, i32, i32 immarg)
declare bfloat @llvm.amdgcn.raw.ptr.buffer.atomic.fadd.bf16(bfloat, ptr addrspace(8), i32, i32, i32 immarg)
declare <3 x i16> @llvm.amdgcn.struct.ptr.buffer.atomic.add.v3i16(<3 x i16>, ptr addrspace(8), i32, i32, i32, i32 immarg)
declare i128 @llvm.amdgcn.raw.ptr.buffer.atomic.cmpswap.i128(i128, i128, ptr addrspace(8), i32, i32, i32 immarg)
declare i128 @llvm.amdgcn.struct.ptr.buffer.atomic.cmpswap.i128(i128, i128, ptr addrspace(8), i32, i32, i32, i32 immarg)
