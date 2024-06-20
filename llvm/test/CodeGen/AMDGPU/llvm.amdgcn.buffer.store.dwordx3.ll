;RUN: llc < %s -mtriple=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck %s -check-prefix=CHECK

;CHECK-LABEL: {{^}}buffer_store_format_immoffs_x3:
;CHECK-NOT: s_waitcnt
;CHECK: buffer_store_format_xyz v[0:2], off, s[0:3], 0 offset:42
define amdgpu_ps void @buffer_store_format_immoffs_x3(<4 x i32> inreg, <3 x float>) {
main_body:
  call void @llvm.amdgcn.buffer.store.format.v3f32(<3 x float> %1, <4 x i32> %0, i32 0, i32 42, i1 0, i1 0)
  ret void
}

;CHECK-LABEL: {{^}}buffer_store_immoffs_x3:
;CHECK-NOT: s_waitcnt
;CHECK: buffer_store_dwordx3 v[0:2], off, s[0:3], 0 offset:42
define amdgpu_ps void @buffer_store_immoffs_x3(<4 x i32> inreg, <3 x float>) {
main_body:
  call void @llvm.amdgcn.buffer.store.v3f32(<3 x float> %1, <4 x i32> %0, i32 0, i32 42, i1 0, i1 0)
  ret void
}

;CHECK-LABEL: {{^}}raw_buffer_store_format_immoffs_x3:
;CHECK-NOT: s_waitcnt
;CHECK: buffer_store_format_xyz v[0:2], off, s[0:3], 0 offset:42
define amdgpu_ps void @raw_buffer_store_format_immoffs_x3(<4 x i32> inreg, <3 x float>) {
main_body:
  call void @llvm.amdgcn.raw.buffer.store.format.v3f32(<3 x float> %1, <4 x i32> %0, i32 42, i32 0, i32 0)
  ret void
}

;CHECK-LABEL: {{^}}raw_buffer_store_immoffs_x3:
;CHECK-NOT: s_waitcnt
;CHECK: buffer_store_dwordx3 v[0:2], off, s[0:3], 0 offset:42
define amdgpu_ps void @raw_buffer_store_immoffs_x3(<4 x i32> inreg, <3 x float>) {
main_body:
  call void @llvm.amdgcn.raw.buffer.store.v3f32(<3 x float> %1, <4 x i32> %0, i32 42, i32 0, i32 0)
  ret void
}

;CHECK-LABEL: {{^}}struct_buffer_store_immoffs_x3:
;CHECK-NOT: s_waitcnt
;CHECK: buffer_store_dwordx3 v[0:2], {{v[0-9]+}}, s[0:3], 0 idxen offset:42
define amdgpu_ps void @struct_buffer_store_immoffs_x3(<4 x i32> inreg, <3 x float>) {
main_body:
  call void @llvm.amdgcn.struct.buffer.store.v3f32(<3 x float> %1, <4 x i32> %0, i32 0, i32 42, i32 0, i32 0)
  ret void
}

;CHECK-LABEL: {{^}}raw_ptr_buffer_store_format_immoffs_x3:
;CHECK-NOT: s_waitcnt
;CHECK: buffer_store_format_xyz v[0:2], off, s[0:3], 0 offset:42
define amdgpu_ps void @raw_ptr_buffer_store_format_immoffs_x3(ptr addrspace(8) inreg, <3 x float>) {
main_body:
  call void @llvm.amdgcn.raw.ptr.buffer.store.format.v3f32(<3 x float> %1, ptr addrspace(8) %0, i32 42, i32 0, i32 0)
  ret void
}

;CHECK-LABEL: {{^}}raw_ptr_buffer_store_immoffs_x3:
;CHECK-NOT: s_waitcnt
;CHECK: buffer_store_dwordx3 v[0:2], off, s[0:3], 0 offset:42
define amdgpu_ps void @raw_ptr_buffer_store_immoffs_x3(ptr addrspace(8) inreg, <3 x float>) {
main_body:
  call void @llvm.amdgcn.raw.ptr.buffer.store.v3f32(<3 x float> %1, ptr addrspace(8) %0, i32 42, i32 0, i32 0)
  ret void
}

;CHECK-LABEL: {{^}}struct_ptr_buffer_store_immoffs_x3:
;CHECK-NOT: s_waitcnt
;CHECK: buffer_store_dwordx3 v[0:2], {{v[0-9]+}}, s[0:3], 0 idxen offset:42
define amdgpu_ps void @struct_ptr_buffer_store_immoffs_x3(ptr addrspace(8) inreg, <3 x float>) {
main_body:
  call void @llvm.amdgcn.struct.ptr.buffer.store.v3f32(<3 x float> %1, ptr addrspace(8) %0, i32 0, i32 42, i32 0, i32 0)
  ret void
}

declare void @llvm.amdgcn.buffer.store.v3f32(<3 x float>, <4 x i32>, i32, i32, i1, i1) #0
declare void @llvm.amdgcn.buffer.store.format.v3f32(<3 x float>, <4 x i32>, i32, i32, i1, i1) #0
declare void @llvm.amdgcn.raw.buffer.store.format.v3f32(<3 x float>, <4 x i32>, i32, i32, i32) #0
declare void @llvm.amdgcn.raw.buffer.store.v3f32(<3 x float>, <4 x i32>, i32, i32, i32) #0
declare void @llvm.amdgcn.struct.buffer.store.format.v3f32(<3 x float>, <4 x i32>, i32, i32, i32, i32) #0
declare void @llvm.amdgcn.struct.buffer.store.v3f32(<3 x float>, <4 x i32>, i32, i32, i32, i32) #0
declare void @llvm.amdgcn.raw.ptr.buffer.store.format.v3f32(<3 x float>, ptr addrspace(8), i32, i32, i32) #0
declare void @llvm.amdgcn.raw.ptr.buffer.store.v3f32(<3 x float>, ptr addrspace(8), i32, i32, i32) #0
declare void @llvm.amdgcn.struct.ptr.buffer.store.format.v3f32(<3 x float>, ptr addrspace(8), i32, i32, i32, i32) #0
declare void @llvm.amdgcn.struct.ptr.buffer.store.v3f32(<3 x float>, ptr addrspace(8), i32, i32, i32, i32) #0
