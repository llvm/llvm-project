;RUN: llc < %s -march=amdgcn -mcpu=verde -verify-machineinstrs | FileCheck %s
;RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck %s

;CHECK-LABEL: {{^}}buffer_load:
;CHECK: buffer_load_format_xyzw v[0:3], off, s[0:3], 0
;CHECK: buffer_load_format_xyzw v[4:7], off, s[0:3], 0 glc
;CHECK: buffer_load_format_xyzw v[8:11], off, s[0:3], 0 slc
;CHECK: s_waitcnt
define amdgpu_ps {<4 x float>, <4 x float>, <4 x float>} @buffer_load(ptr addrspace(8) inreg) {
main_body:
  %data = call <4 x float> @llvm.amdgcn.raw.ptr.buffer.load.format.v4f32(ptr addrspace(8) %0, i32 0, i32 0, i32 0)
  %data_glc = call <4 x float> @llvm.amdgcn.raw.ptr.buffer.load.format.v4f32(ptr addrspace(8) %0, i32 0, i32 0, i32 1)
  %data_slc = call <4 x float> @llvm.amdgcn.raw.ptr.buffer.load.format.v4f32(ptr addrspace(8) %0, i32 0, i32 0, i32 2)
  %r0 = insertvalue {<4 x float>, <4 x float>, <4 x float>} undef, <4 x float> %data, 0
  %r1 = insertvalue {<4 x float>, <4 x float>, <4 x float>} %r0, <4 x float> %data_glc, 1
  %r2 = insertvalue {<4 x float>, <4 x float>, <4 x float>} %r1, <4 x float> %data_slc, 2
  ret {<4 x float>, <4 x float>, <4 x float>} %r2
}

;CHECK-LABEL: {{^}}buffer_load_immoffs:
;CHECK: buffer_load_format_xyzw v[0:3], off, s[0:3], 0 offset:42
;CHECK: s_waitcnt
define amdgpu_ps <4 x float> @buffer_load_immoffs(ptr addrspace(8) inreg) {
main_body:
  %data = call <4 x float> @llvm.amdgcn.raw.ptr.buffer.load.format.v4f32(ptr addrspace(8) %0, i32 42, i32 0, i32 0)
  ret <4 x float> %data
}

;CHECK-LABEL: {{^}}buffer_load_immoffs_large:
;CHECK-DAG: buffer_load_format_xyzw {{v\[[0-9]+:[0-9]+\]}}, off, s[0:3], 60 offset:4092
;CHECK-DAG: s_movk_i32 [[OFS1:s[0-9]+]], 0x7ffc
;CHECK-DAG: buffer_load_format_xyzw {{v\[[0-9]+:[0-9]+\]}}, off, s[0:3], [[OFS1]] offset:4092
;CHECK-DAG: s_mov_b32 [[OFS2:s[0-9]+]], 0x8ffc
;CHECK-DAG: buffer_load_format_xyzw {{v\[[0-9]+:[0-9]+\]}}, off, s[0:3], [[OFS2]] offset:4
;CHECK: s_waitcnt
define amdgpu_ps <4 x float> @buffer_load_immoffs_large(ptr addrspace(8) inreg) {
main_body:
  %d.0 = call <4 x float> @llvm.amdgcn.raw.ptr.buffer.load.format.v4f32(ptr addrspace(8) %0, i32 4092, i32 60, i32 0)
  %d.1 = call <4 x float> @llvm.amdgcn.raw.ptr.buffer.load.format.v4f32(ptr addrspace(8) %0, i32 4092, i32 32764, i32 0)
  %d.2 = call <4 x float> @llvm.amdgcn.raw.ptr.buffer.load.format.v4f32(ptr addrspace(8) %0, i32 4, i32 36860, i32 0)
  %d.3 = fadd <4 x float> %d.0, %d.1
  %data = fadd <4 x float> %d.2, %d.3
  ret <4 x float> %data
}

;CHECK-LABEL: {{^}}buffer_load_ofs:
;CHECK: buffer_load_format_xyzw v[0:3], v0, s[0:3], 0 offen
;CHECK: s_waitcnt
define amdgpu_ps <4 x float> @buffer_load_ofs(ptr addrspace(8) inreg, i32) {
main_body:
  %data = call <4 x float> @llvm.amdgcn.raw.ptr.buffer.load.format.v4f32(ptr addrspace(8) %0, i32 %1, i32 0, i32 0)
  ret <4 x float> %data
}

;CHECK-LABEL: {{^}}buffer_load_ofs_imm:
;CHECK: buffer_load_format_xyzw v[0:3], v0, s[0:3], 0 offen offset:60
;CHECK: s_waitcnt
define amdgpu_ps <4 x float> @buffer_load_ofs_imm(ptr addrspace(8) inreg, i32) {
main_body:
  %ofs = add i32 %1, 60
  %data = call <4 x float> @llvm.amdgcn.raw.ptr.buffer.load.format.v4f32(ptr addrspace(8) %0, i32 %ofs, i32 0, i32 0)
  ret <4 x float> %data
}

;CHECK-LABEL: {{^}}buffer_load_x:
;CHECK: buffer_load_format_x v0, off, s[0:3], 0
;CHECK: s_waitcnt
define amdgpu_ps float @buffer_load_x(ptr addrspace(8) inreg %rsrc) {
main_body:
  %data = call float @llvm.amdgcn.raw.ptr.buffer.load.format.f32(ptr addrspace(8) %rsrc, i32 0, i32 0, i32 0)
  ret float %data
}

;CHECK-LABEL: {{^}}buffer_load_xy:
;CHECK: buffer_load_format_xy v[0:1], off, s[0:3], 0
;CHECK: s_waitcnt
define amdgpu_ps <2 x float> @buffer_load_xy(ptr addrspace(8) inreg %rsrc) {
main_body:
  %data = call <2 x float> @llvm.amdgcn.raw.ptr.buffer.load.format.v2f32(ptr addrspace(8) %rsrc, i32 0, i32 0, i32 0)
  ret <2 x float> %data
}

declare float @llvm.amdgcn.raw.ptr.buffer.load.format.f32(ptr addrspace(8), i32, i32, i32) #0
declare <2 x float> @llvm.amdgcn.raw.ptr.buffer.load.format.v2f32(ptr addrspace(8), i32, i32, i32) #0
declare <4 x float> @llvm.amdgcn.raw.ptr.buffer.load.format.v4f32(ptr addrspace(8), i32, i32, i32) #0

attributes #0 = { nounwind readonly }
