;RUN: llc < %s -march=amdgcn -mcpu=gfx600 -verify-machineinstrs | FileCheck %s -check-prefixes=CHECK,SI
;RUN: llc < %s -march=amdgcn -mcpu=gfx700 -verify-machineinstrs | FileCheck %s -check-prefixes=CHECK,GCNX3

;CHECK-LABEL: {{^}}buffer_load_format_immoffs_x3:
;SI: buffer_load_format_xyzw v[0:3], off, s[0:3], 0 offset:42
;GCNX3: buffer_load_format_xyz v[0:2], off, s[0:3], 0 offset:42
;CHECK: s_waitcnt
define amdgpu_ps <3 x float> @buffer_load_format_immoffs_x3(<4 x i32> inreg) {
main_body:
  %data = call <3 x float> @llvm.amdgcn.buffer.load.format.v3f32(<4 x i32> %0, i32 0, i32 42, i1 0, i1 0)
  ret <3 x float> %data
}

;CHECK-LABEL: {{^}}buffer_load_immoffs_x3:
;SI: buffer_load_dwordx4 v[0:3], off, s[0:3], 0 offset:40
;GCNX3: buffer_load_dwordx3 v[0:2], off, s[0:3], 0 offset:40
;CHECK: s_waitcnt
define amdgpu_ps <3 x float> @buffer_load_immoffs_x3(<4 x i32> inreg) {
main_body:
  %data = call <3 x float> @llvm.amdgcn.buffer.load.v3f32(<4 x i32> %0, i32 0, i32 40, i1 0, i1 0)
  ret <3 x float> %data
}

;CHECK-LABEL: {{^}}buffer_raw_load_immoffs_x3:
;SI: buffer_load_dwordx4 v[0:3], off, s[0:3], 0 offset:40
;GCNX3: buffer_load_dwordx3 v[0:2], off, s[0:3], 0 offset:40
;CHECK: s_waitcnt
define amdgpu_ps <3 x float> @buffer_raw_load_immoffs_x3(<4 x i32> inreg) {
main_body:
  %data = call <3 x float> @llvm.amdgcn.raw.buffer.load.v3f32(<4 x i32> %0, i32 40, i32 0, i32 0)
  ret <3 x float> %data
}

;CHECK-LABEL: {{^}}buffer_struct_load_format_immoffs_x3:
;SI: buffer_load_format_xyzw v[0:3], {{v[0-9]+}}, s[0:3], 0 idxen offset:42
;GCNX3: buffer_load_format_xyz v[0:2], {{v[0-9]+}}, s[0:3], 0 idxen offset:42
;CHECK: s_waitcnt
define amdgpu_ps <3 x float> @buffer_struct_load_format_immoffs_x3(<4 x i32> inreg) {
main_body:
  %data = call <3 x float> @llvm.amdgcn.struct.buffer.load.format.v3f32(<4 x i32> %0, i32 0, i32 42, i32 0, i32 0)
  ret <3 x float> %data
}

;CHECK-LABEL: {{^}}struct_buffer_load_immoffs_x3:
;SI: buffer_load_dwordx4 v[0:3], {{v[0-9]+}}, s[0:3], 0 idxen offset:40
;GCNX3: buffer_load_dwordx3 v[0:2], {{v[0-9]+}}, s[0:3], 0 idxen offset:40
;CHECK: s_waitcnt
define amdgpu_ps <3 x float> @struct_buffer_load_immoffs_x3(<4 x i32> inreg) {
main_body:
  %data = call <3 x float> @llvm.amdgcn.struct.buffer.load.v3f32(<4 x i32> %0, i32 0, i32 40, i32 0, i32 0)
  ret <3 x float> %data
}

declare <3 x float> @llvm.amdgcn.buffer.load.format.v3f32(<4 x i32>, i32, i32, i1, i1) #0
declare <3 x float> @llvm.amdgcn.buffer.load.v3f32(<4 x i32>, i32, i32, i1, i1) #0
declare <3 x float> @llvm.amdgcn.raw.buffer.load.format.v3f32(<4 x i32>, i32, i32, i32) #0
declare <3 x float> @llvm.amdgcn.raw.buffer.load.v3f32(<4 x i32>, i32, i32, i32) #0
declare <3 x float> @llvm.amdgcn.struct.buffer.load.format.v3f32(<4 x i32>, i32, i32, i32, i32) #0
declare <3 x float> @llvm.amdgcn.struct.buffer.load.v3f32(<4 x i32>, i32, i32, i32, i32) #0

