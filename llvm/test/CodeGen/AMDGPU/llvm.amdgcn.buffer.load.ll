;RUN: llc < %s -march=amdgcn -mcpu=verde -verify-machineinstrs | FileCheck %s -check-prefix=CHECK -check-prefix=SICI
;RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck %s -check-prefix=CHECK -check-prefix=VI

;CHECK-LABEL: {{^}}buffer_load:
;CHECK: buffer_load_dwordx4 v[0:3], off, s[0:3], 0
;CHECK: buffer_load_dwordx4 v[4:7], off, s[0:3], 0 glc
;CHECK: buffer_load_dwordx4 v[8:11], off, s[0:3], 0 slc
;CHECK: s_waitcnt
define amdgpu_ps {<4 x float>, <4 x float>, <4 x float>} @buffer_load(<4 x i32> inreg) {
main_body:
  %data = call <4 x float> @llvm.amdgcn.buffer.load.v4f32(<4 x i32> %0, i32 0, i32 0, i1 0, i1 0)
  %data_glc = call <4 x float> @llvm.amdgcn.buffer.load.v4f32(<4 x i32> %0, i32 0, i32 0, i1 1, i1 0)
  %data_slc = call <4 x float> @llvm.amdgcn.buffer.load.v4f32(<4 x i32> %0, i32 0, i32 0, i1 0, i1 1)
  %r0 = insertvalue {<4 x float>, <4 x float>, <4 x float>} undef, <4 x float> %data, 0
  %r1 = insertvalue {<4 x float>, <4 x float>, <4 x float>} %r0, <4 x float> %data_glc, 1
  %r2 = insertvalue {<4 x float>, <4 x float>, <4 x float>} %r1, <4 x float> %data_slc, 2
  ret {<4 x float>, <4 x float>, <4 x float>} %r2
}

;CHECK-LABEL: {{^}}buffer_load_immoffs:
;CHECK: buffer_load_dwordx4 v[0:3], off, s[0:3], 0 offset:42
;CHECK: s_waitcnt
define amdgpu_ps <4 x float> @buffer_load_immoffs(<4 x i32> inreg) {
main_body:
  %data = call <4 x float> @llvm.amdgcn.buffer.load.v4f32(<4 x i32> %0, i32 0, i32 42, i1 0, i1 0)
  ret <4 x float> %data
}

;CHECK-LABEL: {{^}}buffer_load_immoffs_large:
;SICI: buffer_load_dwordx4 v[0:3], {{v[0-9]+}}, s[0:3], 0 offen
;VI: s_movk_i32 [[OFFSET:s[0-9]+]], 0x1fff
;VI: buffer_load_dwordx4 v[0:3], off, s[0:3], [[OFFSET]] offset:1
;CHECK: s_waitcnt
define amdgpu_ps <4 x float> @buffer_load_immoffs_large(<4 x i32> inreg) {
main_body:
  %data = call <4 x float> @llvm.amdgcn.buffer.load.v4f32(<4 x i32> %0, i32 0, i32 8192, i1 0, i1 0)
  ret <4 x float> %data
}

;CHECK-LABEL: {{^}}buffer_load_idx:
;CHECK: buffer_load_dwordx4 v[0:3], v0, s[0:3], 0 idxen
;CHECK: s_waitcnt
define amdgpu_ps <4 x float> @buffer_load_idx(<4 x i32> inreg, i32) {
main_body:
  %data = call <4 x float> @llvm.amdgcn.buffer.load.v4f32(<4 x i32> %0, i32 %1, i32 0, i1 0, i1 0)
  ret <4 x float> %data
}

;CHECK-LABEL: {{^}}buffer_load_ofs:
;CHECK: buffer_load_dwordx4 v[0:3], v0, s[0:3], 0 offen
;CHECK: s_waitcnt
define amdgpu_ps <4 x float> @buffer_load_ofs(<4 x i32> inreg, i32) {
main_body:
  %data = call <4 x float> @llvm.amdgcn.buffer.load.v4f32(<4 x i32> %0, i32 0, i32 %1, i1 0, i1 0)
  ret <4 x float> %data
}

;CHECK-LABEL: {{^}}buffer_load_ofs_imm:
;CHECK: buffer_load_dwordx4 v[0:3], v0, s[0:3], 0 offen offset:58
;CHECK: s_waitcnt
define amdgpu_ps <4 x float> @buffer_load_ofs_imm(<4 x i32> inreg, i32) {
main_body:
  %ofs = add i32 %1, 58
  %data = call <4 x float> @llvm.amdgcn.buffer.load.v4f32(<4 x i32> %0, i32 0, i32 %ofs, i1 0, i1 0)
  ret <4 x float> %data
}

;CHECK-LABEL: {{^}}buffer_load_both:
;CHECK: buffer_load_dwordx4 v[0:3], v[0:1], s[0:3], 0 idxen offen
;CHECK: s_waitcnt
define amdgpu_ps <4 x float> @buffer_load_both(<4 x i32> inreg, i32, i32) {
main_body:
  %data = call <4 x float> @llvm.amdgcn.buffer.load.v4f32(<4 x i32> %0, i32 %1, i32 %2, i1 0, i1 0)
  ret <4 x float> %data
}

;CHECK-LABEL: {{^}}buffer_load_both_reversed:
;CHECK: v_mov_b32_e32 v2, v0
;CHECK: buffer_load_dwordx4 v[0:3], v[1:2], s[0:3], 0 idxen offen
;CHECK: s_waitcnt
define amdgpu_ps <4 x float> @buffer_load_both_reversed(<4 x i32> inreg, i32, i32) {
main_body:
  %data = call <4 x float> @llvm.amdgcn.buffer.load.v4f32(<4 x i32> %0, i32 %2, i32 %1, i1 0, i1 0)
  ret <4 x float> %data
}

;CHECK-LABEL: {{^}}buffer_load_x1:
;CHECK: buffer_load_dword v0, v[0:1], s[0:3], 0 idxen offen
;CHECK: s_waitcnt
define amdgpu_ps float @buffer_load_x1(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs) {
main_body:
  %data = call float @llvm.amdgcn.buffer.load.f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i1 0, i1 0)
  ret float %data
}

;CHECK-LABEL: {{^}}buffer_load_x2:
;CHECK: buffer_load_dwordx2 v[0:1], v[0:1], s[0:3], 0 idxen offen
;CHECK: s_waitcnt
define amdgpu_ps <2 x float> @buffer_load_x2(<4 x i32> inreg %rsrc, i32 %idx, i32 %ofs) {
main_body:
  %data = call <2 x float> @llvm.amdgcn.buffer.load.v2f32(<4 x i32> %rsrc, i32 %idx, i32 %ofs, i1 0, i1 0)
  ret <2 x float> %data
}

;CHECK-LABEL: {{^}}buffer_load_negative_offset:
;CHECK: v_add_i32_e32 [[VOFS:v[0-9]+]], vcc, -16, v0
;CHECK: buffer_load_dwordx4 v[0:3], [[VOFS]], s[0:3], 0 offen
define amdgpu_ps <4 x float> @buffer_load_negative_offset(<4 x i32> inreg, i32 %ofs) {
main_body:
  %ofs.1 = add i32 %ofs, -16
  %data = call <4 x float> @llvm.amdgcn.buffer.load.v4f32(<4 x i32> %0, i32 0, i32 %ofs.1, i1 0, i1 0)
  ret <4 x float> %data
}

declare float @llvm.amdgcn.buffer.load.f32(<4 x i32>, i32, i32, i1, i1) #0
declare <2 x float> @llvm.amdgcn.buffer.load.v2f32(<4 x i32>, i32, i32, i1, i1) #0
declare <4 x float> @llvm.amdgcn.buffer.load.v4f32(<4 x i32>, i32, i32, i1, i1) #0

attributes #0 = { nounwind readonly }
