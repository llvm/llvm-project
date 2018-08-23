;RUN: llc < %s -march=amdgcn -mcpu=verde -verify-machineinstrs | FileCheck %s -check-prefix=CHECK -check-prefix=SICI
;RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck %s -check-prefix=CHECK -check-prefix=VI

;CHECK-LABEL: {{^}}buffer_load:
;CHECK: buffer_load_format_xyzw v[0:3], {{v[0-9]+}}, s[0:3], 0 idxen
;CHECK: buffer_load_format_xyzw v[4:7], {{v[0-9]+}}, s[0:3], 0 idxen glc
;CHECK: buffer_load_format_xyzw v[8:11], {{v[0-9]+}}, s[0:3], 0 idxen slc
;CHECK: s_waitcnt
define amdgpu_ps {<4 x float>, <4 x float>, <4 x float>} @buffer_load(<4 x i32> inreg) {
main_body:
  %data = call <4 x float> @llvm.amdgcn.struct.buffer.load.format.v4f32(<4 x i32> %0, i32 0, i32 0, i32 0, i32 0)
  %data_glc = call <4 x float> @llvm.amdgcn.struct.buffer.load.format.v4f32(<4 x i32> %0, i32 0, i32 0, i32 0, i32 1)
  %data_slc = call <4 x float> @llvm.amdgcn.struct.buffer.load.format.v4f32(<4 x i32> %0, i32 0, i32 0, i32 0, i32 2)
  %r0 = insertvalue {<4 x float>, <4 x float>, <4 x float>} undef, <4 x float> %data, 0
  %r1 = insertvalue {<4 x float>, <4 x float>, <4 x float>} %r0, <4 x float> %data_glc, 1
  %r2 = insertvalue {<4 x float>, <4 x float>, <4 x float>} %r1, <4 x float> %data_slc, 2
  ret {<4 x float>, <4 x float>, <4 x float>} %r2
}

;CHECK-LABEL: {{^}}buffer_load_immoffs:
;CHECK: buffer_load_format_xyzw v[0:3], {{v[0-9]+}}, s[0:3], 0 idxen offset:42
;CHECK: s_waitcnt
define amdgpu_ps <4 x float> @buffer_load_immoffs(<4 x i32> inreg) {
main_body:
  %data = call <4 x float> @llvm.amdgcn.struct.buffer.load.format.v4f32(<4 x i32> %0, i32 0, i32 42, i32 0, i32 0)
  ret <4 x float> %data
}

;CHECK-LABEL: {{^}}buffer_load_immoffs_large:
;CHECK-DAG: buffer_load_format_xyzw {{v\[[0-9]+:[0-9]+\]}}, {{v[0-9]+}}, s[0:3], 60 idxen offset:4092
;CHECK-DAG: s_movk_i32 [[OFS1:s[0-9]+]], 0x7ffc
;CHECK-DAG: buffer_load_format_xyzw {{v\[[0-9]+:[0-9]+\]}}, {{v[0-9]+}}, s[0:3], [[OFS1]] idxen offset:4092
;CHECK-DAG: s_mov_b32 [[OFS2:s[0-9]+]], 0x8ffc
;CHECK-DAG: buffer_load_format_xyzw {{v\[[0-9]+:[0-9]+\]}}, {{v[0-9]+}}, s[0:3], [[OFS2]] idxen offset:4
;CHECK: s_waitcnt
define amdgpu_ps <4 x float> @buffer_load_immoffs_large(<4 x i32> inreg) {
main_body:
  %d.0 = call <4 x float> @llvm.amdgcn.struct.buffer.load.format.v4f32(<4 x i32> %0, i32 0, i32 4092, i32 60, i32 0)
  %d.1 = call <4 x float> @llvm.amdgcn.struct.buffer.load.format.v4f32(<4 x i32> %0, i32 0, i32 4092, i32 32764, i32 0)
  %d.2 = call <4 x float> @llvm.amdgcn.struct.buffer.load.format.v4f32(<4 x i32> %0, i32 0, i32 4, i32 36860, i32 0)
  %d.3 = fadd <4 x float> %d.0, %d.1
  %data = fadd <4 x float> %d.2, %d.3
  ret <4 x float> %data
}

;CHECK-LABEL: {{^}}buffer_load_idx:
;CHECK: buffer_load_format_xyzw v[0:3], v0, s[0:3], 0 idxen
;CHECK: s_waitcnt
define amdgpu_ps <4 x float> @buffer_load_idx(<4 x i32> inreg, i32) {
main_body:
  %data = call <4 x float> @llvm.amdgcn.struct.buffer.load.format.v4f32(<4 x i32> %0, i32 %1, i32 0, i32 0, i32 0)
  ret <4 x float> %data
}

;CHECK-LABEL: {{^}}buffer_load_ofs:
;CHECK: buffer_load_format_xyzw v[0:3], v[0:1], s[0:3], 0 idxen offen
;CHECK: s_waitcnt
define amdgpu_ps <4 x float> @buffer_load_ofs(<4 x i32> inreg, i32) {
main_body:
  %data = call <4 x float> @llvm.amdgcn.struct.buffer.load.format.v4f32(<4 x i32> %0, i32 0, i32 %1, i32 0, i32 0)
  ret <4 x float> %data
}

;CHECK-LABEL: {{^}}buffer_load_ofs_imm:
;CHECK: buffer_load_format_xyzw v[0:3], v[0:1], s[0:3], 0 idxen offen offset:60
;CHECK: s_waitcnt
define amdgpu_ps <4 x float> @buffer_load_ofs_imm(<4 x i32> inreg, i32) {
main_body:
  %ofs = add i32 %1, 60
  %data = call <4 x float> @llvm.amdgcn.struct.buffer.load.format.v4f32(<4 x i32> %0, i32 0, i32 %ofs, i32 0, i32 0)
  ret <4 x float> %data
}

;CHECK-LABEL: {{^}}buffer_load_both:
;CHECK: buffer_load_format_xyzw v[0:3], v[0:1], s[0:3], 0 idxen offen
;CHECK: s_waitcnt
define amdgpu_ps <4 x float> @buffer_load_both(<4 x i32> inreg, i32, i32) {
main_body:
  %data = call <4 x float> @llvm.amdgcn.struct.buffer.load.format.v4f32(<4 x i32> %0, i32 %1, i32 %2, i32 0, i32 0)
  ret <4 x float> %data
}

;CHECK-LABEL: {{^}}buffer_load_both_reversed:
;CHECK: v_mov_b32_e32 v2, v0
;CHECK: buffer_load_format_xyzw v[0:3], v[1:2], s[0:3], 0 idxen offen
;CHECK: s_waitcnt
define amdgpu_ps <4 x float> @buffer_load_both_reversed(<4 x i32> inreg, i32, i32) {
main_body:
  %data = call <4 x float> @llvm.amdgcn.struct.buffer.load.format.v4f32(<4 x i32> %0, i32 %2, i32 %1, i32 0, i32 0)
  ret <4 x float> %data
}

;CHECK-LABEL: {{^}}buffer_load_x:
;CHECK: buffer_load_format_x v0, {{v[0-9]+}}, s[0:3], 0 idxen
;CHECK: s_waitcnt
define amdgpu_ps float @buffer_load_x(<4 x i32> inreg %rsrc) {
main_body:
  %data = call float @llvm.amdgcn.struct.buffer.load.format.f32(<4 x i32> %rsrc, i32 0, i32 0, i32 0, i32 0)
  ret float %data
}

;CHECK-LABEL: {{^}}buffer_load_xy:
;CHECK: buffer_load_format_xy v[0:1], {{v[0-9]+}}, s[0:3], 0 idxen
;CHECK: s_waitcnt
define amdgpu_ps <2 x float> @buffer_load_xy(<4 x i32> inreg %rsrc) {
main_body:
  %data = call <2 x float> @llvm.amdgcn.struct.buffer.load.format.v2f32(<4 x i32> %rsrc, i32 0, i32 0, i32 0, i32 0)
  ret <2 x float> %data
}

declare float @llvm.amdgcn.struct.buffer.load.format.f32(<4 x i32>, i32, i32, i32, i32) #0
declare <2 x float> @llvm.amdgcn.struct.buffer.load.format.v2f32(<4 x i32>, i32, i32, i32, i32) #0
declare <4 x float> @llvm.amdgcn.struct.buffer.load.format.v4f32(<4 x i32>, i32, i32, i32, i32) #0

attributes #0 = { nounwind readonly }
