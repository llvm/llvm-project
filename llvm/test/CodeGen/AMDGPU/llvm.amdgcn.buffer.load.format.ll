;RUN: llc < %s -march=amdgcn -mcpu=verde -verify-machineinstrs | FileCheck %s
;RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck %s

;CHECK-LABEL: {{^}}buffer_load:
;CHECK: buffer_load_format_xyzw v[0:3], s[0:3], 0
;CHECK: buffer_load_format_xyzw v[4:7], s[0:3], 0 glc
;CHECK: buffer_load_format_xyzw v[8:11], s[0:3], 0 slc
;CHECK: s_waitcnt
define {<4 x float>, <4 x float>, <4 x float>} @buffer_load(<4 x i32> inreg) #0 {
main_body:
  %data = call <4 x float> @llvm.amdgcn.buffer.load.format.v4f32(<4 x i32> %0, i32 0, i32 0, i1 0, i1 0)
  %data_glc = call <4 x float> @llvm.amdgcn.buffer.load.format.v4f32(<4 x i32> %0, i32 0, i32 0, i1 1, i1 0)
  %data_slc = call <4 x float> @llvm.amdgcn.buffer.load.format.v4f32(<4 x i32> %0, i32 0, i32 0, i1 0, i1 1)
  %r0 = insertvalue {<4 x float>, <4 x float>, <4 x float>} undef, <4 x float> %data, 0
  %r1 = insertvalue {<4 x float>, <4 x float>, <4 x float>} %r0, <4 x float> %data_glc, 1
  %r2 = insertvalue {<4 x float>, <4 x float>, <4 x float>} %r1, <4 x float> %data_slc, 2
  ret {<4 x float>, <4 x float>, <4 x float>} %r2
}

;CHECK-LABEL: {{^}}buffer_load_immoffs:
;CHECK: buffer_load_format_xyzw v[0:3], s[0:3], 0 offset:42
;CHECK: s_waitcnt
define <4 x float> @buffer_load_immoffs(<4 x i32> inreg) #0 {
main_body:
  %data = call <4 x float> @llvm.amdgcn.buffer.load.format.v4f32(<4 x i32> %0, i32 0, i32 42, i1 0, i1 0)
  ret <4 x float> %data
}

;CHECK-LABEL: {{^}}buffer_load_immoffs_large:
;CHECK-DAG: buffer_load_format_xyzw {{v\[[0-9]+:[0-9]+\]}}, s[0:3], 61 offset:4095
;CHECK-DAG: s_movk_i32 [[OFS1:s[0-9]+]], 0x7fff
;CHECK: buffer_load_format_xyzw {{v\[[0-9]+:[0-9]+\]}}, s[0:3], [[OFS1]] offset:4093
;CHECK: s_mov_b32 [[OFS2:s[0-9]+]], 0x8fff
;CHECK: buffer_load_format_xyzw {{v\[[0-9]+:[0-9]+\]}}, s[0:3], [[OFS2]] offset:1
;CHECK: s_waitcnt
define <4 x float> @buffer_load_immoffs_large(<4 x i32> inreg) #0 {
main_body:
  %d.0 = call <4 x float> @llvm.amdgcn.buffer.load.format.v4f32(<4 x i32> %0, i32 0, i32 4156, i1 0, i1 0)
  %d.1 = call <4 x float> @llvm.amdgcn.buffer.load.format.v4f32(<4 x i32> %0, i32 0, i32 36860, i1 0, i1 0)
  %d.2 = call <4 x float> @llvm.amdgcn.buffer.load.format.v4f32(<4 x i32> %0, i32 0, i32 36864, i1 0, i1 0)
  %d.3 = fadd <4 x float> %d.0, %d.1
  %data = fadd <4 x float> %d.2, %d.3
  ret <4 x float> %data
}

;CHECK-LABEL: {{^}}buffer_load_immoffs_reuse:
;CHECK: s_movk_i32 [[OFS:s[0-9]+]], 0xfff
;CHECK: buffer_load_format_xyzw {{v\[[0-9]+:[0-9]+\]}}, s[0:3], [[OFS]] offset:65
;CHECK-NOT: s_mov
;CHECK: buffer_load_format_xyzw {{v\[[0-9]+:[0-9]+\]}}, s[0:3], [[OFS]] offset:81
;CHECK: s_waitcnt
define <4 x float> @buffer_load_immoffs_reuse(<4 x i32> inreg) #0 {
main_body:
  %d.0 = call <4 x float> @llvm.amdgcn.buffer.load.format.v4f32(<4 x i32> %0, i32 0, i32 4160, i1 0, i1 0)
  %d.1 = call <4 x float> @llvm.amdgcn.buffer.load.format.v4f32(<4 x i32> %0, i32 0, i32 4176, i1 0, i1 0)
  %data = fadd <4 x float> %d.0, %d.1
  ret <4 x float> %data
}

;CHECK-LABEL: {{^}}buffer_load_idx:
;CHECK: buffer_load_format_xyzw v[0:3], v0, s[0:3], 0 idxen
;CHECK: s_waitcnt
define <4 x float> @buffer_load_idx(<4 x i32> inreg, i32) #0 {
main_body:
  %data = call <4 x float> @llvm.amdgcn.buffer.load.format.v4f32(<4 x i32> %0, i32 %1, i32 0, i1 0, i1 0)
  ret <4 x float> %data
}

;CHECK-LABEL: {{^}}buffer_load_ofs:
;CHECK: buffer_load_format_xyzw v[0:3], v0, s[0:3], 0 offen
;CHECK: s_waitcnt
define <4 x float> @buffer_load_ofs(<4 x i32> inreg, i32) #0 {
main_body:
  %data = call <4 x float> @llvm.amdgcn.buffer.load.format.v4f32(<4 x i32> %0, i32 0, i32 %1, i1 0, i1 0)
  ret <4 x float> %data
}

;CHECK-LABEL: {{^}}buffer_load_ofs_imm:
;CHECK: buffer_load_format_xyzw v[0:3], v0, s[0:3], 0 offen offset:58
;CHECK: s_waitcnt
define <4 x float> @buffer_load_ofs_imm(<4 x i32> inreg, i32) #0 {
main_body:
  %ofs = add i32 %1, 58
  %data = call <4 x float> @llvm.amdgcn.buffer.load.format.v4f32(<4 x i32> %0, i32 0, i32 %ofs, i1 0, i1 0)
  ret <4 x float> %data
}

;CHECK-LABEL: {{^}}buffer_load_both:
;CHECK: buffer_load_format_xyzw v[0:3], v[0:1], s[0:3], 0 idxen offen
;CHECK: s_waitcnt
define <4 x float> @buffer_load_both(<4 x i32> inreg, i32, i32) #0 {
main_body:
  %data = call <4 x float> @llvm.amdgcn.buffer.load.format.v4f32(<4 x i32> %0, i32 %1, i32 %2, i1 0, i1 0)
  ret <4 x float> %data
}

;CHECK-LABEL: {{^}}buffer_load_both_reversed:
;CHECK: v_mov_b32_e32 v2, v0
;CHECK: buffer_load_format_xyzw v[0:3], v[1:2], s[0:3], 0 idxen offen
;CHECK: s_waitcnt
define <4 x float> @buffer_load_both_reversed(<4 x i32> inreg, i32, i32) #0 {
main_body:
  %data = call <4 x float> @llvm.amdgcn.buffer.load.format.v4f32(<4 x i32> %0, i32 %2, i32 %1, i1 0, i1 0)
  ret <4 x float> %data
}

;CHECK-LABEL: {{^}}buffer_load_x:
;CHECK: buffer_load_format_x v0, s[0:3], 0
;CHECK: s_waitcnt
define float @buffer_load_x(<4 x i32> inreg %rsrc) #0 {
main_body:
  %data = call float @llvm.amdgcn.buffer.load.format.f32(<4 x i32> %rsrc, i32 0, i32 0, i1 0, i1 0)
  ret float %data
}

;CHECK-LABEL: {{^}}buffer_load_xy:
;CHECK: buffer_load_format_xy v[0:1], s[0:3], 0
;CHECK: s_waitcnt
define <2 x float> @buffer_load_xy(<4 x i32> inreg %rsrc) #0 {
main_body:
  %data = call <2 x float> @llvm.amdgcn.buffer.load.format.v2f32(<4 x i32> %rsrc, i32 0, i32 0, i1 0, i1 0)
  ret <2 x float> %data
}

declare float @llvm.amdgcn.buffer.load.format.f32(<4 x i32>, i32, i32, i1, i1) #1
declare <2 x float> @llvm.amdgcn.buffer.load.format.v2f32(<4 x i32>, i32, i32, i1, i1) #1
declare <4 x float> @llvm.amdgcn.buffer.load.format.v4f32(<4 x i32>, i32, i32, i1, i1) #1

attributes #0 = { "ShaderType"="0" }
attributes #1 = { nounwind readonly }
