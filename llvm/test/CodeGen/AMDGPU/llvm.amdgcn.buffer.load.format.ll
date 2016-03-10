;RUN: llc < %s -march=amdgcn -mcpu=verde -verify-machineinstrs | FileCheck %s
;RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck %s

;CHECK-LABEL: {{^}}buffer_load:
;CHECK: buffer_load_format_xyzw v[0:3], s[0:3], s4
;CHECK: buffer_load_format_xyzw v[4:7], s[0:3], s4 glc
;CHECK: buffer_load_format_xyzw v[8:11], s[0:3], s4 slc
;CHECK: s_waitcnt
define {<4 x float>, <4 x float>, <4 x float>} @buffer_load(<4 x i32> inreg, i32 inreg) #0 {
main_body:
  %data = call <4 x float> @llvm.amdgcn.buffer.load.format(<4 x i32> %0, i32 %1, i32 0, i32 0, i32 0, i1 0, i1 0)
  %data_glc = call <4 x float> @llvm.amdgcn.buffer.load.format(<4 x i32> %0, i32 %1, i32 0, i32 0, i32 0, i1 1, i1 0)
  %data_slc = call <4 x float> @llvm.amdgcn.buffer.load.format(<4 x i32> %0, i32 %1, i32 0, i32 0, i32 0, i1 0, i1 1)
  %r0 = insertvalue {<4 x float>, <4 x float>, <4 x float>} undef, <4 x float> %data, 0
  %r1 = insertvalue {<4 x float>, <4 x float>, <4 x float>} %r0, <4 x float> %data_glc, 1
  %r2 = insertvalue {<4 x float>, <4 x float>, <4 x float>} %r1, <4 x float> %data_slc, 2
  ret {<4 x float>, <4 x float>, <4 x float>} %r2
}

;CHECK-LABEL: {{^}}buffer_load_immoffs:
;CHECK: buffer_load_format_xyzw v[0:3], s[0:3], s4 offset:42
;CHECK: s_waitcnt
define <4 x float> @buffer_load_immoffs(<4 x i32> inreg, i32 inreg) #0 {
main_body:
  %data = call <4 x float> @llvm.amdgcn.buffer.load.format(<4 x i32> %0, i32 %1, i32 42, i32 0, i32 0, i1 0, i1 0)
  ret <4 x float> %data
}

;CHECK-LABEL: {{^}}buffer_load_idx:
;CHECK: buffer_load_format_xyzw v[0:3], v0, s[0:3], 0 idxen
;CHECK: s_waitcnt
define <4 x float> @buffer_load_idx(<4 x i32> inreg, i32) #0 {
main_body:
  %data = call <4 x float> @llvm.amdgcn.buffer.load.format(<4 x i32> %0, i32 0, i32 0, i32 %1, i32 0, i1 0, i1 0)
  ret <4 x float> %data
}

;CHECK-LABEL: {{^}}buffer_load_ofs:
;CHECK: buffer_load_format_xyzw v[0:3], v0, s[0:3], 0 offen
;CHECK: s_waitcnt
define <4 x float> @buffer_load_ofs(<4 x i32> inreg, i32) #0 {
main_body:
  %data = call <4 x float> @llvm.amdgcn.buffer.load.format(<4 x i32> %0, i32 0, i32 0, i32 0, i32 %1, i1 0, i1 0)
  ret <4 x float> %data
}

;CHECK-LABEL: {{^}}buffer_load_both:
;CHECK: buffer_load_format_xyzw v[0:3], v[0:1], s[0:3], 0 idxen offen
;CHECK: s_waitcnt
define <4 x float> @buffer_load_both(<4 x i32> inreg, i32, i32) #0 {
main_body:
  %data = call <4 x float> @llvm.amdgcn.buffer.load.format(<4 x i32> %0, i32 0, i32 0, i32 %1, i32 %2, i1 0, i1 0)
  ret <4 x float> %data
}

;CHECK-LABEL: {{^}}buffer_load_both_reversed:
;CHECK: v_mov_b32_e32 v2, v0
;CHECK: buffer_load_format_xyzw v[0:3], v[1:2], s[0:3], 0 idxen offen
;CHECK: s_waitcnt
define <4 x float> @buffer_load_both_reversed(<4 x i32> inreg, i32, i32) #0 {
main_body:
  %data = call <4 x float> @llvm.amdgcn.buffer.load.format(<4 x i32> %0, i32 0, i32 0, i32 %2, i32 %1, i1 0, i1 0)
  ret <4 x float> %data
}

declare <4 x float> @llvm.amdgcn.buffer.load.format(<4 x i32>, i32, i32, i32, i32, i1, i1) #1

attributes #0 = { "ShaderType"="0" }
attributes #1 = { nounwind readonly }
