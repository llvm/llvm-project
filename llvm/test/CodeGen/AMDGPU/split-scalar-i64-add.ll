; RUN: llc -mtriple=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

declare i32 @llvm.amdgcn.workitem.id.x() readnone

; This is broken because the low half of the 64-bit add remains on the
; SALU, but the upper half does not. The addc expects the carry bit
; set in vcc, which is undefined since the low scalar half add sets
; scc instead.

; FIXME: SIShrinkInstructions should force immediate fold.

; FUNC-LABEL: {{^}}imp_def_vcc_split_i64_add_0:
; SI: v_mov_b32_e32 [[V_VAL:v[0-9]+]], s
; SI: v_add_i32_e32 v{{[0-9]+}}, vcc, 0x18f, [[V_VAL]]
; SI: v_addc_u32_e32 v{{[0-9]+}}, vcc, 0, v{{[0-9]+}}, vcc
define amdgpu_kernel void @imp_def_vcc_split_i64_add_0(ptr addrspace(1) %out, ptr addrspace(1) %in, i32 %s.val) {
  %v.val = load volatile i32, ptr addrspace(1) %in
  %vec.0 = insertelement <2 x i32> undef, i32 %s.val, i32 0
  %vec.1 = insertelement <2 x i32> %vec.0, i32 %v.val, i32 1
  %bc = bitcast <2 x i32> %vec.1 to i64
  %add = add i64 %bc, 399
  store i64 %add, ptr addrspace(1) %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}s_imp_def_vcc_split_i64_add_0:
; SI: s_add_u32 {{s[0-9]+}}, {{s[0-9]+}}, 0x18f
; SI: s_addc_u32 {{s[0-9]+}}, 0xf423f, 0
define amdgpu_kernel void @s_imp_def_vcc_split_i64_add_0(ptr addrspace(1) %out, i32 %val) {
  %vec.0 = insertelement <2 x i32> undef, i32 %val, i32 0
  %vec.1 = insertelement <2 x i32> %vec.0, i32 999999, i32 1
  %bc = bitcast <2 x i32> %vec.1 to i64
  %add = add i64 %bc, 399
  store i64 %add, ptr addrspace(1) %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}imp_def_vcc_split_i64_add_1:
; SI: v_add_i32
; SI: v_addc_u32
define amdgpu_kernel void @imp_def_vcc_split_i64_add_1(ptr addrspace(1) %out, ptr addrspace(1) %in, i32 %val0, i64 %val1) {
  %v.val = load volatile i32, ptr addrspace(1) %in
  %vec.0 = insertelement <2 x i32> undef, i32 %val0, i32 0
  %vec.1 = insertelement <2 x i32> %vec.0, i32 %v.val, i32 1
  %bc = bitcast <2 x i32> %vec.1 to i64
  %add = add i64 %bc, %val1
  store i64 %add, ptr addrspace(1) %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}s_imp_def_vcc_split_i64_add_1:
; SI: s_add_u32 {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
; SI: s_addc_u32 {{s[0-9]+}}, 0x1869f, {{s[0-9]+}}
define amdgpu_kernel void @s_imp_def_vcc_split_i64_add_1(ptr addrspace(1) %out, i32 %val0, i64 %val1) {
  %vec.0 = insertelement <2 x i32> undef, i32 %val0, i32 0
  %vec.1 = insertelement <2 x i32> %vec.0, i32 99999, i32 1
  %bc = bitcast <2 x i32> %vec.1 to i64
  %add = add i64 %bc, %val1
  store i64 %add, ptr addrspace(1) %out, align 8
  ret void
}

; Doesn't use constants
; FUNC-LABEL: {{^}}imp_def_vcc_split_i64_add_2:
; SI: v_add_i32_e32 {{v[0-9]+}}, vcc, {{s[0-9]+}}, {{v[0-9]+}}
; SI: v_addc_u32_e32 {{v[0-9]+}}, vcc, {{v[0-9]+}}, {{v[0-9]+}}, vcc
define amdgpu_kernel void @imp_def_vcc_split_i64_add_2(ptr addrspace(1) %out, ptr addrspace(1) %in, i32 %val0, i64 %val1) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() readnone
  %gep = getelementptr i32, ptr addrspace(1) %in, i32 %tid
  %load = load i32, ptr addrspace(1) %gep
  %vec.0 = insertelement <2 x i32> undef, i32 %val0, i32 0
  %vec.1 = insertelement <2 x i32> %vec.0, i32 %load, i32 1
  %bc = bitcast <2 x i32> %vec.1 to i64
  %add = add i64 %bc, %val1
  store i64 %add, ptr addrspace(1) %out, align 8
  ret void
}
