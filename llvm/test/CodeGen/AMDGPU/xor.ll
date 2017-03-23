; RUN: llc -march=amdgcn -mcpu=verde -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s


; FUNC-LABEL: {{^}}xor_v2i32:
; EG: XOR_INT {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; EG: XOR_INT {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

; SI: v_xor_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
; SI: v_xor_b32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}

define amdgpu_kernel void @xor_v2i32(<2 x i32> addrspace(1)* %out, <2 x i32> addrspace(1)* %in0, <2 x i32> addrspace(1)* %in1) {
  %a = load <2 x i32>, <2 x i32> addrspace(1) * %in0
  %b = load <2 x i32>, <2 x i32> addrspace(1) * %in1
  %result = xor <2 x i32> %a, %b
  store <2 x i32> %result, <2 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}xor_v4i32:
; EG: XOR_INT {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; EG: XOR_INT {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; EG: XOR_INT {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; EG: XOR_INT {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

; SI: v_xor_b32_e32 {{v[0-9]+, v[0-9]+, v[0-9]+}}
; SI: v_xor_b32_e32 {{v[0-9]+, v[0-9]+, v[0-9]+}}
; SI: v_xor_b32_e32 {{v[0-9]+, v[0-9]+, v[0-9]+}}
; SI: v_xor_b32_e32 {{v[0-9]+, v[0-9]+, v[0-9]+}}

define amdgpu_kernel void @xor_v4i32(<4 x i32> addrspace(1)* %out, <4 x i32> addrspace(1)* %in0, <4 x i32> addrspace(1)* %in1) {
  %a = load <4 x i32>, <4 x i32> addrspace(1) * %in0
  %b = load <4 x i32>, <4 x i32> addrspace(1) * %in1
  %result = xor <4 x i32> %a, %b
  store <4 x i32> %result, <4 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}xor_i1:
; EG: XOR_INT {{\** *}}{{T[0-9]+\.[XYZW]}}, {{PS|PV\.[XYZW]}}, {{PS|PV\.[XYZW]}}

; SI-DAG: v_cmp_le_f32_e32 [[CMP0:vcc]], 0, {{v[0-9]+}}
; SI-DAG: v_cmp_le_f32_e64 [[CMP1:s\[[0-9]+:[0-9]+\]]], 1.0, {{v[0-9]+}}
; SI: s_xor_b64 [[XOR:vcc]], [[CMP0]], [[CMP1]]
; SI: v_cndmask_b32_e32 [[RESULT:v[0-9]+]], {{v[0-9]+}}, {{v[0-9]+}}
; SI: buffer_store_dword [[RESULT]]
; SI: s_endpgm
define amdgpu_kernel void @xor_i1(float addrspace(1)* %out, float addrspace(1)* %in0, float addrspace(1)* %in1) {
  %a = load float, float addrspace(1) * %in0
  %b = load float, float addrspace(1) * %in1
  %acmp = fcmp oge float %a, 0.000000e+00
  %bcmp = fcmp oge float %b, 1.000000e+00
  %xor = xor i1 %acmp, %bcmp
  %result = select i1 %xor, float %a, float %b
  store float %result, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v_xor_i1:
; SI: buffer_load_ubyte [[B:v[0-9]+]]
; SI: buffer_load_ubyte [[A:v[0-9]+]]
; SI: v_xor_b32_e32 [[XOR:v[0-9]+]], [[A]], [[B]]
; SI: v_and_b32_e32 [[RESULT:v[0-9]+]], 1, [[XOR]]
; SI: buffer_store_byte [[RESULT]]
define amdgpu_kernel void @v_xor_i1(i1 addrspace(1)* %out, i1 addrspace(1)* %in0, i1 addrspace(1)* %in1) {
  %a = load volatile i1, i1 addrspace(1)* %in0
  %b = load volatile i1, i1 addrspace(1)* %in1
  %xor = xor i1 %a, %b
  store i1 %xor, i1 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}vector_xor_i32:
; SI: v_xor_b32_e32
define amdgpu_kernel void @vector_xor_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in0, i32 addrspace(1)* %in1) {
  %a = load i32, i32 addrspace(1)* %in0
  %b = load i32, i32 addrspace(1)* %in1
  %result = xor i32 %a, %b
  store i32 %result, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}scalar_xor_i32:
; SI: s_xor_b32
define amdgpu_kernel void @scalar_xor_i32(i32 addrspace(1)* %out, i32 %a, i32 %b) {
  %result = xor i32 %a, %b
  store i32 %result, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}scalar_not_i32:
; SI: s_not_b32
define amdgpu_kernel void @scalar_not_i32(i32 addrspace(1)* %out, i32 %a) {
  %result = xor i32 %a, -1
  store i32 %result, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}vector_not_i32:
; SI: v_not_b32
define amdgpu_kernel void @vector_not_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in0, i32 addrspace(1)* %in1) {
  %a = load i32, i32 addrspace(1)* %in0
  %b = load i32, i32 addrspace(1)* %in1
  %result = xor i32 %a, -1
  store i32 %result, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}vector_xor_i64:
; SI: v_xor_b32_e32
; SI: v_xor_b32_e32
; SI: s_endpgm
define amdgpu_kernel void @vector_xor_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %in0, i64 addrspace(1)* %in1) {
  %a = load i64, i64 addrspace(1)* %in0
  %b = load i64, i64 addrspace(1)* %in1
  %result = xor i64 %a, %b
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}scalar_xor_i64:
; SI: s_xor_b64
; SI: s_endpgm
define amdgpu_kernel void @scalar_xor_i64(i64 addrspace(1)* %out, i64 %a, i64 %b) {
  %result = xor i64 %a, %b
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}scalar_not_i64:
; SI: s_not_b64
define amdgpu_kernel void @scalar_not_i64(i64 addrspace(1)* %out, i64 %a) {
  %result = xor i64 %a, -1
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}vector_not_i64:
; SI: v_not_b32
; SI: v_not_b32
define amdgpu_kernel void @vector_not_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %in0, i64 addrspace(1)* %in1) {
  %a = load i64, i64 addrspace(1)* %in0
  %b = load i64, i64 addrspace(1)* %in1
  %result = xor i64 %a, -1
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; Test that we have a pattern to match xor inside a branch.
; Note that in the future the backend may be smart enough to
; use an SALU instruction for this.

; FUNC-LABEL: {{^}}xor_cf:
; SI: s_xor_b64
define amdgpu_kernel void @xor_cf(i64 addrspace(1)* %out, i64 addrspace(1)* %in, i64 %a, i64 %b) {
entry:
  %0 = icmp eq i64 %a, 0
  br i1 %0, label %if, label %else

if:
  %1 = xor i64 %a, %b
  br label %endif

else:
  %2 = load i64, i64 addrspace(1)* %in
  br label %endif

endif:
  %3 = phi i64 [%1, %if], [%2, %else]
  store i64 %3, i64 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}scalar_xor_literal_i64:
; SI: s_load_dwordx2 s{{\[}}[[LO:[0-9]+]]:[[HI:[0-9]+]]{{\]}}, s{{\[[0-9]+:[0-9]+\]}}, {{0xb|0x2c}}
; SI-DAG: s_xor_b32 s[[RES_HI:[0-9]+]], s[[HI]], 0xf237b
; SI-DAG: s_xor_b32 s[[RES_LO:[0-9]+]], s[[LO]], 0x3039
; SI-DAG: v_mov_b32_e32 v{{[0-9]+}}, s[[RES_LO]]
; SI-DAG: v_mov_b32_e32 v{{[0-9]+}}, s[[RES_HI]]
define amdgpu_kernel void @scalar_xor_literal_i64(i64 addrspace(1)* %out, i64 %a) {
  %or = xor i64 %a, 4261135838621753
  store i64 %or, i64 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}scalar_xor_literal_multi_use_i64:
; SI: s_load_dwordx2 s{{\[}}[[LO:[0-9]+]]:[[HI:[0-9]+]]{{\]}}, s{{\[[0-9]+:[0-9]+\]}}, {{0xb|0x2c}}
; SI-DAG: s_mov_b32 s[[K_HI:[0-9]+]], 0xf237b
; SI-DAG: s_movk_i32 s[[K_LO:[0-9]+]], 0x3039
; SI: s_xor_b64 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{\[}}[[K_LO]]:[[K_HI]]{{\]}}

; SI: s_add_u32 s{{[0-9]+}}, s{{[0-9]+}}, s[[K_LO]]
; SI: s_addc_u32 s{{[0-9]+}}, s{{[0-9]+}}, s[[K_HI]]
define amdgpu_kernel void @scalar_xor_literal_multi_use_i64(i64 addrspace(1)* %out, i64 %a, i64 %b) {
  %or = xor i64 %a, 4261135838621753
  store i64 %or, i64 addrspace(1)* %out

  %foo = add i64 %b, 4261135838621753
  store volatile i64 %foo, i64 addrspace(1)* undef
  ret void
}

; FUNC-LABEL: {{^}}scalar_xor_inline_imm_i64:
; SI: s_load_dwordx2 s{{\[}}[[VAL_LO:[0-9]+]]:[[VAL_HI:[0-9]+]]{{\]}}, s{{\[[0-9]+:[0-9]+\]}}, {{0xb|0x2c}}
; SI-NOT: xor_b32
; SI: s_xor_b32 s[[VAL_LO]], s[[VAL_LO]], 63
; SI-NOT: xor_b32
; SI: v_mov_b32_e32 v[[VLO:[0-9]+]], s[[VAL_LO]]
; SI-NOT: xor_b32
; SI: v_mov_b32_e32 v[[VHI:[0-9]+]], s[[VAL_HI]]
; SI-NOT: xor_b32
; SI: buffer_store_dwordx2 v{{\[}}[[VLO]]:[[VHI]]{{\]}}
define amdgpu_kernel void @scalar_xor_inline_imm_i64(i64 addrspace(1)* %out, i64 %a) {
  %or = xor i64 %a, 63
  store i64 %or, i64 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}scalar_xor_neg_inline_imm_i64:
; SI: s_load_dwordx2 [[VAL:s\[[0-9]+:[0-9]+\]]], s{{\[[0-9]+:[0-9]+\]}}, {{0xb|0x2c}}
; SI: s_xor_b64 [[VAL]], [[VAL]], -8
define amdgpu_kernel void @scalar_xor_neg_inline_imm_i64(i64 addrspace(1)* %out, i64 %a) {
  %or = xor i64 %a, -8
  store i64 %or, i64 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}vector_xor_i64_neg_inline_imm:
; SI: buffer_load_dwordx2 v{{\[}}[[LO_VREG:[0-9]+]]:[[HI_VREG:[0-9]+]]{{\]}},
; SI: v_xor_b32_e32 {{v[0-9]+}}, -8, v[[LO_VREG]]
; SI: v_xor_b32_e32 {{v[0-9]+}}, -1, {{.*}}
; SI: s_endpgm
define amdgpu_kernel void @vector_xor_i64_neg_inline_imm(i64 addrspace(1)* %out, i64 addrspace(1)* %a, i64 addrspace(1)* %b) {
  %loada = load i64, i64 addrspace(1)* %a, align 8
  %or = xor i64 %loada, -8
  store i64 %or, i64 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}vector_xor_literal_i64:
; SI-DAG: buffer_load_dwordx2 v{{\[}}[[LO_VREG:[0-9]+]]:[[HI_VREG:[0-9]+]]{{\]}},
; SI-DAG: v_xor_b32_e32 {{v[0-9]+}}, 0xdf77987f, v[[LO_VREG]]
; SI-DAG: v_xor_b32_e32 {{v[0-9]+}}, 0x146f, v[[HI_VREG]]
; SI: s_endpgm
define amdgpu_kernel void @vector_xor_literal_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %a, i64 addrspace(1)* %b) {
  %loada = load i64, i64 addrspace(1)* %a, align 8
  %or = xor i64 %loada, 22470723082367
  store i64 %or, i64 addrspace(1)* %out
  ret void
}
