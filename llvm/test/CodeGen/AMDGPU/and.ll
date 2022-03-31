; RUN: llc -march=amdgcn -mcpu=verde -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global,-xnack -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

declare i32 @llvm.amdgcn.workitem.id.x() #0

; FUNC-LABEL: {{^}}test2:
; EG: AND_INT {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; EG: AND_INT {{\*? *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

; SI: s_and_b32 s{{[0-9]+, s[0-9]+, s[0-9]+}}
; SI: s_and_b32 s{{[0-9]+, s[0-9]+, s[0-9]+}}

define amdgpu_kernel void @test2(<2 x i32> addrspace(1)* %out, <2 x i32> addrspace(1)* %in) {
  %b_ptr = getelementptr <2 x i32>, <2 x i32> addrspace(1)* %in, i32 1
  %a = load <2 x i32>, <2 x i32> addrspace(1) * %in
  %b = load <2 x i32>, <2 x i32> addrspace(1) * %b_ptr
  %result = and <2 x i32> %a, %b
  store <2 x i32> %result, <2 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}test4:
; EG: AND_INT {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; EG: AND_INT {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; EG: AND_INT {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; EG: AND_INT {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}


; SI: s_and_b32 s{{[0-9]+, s[0-9]+, s[0-9]+}}
; SI: s_and_b32 s{{[0-9]+, s[0-9]+, s[0-9]+}}
; SI: s_and_b32 s{{[0-9]+, s[0-9]+, s[0-9]+}}
; SI: s_and_b32 s{{[0-9]+, s[0-9]+, s[0-9]+}}

define amdgpu_kernel void @test4(<4 x i32> addrspace(1)* %out, <4 x i32> addrspace(1)* %in) {
  %b_ptr = getelementptr <4 x i32>, <4 x i32> addrspace(1)* %in, i32 1
  %a = load <4 x i32>, <4 x i32> addrspace(1) * %in
  %b = load <4 x i32>, <4 x i32> addrspace(1) * %b_ptr
  %result = and <4 x i32> %a, %b
  store <4 x i32> %result, <4 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}s_and_i32:
; SI: s_and_b32
define amdgpu_kernel void @s_and_i32(i32 addrspace(1)* %out, i32 %a, i32 %b) {
  %and = and i32 %a, %b
  store i32 %and, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}s_and_constant_i32:
; SI: s_and_b32 s{{[0-9]+}}, s{{[0-9]+}}, 0x12d687
define amdgpu_kernel void @s_and_constant_i32(i32 addrspace(1)* %out, i32 %a) {
  %and = and i32 %a, 1234567
  store i32 %and, i32 addrspace(1)* %out, align 4
  ret void
}

; FIXME: We should really duplicate the constant so that the SALU use
; can fold into the s_and_b32 and the VALU one is materialized
; directly without copying from the SGPR.

; Second use is a VGPR use of the constant.
; FUNC-LABEL: {{^}}s_and_multi_use_constant_i32_0:
; SI: s_mov_b32 [[K:s[0-9]+]], 0x12d687
; SI-DAG: s_and_b32 [[AND:s[0-9]+]], s{{[0-9]+}}, [[K]]
; SI-DAG: v_mov_b32_e32 [[VK:v[0-9]+]], [[K]]
; SI: buffer_store_dword [[VK]]
define amdgpu_kernel void @s_and_multi_use_constant_i32_0(i32 addrspace(1)* %out, i32 %a, i32 %b) {
  %and = and i32 %a, 1234567

  ; Just to stop future replacement of copy to vgpr + store with VALU op.
  %foo = add i32 %and, %b
  store volatile i32 %foo, i32 addrspace(1)* %out
  store volatile i32 1234567, i32 addrspace(1)* %out
  ret void
}

; Second use is another SGPR use of the constant.
; FUNC-LABEL: {{^}}s_and_multi_use_constant_i32_1:
; SI: s_mov_b32 [[K:s[0-9]+]], 0x12d687
; SI: s_and_b32 [[AND:s[0-9]+]], s{{[0-9]+}}, [[K]]
; SI: s_add_i32
; SI: s_add_i32 [[ADD:s[0-9]+]], s{{[0-9]+}}, [[K]]
; SI: v_mov_b32_e32 [[VADD:v[0-9]+]], [[ADD]]
; SI: buffer_store_dword [[VADD]]
define amdgpu_kernel void @s_and_multi_use_constant_i32_1(i32 addrspace(1)* %out, i32 %a, i32 %b) {
  %and = and i32 %a, 1234567
  %foo = add i32 %and, 1234567
  %bar = add i32 %foo, %b
  store volatile i32 %bar, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v_and_i32_vgpr_vgpr:
; SI: v_and_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @v_and_i32_vgpr_vgpr(i32 addrspace(1)* %out, i32 addrspace(1)* %aptr, i32 addrspace(1)* %bptr) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.a = getelementptr i32, i32 addrspace(1)* %aptr, i32 %tid
  %gep.b = getelementptr i32, i32 addrspace(1)* %bptr, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %a = load i32, i32 addrspace(1)* %gep.a
  %b = load i32, i32 addrspace(1)* %gep.b
  %and = and i32 %a, %b
  store i32 %and, i32 addrspace(1)* %gep.out
  ret void
}

; FUNC-LABEL: {{^}}v_and_i32_sgpr_vgpr:
; SI-DAG: s_load_dword [[SA:s[0-9]+]]
; SI-DAG: {{buffer|flat}}_load_dword [[VB:v[0-9]+]]
; SI: v_and_b32_e32 v{{[0-9]+}}, [[SA]], [[VB]]
define amdgpu_kernel void @v_and_i32_sgpr_vgpr(i32 addrspace(1)* %out, i32 %a, i32 addrspace(1)* %bptr) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.b = getelementptr i32, i32 addrspace(1)* %bptr, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %b = load i32, i32 addrspace(1)* %gep.b
  %and = and i32 %a, %b
  store i32 %and, i32 addrspace(1)* %gep.out
  ret void
}

; FUNC-LABEL: {{^}}v_and_i32_vgpr_sgpr:
; SI-DAG: s_load_dword [[SA:s[0-9]+]]
; SI-DAG: {{buffer|flat}}_load_dword [[VB:v[0-9]+]]
; SI: v_and_b32_e32 v{{[0-9]+}}, [[SA]], [[VB]]
define amdgpu_kernel void @v_and_i32_vgpr_sgpr(i32 addrspace(1)* %out, i32 addrspace(1)* %aptr, i32 %b) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.a = getelementptr i32, i32 addrspace(1)* %aptr, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %a = load i32, i32 addrspace(1)* %gep.a
  %and = and i32 %a, %b
  store i32 %and, i32 addrspace(1)* %gep.out
  ret void
}

; FUNC-LABEL: {{^}}v_and_constant_i32
; SI: v_and_b32_e32 v{{[0-9]+}}, 0x12d687, v{{[0-9]+}}
define amdgpu_kernel void @v_and_constant_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %aptr) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep = getelementptr i32, i32 addrspace(1)* %aptr, i32 %tid
  %a = load i32, i32 addrspace(1)* %gep, align 4
  %and = and i32 %a, 1234567
  store i32 %and, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v_and_inline_imm_64_i32
; SI: v_and_b32_e32 v{{[0-9]+}}, 64, v{{[0-9]+}}
define amdgpu_kernel void @v_and_inline_imm_64_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %aptr) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep = getelementptr i32, i32 addrspace(1)* %aptr, i32 %tid
  %a = load i32, i32 addrspace(1)* %gep, align 4
  %and = and i32 %a, 64
  store i32 %and, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v_and_inline_imm_neg_16_i32
; SI: v_and_b32_e32 v{{[0-9]+}}, -16, v{{[0-9]+}}
define amdgpu_kernel void @v_and_inline_imm_neg_16_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %aptr) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep = getelementptr i32, i32 addrspace(1)* %aptr, i32 %tid
  %a = load i32, i32 addrspace(1)* %gep, align 4
  %and = and i32 %a, -16
  store i32 %and, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}s_and_i64
; SI: s_and_b64
define amdgpu_kernel void @s_and_i64(i64 addrspace(1)* %out, i64 %a, i64 %b) {
  %and = and i64 %a, %b
  store i64 %and, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}s_and_i1:
; SI: s_load_dword [[LOAD:s[0-9]+]]
; SI: s_lshr_b32 [[B_SHIFT:s[0-9]+]], [[LOAD]], 8
; SI: s_and_b32 [[AND:s[0-9]+]], [[LOAD]], [[B_SHIFT]]
; SI: s_and_b32 [[AND_TRUNC:s[0-9]+]], [[AND]], 1{{$}}
; SI: v_mov_b32_e32 [[V_AND_TRUNC:v[0-9]+]], [[AND_TRUNC]]
; SI: buffer_store_byte [[V_AND_TRUNC]]
define amdgpu_kernel void @s_and_i1(i1 addrspace(1)* %out, i1 %a, i1 %b) {
  %and = and i1 %a, %b
  store i1 %and, i1 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}s_and_constant_i64:
; SI-DAG: s_and_b32 s{{[0-9]+}}, s{{[0-9]+}}, 0x80000{{$}}
; SI-DAG: s_and_b32 s{{[0-9]+}}, s{{[0-9]+}}, 0x80{{$}}
; SI: buffer_store_dwordx2
define amdgpu_kernel void @s_and_constant_i64(i64 addrspace(1)* %out, i64 %a) {
  %and = and i64 %a, 549756338176
  store i64 %and, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}s_and_multi_use_constant_i64:
; XSI-DAG: s_mov_b32 s[[KLO:[0-9]+]], 0x80000{{$}}
; XSI-DAG: s_mov_b32 s[[KHI:[0-9]+]], 0x80{{$}}
; XSI: s_and_b64 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s[[[KLO]]:[[KHI]]]
define amdgpu_kernel void @s_and_multi_use_constant_i64(i64 addrspace(1)* %out, i64 %a, i64 %b) {
  %and0 = and i64 %a, 549756338176
  %and1 = and i64 %b, 549756338176
  store volatile i64 %and0, i64 addrspace(1)* %out
  store volatile i64 %and1, i64 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}s_and_32_bit_constant_i64:
; SI: s_load_dwordx2
; SI-NOT: and
; SI: s_and_b32 s{{[0-9]+}}, s{{[0-9]+}}, 0x12d687{{$}}
; SI-NOT: and
; SI: buffer_store_dwordx2
define amdgpu_kernel void @s_and_32_bit_constant_i64(i64 addrspace(1)* %out, i32, i64 %a) {
  %and = and i64 %a, 1234567
  store i64 %and, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}s_and_multi_use_inline_imm_i64:
; SI: s_load_dwordx2
; SI: s_load_dword [[A:s[0-9]+]]
; SI: s_load_dword [[B:s[0-9]+]]
; SI: s_load_dwordx2
; SI-NOT: and
; SI: s_lshl_b32 [[A]], [[A]], 1
; SI: s_lshl_b32 [[B]], [[B]], 1
; SI: s_and_b32 s{{[0-9]+}}, [[A]], 62
; SI: s_and_b32 s{{[0-9]+}}, [[B]], 62
; SI-NOT: and
; SI: buffer_store_dwordx2
define amdgpu_kernel void @s_and_multi_use_inline_imm_i64(i64 addrspace(1)* %out, i32, i64 %a, i32, i64 %b, i32, i64 %c) {
  %shl.a = shl i64 %a, 1
  %shl.b = shl i64 %b, 1
  %and0 = and i64 %shl.a, 62
  %and1 = and i64 %shl.b, 62
  %add0 = add i64 %and0, %c
  %add1 = add i64 %and1, %c
  store volatile i64 %add0, i64 addrspace(1)* %out
  store volatile i64 %add1, i64 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v_and_i64:
; SI: v_and_b32
; SI: v_and_b32
define amdgpu_kernel void @v_and_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %aptr, i64 addrspace(1)* %bptr) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.a = getelementptr i64, i64 addrspace(1)* %aptr, i32 %tid
  %a = load i64, i64 addrspace(1)* %gep.a, align 8
  %gep.b = getelementptr i64, i64 addrspace(1)* %bptr, i32 %tid
  %b = load i64, i64 addrspace(1)* %gep.b, align 8
  %and = and i64 %a, %b
  store i64 %and, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}v_and_constant_i64:
; SI-DAG: v_and_b32_e32 {{v[0-9]+}}, 0xab19b207, {{v[0-9]+}}
; SI-DAG: v_and_b32_e32 {{v[0-9]+}}, 0x11e, {{v[0-9]+}}
; SI: buffer_store_dwordx2
define amdgpu_kernel void @v_and_constant_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %aptr) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.a = getelementptr i64, i64 addrspace(1)* %aptr, i32 %tid
  %a = load i64, i64 addrspace(1)* %gep.a, align 8
  %and = and i64 %a, 1231231234567
  store i64 %and, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}v_and_multi_use_constant_i64:
; SI-DAG: buffer_load_dwordx2 v[[[LO0:[0-9]+]]:[[HI0:[0-9]+]]]
; SI-DAG: buffer_load_dwordx2 v[[[LO1:[0-9]+]]:[[HI1:[0-9]+]]]
; SI-DAG: s_movk_i32 [[KHI:s[0-9]+]], 0x11e{{$}}
; SI-DAG: s_mov_b32 [[KLO:s[0-9]+]], 0xab19b207{{$}}
; SI-DAG: v_and_b32_e32 {{v[0-9]+}}, [[KLO]], v[[LO0]]
; SI-DAG: v_and_b32_e32 {{v[0-9]+}}, [[KHI]], v[[HI0]]
; SI-DAG: v_and_b32_e32 {{v[0-9]+}}, [[KLO]], v[[LO1]]
; SI-DAG: v_and_b32_e32 {{v[0-9]+}}, [[KHI]], v[[HI1]]
; SI: buffer_store_dwordx2
; SI: buffer_store_dwordx2
define amdgpu_kernel void @v_and_multi_use_constant_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %aptr) {
  %a = load volatile i64, i64 addrspace(1)* %aptr
  %b = load volatile i64, i64 addrspace(1)* %aptr
  %and0 = and i64 %a, 1231231234567
  %and1 = and i64 %b, 1231231234567
  store volatile i64 %and0, i64 addrspace(1)* %out
  store volatile i64 %and1, i64 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v_and_multi_use_inline_imm_i64:
; SI: buffer_load_dwordx2 v[[[LO0:[0-9]+]]:[[HI0:[0-9]+]]]
; SI-NOT: and
; SI: buffer_load_dwordx2 v[[[LO1:[0-9]+]]:[[HI1:[0-9]+]]]
; SI-NOT: and
; SI: v_and_b32_e32 v[[RESLO0:[0-9]+]], 63, v[[LO0]]
; SI: v_and_b32_e32 v[[RESLO1:[0-9]+]], 63, v[[LO1]]
; SI-NOT: and
; SI: buffer_store_dwordx2 v[[[RESLO0]]
; SI: buffer_store_dwordx2 v[[[RESLO1]]
define amdgpu_kernel void @v_and_multi_use_inline_imm_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %aptr) {
  %a = load volatile i64, i64 addrspace(1)* %aptr
  %b = load volatile i64, i64 addrspace(1)* %aptr
  %and0 = and i64 %a, 63
  %and1 = and i64 %b, 63
  store volatile i64 %and0, i64 addrspace(1)* %out
  store volatile i64 %and1, i64 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v_and_i64_32_bit_constant:
; SI: {{buffer|flat}}_load_dword [[VAL:v[0-9]+]]
; SI-NOT: and
; SI: v_and_b32_e32 {{v[0-9]+}}, 0x12d687, [[VAL]]
; SI-NOT: and
; SI: buffer_store_dwordx2
define amdgpu_kernel void @v_and_i64_32_bit_constant(i64 addrspace(1)* %out, i64 addrspace(1)* %aptr) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.a = getelementptr i64, i64 addrspace(1)* %aptr, i32 %tid
  %a = load i64, i64 addrspace(1)* %gep.a, align 8
  %and = and i64 %a, 1234567
  store i64 %and, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}v_and_inline_imm_i64:
; SI: {{buffer|flat}}_load_dword v{{[0-9]+}}
; SI-NOT: and
; SI: v_and_b32_e32 {{v[0-9]+}}, 64, {{v[0-9]+}}
; SI-NOT: and
; SI: buffer_store_dwordx2
define amdgpu_kernel void @v_and_inline_imm_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %aptr) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.a = getelementptr i64, i64 addrspace(1)* %aptr, i32 %tid
  %a = load i64, i64 addrspace(1)* %gep.a, align 8
  %and = and i64 %a, 64
  store i64 %and, i64 addrspace(1)* %out, align 8
  ret void
}

; FIXME: Should be able to reduce load width
; FUNC-LABEL: {{^}}v_and_inline_neg_imm_i64:
; SI: {{buffer|flat}}_load_dwordx2 v[[[VAL_LO:[0-9]+]]:[[VAL_HI:[0-9]+]]]
; SI-NOT: and
; SI: v_and_b32_e32 v[[VAL_LO]], -8, v[[VAL_LO]]
; SI-NOT: and
; SI: buffer_store_dwordx2 v[[[VAL_LO]]:[[VAL_HI]]]
define amdgpu_kernel void @v_and_inline_neg_imm_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %aptr) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %gep.a = getelementptr i64, i64 addrspace(1)* %aptr, i32 %tid
  %a = load i64, i64 addrspace(1)* %gep.a, align 8
  %and = and i64 %a, -8
  store i64 %and, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}s_and_inline_imm_64_i64
; SI: s_load_dword
; SI-NOT: and
; SI: s_and_b32 s{{[0-9]+}}, s{{[0-9]+}}, 64
; SI-NOT: and
; SI: buffer_store_dword
define amdgpu_kernel void @s_and_inline_imm_64_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %aptr, i64 %a) {
  %and = and i64 %a, 64
  store i64 %and, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}s_and_inline_imm_64_i64_noshrink:
; SI: s_load_dword [[A:s[0-9]+]]
; SI: s_lshl_b32 [[A]], [[A]], 1{{$}}
; SI-NOT: and
; SI: s_and_b32 s{{[0-9]+}}, [[A]], 64
; SI-NOT: and
; SI: s_add_u32
; SI-NEXT: s_addc_u32
define amdgpu_kernel void @s_and_inline_imm_64_i64_noshrink(i64 addrspace(1)* %out, i64 addrspace(1)* %aptr, i64 %a, i32, i64 %b) {
  %shl = shl i64 %a, 1
  %and = and i64 %shl, 64
  %add = add i64 %and, %b
  store i64 %add, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}s_and_inline_imm_1_i64
; SI: s_load_dwordx2
; SI-NOT: and
; SI: s_and_b32 s{{[0-9]+}}, s{{[0-9]+}}, 1
; SI-NOT: and
; SI: buffer_store_dwordx2
define amdgpu_kernel void @s_and_inline_imm_1_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %aptr, i64 %a) {
  %and = and i64 %a, 1
  store i64 %and, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}s_and_inline_imm_1.0_i64
; XSI: s_and_b64 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 1.0

; SI: s_load_dword
; SI: s_load_dwordx2
; SI-NOT: and
; SI: s_and_b32 {{s[0-9]+}}, {{s[0-9]+}}, 0x3ff00000
; SI-NOT: and
; SI: buffer_store_dwordx2
define amdgpu_kernel void @s_and_inline_imm_1.0_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %aptr, i64 %a) {
  %and = and i64 %a, 4607182418800017408
  store i64 %and, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}s_and_inline_imm_neg_1.0_i64
; XSI: s_and_b64 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, -1.0

; SI: s_load_dword
; SI: s_load_dwordx2
; SI-NOT: and
; SI: s_and_b32 {{s[0-9]+}}, {{s[0-9]+}}, 0xbff00000
; SI-NOT: and
; SI: buffer_store_dwordx2
define amdgpu_kernel void @s_and_inline_imm_neg_1.0_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %aptr, i64 %a) {
  %and = and i64 %a, 13830554455654793216
  store i64 %and, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}s_and_inline_imm_0.5_i64
; XSI: s_and_b64 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0.5

; SI: s_load_dword
; SI: s_load_dwordx2
; SI-NOT: and
; SI: s_and_b32 {{s[0-9]+}}, {{s[0-9]+}}, 0x3fe00000
; SI-NOT: and
; SI: buffer_store_dwordx2
define amdgpu_kernel void @s_and_inline_imm_0.5_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %aptr, i64 %a) {
  %and = and i64 %a, 4602678819172646912
  store i64 %and, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}s_and_inline_imm_neg_0.5_i64:
; XSI: s_and_b64 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, -0.5

; SI: s_load_dword
; SI: s_load_dwordx2
; SI-NOT: and
; SI: s_and_b32 {{s[0-9]+}}, {{s[0-9]+}}, 0xbfe00000
; SI-NOT: and
; SI: buffer_store_dwordx2
define amdgpu_kernel void @s_and_inline_imm_neg_0.5_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %aptr, i64 %a) {
  %and = and i64 %a, 13826050856027422720
  store i64 %and, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}s_and_inline_imm_2.0_i64:
; SI: s_load_dword
; SI: s_load_dwordx2
; SI-NOT: and
; SI: s_and_b32 {{s[0-9]+}}, {{s[0-9]+}}, 2.0
; SI-NOT: and
; SI: buffer_store_dwordx2
define amdgpu_kernel void @s_and_inline_imm_2.0_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %aptr, i64 %a) {
  %and = and i64 %a, 4611686018427387904
  store i64 %and, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}s_and_inline_imm_neg_2.0_i64:
; SI: s_load_dword
; SI: s_load_dwordx2
; SI-NOT: and
; SI: s_and_b32 {{s[0-9]+}}, {{s[0-9]+}}, -2.0
; SI-NOT: and
; SI: buffer_store_dwordx2
define amdgpu_kernel void @s_and_inline_imm_neg_2.0_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %aptr, i64 %a) {
  %and = and i64 %a, 13835058055282163712
  store i64 %and, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}s_and_inline_imm_4.0_i64:
; XSI: s_and_b64 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 4.0

; SI: s_load_dword
; SI: s_load_dwordx2
; SI-NOT: and
; SI: s_and_b32 {{s[0-9]+}}, {{s[0-9]+}}, 0x40100000
; SI-NOT: and
; SI: buffer_store_dwordx2
define amdgpu_kernel void @s_and_inline_imm_4.0_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %aptr, i64 %a) {
  %and = and i64 %a, 4616189618054758400
  store i64 %and, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}s_and_inline_imm_neg_4.0_i64:
; XSI: s_and_b64 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, -4.0

; SI: s_load_dword
; SI: s_load_dwordx2
; SI-NOT: and
; SI: s_and_b32 {{s[0-9]+}}, {{s[0-9]+}}, 0xc0100000
; SI-NOT: and
; SI: buffer_store_dwordx2
define amdgpu_kernel void @s_and_inline_imm_neg_4.0_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %aptr, i64 %a) {
  %and = and i64 %a, 13839561654909534208
  store i64 %and, i64 addrspace(1)* %out, align 8
  ret void
}


; Test with the 64-bit integer bitpattern for a 32-bit float in the
; low 32-bits, which is not a valid 64-bit inline immmediate.

; FUNC-LABEL: {{^}}s_and_inline_imm_f32_4.0_i64:
; SI: s_load_dword s
; SI: s_load_dwordx2
; SI-NOT: and
; SI: s_and_b32 s[[K_HI:[0-9]+]], s{{[0-9]+}}, 4.0
; SI-NOT: and
; SI: buffer_store_dwordx2
define amdgpu_kernel void @s_and_inline_imm_f32_4.0_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %aptr, i64 %a) {
  %and = and i64 %a, 1082130432
  store i64 %and, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}s_and_inline_imm_f32_neg_4.0_i64:
; SI: s_load_dwordx2
; SI: s_load_dwordx2
; SI-NOT: and
; SI: s_and_b32 s[[K_HI:[0-9]+]], s{{[0-9]+}}, -4.0
; SI-NOT: and
; SI: buffer_store_dwordx2
define amdgpu_kernel void @s_and_inline_imm_f32_neg_4.0_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %aptr, i64 %a) {
  %and = and i64 %a, -1065353216
  store i64 %and, i64 addrspace(1)* %out, align 8
  ret void
}

; Shift into upper 32-bits
; SI: s_load_dword
; SI: s_load_dwordx2
; SI-NOT: and
; SI: s_and_b32 s[[K_HI:[0-9]+]], s{{[0-9]+}}, 4.0
; SI-NOT: and
; SI: buffer_store_dwordx2
define amdgpu_kernel void @s_and_inline_high_imm_f32_4.0_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %aptr, i64 %a) {
  %and = and i64 %a, 4647714815446351872
  store i64 %and, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}s_and_inline_high_imm_f32_neg_4.0_i64:
; SI: s_load_dword
; SI: s_load_dwordx2
; SI-NOT: and
; SI: s_and_b32 s[[K_HI:[0-9]+]], s{{[0-9]+}}, -4.0
; SI-NOT: and
; SI: buffer_store_dwordx2
define amdgpu_kernel void @s_and_inline_high_imm_f32_neg_4.0_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %aptr, i64 %a) {
  %and = and i64 %a, 13871086852301127680
  store i64 %and, i64 addrspace(1)* %out, align 8
  ret void
}
attributes #0 = { nounwind readnone }
