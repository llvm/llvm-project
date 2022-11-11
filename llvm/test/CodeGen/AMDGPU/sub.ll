; RUN: llc -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=verde -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,SI %s
; RUN: llc -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=fiji -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,VI,GFX89 %s
; RUN: llc -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX9,GFX89 %s

declare i32 @llvm.amdgcn.workitem.id.x() nounwind readnone speculatable

; GCN-LABEL: {{^}}s_sub_i32:
; GCN: s_load_dwordx4 s[[[#LOAD:]]:{{[0-9]+}}]
; GCN: s_sub_i32 s{{[0-9]+}}, s[[#LOAD + 2]], s[[#LOAD + 3]]
define amdgpu_kernel void @s_sub_i32(i32 addrspace(1)* %out, i32 %a, i32 %b) {
  %result = sub i32 %a, %b
  store i32 %result, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}s_sub_imm_i32:
; GCN: s_load_dword [[A:s[0-9]+]]
; GCN: s_sub_i32 s{{[0-9]+}}, 0x4d2, [[A]]
define amdgpu_kernel void @s_sub_imm_i32(i32 addrspace(1)* %out, i32 %a) {
  %result = sub i32 1234, %a
  store i32 %result, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_sub_i32:
; SI: v_subrev_i32_e32 v{{[0-9]+, vcc, v[0-9]+, v[0-9]+}}
; GFX9: v_sub_u32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
define amdgpu_kernel void @test_sub_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in) {
  %b_ptr = getelementptr i32, i32 addrspace(1)* %in, i32 1
  %a = load i32, i32 addrspace(1)* %in
  %b = load i32, i32 addrspace(1)* %b_ptr
  %result = sub i32 %a, %b
  store i32 %result, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_sub_imm_i32:
; SI: v_sub_i32_e32 v{{[0-9]+}}, vcc, 0x7b, v{{[0-9]+}}
; GFX9: v_sub_u32_e32 v{{[0-9]+}}, 0x7b, v{{[0-9]+}}
define amdgpu_kernel void @test_sub_imm_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in) {
  %a = load i32, i32 addrspace(1)* %in
  %result = sub i32 123, %a
  store i32 %result, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_sub_v2i32:
; SI: v_sub_i32_e32 v{{[0-9]+, vcc, v[0-9]+, v[0-9]+}}
; SI: v_subrev_i32_e32 v{{[0-9]+, vcc, v[0-9]+, v[0-9]+}}

; GFX9: v_sub_u32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
; GFX9: v_sub_u32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
define amdgpu_kernel void @test_sub_v2i32(<2 x i32> addrspace(1)* %out, <2 x i32> addrspace(1)* %in) {
  %b_ptr = getelementptr <2 x i32>, <2 x i32> addrspace(1)* %in, i32 1
  %a = load <2 x i32>, <2 x i32> addrspace(1) * %in
  %b = load <2 x i32>, <2 x i32> addrspace(1) * %b_ptr
  %result = sub <2 x i32> %a, %b
  store <2 x i32> %result, <2 x i32> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_sub_v4i32:
; SI: v_subrev_i32_e32 v{{[0-9]+, vcc, v[0-9]+, v[0-9]+}}
; SI: v_sub_i32_e32 v{{[0-9]+, vcc, v[0-9]+, v[0-9]+}}
; SI: v_sub_i32_e32 v{{[0-9]+, vcc, v[0-9]+, v[0-9]+}}
; SI: v_sub_i32_e32 v{{[0-9]+, vcc, v[0-9]+, v[0-9]+}}

; GFX9: v_sub_u32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
; GFX9: v_sub_u32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
; GFX9: v_sub_u32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
; GFX9: v_sub_u32_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
define amdgpu_kernel void @test_sub_v4i32(<4 x i32> addrspace(1)* %out, <4 x i32> addrspace(1)* %in) {
  %b_ptr = getelementptr <4 x i32>, <4 x i32> addrspace(1)* %in, i32 1
  %a = load <4 x i32>, <4 x i32> addrspace(1) * %in
  %b = load <4 x i32>, <4 x i32> addrspace(1) * %b_ptr
  %result = sub <4 x i32> %a, %b
  store <4 x i32> %result, <4 x i32> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_sub_i16:
; SI: v_sub_i32_e32 v{{[0-9]+}}, vcc,
; GFX89: v_sub_u16_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
define amdgpu_kernel void @test_sub_i16(i16 addrspace(1)* %out, i16 addrspace(1)* %in) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr i16, i16 addrspace(1)* %in, i32 %tid
  %b_ptr = getelementptr i16, i16 addrspace(1)* %gep, i32 1
  %a = load volatile i16, i16 addrspace(1)* %gep
  %b = load volatile i16, i16 addrspace(1)* %b_ptr
  %result = sub i16 %a, %b
  store i16 %result, i16 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_sub_v2i16:
; VI: v_sub_u16_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
; VI: v_sub_u16_sdwa v{{[0-9]+, v[0-9]+, v[0-9]+}}

; GFX9: v_pk_sub_i16
define amdgpu_kernel void @test_sub_v2i16(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(1)* %in) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr <2 x i16>, <2 x i16> addrspace(1)* %in, i32 %tid
  %b_ptr = getelementptr <2 x i16>, <2 x i16> addrspace(1)* %gep, i16 1
  %a = load <2 x i16>, <2 x i16> addrspace(1)* %gep
  %b = load <2 x i16>, <2 x i16> addrspace(1)* %b_ptr
  %result = sub <2 x i16> %a, %b
  store <2 x i16> %result, <2 x i16> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_sub_v4i16:
; VI: v_sub_u16_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
; VI: v_sub_u16_sdwa v{{[0-9]+, v[0-9]+, v[0-9]+}}
; VI: v_sub_u16_e32 v{{[0-9]+, v[0-9]+, v[0-9]+}}
; VI: v_sub_u16_sdwa v{{[0-9]+, v[0-9]+, v[0-9]+}}

; GFX9: v_pk_sub_i16
; GFX9: v_pk_sub_i16
define amdgpu_kernel void @test_sub_v4i16(<4 x i16> addrspace(1)* %out, <4 x i16> addrspace(1)* %in) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr <4 x i16>, <4 x i16> addrspace(1)* %in, i32 %tid
  %b_ptr = getelementptr <4 x i16>, <4 x i16> addrspace(1)* %gep, i16 1
  %a = load <4 x i16>, <4 x i16> addrspace(1) * %gep
  %b = load <4 x i16>, <4 x i16> addrspace(1) * %b_ptr
  %result = sub <4 x i16> %a, %b
  store <4 x i16> %result, <4 x i16> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}s_sub_i64:
; GCN: s_sub_u32
; GCN: s_subb_u32
define amdgpu_kernel void @s_sub_i64(i64 addrspace(1)* noalias %out, i64 %a, i64 %b) nounwind {
  %result = sub i64 %a, %b
  store i64 %result, i64 addrspace(1)* %out, align 8
  ret void
}

; GCN-LABEL: {{^}}v_sub_i64:
; SI: v_sub_i32_e32
; SI: v_subb_u32_e32

; VI: v_sub_u32_e32
; VI: v_subb_u32_e32

; GFX9: v_sub_co_u32_e32
; GFX9: v_subb_co_u32_e32
define amdgpu_kernel void @v_sub_i64(i64 addrspace(1)* noalias %out, i64 addrspace(1)* noalias %inA, i64 addrspace(1)* noalias %inB) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() readnone
  %a_ptr = getelementptr i64, i64 addrspace(1)* %inA, i32 %tid
  %b_ptr = getelementptr i64, i64 addrspace(1)* %inB, i32 %tid
  %a = load i64, i64 addrspace(1)* %a_ptr
  %b = load i64, i64 addrspace(1)* %b_ptr
  %result = sub i64 %a, %b
  store i64 %result, i64 addrspace(1)* %out, align 8
  ret void
}

; GCN-LABEL: {{^}}v_test_sub_v2i64:
; SI: v_sub_i32_e32 v{{[0-9]+}}, vcc,
; SI: v_subb_u32_e32 v{{[0-9]+}}, vcc,
; SI: v_sub_i32_e32 v{{[0-9]+}}, vcc,
; SI: v_subb_u32_e32 v{{[0-9]+}}, vcc,

; VI: v_sub_u32_e32 v{{[0-9]+}}, vcc,
; VI: v_subb_u32_e32 v{{[0-9]+}}, vcc,
; VI: v_sub_u32_e32 v{{[0-9]+}}, vcc,
; VI: v_subb_u32_e32 v{{[0-9]+}}, vcc,

; GFX9: v_sub_co_u32_e32 v{{[0-9]+}}, vcc,
; GFX9: v_subb_co_u32_e32 v{{[0-9]+}}, vcc,
; GFX9: v_sub_co_u32_e32 v{{[0-9]+}}, vcc,
; GFX9: v_subb_co_u32_e32 v{{[0-9]+}}, vcc,
define amdgpu_kernel void @v_test_sub_v2i64(<2 x i64> addrspace(1)* %out, <2 x i64> addrspace(1)* noalias %inA, <2 x i64> addrspace(1)* noalias %inB) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() readnone
  %a_ptr = getelementptr <2 x i64>, <2 x i64> addrspace(1)* %inA, i32 %tid
  %b_ptr = getelementptr <2 x i64>, <2 x i64> addrspace(1)* %inB, i32 %tid
  %a = load <2 x i64>, <2 x i64> addrspace(1)* %a_ptr
  %b = load <2 x i64>, <2 x i64> addrspace(1)* %b_ptr
  %result = sub <2 x i64> %a, %b
  store <2 x i64> %result, <2 x i64> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_test_sub_v4i64:
; SI: v_sub_i32_e32 v{{[0-9]+}}, vcc,
; SI: v_subb_u32_e32 v{{[0-9]+}}, vcc,
; SI: v_sub_i32_e32 v{{[0-9]+}}, vcc,
; SI: v_subb_u32_e32 v{{[0-9]+}}, vcc,
; SI: v_sub_i32_e32 v{{[0-9]+}}, vcc,
; SI: v_subb_u32_e32 v{{[0-9]+}}, vcc,
; SI: v_sub_i32_e32 v{{[0-9]+}}, vcc,
; SI: v_subb_u32_e32 v{{[0-9]+}}, vcc,

; VI: v_sub_u32_e32 v{{[0-9]+}}, vcc,
; VI: v_subb_u32_e32 v{{[0-9]+}}, vcc,
; VI: v_sub_u32_e32 v{{[0-9]+}}, vcc,
; VI: v_subb_u32_e32 v{{[0-9]+}}, vcc,
; VI: v_sub_u32_e32 v{{[0-9]+}}, vcc,
; VI: v_subb_u32_e32 v{{[0-9]+}}, vcc,
; VI: v_sub_u32_e32 v{{[0-9]+}}, vcc,
; VI: v_subb_u32_e32 v{{[0-9]+}}, vcc,

; GFX9: v_sub_co_u32_e32 v{{[0-9]+}}, vcc,
; GFX9: v_subb_co_u32_e32 v{{[0-9]+}}, vcc,
; GFX9: v_sub_co_u32_e32 v{{[0-9]+}}, vcc,
; GFX9: v_subb_co_u32_e32 v{{[0-9]+}}, vcc,
; GFX9: v_sub_co_u32_e32 v{{[0-9]+}}, vcc,
; GFX9: v_subb_co_u32_e32 v{{[0-9]+}}, vcc,
; GFX9: v_sub_co_u32_e32 v{{[0-9]+}}, vcc,
; GFX9: v_subb_co_u32_e32 v{{[0-9]+}}, vcc,
define amdgpu_kernel void @v_test_sub_v4i64(<4 x i64> addrspace(1)* %out, <4 x i64> addrspace(1)* noalias %inA, <4 x i64> addrspace(1)* noalias %inB) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() readnone
  %a_ptr = getelementptr <4 x i64>, <4 x i64> addrspace(1)* %inA, i32 %tid
  %b_ptr = getelementptr <4 x i64>, <4 x i64> addrspace(1)* %inB, i32 %tid
  %a = load <4 x i64>, <4 x i64> addrspace(1)* %a_ptr
  %b = load <4 x i64>, <4 x i64> addrspace(1)* %b_ptr
  %result = sub <4 x i64> %a, %b
  store <4 x i64> %result, <4 x i64> addrspace(1)* %out
  ret void
}

; Make sure the VOP3 form of sub is initially selected. Otherwise pair
; of opies from/to VCC would be necessary

; GCN-LABEL: {{^}}sub_select_vop3:
; SI: v_subrev_i32_e64 v0, s[0:1], s0, v0
; VI: v_subrev_u32_e64 v0, s[0:1], s0, v0
; GFX9: v_subrev_u32_e32 v0, s0, v0

; GCN: ; def vcc
; GCN: ds_write_b32
; GCN: ; use vcc
define amdgpu_ps void @sub_select_vop3(i32 inreg %s, i32 %v) {
  %vcc = call i64 asm sideeffect "; def vcc", "={vcc}"()
  %sub = sub i32 %v, %s
  store i32 %sub, i32 addrspace(3)* undef
  call void asm sideeffect "; use vcc", "{vcc}"(i64 %vcc)
  ret void
}
