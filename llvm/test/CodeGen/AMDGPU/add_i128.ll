; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}test_i128_vreg:
; GCN: v_add_i32_e32 v[[LO:[0-9]+]], vcc,
; GCN-NEXT: v_addc_u32_e32 v{{[0-9]+}}, vcc, v{{[0-9]+}}, v{{[0-9]+}}, vcc
; GCN-NEXT: v_addc_u32_e32 v{{[0-9]+}}, vcc, v{{[0-9]+}}, v{{[0-9]+}}, vcc
; GCN-NEXT: v_addc_u32_e32 v[[HI:[0-9]+]], vcc, v{{[0-9]+}}, v{{[0-9]+}}, vcc
; GCN: buffer_store_dwordx4 v{{\[}}[[LO]]:[[HI]]],
define amdgpu_kernel void @test_i128_vreg(i128 addrspace(1)* noalias %out, i128 addrspace(1)* noalias %inA, i128 addrspace(1)* noalias %inB) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() readnone
  %a_ptr = getelementptr i128, i128 addrspace(1)* %inA, i32 %tid
  %b_ptr = getelementptr i128, i128 addrspace(1)* %inB, i32 %tid
  %a = load i128, i128 addrspace(1)* %a_ptr
  %b = load i128, i128 addrspace(1)* %b_ptr
  %result = add i128 %a, %b
  store i128 %result, i128 addrspace(1)* %out
  ret void
}

; Check that the SGPR add operand is correctly moved to a VGPR.
; GCN-LABEL: {{^}}sgpr_operand:
; GCN: s_add_u32
; GCN: s_addc_u32
; GCN: s_addc_u32
; GCN: s_addc_u32
define amdgpu_kernel void @sgpr_operand(i128 addrspace(1)* noalias %out, i128 addrspace(1)* noalias %in, i128 %a) {
  %foo = load i128, i128 addrspace(1)* %in, align 8
  %result = add i128 %foo, %a
  store i128 %result, i128 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}sgpr_operand_reversed:
; GCN: s_add_u32
; GCN: s_addc_u32
; GCN: s_addc_u32
; GCN: s_addc_u32
define amdgpu_kernel void @sgpr_operand_reversed(i128 addrspace(1)* noalias %out, i128 addrspace(1)* noalias %in, i128 %a) {
  %foo = load i128, i128 addrspace(1)* %in, align 8
  %result = add i128 %a, %foo
  store i128 %result, i128 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_sreg:
; GCN: s_add_u32
; GCN: s_addc_u32
; GCN: s_addc_u32
; GCN: s_addc_u32
define amdgpu_kernel void @test_sreg(i128 addrspace(1)* noalias %out, i128 %a, i128 %b) {
  %result = add i128 %a, %b
  store i128 %result, i128 addrspace(1)* %out
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() readnone
