; RUN: llc -march=amdgcn -mcpu=gfx600 -verify-machineinstrs < %s | FileCheck --check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=gfx700 -verify-machineinstrs < %s | FileCheck --check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=gfx801 -verify-machineinstrs < %s | FileCheck --check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck --check-prefix=GCN %s

; GCN-LABEL: {{^}}scalar_nand_i32_one_use
; GCN: s_nand_b32
define amdgpu_kernel void @scalar_nand_i32_one_use(
    ptr addrspace(1) %r0, i32 %a, i32 %b) {
entry:
  %and = and i32 %a, %b
  %r0.val = xor i32 %and, -1
  store i32 %r0.val, ptr addrspace(1) %r0
  ret void
}

; GCN-LABEL: {{^}}scalar_nand_i32_mul_use
; GCN-NOT: s_nand_b32
; GCN: s_and_b32
; GCN: s_not_b32
; GCN: s_add_i32
define amdgpu_kernel void @scalar_nand_i32_mul_use(
    ptr addrspace(1) %r0, ptr addrspace(1) %r1, i32 %a, i32 %b) {
entry:
  %and = and i32 %a, %b
  %r0.val = xor i32 %and, -1
  %r1.val = add i32 %and, %a
  store i32 %r0.val, ptr addrspace(1) %r0
  store i32 %r1.val, ptr addrspace(1) %r1
  ret void
}

; GCN-LABEL: {{^}}scalar_nand_i64_one_use
; GCN: s_nand_b64
define amdgpu_kernel void @scalar_nand_i64_one_use(
    ptr addrspace(1) %r0, i64 %a, i64 %b) {
entry:
  %and = and i64 %a, %b
  %r0.val = xor i64 %and, -1
  store i64 %r0.val, ptr addrspace(1) %r0
  ret void
}

; GCN-LABEL: {{^}}scalar_nand_i64_mul_use
; GCN-NOT: s_nand_b64
; GCN: s_and_b64
; GCN: s_not_b64
; GCN: s_add_u32
; GCN: s_addc_u32
define amdgpu_kernel void @scalar_nand_i64_mul_use(
    ptr addrspace(1) %r0, ptr addrspace(1) %r1, i64 %a, i64 %b) {
entry:
  %and = and i64 %a, %b
  %r0.val = xor i64 %and, -1
  %r1.val = add i64 %and, %a
  store i64 %r0.val, ptr addrspace(1) %r0
  store i64 %r1.val, ptr addrspace(1) %r1
  ret void
}

; GCN-LABEL: {{^}}vector_nand_i32_one_use
; GCN-NOT: s_nand_b32
; GCN: v_and_b32
; GCN: v_not_b32
define i32 @vector_nand_i32_one_use(i32 %a, i32 %b) {
entry:
  %and = and i32 %a, %b
  %r = xor i32 %and, -1
  ret i32 %r
}

; GCN-LABEL: {{^}}vector_nand_i64_one_use
; GCN-NOT: s_nand_b64
; GCN: v_and_b32
; GCN: v_and_b32
; GCN: v_not_b32
; GCN: v_not_b32
define i64 @vector_nand_i64_one_use(i64 %a, i64 %b) {
entry:
  %and = and i64 %a, %b
  %r = xor i64 %and, -1
  ret i64 %r
}
