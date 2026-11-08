; RUN: llc -mtriple=amdgpu6.00 < %s | FileCheck --check-prefix=GCN %s
; RUN: llc -mtriple=amdgpu7.00 < %s | FileCheck --check-prefix=GCN %s
; RUN: llc -mtriple=amdgpu8.01 < %s | FileCheck --check-prefix=GCN %s
; RUN: llc -mtriple=amdgpu9.00 < %s | FileCheck --check-prefix=GCN %s

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

; GCN-LABEL: {{^}}scalar_nand_v2i16_one_use
; GCN: s_nand_b32
define amdgpu_kernel void @scalar_nand_v2i16_one_use(
    ptr addrspace(1) %r0, <2 x i16> %a, <2 x i16> %b) {
entry:
  %and = and <2 x i16> %a, %b
  %r0.val = xor <2 x i16> %and, <i16 -1, i16 -1>
  store <2 x i16> %r0.val, ptr addrspace(1) %r0
  ret void
}

; GCN-LABEL: {{^}}scalar_nand_v4i16_one_use
; GCN: s_nand_b64
define amdgpu_kernel void @scalar_nand_v4i16_one_use(
    ptr addrspace(1) %r0, <4 x i16> %a, <4 x i16> %b) {
entry:
  %and = and <4 x i16> %a, %b
  %r0.val = xor <4 x i16> %and, <i16 -1, i16 -1, i16 -1, i16 -1>
  store <4 x i16> %r0.val, ptr addrspace(1) %r0
  ret void
}

; GCN-LABEL: {{^}}scalar_nand_v2i32_one_use
; GCN: s_nand_b64
define amdgpu_kernel void @scalar_nand_v2i32_one_use(
    ptr addrspace(1) %r0, <2 x i32> %a, <2 x i32> %b) {
entry:
  %and = and <2 x i32> %a, %b
  %r0.val = xor <2 x i32> %and, <i32 -1, i32 -1>
  store <2 x i32> %r0.val, ptr addrspace(1) %r0
  ret void
}

; GCN-LABEL: {{^}}scalar_nand_v4i16_mul_use
; GCN-NOT: s_nand_b64
define amdgpu_kernel void @scalar_nand_v4i16_mul_use(
    ptr addrspace(1) %r0, ptr addrspace(1) %r1, <4 x i16> %a, <4 x i16> %b) {
entry:
  %and = and <4 x i16> %a, %b
  %r0.val = xor <4 x i16> %and, <i16 -1, i16 -1, i16 -1, i16 -1>
  store <4 x i16> %r0.val, ptr addrspace(1) %r0
  store <4 x i16> %and, ptr addrspace(1) %r1
  ret void
}
