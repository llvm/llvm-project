; RUN: llc -march=amdgcn -mcpu=gfx600 -verify-machineinstrs < %s | FileCheck --check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=gfx700 -verify-machineinstrs < %s | FileCheck --check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=gfx801 -verify-machineinstrs < %s | FileCheck --check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck --check-prefix=GCN %s

; GCN-LABEL: {{^}}scalar_nor_i32_one_use
; GCN: s_nor_b32
define amdgpu_kernel void @scalar_nor_i32_one_use(
    i32 addrspace(1)* %r0, i32 %a, i32 %b) {
entry:
  %or = or i32 %a, %b
  %r0.val = xor i32 %or, -1
  store i32 %r0.val, i32 addrspace(1)* %r0
  ret void
}

; GCN-LABEL: {{^}}scalar_nor_i32_mul_use
; GCN-NOT: s_nor_b32
; GCN: s_or_b32
; GCN: s_not_b32
; GCN: s_add_i32
define amdgpu_kernel void @scalar_nor_i32_mul_use(
    i32 addrspace(1)* %r0, i32 addrspace(1)* %r1, i32 %a, i32 %b) {
entry:
  %or = or i32 %a, %b
  %r0.val = xor i32 %or, -1
  %r1.val = add i32 %or, %a
  store i32 %r0.val, i32 addrspace(1)* %r0
  store i32 %r1.val, i32 addrspace(1)* %r1
  ret void
}

; GCN-LABEL: {{^}}scalar_nor_i64_one_use
; GCN: s_nor_b64
define amdgpu_kernel void @scalar_nor_i64_one_use(
    i64 addrspace(1)* %r0, i64 %a, i64 %b) {
entry:
  %or = or i64 %a, %b
  %r0.val = xor i64 %or, -1
  store i64 %r0.val, i64 addrspace(1)* %r0
  ret void
}

; GCN-LABEL: {{^}}scalar_nor_i64_mul_use
; GCN-NOT: s_nor_b64
; GCN: s_or_b64
; GCN: s_not_b64
; GCN: s_add_u32
; GCN: s_addc_u32
define amdgpu_kernel void @scalar_nor_i64_mul_use(
    i64 addrspace(1)* %r0, i64 addrspace(1)* %r1, i64 %a, i64 %b) {
entry:
  %or = or i64 %a, %b
  %r0.val = xor i64 %or, -1
  %r1.val = add i64 %or, %a
  store i64 %r0.val, i64 addrspace(1)* %r0
  store i64 %r1.val, i64 addrspace(1)* %r1
  ret void
}

; GCN-LABEL: {{^}}vector_nor_i32_one_use
; GCN-NOT: s_nor_b32
; GCN: v_or_b32
; GCN: v_not_b32
define i32 @vector_nor_i32_one_use(i32 %a, i32 %b) {
entry:
  %or = or i32 %a, %b
  %r = xor i32 %or, -1
  ret i32 %r
}

; GCN-LABEL: {{^}}vector_nor_i64_one_use
; GCN-NOT: s_nor_b64
; GCN: v_or_b32
; GCN: v_or_b32
; GCN: v_not_b32
; GCN: v_not_b32
define i64 @vector_nor_i64_one_use(i64 %a, i64 %b) {
entry:
  %or = or i64 %a, %b
  %r = xor i64 %or, -1
  ret i64 %r
}
