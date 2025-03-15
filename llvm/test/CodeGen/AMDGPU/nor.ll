; RUN: llc -mtriple=amdgcn -mcpu=gfx600 -verify-machineinstrs < %s | FileCheck --check-prefixes=W64,GCN %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx700 -verify-machineinstrs < %s | FileCheck --check-prefixes=W64,GCN %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx801 -verify-machineinstrs < %s | FileCheck --check-prefixes=W64,GCN %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck --check-prefixes=W64,GCN %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx1100 -verify-machineinstrs < %s | FileCheck --check-prefixes=W32,GCN %s

; GCN-LABEL: {{^}}scalar_nor_i32_one_use
; GCN: s_nor_b32
define amdgpu_kernel void @scalar_nor_i32_one_use(
    ptr addrspace(1) %r0, i32 %a, i32 %b) {
entry:
  %or = or i32 %a, %b
  %r0.val = xor i32 %or, -1
  store i32 %r0.val, ptr addrspace(1) %r0
  ret void
}

; GCN-LABEL: {{^}}scalar_nor_i32_mul_use
; GCN-NOT: s_nor_b32
; GCN: s_or_b32
; GCN: s_not_b32
; GCN: s_add_i32
define amdgpu_kernel void @scalar_nor_i32_mul_use(
    ptr addrspace(1) %r0, ptr addrspace(1) %r1, i32 %a, i32 %b) {
entry:
  %or = or i32 %a, %b
  %r0.val = xor i32 %or, -1
  %r1.val = add i32 %or, %a
  store i32 %r0.val, ptr addrspace(1) %r0
  store i32 %r1.val, ptr addrspace(1) %r1
  ret void
}

; GCN-LABEL: {{^}}scalar_nor_i64_one_use
; GCN: s_nor_b64
define amdgpu_kernel void @scalar_nor_i64_one_use(
    ptr addrspace(1) %r0, i64 %a, i64 %b) {
entry:
  %or = or i64 %a, %b
  %r0.val = xor i64 %or, -1
  store i64 %r0.val, ptr addrspace(1) %r0
  ret void
}

; GCN-LABEL: {{^}}scalar_nor_i64_mul_use
; GCN-NOT: s_nor_b64
; GCN: s_or_b64
; GCN: s_not_b64
; GCN: s_add_u32
; GCN: s_addc_u32
define amdgpu_kernel void @scalar_nor_i64_mul_use(
    ptr addrspace(1) %r0, ptr addrspace(1) %r1, i64 %a, i64 %b) {
entry:
  %or = or i64 %a, %b
  %r0.val = xor i64 %or, -1
  %r1.val = add i64 %or, %a
  store i64 %r0.val, ptr addrspace(1) %r0
  store i64 %r1.val, ptr addrspace(1) %r1
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

; GCN-LABEL: {{^}}test_nor_in_control_flow

; W32-NOT: s_nor_b64
; W32: s_nor_b32

; W64-NOT: s_nor_b32
; W64: s_nor_b64
define amdgpu_ps void @test_nor_in_control_flow(ptr addrspace(1) %out, i32  %a) {

entry:
  %x = icmp ule i32 %a, 0
  br i1 %x, label %If2, label %MergeCF

If2:
  %y = icmp ule i32 %a, 1
  br label %MergeCF

MergeCF:
  %z = phi i1 [ %x, %entry ], [ %y, %If2 ]
  %or = or i1 %x, %z
  br i1 %or, label %If, label %Else

If:
  %val_A = icmp uge i32 %a, 3
  br label %exit

Else:
  %val_B = icmp ult i32 %a, 4
  br label %exit

exit:
  %phi = phi i1 [ %val_A, %If ], [ %val_B, %Else ]
  store i1 %phi, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}test_or_two_uses
; GCN-NOT: s_nor_b64
; GCN-NOT: s_nor_b32

; W32: s_or_b32
; W32: s_xor_b32

; W64: s_or_b64
; W64: s_xor_b64
define amdgpu_ps void @test_or_two_uses(ptr addrspace(1) %out, i32  %a) {
entry:
  %x = icmp ule i32 %a, 0
  br i1 %x, label %If2, label %MergeCF

If2:
  %y = icmp ule i32 %a, 1
  br label %MergeCF

MergeCF:
  %z = phi i1 [ %x, %entry ], [ %y, %If2 ]
  %or = or i1 %x, %z
  br i1 %or, label %If, label %Else

If:
  %val_A = icmp uge i32 %a, 1
  br label %exit

Else:
  %val_B = icmp ult i32 %a, 4
  br label %exit

exit:
  %phi = phi i1 [ %val_A, %If ], [ %val_B, %Else ]
  %or2 = or i1 %phi, %or
  store i1 %or2, ptr addrspace(1) %out
  ret void
}
