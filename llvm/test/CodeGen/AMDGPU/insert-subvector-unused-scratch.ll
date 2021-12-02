; RUN: llc -mtriple amdgcn-amd-- -mcpu=bonaire -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; Before the fix that this test was committed with, this code would leave
; an unused stack slot, causing ScratchSize to be non-zero.

; GCN-LABEL: store_v3i32:
; GCN:        ds_read_b96
; GCN:        ds_write_b96
; GCN: ScratchSize: 0
define amdgpu_kernel void @store_v3i32(<3 x i32> addrspace(3)* %out, <3 x i32> %a) nounwind {
  %val = load <3 x i32>, <3 x i32> addrspace(3)* %out
  %val.1 = add <3 x i32> %a, %val
  store <3 x i32> %val.1, <3 x i32> addrspace(3)* %out, align 16
  ret void
}

; GCN-LABEL: store_v5i32:
; GCN:        ds_read_b128
; GCN:        ds_read_b32
; GCN:        ds_write_b32
; GCN:        ds_write_b128
; GCN: ScratchSize: 0
define amdgpu_kernel void @store_v5i32(<5 x i32> addrspace(3)* %out, <5 x i32> %a) nounwind {
  %val = load <5 x i32>, <5 x i32> addrspace(3)* %out
  %val.1 = add <5 x i32> %a, %val
  store <5 x i32> %val.1, <5 x i32> addrspace(3)* %out, align 16
  ret void
}
