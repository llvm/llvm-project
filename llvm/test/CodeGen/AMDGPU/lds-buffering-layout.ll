; NOTE: Do not autogenerate. This checks the final LDS layout, not instructions.
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -amdgpu-enable-lds-buffering < %s | FileCheck %s

target triple = "amdgcn-amd-amdhsa"

@used = internal addrspace(3) global [130500 x i8] poison, align 4

; The backend lays out LDS globals created after module LDS lowering in inverse
; use order. Include worst-case leading padding when reserving each slot so
; this does not exceed the gfx950 LDS limit during code generation.
; CHECK: .amdhsa_group_segment_fixed_size 131536
define amdgpu_kernel void @layout_order(ptr addrspace(1) %high,
                                        ptr addrspace(1) %low) #0 {
  %used = load volatile i8, ptr addrspace(3) @used, align 1
  %high.value = load i8, ptr addrspace(1) %high, align 131072
  store i8 %high.value, ptr addrspace(1) %high, align 131072
  %low.value = load i8, ptr addrspace(1) %low, align 16
  store i8 %low.value, ptr addrspace(1) %low, align 16
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="1024,1024" }
