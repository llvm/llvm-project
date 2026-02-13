; RUN: llc -mtriple=amdgcn--amdpal < %s | FileCheck -check-prefixes=GCN,GCN-PAL %s
; RUN: llc -mtriple=amdgcn-- -mcpu=kaveri < %s | FileCheck -check-prefixes=GCN,GCN-DEFAULT %s
; RUN: llc -mtriple=amdgcn-mesa-mesa3d -mcpu=kaveri < %s | FileCheck -check-prefixes=GCN,GCN-MESA %s
; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=kaveri < %s | FileCheck -check-prefixes=GCN,GCN-DEFAULT %s
; RUN: llc -mtriple=r600-- -mcpu=cypress < %s | FileCheck -check-prefix=R600 %s

@private1 = private unnamed_addr addrspace(4) constant [4 x float] [float 0.0, float 1.0, float 2.0, float 3.0]
@private2 = private unnamed_addr addrspace(4) constant [4 x float] [float 4.0, float 5.0, float 6.0, float 7.0]
@available_externally = available_externally addrspace(4) global [256 x i32] zeroinitializer

; GCN-LABEL: {{^}}private_test:

; Non-R600 OSes use relocations.
; GCN-DEFAULT: s_getpc_b64 s[[[PC0_LO:[0-9]+]]:[[PC0_HI:[0-9]+]]]
; GCN-DEFAULT: s_add_u32 s{{[0-9]+}}, s[[PC0_LO]], .Lprivate1@rel32@lo+4
; GCN-DEFAULT: s_addc_u32 s{{[0-9]+}}, s[[PC0_HI]], .Lprivate1@rel32@hi+12
; GCN-DEFAULT: s_getpc_b64 s[[[PC1_LO:[0-9]+]]:[[PC1_HI:[0-9]+]]]
; GCN-DEFAULT: s_add_u32 s{{[0-9]+}}, s[[PC1_LO]], .Lprivate2@rel32@lo+4
; GCN-DEFAULT: s_addc_u32 s{{[0-9]+}}, s[[PC1_HI]], .Lprivate2@rel32@hi+12

; MESA uses absolute relocations.
; GCN-MESA: s_add_u32 s2, .Lprivate1@abs32@lo, s4
; GCN-MESA: s_addc_u32 s3, .Lprivate1@abs32@hi, s5

; PAL uses absolute relocations.
; GCN-PAL:    s_add_u32 s2, .Lprivate1@abs32@lo, s4
; GCN-PAL:    s_addc_u32 s3, .Lprivate1@abs32@hi, s5
; GCN-PAL:    s_add_u32 s4, .Lprivate2@abs32@lo, s4
; GCN-PAL:    s_addc_u32 s5, .Lprivate2@abs32@hi, s5

; R600-LABEL: private_test
define amdgpu_kernel void @private_test(i32 %index, ptr addrspace(1) %out) {
  %ptr = getelementptr [4 x float], ptr addrspace(4) @private1, i32 0, i32 %index
  %val = load float, ptr addrspace(4) %ptr
  store volatile float %val, ptr addrspace(1) %out
  %ptr2 = getelementptr [4 x float], ptr addrspace(4) @private2, i32 0, i32 %index
  %val2 = load float, ptr addrspace(4) %ptr2
  store volatile float %val2, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}available_externally_test:
; GCN-DEFAULT: s_getpc_b64 s[[[PC0_LO:[0-9]+]]:[[PC0_HI:[0-9]+]]]
; GCN-DEFAULT: s_add_u32 s{{[0-9]+}}, s[[PC0_LO]], available_externally@gotpcrel32@lo+4
; GCN-DEFAULT: s_addc_u32 s{{[0-9]+}}, s[[PC0_HI]], available_externally@gotpcrel32@hi+12

; GCN-MESA:    s_mov_b32 s1, available_externally@abs32@hi
; GCN-MESA:    s_mov_b32 s0, available_externally@abs32@lo

; R600-LABEL: available_externally_test

; GCN-PAL:    s_mov_b32 s1, available_externally@abs32@hi
; GCN-PAL:    s_mov_b32 s0, available_externally@abs32@lo
define amdgpu_kernel void @available_externally_test(ptr addrspace(1) %out) {
  %ptr = getelementptr [256 x i32], ptr addrspace(4) @available_externally, i32 0, i32 1
  %val = load i32, ptr addrspace(4) %ptr
  store i32 %val, ptr addrspace(1) %out
  ret void
}

; GCN: .section .rodata
; R600: .text

; GCN: private1:
; GCN: private2:
; R600: private1:
; R600: private2:
