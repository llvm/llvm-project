; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=NOHSA %s
; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=kaveri -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=HSA %s

@private1 = private unnamed_addr addrspace(2) constant [4 x float] [float 0.0, float 1.0, float 2.0, float 3.0]
@private2 = private unnamed_addr addrspace(2) constant [4 x float] [float 4.0, float 5.0, float 6.0, float 7.0]
@available_externally = available_externally addrspace(2) global [256 x i32] zeroinitializer

; GCN-LABEL: {{^}}private_test:
; GCN: s_getpc_b64 s{{\[}}[[PC0_LO:[0-9]+]]:[[PC0_HI:[0-9]+]]{{\]}}

; Non-HSA OSes use fixup into .text section.
; NOHSA: s_add_u32 s{{[0-9]+}}, s[[PC0_LO]], private1
; NOHSA: s_addc_u32 s{{[0-9]+}}, s[[PC0_HI]], 0

; HSA OSes use relocations.
; HSA: s_add_u32 s{{[0-9]+}}, s[[PC0_LO]], private1@rel32@lo+4
; HSA: s_addc_u32 s{{[0-9]+}}, s[[PC0_HI]], private1@rel32@hi+4

; GCN: s_getpc_b64 s{{\[}}[[PC1_LO:[0-9]+]]:[[PC1_HI:[0-9]+]]{{\]}}

; Non-HSA OSes use fixup into .text section.
; NOHSA: s_add_u32 s{{[0-9]+}}, s[[PC1_LO]], private2
; NOHSA: s_addc_u32 s{{[0-9]+}}, s[[PC1_HI]], 0

; HSA OSes use relocations.
; HSA: s_add_u32 s{{[0-9]+}}, s[[PC1_LO]], private2@rel32@lo+4
; HSA: s_addc_u32 s{{[0-9]+}}, s[[PC1_HI]], private2@rel32@hi+4

define amdgpu_kernel void @private_test(i32 %index, float addrspace(1)* %out) {
  %ptr = getelementptr [4 x float], [4 x float] addrspace(2) * @private1, i32 0, i32 %index
  %val = load float, float addrspace(2)* %ptr
  store volatile float %val, float addrspace(1)* %out
  %ptr2 = getelementptr [4 x float], [4 x float] addrspace(2) * @private2, i32 0, i32 %index
  %val2 = load float, float addrspace(2)* %ptr2
  store volatile float %val2, float addrspace(1)* %out
  ret void
}

; HSA-LABEL: {{^}}available_externally_test:
; HSA: s_getpc_b64 s{{\[}}[[PC0_LO:[0-9]+]]:[[PC0_HI:[0-9]+]]{{\]}}
; HSA: s_add_u32 s{{[0-9]+}}, s[[PC0_LO]], available_externally@gotpcrel32@lo+4
; HSA: s_addc_u32 s{{[0-9]+}}, s[[PC0_HI]], available_externally@gotpcrel32@hi+4
define amdgpu_kernel void @available_externally_test(i32 addrspace(1)* %out) {
  %ptr = getelementptr [256 x i32], [256 x i32] addrspace(2)* @available_externally, i32 0, i32 1
  %val = load i32, i32 addrspace(2)* %ptr
  store i32 %val, i32 addrspace(1)* %out
  ret void
}

; NOHSA: .text
; HSA: .section .rodata

; GCN: private1:
; GCN: private2:
