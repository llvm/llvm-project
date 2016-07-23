; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=NOHSA %s
; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=kaveri -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=HSA %s

@readonly = private unnamed_addr addrspace(2) constant [4 x float] [float 0.0, float 1.0, float 2.0, float 3.0]
@readonly2 = private unnamed_addr addrspace(2) constant [4 x float] [float 4.0, float 5.0, float 6.0, float 7.0]

; GCN-LABEL: {{^}}main:
; GCN: s_getpc_b64 s{{\[}}[[PC0_LO:[0-9]+]]:[[PC0_HI:[0-9]+]]{{\]}}
; GCN-NEXT: s_add_u32 s{{[0-9]+}}, s[[PC0_LO]], readonly
; GCN: s_addc_u32 s{{[0-9]+}}, s[[PC0_HI]], 0
; GCN: s_getpc_b64 s{{\[}}[[PC1_LO:[0-9]+]]:[[PC1_HI:[0-9]+]]{{\]}}
; GCN-NEXT: s_add_u32 s{{[0-9]+}}, s[[PC1_LO]], readonly
; GCN: s_addc_u32 s{{[0-9]+}}, s[[PC1_HI]], 0
; NOHSA: .text
; HSA: .text
; GCN: readonly:
; GCN: readonly2:
define void @main(i32 %index, float addrspace(1)* %out) {
  %ptr = getelementptr [4 x float], [4 x float] addrspace(2) * @readonly, i32 0, i32 %index
  %val = load float, float addrspace(2)* %ptr
  store float %val, float addrspace(1)* %out
  %ptr2 = getelementptr [4 x float], [4 x float] addrspace(2) * @readonly2, i32 0, i32 %index
  %val2 = load float, float addrspace(2)* %ptr2
  store float %val2, float addrspace(1)* %out
  ret void
}

