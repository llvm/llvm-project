; Flag is ignored on targets without sramecc support (like gfx900)
; RUN: llc -mtriple=amdgpu9.00 < %s | FileCheck -check-prefixes=GCN,NO-ECC %s
; RUN: llc -mtriple=amdgpu9.00 -amdgpu-sramecc=1 < %s | FileCheck -check-prefixes=GCN,NO-ECC %s
; RUN: llc -mtriple=amdgpu9.00 -amdgpu-sramecc=0 < %s | FileCheck -check-prefixes=GCN,NO-ECC %s

; RUN: llc -mtriple=amdgpu9.06 -amdgpu-sramecc=1 < %s | FileCheck -check-prefixes=GCN,ECC %s
; RUN: llc -mtriple=amdgpu9.06 -amdgpu-sramecc=0 < %s | FileCheck -check-prefixes=GCN,NO-ECC %s
; RUN: llc -mtriple=amdgpu12.50 < %s | FileCheck -check-prefixes=GCN,ECC %s

; Make sure the correct set of targets are marked with
; FeatureDoesNotSupportSRAMECC, and +sramecc is ignored if it's never
; supported.

; GCN-LABEL: {{^}}load_global_hi_v2i16_reglo_vreg:
; NO-ECC: global_load_short_d16_hi
; ECC: global_load_{{ushort|u16}}
define void @load_global_hi_v2i16_reglo_vreg(ptr addrspace(1) %in, i16 %reg) {
entry:
  %gep = getelementptr inbounds i16, ptr addrspace(1) %in, i64 -2047
  %load = load i16, ptr addrspace(1) %gep
  %build0 = insertelement <2 x i16> poison, i16 %reg, i32 0
  %build1 = insertelement <2 x i16> %build0, i16 %load, i32 1
  store <2 x i16> %build1, ptr addrspace(1) poison
  ret void
}
