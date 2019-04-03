; RUN: llc -march=amdgcn -mcpu=gfx900 < %s | FileCheck -check-prefixes=GCN,NO-ECC %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -mattr=+sram-ecc < %s | FileCheck -check-prefixes=GCN,NO-ECC %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -mattr=-sram-ecc < %s | FileCheck -check-prefixes=GCN,NO-ECC %s
; RUN: llc -march=amdgcn -mcpu=gfx902 -mattr=+sram-ecc < %s | FileCheck -check-prefixes=GCN,NO-ECC %s
; RUN: llc -march=amdgcn -mcpu=gfx904 -mattr=+sram-ecc < %s | FileCheck -check-prefixes=GCN,NO-ECC %s
; RUN: llc -march=amdgcn -mcpu=gfx906 -mattr=+sram-ecc < %s | FileCheck -check-prefixes=GCN,ECC %s
; RUN: llc -march=amdgcn -mcpu=gfx906 -mattr=-sram-ecc < %s | FileCheck -check-prefixes=GCN,NO-ECC %s

; Make sure the correct set of targets are marked with
; FeatureDoesNotSupportSRAMECC, and +sram-ecc is ignored if it's never
; supported.

; GCN-LABEL: {{^}}load_global_hi_v2i16_reglo_vreg:
; NO-ECC: global_load_short_d16_hi
; ECC: global_load_ushort
define void @load_global_hi_v2i16_reglo_vreg(i16 addrspace(1)* %in, i16 %reg) {
entry:
  %gep = getelementptr inbounds i16, i16 addrspace(1)* %in, i64 -2047
  %load = load i16, i16 addrspace(1)* %gep
  %build0 = insertelement <2 x i16> undef, i16 %reg, i32 0
  %build1 = insertelement <2 x i16> %build0, i16 %load, i32 1
  store <2 x i16> %build1, <2 x i16> addrspace(1)* undef
  ret void
}
