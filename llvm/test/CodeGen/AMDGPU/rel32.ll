; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck %s -check-prefixes=CHECK,GFX910
; RUN: llc -march=amdgcn -mcpu=gfx1010 -verify-machineinstrs < %s | FileCheck %s -check-prefixes=CHECK,GFX910
; RUN: llc -march=amdgcn -mcpu=gfx1100 -verify-machineinstrs < %s | FileCheck %s -check-prefixes=CHECK,GFX11

@g = protected local_unnamed_addr addrspace(4) externally_initialized global i32 0, align 4

; CHECK-LABEL: rel32_neg_offset:
; CHECK: s_getpc_b64 s[[[LO:[0-9]+]]:[[HI:[0-9]+]]]
; GFX910-NEXT: s_add_u32 s[[LO]], s[[LO]], g@rel32@lo-4
; GFX910-NEXT: s_addc_u32 s[[HI]], s[[HI]], g@rel32@hi+4
; GFX11-NEXT: s_delay_alu
; GFX11-NEXT: s_add_u32 s[[LO]], s[[LO]], g@rel32@lo
; GFX11-NEXT: s_addc_u32 s[[HI]], s[[HI]], g@rel32@hi+8
define ptr addrspace(4) @rel32_neg_offset() {
  %r = getelementptr i32, ptr addrspace(4) @g, i64 -2
  ret ptr addrspace(4) %r
}
