; Test generation of _dvgpr$ symbol for an amdgpu_cs_chain function with +dynamic-vgpr.

; RUN: llc -mtriple=amdgcn-amd-amdpal -mcpu=gfx1200 -asm-verbose=0 < %s | FileCheck -check-prefixes=DVGPR %s

; DVGPR-LABEL: func:
; DVGPR: .Ltmp0:
; DVGPR: .set _dvgpr$func, .Ltmp0+{{[0-9]+}}

define amdgpu_cs_chain void @func() #0 {
  ret void
}
attributes #0 = { "target-features"="+dynamic-vgpr" }
