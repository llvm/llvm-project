; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1010 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX10 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX11 %s

; GCN: foo1:
; v_cndmask_b32_e64 v0, 0, 1, vcc_lo{{$}}
; GCN: kernel1:
; GCN: s_getpc_b64
; GFX10-NEXT: foo1@gotpcrel32@lo+4
; GFX10-NEXT: foo1@gotpcrel32@hi+12
; GFX11-NEXT: s_delay_alu
; GFX11-NEXT: foo1@gotpcrel32@lo+8
; GFX11-NEXT: foo1@gotpcrel32@hi+16

define void @foo1(i32 %x) #1 {
entry:
  %cc = icmp eq i32 %x, 0
  store volatile i1 %cc, ptr undef
  ret void
}

define amdgpu_kernel void @kernel1(float %x) #0 {
entry:
  call void @foo1(float %x)
  ret void
}

attributes #0 = { nounwind "target-features"="+wavefrontsize32" }
attributes #1 = { noinline nounwind "target-features"="+wavefrontsize64" }
