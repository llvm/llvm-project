; RUN: llc -mtriple=amdgpu12.00--amdpal < %s | FileCheck --check-prefixes=CHECK,EXTSTACK0 %s
; RUN: llc -mtriple=amdgpu12.00--amdpal -amdgpu-assume-external-call-stack-size=48 < %s | FileCheck --check-prefixes=CHECK,EXTSTACK48 %s

; The dynamic VGPR variant of the chain pseudo is modelled as an indirect call,
; so neither the chain callee nor the chain_preserve fallback contributes to the
; caller's stack size.

; CHECK-LABEL: {{^}}dvgpr_callee:
; CHECK:         .set .Ldvgpr_callee.private_seg_size, 12{{$}}

; CHECK-LABEL: {{^}}dvgpr_preserve_fallback:
; CHECK:         .set .Ldvgpr_preserve_fallback.private_seg_size, 52{{$}}

; CHECK-LABEL: {{^}}chain_dvgpr:
; CHECK:         .set .Lchain_dvgpr.private_seg_size, 40{{$}}
; CHECK:         .set .Lchain_dvgpr.has_indirect_call, 1{{$}}

; A non-chain indirect call still contributes the assumed external stack size.
; CHECK-LABEL: {{^}}chain_dvgpr_and_indirect_gfx:
; EXTSTACK0:     .set .Lchain_dvgpr_and_indirect_gfx.private_seg_size, 32{{$}}
; EXTSTACK48:    .set .Lchain_dvgpr_and_indirect_gfx.private_seg_size, 32+max(48){{$}}

define amdgpu_cs_chain void @dvgpr_callee(<40 x i32> %v) #0 {
.entry:
  %a = alloca [2 x i32], addrspace(5)
  store volatile i32 1, ptr addrspace(5) %a
  ret void
}

define amdgpu_cs_chain_preserve void @dvgpr_preserve_fallback() #0 {
.entry:
  %a = alloca [12 x i32], addrspace(5)
  store volatile i32 2, ptr addrspace(5) %a
  ret void
}

define amdgpu_cs_chain void @chain_dvgpr(<40 x i32> %v) #0 {
.entry:
  %a = alloca [9 x i32], addrspace(5)
  store volatile i32 3, ptr addrspace(5) %a
  call void (ptr, i32, {}, <40 x i32>, i32, ...) @llvm.amdgcn.cs.chain.p0.i32.s.v40i32(ptr @dvgpr_callee, i32 -1, {} inreg poison, <40 x i32> %v, i32 1, i32 inreg 64, i32 inreg -1, ptr @dvgpr_preserve_fallback)
  unreachable
}

define amdgpu_cs_chain void @chain_dvgpr_and_indirect_gfx(ptr inreg %fp, <40 x i32> %v) #0 {
.entry:
  %a = alloca [5 x i32], addrspace(5)
  store volatile i32 4, ptr addrspace(5) %a
  call amdgpu_gfx void %fp(ptr addrspace(5) %a)
  call void (ptr, i32, {}, <40 x i32>, i32, ...) @llvm.amdgcn.cs.chain.p0.i32.s.v40i32(ptr @dvgpr_callee, i32 -1, {} inreg poison, <40 x i32> %v, i32 1, i32 inreg 64, i32 inreg -1, ptr @dvgpr_preserve_fallback)
  unreachable
}

attributes #0 = { nounwind "amdgpu-dynamic-vgpr-block-size"="16" }
