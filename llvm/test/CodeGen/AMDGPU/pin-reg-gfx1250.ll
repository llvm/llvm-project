; RUN: llc -mtriple=amdgcn -mcpu=gfx1250 -verify-machineinstrs < %s | FileCheck -check-prefixes=CHECK %s

; gfx1250 has 1024 addressable VGPRs. A pin to a VGPR index >= 256 must be
; honored: the value is placed in the requested high tuple (reachable via the
; S_SET_VGPR_MSB addressing mode) and the VGPR count / occupancy cap grows to
; cover the pinned range.

declare <4 x float> @llvm.amdgcn.pin.vgpr.v4f32(<4 x float>, i32 immarg)

; CHECK-LABEL: {{^}}pin_high_vgpr:
; CHECK: s_set_vgpr_msb
; CHECK: v[{{[0-9:]+}}] /*v[300:303]*/
; CHECK: .set .Lpin_high_vgpr.num_vgpr, 304
define amdgpu_kernel void @pin_high_vgpr(ptr addrspace(1) %p) {
  %v = load <4 x float>, ptr addrspace(1) %p
  %pv = call <4 x float> @llvm.amdgcn.pin.vgpr.v4f32(<4 x float> %v, i32 300)
  store <4 x float> %pv, ptr addrspace(1) %p
  ret void
}
