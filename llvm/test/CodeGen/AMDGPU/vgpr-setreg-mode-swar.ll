; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 < %s | FileCheck %s

; llvm.set.rounding lowers to s_setreg_b32(MODE), leaving garbage in
; bits[19:12] of the SGPR operand, so the tracked mode must be re-established.

declare void @llvm.set.rounding(i32)

; CHECK-LABEL: {{^}}dynamic_rounding_mode_with_high_vgprs:
; CHECK: s_setreg_b32 hwreg(HW_REG_WAVE_MODE, 0, 4), s0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_set_vgpr_msb 0x4141
define amdgpu_kernel void @dynamic_rounding_mode_with_high_vgprs(ptr addrspace(1) %p, ptr addrspace(1) %q, i32 %mode) #0 {
  %a = load volatile <64 x float>, ptr addrspace(1) %p
  %b = load volatile <64 x float>, ptr addrspace(1) %p
  %c = load volatile <64 x float>, ptr addrspace(1) %p
  %d = load volatile <64 x float>, ptr addrspace(1) %p
  %e = load volatile <64 x float>, ptr addrspace(1) %p
  store volatile <64 x float> %a, ptr addrspace(1) %q
  call void @llvm.set.rounding(i32 %mode)
  %sum = call <64 x float> @llvm.experimental.constrained.fadd.v64f32(<64 x float> %b, <64 x float> %c, metadata !"round.dynamic", metadata !"fpexcept.strict")
  store volatile <64 x float> %sum, ptr addrspace(1) %q
  store volatile <64 x float> %d, ptr addrspace(1) %q
  store volatile <64 x float> %e, ptr addrspace(1) %q
  ret void
}

attributes #0 = { strictfp "amdgpu-flat-work-group-size"="1,32" "amdgpu-waves-per-eu"="1,1" }
