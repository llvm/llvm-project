; RUN: llc -O0 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 < %s | FileCheck %s

; CallGraphAnalysis, which CodeGenSCC order depends on, does not look
; through aliases. If GlobalOpt is never run, we do not see direct
; calls,

@alias1 = hidden alias void (), void ()* @aliasee_vgpr32_sgpr76

; The parent kernel has a higher VGPR usage than the possible callees.

; CHECK-LABEL: {{^}}kernel1:
; CHECK: .amdhsa_next_free_vgpr 42
; CHECK-NEXT: .amdhsa_next_free_sgpr 33
define amdgpu_kernel void @kernel1() #0 {
bb:
  call void asm sideeffect "; clobber v40 ", "~{v40}"()
  call void @alias1() #2
  ret void
}

define internal void @aliasee_vgpr32_sgpr76() #1 {
bb:
  call void asm sideeffect "; clobber v26 ", "~{v26}"()
  ret void
}

attributes #0 = { noinline norecurse nounwind optnone }
attributes #1 = { noinline norecurse nounwind readnone willreturn "amdgpu-waves-per-eu"="8,10" }
attributes #2 = { nounwind readnone willreturn }
