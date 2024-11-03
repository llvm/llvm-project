; RUN: llc -O0 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 < %s | FileCheck %s

; CallGraphAnalysis, which CodeGenSCC order depends on, does not look
; through aliases. If GlobalOpt is never run, we do not see direct
; calls,

@alias0 = hidden alias void (), ptr @aliasee_default_vgpr64_sgpr102

; CHECK-LABEL: {{^}}kernel0:
; CHECK: .amdhsa_next_free_vgpr 53
; CHECK-NEXT: .amdhsa_next_free_sgpr 36
define amdgpu_kernel void @kernel0() #0 {
bb:
  call void @alias0() #2
  ret void
}

define internal void @aliasee_default_vgpr64_sgpr102() #1 {
bb:
  call void asm sideeffect "; clobber v52 ", "~{v52}"()
  ret void
}

attributes #0 = { noinline norecurse nounwind optnone }
attributes #1 = { noinline norecurse nounwind readnone willreturn }
attributes #2 = { nounwind readnone willreturn }
