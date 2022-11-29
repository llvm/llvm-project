; RUN: llc -O0 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 < %s | FileCheck %s

; CallGraphAnalysis, which CodeGenSCC order depends on, does not look
; through aliases. If GlobalOpt is never run, we do not see direct
; calls,

@alias3 = hidden alias void (), ptr @aliasee_vgpr256_sgpr102

; CHECK-LABEL: {{^}}kernel3:
; CHECK: .amdhsa_next_free_vgpr 253
; CHECK-NEXT: .amdhsa_next_free_sgpr 33
define amdgpu_kernel void @kernel3() #0 {
bb:
  call void @alias3() #2
  ret void
}

define internal void @aliasee_vgpr256_sgpr102() #1 {
bb:
  call void asm sideeffect "; clobber v252 ", "~{v252}"()
  ret void
}

attributes #0 = { noinline norecurse nounwind optnone }
attributes #1 = { noinline norecurse nounwind readnone willreturn "amdgpu-flat-work-group-size"="1,256" "amdgpu-waves-per-eu"="1,1" }
attributes #2 = { nounwind readnone willreturn }
