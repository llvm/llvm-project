; RUN: llc -O0 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 < %s | FileCheck %s

; CallGraphAnalysis, which CodeGenSCC order depends on, does not look
; through aliases. If GlobalOpt is never run, we do not see direct
; calls,

@alias3 = hidden alias void (), ptr @aliasee_vgpr256_sgpr102

; CHECK-LABEL: {{^}}kernel3:
; CHECK: .amdhsa_next_free_vgpr 253
; CHECK-NEXT: .amdhsa_next_free_sgpr 33
define amdgpu_kernel void @kernel3() noinline norecurse nounwind optnone {
bb:
  call void @alias3() nounwind readnone willreturn
  ret void
}

define internal void @aliasee_vgpr256_sgpr102() noinline norecurse nounwind readnone willreturn "amdgpu-flat-work-group-size"="1,256" "amdgpu-waves-per-eu"="1,1" {
bb:
  call void asm sideeffect "; clobber v252 ", "~{v252}"()
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 500}
