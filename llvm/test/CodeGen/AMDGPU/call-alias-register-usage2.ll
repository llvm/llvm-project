; RUN: llc -O0 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 < %s | FileCheck %s

; CallGraphAnalysis, which CodeGenSCC order depends on, does not look
; through aliases. If GlobalOpt is never run, we do not see direct
; calls,

@alias2 = hidden alias void (), ptr @aliasee_vgpr64_sgpr102

; CHECK-LABEL: {{^}}kernel2:
; CHECK: .amdhsa_next_free_vgpr 53
; CHECK-NEXT: .amdhsa_next_free_sgpr 33
define amdgpu_kernel void @kernel2() noinline norecurse nounwind optnone {
bb:
  call void @alias2() nounwind readnone willreturn
  ret void
}

define internal void @aliasee_vgpr64_sgpr102() noinline norecurse nounwind readnone willreturn "amdgpu-waves-per-eu"="4,10" {
bb:
  call void asm sideeffect "; clobber v52 ", "~{v52}"()
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 500}
