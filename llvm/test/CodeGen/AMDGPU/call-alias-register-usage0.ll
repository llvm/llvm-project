; RUN: llc -O0 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 < %s | FileCheck %s

; CallGraphAnalysis, which CodeGenSCC order depends on, does not look
; through aliases. If GlobalOpt is never run, we do not see direct
; calls,

@alias0 = hidden alias void (), ptr @aliasee_default_vgpr64_sgpr102

; CHECK-LABEL: {{^}}kernel0:
; CHECK:      .set kernel0.num_vgpr, max(41, .Laliasee_default_vgpr64_sgpr102.num_vgpr)
; CHECK-NEXT: .set kernel0.num_agpr, max(0, .Laliasee_default_vgpr64_sgpr102.num_agpr)
; CHECK-NEXT: .set kernel0.numbered_sgpr, max(33, .Laliasee_default_vgpr64_sgpr102.numbered_sgpr)
define amdgpu_kernel void @kernel0() #0 {
bb:
  call void @alias0() #2
  ret void
}

; CHECK:      .set .Laliasee_default_vgpr64_sgpr102.num_vgpr, 53
; CHECK-NEXT: .set .Laliasee_default_vgpr64_sgpr102.num_agpr, 0
; CHECK-NEXT: .set .Laliasee_default_vgpr64_sgpr102.numbered_sgpr, 32
define internal void @aliasee_default_vgpr64_sgpr102() #1 {
bb:
  call void asm sideeffect "; clobber v52 ", "~{v52}"()
  ret void
}

attributes #0 = { noinline norecurse nounwind optnone }
attributes #1 = { noinline norecurse nounwind readnone willreturn }
attributes #2 = { nounwind readnone willreturn }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 500}
