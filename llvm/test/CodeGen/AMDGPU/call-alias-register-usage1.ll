; RUN: llc -O0 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 < %s | FileCheck %s

; CallGraphAnalysis, which CodeGenSCC order depends on, does not look
; through aliases. If GlobalOpt is never run, we do not see direct
; calls,

@alias1 = hidden alias void (), ptr @aliasee_vgpr32_sgpr76

; The parent kernel has a higher VGPR usage than the possible callees.

; CHECK-LABEL: {{^}}kernel1:
; CHECK:      .amdhsa_next_free_vgpr max(totalnumvgprs(kernel1.num_agpr, kernel1.num_vgpr), 1, 0)
; CHECK-NEXT: .amdhsa_next_free_sgpr (max(kernel1.numbered_sgpr+(extrasgprs(kernel1.uses_vcc, kernel1.uses_flat_scratch, 1)), 1, 0))-(extrasgprs(kernel1.uses_vcc, kernel1.uses_flat_scratch, 1))

; CHECK:      .set kernel1.num_vgpr, max(42, .Laliasee_vgpr32_sgpr76.num_vgpr)
; CHECK-NEXT: .set kernel1.num_agpr, max(0, .Laliasee_vgpr32_sgpr76.num_agpr)
; CHECK-NEXT: .set kernel1.numbered_sgpr, max(33, .Laliasee_vgpr32_sgpr76.numbered_sgpr)
define amdgpu_kernel void @kernel1() #0 {
bb:
  call void asm sideeffect "; clobber v40 ", "~{v40}"()
  call void @alias1() #2
  ret void
}

; CHECK:      .set .Laliasee_vgpr32_sgpr76.num_vgpr, 27
; CHECK-NEXT: .set .Laliasee_vgpr32_sgpr76.num_agpr, 0
; CHECK-NEXT: .set .Laliasee_vgpr32_sgpr76.numbered_sgpr, 32
define internal void @aliasee_vgpr32_sgpr76() #1 {
bb:
  call void asm sideeffect "; clobber v26 ", "~{v26}"()
  ret void
}

attributes #0 = { noinline norecurse nounwind optnone }
attributes #1 = { noinline norecurse nounwind readnone willreturn "amdgpu-waves-per-eu"="8,10" }
attributes #2 = { nounwind readnone willreturn }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 500}
