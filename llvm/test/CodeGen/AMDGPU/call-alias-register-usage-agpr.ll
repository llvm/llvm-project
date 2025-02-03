; RUN: llc -O0 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 < %s | FileCheck -check-prefix=ALL %s
; RUN: llc -O0 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a < %s | FileCheck -check-prefixes=ALL,GFX90A %s

; CallGraphAnalysis, which CodeGenSCC order depends on, does not look
; through aliases. If GlobalOpt is never run, we do not see direct
; calls,

@alias = hidden alias void (), ptr @aliasee_default

; ALL-LABEL: {{^}}kernel:
; ALL:          .amdhsa_next_free_vgpr max(totalnumvgprs(kernel.num_agpr, kernel.num_vgpr), 1, 0)
; ALL-NEXT:     .amdhsa_next_free_sgpr (max(kernel.numbered_sgpr+(extrasgprs(kernel.uses_vcc, kernel.uses_flat_scratch, 1)), 1, 0))-(extrasgprs(kernel.uses_vcc, kernel.uses_flat_scratch, 1))
; GFX90A-NEXT:  .amdhsa_accum_offset ((((((alignto(max(1, kernel.num_vgpr), 4))/4)-1)&(~65536))&63)+1)*4

; ALL:       .set kernel.num_vgpr, max(41, .Laliasee_default.num_vgpr)
; ALL-NEXT:  .set kernel.num_agpr, max(0, .Laliasee_default.num_agpr)
; ALL-NEXT:  .set kernel.numbered_sgpr, max(33, .Laliasee_default.numbered_sgpr)
define amdgpu_kernel void @kernel() #0 {
bb:
  call void @alias() #2
  ret void
}

define internal void @aliasee_default() #1 {
bb:
  call void asm sideeffect "; clobber a26 ", "~{a26}"()
  ret void
}
; ALL:      .set .Laliasee_default.num_vgpr, 0
; ALL-NEXT: .set .Laliasee_default.num_agpr, 27
; ALL-NEXT: .set .Laliasee_default.numbered_sgpr, 32

attributes #0 = { noinline norecurse nounwind optnone }
attributes #1 = { noinline norecurse nounwind readnone willreturn }
attributes #2 = { nounwind readnone willreturn }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 500}
