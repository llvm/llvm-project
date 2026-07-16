; RUN: llc -O0 -mtriple=amdgpu9.08-amd-amdhsa < %s | FileCheck -check-prefix=ALL,GFX908 %s
; RUN: llc -O0 -mtriple=amdgpu9.0a-amd-amdhsa < %s | FileCheck -check-prefixes=ALL,GFX90A %s

@alias = hidden alias void (), ptr @aliasee_default

define internal void @aliasee_default() #1 {
bb:
  call void asm sideeffect "; clobber a26 ", "~{a26}"()
  ret void
}
; ALL:      .set .Laliasee_default.num_vgpr, 0
; ALL-NEXT: .set .Laliasee_default.num_agpr, 27
; ALL-NEXT: .set .Laliasee_default.numbered_sgpr, 32

; GFX908-LABEL: {{^}}kernel:
; GFX908:          .amdhsa_next_free_vgpr 41
; GFX908-NEXT:     .amdhsa_next_free_sgpr 33

; GFX90A-LABEL: {{^}}kernel:
; GFX90A:          .amdhsa_next_free_vgpr 71
; GFX90A-NEXT:     .amdhsa_next_free_sgpr 33
; GFX90A-NEXT:  .amdhsa_accum_offset 44

; ALL:       .set .Lkernel.num_vgpr, max(41, .Laliasee_default.num_vgpr)
; ALL-NEXT:  .set .Lkernel.num_agpr, max(0, .Laliasee_default.num_agpr)
; ALL-NEXT:  .set .Lkernel.numbered_sgpr, max(33, .Laliasee_default.numbered_sgpr)
define amdgpu_kernel void @kernel() #0 {
bb:
  call void @alias() #2
  ret void
}

attributes #0 = { noinline norecurse nounwind optnone }
attributes #1 = { noinline norecurse nounwind readnone willreturn }
attributes #2 = { nounwind readnone willreturn }

