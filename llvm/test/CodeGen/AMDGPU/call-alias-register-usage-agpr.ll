; RUN: llc -O0 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 < %s | FileCheck -check-prefixes=ALL,GFX908 %s
; RUN: llc -O0 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a < %s | FileCheck -check-prefixes=ALL,GFX90A %s

; CallGraphAnalysis, which CodeGenSCC order depends on, does not look
; through aliases. If GlobalOpt is never run, we do not see direct
; calls,

@alias = hidden alias void (), ptr @aliasee_default

; ALL-LABEL: {{^}}kernel:
; GFX908: .amdhsa_next_free_vgpr 32
; GFX908-NEXT: .amdhsa_next_free_sgpr 33

; GFX90A: .amdhsa_next_free_vgpr 59
; GFX90A-NEXT: .amdhsa_next_free_sgpr 33
; GFX90A-NEXT: .amdhsa_accum_offset 32
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

attributes #0 = { noinline norecurse nounwind optnone }
attributes #1 = { noinline norecurse nounwind readnone willreturn }
attributes #2 = { nounwind readnone willreturn }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 500}
