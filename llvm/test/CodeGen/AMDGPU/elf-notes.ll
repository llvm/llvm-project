; RUN: llc -mtriple=amdgcn-amd-unknown -mcpu=gfx802 < %s | FileCheck --check-prefix=OSABI-UNK %s
; RUN: llc -mtriple=amdgcn-amd-unknown -mcpu=iceland < %s | FileCheck --check-prefix=OSABI-UNK %s
; RUN: llc -mtriple=amdgcn-amd-unknown -mcpu=gfx802 -filetype=obj < %s | llvm-readelf --notes  - | FileCheck --check-prefix=OSABI-UNK-ELF %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx802 < %s| FileCheck --check-prefix=OSABI-HSA %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=iceland < %s | FileCheck --check-prefix=OSABI-HSA %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx802 -filetype=obj < %s | llvm-readelf --notes  - | FileCheck --check-prefix=OSABI-HSA-ELF %s
; RUN: llc -mtriple=amdgcn-amd-amdpal -mcpu=gfx802 < %s | FileCheck --check-prefix=OSABI-PAL %s
; RUN: llc -mtriple=amdgcn-amd-amdpal -mcpu=iceland < %s | FileCheck --check-prefix=OSABI-PAL %s
; RUN: llc -mtriple=amdgcn-amd-amdpal -mcpu=gfx802 -filetype=obj < %s | llvm-readelf --notes  - | FileCheck --check-prefix=OSABI-PAL-ELF %s
; RUN: llc -mtriple=r600 < %s | FileCheck --check-prefix=R600 %s

; OSABI-UNK-NOT: .hsa_code_object_version
; OSABI-UNK-NOT: .hsa_code_object_isa
; OSABI-UNK: .amd_amdgpu_isa "amdgcn-amd-unknown--gfx802"
; OSABI-UNK-NOT: .amd_amdgpu_hsa_metadata
; OSABI-UNK-NOT: .amd_amdgpu_pal_metadata

; OSABI-UNK-ELF-NOT: Unknown note type
; OSABI-UNK-ELF: NT_AMD_HSA_ISA_NAME (AMD HSA ISA Name)
; OSABI-UNK-ELF: AMD HSA ISA Name:
; OSABI-UNK-ELF: amdgcn-amd-unknown--gfx802
; OSABI-UNK-ELF-NOT: Unknown note type
; OSABI-UNK-ELF-NOT: NT_AMD_HSA_METADATA (AMD HSA Metadata)
; OSABI-UNK-ELF-NOT: Unknown note type
; OSABI-UNK-ELF-NOT: NT_AMD_PAL_METADATA (AMD PAL Metadata)
; OSABI-UNK-ELF-NOT: Unknown note type

; OSABI-HSA: amdhsa.target:   amdgcn-amd-amdhsa--gfx802
; OSABI-HSA: amdhsa.version:
; OSABI-HSA: .end_amdgpu_metadata
; OSABI-HSA-NOT: .amd_amdgpu_pal_metadata

; OSABI-HSA-ELF: NT_AMDGPU_METADATA (AMDGPU Metadata)
; OSABI-HSA-ELF: ---
; OSABI-HSA-ELF: amdhsa.kernels:
; OSABI-HSA-ELF:   - .args:           []
; OSABI-HSA-ELF:     .group_segment_fixed_size: 0
; OSABI-HSA-ELF:     .kernarg_segment_align: 4
; OSABI-HSA-ELF:     .kernarg_segment_size: 0
; OSABI-HSA-ELF:     .max_flat_workgroup_size: 1024
; OSABI-HSA-ELF:     .name:           elf_notes
; OSABI-HSA-ELF:     .private_segment_fixed_size: 0
; OSABI-HSA-ELF:     .sgpr_count:     96
; OSABI-HSA-ELF:     .sgpr_spill_count: 0
; OSABI-HSA-ELF:     .symbol:         elf_notes.kd
; OSABI-HSA-ELF:     .vgpr_count:     0
; OSABI-HSA-ELF:     .vgpr_spill_count: 0
; OSABI-HSA-ELF:     .wavefront_size: 64
; OSABI-HSA-ELF: amdhsa.target:   amdgcn-amd-amdhsa--gfx802
; OSABI-HSA-ELF: amdhsa.version:
; OSABI-HSA-ELF:   - 1
; OSABI-HSA-ELF:   - 1
; OSABI-HSA-ELF: ...
; OSABI-HSA-ELF-NOT: NT_AMD_PAL_METADATA (AMD PAL Metadata)

; OSABI-PAL: .amd_amdgpu_isa  "amdgcn-amd-amdpal--gfx802"
; OSABI-PAL: .amdgpu_pal_metadata
; OSABI-PAL-NOT: .amd_amdgpu_hsa_metadata

; OSABI-PAL-ELF: NT_AMD_HSA_ISA_NAME (AMD HSA ISA Name)
; OSABI-PAL-ELF: AMD HSA ISA Name:
; OSABI-PAL-ELF: amdgcn-amd-amdpal--gfx802
; OSABI-PAL-ELF-NOT: NT_AMD_HSA_METADATA (AMD HSA Metadata)
; OSABI-PAL-ELF: NT_AMDGPU_METADATA (AMDGPU Metadata)
; OSABI-PAL-ELF: AMDGPU Metadata:
; OSABI-PAL-ELF: amdpal.pipelines:
; OSABI-PAL-ELF:   - .hardware_stages:
; OSABI-PAL-ELF:       .cs:
; OSABI-PAL-ELF:         .entry_point:    elf_notes
; OSABI-PAL-ELF:         .scratch_memory_size: 0
; OSABI-PAL-ELF:         .sgpr_count:     96
; OSABI-PAL-ELF:         .vgpr_count:     1
; OSABI-PAL-ELF:     .registers:
; OSABI-PAL-ELF:       11794:           11469504
; OSABI-PAL-ELF:       11795:           128

; R600-NOT: .hsa_code_object_version
; R600-NOT: .hsa_code_object_isa
; R600-NOT: .amd_amdgpu_isa
; R600-NOT: .amd_amdgpu_hsa_metadata
; R600-NOT: .amd_amdgpu_pal_metadata

define amdgpu_kernel void @elf_notes() #0 {
  ret void
}

attributes #0 = { "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 400}
