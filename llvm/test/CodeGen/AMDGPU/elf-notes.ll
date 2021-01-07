; RUN: llc -mtriple=amdgcn-amd-unknown -mcpu=gfx802 --amdhsa-code-object-version=2 < %s | FileCheck --check-prefix=OSABI-UNK %s
; RUN: llc -mtriple=amdgcn-amd-unknown -mcpu=iceland --amdhsa-code-object-version=2 < %s | FileCheck --check-prefix=OSABI-UNK %s
; RUN: llc -mtriple=amdgcn-amd-unknown -mcpu=gfx802 -filetype=obj --amdhsa-code-object-version=2 < %s | llvm-readelf --notes  - | FileCheck --check-prefix=OSABI-UNK-ELF %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx802 --amdhsa-code-object-version=2 < %s | FileCheck --check-prefix=OSABI-HSA %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=iceland --amdhsa-code-object-version=2 < %s | FileCheck --check-prefix=OSABI-HSA %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx802 -filetype=obj --amdhsa-code-object-version=2 < %s | llvm-readelf --notes  - | FileCheck --check-prefix=OSABI-HSA-ELF %s
; RUN: llc -mtriple=amdgcn-amd-amdpal -mcpu=gfx802 --amdhsa-code-object-version=2 < %s | FileCheck --check-prefix=OSABI-PAL %s
; RUN: llc -mtriple=amdgcn-amd-amdpal -mcpu=iceland --amdhsa-code-object-version=2 < %s | FileCheck --check-prefix=OSABI-PAL %s
; RUN: llc -mtriple=amdgcn-amd-amdpal -mcpu=gfx802 -filetype=obj --amdhsa-code-object-version=2 < %s | llvm-readelf --notes  - | FileCheck --check-prefix=OSABI-PAL-ELF %s
; RUN: llc -march=r600 < %s | FileCheck --check-prefix=R600 %s

; OSABI-UNK-NOT: .hsa_code_object_version
; OSABI-UNK-NOT: .hsa_code_object_isa
; OSABI-UNK: .amd_amdgpu_isa "amdgcn-amd-unknown--gfx802"
; OSABI-UNK-NOT: .amd_amdgpu_hsa_metadata
; OSABI-UNK-NOT: .amd_amdgpu_pal_metadata

; OSABI-UNK-ELF-NOT: Unknown note type
; OSABI-UNK-ELF: NT_AMD_AMDGPU_ISA (ISA Version)
; OSABI-UNK-ELF: ISA Version:
; OSABI-UNK-ELF: amdgcn-amd-unknown--gfx802
; OSABI-UNK-ELF-NOT: Unknown note type
; OSABI-UNK-ELF-NOT: NT_AMD_AMDGPU_HSA_METADATA (HSA Metadata)
; OSABI-UNK-ELF-NOT: Unknown note type
; OSABI-UNK-ELF-NOT: NT_AMD_AMDGPU_PAL_METADATA (PAL Metadata)
; OSABI-UNK-ELF-NOT: Unknown note type

; OSABI-HSA: .hsa_code_object_version
; OSABI-HSA: .hsa_code_object_isa
; OSABI-HSA: .amd_amdgpu_isa "amdgcn-amd-amdhsa--gfx802"
; OSABI-HSA: .amd_amdgpu_hsa_metadata
; OSABI-HSA-NOT: .amd_amdgpu_pal_metadata

; OSABI-HSA-ELF: Unknown note type: (0x00000001)
; OSABI-HSA-ELF: Unknown note type: (0x00000003)
; OSABI-HSA-ELF: NT_AMD_AMDGPU_ISA (ISA Version)
; OSABI-HSA-ELF: ISA Version:
; OSABI-HSA-ELF: amdgcn-amd-amdhsa--gfx802
; OSABI-HSA-ELF: NT_AMD_AMDGPU_HSA_METADATA (HSA Metadata)
; OSABI-HSA-ELF: HSA Metadata:
; OSABI-HSA-ELF: ---
; OSABI-HSA-ELF: Version: [ 1, 0 ]
; OSABI-HSA-ELF: Kernels:
; OSABI-HSA-ELF:   - Name:       elf_notes
; OSABI-HSA-ELF:     SymbolName: 'elf_notes@kd'
; OSABI-HSA-ELF:     CodeProps:
; OSABI-HSA-ELF:       KernargSegmentSize: 0
; OSABI-HSA-ELF:       GroupSegmentFixedSize: 0
; OSABI-HSA-ELF:       PrivateSegmentFixedSize: 0
; OSABI-HSA-ELF:       KernargSegmentAlign: 4
; OSABI-HSA-ELF:       WavefrontSize:   64
; OSABI-HSA-ELF:       NumSGPRs:        96
; OSABI-HSA-ELF: ...
; OSABI-HSA-ELF-NOT: NT_AMD_AMDGPU_PAL_METADATA (PAL Metadata)

; OSABI-PAL-NOT: .hsa_code_object_version
; OSABI-PAL: .hsa_code_object_isa
; OSABI-PAL: .amd_amdgpu_isa "amdgcn-amd-amdpal--gfx802"
; OSABI-PAL-NOT: .amd_amdgpu_hsa_metadata

; OSABI-PAL-ELF: Unknown note type: (0x00000003)
; OSABI-PAL-ELF: NT_AMD_AMDGPU_ISA (ISA Version)
; OSABI-PAL-ELF: ISA Version:
; OSABI-PAL-ELF: amdgcn-amd-amdpal--gfx802
; OSABI-PAL-ELF-NOT: NT_AMD_AMDGPU_HSA_METADATA (HSA Metadata)
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
; OSABI-PAL: amdpal.pipelines:
; OSABI-PAL:   - .hardware_stages:
; OSABI-PAL:       .cs:
; OSABI-PAL:         .entry_point:    elf_notes
; OSABI-PAL:         .scratch_memory_size: 0
; OSABI-PAL:         .sgpr_count:     0x60
; OSABI-PAL:         .vgpr_count:     0x1
; OSABI-PAL:     .registers:
; OSABI-PAL:       0x2e12 (COMPUTE_PGM_RSRC1): 0xaf02c0
; OSABI-PAL:       0x2e13 (COMPUTE_PGM_RSRC2): 0x80

; R600-NOT: .hsa_code_object_version
; R600-NOT: .hsa_code_object_isa
; R600-NOT: .amd_amdgpu_isa
; R600-NOT: .amd_amdgpu_hsa_metadata
; R600-NOT: .amd_amdgpu_pal_metadata

define amdgpu_kernel void @elf_notes() {
  ret void
}
