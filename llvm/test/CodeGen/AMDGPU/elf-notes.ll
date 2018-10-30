; RUN: llc -mtriple=amdgcn-amd-unknown -mcpu=gfx802 -mattr=-code-object-v3 < %s | FileCheck --check-prefix=GCN --check-prefix=OSABI-UNK --check-prefix=GFX802 %s
; RUN: llc -mtriple=amdgcn-amd-unknown -mcpu=iceland -mattr=-code-object-v3 < %s | FileCheck --check-prefix=GCN --check-prefix=OSABI-UNK --check-prefix=GFX802 %s
; RUN: llc -mtriple=amdgcn-amd-unknown -mcpu=gfx802 -filetype=obj -mattr=-code-object-v3 < %s | llvm-readobj -elf-output-style=GNU -notes  | FileCheck --check-prefix=GCN --check-prefix=OSABI-UNK-ELF --check-prefix=GFX802 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx802 -mattr=-code-object-v3 < %s | FileCheck --check-prefix=GCN --check-prefix=OSABI-HSA --check-prefix=GFX802 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=iceland -mattr=-code-object-v3 < %s | FileCheck --check-prefix=GCN --check-prefix=OSABI-HSA --check-prefix=GFX802 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx802 -filetype=obj -mattr=-code-object-v3 < %s | llvm-readobj -elf-output-style=GNU -notes  | FileCheck --check-prefix=GCN --check-prefix=OSABI-HSA-ELF --check-prefix=GFX802 %s
; RUN: llc -mtriple=amdgcn-amd-amdpal -mcpu=gfx802 -mattr=-code-object-v3 < %s | FileCheck --check-prefix=GCN --check-prefix=OSABI-PAL --check-prefix=GFX802 %s
; RUN: llc -mtriple=amdgcn-amd-amdpal -mcpu=iceland -mattr=-code-object-v3 < %s | FileCheck --check-prefix=GCN --check-prefix=OSABI-PAL --check-prefix=GFX802 %s
; RUN: llc -mtriple=amdgcn-amd-amdpal -mcpu=gfx802 -filetype=obj -mattr=-code-object-v3 < %s | llvm-readobj -elf-output-style=GNU -notes  | FileCheck --check-prefix=GCN --check-prefix=OSABI-PAL-ELF --check-prefix=GFX802 %s
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

; OSABI-HSA-ELF: Unknown note type (0x00000001)
; OSABI-HSA-ELF: Unknown note type (0x00000003)
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
; OSABI-PAL: .amd_amdgpu_pal_metadata

; OSABI-PAL-ELF: Unknown note type (0x00000003)
; OSABI-PAL-ELF: NT_AMD_AMDGPU_ISA (ISA Version)
; OSABI-PAL-ELF: ISA Version:
; OSABI-PAL-ELF: amdgcn-amd-amdpal--gfx802
; OSABI-PAL-ELF-NOT: NT_AMD_AMDGPU_HSA_METADATA (HSA Metadata)
; OSABI-PAL-ELF: NT_AMD_AMDGPU_PAL_METADATA (PAL Metadata)
; OSABI-PAL-ELF: PAL Metadata:
; TODO: Following check line fails on mips:
; OSABI-PAL-ELF-XXX: 0x2e12,0xac02c0,0x2e13,0x80,0x1000001b,0x1,0x10000022,0x60,0x1000003e,0x0

; R600-NOT: .hsa_code_object_version
; R600-NOT: .hsa_code_object_isa
; R600-NOT: .amd_amdgpu_isa
; R600-NOT: .amd_amdgpu_hsa_metadata
; R600-NOT: .amd_amdgpu_pal_metadata

define amdgpu_kernel void @elf_notes() {
  ret void
}
