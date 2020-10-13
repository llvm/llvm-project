; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=kaveri -mattr=-code-object-v3 | FileCheck --check-prefix=HSA %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=kaveri -mattr=-code-object-v3,-flat-for-global | FileCheck --check-prefix=HSA-CI %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=carrizo -mattr=-code-object-v3 | FileCheck --check-prefix=HSA %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=carrizo -mattr=-code-object-v3,-flat-for-global | FileCheck --check-prefix=HSA-VI %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=kaveri -filetype=obj -mattr=-code-object-v3 | llvm-readobj -symbols -s -sd - | FileCheck --check-prefix=ELF %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=kaveri -mattr=-code-object-v3 | llvm-mc -filetype=obj -triple amdgcn--amdhsa -mcpu=kaveri -mattr=-code-object-v3 | llvm-readobj -symbols -s -sd - | FileCheck %s --check-prefix=ELF
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx1010 -mattr=+wavefrontsize32,-wavefrontsize64,-code-object-v3 | FileCheck --check-prefix=HSA --check-prefix=GFX10 --check-prefix=GFX10-W32 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx1010 -mattr=-wavefrontsize32,+wavefrontsize64,-code-object-v3 | FileCheck --check-prefix=HSA --check-prefix=GFX10 --check-prefix=GFX10-W64 %s

; The SHT_NOTE section contains the output from the .hsa_code_object_*
; directives.

; ELF: Section {
; ELF: Name: .text
; ELF: Type: SHT_PROGBITS (0x1)
; ELF: Flags [ (0x6)
; ELF: SHF_ALLOC (0x2)
; ELF: SHF_EXECINSTR (0x4)
; ELF: }

; ELF: SHT_NOTE
; ELF: Flags [ (0x2)
; ELF: SHF_ALLOC (0x2)
; ELF: ]
; ELF: SectionData (
; ELF: 0000: 04000000 08000000 01000000 414D4400
; ELF: 0010: 02000000 01000000 04000000 1B000000
; ELF: 0020: 03000000 414D4400 04000700 07000000
; ELF: 0030: 00000000 00000000 414D4400 414D4447
; ELF: 0040: 50550000
; ELF: )

; ELF: Symbol {
; ELF: Name: simple
; ELF: Size: 288
; ELF: Type: AMDGPU_HSA_KERNEL (0xA)
; ELF: }

; HSA-NOT: .AMDGPU.config
; HSA: .text
; HSA: .hsa_code_object_version 2,1
; HSA-CI: .hsa_code_object_isa 7,0,0,"AMD","AMDGPU"
; HSA-VI: .hsa_code_object_isa 8,0,1,"AMD","AMDGPU"

; HSA-LABEL: .amdgpu_hsa_kernel simple
; HSA: {{^}}simple:
; HSA: .amd_kernel_code_t
; HSA: enable_sgpr_private_segment_buffer = 1
; HSA: enable_sgpr_kernarg_segment_ptr = 1

; PRE-GFX10: enable_wavefront_size32 = 0
; GFX10-W32: enable_wavefront_size32 = 1
; GFX10-W64: enable_wavefront_size32 = 0

; PRE-GFX10: wavefront_size = 6
; GFX10-W32: wavefront_size = 5
; GFX10-W64: wavefront_size = 6

; HSA: call_convention = -1
; HSA: .end_amd_kernel_code_t
; HSA: s_load_dwordx2 s[{{[0-9]+:[0-9]+}}], s[4:5], 0x0

; Make sure we are setting the ATC bit:
; HSA-CI: s_mov_b32 s[[HI:[0-9]]], 0x100f000
; On VI+ we also need to set MTYPE = 2
; HSA-VI: s_mov_b32 s[[HI:[0-9]]], 0x1100f000
; Make sure we generate flat store for HSA
; PRE-GFX10: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}
; GFX10: global_store_dword v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}

; HSA: .Lfunc_end0:
; HSA: .size   simple, .Lfunc_end0-simple

define amdgpu_kernel void @simple(i32 addrspace(1)* %out) {
entry:
  store i32 0, i32 addrspace(1)* %out
  ret void
}

; HSA-LABEL: .amdgpu_hsa_kernel simple_no_kernargs
; HSA: enable_sgpr_kernarg_segment_ptr = 0
define amdgpu_kernel void @simple_no_kernargs() {
entry:
  store volatile i32 0, i32 addrspace(1)* undef
  ret void
}
