; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=kaveri | FileCheck --check-prefix=ASM %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=kaveri | llvm-mc -filetype=obj -triple amdgcn--amdhsa -mcpu=kaveri | llvm-readobj -symbols -s | FileCheck %s --check-prefix=ELF

@linkonce_odr_global_program = linkonce_odr addrspace(1) global i32 0
@linkonce_global_program = linkonce addrspace(1) global i32 0
@internal_global_program = internal addrspace(1) global i32 0
@common_global_program = common addrspace(1) global i32 0
@external_global_program = addrspace(1) global i32 0

@linkonce_odr_global_agent = linkonce_odr addrspace(1) global i32 0, section ".hsadata_global_agent"
@linkonce_global_agent = linkonce addrspace(1) global i32 0, section ".hsadata_global_agent"
@internal_global_agent = internal addrspace(1) global i32 0, section ".hsadata_global_agent"
@common_global_agent = common addrspace(1) global i32 0, section ".hsadata_global_agent"
@external_global_agent = addrspace(1) global i32 0, section ".hsadata_global_agent"

@internal_readonly = internal unnamed_addr addrspace(2) constant i32 0
@external_readonly = unnamed_addr addrspace(2) constant i32 0

define void @test() {
  ret void
}

; ASM: .amdgpu_hsa_module_global linkonce_odr_global
; ASM: .size linkonce_odr_global_program, 4
; ASM: .hsadata_global_program
; ASM: linkonce_odr_global_program:
; ASM: .long 0

; ASM: .amdgpu_hsa_module_global linkonce_global
; ASM: .size linkonce_global_program, 4
; ASM: .hsadata_global_program
; ASM: linkonce_global_program:
; ASM: .long 0

; ASM: .amdgpu_hsa_module_global internal_global
; ASM: .size internal_global_program, 4
; ASM: .hsadata_global_program
; ASM: internal_global_program:
; ASM: .long 0

; ASM: .amdgpu_hsa_module_global common_global
; ASM: .size common_global_program, 4
; ASM: .hsadata_global_program
; ASM: common_global_program:
; ASM: .long 0

; ASM: .amdgpu_hsa_program_global external_global
; ASM: .size external_global_program, 4
; ASM: .hsadata_global_program
; ASM: external_global_program:
; ASM: .long 0

; ASM: .amdgpu_hsa_module_global linkonce_odr_global
; ASM: .size linkonce_odr_global_agent, 4
; ASM: .hsadata_global_agent
; ASM: linkonce_odr_global_agent:
; ASM: .long 0

; ASM: .amdgpu_hsa_module_global linkonce_global
; ASM: .size linkonce_global_agent, 4
; ASM: .hsadata_global_agent
; ASM: linkonce_global_agent:
; ASM: .long 0

; ASM: .amdgpu_hsa_module_global internal_global
; ASM: .size internal_global_agent, 4
; ASM: .hsadata_global_agent
; ASM: internal_global_agent:
; ASM: .long 0

; ASM: .amdgpu_hsa_module_global common_global
; ASM: .size common_global_agent, 4
; ASM: .hsadata_global_agent
; ASM: common_global_agent:
; ASM: .long 0

; ASM: .amdgpu_hsa_program_global external_global
; ASM: .size external_global_agent, 4
; ASM: .hsadata_global_agent
; ASM: external_global_agent:
; ASM: .long 0

; ASM: .amdgpu_hsa_module_global internal_readonly
; ASM: .size internal_readonly, 4
; ASM: .hsatext
; ASM: internal_readonly:
; ASM: .long 0

; ASM: .amdgpu_hsa_program_global external_readonly
; ASM: .size external_readonly, 4
; ASM: .hsatext
; ASM: external_readonly:
; ASM: .long 0

; ELF: Section {
; ELF: Name: .hsadata_global_program
; ELF: Type: SHT_PROGBITS (0x1)
; ELF: Flags [ (0x100003)
; ELF: SHF_ALLOC (0x2)
; ELF: SHF_AMDGPU_HSA_GLOBAL (0x100000)
; ELF: SHF_WRITE (0x1)
; ELF: ]
; ELF: }

; ELF: Section {
; ELF: Name: .hsadata_global_agent
; ELF: Type: SHT_PROGBITS (0x1)
; ELF: Flags [ (0x900003)
; ELF: SHF_ALLOC (0x2)
; ELF: SHF_AMDGPU_HSA_AGENT (0x800000)
; ELF: SHF_AMDGPU_HSA_GLOBAL (0x100000)
; ELF: SHF_WRITE (0x1)
; ELF: ]
; ELF: }

; ELF: Symbol {
; ELF: Name: common_global_agent
; ELF: Size: 4
; ELF: Binding: Local
; ELF: Section: .hsadata_global_agent
; ELF: }

; ELF: Symbol {
; ELF: Name: common_global_program
; ELF: Size: 4
; ELF: Binding: Local
; ELF: Section: .hsadata_global_program
; ELF: }

; ELF: Symbol {
; ELF: Name: internal_global_agent
; ELF: Size: 4
; ELF: Binding: Local
; ELF: Type: Object
; ELF: Section: .hsadata_global_agent
; ELF: }

; ELF: Symbol {
; ELF: Name: internal_global_program
; ELF: Size: 4
; ELF: Binding: Local
; ELF: Type: Object
; ELF: Section: .hsadata_global_program
; ELF: }

; ELF: Symbol {
; ELF: Name: internal_readonly
; ELF: Size: 4
; ELF: Binding: Local
; ELF: Type: Object
; ELF: Section: .hsatext
; ELF: }

; ELF: Symbol {
; ELF: Name: linkonce_global_agent
; ELF: Size: 4
; ELF: Binding: Local
; ELF: Section: .hsadata_global_agent
; ELF: }

; ELF: Symbol {
; ELF: Name: linkonce_global_program
; ELF: Size: 4
; ELF: Binding: Local
; ELF: Section: .hsadata_global_program
; ELF: }

; ELF: Symbol {
; ELF: Name: linkonce_odr_global_agent
; ELF: Size: 4
; ELF: Binding: Local
; ELF: Section: .hsadata_global_agent
; ELF: }

; ELF: Symbol {
; ELF: Name: linkonce_odr_global_program
; ELF: Size: 4
; ELF: Binding: Local
; ELF: Section: .hsadata_global_program
; ELF: }

; ELF: Symbol {
; ELF: Name: external_global_agent
; ELF: Size: 4
; ELF: Binding: Global
; ELF: Type: Object
; ELF: Section: .hsadata_global_agent
; ELF: }

; ELF: Symbol {
; ELF: Name: external_global_program
; ELF: Size: 4
; ELF: Binding: Global
; ELF: Type: Object
; ELF: Section: .hsadata_global_program
; ELF: }

; ELF: Symbol {
; ELF: Name: external_readonly
; ELF: Size: 4
; ELF: Binding: Global
; ELF: Type: Object
; ELF: Section: .hsatext
; ELF: }
