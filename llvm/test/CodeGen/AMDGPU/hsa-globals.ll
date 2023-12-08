; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=kaveri | FileCheck --check-prefix=ASM %s

@linkonce_odr_global_program = linkonce_odr addrspace(1) global i32 0
@linkonce_global_program = linkonce addrspace(1) global i32 0
@internal_global_program = internal addrspace(1) global i32 0
@common_global_program = common addrspace(1) global i32 0
@external_global_program = addrspace(1) global i32 0

@internal_readonly = internal unnamed_addr addrspace(4) constant i32 0
@external_readonly = unnamed_addr addrspace(4) constant i32 0

define amdgpu_kernel void @test() {
  ret void
}

@weak_global = extern_weak addrspace(1) global i32

; ASM: .type linkonce_odr_global_program,@object
; ASM: .section .bss,#alloc,#write
; ASM: .weak linkonce_odr_global_program
; ASM: linkonce_odr_global_program:
; ASM: .long 0
; ASM: .size linkonce_odr_global_program, 4

; ASM: .type linkonce_global_program,@object
; ASM: .weak linkonce_global_program
; ASM: linkonce_global_program:
; ASM: .long 0
; ASM: .size linkonce_global_program, 4

; ASM: .type internal_global_program,@object
; ASM: .local internal_global_program
; ASM: .comm internal_global_program,4,2

; ASM: .type common_global_program,@object
; ASM: .comm common_global_program,4,2

; ASM: external_global_program:
; ASM: .long 0
; ASM: .size external_global_program, 4

; ASM: .type internal_readonly,@object
; ASM: .section .rodata.cst4,"aM",@progbits,4
; ASM: internal_readonly:
; ASM: .long 0
; ASM: .size internal_readonly, 4

; ASM: .type external_readonly,@object
; ASM: .globl external_readonly
; ASM: external_readonly:
; ASM: .long 0
; ASM: .size external_readonly, 4

; ASM: .weak weak_global
