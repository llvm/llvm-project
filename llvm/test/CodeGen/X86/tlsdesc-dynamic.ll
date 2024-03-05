; RUN: llc -mtriple=x86_64-linux-gnu -relocation-model=pic -enable-tlsdesc %s -o - | FileCheck %s --check-prefixes=GD
; RUN: llc -mtriple=x86_64-linux-gnu -relocation-model=pic -enable-tlsdesc -filetype=obj < %s | llvm-objdump -r - | FileCheck --check-prefixes=GD-RELOC %s

@general_dynamic_var = external thread_local global i32

define i32 @test_generaldynamic() {
  %val = load i32, ptr @general_dynamic_var
  ret i32 %val
;      GD: test_generaldynamic:
;      GD: leaq general_dynamic_var@tlsdesc(%rip), [[REG:%.*]]
; GD-NEXT: callq *general_dynamic_var@tlscall([[REG]])
; GD-NEXT: movl %fs:([[REG]]),

; GD-RELOC: R_X86_64_GOTPC32_TLSDESC general_dynamic_var
; GD-RELOC: R_X86_64_TLSDESC_CALL general_dynamic_var
}

define ptr @test_generaldynamic_addr() {
  ret ptr @general_dynamic_var
;      GD: test_generaldynamic_addr:
;      GD: leaq general_dynamic_var@tlsdesc(%rip), [[REG:%.*]]
; GD-NEXT: callq *general_dynamic_var@tlscall([[REG]])
; GD-NEXT: addq %fs:0, %rax

; GD-RELOC: R_X86_64_GOTPC32_TLSDESC general_dynamic_var
; GD-RELOC: R_X86_64_TLSDESC_CALL general_dynamic_var
}

@local_dynamic_var = external thread_local(localdynamic) global i32

define i32 @test_localdynamic() {
  %val = load i32, ptr @local_dynamic_var
  ret i32 %val
;      GD: test_localdynamic:
;      GD: leaq _TLS_MODULE_BASE_@tlsdesc(%rip), [[REG:%.*]]
; GD-NEXT: callq *_TLS_MODULE_BASE_@tlscall([[REG]])
; GD-NEXT: movl  %fs:local_dynamic_var@DTPOFF(%rax), %eax

; GD-RELOC: R_X86_64_GOTPC32_TLSDESC _TLS_MODULE_BASE_
; GD-RELOC: R_X86_64_TLSDESC_CALL _TLS_MODULE_BASE_
; GD-RELOC: R_X86_64_DTPOFF32 local_dynamic_var
}

define ptr @test_localdynamic_addr() {
  ret ptr @local_dynamic_var
;      GD: test_localdynamic_addr:
;      GD: leaq _TLS_MODULE_BASE_@tlsdesc(%rip), [[REG:%.*]]
; GD-NEXT: callq *_TLS_MODULE_BASE_@tlscall([[REG]])
; GD-NEXT: movq %fs:0, %rcx
; GD-NEXT: leaq local_dynamic_var@DTPOFF(%rcx,[[REG]])

; GD-RELOC: R_X86_64_GOTPC32_TLSDESC _TLS_MODULE_BASE_
; GD-RELOC: R_X86_64_TLSDESC_CALL _TLS_MODULE_BASE_
; GD-RELOC: R_X86_64_DTPOFF32 local_dynamic_var
}

@local_dynamic_var2 = external thread_local(localdynamic) global i32

define i32 @test_localdynamic_deduplicate() {
  %val = load i32, ptr @local_dynamic_var
  %val2 = load i32, ptr @local_dynamic_var2
  %sum = add i32 %val, %val2
  ret i32 %sum
;      GD: test_localdynamic_deduplicate:
;      GD: leaq _TLS_MODULE_BASE_@tlsdesc(%rip), [[REG:%.*]]
; GD-NEXT: callq *_TLS_MODULE_BASE_@tlscall([[REG]])
; GD-NEXT: movl  %fs:local_dynamic_var@DTPOFF(%rax)
; GD-NEXT: addl  %fs:local_dynamic_var2@DTPOFF(%rax)

; GD-RELOC: R_X86_64_GOTPC32_TLSDESC _TLS_MODULE_BASE_
; GD-RELOC: R_X86_64_TLSDESC_CALL _TLS_MODULE_BASE_
; GD-RELOC: R_X86_64_DTPOFF32 local_dynamic_var2
}

