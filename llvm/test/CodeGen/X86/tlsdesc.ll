; RUN: llc -mtriple=x86_64-linux-gnu -relocation-model=pic -enable-tlsdesc %s -o - | FileCheck %s --check-prefixes=INST
; RUN: llc -mtriple=x86_64-linux-gnu -relocation-model=pic -filetype=obj -enable-tlsdesc < %s | llvm-objdump -r - | FileCheck --check-prefixes=RELOC %s

@var = thread_local global i32 zeroinitializer

define i32 @test_thread_local() nounwind {

  %val = load i32, ptr @var
  ret i32 %val

;      INST: test_thread_local:
;      INST: leaq var@tlsdesc(%rip), [[REG:%.*]]
; INST-NEXT: callq *var@tlscall([[REG]])
; INST-NEXT: movl %fs:([[REG]]),

; RELOC: R_X86_64_GOTPC32_TLSDESC var
; RELOC: R_X86_64_TLSDESC_CALL var
}
