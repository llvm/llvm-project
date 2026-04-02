# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=-relax < %s \
# RUN:     | llvm-readobj -r - | FileCheck %s

## .option relax overrides -mno-relax and enables R_RISCV_ALIGN/R_RISCV_RELAX relocations.
# CHECK:      .rela.text
# CHECK:       R_RISCV_CALL_PLT
# CHECK-NEXT:  R_RISCV_RELAX
# CHECK-NEXT:  R_RISCV_ALIGN
.option relax
call foo
.align 4

## Alignments before the first linker-relaxable instruction do not need relocations.
# CHECK-NOT: .rela.text1
.section .text1,"ax"
.align 4
nop

# CHECK: .rela.text2
.section .text2,"ax"
call foo
