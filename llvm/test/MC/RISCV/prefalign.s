# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+relax %s -o %t
# RUN: llvm-readelf -SW %t | FileCheck --check-prefix=OBJ %s
# RUN: llvm-objdump -d -M no-aliases --no-show-raw-insn %t | FileCheck --check-prefix=DIS %s
# RUN: llvm-readobj -r %t | FileCheck --check-prefix=RELOC %s

## Two functions in one section with nop fill.
## f1: body = 12 bytes < 16, ComputedAlign=16, but section start is 16-aligned
##     so pad = 0
## f2: body = 32 bytes >= 16, ComputedAlign=16, pad = 4 (one nop at 0xc)
# OBJ: .text.f1 PROGBITS {{[0-9a-f]+}} {{[0-9a-f]+}} 000030 00 AX 0 0 16
# DIS:       0: addi a0, zero, 0x1
# DIS-NEXT:  4: addi a0, zero, 0x2
# DIS-NEXT:  8: add a0, a0, a1
## Padding nop for f2
# DIS-NEXT:  c: addi zero, zero, 0x0
## f2 starts at 0x10, aligned to 16
# DIS-NEXT: 10: add a0, a0, a1
.section .text.f1,"ax",@progbits
.p2align 2
.prefalign 16, .Lf1_end, nop
addi a0, zero, 1
addi a0, zero, 2
add a0, a0, a1
.Lf1_end:
.prefalign 16, .Lf2_end, nop
.rept 8
add a0, a0, a1
.endr
.Lf2_end:

## .prefalign does not emit R_RISCV_ALIGN relocations. The padding is fully
## resolved at assembly time, so no linker adjustment is needed.
# RELOC: Relocations [
# RELOC-NEXT: ]
