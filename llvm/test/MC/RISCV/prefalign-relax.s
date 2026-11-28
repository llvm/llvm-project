# RUN: llvm-mc -triple=riscv32 -mattr=+c,+relax -filetype=obj %s -o - \
# RUN:   | llvm-readobj -r - | FileCheck %s --check-prefix=RVC
# RUN: llvm-mc -triple=riscv64 -mattr=+c,+relax -filetype=obj %s -o - \
# RUN:   | llvm-readobj -r - | FileCheck %s --check-prefix=RVC
# RUN: llvm-mc -triple=riscv32 -mattr=-c,+relax -filetype=obj %s -o - \
# RUN:   | llvm-readobj -r - | FileCheck %s --check-prefix=RVC
# RUN: llvm-mc -triple=riscv32 -mattr=-c,+relax -riscv-align-rvc=false \
# RUN:   -filetype=obj %s -o - | llvm-readobj -r - | FileCheck %s --check-prefix=NORVC
# RUN: llvm-mc -triple=riscv32 -mattr=+c,-relax -filetype=obj %s -o - \
# RUN:   | llvm-readobj -r - | FileCheck %s --check-prefix=NORELAX

## Preferred alignment following a relaxable call needs an alignment relocation.
## With compressed instructions, reserve 64 - 2 bytes for preferred alignment.
# RVC-LABEL: Section ({{[0-9]+}}) .rela.text {
# RVC:       0x8 R_RISCV_ALIGN - 0x2
# RVC-NEXT:  0xA R_RISCV_ALIGN - 0x3E
# RVC-NEXT:  }

## With compressed instructions and riscv-align-rvc both disabled, reserve
## 64 - 4 bytes. The preceding .p2align 2 requires no padding reservation.
# NORVC-LABEL: Section ({{[0-9]+}}) .rela.text {
# NORVC:       0x8 R_RISCV_ALIGN - 0x3C
# NORVC-NEXT:  }

## The 20-byte body selects alignment 32, despite requesting preferred
## alignment 64. The relocation must describe the computed alignment.
# RVC-LABEL: Section ({{[0-9]+}}) .rela.text.small {
# RVC:       0x8 R_RISCV_ALIGN - 0x2
# RVC-NEXT:  0xA R_RISCV_ALIGN - 0x1E
# RVC-NEXT:  }

# NORVC-LABEL: Section ({{[0-9]+}}) .rela.text.small {
# NORVC:       0x8 R_RISCV_ALIGN - 0x1C
# NORVC-NEXT:  }

## Disabling assembler relaxation must retain ordinary fixed padding.
# NORELAX:     Relocations [
# NORELAX-NOT: R_RISCV_RELAX
# NORELAX-NOT: R_RISCV_ALIGN
# NORELAX:     ]

.text
.globl big
call big

.p2align 2
.prefalign 6, .Lbig_end, nop
.type big, @function
big:
## Keep the body exactly 64 bytes in every configuration.
.option push
.option exact
.rept 15
  nop
.endr
  ret
.option pop
.Lbig_end:
.size big, .Lbig_end-big

.section .text.small,"ax",@progbits
.globl small
call small

.p2align 2
.prefalign 6, .Lsmall_end, nop
.type small, @function
small:
## Keep the body exactly 20 bytes in every configuration.
.option push
.option exact
.rept 4
  nop
.endr
  ret
.option pop
.Lsmall_end:
.size small, .Lsmall_end-small
