# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+relax %s \
# RUN:     | llvm-readobj -r - | FileCheck -check-prefix RELAX %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=-relax %s \
# RUN:     | llvm-readobj -r - | FileCheck -check-prefix NORELAX %s

# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax %s \
# RUN:     | llvm-readobj -r - | FileCheck -check-prefix RELAX %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=-relax %s \
# RUN:     | llvm-readobj -r - | FileCheck -check-prefix NORELAX %s

# NORELAX:      Relocations [
# NORELAX-NEXT:   .rela.text {
# NORELAX-NEXT:     R_RISCV_CALL_PLT
# NORELAX-NEXT:   }
# NORELAX-NEXT: ]

.globl G1
.globl G2
.globl G3
.L1:
G1:
  call extern
.L2:
G2:
  .p2align 3
.L3:
G3:

.data
.dword .L2-.L1
.dword G2-G1
.word .L2-.L1
.word G2-G1
.half .L2-.L1
.half G2-G1
.byte .L2-.L1
.byte G2-G1
.dword .L3-.L2
.dword G3-G2
.word .L3-.L2
.word G3-G2
.half .L3-.L2
.half G3-G2
.byte .L3-.L2
.byte G3-G2
# RELAX:      .rela.data {
# RELAX-NEXT:   0x0 R_RISCV_ADD64 .L2 0x0
# RELAX-NEXT:   0x0 R_RISCV_SUB64 .L1 0x0
# RELAX-NEXT:   0x8 R_RISCV_ADD64 G2 0x0
# RELAX-NEXT:   0x8 R_RISCV_SUB64 G1 0x0
# RELAX-NEXT:   0x10 R_RISCV_ADD32 .L2 0x0
# RELAX-NEXT:   0x10 R_RISCV_SUB32 .L1 0x0
# RELAX-NEXT:   0x14 R_RISCV_ADD32 G2 0x0
# RELAX-NEXT:   0x14 R_RISCV_SUB32 G1 0x0
# RELAX-NEXT:   0x18 R_RISCV_ADD16 .L2 0x0
# RELAX-NEXT:   0x18 R_RISCV_SUB16 .L1 0x0
# RELAX-NEXT:   0x1A R_RISCV_ADD16 G2 0x0
# RELAX-NEXT:   0x1A R_RISCV_SUB16 G1 0x0
# RELAX-NEXT:   0x1C R_RISCV_ADD8 .L2 0x0
# RELAX-NEXT:   0x1C R_RISCV_SUB8 .L1 0x0
# RELAX-NEXT:   0x1D R_RISCV_ADD8 G2 0x0
# RELAX-NEXT:   0x1D R_RISCV_SUB8 G1 0x0
# RELAX-NEXT:   0x1E R_RISCV_ADD64 .L3 0x0
# RELAX-NEXT:   0x1E R_RISCV_SUB64 .L2 0x0
# RELAX-NEXT:   0x26 R_RISCV_ADD64 G3 0x0
# RELAX-NEXT:   0x26 R_RISCV_SUB64 G2 0x0
# RELAX-NEXT:   0x2E R_RISCV_ADD32 .L3 0x0
# RELAX-NEXT:   0x2E R_RISCV_SUB32 .L2 0x0
# RELAX-NEXT:   0x32 R_RISCV_ADD32 G3 0x0
# RELAX-NEXT:   0x32 R_RISCV_SUB32 G2 0x0
# RELAX-NEXT:   0x36 R_RISCV_ADD16 .L3 0x0
# RELAX-NEXT:   0x36 R_RISCV_SUB16 .L2 0x0
# RELAX-NEXT:   0x38 R_RISCV_ADD16 G3 0x0
# RELAX-NEXT:   0x38 R_RISCV_SUB16 G2 0x0
# RELAX-NEXT:   0x3A R_RISCV_ADD8 .L3 0x0
# RELAX-NEXT:   0x3A R_RISCV_SUB8 .L2 0x0
# RELAX-NEXT:   0x3B R_RISCV_ADD8 G3 0x0
# RELAX-NEXT:   0x3B R_RISCV_SUB8 G2 0x0
# RELAX-NEXT: }
