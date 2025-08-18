# RUN: llvm-mc --filetype=obj --triple=loongarch64 -mattr=-relax %s -o %t.n
# RUN: llvm-readobj -r %t.n | FileCheck %s
# RUN: llvm-mc --filetype=obj --triple=loongarch64 -mattr=+relax %s -o %t.r
# RUN: llvm-readobj -r %t.r | FileCheck %s --check-prefix=CHECKR

# CHECK:      Relocations [
# CHECK-NEXT:   Section ({{.*}}) .rela.text {
# CHECK-NEXT:     0x4 R_LARCH_CALL36 foo 0x0
# CHECK-NEXT:   }
# CHECK-NEXT:   Section ({{.*}}) .rela.data {
# CHECK-NEXT:     0x0 R_LARCH_64 .text 0xC
# CHECK-NEXT:   }
# CHECK-NEXT: ]

# CHECKR:      Relocations [
# CHECKR-NEXT:   Section ({{.*}}) .rela.text {
# CHECKR-NEXT:     0x4 R_LARCH_CALL36 foo 0x0
# CHECKR-NEXT:     0x4 R_LARCH_RELAX - 0x0
# CHECKR-NEXT:     0x10 R_LARCH_B21 .L0 0x0
# CHECKR-NEXT:     0x14 R_LARCH_B21 .L1 0x0
# CHECKR-NEXT:     0x18 R_LARCH_B16 .L0 0x0
# CHECKR-NEXT:     0x1C R_LARCH_B16 .L1 0x0
# CHECKR-NEXT:     0x20 R_LARCH_B26 .L0 0x0
# CHECKR-NEXT:     0x24 R_LARCH_B26 .L1 0x0
# CHECKR-NEXT:   }
# CHECKR-NEXT:   Section ({{.*}}) .rela.data {
# CHECKR-NEXT:     0x0 R_LARCH_64 .L1 0x0
# CHECKR-NEXT:   }
# CHECKR-NEXT: ]

.text
  nop

.L0:
  call36 foo

.L1:
  nop
  bnez $a0, .L0
  beqz $a0, .L1
  beq  $a0, $a1, .L0
  blt  $a0, $a1, .L1
  bl   .L0
  b    .L1

.data
.dword .L1
