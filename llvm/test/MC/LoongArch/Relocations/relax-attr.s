# RUN: llvm-mc --filetype=obj --triple=loongarch64 %s -o %t
# RUN: llvm-readobj -r %t | FileCheck %s
# RUN: llvm-mc --filetype=obj --triple=loongarch64 -mattr=+relax %s -o %t
# RUN: llvm-readobj -r %t | FileCheck %s --check-prefix=CHECKR

# CHECK:      Relocations [
# CHECK-NEXT:   Section ({{.*}}) .rela.data {
# CHECK-NEXT:     0x0 R_LARCH_64 .text 0x4
# CHECK-NEXT:   }
# CHECK-NEXT: ]

# CHECKR:      Relocations [
# CHECKR-NEXT:   Section ({{.*}}) .rela.text {
# CHECKR-NEXT:     0x8 R_LARCH_B21 .L1 0x0
# CHECKR-NEXT:     0xC R_LARCH_B16 .L1 0x0
# CHECKR-NEXT:     0x10 R_LARCH_B26 .L1 0x0
# CHECKR-NEXT:   }
# CHECKR-NEXT:   Section ({{.*}}) .rela.data {
# CHECKR-NEXT:     0x0 R_LARCH_64 .L1 0x0
# CHECKR-NEXT:   }
# CHECKR-NEXT: ]

.text
  nop
.L1:
  nop
  beqz $a0, .L1
  blt  $a0, $a1, .L1
  b    .L1

.data
.dword .L1
