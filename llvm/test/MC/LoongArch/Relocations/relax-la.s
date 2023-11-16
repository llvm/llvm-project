# RUN: llvm-mc --filetype=obj --triple=loongarch64 -mattr=+relax %s -o %t
# RUN: llvm-readobj -r %t | FileCheck %s

# CHECK:       Relocations [
# CHECK-NEXT:    Section ({{.*}}) .rela.text {
# CHECK-NEXT:      0x0 R_LARCH_PCALA_HI20 .L1 0x0
# CHECK-NEXT:      0x0 R_LARCH_RELAX - 0x0
# CHECK-NEXT:      0x4 R_LARCH_PCALA_LO12 .L1 0x0
# CHECK-NEXT:      0x4 R_LARCH_RELAX - 0x0
# CHECK-NEXT:      0x8 R_LARCH_GOT_PC_HI20 .L1 0x0
# CHECK-NEXT:      0x8 R_LARCH_RELAX - 0x0
# CHECK-NEXT:      0xC R_LARCH_GOT_PC_LO12 .L1 0x0
# CHECK-NEXT:      0xC R_LARCH_RELAX - 0x0
# CHECK-NEXT:    }

.text
.L1:
  la.pcrel $a0, .L1
  la.got   $a0, .L1
