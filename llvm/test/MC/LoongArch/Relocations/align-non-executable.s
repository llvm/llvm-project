## A label difference separated by an alignment directive, when the
## referenced symbols are in a non-executable section with instructions,
## should generate ADD/SUB relocations.
## https://github.com/llvm/llvm-project/pull/76552

# RUN: llvm-mc --filetype=obj --triple=loongarch64 --mattr=+relax %s \
# RUN:     | llvm-readobj -r - | FileCheck --check-prefixes=CHECK,RELAX %s
# RUN: llvm-mc --filetype=obj --triple=loongarch64 --mattr=-relax %s \
# RUN:     | llvm-readobj -r - | FileCheck %s

.section ".dummy", "a"
.L1:
  la.pcrel $t0, sym
.p2align 3
.L2:
.dword .L2 - .L1

# CHECK:       Relocations [
# CHECK-NEXT:    Section ({{.*}}) .rela.dummy {
# CHECK-NEXT:      0x0 R_LARCH_PCALA_HI20 sym 0x0
# RELAX-NEXT:      0x0 R_LARCH_RELAX - 0x0
# CHECK-NEXT:      0x4 R_LARCH_PCALA_LO12 sym 0x0
# RELAX-NEXT:      0x4 R_LARCH_RELAX - 0x0
# RELAX-NEXT:      0x8 R_LARCH_ADD64 .L2 0x0
# RELAX-NEXT:      0x8 R_LARCH_SUB64 .L1 0x0
# CHECK-NEXT:    }
# CHECK-NEXT:  ]
