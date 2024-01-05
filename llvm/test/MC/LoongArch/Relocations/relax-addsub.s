# RUN: llvm-mc --filetype=obj --triple=loongarch64 --mattr=-relax %s \
# RUN:     | llvm-readobj -r -x .data - | FileCheck %s --check-prefix=NORELAX
# RUN: llvm-mc --filetype=obj --triple=loongarch64 --mattr=+relax %s \
# RUN:     | llvm-readobj -r -x .data - | FileCheck %s --check-prefix=RELAX

# NORELAX:       Relocations [
# NORELAX-NEXT:    Section ({{.*}}) .rela.text {
# NORELAX-NEXT:      0x10 R_LARCH_PCALA_HI20 .text 0x0
# NORELAX-NEXT:      0x14 R_LARCH_PCALA_LO12 .text 0x0
# NORELAX-NEXT:    }
# NORELAX-NEXT:  ]

# NORELAX:      Hex dump of section '.data':
# NORELAX-NEXT: 0x00000000 04040004 00000004 00000000 0000000c
# NORELAX-NEXT: 0x00000010 0c000c00 00000c00 00000000 00000808
# NORELAX-NEXT: 0x00000020 00080000 00080000 00000000 00

# RELAX:       Relocations [
# RELAX-NEXT:    Section ({{.*}}) .rela.text {
# RELAX-NEXT:      0x10 R_LARCH_PCALA_HI20 .L1 0x0
# RELAX-NEXT:      0x10 R_LARCH_RELAX - 0x0
# RELAX-NEXT:      0x14 R_LARCH_PCALA_LO12 .L1 0x0
# RELAX-NEXT:      0x14 R_LARCH_RELAX - 0x0
# RELAX-NEXT:    }
# RELAX-NEXT:    Section ({{.*}}) .rela.data {
# RELAX-NEXT:      0x1E R_LARCH_ADD8 .L4 0x0
# RELAX-NEXT:      0x1E R_LARCH_SUB8 .L3 0x0
# RELAX-NEXT:      0x1F R_LARCH_ADD16 .L4 0x0
# RELAX-NEXT:      0x1F R_LARCH_SUB16 .L3 0x0
# RELAX-NEXT:      0x21 R_LARCH_ADD32 .L4 0x0
# RELAX-NEXT:      0x21 R_LARCH_SUB32 .L3 0x0
# RELAX-NEXT:      0x25 R_LARCH_ADD64 .L4 0x0
# RELAX-NEXT:      0x25 R_LARCH_SUB64 .L3 0x0
# RELAX-NEXT:    }
# RELAX-NEXT:  ]

# RELAX:      Hex dump of section '.data':
# RELAX-NEXT: 0x00000000 04040004 00000004 00000000 0000000c
# RELAX-NEXT: 0x00000010 0c000c00 00000c00 00000000 00000000
# RELAX-NEXT: 0x00000020 00000000 00000000 00000000 00

.text
.L1:
  nop
.L2:
  .align 4
.L3:
  la.pcrel $t0, .L1
.L4:
  ret

.data
## Not emit relocs
.byte  .L2 - .L1
.short .L2 - .L1
.word  .L2 - .L1
.dword .L2 - .L1
## TODO Handle alignment directive.
.byte  .L3 - .L2
.short .L3 - .L2
.word  .L3 - .L2
.dword .L3 - .L2
## With relaxation, emit relocs because the la.pcrel makes the diff variable.
.byte  .L4 - .L3
.short .L4 - .L3
.word  .L4 - .L3
.dword .L4 - .L3
