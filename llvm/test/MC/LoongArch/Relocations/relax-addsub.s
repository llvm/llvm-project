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
# RELAX-NEXT:      0x14 R_LARCH_PCALA_LO12 .L1 0x0
# RELAX-NEXT:    }
# RELAX-NEXT:    Section ({{.*}}) .rela.data {
# RELAX-NEXT:      0xF R_LARCH_ADD8 .L3 0x0
# RELAX-NEXT:      0xF R_LARCH_SUB8 .L2 0x0
# RELAX-NEXT:      0x10 R_LARCH_ADD16 .L3 0x0
# RELAX-NEXT:      0x10 R_LARCH_SUB16 .L2 0x0
# RELAX-NEXT:      0x12 R_LARCH_ADD32 .L3 0x0
# RELAX-NEXT:      0x12 R_LARCH_SUB32 .L2 0x0
# RELAX-NEXT:      0x16 R_LARCH_ADD64 .L3 0x0
# RELAX-NEXT:      0x16 R_LARCH_SUB64 .L2 0x0
# RELAX-NEXT:    }
# RELAX-NEXT:  ]

# RELAX:      Hex dump of section '.data':
# RELAX-NEXT: 0x00000000 04040004 00000004 00000000 00000000
# RELAX-NEXT: 0x00000010 00000000 00000000 00000000 00000808
# RELAX-NEXT: 0x00000020 00080000 00080000 00000000 00

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
## With relaxation, emit relocs because of the .align making the diff variable.
## TODO Handle alignment directive. Why they emit relocs now? They returns
## without folding symbols offset in AttemptToFoldSymbolOffsetDifference().
.byte  .L3 - .L2
.short .L3 - .L2
.word  .L3 - .L2
.dword .L3 - .L2
## TODO
## With relaxation, emit relocs because la.pcrel is a linker-relaxable inst.
.byte  .L4 - .L3
.short .L4 - .L3
.word  .L4 - .L3
.dword .L4 - .L3
