# RUN: llvm-mc --filetype=obj --triple=loongarch64 --mattr=-relax %s \
# RUN:     | llvm-readobj -r -x .data - | FileCheck %s --check-prefix=NORELAX
# RUN: llvm-mc --filetype=obj --triple=loongarch64 --mattr=+relax %s \
# RUN:     | llvm-readobj -r -x .data - | FileCheck %s --check-prefix=RELAX

# NORELAX:       Relocations [
# NORELAX-NEXT:    Section ({{.*}}) .rela.text {
# NORELAX-NEXT:      0x10 R_LARCH_PCALA_HI20 .text 0x0
# NORELAX-NEXT:      0x14 R_LARCH_PCALA_LO12 .text 0x0
# NORELAX-NEXT:    }
# NORELAX-NEXT:    Section ({{.*}}) .rela.data {
# NORELAX-NEXT:      0x30 R_LARCH_ADD8 foo 0x0
# NORELAX-NEXT:      0x30 R_LARCH_SUB8 .text 0x10
# NORELAX-NEXT:      0x31 R_LARCH_ADD16 foo 0x0
# NORELAX-NEXT:      0x31 R_LARCH_SUB16 .text 0x10
# NORELAX-NEXT:      0x33 R_LARCH_ADD32 foo 0x0
# NORELAX-NEXT:      0x33 R_LARCH_SUB32 .text 0x10
# NORELAX-NEXT:      0x37 R_LARCH_ADD64 foo 0x0
# NORELAX-NEXT:      0x37 R_LARCH_SUB64 .text 0x10
# NORELAX-NEXT:    }
# NORELAX-NEXT:  ]

# NORELAX:      Hex dump of section '.data':
# NORELAX-NEXT: 0x00000000 04040004 00000004 00000000 00000004
# NORELAX-NEXT: 0x00000010 0c0c000c 0000000c 00000000 0000000c
# NORELAX-NEXT: 0x00000020 08080008 00000008 00000000 00000008
# NORELAX-NEXT: 0x00000030 00000000 00000000 00000000 000000

# RELAX:       Relocations [
# RELAX-NEXT:    Section ({{.*}}) .rela.text {
# RELAX-NEXT:      0x4 R_LARCH_ALIGN - 0xC
# RELAX-NEXT:      0x10 R_LARCH_PCALA_HI20 .L1 0x0
# RELAX-NEXT:      0x10 R_LARCH_RELAX - 0x0
# RELAX-NEXT:      0x14 R_LARCH_PCALA_LO12 .L1 0x0
# RELAX-NEXT:      0x14 R_LARCH_RELAX - 0x0
# RELAX-NEXT:    }
# RELAX-NEXT:    Section ({{.*}}) .rela.data {
# RELAX-NEXT:      0x10 R_LARCH_ADD8 .L3 0x0
# RELAX-NEXT:      0x10 R_LARCH_SUB8 .L2 0x0
# RELAX-NEXT:      0x11 R_LARCH_ADD16 .L3 0x0
# RELAX-NEXT:      0x11 R_LARCH_SUB16 .L2 0x0
# RELAX-NEXT:      0x13 R_LARCH_ADD32 .L3 0x0
# RELAX-NEXT:      0x13 R_LARCH_SUB32 .L2 0x0
# RELAX-NEXT:      0x17 R_LARCH_ADD64 .L3 0x0
# RELAX-NEXT:      0x17 R_LARCH_SUB64 .L2 0x0
# RELAX-NEXT:      0x1F R_LARCH_ADD_ULEB128 .L3 0x0
# RELAX-NEXT:      0x1F R_LARCH_SUB_ULEB128 .L2 0x0
# RELAX-NEXT:      0x20 R_LARCH_ADD8 .L4 0x0
# RELAX-NEXT:      0x20 R_LARCH_SUB8 .L3 0x0
# RELAX-NEXT:      0x21 R_LARCH_ADD16 .L4 0x0
# RELAX-NEXT:      0x21 R_LARCH_SUB16 .L3 0x0
# RELAX-NEXT:      0x23 R_LARCH_ADD32 .L4 0x0
# RELAX-NEXT:      0x23 R_LARCH_SUB32 .L3 0x0
# RELAX-NEXT:      0x27 R_LARCH_ADD64 .L4 0x0
# RELAX-NEXT:      0x27 R_LARCH_SUB64 .L3 0x0
# RELAX-NEXT:      0x2F R_LARCH_ADD_ULEB128 .L4 0x0
# RELAX-NEXT:      0x2F R_LARCH_SUB_ULEB128 .L3 0x0
# RELAX-NEXT:      0x30 R_LARCH_ADD8 foo 0x0
# RELAX-NEXT:      0x30 R_LARCH_SUB8 .L3 0x0
# RELAX-NEXT:      0x31 R_LARCH_ADD16 foo 0x0
# RELAX-NEXT:      0x31 R_LARCH_SUB16 .L3 0x0
# RELAX-NEXT:      0x33 R_LARCH_ADD32 foo 0x0
# RELAX-NEXT:      0x33 R_LARCH_SUB32 .L3 0x0
# RELAX-NEXT:      0x37 R_LARCH_ADD64 foo 0x0
# RELAX-NEXT:      0x37 R_LARCH_SUB64 .L3 0x0
# RELAX-NEXT:    }
# RELAX-NEXT:  ]

# RELAX:      Hex dump of section '.data':
# RELAX-NEXT: 0x00000000 04040004 00000004 00000000 00000004
# RELAX-NEXT: 0x00000010 00000000 00000000 00000000 00000000
# RELAX-NEXT: 0x00000020 00000000 00000000 00000000 00000000
# RELAX-NEXT: 0x00000030 00000000 00000000 00000000 000000

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
.uleb128 .L2 - .L1
## With relaxation, emit relocs because the .align makes the diff variable.
.byte  .L3 - .L2
.short .L3 - .L2
.word  .L3 - .L2
.dword .L3 - .L2
.uleb128 .L3 - .L2
## With relaxation, emit relocs because the la.pcrel makes the diff variable.
.byte  .L4 - .L3
.short .L4 - .L3
.word  .L4 - .L3
.dword .L4 - .L3
.uleb128 .L4 - .L3
.byte  foo - .L3
.short foo - .L3
.word  foo - .L3
.dword foo - .L3
