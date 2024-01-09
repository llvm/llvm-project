# RUN: llvm-mc --filetype=obj --triple=loongarch64 --mattr=-relax %s \
# RUN:     | llvm-readelf -rs - | FileCheck %s --check-prefix=NORELAX
# RUN: llvm-mc --filetype=obj --triple=loongarch64 --mattr=+relax %s \
# RUN:     | llvm-readelf -rs - | FileCheck %s --check-prefix=RELAX
# RUN: llvm-mc --filetype=obj --triple=loongarch64 --mattr=+relax %s \
# RUN:     | llvm-objdump -d - | FileCheck -check-prefix=RELAX-INST %s

# NORELAX: There are no relocations in this file.
# NORELAX: Symbol table '.symtab' contains 1 entries:

# RELAX:       0000000000000000  0000000100000066 R_LARCH_ALIGN          0000000000000000 {{.*}} + 4
# RELAX-NEXT:  0000000000000010  0000000100000066 R_LARCH_ALIGN          0000000000000000 {{.*}} + 5
# RELAX-NEXT:  000000000000002c  0000000100000066 R_LARCH_ALIGN          0000000000000000 {{.*}} + 4
# RELAX-NEXT:  000000000000003c  0000000100000066 R_LARCH_ALIGN          0000000000000000 {{.*}} + b04
# RELAX-NEXT:  0000000000000048  0000000100000066 R_LARCH_ALIGN          0000000000000000 {{.*}} + 4
# RELAX-EMPTY:
# RELAX:       0000000000000000  0000000200000066 R_LARCH_ALIGN          0000000000000000 <null> + 4
# RELAX-EMPTY:
# RELAX:       Symbol table '.symtab' contains 3 entries:
# RELAX:       0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT   UND
# RELAX-NEXT:  1: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT     2
# RELAX-NEXT:  2: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT     4

.text
.p2align 4        # A = 0x0
nop
.p2align 5        # B = A + 3 * NOP + NOP = 0x10
.p2align 4        # C = B + 7 * NOP = 0x2C
nop
.p2align 4, , 11  # D = C + 3 * NOP + NOP = 0x3C
## Not emit the third parameter.
.p2align 4, , 12  # E = D + 3 * NOP = 0x48
                  # END = E + 3 * NOP = 0x54 = 21 * NOP

## Not emit R_LARCH_ALIGN if code alignment great than alignment directive.
.p2align 2
.p2align 1
.p2align 0
## Not emit instructions if max emit bytes less than min nop size.
.p2align 4, , 2
## Not emit R_LARCH_ALIGN if alignment directive with specific padding value.
.p2align 4, 1
nop
.p2align 4, 1, 12

# RELAX-INST:           <.text>:
# RELAX-INST-COUNT-21:    nop
# RELAX-INST-COUNT-3:     01 01 01 01
# RELAX-INST-NEXT:        nop
# RELAX-INST-COUNT-3:     01 01 01 01

.section .text2, "ax"
.p2align 4
