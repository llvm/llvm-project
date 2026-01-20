# RUN: llvm-mc -triple=hexagon -filetype=obj %s | llvm-readelf -x .data - \
# RUN:   | FileCheck %s --match-full-lines

# Illustrate the case when padding packets across labels also breaks leb128
# relocations. This happens because .align padding is inserted once at the
# very end of the section layout.
L1:
  nop
L2:
.size L1, L2-L1
.align 16
  nop
.data
.word L2-L1
.uleb128 L2-L1

# CHECK: Hex dump of section '.data':
# CHECK-NEXT: 0x00000000 04000000 04 .....
