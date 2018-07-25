# RUN: llvm-mc %s -triple=mipsel-unknown-linux -mcpu=mips32r2 \
# RUN:   -mattr=+micromips -filetype=obj -o - | llvm-readobj -t | FileCheck %s
  .text
  .set nomicromips
f:
  nop
g:
  .set micromips
  nop
h:
  .word 0
k:
  .long 0
l:
  .hword 0
m:
  .2byte 0
n:
  .4byte 0
o:
  .8byte 0
i:
  nop
j:
  .set nomicromips
  nop
# CHECK: Symbols [
# CHECK:   Symbol {
# CHECK:     Name: f
# CHECK:     Binding: Local
# CHECK:     Type: None
# CHECK:     Other: 0
# CHECK:     Section: .text
# CHECK:   }
# CHECK:   Symbol {
# CHECK:     Name: g
# CHECK:     Binding: Local
# CHECK:     Type: None
# CHECK:     Other [ (0x80)
# CHECK:       STO_MIPS_MICROMIPS
# CHECK:     ]
# CHECK:     Section: .text
# CHECK:   }
# CHECK:   Symbol {
# CHECK:     Name: h
# CHECK:     Binding: Local
# CHECK:     Type: None
# CHECK:     Other: 0
# CHECK:     Section: .text
# CHECK:   }
# CHECK:   Symbol {
# CHECK:     Name: i
# CHECK:     Binding: Local
# CHECK:     Type: None
# CHECK:     Other [ (0x80)
# CHECK:       STO_MIPS_MICROMIPS
# CHECK:     ]
# CHECK:     Section: .text
# CHECK:   }
# CHECK:   Symbol {
# CHECK:     Name: j
# CHECK:     Binding: Local
# CHECK:     Type: None
# CHECK:     Other: 0
# CHECK:     Section: .text
# CHECK:   }
# CHECK:   Symbol {
# CHECK:     Name: k
# CHECK:     Binding: Local
# CHECK:     Type: None
# CHECK:     Other: 0
# CHECK:     Section: .text
# CHECK:   }
# CHECK:   Symbol {
# CHECK:     Name: l
# CHECK:     Binding: Local
# CHECK:     Type: None
# CHECK:     Other: 0
# CHECK:     Section: .text
# CHECK:   }
# CHECK:   Symbol {
# CHECK:     Name: m
# CHECK:     Binding: Local
# CHECK:     Type: None
# CHECK:     Other: 0
# CHECK:     Section: .text
# CHECK:   }
# CHECK:   Symbol {
# CHECK:     Name: n
# CHECK:     Binding: Local
# CHECK:     Type: None
# CHECK:     Other: 0
# CHECK:     Section: .text
# CHECK:   }
# CHECK:   Symbol {
# CHECK:     Name: o
# CHECK:     Binding: Local
# CHECK:     Type: None
# CHECK:     Other: 0
# CHECK:     Section: .text
# CHECK:   }
# CHECK: ]
