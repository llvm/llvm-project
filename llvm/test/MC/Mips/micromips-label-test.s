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
# CHECK: ]

