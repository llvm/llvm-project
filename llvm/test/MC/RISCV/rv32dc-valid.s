# RUN: llvm-mc %s -triple=riscv32 -mattr=+c,+d -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+c,+d < %s \
# RUN:     | llvm-objdump -mattr=+c,+d -d - | FileCheck -check-prefix=CHECK-INST %s

# CHECK-INST: c.fldsp  fs0, 504(sp)
# CHECK: encoding: [0x7e,0x34]
c.fldsp  fs0, 504(sp)
# CHECK-INST: c.fsdsp  fa7, 504(sp)
# CHECK: encoding: [0xc6,0xbf]
c.fsdsp  fa7, 504(sp)

# CHECK-INST: c.fld  fa3, 248(a5)
# CHECK: encoding: [0xf4,0x3f]
c.fld  fa3, 248(a5)
# CHECK-INST: c.fsd  fa2, 248(a1)
# CHECK: encoding: [0xf0,0xbd]
c.fsd  fa2, 248(a1)
