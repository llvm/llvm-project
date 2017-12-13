# RUN: llvm-mc %s -triple=riscv32 -mattr=+c,+f -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+c,+f < %s \
# RUN:     | llvm-objdump -mattr=+c,+f -d - | FileCheck -check-prefix=CHECK-INST %s

# CHECK-INST: c.flwsp  fs0, 252(sp)
# CHECK: encoding: [0x7e,0x74]
c.flwsp  fs0, 252(sp)
# CHECK-INST: c.fswsp  fa7, 252(sp)
# CHECK: encoding: [0xc6,0xff]
c.fswsp  fa7, 252(sp)

# CHECK-INST: c.flw  fa3, 124(a5)
# CHECK: encoding: [0xf4,0x7f]
c.flw  fa3, 124(a5)
# CHECK-INST: c.fsw  fa2, 124(a1)
# CHECK: encoding: [0xf0,0xfd]
c.fsw  fa2, 124(a1)
