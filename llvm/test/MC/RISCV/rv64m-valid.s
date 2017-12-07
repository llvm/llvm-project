# RUN: llvm-mc %s -triple=riscv64 -mattr=+m -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+m < %s \
# RUN:     | llvm-objdump -mattr=+m -d - | FileCheck -check-prefix=CHECK-INST %s

# CHECK-INST: mulw ra, sp, gp
# CHECK: encoding: [0xbb,0x00,0x31,0x02]
mulw ra, sp, gp
# CHECK-INST: divw tp, t0, t1
# CHECK: encoding: [0x3b,0xc2,0x62,0x02]
divw tp, t0, t1
# CHECK-INST: divuw t2, s0, s2
# CHECK: encoding: [0xbb,0x53,0x24,0x03]
divuw t2, s0, s2
# CHECK-INST: remw a0, a1, a2
# CHECK: encoding: [0x3b,0xe5,0xc5,0x02]
remw a0, a1, a2
# CHECK-INST: remuw a3, a4, a5
# CHECK: encoding: [0xbb,0x76,0xf7,0x02]
remuw a3, a4, a5
