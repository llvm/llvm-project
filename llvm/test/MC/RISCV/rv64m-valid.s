# RUN: llvm-mc %s -triple=riscv64 -mattr=+m -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+m < %s \
# RUN:     | llvm-objdump -mattr=+m -M no-aliases -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-OBJ,CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: mulw ra, sp, gp
# CHECK-ASM: encoding: [0xbb,0x00,0x31,0x02]
mulw ra, sp, gp
# CHECK-ASM-AND-OBJ: divw tp, t0, t1
# CHECK-ASM: encoding: [0x3b,0xc2,0x62,0x02]
divw tp, t0, t1
# CHECK-ASM-AND-OBJ: divuw t2, s0, s2
# CHECK-ASM: encoding: [0xbb,0x53,0x24,0x03]
divuw t2, s0, s2
# CHECK-ASM-AND-OBJ: remw a0, a1, a2
# CHECK-ASM: encoding: [0x3b,0xe5,0xc5,0x02]
remw a0, a1, a2
# CHECK-ASM-AND-OBJ: remuw a3, a4, a5
# CHECK-ASM: encoding: [0xbb,0x76,0xf7,0x02]
remuw a3, a4, a5
