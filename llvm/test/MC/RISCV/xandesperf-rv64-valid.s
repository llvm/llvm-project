# XAndesPerf - Andes Performance Extension
# RUN: llvm-mc %s -triple=riscv64 -mattr=+xandesperf -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+xandesperf < %s \
# RUN:     | llvm-objdump --mattr=+xandesperf -M no-aliases --no-print-imm-hex -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: nds.lea.b.ze t0, t1, t3
# CHECK-ASM: encoding: [0xdb,0x02,0xc3,0x11]
nds.lea.b.ze t0, t1, t3

# CHECK-ASM-AND-OBJ: nds.lea.h.ze a0, a1, a3
# CHECK-ASM: encoding: [0x5b,0x85,0xd5,0x12]
nds.lea.h.ze a0, a1, a3

# CHECK-ASM-AND-OBJ: nds.lea.w.ze s0, s1, s3
# CHECK-ASM: encoding: [0x5b,0x84,0x34,0x15]
nds.lea.w.ze s0, s1, s3

# CHECK-ASM-AND-OBJ: nds.lea.d.ze a3, a4, a5
# CHECK-ASM: encoding: [0xdb,0x06,0xf7,0x16]
nds.lea.d.ze a3, a4, a5

# CHECK-ASM-AND-OBJ: nds.lwugp t0, 9568
# CHECK-ASM: encoding: [0xab,0x62,0x04,0x56]
nds.lwugp t0, 0x2560

# CHECK-ASM-AND-OBJ: nds.ldgp t0, -9568
# CHECK-ASM: encoding: [0xab,0xb2,0x7b,0xaa]
nds.ldgp t0, -0x2560

# CHECK-ASM-AND-OBJ: nds.sdgp t0, 9568
# CHECK-ASM: encoding: [0x2b,0x70,0x54,0x56]
nds.sdgp t0, 0x2560
