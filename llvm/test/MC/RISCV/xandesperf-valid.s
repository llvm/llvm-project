# XAndesPerf - Andes Performance Extension
# RUN: llvm-mc %s -triple=riscv32 -mattr=+xandesperf -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+xandesperf < %s \
# RUN:     | llvm-objdump --mattr=+xandesperf -M no-aliases --no-print-imm-hex -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-OBJ,CHECK-OBJ32,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+xandesperf -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+xandesperf < %s \
# RUN:     | llvm-objdump --mattr=+xandesperf -M no-aliases --no-print-imm-hex -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-OBJ,CHECK-OBJ64,CHECK-ASM-AND-OBJ %s

# CHECK-OBJ: nds.bbc a0, 16, 0x200
# CHECK-ASM: nds.bbc a0, 16, 512
# CHECK-ASM: encoding: [0x5b,0x70,0x05,0x21]
nds.bbc a0, 16, 512

# CHECK-OBJ32: nds.bbs a1, 21, 0xffffff04
# CHECK-OBJ64: nds.bbs a1, 21, 0xffffffffffffff04
# CHECK-ASM: nds.bbs a1, 21, -256
# CHECK-ASM: encoding: [0x5b,0xf0,0x55,0xf1]
nds.bbs a1, 21, -256

# CHECK-OBJ: beqc t0, 23, 0x108
# CHECK-ASM: beqc t0, 23, 256
# CHECK-ASM: encoding: [0x5b,0xd0,0x72,0x11]
nds.beqc t0, 23, 256

# CHECK-OBJ32: bnec t1, 9, 0xffffff8c
# CHECK-OBJ64: bnec t1, 9, 0xffffffffffffff8c
# CHECK-ASM: bnec t1, 9, -128
# CHECK-ASM: encoding: [0x5b,0x60,0x93,0xb8]
nds.bnec t1, 9, -128

# CHECK-ASM-AND-OBJ: bfoz a0, a1, 9, 13
# CHECK-ASM: encoding: [0x5b,0xa5,0xd5,0x24]
nds.bfoz a0, a1, 9, 13

# CHECK-ASM-AND-OBJ: bfos t0, t1, 23, 3
# CHECK-ASM: encoding: [0xdb,0x32,0x33,0x5c]
nds.bfos t0, t1, 23, 3

# CHECK-ASM-AND-OBJ: lea.h t0, t1, t3
# CHECK-ASM: encoding: [0xdb,0x02,0xc3,0x0b]
nds.lea.h t0, t1, t3

# CHECK-ASM-AND-OBJ: lea.w a0, a1, a3
# CHECK-ASM: encoding: [0x5b,0x85,0xd5,0x0c]
nds.lea.w a0, a1, a3

# CHECK-ASM-AND-OBJ: lea.d s0, s1, s3
# CHECK-ASM: encoding: [0x5b,0x84,0x34,0x0f]
nds.lea.d s0, s1, s3

# CHECK-ASM-AND-OBJ: addigp t0, 9568
# CHECK-ASM: encoding: [0x8b,0x12,0x04,0x56]
nds.addigp t0, 0x2560

# CHECK-ASM-AND-OBJ: lbgp t0, -9568
# CHECK-ASM: encoding: [0x8b,0x82,0x1b,0xaa]
nds.lbgp t0, -0x2560

# CHECK-ASM-AND-OBJ: lbugp t0, 9568
# CHECK-ASM: encoding: [0x8b,0x22,0x04,0x56]
nds.lbugp t0, 0x2560

# CHECK-ASM-AND-OBJ: lhgp t0, -9568
# CHECK-ASM: encoding: [0xab,0x92,0x1b,0xaa]
nds.lhgp t0, -0x2560

# CHECK-ASM-AND-OBJ: lhugp t0, 9568
# CHECK-ASM: encoding: [0xab,0x52,0x04,0x56]
nds.lhugp t0, 0x2560

# CHECK-ASM-AND-OBJ: lwgp t0, -9568
# CHECK-ASM: encoding: [0xab,0xa2,0x3b,0xaa]
nds.lwgp t0, -0x2560

# CHECK-ASM-AND-OBJ: sbgp t0, 9568
# CHECK-ASM: encoding: [0x0b,0x30,0x54,0x56]
nds.sbgp t0, 0x2560

# CHECK-ASM-AND-OBJ: shgp t0, -9568
# CHECK-ASM: encoding: [0xab,0x80,0x5b,0xaa]
nds.shgp t0, -0x2560

# CHECK-ASM-AND-OBJ: swgp t0, 9568
# CHECK-ASM: encoding: [0x2b,0x40,0x54,0x56]
nds.swgp t0, 0x2560

# CHECK-ASM-AND-OBJ: ffb t0, t1, t3
# CHECK-ASM: encoding: [0xdb,0x02,0xc3,0x21]
nds.ffb t0, t1, t3

# CHECK-ASM-AND-OBJ: ffzmism a0, a1, a3
# CHECK-ASM: encoding: [0x5b,0x85,0xd5,0x22]
nds.ffzmism a0, a1, a3

# CHECK-ASM-AND-OBJ: ffmism s0, s1, s3
# CHECK-ASM: encoding: [0x5b,0x84,0x34,0x25]
nds.ffmism s0, s1, s3

# CHECK-ASM-AND-OBJ: flmism s0, s1, s3
# CHECK-ASM: encoding: [0x5b,0x84,0x34,0x27]
nds.flmism s0, s1, s3
