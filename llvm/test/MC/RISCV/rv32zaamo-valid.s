# RUN: llvm-mc %s -triple=riscv32 -mattr=+a -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+a -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+a < %s \
# RUN:     | llvm-objdump --mattr=+a -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+a < %s \
# RUN:     | llvm-objdump --mattr=+a -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: amoswap.w a4, ra, (s0)
# CHECK-ASM: encoding: [0x2f,0x27,0x14,0x08]
amoswap.w a4, ra, (s0)
# CHECK-ASM-AND-OBJ: amoadd.w a1, a2, (a3)
# CHECK-ASM: encoding: [0xaf,0xa5,0xc6,0x00]
amoadd.w a1, a2, (a3)
# CHECK-ASM-AND-OBJ: amoxor.w a2, a3, (a4)
# CHECK-ASM: encoding: [0x2f,0x26,0xd7,0x20]
amoxor.w a2, a3, (a4)
# CHECK-ASM-AND-OBJ: amoand.w a3, a4, (a5)
# CHECK-ASM: encoding: [0xaf,0xa6,0xe7,0x60]
amoand.w a3, a4, (a5)
# CHECK-ASM-AND-OBJ: amoor.w a4, a5, (a6)
# CHECK-ASM: encoding: [0x2f,0x27,0xf8,0x40]
amoor.w a4, a5, (a6)
# CHECK-ASM-AND-OBJ: amomin.w a5, a6, (a7)
# CHECK-ASM: encoding: [0xaf,0xa7,0x08,0x81]
amomin.w a5, a6, (a7)
# CHECK-ASM-AND-OBJ: amomax.w s7, s6, (s5)
# CHECK-ASM: encoding: [0xaf,0xab,0x6a,0xa1]
amomax.w s7, s6, (s5)
# CHECK-ASM-AND-OBJ: amominu.w s6, s5, (s4)
# CHECK-ASM: encoding: [0x2f,0x2b,0x5a,0xc1]
amominu.w s6, s5, (s4)
# CHECK-ASM-AND-OBJ: amomaxu.w s5, s4, (s3)
# CHECK-ASM: encoding: [0xaf,0xaa,0x49,0xe1]
amomaxu.w s5, s4, (s3)

# CHECK-ASM-AND-OBJ: amoswap.w.aq a4, ra, (s0)
# CHECK-ASM: encoding: [0x2f,0x27,0x14,0x0c]
amoswap.w.aq a4, ra, (s0)
# CHECK-ASM-AND-OBJ: amoadd.w.aq a1, a2, (a3)
# CHECK-ASM: encoding: [0xaf,0xa5,0xc6,0x04]
amoadd.w.aq a1, a2, (a3)
# CHECK-ASM-AND-OBJ: amoxor.w.aq a2, a3, (a4)
# CHECK-ASM: encoding: [0x2f,0x26,0xd7,0x24]
amoxor.w.aq a2, a3, (a4)
# CHECK-ASM-AND-OBJ: amoand.w.aq a3, a4, (a5)
# CHECK-ASM: encoding: [0xaf,0xa6,0xe7,0x64]
amoand.w.aq a3, a4, (a5)
# CHECK-ASM-AND-OBJ: amoor.w.aq a4, a5, (a6)
# CHECK-ASM: encoding: [0x2f,0x27,0xf8,0x44]
amoor.w.aq a4, a5, (a6)
# CHECK-ASM-AND-OBJ: amomin.w.aq a5, a6, (a7)
# CHECK-ASM: encoding: [0xaf,0xa7,0x08,0x85]
amomin.w.aq a5, a6, (a7)
# CHECK-ASM-AND-OBJ: amomax.w.aq s7, s6, (s5)
# CHECK-ASM: encoding: [0xaf,0xab,0x6a,0xa5]
amomax.w.aq s7, s6, (s5)
# CHECK-ASM-AND-OBJ: amominu.w.aq s6, s5, (s4)
# CHECK-ASM: encoding: [0x2f,0x2b,0x5a,0xc5]
amominu.w.aq s6, s5, (s4)
# CHECK-ASM-AND-OBJ: amomaxu.w.aq s5, s4, (s3)
# CHECK-ASM: encoding: [0xaf,0xaa,0x49,0xe5]
amomaxu.w.aq s5, s4, (s3)

# CHECK-ASM-AND-OBJ: amoswap.w.rl a4, ra, (s0)
# CHECK-ASM: encoding: [0x2f,0x27,0x14,0x0a]
amoswap.w.rl a4, ra, (s0)
# CHECK-ASM-AND-OBJ: amoadd.w.rl a1, a2, (a3)
# CHECK-ASM: encoding: [0xaf,0xa5,0xc6,0x02]
amoadd.w.rl a1, a2, (a3)
# CHECK-ASM-AND-OBJ: amoxor.w.rl a2, a3, (a4)
# CHECK-ASM: encoding: [0x2f,0x26,0xd7,0x22]
amoxor.w.rl a2, a3, (a4)
# CHECK-ASM-AND-OBJ: amoand.w.rl a3, a4, (a5)
# CHECK-ASM: encoding: [0xaf,0xa6,0xe7,0x62]
amoand.w.rl a3, a4, (a5)
# CHECK-ASM-AND-OBJ: amoor.w.rl a4, a5, (a6)
# CHECK-ASM: encoding: [0x2f,0x27,0xf8,0x42]
amoor.w.rl a4, a5, (a6)
# CHECK-ASM-AND-OBJ: amomin.w.rl a5, a6, (a7)
# CHECK-ASM: encoding: [0xaf,0xa7,0x08,0x83]
amomin.w.rl a5, a6, (a7)
# CHECK-ASM-AND-OBJ: amomax.w.rl s7, s6, (s5)
# CHECK-ASM: encoding: [0xaf,0xab,0x6a,0xa3]
amomax.w.rl s7, s6, (s5)
# CHECK-ASM-AND-OBJ: amominu.w.rl s6, s5, (s4)
# CHECK-ASM: encoding: [0x2f,0x2b,0x5a,0xc3]
amominu.w.rl s6, s5, (s4)
# CHECK-ASM-AND-OBJ: amomaxu.w.rl s5, s4, (s3)
# CHECK-ASM: encoding: [0xaf,0xaa,0x49,0xe3]
amomaxu.w.rl s5, s4, (s3)

# CHECK-ASM-AND-OBJ: amoswap.w.aqrl a4, ra, (s0)
# CHECK-ASM: encoding: [0x2f,0x27,0x14,0x0e]
amoswap.w.aqrl a4, ra, (s0)
# CHECK-ASM-AND-OBJ: amoadd.w.aqrl a1, a2, (a3)
# CHECK-ASM: encoding: [0xaf,0xa5,0xc6,0x06]
amoadd.w.aqrl a1, a2, (a3)
# CHECK-ASM-AND-OBJ: amoxor.w.aqrl a2, a3, (a4)
# CHECK-ASM: encoding: [0x2f,0x26,0xd7,0x26]
amoxor.w.aqrl a2, a3, (a4)
# CHECK-ASM-AND-OBJ: amoand.w.aqrl a3, a4, (a5)
# CHECK-ASM: encoding: [0xaf,0xa6,0xe7,0x66]
amoand.w.aqrl a3, a4, (a5)
# CHECK-ASM-AND-OBJ: amoor.w.aqrl a4, a5, (a6)
# CHECK-ASM: encoding: [0x2f,0x27,0xf8,0x46]
amoor.w.aqrl a4, a5, (a6)
# CHECK-ASM-AND-OBJ: amomin.w.aqrl a5, a6, (a7)
# CHECK-ASM: encoding: [0xaf,0xa7,0x08,0x87]
amomin.w.aqrl a5, a6, (a7)
# CHECK-ASM-AND-OBJ: amomax.w.aqrl s7, s6, (s5)
# CHECK-ASM: encoding: [0xaf,0xab,0x6a,0xa7]
amomax.w.aqrl s7, s6, (s5)
# CHECK-ASM-AND-OBJ: amominu.w.aqrl s6, s5, (s4)
# CHECK-ASM: encoding: [0x2f,0x2b,0x5a,0xc7]
amominu.w.aqrl s6, s5, (s4)
# CHECK-ASM-AND-OBJ: amomaxu.w.aqrl s5, s4, (s3)
# CHECK-ASM: encoding: [0xaf,0xaa,0x49,0xe7]
amomaxu.w.aqrl s5, s4, (s3)
