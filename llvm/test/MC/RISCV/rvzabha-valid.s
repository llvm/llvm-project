# RUN: llvm-mc %s -triple=riscv32 -mattr=+a,+zabha -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+a,+zabha -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+a,+zabha < %s \
# RUN:     | llvm-objdump --mattr=+a,+zabha -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+a,+zabha < %s \
# RUN:     | llvm-objdump --mattr=+a,+zabha -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: amoswap.b a4, ra, (s0)
# CHECK-ASM: encoding: [0x2f,0x07,0x14,0x08]
amoswap.b a4, ra, (s0)
# CHECK-ASM-AND-OBJ: amoadd.b a1, a2, (a3)
# CHECK-ASM: encoding: [0xaf,0x85,0xc6,0x00]
amoadd.b a1, a2, (a3)
# CHECK-ASM-AND-OBJ: amoxor.b a2, a3, (a4)
# CHECK-ASM: encoding: [0x2f,0x06,0xd7,0x20]
amoxor.b a2, a3, (a4)
# CHECK-ASM-AND-OBJ: amoand.b a3, a4, (a5)
# CHECK-ASM: encoding: [0xaf,0x86,0xe7,0x60]
amoand.b a3, a4, (a5)
# CHECK-ASM-AND-OBJ: amoor.b a4, a5, (a6)
# CHECK-ASM: encoding: [0x2f,0x07,0xf8,0x40]
amoor.b a4, a5, (a6)
# CHECK-ASM-AND-OBJ: amomin.b a5, a6, (a7)
# CHECK-ASM: encoding: [0xaf,0x87,0x08,0x81]
amomin.b a5, a6, (a7)
# CHECK-ASM-AND-OBJ: amomax.b s7, s6, (s5)
# CHECK-ASM: encoding: [0xaf,0x8b,0x6a,0xa1]
amomax.b s7, s6, (s5)
# CHECK-ASM-AND-OBJ: amominu.b s6, s5, (s4)
# CHECK-ASM: encoding: [0x2f,0x0b,0x5a,0xc1]
amominu.b s6, s5, (s4)
# CHECK-ASM-AND-OBJ: amomaxu.b s5, s4, (s3)
# CHECK-ASM: encoding: [0xaf,0x8a,0x49,0xe1]
amomaxu.b s5, s4, (s3)

# CHECK-ASM-AND-OBJ: amoswap.b.aq a4, ra, (s0)
# CHECK-ASM: encoding: [0x2f,0x07,0x14,0x0c]
amoswap.b.aq a4, ra, (s0)
# CHECK-ASM-AND-OBJ: amoadd.b.aq a1, a2, (a3)
# CHECK-ASM: encoding: [0xaf,0x85,0xc6,0x04]
amoadd.b.aq a1, a2, (a3)
# CHECK-ASM-AND-OBJ: amoxor.b.aq a2, a3, (a4)
# CHECK-ASM: encoding: [0x2f,0x06,0xd7,0x24]
amoxor.b.aq a2, a3, (a4)
# CHECK-ASM-AND-OBJ: amoand.b.aq a3, a4, (a5)
# CHECK-ASM: encoding: [0xaf,0x86,0xe7,0x64]
amoand.b.aq a3, a4, (a5)
# CHECK-ASM-AND-OBJ: amoor.b.aq a4, a5, (a6)
# CHECK-ASM: encoding: [0x2f,0x07,0xf8,0x44]
amoor.b.aq a4, a5, (a6)
# CHECK-ASM-AND-OBJ: amomin.b.aq a5, a6, (a7)
# CHECK-ASM: encoding: [0xaf,0x87,0x08,0x85]
amomin.b.aq a5, a6, (a7)
# CHECK-ASM-AND-OBJ: amomax.b.aq s7, s6, (s5)
# CHECK-ASM: encoding: [0xaf,0x8b,0x6a,0xa5]
amomax.b.aq s7, s6, (s5)
# CHECK-ASM-AND-OBJ: amominu.b.aq s6, s5, (s4)
# CHECK-ASM: encoding: [0x2f,0x0b,0x5a,0xc5]
amominu.b.aq s6, s5, (s4)
# CHECK-ASM-AND-OBJ: amomaxu.b.aq s5, s4, (s3)
# CHECK-ASM: encoding: [0xaf,0x8a,0x49,0xe5]
amomaxu.b.aq s5, s4, (s3)

# CHECK-ASM-AND-OBJ: amoswap.b.rl a4, ra, (s0)
# CHECK-ASM: encoding: [0x2f,0x07,0x14,0x0a]
amoswap.b.rl a4, ra, (s0)
# CHECK-ASM-AND-OBJ: amoadd.b.rl a1, a2, (a3)
# CHECK-ASM: encoding: [0xaf,0x85,0xc6,0x02]
amoadd.b.rl a1, a2, (a3)
# CHECK-ASM-AND-OBJ: amoxor.b.rl a2, a3, (a4)
# CHECK-ASM: encoding: [0x2f,0x06,0xd7,0x22]
amoxor.b.rl a2, a3, (a4)
# CHECK-ASM-AND-OBJ: amoand.b.rl a3, a4, (a5)
# CHECK-ASM: encoding: [0xaf,0x86,0xe7,0x62]
amoand.b.rl a3, a4, (a5)
# CHECK-ASM-AND-OBJ: amoor.b.rl a4, a5, (a6)
# CHECK-ASM: encoding: [0x2f,0x07,0xf8,0x42]
amoor.b.rl a4, a5, (a6)
# CHECK-ASM-AND-OBJ: amomin.b.rl a5, a6, (a7)
# CHECK-ASM: encoding: [0xaf,0x87,0x08,0x83]
amomin.b.rl a5, a6, (a7)
# CHECK-ASM-AND-OBJ: amomax.b.rl s7, s6, (s5)
# CHECK-ASM: encoding: [0xaf,0x8b,0x6a,0xa3]
amomax.b.rl s7, s6, (s5)
# CHECK-ASM-AND-OBJ: amominu.b.rl s6, s5, (s4)
# CHECK-ASM: encoding: [0x2f,0x0b,0x5a,0xc3]
amominu.b.rl s6, s5, (s4)
# CHECK-ASM-AND-OBJ: amomaxu.b.rl s5, s4, (s3)
# CHECK-ASM: encoding: [0xaf,0x8a,0x49,0xe3]
amomaxu.b.rl s5, s4, (s3)

# CHECK-ASM-AND-OBJ: amoswap.b.aqrl a4, ra, (s0)
# CHECK-ASM: encoding: [0x2f,0x07,0x14,0x0e]
amoswap.b.aqrl a4, ra, (s0)
# CHECK-ASM-AND-OBJ: amoadd.b.aqrl a1, a2, (a3)
# CHECK-ASM: encoding: [0xaf,0x85,0xc6,0x06]
amoadd.b.aqrl a1, a2, (a3)
# CHECK-ASM-AND-OBJ: amoxor.b.aqrl a2, a3, (a4)
# CHECK-ASM: encoding: [0x2f,0x06,0xd7,0x26]
amoxor.b.aqrl a2, a3, (a4)
# CHECK-ASM-AND-OBJ: amoand.b.aqrl a3, a4, (a5)
# CHECK-ASM: encoding: [0xaf,0x86,0xe7,0x66]
amoand.b.aqrl a3, a4, (a5)
# CHECK-ASM-AND-OBJ: amoor.b.aqrl a4, a5, (a6)
# CHECK-ASM: encoding: [0x2f,0x07,0xf8,0x46]
amoor.b.aqrl a4, a5, (a6)
# CHECK-ASM-AND-OBJ: amomin.b.aqrl a5, a6, (a7)
# CHECK-ASM: encoding: [0xaf,0x87,0x08,0x87]
amomin.b.aqrl a5, a6, (a7)
# CHECK-ASM-AND-OBJ: amomax.b.aqrl s7, s6, (s5)
# CHECK-ASM: encoding: [0xaf,0x8b,0x6a,0xa7]
amomax.b.aqrl s7, s6, (s5)
# CHECK-ASM-AND-OBJ: amominu.b.aqrl s6, s5, (s4)
# CHECK-ASM: encoding: [0x2f,0x0b,0x5a,0xc7]
amominu.b.aqrl s6, s5, (s4)
# CHECK-ASM-AND-OBJ: amomaxu.b.aqrl s5, s4, (s3)
# CHECK-ASM: encoding: [0xaf,0x8a,0x49,0xe7]
amomaxu.b.aqrl s5, s4, (s3)


# CHECK-ASM-AND-OBJ: amoswap.h a4, ra, (s0)
# CHECK-ASM: encoding: [0x2f,0x17,0x14,0x08]
amoswap.h a4, ra, (s0)
# CHECK-ASM-AND-OBJ: amoadd.h a1, a2, (a3)
# CHECK-ASM: encoding: [0xaf,0x95,0xc6,0x00]
amoadd.h a1, a2, (a3)
# CHECK-ASM-AND-OBJ: amoxor.h a2, a3, (a4)
# CHECK-ASM: encoding: [0x2f,0x16,0xd7,0x20]
amoxor.h a2, a3, (a4)
# CHECK-ASM-AND-OBJ: amoand.h a3, a4, (a5)
# CHECK-ASM: encoding: [0xaf,0x96,0xe7,0x60]
amoand.h a3, a4, (a5)
# CHECK-ASM-AND-OBJ: amoor.h a4, a5, (a6)
# CHECK-ASM: encoding: [0x2f,0x17,0xf8,0x40]
amoor.h a4, a5, (a6)
# CHECK-ASM-AND-OBJ: amomin.h a5, a6, (a7)
# CHECK-ASM: encoding: [0xaf,0x97,0x08,0x81]
amomin.h a5, a6, (a7)
# CHECK-ASM-AND-OBJ: amomax.h s7, s6, (s5)
# CHECK-ASM: encoding: [0xaf,0x9b,0x6a,0xa1]
amomax.h s7, s6, (s5)
# CHECK-ASM-AND-OBJ: amominu.h s6, s5, (s4)
# CHECK-ASM: encoding: [0x2f,0x1b,0x5a,0xc1]
amominu.h s6, s5, (s4)
# CHECK-ASM-AND-OBJ: amomaxu.h s5, s4, (s3)
# CHECK-ASM: encoding: [0xaf,0x9a,0x49,0xe1]
amomaxu.h s5, s4, (s3)

# CHECK-ASM-AND-OBJ: amoswap.h.aq a4, ra, (s0)
# CHECK-ASM: encoding: [0x2f,0x17,0x14,0x0c]
amoswap.h.aq a4, ra, (s0)
# CHECK-ASM-AND-OBJ: amoadd.h.aq a1, a2, (a3)
# CHECK-ASM: encoding: [0xaf,0x95,0xc6,0x04]
amoadd.h.aq a1, a2, (a3)
# CHECK-ASM-AND-OBJ: amoxor.h.aq a2, a3, (a4)
# CHECK-ASM: encoding: [0x2f,0x16,0xd7,0x24]
amoxor.h.aq a2, a3, (a4)
# CHECK-ASM-AND-OBJ: amoand.h.aq a3, a4, (a5)
# CHECK-ASM: encoding: [0xaf,0x96,0xe7,0x64]
amoand.h.aq a3, a4, (a5)
# CHECK-ASM-AND-OBJ: amoor.h.aq a4, a5, (a6)
# CHECK-ASM: encoding: [0x2f,0x17,0xf8,0x44]
amoor.h.aq a4, a5, (a6)
# CHECK-ASM-AND-OBJ: amomin.h.aq a5, a6, (a7)
# CHECK-ASM: encoding: [0xaf,0x97,0x08,0x85]
amomin.h.aq a5, a6, (a7)
# CHECK-ASM-AND-OBJ: amomax.h.aq s7, s6, (s5)
# CHECK-ASM: encoding: [0xaf,0x9b,0x6a,0xa5]
amomax.h.aq s7, s6, (s5)
# CHECK-ASM-AND-OBJ: amominu.h.aq s6, s5, (s4)
# CHECK-ASM: encoding: [0x2f,0x1b,0x5a,0xc5]
amominu.h.aq s6, s5, (s4)
# CHECK-ASM-AND-OBJ: amomaxu.h.aq s5, s4, (s3)
# CHECK-ASM: encoding: [0xaf,0x9a,0x49,0xe5]
amomaxu.h.aq s5, s4, (s3)

# CHECK-ASM-AND-OBJ: amoswap.h.rl a4, ra, (s0)
# CHECK-ASM: encoding: [0x2f,0x17,0x14,0x0a]
amoswap.h.rl a4, ra, (s0)
# CHECK-ASM-AND-OBJ: amoadd.h.rl a1, a2, (a3)
# CHECK-ASM: encoding: [0xaf,0x95,0xc6,0x02]
amoadd.h.rl a1, a2, (a3)
# CHECK-ASM-AND-OBJ: amoxor.h.rl a2, a3, (a4)
# CHECK-ASM: encoding: [0x2f,0x16,0xd7,0x22]
amoxor.h.rl a2, a3, (a4)
# CHECK-ASM-AND-OBJ: amoand.h.rl a3, a4, (a5)
# CHECK-ASM: encoding: [0xaf,0x96,0xe7,0x62]
amoand.h.rl a3, a4, (a5)
# CHECK-ASM-AND-OBJ: amoor.h.rl a4, a5, (a6)
# CHECK-ASM: encoding: [0x2f,0x17,0xf8,0x42]
amoor.h.rl a4, a5, (a6)
# CHECK-ASM-AND-OBJ: amomin.h.rl a5, a6, (a7)
# CHECK-ASM: encoding: [0xaf,0x97,0x08,0x83]
amomin.h.rl a5, a6, (a7)
# CHECK-ASM-AND-OBJ: amomax.h.rl s7, s6, (s5)
# CHECK-ASM: encoding: [0xaf,0x9b,0x6a,0xa3]
amomax.h.rl s7, s6, (s5)
# CHECK-ASM-AND-OBJ: amominu.h.rl s6, s5, (s4)
# CHECK-ASM: encoding: [0x2f,0x1b,0x5a,0xc3]
amominu.h.rl s6, s5, (s4)
# CHECK-ASM-AND-OBJ: amomaxu.h.rl s5, s4, (s3)
# CHECK-ASM: encoding: [0xaf,0x9a,0x49,0xe3]
amomaxu.h.rl s5, s4, (s3)

# CHECK-ASM-AND-OBJ: amoswap.h.aqrl a4, ra, (s0)
# CHECK-ASM: encoding: [0x2f,0x17,0x14,0x0e]
amoswap.h.aqrl a4, ra, (s0)
# CHECK-ASM-AND-OBJ: amoadd.h.aqrl a1, a2, (a3)
# CHECK-ASM: encoding: [0xaf,0x95,0xc6,0x06]
amoadd.h.aqrl a1, a2, (a3)
# CHECK-ASM-AND-OBJ: amoxor.h.aqrl a2, a3, (a4)
# CHECK-ASM: encoding: [0x2f,0x16,0xd7,0x26]
amoxor.h.aqrl a2, a3, (a4)
# CHECK-ASM-AND-OBJ: amoand.h.aqrl a3, a4, (a5)
# CHECK-ASM: encoding: [0xaf,0x96,0xe7,0x66]
amoand.h.aqrl a3, a4, (a5)
# CHECK-ASM-AND-OBJ: amoor.h.aqrl a4, a5, (a6)
# CHECK-ASM: encoding: [0x2f,0x17,0xf8,0x46]
amoor.h.aqrl a4, a5, (a6)
# CHECK-ASM-AND-OBJ: amomin.h.aqrl a5, a6, (a7)
# CHECK-ASM: encoding: [0xaf,0x97,0x08,0x87]
amomin.h.aqrl a5, a6, (a7)
# CHECK-ASM-AND-OBJ: amomax.h.aqrl s7, s6, (s5)
# CHECK-ASM: encoding: [0xaf,0x9b,0x6a,0xa7]
amomax.h.aqrl s7, s6, (s5)
# CHECK-ASM-AND-OBJ: amominu.h.aqrl s6, s5, (s4)
# CHECK-ASM: encoding: [0x2f,0x1b,0x5a,0xc7]
amominu.h.aqrl s6, s5, (s4)
# CHECK-ASM-AND-OBJ: amomaxu.h.aqrl s5, s4, (s3)
# CHECK-ASM: encoding: [0xaf,0x9a,0x49,0xe7]
amomaxu.h.aqrl s5, s4, (s3)
