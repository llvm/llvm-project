# RUN: llvm-mc %s -triple=riscv64 -mattr=+a -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+a < %s \
# RUN:     | llvm-objdump --mattr=+a -M no-aliases -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-OBJ,CHECK-ASM-AND-OBJ %s
#
# RUN: not llvm-mc -triple riscv32 -mattr=+a < %s 2>&1 \
# RUN:     | FileCheck -check-prefix=CHECK-RV32 %s

# CHECK-ASM-AND-OBJ: lr.d t0, (t1)
# CHECK-ASM: encoding: [0xaf,0x32,0x03,0x10]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
lr.d t0, (t1)
# CHECK-ASM-AND-OBJ: lr.d.aq t1, (t2)
# CHECK-ASM: encoding: [0x2f,0xb3,0x03,0x14]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
lr.d.aq t1, (t2)
# CHECK-ASM-AND-OBJ: lr.d.rl t2, (t3)
# CHECK-ASM: encoding: [0xaf,0x33,0x0e,0x12]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
lr.d.rl t2, (t3)
# CHECK-ASM-AND-OBJ: lr.d.aqrl t3, (t4)
# CHECK-ASM: encoding: [0x2f,0xbe,0x0e,0x16]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
lr.d.aqrl t3, (t4)

# CHECK-ASM-AND-OBJ: sc.d t6, t5, (t4)
# CHECK-ASM: encoding: [0xaf,0xbf,0xee,0x19]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
sc.d t6, t5, (t4)
# CHECK-ASM-AND-OBJ: sc.d.aq t5, t4, (t3)
# CHECK-ASM: encoding: [0x2f,0x3f,0xde,0x1d]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
sc.d.aq t5, t4, (t3)
# CHECK-ASM-AND-OBJ: sc.d.rl t4, t3, (t2)
# CHECK-ASM: encoding: [0xaf,0xbe,0xc3,0x1b]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
sc.d.rl t4, t3, (t2)
# CHECK-ASM-AND-OBJ: sc.d.aqrl t3, t2, (t1)
# CHECK-ASM: encoding: [0x2f,0x3e,0x73,0x1e]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
sc.d.aqrl t3, t2, (t1)

# CHECK-ASM-AND-OBJ: amoswap.d a4, ra, (s0)
# CHECK-ASM: encoding: [0x2f,0x37,0x14,0x08]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
amoswap.d a4, ra, (s0)
# CHECK-ASM-AND-OBJ: amoadd.d a1, a2, (a3)
# CHECK-ASM: encoding: [0xaf,0xb5,0xc6,0x00]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
amoadd.d a1, a2, (a3)
# CHECK-ASM-AND-OBJ: amoxor.d a2, a3, (a4)
# CHECK-ASM: encoding: [0x2f,0x36,0xd7,0x20]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
amoxor.d a2, a3, (a4)
# CHECK-ASM-AND-OBJ: amoand.d a3, a4, (a5)
# CHECK-ASM: encoding: [0xaf,0xb6,0xe7,0x60]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
amoand.d a3, a4, (a5)
# CHECK-ASM-AND-OBJ: amoor.d a4, a5, (a6)
# CHECK-ASM: encoding: [0x2f,0x37,0xf8,0x40]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
amoor.d a4, a5, (a6)
# CHECK-ASM-AND-OBJ: amomin.d a5, a6, (a7)
# CHECK-ASM: encoding: [0xaf,0xb7,0x08,0x81]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
amomin.d a5, a6, (a7)
# CHECK-ASM-AND-OBJ: amomax.d s7, s6, (s5)
# CHECK-ASM: encoding: [0xaf,0xbb,0x6a,0xa1]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
amomax.d s7, s6, (s5)
# CHECK-ASM-AND-OBJ: amominu.d s6, s5, (s4)
# CHECK-ASM: encoding: [0x2f,0x3b,0x5a,0xc1]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
amominu.d s6, s5, (s4)
# CHECK-ASM-AND-OBJ: amomaxu.d s5, s4, (s3)
# CHECK-ASM: encoding: [0xaf,0xba,0x49,0xe1]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
amomaxu.d s5, s4, (s3)


# CHECK-ASM-AND-OBJ: amoswap.d.aq a4, ra, (s0)
# CHECK-ASM: encoding: [0x2f,0x37,0x14,0x0c]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
amoswap.d.aq a4, ra, (s0)
# CHECK-ASM-AND-OBJ: amoadd.d.aq a1, a2, (a3)
# CHECK-ASM: encoding: [0xaf,0xb5,0xc6,0x04]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
amoadd.d.aq a1, a2, (a3)
# CHECK-ASM-AND-OBJ: amoxor.d.aq a2, a3, (a4)
# CHECK-ASM: encoding: [0x2f,0x36,0xd7,0x24]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
amoxor.d.aq a2, a3, (a4)
# CHECK-ASM-AND-OBJ: amoand.d.aq a3, a4, (a5)
# CHECK-ASM: encoding: [0xaf,0xb6,0xe7,0x64]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
amoand.d.aq a3, a4, (a5)
# CHECK-ASM-AND-OBJ: amoor.d.aq a4, a5, (a6)
# CHECK-ASM: encoding: [0x2f,0x37,0xf8,0x44]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
amoor.d.aq a4, a5, (a6)
# CHECK-ASM-AND-OBJ: amomin.d.aq a5, a6, (a7)
# CHECK-ASM: encoding: [0xaf,0xb7,0x08,0x85]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
amomin.d.aq a5, a6, (a7)
# CHECK-ASM-AND-OBJ: amomax.d.aq s7, s6, (s5)
# CHECK-ASM: encoding: [0xaf,0xbb,0x6a,0xa5]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
amomax.d.aq s7, s6, (s5)
# CHECK-ASM-AND-OBJ: amominu.d.aq s6, s5, (s4)
# CHECK-ASM: encoding: [0x2f,0x3b,0x5a,0xc5]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
amominu.d.aq s6, s5, (s4)
# CHECK-ASM-AND-OBJ: amomaxu.d.aq s5, s4, (s3)
# CHECK-ASM: encoding: [0xaf,0xba,0x49,0xe5]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
amomaxu.d.aq s5, s4, (s3)

# CHECK-ASM-AND-OBJ: amoswap.d.rl a4, ra, (s0)
# CHECK-ASM: encoding: [0x2f,0x37,0x14,0x0a]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
amoswap.d.rl a4, ra, (s0)
# CHECK-ASM-AND-OBJ: amoadd.d.rl a1, a2, (a3)
# CHECK-ASM: encoding: [0xaf,0xb5,0xc6,0x02]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
amoadd.d.rl a1, a2, (a3)
# CHECK-ASM-AND-OBJ: amoxor.d.rl a2, a3, (a4)
# CHECK-ASM: encoding: [0x2f,0x36,0xd7,0x22]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
amoxor.d.rl a2, a3, (a4)
# CHECK-ASM-AND-OBJ: amoand.d.rl a3, a4, (a5)
# CHECK-ASM: encoding: [0xaf,0xb6,0xe7,0x62]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
amoand.d.rl a3, a4, (a5)
# CHECK-ASM-AND-OBJ: amoor.d.rl a4, a5, (a6)
# CHECK-ASM: encoding: [0x2f,0x37,0xf8,0x42]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
amoor.d.rl a4, a5, (a6)
# CHECK-ASM-AND-OBJ: amomin.d.rl a5, a6, (a7)
# CHECK-ASM: encoding: [0xaf,0xb7,0x08,0x83]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
amomin.d.rl a5, a6, (a7)
# CHECK-ASM-AND-OBJ: amomax.d.rl s7, s6, (s5)
# CHECK-ASM: encoding: [0xaf,0xbb,0x6a,0xa3]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
amomax.d.rl s7, s6, (s5)
# CHECK-ASM-AND-OBJ: amominu.d.rl s6, s5, (s4)
# CHECK-ASM: encoding: [0x2f,0x3b,0x5a,0xc3]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
amominu.d.rl s6, s5, (s4)
# CHECK-ASM-AND-OBJ: amomaxu.d.rl s5, s4, (s3)
# CHECK-ASM: encoding: [0xaf,0xba,0x49,0xe3]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
amomaxu.d.rl s5, s4, (s3)

# CHECK-ASM-AND-OBJ: amoswap.d.aqrl a4, ra, (s0)
# CHECK-ASM: encoding: [0x2f,0x37,0x14,0x0e]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
amoswap.d.aqrl a4, ra, (s0)
# CHECK-ASM-AND-OBJ: amoadd.d.aqrl a1, a2, (a3)
# CHECK-ASM: encoding: [0xaf,0xb5,0xc6,0x06]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
amoadd.d.aqrl a1, a2, (a3)
# CHECK-ASM-AND-OBJ: amoxor.d.aqrl a2, a3, (a4)
# CHECK-ASM: encoding: [0x2f,0x36,0xd7,0x26]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
amoxor.d.aqrl a2, a3, (a4)
# CHECK-ASM-AND-OBJ: amoand.d.aqrl a3, a4, (a5)
# CHECK-ASM: encoding: [0xaf,0xb6,0xe7,0x66]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
amoand.d.aqrl a3, a4, (a5)
# CHECK-ASM-AND-OBJ: amoor.d.aqrl a4, a5, (a6)
# CHECK-ASM: encoding: [0x2f,0x37,0xf8,0x46]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
amoor.d.aqrl a4, a5, (a6)
# CHECK-ASM-AND-OBJ: amomin.d.aqrl a5, a6, (a7)
# CHECK-ASM: encoding: [0xaf,0xb7,0x08,0x87]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
amomin.d.aqrl a5, a6, (a7)
# CHECK-ASM-AND-OBJ: amomax.d.aqrl s7, s6, (s5)
# CHECK-ASM: encoding: [0xaf,0xbb,0x6a,0xa7]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
amomax.d.aqrl s7, s6, (s5)
# CHECK-ASM-AND-OBJ: amominu.d.aqrl s6, s5, (s4)
# CHECK-ASM: encoding: [0x2f,0x3b,0x5a,0xc7]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
amominu.d.aqrl s6, s5, (s4)
# CHECK-ASM-AND-OBJ: amomaxu.d.aqrl s5, s4, (s3)
# CHECK-ASM: encoding: [0xaf,0xba,0x49,0xe7]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
amomaxu.d.aqrl s5, s4, (s3)
