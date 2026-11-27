# RUN: llvm-mc %s -triple=riscv64 -mattr=+a -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+a < %s \
# RUN:     | llvm-objdump --mattr=+a -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
#
# RUN: llvm-mc --verify -triple riscv32 -mattr=+a < %s
#
# RUN: llvm-mc %s -triple=riscv64 -mattr=+zaamo -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+zaamo < %s \
# RUN:     | llvm-objdump --mattr=+zaamo -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
#
# RUN: llvm-mc --verify -triple riscv32 -mattr=+zaamo < %s

# CHECK-ASM-AND-OBJ: amoswap.d a4, ra, (s0)
# CHECK-ASM: encoding: [0x2f,0x37,0x14,0x08]
amoswap.d a4, ra, (s0) # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: amoadd.d a1, a2, (a3)
# CHECK-ASM: encoding: [0xaf,0xb5,0xc6,0x00]
amoadd.d a1, a2, (a3) # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: amoxor.d a2, a3, (a4)
# CHECK-ASM: encoding: [0x2f,0x36,0xd7,0x20]
amoxor.d a2, a3, (a4) # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: amoand.d a3, a4, (a5)
# CHECK-ASM: encoding: [0xaf,0xb6,0xe7,0x60]
amoand.d a3, a4, (a5) # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: amoor.d a4, a5, (a6)
# CHECK-ASM: encoding: [0x2f,0x37,0xf8,0x40]
amoor.d a4, a5, (a6) # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: amomin.d a5, a6, (a7)
# CHECK-ASM: encoding: [0xaf,0xb7,0x08,0x81]
amomin.d a5, a6, (a7) # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: amomax.d s7, s6, (s5)
# CHECK-ASM: encoding: [0xaf,0xbb,0x6a,0xa1]
amomax.d s7, s6, (s5) # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: amominu.d s6, s5, (s4)
# CHECK-ASM: encoding: [0x2f,0x3b,0x5a,0xc1]
amominu.d s6, s5, (s4) # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: amomaxu.d s5, s4, (s3)
# CHECK-ASM: encoding: [0xaf,0xba,0x49,0xe1]
amomaxu.d s5, s4, (s3) # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}


# CHECK-ASM-AND-OBJ: amoswap.d.aq a4, ra, (s0)
# CHECK-ASM: encoding: [0x2f,0x37,0x14,0x0c]
amoswap.d.aq a4, ra, (s0) # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: amoadd.d.aq a1, a2, (a3)
# CHECK-ASM: encoding: [0xaf,0xb5,0xc6,0x04]
amoadd.d.aq a1, a2, (a3) # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: amoxor.d.aq a2, a3, (a4)
# CHECK-ASM: encoding: [0x2f,0x36,0xd7,0x24]
amoxor.d.aq a2, a3, (a4) # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: amoand.d.aq a3, a4, (a5)
# CHECK-ASM: encoding: [0xaf,0xb6,0xe7,0x64]
amoand.d.aq a3, a4, (a5) # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: amoor.d.aq a4, a5, (a6)
# CHECK-ASM: encoding: [0x2f,0x37,0xf8,0x44]
amoor.d.aq a4, a5, (a6) # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: amomin.d.aq a5, a6, (a7)
# CHECK-ASM: encoding: [0xaf,0xb7,0x08,0x85]
amomin.d.aq a5, a6, (a7) # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: amomax.d.aq s7, s6, (s5)
# CHECK-ASM: encoding: [0xaf,0xbb,0x6a,0xa5]
amomax.d.aq s7, s6, (s5) # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: amominu.d.aq s6, s5, (s4)
# CHECK-ASM: encoding: [0x2f,0x3b,0x5a,0xc5]
amominu.d.aq s6, s5, (s4) # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: amomaxu.d.aq s5, s4, (s3)
# CHECK-ASM: encoding: [0xaf,0xba,0x49,0xe5]
amomaxu.d.aq s5, s4, (s3) # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}

# CHECK-ASM-AND-OBJ: amoswap.d.rl a4, ra, (s0)
# CHECK-ASM: encoding: [0x2f,0x37,0x14,0x0a]
amoswap.d.rl a4, ra, (s0) # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: amoadd.d.rl a1, a2, (a3)
# CHECK-ASM: encoding: [0xaf,0xb5,0xc6,0x02]
amoadd.d.rl a1, a2, (a3) # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: amoxor.d.rl a2, a3, (a4)
# CHECK-ASM: encoding: [0x2f,0x36,0xd7,0x22]
amoxor.d.rl a2, a3, (a4) # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: amoand.d.rl a3, a4, (a5)
# CHECK-ASM: encoding: [0xaf,0xb6,0xe7,0x62]
amoand.d.rl a3, a4, (a5) # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: amoor.d.rl a4, a5, (a6)
# CHECK-ASM: encoding: [0x2f,0x37,0xf8,0x42]
amoor.d.rl a4, a5, (a6) # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: amomin.d.rl a5, a6, (a7)
# CHECK-ASM: encoding: [0xaf,0xb7,0x08,0x83]
amomin.d.rl a5, a6, (a7) # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: amomax.d.rl s7, s6, (s5)
# CHECK-ASM: encoding: [0xaf,0xbb,0x6a,0xa3]
amomax.d.rl s7, s6, (s5) # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: amominu.d.rl s6, s5, (s4)
# CHECK-ASM: encoding: [0x2f,0x3b,0x5a,0xc3]
amominu.d.rl s6, s5, (s4) # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: amomaxu.d.rl s5, s4, (s3)
# CHECK-ASM: encoding: [0xaf,0xba,0x49,0xe3]
amomaxu.d.rl s5, s4, (s3) # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}

# CHECK-ASM-AND-OBJ: amoswap.d.aqrl a4, ra, (s0)
# CHECK-ASM: encoding: [0x2f,0x37,0x14,0x0e]
amoswap.d.aqrl a4, ra, (s0) # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: amoadd.d.aqrl a1, a2, (a3)
# CHECK-ASM: encoding: [0xaf,0xb5,0xc6,0x06]
amoadd.d.aqrl a1, a2, (a3) # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: amoxor.d.aqrl a2, a3, (a4)
# CHECK-ASM: encoding: [0x2f,0x36,0xd7,0x26]
amoxor.d.aqrl a2, a3, (a4) # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: amoand.d.aqrl a3, a4, (a5)
# CHECK-ASM: encoding: [0xaf,0xb6,0xe7,0x66]
amoand.d.aqrl a3, a4, (a5) # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: amoor.d.aqrl a4, a5, (a6)
# CHECK-ASM: encoding: [0x2f,0x37,0xf8,0x46]
amoor.d.aqrl a4, a5, (a6) # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: amomin.d.aqrl a5, a6, (a7)
# CHECK-ASM: encoding: [0xaf,0xb7,0x08,0x87]
amomin.d.aqrl a5, a6, (a7) # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: amomax.d.aqrl s7, s6, (s5)
# CHECK-ASM: encoding: [0xaf,0xbb,0x6a,0xa7]
amomax.d.aqrl s7, s6, (s5) # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: amominu.d.aqrl s6, s5, (s4)
# CHECK-ASM: encoding: [0x2f,0x3b,0x5a,0xc7]
amominu.d.aqrl s6, s5, (s4) # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: amomaxu.d.aqrl s5, s4, (s3)
# CHECK-ASM: encoding: [0xaf,0xba,0x49,0xe7]
amomaxu.d.aqrl s5, s4, (s3) # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
