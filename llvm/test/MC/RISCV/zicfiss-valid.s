# RUN: llvm-mc %s -triple=riscv32 -mattr=+a,+experimental-zicfiss -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+a,+experimental-zicfiss < %s \
# RUN:     | llvm-objdump --mattr=+a,+experimental-zicfiss -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -defsym=RV64=1 -mattr=+a,+experimental-zicfiss -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM-RV64,CHECK-ASM,CHECK-ASM-AND-OBJ-RV64,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -defsym=RV64=1 -mattr=+a,+experimental-zicfiss < %s \
# RUN:     | llvm-objdump --mattr=+a,+experimental-zicfiss -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefixes=CHECK-ASM-AND-OBJ-RV64,CHECK-ASM-AND-OBJ %s
#
# RUN: not llvm-mc -triple riscv32 -riscv-no-aliases -show-encoding < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-NO-EXT %s
# RUN: not llvm-mc -triple riscv64 -defsym=RV64=1 -riscv-no-aliases -show-encoding < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-NO-EXT-RV64 %s

# CHECK-ASM-AND-OBJ: sspopchk ra
# CHECK-ASM: encoding: [0x73,0xc0,0xc0,0xcd]
# CHECK-NO-EXT: error: instruction requires the following: 'Zicfiss' (Shadow stack)
sspopchk x1

# CHECK-ASM-AND-OBJ: sspopchk ra
# CHECK-ASM: encoding: [0x73,0xc0,0xc0,0xcd]
# CHECK-NO-EXT: error: instruction requires the following: 'Zicfiss' (Shadow stack)
sspopchk ra

# CHECK-ASM-AND-OBJ: sspopchk t0
# CHECK-ASM: encoding: [0x73,0xc0,0xc2,0xcd]
# CHECK-NO-EXT: error: instruction requires the following: 'Zicfiss' (Shadow stack)
sspopchk x5

# CHECK-ASM-AND-OBJ: sspopchk t0
# CHECK-ASM: encoding: [0x73,0xc0,0xc2,0xcd]
# CHECK-NO-EXT: error: instruction requires the following: 'Zicfiss' (Shadow stack)
sspopchk t0

# CHECK-ASM-AND-OBJ: sspush ra
# CHECK-ASM: encoding: [0x73,0x40,0x10,0xce]
# CHECK-NO-EXT: error: instruction requires the following: 'Zicfiss' (Shadow stack)
sspush x1

# CHECK-ASM-AND-OBJ: sspush ra
# CHECK-ASM: encoding: [0x73,0x40,0x10,0xce]
# CHECK-NO-EXT: error: instruction requires the following: 'Zicfiss' (Shadow stack)
sspush ra

# check-asm-and-obj: sspush t0
# check-asm: encoding: [0x73,0x40,0x50,0xce]
# check-no-ext: error: instruction requires the following: 'Zicfiss' (Shadow stack)
sspush x5

# check-asm-and-obj: sspush t0
# check-asm: encoding: [0x73,0x40,0x50,0xce]
# check-no-ext: error: instruction requires the following: 'Zicfiss' (Shadow stack)
sspush t0

# CHECK-ASM-AND-OBJ: ssrdp ra
# CHECK-ASM: encoding: [0xf3,0x40,0xc0,0xcd]
# CHECK-NO-EXT: error: instruction requires the following: 'Zicfiss' (Shadow stack)
ssrdp ra

# CHECK-ASM-AND-OBJ: ssamoswap.w a4, ra, (s0)
# CHECK-ASM: encoding: [0x2f,0x27,0x14,0x48]
# CHECK-NO-EXT: error: instruction requires the following: 'Zicfiss' (Shadow stack)
ssamoswap.w a4, ra, (s0)

# CHECK-ASM-AND-OBJ: ssamoswap.w.aq a4, ra, (s0)
# CHECK-ASM: encoding: [0x2f,0x27,0x14,0x4c]
# CHECK-NO-EXT: error: instruction requires the following: 'Zicfiss' (Shadow stack)
ssamoswap.w.aq a4, ra, (s0)

# CHECK-ASM-AND-OBJ: ssamoswap.w.rl a4, ra, (s0)
# CHECK-ASM: encoding: [0x2f,0x27,0x14,0x4a]
# CHECK-NO-EXT: error: instruction requires the following: 'Zicfiss' (Shadow stack)
ssamoswap.w.rl a4, ra, (s0)

# CHECK-ASM-AND-OBJ: ssamoswap.w.aqrl a4, ra, (s0)
# CHECK-ASM: encoding: [0x2f,0x27,0x14,0x4e]
# CHECK-NO-EXT: error: instruction requires the following: 'Zicfiss' (Shadow stack)
ssamoswap.w.aqrl a4, ra, (s0)

.ifdef RV64
# CHECK-ASM-AND-OBJ-RV64: ssamoswap.d a4, ra, (s0)
# CHECK-ASM-RV64: encoding: [0x2f,0x37,0x14,0x48]
# CHECK-NO-EXT-RV64: error: instruction requires the following: 'Zicfiss' (Shadow stack)
ssamoswap.d a4, ra, (s0)

# CHECK-ASM-AND-OBJ-RV64: ssamoswap.d.aq a4, ra, (s0)
# CHECK-ASM-RV64: encoding: [0x2f,0x37,0x14,0x4c]
# CHECK-NO-EXT-RV64: error: instruction requires the following: 'Zicfiss' (Shadow stack)
ssamoswap.d.aq a4, ra, (s0)

# CHECK-ASM-AND-OBJ-RV64: ssamoswap.d.rl a4, ra, (s0)
# CHECK-ASM-RV64: encoding: [0x2f,0x37,0x14,0x4a]
# CHECK-NO-EXT-RV64: error: instruction requires the following: 'Zicfiss' (Shadow stack)
ssamoswap.d.rl a4, ra, (s0)

# CHECK-ASM-AND-OBJ-RV64: ssamoswap.d.aqrl a4, ra, (s0)
# CHECK-ASM-RV64: encoding: [0x2f,0x37,0x14,0x4e]
# CHECK-NO-EXT-RV64: error: instruction requires the following: 'Zicfiss' (Shadow stack)
ssamoswap.d.aqrl a4, ra, (s0)
.endif
