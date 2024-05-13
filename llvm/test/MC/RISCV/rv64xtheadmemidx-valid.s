# RUN: llvm-mc %s -triple=riscv64 -mattr=+xtheadmemidx -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+xtheadmemidx < %s \
# RUN:     | llvm-objdump --mattr=+xtheadmemidx -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: th.ldia
# CHECK-ASM: encoding: [0x0b,0xc5,0x05,0x78]
th.ldia		a0, (a1), 0, 0

# CHECK-ASM-AND-OBJ: th.ldib
# CHECK-ASM: encoding: [0x0b,0xc5,0xf5,0x6a]
th.ldib		a0, (a1), 15, 1

# CHECK-ASM-AND-OBJ: th.lwia
# CHECK-ASM: encoding: [0x0b,0xc5,0x05,0x5c]
th.lwia		a0, (a1), 0, 2

# CHECK-ASM-AND-OBJ: th.lwib
# CHECK-ASM: encoding: [0x0b,0xc5,0x05,0x4f]
th.lwib		a0, (a1), -16, 3

# CHECK-ASM-AND-OBJ: th.lwuia
# CHECK-ASM: encoding: [0x0b,0xc5,0x05,0xd8]
th.lwuia	a0, (a1), 0, 0

# CHECK-ASM-AND-OBJ: th.lwuib
# CHECK-ASM: encoding: [0x0b,0xc5,0xf5,0xca]
th.lwuib	a0, (a1), 15, 1

# CHECK-ASM-AND-OBJ: th.lhia
# CHECK-ASM: encoding: [0x0b,0xc5,0x05,0x3c]
th.lhia		a0, (a1), 0, 2

# CHECK-ASM-AND-OBJ: th.lhib
# CHECK-ASM: encoding: [0x0b,0xc5,0x05,0x2f]
th.lhib		a0, (a1), -16, 3

# CHECK-ASM-AND-OBJ: th.lhuia
# CHECK-ASM: encoding: [0x0b,0xc5,0x05,0xb8]
th.lhuia	a0, (a1), 0, 0

# CHECK-ASM-AND-OBJ: th.lhuib
# CHECK-ASM: encoding: [0x0b,0xc5,0xf5,0xaa]
th.lhuib	a0, (a1), 15, 1

# CHECK-ASM-AND-OBJ: th.lbia
# CHECK-ASM: encoding: [0x0b,0xc5,0x05,0x1c]
th.lbia		a0, (a1), 0, 2

# CHECK-ASM-AND-OBJ: th.lbib
# CHECK-ASM: encoding: [0x0b,0xc5,0x05,0x0f]
th.lbib		a0, (a1), -16, 3

# CHECK-ASM-AND-OBJ: th.lbuia
# CHECK-ASM: encoding: [0x0b,0xc5,0x05,0x98]
th.lbuia	a0, (a1), 0, 0

# CHECK-ASM-AND-OBJ: th.lbuib
# CHECK-ASM: encoding: [0x0b,0xc5,0xf5,0x8a]
th.lbuib	a0, (a1), 15, 1

# CHECK-ASM-AND-OBJ: th.sdia
# CHECK-ASM: encoding: [0x0b,0xd5,0x05,0x79]
th.sdia		a0, (a1), -16, 0

# CHECK-ASM-AND-OBJ: th.sdib
# CHECK-ASM: encoding: [0x0b,0xd5,0xf5,0x6b]
th.sdib		a0, (a1), -1, 1

# CHECK-ASM-AND-OBJ: th.swia
# CHECK-ASM: encoding: [0x0b,0xd5,0x05,0x5c]
th.swia		a0, (a1), 0, 2

# CHECK-ASM-AND-OBJ: th.swib
# CHECK-ASM: encoding: [0x0b,0xd5,0x15,0x4e]
th.swib		a0, (a1), 1, 3

# CHECK-ASM-AND-OBJ: th.shia
# CHECK-ASM: encoding: [0x0b,0xd5,0x45,0x38]
th.shia		a0, (a1), 4, 0

# CHECK-ASM-AND-OBJ: th.shib
# CHECK-ASM: encoding: [0x0b,0xd5,0xd5,0x2a]
th.shib		a0, (a1), 13, 1

# CHECK-ASM-AND-OBJ: th.sbia
# CHECK-ASM: encoding: [0x0b,0xd5,0xe5,0x1c]
th.sbia		a0, (a1), 14, 2

# CHECK-ASM-AND-OBJ: th.sbib
# CHECK-ASM: encoding: [0x0b,0xd5,0xf5,0x0e]
th.sbib		a0, (a1), 15, 3

# CHECK-ASM-AND-OBJ: th.lrd
# CHECK-ASM: encoding: [0x0b,0xc5,0xc5,0x60]
th.lrd		a0, a1, a2, 0

# CHECK-ASM-AND-OBJ: th.lrw
# CHECK-ASM: encoding: [0x0b,0xc5,0xc5,0x42]
th.lrw		a0, a1, a2, 1

# CHECK-ASM-AND-OBJ: th.lrwu
# CHECK-ASM: encoding: [0x0b,0xc5,0xc5,0xc4]
th.lrwu		a0, a1, a2, 2

# CHECK-ASM-AND-OBJ: th.lrh
# CHECK-ASM: encoding: [0x0b,0xc5,0xc5,0x26]
th.lrh		a0, a1, a2, 3

# CHECK-ASM-AND-OBJ: th.lrhu
# CHECK-ASM: encoding: [0x0b,0xc5,0xc5,0xa0]
th.lrhu		a0, a1, a2, 0

# CHECK-ASM-AND-OBJ: th.lrb
# CHECK-ASM: encoding: [0x0b,0xc5,0xc5,0x02]
th.lrb		a0, a1, a2, 1

# CHECK-ASM-AND-OBJ: th.lrbu
# CHECK-ASM: encoding: [0x0b,0xc5,0xc5,0x84]
th.lrbu		a0, a1, a2, 2

# CHECK-ASM-AND-OBJ: th.srd
# CHECK-ASM: encoding: [0x0b,0xd5,0xc5,0x66]
th.srd		a0, a1, a2, 3

# CHECK-ASM-AND-OBJ: th.srw
# CHECK-ASM: encoding: [0x0b,0xd5,0xc5,0x40]
th.srw		a0, a1, a2, 0

# CHECK-ASM-AND-OBJ: th.srh
# CHECK-ASM: encoding: [0x0b,0xd5,0xc5,0x22]
th.srh		a0, a1, a2, 1

# CHECK-ASM-AND-OBJ: th.srb
# CHECK-ASM: encoding: [0x0b,0xd5,0xc5,0x04]
th.srb		a0, a1, a2, 2

# CHECK-ASM-AND-OBJ: th.lurd
# CHECK-ASM: encoding: [0x0b,0xc5,0xc5,0x70]
th.lurd		a0, a1, a2, 0

# CHECK-ASM-AND-OBJ: th.lurw
# CHECK-ASM: encoding: [0x0b,0xc5,0xc5,0x52]
th.lurw		a0, a1, a2, 1

# CHECK-ASM-AND-OBJ: th.lurwu
# CHECK-ASM: encoding: [0x0b,0xc5,0xc5,0xd4]
th.lurwu	a0, a1, a2, 2

# CHECK-ASM-AND-OBJ: th.lurh
# CHECK-ASM: encoding: [0x0b,0xc5,0xc5,0x36]
th.lurh		a0, a1, a2, 3

# CHECK-ASM-AND-OBJ: th.lurhu
# CHECK-ASM: encoding: [0x0b,0xc5,0xc5,0xb0]
th.lurhu	a0, a1, a2, 0

# CHECK-ASM-AND-OBJ: th.lurb
# CHECK-ASM: encoding: [0x0b,0xc5,0xc5,0x12]
th.lurb		a0, a1, a2, 1

# CHECK-ASM-AND-OBJ: th.lurbu
# CHECK-ASM: encoding: [0x0b,0xc5,0xc5,0x94]
th.lurbu	a0, a1, a2, 2

# CHECK-ASM-AND-OBJ: th.surd
# CHECK-ASM: encoding: [0x0b,0xd5,0xc5,0x76]
th.surd		a0, a1, a2, 3

# CHECK-ASM-AND-OBJ: th.surw
# CHECK-ASM: encoding: [0x0b,0xd5,0xc5,0x50]
th.surw		a0, a1, a2, 0

# CHECK-ASM-AND-OBJ: th.surh
# CHECK-ASM: encoding: [0x0b,0xd5,0xc5,0x32]
th.surh		a0, a1, a2, 1

# CHECK-ASM-AND-OBJ: th.surb
# CHECK-ASM: encoding: [0x0b,0xd5,0xc5,0x14]
th.surb		a0, a1, a2, 2
