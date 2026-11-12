# RUN: llvm-mc %s -triple=xtensa -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s


.align	4
LBL0:

# Instruction format RRI8
# CHECK-INST:  ball    a1, a3, LBL0
# CHECK: encoding: [0x37,0x41,A]
ball a1, a3, LBL0

# Instruction format RRI8
# CHECK-INST:  bany    a8, a13, LBL0
# CHECK: encoding: [0xd7,0x88,A]
bany a8, a13, LBL0

# Instruction format RRI8
# CHECK-INST:  bbc     a8, a7, LBL0
# CHECK: encoding: [0x77,0x58,A]
bbc a8, a7, LBL0

# Instruction format RRI8
# CHECK-INST:  bbci    a3, 16, LBL0
# CHECK: encoding: [0x07,0x73,A]
bbci a3, 16, LBL0

# CHECK-INST:  bbci    a3, 16, LBL0
# CHECK: encoding: [0x07,0x73,A]
bbci a3, (16), LBL0

# Instruction format RRI8
# CHECK-INST:  bbs     a12, a5, LBL0
# CHECK: encoding: [0x57,0xdc,A]
bbs a12, a5, LBL0

# Instruction format RRI8
# CHECK-INST:  bbsi    a3, 16, LBL0
# CHECK: encoding: [0x07,0xf3,A]
bbsi a3, 16, LBL0

# Instruction format RRI8
# CHECK-INST:  bnall   a7, a3, LBL0
# CHECK: encoding: [0x37,0xc7,A]
bnall a7, a3, LBL0

# Instruction format RRI8
# CHECK-INST:  bnone   a2, a4, LBL0
# CHECK: encoding: [0x47,0x02,A]
bnone a2, a4, LBL0

# Instruction format RRI8
# CHECK-INST:  beq     a1, a2, LBL0
# CHECK: encoding: [0x27,0x11,A]
beq a1, a2, LBL0

# CHECK-INST:  beq     a11, a5, LBL0
# CHECK: encoding: [0x57,0x1b,A]
beq a11, a5, LBL0

# Instruction format BRI8
# CHECK-INST:  beqi    a1, 256, LBL0
# CHECK: encoding: [0x26,0xf1,A]
beqi a1, 256, LBL0

# CHECK-INST:  beqi    a11, -1, LBL0
# CHECK: encoding: [0x26,0x0b,A]
beqi a11, -1, LBL0

# Instruction format BRI12
# CHECK-INST:  beqz    a8, LBL0
# CHECK: encoding: [0x16,0bAAAA1000,A]
beqz a8, LBL0

# Instruction format RRI8
# CHECK-INST:  bge     a14, a2, LBL0
# CHECK: encoding: [0x27,0xae,A]
bge a14, a2, LBL0

# Instruction format BRI8
# CHECK-INST:  bgei    a11, -1, LBL0
# CHECK: encoding: [0xe6,0x0b,A]
bgei a11, -1, LBL0

# CHECK-INST:  bgei    a11, 128, LBL0
# CHECK: encoding: [0xe6,0xeb,A]
bgei a11, 128, LBL0

# Instruction format RRI8
# CHECK-INST:  bgeu    a14, a2, LBL0
# CHECK: encoding: [0x27,0xbe,A]
bgeu a14, a2, LBL0

# CHECK-INST:  bgeu    a13, a1, LBL0
# CHECK: encoding: [0x17,0xbd,A]
bgeu a13, a1, LBL0

# Instruction format BRI8
# CHECK-INST:  bgeui   a9, 32768, LBL0
# CHECK: encoding: [0xf6,0x09,A]
bgeui a9, 32768, LBL0

# CHECK-INST:  bgeui   a7, 65536, LBL0
# CHECK: encoding: [0xf6,0x17,A]
bgeui a7, 65536, LBL0

# CHECK-INST:  bgeui   a7, 64, LBL0
# CHECK: encoding: [0xf6,0xd7,A]
bgeui a7, 64, LBL0

# Instruction format BRI12
# CHECK-INST:  bgez    a8, LBL0
# CHECK: encoding: [0xd6,0bAAAA1000,A]
bgez a8, LBL0

# Instruction format RRI8
# CHECK-INST:  blt     a14, a2, LBL0
# CHECK: encoding: [0x27,0x2e,A]
blt a14, a2, LBL0

# Instruction format BRI8
# CHECK-INST:  blti    a12, -1, LBL0
# CHECK: encoding: [0xa6,0x0c,A]
blti a12, -1, LBL0

# CHECK-INST:  blti    a0, 32, LBL0
# CHECK: encoding: [0xa6,0xc0,A]
blti a0, 32, LBL0

# Instruction format BRI8
# CHECK-INST:  bltui   a7, 16, LBL0
# CHECK: encoding: [0xb6,0xb7,A]
bltui a7, 16, LBL0

# Instruction format BRI12
# CHECK-INST:  bltz    a6, LBL0
# CHECK: encoding: [0x96,0bAAAA0110,A]
bltz a6, LBL0

# Instruction format RRI8
# CHECK-INST:  bne     a3, a4, LBL0
# CHECK: encoding: [0x47,0x93,A]
bne a3, a4, LBL0

# Instruction format BRI8
# CHECK-INST:  bnei    a5, 12, LBL0
# CHECK: encoding: [0x66,0xa5,A]
bnei a5, 12, LBL0

# Instruction format BRI12
# CHECK-INST:  bnez    a5, LBL0
# CHECK: encoding: [0x56,0bAAAA0101,A]
bnez a5, LBL0
