## Test valid bit shift instructions.

# RUN: llvm-mc %s --triple=loongarch32 -show-encoding \
# RUN:     | FileCheck --check-prefix=CHECK-ASM %s
# RUN: llvm-mc %s --triple=loongarch64 -show-encoding --defsym=LA64=1 \
# RUN:     | FileCheck --check-prefixes=CHECK-ASM,CHECK64-ASM %s

#############################################################
## Instructions for both loongarch32 and loongarch64
#############################################################

# CHECK-ASM: sll.w $s1, $s4, $s0
# CHECK-ASM: encoding: [0x78,0x5f,0x17,0x00]
sll.w $s1, $s4, $s0

# CHECK-ASM: srl.w $s8, $t5, $a3
# CHECK-ASM: encoding: [0x3f,0x9e,0x17,0x00]
srl.w $s8, $t5, $a3

# CHECK-ASM: sra.w $t0, $s5, $a6
# CHECK-ASM: encoding: [0x8c,0x2b,0x18,0x00]
sra.w $t0, $s5, $a6

# CHECK-ASM: rotr.w $ra, $s3, $t6
# CHECK-ASM: encoding: [0x41,0x4b,0x1b,0x00]
rotr.w $ra, $s3, $t6

# CHECK-ASM: slli.w $s3, $t6, 0
# CHECK-ASM: encoding: [0x5a,0x82,0x40,0x00]
slli.w $s3, $t6, 0

# CHECK-ASM: srli.w $a6, $t2, 30
# CHECK-ASM: encoding: [0xca,0xf9,0x44,0x00]
srli.w $a6, $t2, 30

# CHECK-ASM: srai.w $a4, $t5, 24
# CHECK-ASM: encoding: [0x28,0xe2,0x48,0x00]
srai.w $a4, $t5, 24

# CHECK-ASM: rotri.w $s0, $t8, 23
# CHECK-ASM: encoding: [0x97,0xde,0x4c,0x00]
rotri.w $s0, $t8, 23


#############################################################
## Instructions only for loongarch64
#############################################################

.ifdef LA64

# CHECK64-ASM: sll.d $t8, $t3, $sp
# CHECK64-ASM: encoding: [0xf4,0x8d,0x18,0x00]
sll.d $t8, $t3, $sp

# CHECK64-ASM: srl.d $t2, $s2, $zero
# CHECK64-ASM: encoding: [0x2e,0x03,0x19,0x00]
srl.d $t2, $s2, $zero

# CHECK64-ASM: sra.d $a3, $fp, $s8
# CHECK64-ASM: encoding: [0xc7,0xfe,0x19,0x00]
sra.d $a3, $fp, $s8

# CHECK64-ASM: rotr.d $s8, $sp, $ra
# CHECK64-ASM: encoding: [0x7f,0x84,0x1b,0x00]
rotr.d $s8, $sp, $ra

# CHECK64-ASM: slli.d $a6, $s8, 39
# CHECK64-ASM: encoding: [0xea,0x9f,0x41,0x00]
slli.d $a6, $s8, 39

# CHECK64-ASM: srli.d $s8, $fp, 38
# CHECK64-ASM: encoding: [0xdf,0x9a,0x45,0x00]
srli.d $s8, $fp, 38

# CHECK64-ASM: srai.d $a5, $r21, 27
# CHECK64-ASM: encoding: [0xa9,0x6e,0x49,0x00]
srai.d $a5, $r21, 27

# CHECK64-ASM: rotri.d $s6, $zero, 7
# CHECK64-ASM: encoding: [0x1d,0x1c,0x4d,0x00]
rotri.d $s6, $zero, 7

.endif

