## Test valid arithmetic operation instructions

# RUN: llvm-mc %s --triple=loongarch32 --show-encoding \
# RUN:     | FileCheck --check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s --triple=loongarch64 --show-encoding --defsym=LA64=1 \
# RUN:     | FileCheck --check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ,CHECK64-ASM,CHECK64-ASM-AND-OBJ %s
# RUN: llvm-mc %s --triple=loongarch32 --filetype=obj | llvm-objdump -d - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s --triple=loongarch64 --filetype=obj --defsym=LA64=1 | llvm-objdump -d - \
# RUN:     | FileCheck --check-prefixes=CHECK-ASM-AND-OBJ,CHECK64-ASM-AND-OBJ %s

#############################################################
## Instructions for both loongarch32 and loongarch64
#############################################################

# CHECK-ASM-AND-OBJ: add.w $a5, $ra, $s8
# CHECK-ASM: encoding: [0x29,0x7c,0x10,0x00]
add.w $a5, $ra, $s8

# CHECK-ASM-AND-OBJ: sub.w $r21, $s2, $t7
# CHECK-ASM: encoding: [0x35,0x4f,0x11,0x00]
sub.w $r21, $s2, $t7

# CHECK-ASM-AND-OBJ: addi.w $a1, $a3, 246
# CHECK-ASM: encoding: [0xe5,0xd8,0x83,0x02]
addi.w $a1, $a3, 246

# CHECK-ASM-AND-OBJ: alsl.w $tp, $t5, $tp, 4
# CHECK-ASM: encoding: [0x22,0x8a,0x05,0x00]
alsl.w $tp, $t5, $tp, 4

# CHECK-ASM-AND-OBJ: lu12i.w $t4, 49
# CHECK-ASM: encoding: [0x30,0x06,0x00,0x14]
lu12i.w $t4, 49

# CHECK-ASM-AND-OBJ: lu12i.w $a0, -1
# CHECK-ASM: encoding: [0xe4,0xff,0xff,0x15]
lu12i.w $a0, -1

# CHECK-ASM-AND-OBJ: slt $s6, $s3, $tp
# CHECK-ASM: encoding: [0x5d,0x0b,0x12,0x00]
slt $s6, $s3, $tp

# CHECK-ASM-AND-OBJ: sltu $a7, $r21, $s6
# CHECK-ASM: encoding: [0xab,0xf6,0x12,0x00]
sltu $a7, $r21, $s6

# CHECK-ASM-AND-OBJ: slti $s4, $ra, 235
# CHECK-ASM: encoding: [0x3b,0xac,0x03,0x02]
slti $s4, $ra, 235

# CHECK-ASM-AND-OBJ: sltui $zero, $a4, 162
# CHECK-ASM: encoding: [0x00,0x89,0x42,0x02]
sltui $zero, $a4, 162

# CHECK-ASM-AND-OBJ: pcaddi $a5, 187
# CHECK-ASM: encoding: [0x69,0x17,0x00,0x18]
pcaddi $a5, 187

# CHECK-ASM-AND-OBJ: pcaddu12i $zero, 37
# CHECK-ASM: encoding: [0xa0,0x04,0x00,0x1c]
pcaddu12i $zero, 37

# CHECK-ASM-AND-OBJ: pcalau12i $a6, 89
# CHECK-ASM: encoding: [0x2a,0x0b,0x00,0x1a]
pcalau12i $a6, 89

# CHECK-ASM-AND-OBJ: and $t7, $s8, $ra
# CHECK-ASM: encoding: [0xf3,0x87,0x14,0x00]
and $t7, $s8, $ra

# CHECK-ASM-AND-OBJ: or $t5, $t4, $s7
# CHECK-ASM: encoding: [0x11,0x7a,0x15,0x00]
or $t5, $t4, $s7

# CHECK-ASM-AND-OBJ: nor $a1, $t6, $a1
# CHECK-ASM: encoding: [0x45,0x16,0x14,0x00]
nor $a1, $t6, $a1

# CHECK-ASM-AND-OBJ: xor $t3, $t7, $a4
# CHECK-ASM: encoding: [0x6f,0xa2,0x15,0x00]
xor $t3, $t7, $a4

# CHECK-ASM-AND-OBJ: andn $s5, $s2, $a1
# CHECK-ASM: encoding: [0x3c,0x97,0x16,0x00]
andn $s5, $s2, $a1

# CHECK-ASM-AND-OBJ: orn $tp, $sp, $s2
# CHECK-ASM: encoding: [0x62,0x64,0x16,0x00]
orn $tp, $sp, $s2

# CHECK-ASM-AND-OBJ: andi $s2, $zero, 106
# CHECK-ASM: encoding: [0x19,0xa8,0x41,0x03]
andi $s2, $zero, 106

# CHECK-ASM-AND-OBJ: ori $t5, $a1, 47
# CHECK-ASM: encoding: [0xb1,0xbc,0x80,0x03]
ori $t5, $a1, 47

# CHECK-ASM-AND-OBJ: xori $t6, $s0, 99
# CHECK-ASM: encoding: [0xf2,0x8e,0xc1,0x03]
xori $t6, $s0, 99

# CHECK-ASM-AND-OBJ: mul.w $a0, $t6, $sp
# CHECK-ASM: encoding: [0x44,0x0e,0x1c,0x00]
mul.w $a0, $t6, $sp

# CHECK-ASM-AND-OBJ: mulh.w $s4, $s0, $zero
# CHECK-ASM: encoding: [0xfb,0x82,0x1c,0x00]
mulh.w $s4, $s0, $zero

# CHECK-ASM-AND-OBJ: mulh.wu $a6, $t5, $s1
# CHECK-ASM: encoding: [0x2a,0x62,0x1d,0x00]
mulh.wu $a6, $t5, $s1

# CHECK-ASM-AND-OBJ: div.w $s7, $t1, $s2
# CHECK-ASM: encoding: [0xbe,0x65,0x20,0x00]
div.w $s7, $t1, $s2

# CHECK-ASM-AND-OBJ: mod.w $ra, $s3, $a6
# CHECK-ASM: encoding: [0x41,0xab,0x20,0x00]
mod.w $ra, $s3, $a6

# CHECK-ASM-AND-OBJ: div.wu $t7, $s0, $zero
# CHECK-ASM: encoding: [0xf3,0x02,0x21,0x00]
div.wu $t7, $s0, $zero

# CHECK-ASM-AND-OBJ: mod.wu $s4, $a5, $t5
# CHECK-ASM: encoding: [0x3b,0xc5,0x21,0x00]
mod.wu $s4, $a5, $t5


#############################################################
## Instructions only for loongarch64
#############################################################

.ifdef LA64

# CHECK64-ASM-AND-OBJ: add.d $tp, $t6, $s4
# CHECK64-ASM: encoding: [0x42,0xee,0x10,0x00]
add.d $tp, $t6, $s4

# CHECK64-ASM-AND-OBJ: sub.d $a3, $t0, $a3
# CHECK64-ASM: encoding: [0x87,0x9d,0x11,0x00]
sub.d $a3, $t0, $a3

# CHECK64-ASM-AND-OBJ: addi.d $s5, $a2, 75
# CHECK64-ASM: encoding: [0xdc,0x2c,0xc1,0x02]
addi.d $s5, $a2, 75

# CHECK64-ASM-AND-OBJ: addu16i.d $a5, $s0, 23
# CHECK64-ASM: encoding: [0xe9,0x5e,0x00,0x10]
addu16i.d $a5, $s0, 23

# CHECK64-ASM-AND-OBJ: alsl.wu $t7, $a4, $s2, 1
# CHECK64-ASM: encoding: [0x13,0x65,0x06,0x00]
alsl.wu $t7, $a4, $s2, 1

# CHECK64-ASM-AND-OBJ: alsl.d $t5, $a7, $a1, 3
# CHECK64-ASM: encoding: [0x71,0x15,0x2d,0x00]
alsl.d $t5, $a7, $a1, 3

# CHECK64-ASM-AND-OBJ: lu32i.d $sp, 196
# CHECK64-ASM: encoding: [0x83,0x18,0x00,0x16]
lu32i.d $sp, 196

# CHECK64-ASM-AND-OBJ: lu52i.d $t1, $a0, 195
# CHECK64-ASM: encoding: [0x8d,0x0c,0x03,0x03]
lu52i.d $t1, $a0, 195

# CHECK64-ASM-AND-OBJ: pcaddu18i $t0, 26
# CHECK64-ASM: encoding: [0x4c,0x03,0x00,0x1e]
pcaddu18i $t0, 26

# CHECK64-ASM-AND-OBJ: mul.d $ra, $t2, $s1
# CHECK64-ASM: encoding: [0xc1,0xe1,0x1d,0x00]
mul.d $ra, $t2, $s1

# CHECK64-ASM-AND-OBJ: mulh.d $s5, $ra, $s4
# CHECK64-ASM: encoding: [0x3c,0x6c,0x1e,0x00]
mulh.d $s5, $ra, $s4

# CHECK64-ASM-AND-OBJ: mulh.du $t1, $s4, $s6
# CHECK64-ASM: encoding: [0x6d,0xf7,0x1e,0x00]
mulh.du $t1, $s4, $s6

# CHECK64-ASM-AND-OBJ: mulw.d.w $s4, $a2, $t5
# CHECK64-ASM: encoding: [0xdb,0x44,0x1f,0x00]
mulw.d.w $s4, $a2, $t5

# CHECK64-ASM-AND-OBJ: mulw.d.wu $t5, $fp, $s7
# CHECK64-ASM: encoding: [0xd1,0xfa,0x1f,0x00]
mulw.d.wu $t5, $fp, $s7

# CHECK64-ASM-AND-OBJ: div.d $s0, $a2, $r21
# CHECK64-ASM: encoding: [0xd7,0x54,0x22,0x00]
div.d $s0, $a2, $r21

# CHECK64-ASM-AND-OBJ: mod.d $t4, $sp, $t3
# CHECK64-ASM: encoding: [0x70,0xbc,0x22,0x00]
mod.d $t4, $sp, $t3

# CHECK64-ASM-AND-OBJ: div.du $s8, $s1, $t2
# CHECK64-ASM: encoding: [0x1f,0x3b,0x23,0x00]
div.du $s8, $s1, $t2

# CHECK64-ASM-AND-OBJ: mod.du $s2, $s0, $s1
# CHECK64-ASM: encoding: [0xf9,0xe2,0x23,0x00]
mod.du $s2, $s0, $s1

.endif

