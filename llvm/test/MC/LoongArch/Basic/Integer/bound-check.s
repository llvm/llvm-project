## Test valid boundary check memory access instructions.

# RUN: llvm-mc %s --triple=loongarch64 --show-encoding \
# RUN:     | FileCheck --check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s --triple=loongarch64 --filetype=obj | llvm-objdump -d - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: ldgt.b $a2, $a2, $s6
# CHECK-ASM: encoding: [0xc6,0x74,0x78,0x38]
ldgt.b $a2, $a2, $s6

# CHECK-ASM-AND-OBJ: ldgt.h $a1, $s8, $ra
# CHECK-ASM: encoding: [0xe5,0x87,0x78,0x38]
ldgt.h $a1, $s8, $ra

# CHECK-ASM-AND-OBJ: ldgt.w $t3, $s3, $a4
# CHECK-ASM: encoding: [0x4f,0x23,0x79,0x38]
ldgt.w $t3, $s3, $a4

# CHECK-ASM-AND-OBJ: ldgt.d $s0, $s2, $s8
# CHECK-ASM: encoding: [0x37,0xff,0x79,0x38]
ldgt.d $s0, $s2, $s8

# CHECK-ASM-AND-OBJ: ldle.b $a5, $t0, $t3
# CHECK-ASM: encoding: [0x89,0x3d,0x7a,0x38]
ldle.b $a5, $t0, $t3

# CHECK-ASM-AND-OBJ: ldle.h $a7, $a7, $s0
# CHECK-ASM: encoding: [0x6b,0xdd,0x7a,0x38]
ldle.h $a7, $a7, $s0

# CHECK-ASM-AND-OBJ: ldle.w $s1, $tp, $tp
# CHECK-ASM: encoding: [0x58,0x08,0x7b,0x38]
ldle.w $s1, $tp, $tp

# CHECK-ASM-AND-OBJ: ldle.d $t8, $t3, $t4
# CHECK-ASM: encoding: [0xf4,0xc1,0x7b,0x38]
ldle.d $t8, $t3, $t4

# CHECK-ASM-AND-OBJ: stgt.b $s4, $t7, $t8
# CHECK-ASM: encoding: [0x7b,0x52,0x7c,0x38]
stgt.b $s4, $t7, $t8

# CHECK-ASM-AND-OBJ: stgt.h $t4, $a0, $a2
# CHECK-ASM: encoding: [0x90,0x98,0x7c,0x38]
stgt.h $t4, $a0, $a2

# CHECK-ASM-AND-OBJ: stgt.w $s8, $s5, $t2
# CHECK-ASM: encoding: [0x9f,0x3b,0x7d,0x38]
stgt.w $s8, $s5, $t2

# CHECK-ASM-AND-OBJ: stgt.d $s7, $r21, $s1
# CHECK-ASM: encoding: [0xbe,0xe2,0x7d,0x38]
stgt.d $s7, $r21, $s1

# CHECK-ASM-AND-OBJ: stle.b $a6, $a0, $t4
# CHECK-ASM: encoding: [0x8a,0x40,0x7e,0x38]
stle.b $a6, $a0, $t4

# CHECK-ASM-AND-OBJ: stle.h $t5, $t5, $r21
# CHECK-ASM: encoding: [0x31,0xd6,0x7e,0x38]
stle.h $t5, $t5, $r21

# CHECK-ASM-AND-OBJ: stle.w $s0, $s5, $s6
# CHECK-ASM: encoding: [0x97,0x77,0x7f,0x38]
stle.w $s0, $s5, $s6

# CHECK-ASM-AND-OBJ: stle.d $s2, $s1, $s6
# CHECK-ASM: encoding: [0x19,0xf7,0x7f,0x38]
stle.d $s2, $s1, $s6

