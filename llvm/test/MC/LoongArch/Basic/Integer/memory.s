## Test valid memory access instructions.

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

# CHECK-ASM-AND-OBJ: ld.b $s1, $a4, 21
# CHECK-ASM: encoding: [0x18,0x55,0x00,0x28]
ld.b $s1, $a4, 21

# CHECK-ASM-AND-OBJ: ld.h $a3, $t6, 80
# CHECK-ASM: encoding: [0x47,0x42,0x41,0x28]
ld.h $a3, $t6, 80

# CHECK-ASM-AND-OBJ: ld.w $t6, $s3, 92
# CHECK-ASM: encoding: [0x52,0x73,0x81,0x28]
ld.w $t6, $s3, 92

# CHECK-ASM-AND-OBJ: ld.bu $t1, $t1, 150
# CHECK-ASM: encoding: [0xad,0x59,0x02,0x2a]
ld.bu $t1, $t1, 150

# CHECK-ASM-AND-OBJ: ld.hu $t6, $s6, 198
# CHECK-ASM: encoding: [0xb2,0x1b,0x43,0x2a]
ld.hu $t6, $s6, 198

# CHECK-ASM-AND-OBJ: st.b $sp, $a3, 95
# CHECK-ASM: encoding: [0xe3,0x7c,0x01,0x29]
st.b $sp, $a3, 95

# CHECK-ASM-AND-OBJ: st.h $s2, $t4, 122
# CHECK-ASM: encoding: [0x19,0xea,0x41,0x29]
st.h $s2, $t4, 122

# CHECK-ASM-AND-OBJ: st.w $t1, $t1, 175
# CHECK-ASM: encoding: [0xad,0xbd,0x82,0x29]
st.w $t1, $t1, 175

# CHECK-ASM-AND-OBJ: preld 10, $zero, 23
# CHECK-ASM: encoding: [0x0a,0x5c,0xc0,0x2a]
preld 10, $zero, 23


#############################################################
## Instructions only for loongarch64
#############################################################

.ifdef LA64

# CHECK64-ASM-AND-OBJ: ld.wu $t2, $t7, 31
# CHECK64-ASM: encoding: [0x6e,0x7e,0x80,0x2a]
ld.wu $t2, $t7, 31

# CHECK64-ASM-AND-OBJ: st.d $s7, $s7, 60
# CHECK64-ASM: encoding: [0xde,0xf3,0xc0,0x29]
st.d $s7, $s7, 60

# CHECK64-ASM-AND-OBJ: ldx.b $s1, $ra, $tp
# CHECK64-ASM: encoding: [0x38,0x08,0x00,0x38]
ldx.b $s1, $ra, $tp

# CHECK64-ASM-AND-OBJ: ldx.h $fp, $fp, $t5
# CHECK64-ASM: encoding: [0xd6,0x46,0x04,0x38]
ldx.h $fp, $fp, $t5

# CHECK64-ASM-AND-OBJ: ldx.w $s2, $a7, $s0
# CHECK64-ASM: encoding: [0x79,0x5d,0x08,0x38]
ldx.w $s2, $a7, $s0

# CHECK64-ASM-AND-OBJ: ldx.d $t6, $s0, $t8
# CHECK64-ASM: encoding: [0xf2,0x52,0x0c,0x38]
ldx.d $t6, $s0, $t8

# CHECK64-ASM-AND-OBJ: ldx.bu $a7, $a5, $a5
# CHECK64-ASM: encoding: [0x2b,0x25,0x20,0x38]
ldx.bu $a7, $a5, $a5

# CHECK64-ASM-AND-OBJ: ldx.hu $fp, $s0, $s4
# CHECK64-ASM: encoding: [0xf6,0x6e,0x24,0x38]
ldx.hu $fp, $s0, $s4

# CHECK64-ASM-AND-OBJ: ldx.wu $a4, $s1, $s5
# CHECK64-ASM: encoding: [0x08,0x73,0x28,0x38]
ldx.wu $a4, $s1, $s5

# CHECK64-ASM-AND-OBJ: stx.b $t7, $ra, $sp
# CHECK64-ASM: encoding: [0x33,0x0c,0x10,0x38]
stx.b $t7, $ra, $sp

# CHECK64-ASM-AND-OBJ: stx.h $zero, $s5, $s3
# CHECK64-ASM: encoding: [0x80,0x6b,0x14,0x38]
stx.h $zero, $s5, $s3

# CHECK64-ASM-AND-OBJ: stx.w $a3, $a0, $s8
# CHECK64-ASM: encoding: [0x87,0x7c,0x18,0x38]
stx.w $a3, $a0, $s8

# CHECK64-ASM-AND-OBJ: stx.d $a3, $s8, $a6
# CHECK64-ASM: encoding: [0xe7,0x2b,0x1c,0x38]
stx.d $a3, $s8, $a6

# CHECK64-ASM-AND-OBJ: ldptr.w $s3, $a2, 60
# CHECK64-ASM: encoding: [0xda,0x3c,0x00,0x24]
ldptr.w $s3, $a2, 60

# CHECK64-ASM-AND-OBJ: ldptr.d $a1, $s6, 244
# CHECK64-ASM: encoding: [0xa5,0xf7,0x00,0x26]
ldptr.d $a1, $s6, 244

# CHECK64-ASM-AND-OBJ: stptr.w $s5, $a1, 216
# CHECK64-ASM: encoding: [0xbc,0xd8,0x00,0x25]
stptr.w $s5, $a1, 216

# CHECK64-ASM-AND-OBJ: stptr.d $t2, $s1, 196
# CHECK64-ASM: encoding: [0x0e,0xc7,0x00,0x27]
stptr.d $t2, $s1, 196

.endif

