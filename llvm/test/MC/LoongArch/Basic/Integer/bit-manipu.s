## Test valid bit manipulation instructions.

# RUN: llvm-mc %s --triple=loongarch32 --show-encoding \
# RUN:     | FileCheck --check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s --triple=loongarch64 --show-encoding --defsym=LA64=1 \
# RUN:     | FileCheck --check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ,CHECK64-ASM,CHECK64-ASM-AND-OBJ %s
# RUN: llvm-mc %s --triple=loongarch32 --filetype=obj | llvm-objdump -d - \
# RUN:     | FileCheck --check-prefixes=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s --triple=loongarch64 --filetype=obj --defsym=LA64=1 | llvm-objdump -d - \
# RUN:     | FileCheck --check-prefixes=CHECK-ASM-AND-OBJ,CHECK64-ASM-AND-OBJ %s

#############################################################
## Instructions for both loongarch32 and loongarch64
#############################################################

# CHECK-ASM: ext.w.b $t8, $t6
# CHECK-ASM: encoding: [0x54,0x5e,0x00,0x00]
ext.w.b $t8, $t6

# CHECK-ASM: ext.w.h $s0, $s0
# CHECK-ASM: encoding: [0xf7,0x5a,0x00,0x00]
ext.w.h $s0, $s0

# CHECK-ASM-AND-OBJ: clo.w $ra, $sp
# CHECK-ASM: encoding: [0x61,0x10,0x00,0x00]
clo.w $ra, $sp

# CHECK-ASM-AND-OBJ: clz.w $a3, $a6
# CHECK-ASM: encoding: [0x47,0x15,0x00,0x00]
clz.w $a3, $a6

# CHECK-ASM-AND-OBJ: cto.w $tp, $a2
# CHECK-ASM: encoding: [0xc2,0x18,0x00,0x00]
cto.w $tp, $a2

# CHECK-ASM-AND-OBJ: ctz.w $a1, $fp
# CHECK-ASM: encoding: [0xc5,0x1e,0x00,0x00]
ctz.w $a1, $fp

# CHECK-ASM-AND-OBJ: bytepick.w $s6, $zero, $t4, 0
# CHECK-ASM: encoding: [0x1d,0x40,0x08,0x00]
bytepick.w $s6, $zero, $t4, 0

# CHECK-ASM-AND-OBJ: revb.2h $t8, $a7
# CHECK-ASM: encoding: [0x74,0x31,0x00,0x00]
revb.2h $t8, $a7

# CHECK-ASM-AND-OBJ: bitrev.4b $r21, $s4
# CHECK-ASM: encoding: [0x75,0x4b,0x00,0x00]
bitrev.4b $r21, $s4

# CHECK-ASM-AND-OBJ: bitrev.w $s2, $a1
# CHECK-ASM: encoding: [0xb9,0x50,0x00,0x00]
bitrev.w $s2, $a1

# CHECK-ASM-AND-OBJ: bstrins.w $a4, $a7, 7, 2
# CHECK-ASM: encoding: [0x68,0x09,0x67,0x00]
bstrins.w $a4, $a7, 7, 2

# CHECK-ASM-AND-OBJ: bstrpick.w $ra, $a5, 10, 4
# CHECK-ASM: encoding: [0x21,0x91,0x6a,0x00]
bstrpick.w $ra, $a5, 10, 4

# CHECK-ASM-AND-OBJ: maskeqz $t8, $a7, $t6
# CHECK-ASM: encoding: [0x74,0x49,0x13,0x00]
maskeqz $t8, $a7, $t6

# CHECK-ASM-AND-OBJ: masknez $t8, $t1, $s3
# CHECK-ASM: encoding: [0xb4,0xe9,0x13,0x00]
masknez $t8, $t1, $s3


#############################################################
## Instructions only for loongarch64
#############################################################

.ifdef LA64

# CHECK64-ASM-AND-OBJ: clo.d $s6, $ra
# CHECK64-ASM: encoding: [0x3d,0x20,0x00,0x00]
clo.d $s6, $ra

# CHECK64-ASM-AND-OBJ: clz.d $s3, $s3
# CHECK64-ASM: encoding: [0x5a,0x27,0x00,0x00]
clz.d $s3, $s3

# CHECK64-ASM-AND-OBJ: cto.d $t6, $t8
# CHECK64-ASM: encoding: [0x92,0x2a,0x00,0x00]
cto.d $t6, $t8

# CHECK64-ASM-AND-OBJ: ctz.d $t5, $a6
# CHECK64-ASM: encoding: [0x51,0x2d,0x00,0x00]
ctz.d $t5, $a6

# CHECK64-ASM-AND-OBJ: bytepick.d $t3, $t5, $t8, 4
# CHECK64-ASM: encoding: [0x2f,0x52,0x0e,0x00]
bytepick.d $t3, $t5, $t8, 4

# CHECK64-ASM-AND-OBJ: revb.4h $t1, $t7
# CHECK64-ASM: encoding: [0x6d,0x36,0x00,0x00]
revb.4h $t1, $t7

# CHECK64-ASM-AND-OBJ: revb.2w $s5, $s4
# CHECK64-ASM: encoding: [0x7c,0x3b,0x00,0x00]
revb.2w $s5, $s4

# CHECK64-ASM-AND-OBJ: revb.d $zero, $s0
# CHECK64-ASM: encoding: [0xe0,0x3e,0x00,0x00]
revb.d $zero, $s0

# CHECK64-ASM-AND-OBJ: revh.2w $s5, $a6
# CHECK64-ASM: encoding: [0x5c,0x41,0x00,0x00]
revh.2w $s5, $a6

# CHECK64-ASM-AND-OBJ: revh.d $a5, $a3
# CHECK64-ASM: encoding: [0xe9,0x44,0x00,0x00]
revh.d $a5, $a3

# CHECK64-ASM-AND-OBJ: bitrev.8b $t1, $s2
# CHECK64-ASM: encoding: [0x2d,0x4f,0x00,0x00]
bitrev.8b $t1, $s2

# CHECK64-ASM-AND-OBJ: bitrev.d $t7, $s0
# CHECK64-ASM: encoding: [0xf3,0x56,0x00,0x00]
bitrev.d $t7, $s0

# CHECK64-ASM-AND-OBJ: bstrins.d $a4, $a7, 7, 2
# CHECK64-ASM: encoding: [0x68,0x09,0x87,0x00]
bstrins.d $a4, $a7, 7, 2

# CHECK64-ASM-AND-OBJ: bstrpick.d $s8, $s4, 39, 22
# CHECK64-ASM: encoding: [0x7f,0x5b,0xe7,0x00]
bstrpick.d $s8, $s4, 39, 22

.endif

