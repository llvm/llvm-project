## Test valid atomic memory access instructions.

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

# CHECK-ASM-AND-OBJ: ll.w $tp, $s4, 220
# CHECK-ASM: encoding: [0x62,0xdf,0x00,0x20]
ll.w $tp, $s4, 220

# CHECK-ASM-AND-OBJ: sc.w $t7, $t2, 56
# CHECK-ASM: encoding: [0xd3,0x39,0x00,0x21]
sc.w $t7, $t2, 56



#############################################################
## Instructions only for loongarch64
#############################################################

.ifdef LA64

# CHECK64-ASM-AND-OBJ: amswap.w $a2, $t0, $s1
# CHECK64-ASM: encoding: [0x06,0x33,0x60,0x38]
amswap.w $a2, $t0, $s1

# CHECK64-ASM-AND-OBJ: amswap.d $tp, $t2, $fp
# CHECK64-ASM: encoding: [0xc2,0xba,0x60,0x38]
amswap.d $tp, $t2, $fp

# CHECK64-ASM-AND-OBJ: amadd.w $a4, $t0, $r21
# CHECK64-ASM: encoding: [0xa8,0x32,0x61,0x38]
amadd.w $a4, $t0, $r21

# CHECK64-ASM-AND-OBJ: amadd.d $a1, $t5, $s6
# CHECK64-ASM: encoding: [0xa5,0xc7,0x61,0x38]
amadd.d $a1, $t5, $s6

# CHECK64-ASM-AND-OBJ: amand.w $a0, $t7, $fp
# CHECK64-ASM: encoding: [0xc4,0x4e,0x62,0x38]
amand.w $a0, $t7, $fp

# CHECK64-ASM-AND-OBJ: amand.d $a6, $t6, $s6
# CHECK64-ASM: encoding: [0xaa,0xcb,0x62,0x38]
amand.d $a6, $t6, $s6

# CHECK64-ASM-AND-OBJ: amor.w $a2, $t4, $s0
# CHECK64-ASM: encoding: [0xe6,0x42,0x63,0x38]
amor.w $a2, $t4, $s0

# CHECK64-ASM-AND-OBJ: amor.d $sp, $t4, $s1
# CHECK64-ASM: encoding: [0x03,0xc3,0x63,0x38]
amor.d $sp, $t4, $s1

# CHECK64-ASM-AND-OBJ: amxor.w $tp, $t3, $s0
# CHECK64-ASM: encoding: [0xe2,0x3e,0x64,0x38]
amxor.w $tp, $t3, $s0

# CHECK64-ASM-AND-OBJ: amxor.d $a4, $t8, $s5
# CHECK64-ASM: encoding: [0x88,0xd3,0x64,0x38]
amxor.d $a4, $t8, $s5

# CHECK64-ASM-AND-OBJ: ammax.w $ra, $a7, $s0
# CHECK64-ASM: encoding: [0xe1,0x2e,0x65,0x38]
ammax.w $ra, $a7, $s0

# CHECK64-ASM-AND-OBJ: ammax.d $a5, $t8, $s4
# CHECK64-ASM: encoding: [0x69,0xd3,0x65,0x38]
ammax.d $a5, $t8, $s4

# CHECK64-ASM-AND-OBJ: ammin.w $a5, $t2, $s0
# CHECK64-ASM: encoding: [0xe9,0x3a,0x66,0x38]
ammin.w $a5, $t2, $s0

# CHECK64-ASM-AND-OBJ: ammin.d $a5, $t1, $fp
# CHECK64-ASM: encoding: [0xc9,0xb6,0x66,0x38]
ammin.d $a5, $t1, $fp

# CHECK64-ASM-AND-OBJ: ammax.wu $a5, $a7, $fp
# CHECK64-ASM: encoding: [0xc9,0x2e,0x67,0x38]
ammax.wu $a5, $a7, $fp

# CHECK64-ASM-AND-OBJ: ammax.du $a2, $t4, $s2
# CHECK64-ASM: encoding: [0x26,0xc3,0x67,0x38]
ammax.du $a2, $t4, $s2

# CHECK64-ASM-AND-OBJ: ammin.wu $a4, $t6, $s7
# CHECK64-ASM: encoding: [0xc8,0x4b,0x68,0x38]
ammin.wu $a4, $t6, $s7

# CHECK64-ASM-AND-OBJ: ammin.du $a3, $t4, $s2
# CHECK64-ASM: encoding: [0x27,0xc3,0x68,0x38]
ammin.du $a3, $t4, $s2

# CHECK64-ASM-AND-OBJ: amswap_db.w $a2, $t0, $s1
# CHECK64-ASM: encoding: [0x06,0x33,0x69,0x38]
amswap_db.w $a2, $t0, $s1

# CHECK64-ASM-AND-OBJ: amswap_db.d $tp, $t2, $fp
# CHECK64-ASM: encoding: [0xc2,0xba,0x69,0x38]
amswap_db.d $tp, $t2, $fp

# CHECK64-ASM-AND-OBJ: amadd_db.w $a4, $t0, $r21
# CHECK64-ASM: encoding: [0xa8,0x32,0x6a,0x38]
amadd_db.w $a4, $t0, $r21

# CHECK64-ASM-AND-OBJ: amadd_db.d $a1, $t5, $s6
# CHECK64-ASM: encoding: [0xa5,0xc7,0x6a,0x38]
amadd_db.d $a1, $t5, $s6

# CHECK64-ASM-AND-OBJ: amand_db.w $a0, $t7, $fp
# CHECK64-ASM: encoding: [0xc4,0x4e,0x6b,0x38]
amand_db.w $a0, $t7, $fp

# CHECK64-ASM-AND-OBJ: amand_db.d $a6, $t6, $s6
# CHECK64-ASM: encoding: [0xaa,0xcb,0x6b,0x38]
amand_db.d $a6, $t6, $s6

# CHECK64-ASM-AND-OBJ: amor_db.w $a2, $t4, $s0
# CHECK64-ASM: encoding: [0xe6,0x42,0x6c,0x38]
amor_db.w $a2, $t4, $s0

# CHECK64-ASM-AND-OBJ: amor_db.d $sp, $t4, $s1
# CHECK64-ASM: encoding: [0x03,0xc3,0x6c,0x38]
amor_db.d $sp, $t4, $s1

# CHECK64-ASM-AND-OBJ: amxor_db.w $tp, $t3, $s0
# CHECK64-ASM: encoding: [0xe2,0x3e,0x6d,0x38]
amxor_db.w $tp, $t3, $s0

# CHECK64-ASM-AND-OBJ: amxor_db.d $a4, $t8, $s5
# CHECK64-ASM: encoding: [0x88,0xd3,0x6d,0x38]
amxor_db.d $a4, $t8, $s5

# CHECK64-ASM-AND-OBJ: ammax_db.w $ra, $a7, $s0
# CHECK64-ASM: encoding: [0xe1,0x2e,0x6e,0x38]
ammax_db.w $ra, $a7, $s0

# CHECK64-ASM-AND-OBJ: ammax_db.d $a5, $t8, $s4
# CHECK64-ASM: encoding: [0x69,0xd3,0x6e,0x38]
ammax_db.d $a5, $t8, $s4

# CHECK64-ASM-AND-OBJ: ammin_db.w $a5, $t2, $s0
# CHECK64-ASM: encoding: [0xe9,0x3a,0x6f,0x38]
ammin_db.w $a5, $t2, $s0

# CHECK64-ASM-AND-OBJ: ammin_db.d $a5, $t1, $fp
# CHECK64-ASM: encoding: [0xc9,0xb6,0x6f,0x38]
ammin_db.d $a5, $t1, $fp

# CHECK64-ASM-AND-OBJ: ammax_db.wu $a5, $a7, $fp
# CHECK64-ASM: encoding: [0xc9,0x2e,0x70,0x38]
ammax_db.wu $a5, $a7, $fp

# CHECK64-ASM-AND-OBJ: ammax_db.du $a2, $t4, $s2
# CHECK64-ASM: encoding: [0x26,0xc3,0x70,0x38]
ammax_db.du $a2, $t4, $s2

# CHECK64-ASM-AND-OBJ: ammin_db.wu $a4, $t6, $s7
# CHECK64-ASM: encoding: [0xc8,0x4b,0x71,0x38]
ammin_db.wu $a4, $t6, $s7

# CHECK64-ASM-AND-OBJ: ammin_db.du $a3, $t4, $s2
# CHECK64-ASM: encoding: [0x27,0xc3,0x71,0x38]
ammin_db.du $a3, $t4, $s2

# CHECK64-ASM-AND-OBJ: ll.d $s2, $s4, 16
# CHECK64-ASM: encoding: [0x79,0x13,0x00,0x22]
ll.d $s2, $s4, 16

# CHECK64-ASM-AND-OBJ: sc.d $t5, $t5, 244
# CHECK64-ASM: encoding: [0x31,0xf6,0x00,0x23]
sc.d $t5, $t5, 244

.endif

