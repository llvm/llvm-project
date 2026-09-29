## Test valid privilege instructions

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

# CHECK-ASM-AND-OBJ: csrrd $s3, 30
# CHECK-ASM: encoding: [0x1a,0x78,0x00,0x04]
csrrd $s3, 30

# CHECK-ASM-AND-OBJ: csrwr $s1, 194
# CHECK-ASM: encoding: [0x38,0x08,0x03,0x04]
csrwr $s1, 194

# CHECK-ASM-AND-OBJ: csrxchg $a2, $s4, 214
# CHECK-ASM: encoding: [0x66,0x5b,0x03,0x04]
csrxchg $a2, $s4, 214

# CHECK-ASM-AND-OBJ: iocsrrd.b $s3, $s1
# CHECK-ASM: encoding: [0x1a,0x03,0x48,0x06]
iocsrrd.b $s3, $s1

# CHECK-ASM-AND-OBJ: iocsrrd.h $a1, $s4
# CHECK-ASM: encoding: [0x65,0x07,0x48,0x06]
iocsrrd.h $a1, $s4

# CHECK-ASM-AND-OBJ: iocsrrd.w $a6, $t8
# CHECK-ASM: encoding: [0x8a,0x0a,0x48,0x06]
iocsrrd.w $a6, $t8

# CHECK-ASM-AND-OBJ: iocsrwr.b $a0, $s0
# CHECK-ASM: encoding: [0xe4,0x12,0x48,0x06]
iocsrwr.b $a0, $s0

# CHECK-ASM-AND-OBJ: iocsrwr.h $a7, $zero
# CHECK-ASM: encoding: [0x0b,0x14,0x48,0x06]
iocsrwr.h $a7, $zero

# CHECK-ASM-AND-OBJ: iocsrwr.w $t8, $s3
# CHECK-ASM: encoding: [0x54,0x1b,0x48,0x06]
iocsrwr.w $t8, $s3

# CHECK-ASM-AND-OBJ: cacop 0, $a6, 27
# CHECK-ASM: encoding: [0x40,0x6d,0x00,0x06]
cacop 0, $a6, 27

# CHECK-ASM-AND-OBJ: tlbclr
# CHECK-ASM: encoding: [0x00,0x20,0x48,0x06]
tlbclr

# CHECK-ASM-AND-OBJ: tlbflush
# CHECK-ASM: encoding: [0x00,0x24,0x48,0x06]
tlbflush

# CHECK-ASM-AND-OBJ: tlbsrch
# CHECK-ASM: encoding: [0x00,0x28,0x48,0x06]
tlbsrch

# CHECK-ASM-AND-OBJ: tlbrd
# CHECK-ASM: encoding: [0x00,0x2c,0x48,0x06]
tlbrd

# CHECK-ASM-AND-OBJ: tlbwr
# CHECK-ASM: encoding: [0x00,0x30,0x48,0x06]
tlbwr

# CHECK-ASM-AND-OBJ: tlbfill
# CHECK-ASM: encoding: [0x00,0x34,0x48,0x06]
tlbfill

# CHECK-ASM-AND-OBJ: invtlb 16, $s6, $s2
# CHECK-ASM: encoding: [0xb0,0xe7,0x49,0x06]
invtlb 16, $s6, $s2

# CHECK-ASM-AND-OBJ: lddir $t0, $s7, 92
# CHECK-ASM: encoding: [0xcc,0x73,0x41,0x06]
lddir $t0, $s7, 92

# CHECK-ASM-AND-OBJ: ldpte $t6, 200
# CHECK-ASM: encoding: [0x40,0x22,0x47,0x06]
ldpte $t6, 200

# CHECK-ASM-AND-OBJ: ertn
# CHECK-ASM: encoding: [0x00,0x38,0x48,0x06]
ertn

# CHECK-ASM-AND-OBJ: dbcl 201
# CHECK-ASM: encoding: [0xc9,0x80,0x2a,0x00]
dbcl 201

# CHECK-ASM-AND-OBJ: idle 204
# CHECK-ASM: encoding: [0xcc,0x80,0x48,0x06]
idle 204

#############################################################
## Instructions only for loongarch64
#############################################################

.ifdef LA64

# CHECK64-ASM-AND-OBJ: iocsrrd.d $t5, $s2
# CHECK64-ASM: encoding: [0x31,0x0f,0x48,0x06]
iocsrrd.d $t5, $s2

# CHECK64-ASM-AND-OBJ: iocsrwr.d $t8, $a3
# CHECK64-ASM: encoding: [0xf4,0x1c,0x48,0x06]
iocsrwr.d $t8, $a3

.endif
