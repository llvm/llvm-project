## Test valid CRC check instructions.

# RUN: llvm-mc %s --triple=loongarch64 -show-encoding \
# RUN:     | FileCheck --check-prefix=CHECK-ASM %s

# CHECK-ASM: crc.w.b.w $s1, $a3, $tp
# CHECK-ASM: encoding: [0xf8,0x08,0x24,0x00]
crc.w.b.w $s1, $a3, $tp

# CHECK-ASM: crc.w.h.w $s8, $a6, $t6
# CHECK-ASM: encoding: [0x5f,0xc9,0x24,0x00]
crc.w.h.w $s8, $a6, $t6

# CHECK-ASM: crc.w.w.w $s5, $a2, $a6
# CHECK-ASM: encoding: [0xdc,0x28,0x25,0x00]
crc.w.w.w $s5, $a2, $a6

# CHECK-ASM: crc.w.d.w $s5, $a7, $s8
# CHECK-ASM: encoding: [0x7c,0xfd,0x25,0x00]
crc.w.d.w $s5, $a7, $s8

# CHECK-ASM: crcc.w.b.w $t3, $t6, $sp
# CHECK-ASM: encoding: [0x4f,0x0e,0x26,0x00]
crcc.w.b.w $t3, $t6, $sp

# CHECK-ASM: crcc.w.h.w $r21, $s6, $t6
# CHECK-ASM: encoding: [0xb5,0xcb,0x26,0x00]
crcc.w.h.w $r21, $s6, $t6

# CHECK-ASM: crcc.w.w.w $t5, $t2, $t1
# CHECK-ASM: encoding: [0xd1,0x35,0x27,0x00]
crcc.w.w.w $t5, $t2, $t1

# CHECK-ASM: crcc.w.d.w $s7, $r21, $s4
# CHECK-ASM: encoding: [0xbe,0xee,0x27,0x00]
crcc.w.d.w $s7, $r21, $s4

