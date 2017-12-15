# RUN: llvm-mc %s -triple=riscv64 -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv64 < %s \
# RUN:     | llvm-objdump -riscv-no-aliases -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s

# CHECK-INST: lwu zero, 4(ra)
# CHECK: encoding: [0x03,0xe0,0x40,0x00]
lwu x0, 4(x1)
# CHECK-INST: lwu sp, 4(gp)
# CHECK: encoding: [0x03,0xe1,0x41,0x00]
lwu x2, +4(x3)
# CHECK-INST: lwu tp, -2048(t0)
# CHECK: encoding: [0x03,0xe2,0x02,0x80]
lwu x4, -2048(x5)
# CHECK-INST: lwu t1, -2048(t2)
# CHECK: encoding: [0x03,0xe3,0x03,0x80]
lwu x6, %lo(2048)(x7)
# CHECK-INST: lwu s0, 2047(s1)
# CHECK: encoding: [0x03,0xe4,0xf4,0x7f]
lwu x8, 2047(x9)

# CHECK-INST: ld a0, -2048(a1)
# CHECK: encoding: [0x03,0xb5,0x05,0x80]
ld x10, -2048(x11)
# CHECK-INST: ld a2, -2048(a3)
# CHECK: encoding: [0x03,0xb6,0x06,0x80]
ld x12, %lo(2048)(x13)
# CHECK-INST: ld a4, 2047(a5)
# CHECK: encoding: [0x03,0xb7,0xf7,0x7f]
ld x14, 2047(x15)

# CHECK-INST: sd a6, -2048(a7)
# CHECK: encoding: [0x23,0xb0,0x08,0x81]
sd x16, -2048(x17)
# CHECK-INST: sd s2, -2048(s3)
# CHECK: encoding: [0x23,0xb0,0x29,0x81]
sd x18, %lo(2048)(x19)
# CHECK-INST: sd s4, 2047(s5)
# CHECK: encoding: [0xa3,0xbf,0x4a,0x7f]
sd x20, 2047(x21)

# CHECK-INST: slli s6, s7, 45
# CHECK: encoding: [0x13,0x9b,0xdb,0x02]
slli x22, x23, 45
# CHECK-INST: srli s8, s9, 0
# CHECK: encoding: [0x13,0xdc,0x0c,0x00]
srli x24, x25, 0
# CHECK-INST: srai s10, s11, 31
# CHECK: encoding: [0x13,0xdd,0xfd,0x41]
srai x26, x27, 31

# CHECK-INST: addiw t3, t4, -2048
# CHECK: encoding: [0x1b,0x8e,0x0e,0x80]
addiw x28, x29, -2048
# CHECK-INST: addiw t5, t6, 2047
# CHECK: encoding: [0x1b,0x8f,0xff,0x7f]
addiw x30, x31, 2047

# CHECK-INST: slliw zero, ra, 0
# CHECK: encoding: [0x1b,0x90,0x00,0x00]
slliw zero, ra, 0
# CHECK-INST: slliw sp, gp, 31
# CHECK: encoding: [0x1b,0x91,0xf1,0x01]
slliw sp, gp, 31
# CHECK-INST: srliw tp, t0, 0
# CHECK: encoding: [0x1b,0xd2,0x02,0x00]
srliw tp, t0, 0
# CHECK-INST: srliw t1, t2, 31
# CHECK: encoding: [0x1b,0xd3,0xf3,0x01]
srliw t1, t2, 31
# CHECK-INST: sraiw s0, s1, 0
# CHECK: encoding: [0x1b,0xd4,0x04,0x40]
sraiw s0, s1, 0
# CHECK-INST: sraiw a0, a1, 31
# CHECK: encoding: [0x1b,0xd5,0xf5,0x41]
sraiw a0, a1, 31

# CHECK-INST: addw a2, a3, a4
# CHECK: encoding: [0x3b,0x86,0xe6,0x00]
addw a2, a3, a4
# CHECK-INST: addw a5, a6, a7
# CHECK: encoding: [0xbb,0x07,0x18,0x01]
addw a5, a6, a7
# CHECK-INST: subw s2, s3, s4
# CHECK: encoding: [0x3b,0x89,0x49,0x41]
subw s2, s3, s4
# CHECK-INST: subw s5, s6, s7
# CHECK: encoding: [0xbb,0x0a,0x7b,0x41]
subw s5, s6, s7
# CHECK-INST: sllw s8, s9, s10
# CHECK: encoding: [0x3b,0x9c,0xac,0x01]
sllw s8, s9, s10
# CHECK-INST: srlw s11, t3, t4
# CHECK: encoding: [0xbb,0x5d,0xde,0x01]
srlw s11, t3, t4
# CHECK-INST: sraw t5, t6, zero
# CHECK: encoding: [0x3b,0xdf,0x0f,0x40]
sraw t5, t6, zero
