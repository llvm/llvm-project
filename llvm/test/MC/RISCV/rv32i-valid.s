# RUN: llvm-mc %s -triple=riscv32 -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv64 -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 < %s \
# RUN:     | llvm-objdump -d - | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv64 < %s \
# RUN:     | llvm-objdump -d - | FileCheck -check-prefix=CHECK-INST %s

# CHECK-INST: lui a0, 2
# CHECK: encoding: [0x37,0x25,0x00,0x00]
lui a0, 2
# CHECK-INST: lui s11, 552960
# CHECK: encoding: [0xb7,0x0d,0x00,0x87]
lui s11, (0x87000000>>12)
# CHECK-INST: lui t0, 1048575
# CHECK: encoding: [0xb7,0xf2,0xff,0xff]
lui t0, 1048575
# CHECK-INST: lui gp, 0
# CHECK: encoding: [0xb7,0x01,0x00,0x00]
lui gp, 0

# CHECK-INST: auipc a0, 2
# CHECK: encoding: [0x17,0x25,0x00,0x00]
auipc a0, 2
# CHECK-INST: auipc s11, 552960
# CHECK: encoding: [0x97,0x0d,0x00,0x87]
auipc s11, (0x87000000>>12)
# CHECK-INST: auipc t0, 1048575
# CHECK: encoding: [0x97,0xf2,0xff,0xff]
auipc t0, 1048575
# CHECK-INST: auipc gp, 0
# CHECK: encoding: [0x97,0x01,0x00,0x00]
auipc gp, 0

# CHECK-INST: jal a2, 1048574
# CHECK: encoding: [0x6f,0xf6,0xff,0x7f]
jal a2, 1048574
# CHECK-INST: jal a3, 256
# CHECK: encoding: [0xef,0x06,0x00,0x10]
jal a3, 256

# CHECK-INST: jalr a0, a1, -2048
# CHECK: encoding: [0x67,0x85,0x05,0x80]
jalr a0, a1, -2048
# CHECK-INST: jalr t2, t1, 2047
# CHECK: encoding: [0xe7,0x03,0xf3,0x7f]
jalr t2, t1, 2047
# CHECK-INST: jalr sp, zero, 256
# CHECK: encoding: [0x67,0x01,0x00,0x10]
jalr sp, zero, 256

# CHECK-INST: beq s1, s1, 102
# CHECK: encoding: [0x63,0x83,0x94,0x06]
beq s1, s1, 102
# CHECK-INST: bne a4, a5, -4096
# CHECK: encoding: [0x63,0x10,0xf7,0x80]
bne a4, a5, -4096
# CHECK-INST: blt sp, gp, 4094
# CHECK: encoding: [0xe3,0x4f,0x31,0x7e]
blt sp, gp, 4094
# CHECK-INST: bge s2, ra, -224
# CHECK: encoding: [0xe3,0x50,0x19,0xf2]
bge s2, ra, -224
# CHECK-INST: bltu zero, zero, 0
# CHECK: encoding: [0x63,0x60,0x00,0x00]
bltu zero, zero, 0
# CHECK-INST: bgeu s8, sp, 512
# CHECK: encoding: [0x63,0x70,0x2c,0x20]
bgeu s8, sp, 512

# CHECK-INST: lb s3, 4(ra)
# CHECK: encoding: [0x83,0x89,0x40,0x00]
lb s3, 4(ra)
# CHECK-INST: lb s3, 4(ra)
# CHECK: encoding: [0x83,0x89,0x40,0x00]
lb s3, +4(ra)
# CHECK-INST: lh t1, -2048(zero)
# CHECK: encoding: [0x03,0x13,0x00,0x80]
lh t1, -2048(zero)
# CHECK-INST: lh sp, 2047(a0)
# CHECK: encoding: [0x03,0x11,0xf5,0x7f]
lh sp, 2047(a0)
# CHECK-INST: lw a0, 97(a2)
# CHECK: encoding: [0x03,0x25,0x16,0x06]
lw a0, 97(a2)
# CHECK-INST: lbu s5, 0(s6)
# CHECK: encoding: [0x83,0x4a,0x0b,0x00]
lbu s5, 0(s6)
# CHECK-INST: lhu t3, 255(t3)
# CHECK: encoding: [0x03,0x5e,0xfe,0x0f]
lhu t3, 255(t3)

# CHECK-INST: sb a0, 2047(a2)
# CHECK: encoding: [0xa3,0x0f,0xa6,0x7e]
sb a0, 2047(a2)
# CHECK-INST: sh t3, -2048(t5)
# CHECK: encoding: [0x23,0x10,0xcf,0x81]
sh t3, -2048(t5)
# CHECK-INST: sw ra, 999(zero)
# CHECK: encoding: [0xa3,0x23,0x10,0x3e]
sw ra, 999(zero)

# CHECK-INST: addi ra, sp, 2
# CHECK: encoding: [0x93,0x00,0x21,0x00]
addi ra, sp, 2
# CHECK-INST: slti a0, a2, -20
# CHECK: encoding: [0x13,0x25,0xc6,0xfe]
slti a0, a2, -20
# CHECK-INST: sltiu s2, s3, 80
# CHECK: encoding: [0x13,0xb9,0x09,0x05]
sltiu s2, s3, 0x50
# CHECK-INST: xori tp, t1, -99
# CHECK: encoding: [0x13,0x42,0xd3,0xf9]
xori tp, t1, -99
# CHECK-INST: ori a0, a1, -2048
# CHECK: encoding: [0x13,0xe5,0x05,0x80]
ori a0, a1, -2048
# CHECK-INST: andi ra, sp, 2047
# CHECK: encoding: [0x93,0x70,0xf1,0x7f]
andi ra, sp, 2047
# CHECK-INST: andi ra, sp, 2047
# CHECK: encoding: [0x93,0x70,0xf1,0x7f]
andi x1, x2, 2047

# CHECK-INST: slli t3, t3, 31
# CHECK: encoding: [0x13,0x1e,0xfe,0x01]
slli t3, t3, 31
# CHECK-INST: srli a0, a4, 0
# CHECK: encoding: [0x13,0x55,0x07,0x00]
srli a0, a4, 0
# CHECK-INST: srai a2, sp, 15
# CHECK: encoding: [0x13,0x56,0xf1,0x40]
srai a2, sp, 15

# CHECK-INST: add ra, zero, zero
# CHECK: encoding: [0xb3,0x00,0x00,0x00]
add ra, zero, zero
# CHECK-INST: add ra, zero, zero
# CHECK: encoding: [0xb3,0x00,0x00,0x00]
add x1, x0, x0
# CHECK-INST: sub t0, t2, t1
# CHECK: encoding: [0xb3,0x82,0x63,0x40]
sub t0, t2, t1
# CHECK-INST: sll a5, a4, a3
# CHECK: encoding: [0xb3,0x17,0xd7,0x00]
sll a5, a4, a3
# CHECK-INST: slt s0, s0, s0
# CHECK: encoding: [0x33,0x24,0x84,0x00]
slt s0, s0, s0
# CHECK-INST: sltu gp, a0, a1
# CHECK: encoding: [0xb3,0x31,0xb5,0x00]
sltu gp, a0, a1
# CHECK-INST: xor s2, s2, s8
# CHECK: encoding: [0x33,0x49,0x89,0x01]
xor s2, s2, s8
# CHECK-INST: xor s2, s2, s8
# CHECK: encoding: [0x33,0x49,0x89,0x01]
xor x18, x18, x24
# CHECK-INST: srl a0, s0, t0
# CHECK: encoding: [0x33,0x55,0x54,0x00]
srl a0, s0, t0
# CHECK-INST: sra t0, s2, zero
# CHECK: encoding: [0xb3,0x52,0x09,0x40]
sra t0, s2, zero
# CHECK-INST: or s10, t1, ra
# CHECK: encoding: [0x33,0x6d,0x13,0x00]
or s10, t1, ra
# CHECK-INST: and a0, s2, s3
# CHECK: encoding: [0x33,0x75,0x39,0x01]
and a0, s2, s3

# CHECK-INST: fence iorw, iorw
# CHECK: encoding: [0x0f,0x00,0xf0,0x0f]
fence iorw, iorw
# CHECK-INST: fence io, rw
# CHECK: encoding: [0x0f,0x00,0x30,0x0c]
fence io, rw
# CHECK-INST: fence r, w
# CHECK: encoding: [0x0f,0x00,0x10,0x02]
fence r,w
# CHECK-INST: fence w, ir
# CHECK: encoding: [0x0f,0x00,0xa0,0x01]
fence w,ir

# CHECK-INST: fence.i
# CHECK: encoding: [0x0f,0x10,0x00,0x00]
fence.i

# CHECK-INST: ecall
# CHECK: encoding: [0x73,0x00,0x00,0x00]
ecall
# CHECK-INST: ebreak
# CHECK: encoding: [0x73,0x00,0x10,0x00]
ebreak

# CHECK-INST: csrrw t0, 4095, t1
# CHECK: encoding: [0xf3,0x12,0xf3,0xff]
csrrw t0, 0xfff, t1
# CHECK-INST: csrrs s0, 3072, zero
# CHECK: encoding: [0x73,0x24,0x00,0xc0]
csrrs s0, 0xc00, x0
# CHECK-INST: csrrs s3, 1, s5
# CHECK: encoding: [0xf3,0xa9,0x1a,0x00]
csrrs s3, 0x001, s5
# CHECK-INST: csrrc sp, 0, ra
# CHECK: encoding: [0x73,0xb1,0x00,0x00]
csrrc sp, 0x000, ra
# CHECK-INST: csrrwi a5, 0, 0
# CHECK: encoding: [0xf3,0x57,0x00,0x00]
csrrwi a5, 0x000, 0
# CHECK-INST: csrrsi t2, 4095, 31
# CHECK: encoding: [0xf3,0xe3,0xff,0xff]
csrrsi t2, 0xfff, 31
# CHECK-INST: csrrci t1, 320, 5
# CHECK: encoding: [0x73,0xf3,0x02,0x14]
csrrci t1, 0x140, 5
