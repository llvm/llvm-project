# RUN: llvm-mc %s -triple=riscv32 -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple riscv64 -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 < %s \
# RUN:     | llvm-objdump -riscv-no-aliases -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-OBJ,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 < %s \
# RUN:     | llvm-objdump -riscv-no-aliases -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-OBJ,CHECK-ASM-AND-OBJ %s

.equ CONST, 30

# CHECK-ASM-AND-OBJ: lui a0, 2
# CHECK-ASM: encoding: [0x37,0x25,0x00,0x00]
lui a0, 2
# CHECK-ASM-AND-OBJ: lui s11, 552960
# CHECK-ASM: encoding: [0xb7,0x0d,0x00,0x87]
lui s11, (0x87000000>>12)
# CHECK-ASM-AND-OBJ: lui a0, 0
# CHECK-ASM: encoding: [0x37,0x05,0x00,0x00]
lui a0, %hi(2)
# CHECK-ASM-AND-OBJ: lui s11, 552960
# CHECK-ASM: encoding: [0xb7,0x0d,0x00,0x87]
lui s11, (0x87000000>>12)
# CHECK-ASM-AND-OBJ: lui s11, 552960
# CHECK-ASM: encoding: [0xb7,0x0d,0x00,0x87]
lui s11, %hi(0x87000000)
# CHECK-ASM-AND-OBJ: lui t0, 1048575
# CHECK-ASM: encoding: [0xb7,0xf2,0xff,0xff]
lui t0, 1048575
# CHECK-ASM-AND-OBJ: lui gp, 0
# CHECK-ASM: encoding: [0xb7,0x01,0x00,0x00]
lui gp, 0
# CHECK-ASM: lui a0, %hi(foo)
# CHECK-ASM: encoding: [0x37,0bAAAA0101,A,A]
# CHECK-OBJ: lui a0, 0
# CHECK-OBJ: R_RISCV_HI20 foo
lui a0, %hi(foo)
# CHECK-ASM-AND-OBJ: lui a0, 30
# CHECK-ASM: encoding: [0x37,0xe5,0x01,0x00]
lui a0, CONST
# CHECK-ASM-AND-OBJ: lui a0, 31
# CHECK-ASM: encoding: [0x37,0xf5,0x01,0x00]
lui a0, CONST+1

# CHECK-ASM-AND-OBJ: auipc a0, 2
# CHECK-ASM: encoding: [0x17,0x25,0x00,0x00]
auipc a0, 2
# CHECK-ASM-AND-OBJ: auipc s11, 552960
# CHECK-ASM: encoding: [0x97,0x0d,0x00,0x87]
auipc s11, (0x87000000>>12)
# CHECK-ASM-AND-OBJ: auipc t0, 1048575
# CHECK-ASM: encoding: [0x97,0xf2,0xff,0xff]
auipc t0, 1048575
# CHECK-ASM-AND-OBJ: auipc gp, 0
# CHECK-ASM: encoding: [0x97,0x01,0x00,0x00]
auipc gp, 0
# CHECK-ASM: auipc a0, %pcrel_hi(foo)
# CHECK-ASM: encoding: [0x17,0bAAAA0101,A,A]
# CHECK-OBJ: auipc a0, 0
# CHECK-OBJ: R_RISCV_PCREL_HI20 foo
auipc a0, %pcrel_hi(foo)
# CHECK-ASM-AND-OBJ: auipc a0, 30
# CHECK-ASM: encoding: [0x17,0xe5,0x01,0x00]
auipc a0, CONST

# CHECK-ASM-AND-OBJ: jal a2, 1048574
# CHECK-ASM: encoding: [0x6f,0xf6,0xff,0x7f]
jal a2, 1048574
# CHECK-ASM-AND-OBJ: jal a3, 256
# CHECK-ASM: encoding: [0xef,0x06,0x00,0x10]
jal a3, 256
# CHECK-ASM: jal a0, foo
# CHECK-ASM: encoding: [0x6f,0bAAAA0101,A,A]
# CHECK-OBJ: jal a0, 0
# CHECK-OBJ: R_RISCV_JAL foo
jal a0, foo
# CHECK-ASM: jal a0, a0
# CHECK-ASM: encoding: [0x6f,0bAAAA0101,A,A]
# CHECK-OBJ: jal a0, 0
# CHECK-OBJ: R_RISCV_JAL a0
jal a0, a0
# CHECK-ASM-AND-OBJ: jal a0, 30
# CHECK-ASM: encoding: [0x6f,0x05,0xe0,0x01]
jal a0, CONST

# CHECK-ASM-AND-OBJ: jalr a0, a1, -2048
# CHECK-ASM: encoding: [0x67,0x85,0x05,0x80]
jalr a0, a1, -2048
# CHECK-ASM-AND-OBJ: jalr a0, a1, -2048
# CHECK-ASM: encoding: [0x67,0x85,0x05,0x80]
jalr a0, a1, %lo(2048)
# CHECK-ASM-AND-OBJ: jalr t2, t1, 2047
# CHECK-ASM: encoding: [0xe7,0x03,0xf3,0x7f]
jalr t2, t1, 2047
# CHECK-ASM-AND-OBJ: jalr sp, zero, 256
# CHECK-ASM: encoding: [0x67,0x01,0x00,0x10]
jalr sp, zero, 256
# CHECK-ASM-AND-OBJ: jalr a1, a2, 30
# CHECK-ASM: encoding: [0xe7,0x05,0xe6,0x01]
jalr a1, a2, CONST

# CHECK-ASM-AND-OBJ: beq s1, s1, 102
# CHECK-ASM: encoding: [0x63,0x83,0x94,0x06]
beq s1, s1, 102
# CHECK-ASM-AND-OBJ: bne a4, a5, -4096
# CHECK-ASM: encoding: [0x63,0x10,0xf7,0x80]
bne a4, a5, -4096
# CHECK-ASM-AND-OBJ: blt sp, gp, 4094
# CHECK-ASM: encoding: [0xe3,0x4f,0x31,0x7e]
blt sp, gp, 4094
# CHECK-ASM-AND-OBJ: bge s2, ra, -224
# CHECK-ASM: encoding: [0xe3,0x50,0x19,0xf2]
bge s2, ra, -224
# CHECK-ASM-AND-OBJ: bltu zero, zero, 0
# CHECK-ASM: encoding: [0x63,0x60,0x00,0x00]
bltu zero, zero, 0
# CHECK-ASM-AND-OBJ: bgeu s8, sp, 512
# CHECK-ASM: encoding: [0x63,0x70,0x2c,0x20]
bgeu s8, sp, 512
# CHECK-ASM-AND-OBJ: bgeu t0, t1, 30
# CHECK-ASM: encoding: [0x63,0xff,0x62,0x00]
bgeu t0, t1, CONST

# CHECK-ASM-AND-OBJ: lb s3, 4(ra)
# CHECK-ASM: encoding: [0x83,0x89,0x40,0x00]
lb s3, 4(ra)
# CHECK-ASM-AND-OBJ: lb s3, 4(ra)
# CHECK-ASM: encoding: [0x83,0x89,0x40,0x00]
lb s3, +4(ra)
# CHECK-ASM-AND-OBJ: lh t1, -2048(zero)
# CHECK-ASM: encoding: [0x03,0x13,0x00,0x80]
lh t1, -2048(zero)
# CHECK-ASM-AND-OBJ: lh t1, -2048(zero)
# CHECK-ASM: encoding: [0x03,0x13,0x00,0x80]
lh t1, %lo(2048)(zero)
# CHECK-ASM-AND-OBJ: lh sp, 2047(a0)
# CHECK-ASM: encoding: [0x03,0x11,0xf5,0x7f]
lh sp, 2047(a0)
# CHECK-ASM-AND-OBJ: lw a0, 97(a2)
# CHECK-ASM: encoding: [0x03,0x25,0x16,0x06]
lw a0, 97(a2)
# CHECK-ASM: lbu s5, %lo(foo)(s6)
# CHECK-ASM: encoding: [0x83,0x4a,0bAAAA1011,A]
# CHECK-OBJ: lbu s5, 0(s6)
# CHECK-OBJ: R_RISCV_LO12
lbu s5, %lo(foo)(s6)
# CHECK-ASM: lhu t3, %pcrel_lo(foo)(t3)
# CHECK-ASM: encoding: [0x03,0x5e,0bAAAA1110,A]
# CHECK-OBJ: lhu t3, 0(t3)
# CHECK-OBJ: R_RISCV_PCREL_LO12
lhu t3, %pcrel_lo(foo)(t3)
# CHECK-ASM-AND-OBJ: lb t0, 30(t1)
# CHECK-ASM: encoding: [0x83,0x02,0xe3,0x01]
lb t0, CONST(t1)

# CHECK-ASM-AND-OBJ: sb a0, 2047(a2)
# CHECK-ASM: encoding: [0xa3,0x0f,0xa6,0x7e]
sb a0, 2047(a2)
# CHECK-ASM-AND-OBJ: sh t3, -2048(t5)
# CHECK-ASM: encoding: [0x23,0x10,0xcf,0x81]
sh t3, -2048(t5)
# CHECK-ASM-AND-OBJ: sh t3, -2048(t5)
# CHECK-ASM: encoding: [0x23,0x10,0xcf,0x81]
sh t3, %lo(2048)(t5)
# CHECK-ASM-AND-OBJ: sw ra, 999(zero)
# CHECK-ASM: encoding: [0xa3,0x23,0x10,0x3e]
sw ra, 999(zero)
# CHECK-ASM-AND-OBJ: sw a0, 30(t0)
# CHECK-ASM: encoding: [0x23,0xaf,0xa2,0x00]
sw a0, CONST(t0)

# CHECK-ASM-AND-OBJ: addi ra, sp, 2
# CHECK-ASM: encoding: [0x93,0x00,0x21,0x00]
addi ra, sp, 2
# CHECK-ASM: addi ra, sp, %lo(foo)
# CHECK-ASM: encoding: [0x93,0x00,0bAAAA0001,A]
# CHECK-OBJ: addi ra, sp, 0
# CHECK-OBJ: R_RISCV_LO12
addi ra, sp, %lo(foo)
# CHECK-ASM-AND-OBJ: addi ra, sp, 30
# CHECK-ASM: encoding: [0x93,0x00,0xe1,0x01]
addi ra, sp, CONST
# CHECK-ASM-AND-OBJ: slti a0, a2, -20
# CHECK-ASM: encoding: [0x13,0x25,0xc6,0xfe]
slti a0, a2, -20
# CHECK-ASM-AND-OBJ: sltiu s2, s3, 80
# CHECK-ASM: encoding: [0x13,0xb9,0x09,0x05]
sltiu s2, s3, 0x50
# CHECK-ASM-AND-OBJ: xori tp, t1, -99
# CHECK-ASM: encoding: [0x13,0x42,0xd3,0xf9]
xori tp, t1, -99
# CHECK-ASM-AND-OBJ: ori a0, a1, -2048
# CHECK-ASM: encoding: [0x13,0xe5,0x05,0x80]
ori a0, a1, -2048
# CHECK-ASM-AND-OBJ: ori a0, a1, -2048
# CHECK-ASM: encoding: [0x13,0xe5,0x05,0x80]
ori a0, a1, %lo(2048)
# CHECK-ASM-AND-OBJ: andi ra, sp, 2047
# CHECK-ASM: encoding: [0x93,0x70,0xf1,0x7f]
andi ra, sp, 2047
# CHECK-ASM-AND-OBJ: andi ra, sp, 2047
# CHECK-ASM: encoding: [0x93,0x70,0xf1,0x7f]
andi x1, x2, 2047

# CHECK-ASM-AND-OBJ: slli t3, t3, 31
# CHECK-ASM: encoding: [0x13,0x1e,0xfe,0x01]
slli t3, t3, 31
# CHECK-ASM-AND-OBJ: srli a0, a4, 0
# CHECK-ASM: encoding: [0x13,0x55,0x07,0x00]
srli a0, a4, 0
# CHECK-ASM-AND-OBJ: srai a2, sp, 15
# CHECK-ASM: encoding: [0x13,0x56,0xf1,0x40]
srai a2, sp, 15
# CHECK-ASM-AND-OBJ: slli t3, t3, 30
# CHECK-ASM: encoding: [0x13,0x1e,0xee,0x01]
slli t3, t3, CONST

# CHECK-ASM-AND-OBJ: add ra, zero, zero
# CHECK-ASM: encoding: [0xb3,0x00,0x00,0x00]
add ra, zero, zero
# CHECK-ASM-AND-OBJ: add ra, zero, zero
# CHECK-ASM: encoding: [0xb3,0x00,0x00,0x00]
add x1, x0, x0
# CHECK-ASM-AND-OBJ: sub t0, t2, t1
# CHECK-ASM: encoding: [0xb3,0x82,0x63,0x40]
sub t0, t2, t1
# CHECK-ASM-AND-OBJ: sll a5, a4, a3
# CHECK-ASM: encoding: [0xb3,0x17,0xd7,0x00]
sll a5, a4, a3
# CHECK-ASM-AND-OBJ: slt s0, s0, s0
# CHECK-ASM: encoding: [0x33,0x24,0x84,0x00]
slt s0, s0, s0
# CHECK-ASM-AND-OBJ: sltu gp, a0, a1
# CHECK-ASM: encoding: [0xb3,0x31,0xb5,0x00]
sltu gp, a0, a1
# CHECK-ASM-AND-OBJ: xor s2, s2, s8
# CHECK-ASM: encoding: [0x33,0x49,0x89,0x01]
xor s2, s2, s8
# CHECK-ASM-AND-OBJ: xor s2, s2, s8
# CHECK-ASM: encoding: [0x33,0x49,0x89,0x01]
xor x18, x18, x24
# CHECK-ASM-AND-OBJ: srl a0, s0, t0
# CHECK-ASM: encoding: [0x33,0x55,0x54,0x00]
srl a0, s0, t0
# CHECK-ASM-AND-OBJ: sra t0, s2, zero
# CHECK-ASM: encoding: [0xb3,0x52,0x09,0x40]
sra t0, s2, zero
# CHECK-ASM-AND-OBJ: or s10, t1, ra
# CHECK-ASM: encoding: [0x33,0x6d,0x13,0x00]
or s10, t1, ra
# CHECK-ASM-AND-OBJ: and a0, s2, s3
# CHECK-ASM: encoding: [0x33,0x75,0x39,0x01]
and a0, s2, s3

# CHECK-ASM-AND-OBJ: fence iorw, iorw
# CHECK-ASM: encoding: [0x0f,0x00,0xf0,0x0f]
fence iorw, iorw
# CHECK-ASM-AND-OBJ: fence io, rw
# CHECK-ASM: encoding: [0x0f,0x00,0x30,0x0c]
fence io, rw
# CHECK-ASM-AND-OBJ: fence r, w
# CHECK-ASM: encoding: [0x0f,0x00,0x10,0x02]
fence r,w
# CHECK-ASM-AND-OBJ: fence w, ir
# CHECK-ASM: encoding: [0x0f,0x00,0xa0,0x01]
fence w,ir
# CHECK-ASM-AND-OBJ: fence.tso
# CHECK-ASM: encoding: [0x0f,0x00,0x30,0x83]
fence.tso

# CHECK-ASM-AND-OBJ: fence.i
# CHECK-ASM: encoding: [0x0f,0x10,0x00,0x00]
fence.i

# CHECK-ASM-AND-OBJ: ecall
# CHECK-ASM: encoding: [0x73,0x00,0x00,0x00]
ecall
# CHECK-ASM-AND-OBJ: ebreak
# CHECK-ASM: encoding: [0x73,0x00,0x10,0x00]
ebreak
# CHECK-ASM-AND-OBJ: unimp
# CHECK-ASM: encoding: [0x73,0x10,0x00,0xc0]
unimp

.equ CONST, 16

# CHECK-ASM-AND-OBJ: csrrw t0, 4095, t1
# CHECK-ASM: encoding: [0xf3,0x12,0xf3,0xff]
csrrw t0, 0xfff, t1
# CHECK-ASM-AND-OBJ: csrrs s0, cycle, zero
# CHECK-ASM: encoding: [0x73,0x24,0x00,0xc0]
csrrs s0, 0xc00, x0
# CHECK-ASM-AND-OBJ: csrrs s3, fflags, s5
# CHECK-ASM: encoding: [0xf3,0xa9,0x1a,0x00]
csrrs s3, 0x001, s5
# CHECK-ASM-AND-OBJ: csrrc sp, ustatus, ra
# CHECK-ASM: encoding: [0x73,0xb1,0x00,0x00]
csrrc sp, 0x000, ra
# CHECK-ASM-AND-OBJ: csrrwi a5, ustatus, 0
# CHECK-ASM: encoding: [0xf3,0x57,0x00,0x00]
csrrwi a5, 0x000, 0
# CHECK-ASM-AND-OBJ: csrrsi t2, 4095, 31
# CHECK-ASM: encoding: [0xf3,0xe3,0xff,0xff]
csrrsi t2, 0xfff, 31
# CHECK-ASM-AND-OBJ: csrrci t1, sscratch, 5
# CHECK-ASM: encoding: [0x73,0xf3,0x02,0x14]
csrrci t1, 0x140, 5
