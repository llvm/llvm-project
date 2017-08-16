# RUN: llvm-mc %s -triple=riscv32 -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv64 -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s

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
