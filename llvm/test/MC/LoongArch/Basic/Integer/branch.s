## Test valid branch instructions.

# RUN: llvm-mc %s --triple=loongarch32 -show-encoding \
# RUN:     | FileCheck --check-prefix=CHECK-ASM %s
# RUN: llvm-mc %s --triple=loongarch64 -show-encoding \
# RUN:     | FileCheck --check-prefix=CHECK-ASM %s

# CHECK-ASM: beq $a6, $a3, 176
# CHECK-ASM: encoding: [0x47,0xb1,0x00,0x58]
beq $a6, $a3, 176

# CHECK-ASM: bne $s2, $ra, 136
# CHECK-ASM: encoding: [0x21,0x8b,0x00,0x5c]
bne $s2, $ra, 136

# CHECK-ASM: blt $t3, $s7, 168
# CHECK-ASM: encoding: [0xfe,0xa9,0x00,0x60]
blt $t3, $s7, 168

# CHECK-ASM: bge $t0, $t3, 148
# CHECK-ASM: encoding: [0x8f,0x95,0x00,0x64]
bge $t0, $t3, 148

# CHECK-ASM: bltu $t5, $a1, 4
# CHECK-ASM: encoding: [0x25,0x06,0x00,0x68]
bltu $t5, $a1, 4

# CHECK-ASM: bgeu $a2, $s0, 140
# CHECK-ASM: encoding: [0xd7,0x8c,0x00,0x6c]
bgeu $a2, $s0, 140

# CHECK-ASM: beqz $a5, 96
# CHECK-ASM: encoding: [0x20,0x61,0x00,0x40]
beqz $a5, 96

# CHECK-ASM: bnez $sp, 212
# CHECK-ASM: encoding: [0x60,0xd4,0x00,0x44]
bnez $sp, 212

# CHECK-ASM: b 248
# CHECK-ASM: encoding: [0x00,0xf8,0x00,0x50]
b 248

# CHECK-ASM: bl 236
# CHECK-ASM: encoding: [0x00,0xec,0x00,0x54]
bl 236

# CHECK-ASM: jirl $ra, $a0, 4
# CHECK-ASM: encoding: [0x81,0x04,0x00,0x4c]
jirl $ra, $a0, 4

