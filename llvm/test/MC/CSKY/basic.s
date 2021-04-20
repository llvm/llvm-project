# RUN: llvm-mc %s -triple=csky -show-encoding | FileCheck -check-prefixes=CHECK-ASM %s

# CHECK-ASM: addi32 a0, sp, 2
# CHECK-ASM: encoding: [0x0e,0xe4,0x01,0x00]
addi32 a0, sp, 2

# CHECK-ASM: subi32 a0, sp, 2
# CHECK-ASM: encoding: [0x0e,0xe4,0x01,0x10]
subi32 a0, sp, 2

# CHECK-ASM: andi32 a0, sp, 2
# CHECK-ASM: encoding: [0x0e,0xe4,0x02,0x20]
andi32 a0, sp, 2

# CHECK-ASM: andni32 a0, sp, 2
# CHECK-ASM: encoding: [0x0e,0xe4,0x02,0x30]
andni32 a0, sp, 2

# CHECK-ASM: xori32 a0, sp, 2
# CHECK-ASM: encoding: [0x0e,0xe4,0x02,0x40]
xori32 a0, sp, 2

# CHECK-ASM: lsli32 a0, sp, 2
# CHECK-ASM: encoding: [0x4e,0xc4,0x20,0x48]
lsli32 a0, sp, 2

# CHECK-ASM: lsri32 a0, sp, 2
# CHECK-ASM: encoding: [0x4e,0xc4,0x40,0x48]
lsri32 a0, sp, 2

# CHECK-ASM: asri32 a0, sp, 2
# CHECK-ASM: encoding: [0x4e,0xc4,0x80,0x48]
asri32 a0, sp, 2


# CHECK-ASM: addu32 a3, l0, l1
# CHECK-ASM: encoding: [0xa4,0xc4,0x23,0x00]
addu32 a3, l0, l1

# CHECK-ASM: subu32 a3, l0, l1
# CHECK-ASM: encoding: [0xa4,0xc4,0x83,0x00]
subu32 a3, l0, l1

# CHECK-ASM: and32 a3, l0, l1
# CHECK-ASM: encoding: [0xa4,0xc4,0x23,0x20]
and32 a3, l0, l1

# CHECK-ASM: andn32 a3, l0, l1
# CHECK-ASM: encoding: [0xa4,0xc4,0x43,0x20]
andn32 a3, l0, l1

# CHECK-ASM: or32 a3, l0, l1
# CHECK-ASM: encoding: [0xa4,0xc4,0x23,0x24]
or32 a3, l0, l1

# CHECK-ASM: xor32 a3, l0, l1
# CHECK-ASM: encoding: [0xa4,0xc4,0x43,0x24]
xor32 a3, l0, l1

# CHECK-ASM: nor32 a3, l0, l1
# CHECK-ASM: encoding: [0xa4,0xc4,0x83,0x24]
nor32 a3, l0, l1

# CHECK-ASM: lsl32 a3, l0, l1
# CHECK-ASM: encoding: [0xa4,0xc4,0x23,0x40]
lsl32 a3, l0, l1

# CHECK-ASM: lsr32 a3, l0, l1
# CHECK-ASM: encoding: [0xa4,0xc4,0x43,0x40]
lsr32 a3, l0, l1

# CHECK-ASM: asr32 a3, l0, l1
# CHECK-ASM: encoding: [0xa4,0xc4,0x83,0x40]
asr32 a3, l0, l1

# CHECK-ASM: mult32 a3, l0, l1
# CHECK-ASM: encoding: [0xa4,0xc4,0x23,0x84]
mult32 a3, l0, l1

# CHECK-ASM: divs32 a3, l0, l1
# CHECK-ASM: encoding: [0xa4,0xc4,0x43,0x80]
divs32 a3, l0, l1

# CHECK-ASM: divu32 a3, l0, l1
# CHECK-ASM: encoding: [0xa4,0xc4,0x23,0x80]
divu32 a3, l0, l1

# CHECK-ASM: not32 a3, l0
# CHECK-ASM: encoding: [0x84,0xc4,0x83,0x24]
not32 a3, l0


# RUN: not llvm-mc -triple csky --defsym=ERR=1 < %s 2>&1 | FileCheck %s

.ifdef ERR

# uimm12/oimm12/nimm12
addi32 a0, a0, 0 # CHECK: :[[#@LINE]]:16: error: immediate must be an integer in the range [1, 4096]
subi32 a0, a0, 4097 # CHECK: :[[#@LINE]]:16: error: immediate must be an integer in the range [1, 4096]
andi32 a0, a0, 4096 # CHECK: :[[#@LINE]]:16: error: immediate must be an integer in the range [0, 4095]
andni32 a0, a0, 4096 # CHECK: :[[#@LINE]]:17: error: immediate must be an integer in the range [0, 4095]
xori32 a0, a0, 4096 # CHECK: :[[#@LINE]]:16: error: immediate must be an integer in the range [0, 4095]


# uimm5
lsli32 a0, a0, 32 # CHECK: :[[#@LINE]]:16: error: immediate must be an integer in the range [0, 31]
lsri32 a0, a0, 32 # CHECK: :[[#@LINE]]:16: error: immediate must be an integer in the range [0, 31]
asri32 a0, a0, 32 # CHECK: :[[#@LINE]]:16: error: immediate must be an integer in the range [0, 31]


# Invalid mnemonics
subs t0, t2, t1 # CHECK: :[[#@LINE]]:1: error: unrecognized instruction mnemonic
nandi t0, t2, 0 # CHECK: :[[#@LINE]]:1: error: unrecognized instruction mnemonic

# Invalid register names
addi32 foo, sp, 10 # CHECK: :[[#@LINE]]:8: error: unknown operand
lsli32 a10, a2, 0x20 # CHECK: :[[#@LINE]]:8: error: unknown operand
asri32 x32, s0, s0 # CHECK: :[[#@LINE]]:8: error: unknown operand

# Invalid operand types
xori32 sp, 22, 220 # CHECK: :[[#@LINE]]:12: error: invalid operand for instruction
subu32 t0, t2, 1 # CHECK: :[[#@LINE]]:16: error: invalid operand for instruction

# Too many operands
subu32 t2, t3, 0x50, 0x60 # CHECK: :[[#@LINE]]:22: error: invalid operand for instruction

# Too few operands
xori32 a0, a1 # CHECK: :[[#@LINE]]:1: error: too few operands for instruction
xor32 a0, a2 # CHECK: :[[#@LINE]]:1: error: too few operands for instruction

.endif
