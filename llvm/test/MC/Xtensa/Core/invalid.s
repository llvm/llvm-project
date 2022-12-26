# RUN: not llvm-mc -triple xtensa %s 2>&1 | FileCheck %s

# Out of range immediates

LBL0:

# imm8
addi a1, a2, 300
# CHECK: :[[#@LINE-1]]:14: error: expected immediate in range [-128, 127]

addi a1, a2, -129
# CHECK: :[[#@LINE-1]]:14: error: expected immediate in range [-128, 127]

# imm8_sh8
addmi a1, a2, 33
# CHECK: :[[#@LINE-1]]:15: error: expected immediate in range [-32768, 32512], first 8 bits should be zero

# Invalid number of operands
addi a1, a2
# CHECK: :[[#@LINE-1]]:1: error: too few operands for instruction
addi a1, a2, 4, 4
# CHECK: :[[#@LINE-1]]:17: error: invalid operand for instruction

# Invalid mnemonics
aaa a10, a12
# CHECK: :[[#@LINE-1]]:1: error: unrecognized instruction mnemonic

# Invalid operand types
and sp, a2, 10
# CHECK: :[[#@LINE-1]]:13: error: invalid operand for instruction
addi sp, a1, a2
# CHECK: :[[#@LINE-1]]:14: error: expected immediate in range [-128, 127]

# Invalid register names
# Instruction format RRR
or r2, sp, a3
# CHECK: :[[#@LINE-1]]:4: error: invalid operand for instruction
and a1, r10, a3
# CHECK: :[[#@LINE-1]]:9: error: invalid operand for instruction
sub a1, sp, a100
# CHECK: :[[#@LINE-1]]:13: error: invalid operand for instruction

# Instruction format RRI8
addi a101, sp, 10
# CHECK: :[[#@LINE-1]]:6: error: invalid operand for instruction
addi a1, r10, 10
# CHECK: :[[#@LINE-1]]:10: error: invalid operand for instruction

# Invalid operands order
# Instruction format RRI8
addi a1, 10, a2
# CHECK: :[[#@LINE-1]]:10: error: invalid operand for instruction
