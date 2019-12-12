# RUN: not llvm-mc  -triple xtensa < %s 2>&1 | FileCheck %s

# Out of range immediates

LBL0:

# imm12m
movi a1, 3000
# CHECK:      error: expected immediate in range [-2048, 2047]

# imm8
addi a1, a2, 300
# CHECK:      error: expected immediate in range [-128, 127]

# imm8
addi a1, a2, -129
# CHECK:      error: expected immediate in range [-128, 127]

# imm8_sh8
addmi a1, a2, 33 
# CHECK:      error: expected immediate in range [-32768, 32512], first 8 bits should be zero

# shimm1_31
slli a1, a2, 0
# CHECK:      error: expected immediate in range [1, 31]

# uimm4
srli a1, a2, 16
# CHECK:      error: expected immediate in range [0, 15]

# uimm5
ssai 32
# CHECK:      error: expected immediate in range [0, 31]

# imm64n_4n
ssai 32
# CHECK:      error: expected immediate in range [0, 31]

# offset8m8
s8i a1, a2, 300
# CHECK:      error: expected immediate in range [0, 255]

# offset16m8
l16si a1, a2, 512
# CHECK:      error: expected immediate in range [0, 510], first bit should be zero

# offset32m8
l32i a1, a2, 1024
# CHECK:      error: expected immediate in range [0, 1020], first 2 bits should be zero

# Invalid number of operands
addi a1, a2 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
addi a1, a2, 4, 4 # CHECK: :[[@LINE]]:17: error: invalid operand for instruction

# Invalid mnemonics
aaa a10, a12 # CHECK: :[[@LINE]]:1: error: unrecognized instruction mnemonic

# Invalid register names
addi a101, sp, 10 # CHECK: :[[@LINE]]:6: error: invalid operand for instruction
wsr.uregister a2 # CHECK: :[[@LINE]]:1: error: invalid register name
or r2, sp, a3 # CHECK: :[[@LINE]]:4: error: invalid operand for instruction

# Invalid operand types
and sp, a2, 10 # CHECK: :[[@LINE]]:13: error: invalid operand for instruction
addi sp, a1, a2 # CHECK: :[[@LINE]]:14: error: expected immediate in range [-128, 127]
