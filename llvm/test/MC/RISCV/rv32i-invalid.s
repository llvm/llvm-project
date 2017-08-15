# RUN: not llvm-mc -triple riscv32 < %s 2>&1 | FileCheck %s

# Out of range immediates
ori a0, a1, -2049 # CHECK: :[[@LINE]]:13: error: immediate must be an integer in the range [-2048, 2047]
andi ra, sp, 2048 # CHECK: :[[@LINE]]:14: error: immediate must be an integer in the range [-2048, 2047]

# Invalid mnemonics
subs t0, t2, t1 # CHECK: :[[@LINE]]:1: error: unrecognized instruction mnemonic
nandi t0, zero, 0 # CHECK: :[[@LINE]]:1: error: unrecognized instruction mnemonic

# Invalid register names
addi foo, sp, 10 # CHECK: :[[@LINE]]:6: error: unknown operand
slti a10, a2, 0x20 # CHECK: :[[@LINE]]:6: error: unknown operand
slt x32, s0, s0 # CHECK: :[[@LINE]]:5: error: unknown operand

# RV64I mnemonics
addiw a0, sp, 100 # CHECK: :[[@LINE]]:1: error: unrecognized instruction mnemonic
sraw t0, s2, zero # CHECK: :[[@LINE]]:1: error: unrecognized instruction mnemonic

# Invalid operand types
xori sp, 22, 220 # CHECK: :[[@LINE]]:10: error: invalid operand for instruction
sub t0, t2, 1 # CHECK: :[[@LINE]]:13: error: invalid operand for instruction

# Too many operands
add ra, zero, zero, zero # CHECK: :[[@LINE]]:21: error: invalid operand for instruction
sltiu s2, s3, 0x50, 0x60 # CHECK: :[[@LINE]]:21: error: invalid operand for instruction

# Too few operands
ori a0, a1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
xor s2, s2 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
