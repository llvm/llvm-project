# RUN: not llvm-mc -triple riscv32 < %s 2>&1 | FileCheck %s

# Out of range immediates
## fencearg
fence iorw, iore # CHECK: :[[@LINE]]:13: error: operand must be formed of letters selected in-order from 'iorw'
fence wr, wr # CHECK: :[[@LINE]]:7: error: operand must be formed of letters selected in-order from 'iorw'
fence rw, rr # CHECK: :[[@LINE]]:11: error: operand must be formed of letters selected in-order from 'iorw'
fence 1, rw # CHECK: :[[@LINE]]:7: error: operand must be formed of letters selected in-order from 'iorw'

## uimm5
slli a0, a0, 32 # CHECK: :[[@LINE]]:14: error: immediate must be an integer in the range [0, 31]
srli a0, a0, -1 # CHECK: :[[@LINE]]:14: error: immediate must be an integer in the range [0, 31]
srai a0, a0, -19 # CHECK: :[[@LINE]]:14: error: immediate must be an integer in the range [0, 31]
csrrwi a1, 0x1, -1 # CHECK: :[[@LINE]]:17: error: immediate must be an integer in the range [0, 31]
csrrsi t1, 999, 32 # CHECK: :[[@LINE]]:17: error: immediate must be an integer in the range [0, 31]
csrrci x0, 43, -90 # CHECK: :[[@LINE]]:16: error: immediate must be an integer in the range [0, 31]

## uimm12
csrrw a0, -1, a0 # CHECK: :[[@LINE]]:11: error: immediate must be an integer in the range [0, 4095]
csrrs a0, 4096, a0 # CHECK: :[[@LINE]]:11: error: immediate must be an integer in the range [0, 4095]
csrrs a0, -0xf, a0 # CHECK: :[[@LINE]]:11: error: immediate must be an integer in the range [0, 4095]
csrrc a0, 0x1000, a0 # CHECK: :[[@LINE]]:11: error: immediate must be an integer in the range [0, 4095]
csrrwi a0, -50, 0 # CHECK: :[[@LINE]]:12: error: immediate must be an integer in the range [0, 4095]
csrrsi a0, 4097, a0 # CHECK: :[[@LINE]]:12: error: immediate must be an integer in the range [0, 4095]
csrrci a0, 0xffff, a0 # CHECK: :[[@LINE]]:12: error: immediate must be an integer in the range [0, 4095]

## simm12
ori a0, a1, -2049 # CHECK: :[[@LINE]]:13: error: immediate must be an integer in the range [-2048, 2047]
andi ra, sp, 2048 # CHECK: :[[@LINE]]:14: error: immediate must be an integer in the range [-2048, 2047]

## simm13_lsb0
beq t0, t1, -4098 # CHECK: :[[@LINE]]:13: error: immediate must be a multiple of 2 bytes in the range [-4096, 4094]
bne t0, t1, -4097 # CHECK: :[[@LINE]]:13: error: immediate must be a multiple of 2 bytes in the range [-4096, 4094]
blt t0, t1, 4095 # CHECK: :[[@LINE]]:13: error: immediate must be a multiple of 2 bytes in the range [-4096, 4094]
bge t0, t1, 4096 # CHECK: :[[@LINE]]:13: error: immediate must be a multiple of 2 bytes in the range [-4096, 4094]
bltu t0, t1, 13 # CHECK: :[[@LINE]]:14: error: immediate must be a multiple of 2 bytes in the range [-4096, 4094]
bgeu t0, t1, -13 # CHECK: :[[@LINE]]:14: error: immediate must be a multiple of 2 bytes in the range [-4096, 4094]

## uimm20
lui a0, -1 # CHECK: :[[@LINE]]:9: error: immediate must be an integer in the range [0, 1048575]
lui s0, 1048576 # CHECK: :[[@LINE]]:9: error: immediate must be an integer in the range [0, 1048575]
auipc zero, -0xf # CHECK: :[[@LINE]]:13: error: immediate must be an integer in the range [0, 1048575]

## simm21_lsb0
jal gp, -1048578 # CHECK: :[[@LINE]]:9: error: immediate must be a multiple of 2 bytes in the range [-1048576, 1048574]
jal gp, -1048577 # CHECK: :[[@LINE]]:9: error: immediate must be a multiple of 2 bytes in the range [-1048576, 1048574]
jal gp, 1048575 # CHECK: :[[@LINE]]:9: error: immediate must be a multiple of 2 bytes in the range [-1048576, 1048574]
jal gp, 1048576 # CHECK: :[[@LINE]]:9: error: immediate must be a multiple of 2 bytes in the range [-1048576, 1048574]
jal gp, 1 # CHECK: :[[@LINE]]:9: error: immediate must be a multiple of 2 bytes in the range [-1048576, 1048574]

# Invalid mnemonics
subs t0, t2, t1 # CHECK: :[[@LINE]]:1: error: unrecognized instruction mnemonic
nandi t0, zero, 0 # CHECK: :[[@LINE]]:1: error: unrecognized instruction mnemonic

# Invalid register names
addi foo, sp, 10 # CHECK: :[[@LINE]]:6: error: invalid operand for instruction
slti a10, a2, 0x20 # CHECK: :[[@LINE]]:6: error: invalid operand for instruction
slt x32, s0, s0 # CHECK: :[[@LINE]]:5: error: invalid operand for instruction

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
