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

## simm12
ori a0, a1, -2049 # CHECK: :[[@LINE]]:13: error: immediate must be an integer in the range [-2048, 2047]
andi ra, sp, 2048 # CHECK: :[[@LINE]]:14: error: immediate must be an integer in the range [-2048, 2047]

## uimm12
csrrw a0, -1, a0 # CHECK: :[[@LINE]]:11: error: immediate must be an integer in the range [0, 4095]
csrrs a0, 4096, a0 # CHECK: :[[@LINE]]:11: error: immediate must be an integer in the range [0, 4095]
csrrs a0, -0xf, a0 # CHECK: :[[@LINE]]:11: error: immediate must be an integer in the range [0, 4095]
csrrc a0, 0x1000, a0 # CHECK: :[[@LINE]]:11: error: immediate must be an integer in the range [0, 4095]
csrrwi a0, -50, 0 # CHECK: :[[@LINE]]:12: error: immediate must be an integer in the range [0, 4095]
csrrsi a0, 4097, a0 # CHECK: :[[@LINE]]:12: error: immediate must be an integer in the range [0, 4095]
csrrci a0, 0xffff, a0 # CHECK: :[[@LINE]]:12: error: immediate must be an integer in the range [0, 4095]

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

# Illegal operand modifier
## fencearg
fence %hi(iorw), iorw # CHECK: :[[@LINE]]:7: error: operand must be formed of letters selected in-order from 'iorw'
fence %lo(iorw), iorw # CHECK: :[[@LINE]]:7: error: operand must be formed of letters selected in-order from 'iorw'
fence %pcrel_hi(iorw), iorw # CHECK: :[[@LINE]]:7: error: operand must be formed of letters selected in-order from 'iorw'
fence %pcrel_lo(iorw), iorw # CHECK: :[[@LINE]]:7: error: operand must be formed of letters selected in-order from 'iorw'

## uimm5
slli a0, a0, %lo(1) # CHECK: :[[@LINE]]:14: error: immediate must be an integer in the range [0, 31]
srli a0, a0, %lo(a) # CHECK: :[[@LINE]]:14: error: immediate must be an integer in the range [0, 31]
srai a0, a0, %hi(2) # CHECK: :[[@LINE]]:14: error: immediate must be an integer in the range [0, 31]
csrrwi a1, 0x1, %hi(b) # CHECK: :[[@LINE]]:17: error: immediate must be an integer in the range [0, 31]
csrrsi t1, 999, %pcrel_hi(3) # CHECK: :[[@LINE]]:17: error: immediate must be an integer in the range [0, 31]
csrrci x0, 43, %pcrel_hi(c) # CHECK: :[[@LINE]]:16: error: immediate must be an integer in the range [0, 31]
csrrsi t1, 999, %pcrel_lo(4) # CHECK: :[[@LINE]]:17: error: immediate must be an integer in the range [0, 31]
csrrci x0, 43, %pcrel_lo(d) # CHECK: :[[@LINE]]:16: error: immediate must be an integer in the range [0, 31]

## simm12
ori a0, a1, %hi(foo) # CHECK: :[[@LINE]]:13: error: immediate must be an integer in the range [-2048, 2047]
andi ra, sp, %pcrel_hi(123) # CHECK: :[[@LINE]]:14: error: immediate must be an integer in the range [-2048, 2047]
xori a2, a3, %hi(345) # CHECK: :[[@LINE]]:14: error: immediate must be an integer in the range [-2048, 2047]

## uimm12
csrrw a0, %lo(1), a0 # CHECK: :[[@LINE]]:11: error: immediate must be an integer in the range [0, 4095]
csrrs a0, %lo(a), a0 # CHECK: :[[@LINE]]:11: error: immediate must be an integer in the range [0, 4095]
csrrs a0, %hi(2), a0 # CHECK: :[[@LINE]]:11: error: immediate must be an integer in the range [0, 4095]
csrrc a0, %hi(b), a0 # CHECK: :[[@LINE]]:11: error: immediate must be an integer in the range [0, 4095]
csrrwi a0, %pcrel_hi(3), 0 # CHECK: :[[@LINE]]:12: error: immediate must be an integer in the range [0, 4095]
csrrsi a0, %pcrel_hi(c), a0 # CHECK: :[[@LINE]]:12: error: immediate must be an integer in the range [0, 4095]
csrrwi a0, %pcrel_lo(4), 0 # CHECK: :[[@LINE]]:12: error: immediate must be an integer in the range [0, 4095]
csrrsi a0, %pcrel_lo(d), a0 # CHECK: :[[@LINE]]:12: error: immediate must be an integer in the range [0, 4095]

## simm13_lsb0
beq t0, t1, %lo(1) # CHECK: :[[@LINE]]:13: error: immediate must be a multiple of 2 bytes in the range [-4096, 4094]
bne t0, t1, %lo(a) # CHECK: :[[@LINE]]:13: error: immediate must be a multiple of 2 bytes in the range [-4096, 4094]
blt t0, t1, %hi(2) # CHECK: :[[@LINE]]:13: error: immediate must be a multiple of 2 bytes in the range [-4096, 4094]
bge t0, t1, %hi(b) # CHECK: :[[@LINE]]:13: error: immediate must be a multiple of 2 bytes in the range [-4096, 4094]
bltu t0, t1, %pcrel_hi(3) # CHECK: :[[@LINE]]:14: error: immediate must be a multiple of 2 bytes in the range [-4096, 4094]
bgeu t0, t1, %pcrel_hi(c) # CHECK: :[[@LINE]]:14: error: immediate must be a multiple of 2 bytes in the range [-4096, 4094]
bltu t0, t1, %pcrel_lo(4) # CHECK: :[[@LINE]]:14: error: immediate must be a multiple of 2 bytes in the range [-4096, 4094]
bgeu t0, t1, %pcrel_lo(d) # CHECK: :[[@LINE]]:14: error: immediate must be a multiple of 2 bytes in the range [-4096, 4094]

## uimm20
lui a0, %lo(1) # CHECK: :[[@LINE]]:9: error: immediate must be an integer in the range [0, 1048575]
auipc a1, %lo(foo) # CHECK: :[[@LINE]]:11: error: immediate must be an integer in the range [0, 1048575]

## simm21_lsb0
jal gp, %lo(1) # CHECK: :[[@LINE]]:9: error: immediate must be a multiple of 2 bytes in the range [-1048576, 1048574]
jal gp, %lo(a) # CHECK: :[[@LINE]]:9: error: immediate must be a multiple of 2 bytes in the range [-1048576, 1048574]
jal gp, %hi(2) # CHECK: :[[@LINE]]:9: error: immediate must be a multiple of 2 bytes in the range [-1048576, 1048574]
jal gp, %hi(b) # CHECK: :[[@LINE]]:9: error: immediate must be a multiple of 2 bytes in the range [-1048576, 1048574]
jal gp, %pcrel_hi(3) # CHECK: :[[@LINE]]:9: error: immediate must be a multiple of 2 bytes in the range [-1048576, 1048574]
jal gp, %pcrel_hi(c) # CHECK: :[[@LINE]]:9: error: immediate must be a multiple of 2 bytes in the range [-1048576, 1048574]
jal gp, %pcrel_lo(4) # CHECK: :[[@LINE]]:9: error: immediate must be a multiple of 2 bytes in the range [-1048576, 1048574]
jal gp, %pcrel_lo(d) # CHECK: :[[@LINE]]:9: error: immediate must be a multiple of 2 bytes in the range [-1048576, 1048574]

# Unrecognized operand modifier
addi t0, sp, %modifer(255) # CHECK: :[[@LINE]]:15: error: unrecognized operand modifier

# Use of operand modifier on register name
addi t1, %lo(t2), 1 # CHECK: :[[@LINE]]:10: error: invalid operand for instruction

# Invalid mnemonics
subs t0, t2, t1 # CHECK: :[[@LINE]]:1: error: unrecognized instruction mnemonic
nandi t0, zero, 0 # CHECK: :[[@LINE]]:1: error: unrecognized instruction mnemonic

# Invalid register names
addi foo, sp, 10 # CHECK: :[[@LINE]]:6: error: invalid operand for instruction
slti a10, a2, 0x20 # CHECK: :[[@LINE]]:6: error: invalid operand for instruction
slt x32, s0, s0 # CHECK: :[[@LINE]]:5: error: invalid operand for instruction

# RV64I mnemonics
addiw a0, sp, 100 # CHECK: :[[@LINE]]:1: error: instruction use requires an option to be enabled
sraw t0, s2, zero # CHECK: :[[@LINE]]:1: error: instruction use requires an option to be enabled

# Invalid operand types
xori sp, 22, 220 # CHECK: :[[@LINE]]:10: error: invalid operand for instruction
sub t0, t2, 1 # CHECK: :[[@LINE]]:13: error: invalid operand for instruction
add a1, a2, (a3) # CHECK: :[[@LINE]]:13: error: invalid operand for instruction

# Too many operands
add ra, zero, zero, zero # CHECK: :[[@LINE]]:21: error: invalid operand for instruction
sltiu s2, s3, 0x50, 0x60 # CHECK: :[[@LINE]]:21: error: invalid operand for instruction

# Memory operand not formatted correctly
lw a4, a5, 111 # CHECK: :[[@LINE]]:8: error: immediate must be an integer in the range [-2048, 2047]

# Too few operands
ori a0, a1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
xor s2, s2 # CHECK: :[[@LINE]]:1: error: too few operands for instruction

# Instruction not in the base ISA
mul a4, ra, s0 # CHECK: :[[@LINE]]:1: error: instruction use requires an option to be enabled
amomaxu.w s5, s4, (s3) # CHECK: :[[@LINE]]:1: error: instruction use requires an option to be enabled
fadd.s ft0, ft1, ft2 # CHECK: :[[@LINE]]:1: error: instruction use requires an option to be enabled

# Using floating point registers when integer registers are expected
addi a2, ft0, 24 # CHECK: :[[@LINE]]:10: error: invalid operand for instruction

# fence.tso accepts no operands
fence.tso rw, rw # CHECK: :[[@LINE]]:11: error: invalid operand for instruction
