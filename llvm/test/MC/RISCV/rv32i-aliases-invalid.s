# UNSUPPORTED: target={{.*-windows.*}}
# RUN: not llvm-mc -triple=riscv32 -M no-aliases < %s -o /dev/null 2>&1 | FileCheck %s
# RUN: not llvm-mc -triple=riscv32 < %s -o /dev/null 2>&1 | FileCheck %s

# TODO ld
# TODO sd

li x0, 4294967296   # CHECK: :[[@LINE]]:8: error: immediate must be an integer in the range [-2147483648, 4294967295]
li x0, -2147483649  # CHECK: :[[@LINE]]:8: error: immediate must be an integer in the range [-2147483648, 4294967295]
li t4, foo          # CHECK: :[[@LINE]]:8: error: immediate must be an integer in the range [-2147483648, 4294967295]

la x0, 4294967296   # CHECK: :[[@LINE]]:8: error: operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]
la x0, -2147483649  # CHECK: :[[@LINE]]:8: error: operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]
la a1, foo+foo # CHECK: :[[@LINE]]:8: error: operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]
la x1, %pcrel_hi(1234) # CHECK: :[[@LINE]]:8: error: operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]
la x1, %pcrel_lo(1234) # CHECK: :[[@LINE]]:8: error: operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]
la x1, %pcrel_hi(foo) # CHECK: :[[@LINE]]:8: error: operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]
la x1, %pcrel_lo(foo) # CHECK: :[[@LINE]]:8: error: operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]
la x1, %hi(1234) # CHECK: :[[@LINE]]:8: error: operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]
la x1, %lo(1234) # CHECK: :[[@LINE]]:8: error: operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]
la x1, %hi(foo) # CHECK: :[[@LINE]]:8: error: operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]
la x1, %lo(foo) # CHECK: :[[@LINE]]:8: error: operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]

lla x0, 4294967296   # CHECK: :[[@LINE]]:9: error: operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]
lla x0, -2147483649  # CHECK: :[[@LINE]]:9: error: operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]
lla a1, foo+foo # CHECK: :[[@LINE]]:9: error: operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]
lla x1, %pcrel_hi(1234) # CHECK: :[[@LINE]]:9: error: operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]
lla x1, %pcrel_lo(1234) # CHECK: :[[@LINE]]:9: error: operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]
lla x1, %pcrel_hi(foo) # CHECK: :[[@LINE]]:9: error: operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]
lla x1, %pcrel_lo(foo) # CHECK: :[[@LINE]]:9: error: operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]
lla x1, %hi(1234) # CHECK: :[[@LINE]]:9: error: operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]
lla x1, %lo(1234) # CHECK: :[[@LINE]]:9: error: operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]
lla x1, %hi(foo) # CHECK: :[[@LINE]]:9: error: operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]
lla x1, %lo(foo) # CHECK: :[[@LINE]]:9: error: operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]
lla a2, foo@plt # CHECK: :[[@LINE]]:17: error: '@plt' operand not valid for instruction

negw x1, x2   # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set{{$}}
sext.w x3, x4 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set{{$}}
zext.w x3, x4 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set{{$}}

sll x2, x3, 32  # CHECK: :[[@LINE]]:13: error: immediate must be an integer in the range [0, 31]
srl x2, x3, 32  # CHECK: :[[@LINE]]:13: error: immediate must be an integer in the range [0, 31]
sra x2, x3, 32  # CHECK: :[[@LINE]]:13: error: immediate must be an integer in the range [0, 31]

sll x2, x3, -1  # CHECK: :[[@LINE]]:13: error: immediate must be an integer in the range [0, 31]
srl x2, x3, -2  # CHECK: :[[@LINE]]:13: error: immediate must be an integer in the range [0, 31]
sra x2, x3, -3  # CHECK: :[[@LINE]]:13: error: immediate must be an integer in the range [0, 31]

addi x1, .      # CHECK: :[[@LINE]]:10: error: invalid operand for instruction

foo:
  .space 4
