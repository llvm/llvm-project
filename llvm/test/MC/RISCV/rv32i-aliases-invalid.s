# UNSUPPORTED: target={{.*-windows.*}}
# RUN: not llvm-mc -triple=riscv32 -M no-aliases < %s -o /dev/null 2>&1 | FileCheck %s
# RUN: not llvm-mc -triple=riscv32 < %s -o /dev/null 2>&1 | FileCheck %s

# TODO ld
# TODO sd

li x0, 4294967296
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:8: note: operand must be a symbol with %lo/%pcrel_lo/%tprel_lo specifier or an integer in the range [-2048, 2047]
# CHECK: :[[@LINE-3]]:8: note: immediate must be an integer in the range [-2147483648, 4294967295]

li x0, -2147483649
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:8: note: operand must be a symbol with %lo/%pcrel_lo/%tprel_lo specifier or an integer in the range [-2048, 2047]
# CHECK: :[[@LINE-3]]:8: note: immediate must be an integer in the range [-2147483648, 4294967295]

li t4, foo
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:8: note: operand must be a symbol with %lo/%pcrel_lo/%tprel_lo specifier or an integer in the range [-2048, 2047]
# CHECK: :[[@LINE-3]]:8: note: immediate must be an integer in the range [-2147483648, 4294967295]

la x0, 4294967296
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:8: note: operand must be a bare symbol name
# CHECK: :[[@LINE-3]]:8: note: operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]

la x0, -2147483649
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:8: note: operand must be a bare symbol name
# CHECK: :[[@LINE-3]]:8: note: operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]

la a1, foo+foo
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:8: note: operand must be a bare symbol name
# CHECK: :[[@LINE-3]]:8: note: operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]

la x1, %pcrel_hi(1234)
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:8: note: operand must be a bare symbol name
# CHECK: :[[@LINE-3]]:8: note: operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]

la x1, %pcrel_lo(1234)
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:8: note: operand must be a bare symbol name
# CHECK: :[[@LINE-3]]:8: note: operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]

la x1, %pcrel_hi(foo)
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:8: note: operand must be a bare symbol name
# CHECK: :[[@LINE-3]]:8: note: operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]

la x1, %pcrel_lo(foo)
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:8: note: operand must be a bare symbol name
# CHECK: :[[@LINE-3]]:8: note: operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]

la x1, %hi(1234)
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:8: note: operand must be a bare symbol name
# CHECK: :[[@LINE-3]]:8: note: operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]

la x1, %lo(1234)
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:8: note: operand must be a bare symbol name
# CHECK: :[[@LINE-3]]:8: note: operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]

la x1, %hi(foo)
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:8: note: operand must be a bare symbol name
# CHECK: :[[@LINE-3]]:8: note: operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]

la x1, %lo(foo)
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:8: note: operand must be a bare symbol name
# CHECK: :[[@LINE-3]]:8: note: operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]

lla x0, 4294967296
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:9: note: operand must be a bare symbol name
# CHECK: :[[@LINE-3]]:9: note: operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]

lla x0, -2147483649
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:9: note: operand must be a bare symbol name
# CHECK: :[[@LINE-3]]:9: note: operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]

lla a1, foo+foo
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:9: note: operand must be a bare symbol name
# CHECK: :[[@LINE-3]]:9: note: operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]

lla x1, %pcrel_hi(1234)
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:9: note: operand must be a bare symbol name
# CHECK: :[[@LINE-3]]:9: note: operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]

lla x1, %pcrel_lo(1234)
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:9: note: operand must be a bare symbol name
# CHECK: :[[@LINE-3]]:9: note: operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]

lla x1, %pcrel_hi(foo)
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:9: note: operand must be a bare symbol name
# CHECK: :[[@LINE-3]]:9: note: operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]

lla x1, %pcrel_lo(foo)
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:9: note: operand must be a bare symbol name
# CHECK: :[[@LINE-3]]:9: note: operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]

lla x1, %hi(1234)
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:9: note: operand must be a bare symbol name
# CHECK: :[[@LINE-3]]:9: note: operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]

lla x1, %lo(1234)
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:9: note: operand must be a bare symbol name
# CHECK: :[[@LINE-3]]:9: note: operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]

lla x1, %hi(foo)
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:9: note: operand must be a bare symbol name
# CHECK: :[[@LINE-3]]:9: note: operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]

lla x1, %lo(foo)
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:9: note: operand must be a bare symbol name
# CHECK: :[[@LINE-3]]:9: note: operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]

lla a2, foo@plt # CHECK: :[[@LINE]]:12: error: unexpected token

negw x1, x2   # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set{{$}}
sext.w x3, x4 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set{{$}}

zext.w x3, x4
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:1: note: instruction requires the following: 'Zba' (Address Generation Instructions), RV64I Base Instruction Set
# CHECK: :[[@LINE-3]]:1: note: instruction requires the following: RV64I Base Instruction Set{{$}}

sll x2, x3, 32
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:13: note: invalid operand for instruction
# CHECK: :[[@LINE-3]]:13: note: immediate must be an integer in the range [0, 31]

srl x2, x3, 32
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:13: note: invalid operand for instruction
# CHECK: :[[@LINE-3]]:13: note: immediate must be an integer in the range [0, 31]

sra x2, x3, 32
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:13: note: invalid operand for instruction
# CHECK: :[[@LINE-3]]:13: note: immediate must be an integer in the range [0, 31]

sll x2, x3, -1
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:13: note: invalid operand for instruction
# CHECK: :[[@LINE-3]]:13: note: immediate must be an integer in the range [0, 31]

srl x2, x3, -2
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:13: note: invalid operand for instruction
# CHECK: :[[@LINE-3]]:13: note: immediate must be an integer in the range [0, 31]

sra x2, x3, -3
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:13: note: invalid operand for instruction
# CHECK: :[[@LINE-3]]:13: note: immediate must be an integer in the range [0, 31]

addi x1, .      # CHECK: :[[@LINE]]:10: error: invalid operand for instruction

foo:
  .space 4
