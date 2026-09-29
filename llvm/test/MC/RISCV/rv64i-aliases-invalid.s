# UNSUPPORTED: target={{.*-windows.*}}
# RUN: not llvm-mc -triple=riscv64 -M no-aliases < %s -o /dev/null 2>&1 | FileCheck %s
# RUN: not llvm-mc -triple=riscv64 < %s 2>&1 -o /dev/null | FileCheck %s

li t5, 0x10000000000000000 # CHECK: :[[@LINE]]:8: error: unknown operand

li t4, foo
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:8: note: operand must be a symbol with %lo/%pcrel_lo/%tprel_lo specifier or an integer in the range [-2048, 2047]
# CHECK: :[[@LINE-3]]:8: note: operand must be a constant 64-bit integer

la t5, 0x10000000000000000 # CHECK: :[[@LINE]]:8: error: unknown operand

la x1, %pcrel_hi(1234)
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:8: note: operand must be a bare symbol name
# CHECK: :[[@LINE-3]]:8: note: operand either must be a constant 64-bit integer or a bare symbol name

la x1, %pcrel_lo(1234)
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:8: note: operand must be a bare symbol name
# CHECK: :[[@LINE-3]]:8: note: operand either must be a constant 64-bit integer or a bare symbol name

la x1, %pcrel_hi(foo)
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:8: note: operand must be a bare symbol name
# CHECK: :[[@LINE-3]]:8: note: operand either must be a constant 64-bit integer or a bare symbol name

la x1, %pcrel_lo(foo)
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:8: note: operand must be a bare symbol name
# CHECK: :[[@LINE-3]]:8: note: operand either must be a constant 64-bit integer or a bare symbol name

la x1, %hi(1234)
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:8: note: operand must be a bare symbol name
# CHECK: :[[@LINE-3]]:8: note: operand either must be a constant 64-bit integer or a bare symbol name

la x1, %lo(1234)
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:8: note: operand must be a bare symbol name
# CHECK: :[[@LINE-3]]:8: note: operand either must be a constant 64-bit integer or a bare symbol name

la x1, %hi(foo)
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:8: note: operand must be a bare symbol name
# CHECK: :[[@LINE-3]]:8: note: operand either must be a constant 64-bit integer or a bare symbol name

la x1, %lo(foo)
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:8: note: operand must be a bare symbol name
# CHECK: :[[@LINE-3]]:8: note: operand either must be a constant 64-bit integer or a bare symbol name

la a1, foo+foo
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:8: note: operand must be a bare symbol name
# CHECK: :[[@LINE-3]]:8: note: operand either must be a constant 64-bit integer or a bare symbol name

lla t5, 0x10000000000000000 # CHECK: :[[@LINE]]:9: error: unknown operand

lla x1, %pcrel_hi(1234)
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:9: note: operand must be a bare symbol name
# CHECK: :[[@LINE-3]]:9: note: operand either must be a constant 64-bit integer or a bare symbol name

lla x1, %pcrel_lo(1234)
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:9: note: operand must be a bare symbol name
# CHECK: :[[@LINE-3]]:9: note: operand either must be a constant 64-bit integer or a bare symbol name

lla x1, %pcrel_hi(foo)
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:9: note: operand must be a bare symbol name
# CHECK: :[[@LINE-3]]:9: note: operand either must be a constant 64-bit integer or a bare symbol name

lla x1, %pcrel_lo(foo)
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:9: note: operand must be a bare symbol name
# CHECK: :[[@LINE-3]]:9: note: operand either must be a constant 64-bit integer or a bare symbol name

lla x1, %hi(1234)
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:9: note: operand must be a bare symbol name
# CHECK: :[[@LINE-3]]:9: note: operand either must be a constant 64-bit integer or a bare symbol name

lla x1, %lo(1234)
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:9: note: operand must be a bare symbol name
# CHECK: :[[@LINE-3]]:9: note: operand either must be a constant 64-bit integer or a bare symbol name

lla x1, %hi(foo)
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:9: note: operand must be a bare symbol name
# CHECK: :[[@LINE-3]]:9: note: operand either must be a constant 64-bit integer or a bare symbol name

lla x1, %lo(foo)
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:9: note: operand must be a bare symbol name
# CHECK: :[[@LINE-3]]:9: note: operand either must be a constant 64-bit integer or a bare symbol name

lla a1, foo+foo
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:9: note: operand must be a bare symbol name
# CHECK: :[[@LINE-3]]:9: note: operand either must be a constant 64-bit integer or a bare symbol name

lla a2, foo@plt # CHECK: :[[@LINE]]:12: error: unexpected token

rdinstreth x29 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV32I Base Instruction Set{{$}}

rdcycleh x27   # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV32I Base Instruction Set{{$}}

rdtimeh x28    # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV32I Base Instruction Set{{$}}

sll x2, x3, 64
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:13: note: register must be a GPR
# CHECK: :[[@LINE-3]]:13: note: immediate must be an integer in the range [0, 63]

srl x2, x3, 64
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:13: note: register must be a GPR
# CHECK: :[[@LINE-3]]:13: note: immediate must be an integer in the range [0, 63]

sra x2, x3, 64
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:13: note: register must be a GPR
# CHECK: :[[@LINE-3]]:13: note: immediate must be an integer in the range [0, 63]

sll x2, x3, -1
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:13: note: register must be a GPR
# CHECK: :[[@LINE-3]]:13: note: immediate must be an integer in the range [0, 63]

srl x2, x3, -2
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:13: note: register must be a GPR
# CHECK: :[[@LINE-3]]:13: note: immediate must be an integer in the range [0, 63]

sra x2, x3, -3
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:13: note: register must be a GPR
# CHECK: :[[@LINE-3]]:13: note: immediate must be an integer in the range [0, 63]

sllw x2, x3, 32
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:14: note: register must be a GPR
# CHECK: :[[@LINE-3]]:14: note: immediate must be an integer in the range [0, 31]

srlw x2, x3, 32
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:14: note: register must be a GPR
# CHECK: :[[@LINE-3]]:14: note: immediate must be an integer in the range [0, 31]

sraw x2, x3, 32
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:14: note: register must be a GPR
# CHECK: :[[@LINE-3]]:14: note: immediate must be an integer in the range [0, 31]

sllw x2, x3, -1
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:14: note: register must be a GPR
# CHECK: :[[@LINE-3]]:14: note: immediate must be an integer in the range [0, 31]

srlw x2, x3, -2
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:14: note: register must be a GPR
# CHECK: :[[@LINE-3]]:14: note: immediate must be an integer in the range [0, 31]

sraw x2, x3, -3
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:14: note: register must be a GPR
# CHECK: :[[@LINE-3]]:14: note: immediate must be an integer in the range [0, 31]

foo:
  .space 8
