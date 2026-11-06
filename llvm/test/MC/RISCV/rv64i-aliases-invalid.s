# UNSUPPORTED: target={{.*-windows.*}}
# RUN: llvm-mc --verify -triple=riscv64 -M no-aliases -o /dev/null < %s
# RUN: llvm-mc --verify -triple=riscv64 -o /dev/null < %s

li t5, 0x10000000000000000 # expected-error:8 {{unknown operand}}

li t4, foo
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:8 {{operand must be a symbol with %lo/%pcrel_lo/%tprel_lo specifier or an integer in the range [-2048, 2047]}}
# expected-note@-3:8 {{operand must be a constant 64-bit integer}}

la t5, 0x10000000000000000 # expected-error:8 {{unknown operand}}

la x1, %pcrel_hi(1234)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:8 {{operand must be a bare symbol name}}
# expected-note@-3:8 {{operand either must be a constant 64-bit integer or a bare symbol name}}

la x1, %pcrel_lo(1234)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:8 {{operand must be a bare symbol name}}
# expected-note@-3:8 {{operand either must be a constant 64-bit integer or a bare symbol name}}

la x1, %pcrel_hi(foo)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:8 {{operand must be a bare symbol name}}
# expected-note@-3:8 {{operand either must be a constant 64-bit integer or a bare symbol name}}

la x1, %pcrel_lo(foo)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:8 {{operand must be a bare symbol name}}
# expected-note@-3:8 {{operand either must be a constant 64-bit integer or a bare symbol name}}

la x1, %hi(1234)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:8 {{operand must be a bare symbol name}}
# expected-note@-3:8 {{operand either must be a constant 64-bit integer or a bare symbol name}}

la x1, %lo(1234)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:8 {{operand must be a bare symbol name}}
# expected-note@-3:8 {{operand either must be a constant 64-bit integer or a bare symbol name}}

la x1, %hi(foo)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:8 {{operand must be a bare symbol name}}
# expected-note@-3:8 {{operand either must be a constant 64-bit integer or a bare symbol name}}

la x1, %lo(foo)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:8 {{operand must be a bare symbol name}}
# expected-note@-3:8 {{operand either must be a constant 64-bit integer or a bare symbol name}}

la a1, foo+foo
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:8 {{operand must be a bare symbol name}}
# expected-note@-3:8 {{operand either must be a constant 64-bit integer or a bare symbol name}}

lla t5, 0x10000000000000000 # expected-error:9 {{unknown operand}}

lla x1, %pcrel_hi(1234)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:9 {{operand must be a bare symbol name}}
# expected-note@-3:9 {{operand either must be a constant 64-bit integer or a bare symbol name}}

lla x1, %pcrel_lo(1234)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:9 {{operand must be a bare symbol name}}
# expected-note@-3:9 {{operand either must be a constant 64-bit integer or a bare symbol name}}

lla x1, %pcrel_hi(foo)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:9 {{operand must be a bare symbol name}}
# expected-note@-3:9 {{operand either must be a constant 64-bit integer or a bare symbol name}}

lla x1, %pcrel_lo(foo)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:9 {{operand must be a bare symbol name}}
# expected-note@-3:9 {{operand either must be a constant 64-bit integer or a bare symbol name}}

lla x1, %hi(1234)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:9 {{operand must be a bare symbol name}}
# expected-note@-3:9 {{operand either must be a constant 64-bit integer or a bare symbol name}}

lla x1, %lo(1234)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:9 {{operand must be a bare symbol name}}
# expected-note@-3:9 {{operand either must be a constant 64-bit integer or a bare symbol name}}

lla x1, %hi(foo)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:9 {{operand must be a bare symbol name}}
# expected-note@-3:9 {{operand either must be a constant 64-bit integer or a bare symbol name}}

lla x1, %lo(foo)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:9 {{operand must be a bare symbol name}}
# expected-note@-3:9 {{operand either must be a constant 64-bit integer or a bare symbol name}}

lla a1, foo+foo
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:9 {{operand must be a bare symbol name}}
# expected-note@-3:9 {{operand either must be a constant 64-bit integer or a bare symbol name}}

lla a2, foo@plt # expected-error:12 {{unexpected token}}

rdinstreth x29 # expected-error:1 {{instruction requires the following: RV32I Base Instruction Set}}

rdcycleh x27   # expected-error:1 {{instruction requires the following: RV32I Base Instruction Set}}

rdtimeh x28    # expected-error:1 {{instruction requires the following: RV32I Base Instruction Set}}

sll x2, x3, 64
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:13 {{register must be a GPR}}
# expected-note@-3:13 {{immediate must be an integer in the range [0, 63]}}

srl x2, x3, 64
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:13 {{register must be a GPR}}
# expected-note@-3:13 {{immediate must be an integer in the range [0, 63]}}

sra x2, x3, 64
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:13 {{register must be a GPR}}
# expected-note@-3:13 {{immediate must be an integer in the range [0, 63]}}

sll x2, x3, -1
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:13 {{register must be a GPR}}
# expected-note@-3:13 {{immediate must be an integer in the range [0, 63]}}

srl x2, x3, -2
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:13 {{register must be a GPR}}
# expected-note@-3:13 {{immediate must be an integer in the range [0, 63]}}

sra x2, x3, -3
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:13 {{register must be a GPR}}
# expected-note@-3:13 {{immediate must be an integer in the range [0, 63]}}

sllw x2, x3, 32
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:14 {{register must be a GPR}}
# expected-note@-3:14 {{immediate must be an integer in the range [0, 31]}}

srlw x2, x3, 32
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:14 {{register must be a GPR}}
# expected-note@-3:14 {{immediate must be an integer in the range [0, 31]}}

sraw x2, x3, 32
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:14 {{register must be a GPR}}
# expected-note@-3:14 {{immediate must be an integer in the range [0, 31]}}

sllw x2, x3, -1
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:14 {{register must be a GPR}}
# expected-note@-3:14 {{immediate must be an integer in the range [0, 31]}}

srlw x2, x3, -2
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:14 {{register must be a GPR}}
# expected-note@-3:14 {{immediate must be an integer in the range [0, 31]}}

sraw x2, x3, -3
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:14 {{register must be a GPR}}
# expected-note@-3:14 {{immediate must be an integer in the range [0, 31]}}

foo:
  .space 8
