# UNSUPPORTED: target={{.*-windows.*}}
# RUN: llvm-mc --verify -triple=riscv32 -M no-aliases -o /dev/null < %s
# RUN: llvm-mc --verify -triple=riscv32 -o /dev/null < %s

# TODO ld
# TODO sd

li x0, 4294967296
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:8 {{operand must be a symbol with %lo/%pcrel_lo/%tprel_lo specifier or an integer in the range [-2048, 2047]}}
# expected-note@-3:8 {{immediate must be an integer in the range [-2147483648, 4294967295]}}

li x0, -2147483649
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:8 {{operand must be a symbol with %lo/%pcrel_lo/%tprel_lo specifier or an integer in the range [-2048, 2047]}}
# expected-note@-3:8 {{immediate must be an integer in the range [-2147483648, 4294967295]}}

li t4, foo
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:8 {{operand must be a symbol with %lo/%pcrel_lo/%tprel_lo specifier or an integer in the range [-2048, 2047]}}
# expected-note@-3:8 {{immediate must be an integer in the range [-2147483648, 4294967295]}}

la x0, 4294967296
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:8 {{operand must be a bare symbol name}}
# expected-note@-3:8 {{operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]}}

la x0, -2147483649
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:8 {{operand must be a bare symbol name}}
# expected-note@-3:8 {{operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]}}

la a1, foo+foo
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:8 {{operand must be a bare symbol name}}
# expected-note@-3:8 {{operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]}}

la x1, %pcrel_hi(1234)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:8 {{operand must be a bare symbol name}}
# expected-note@-3:8 {{operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]}}

la x1, %pcrel_lo(1234)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:8 {{operand must be a bare symbol name}}
# expected-note@-3:8 {{operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]}}

la x1, %pcrel_hi(foo)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:8 {{operand must be a bare symbol name}}
# expected-note@-3:8 {{operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]}}

la x1, %pcrel_lo(foo)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:8 {{operand must be a bare symbol name}}
# expected-note@-3:8 {{operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]}}

la x1, %hi(1234)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:8 {{operand must be a bare symbol name}}
# expected-note@-3:8 {{operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]}}

la x1, %lo(1234)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:8 {{operand must be a bare symbol name}}
# expected-note@-3:8 {{operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]}}

la x1, %hi(foo)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:8 {{operand must be a bare symbol name}}
# expected-note@-3:8 {{operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]}}

la x1, %lo(foo)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:8 {{operand must be a bare symbol name}}
# expected-note@-3:8 {{operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]}}

lla x0, 4294967296
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:9 {{operand must be a bare symbol name}}
# expected-note@-3:9 {{operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]}}

lla x0, -2147483649
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:9 {{operand must be a bare symbol name}}
# expected-note@-3:9 {{operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]}}

lla a1, foo+foo
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:9 {{operand must be a bare symbol name}}
# expected-note@-3:9 {{operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]}}

lla x1, %pcrel_hi(1234)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:9 {{operand must be a bare symbol name}}
# expected-note@-3:9 {{operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]}}

lla x1, %pcrel_lo(1234)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:9 {{operand must be a bare symbol name}}
# expected-note@-3:9 {{operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]}}

lla x1, %pcrel_hi(foo)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:9 {{operand must be a bare symbol name}}
# expected-note@-3:9 {{operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]}}

lla x1, %pcrel_lo(foo)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:9 {{operand must be a bare symbol name}}
# expected-note@-3:9 {{operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]}}

lla x1, %hi(1234)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:9 {{operand must be a bare symbol name}}
# expected-note@-3:9 {{operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]}}

lla x1, %lo(1234)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:9 {{operand must be a bare symbol name}}
# expected-note@-3:9 {{operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]}}

lla x1, %hi(foo)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:9 {{operand must be a bare symbol name}}
# expected-note@-3:9 {{operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]}}

lla x1, %lo(foo)
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:9 {{operand must be a bare symbol name}}
# expected-note@-3:9 {{operand either must be a bare symbol name or an immediate integer in the range [-2147483648, 4294967295]}}

lla a2, foo@plt # expected-error:12 {{unexpected token}}

negw x1, x2   # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
sext.w x3, x4 # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}

zext.w x3, x4
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:1 {{instruction requires the following: 'Zba' (Address Generation Instructions), RV64I Base Instruction Set}}
# expected-note@-3:1 {{instruction requires the following: RV64I Base Instruction Set}}

sll x2, x3, 32
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:13 {{register must be a GPR}}
# expected-note@-3:13 {{immediate must be an integer in the range [0, 31]}}

srl x2, x3, 32
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:13 {{register must be a GPR}}
# expected-note@-3:13 {{immediate must be an integer in the range [0, 31]}}

sra x2, x3, 32
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:13 {{register must be a GPR}}
# expected-note@-3:13 {{immediate must be an integer in the range [0, 31]}}

sll x2, x3, -1
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:13 {{register must be a GPR}}
# expected-note@-3:13 {{immediate must be an integer in the range [0, 31]}}

srl x2, x3, -2
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:13 {{register must be a GPR}}
# expected-note@-3:13 {{immediate must be an integer in the range [0, 31]}}

sra x2, x3, -3
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:13 {{register must be a GPR}}
# expected-note@-3:13 {{immediate must be an integer in the range [0, 31]}}

addi x1, .      # expected-error:10 {{register must be a GPR}}

foo:
  .space 4
