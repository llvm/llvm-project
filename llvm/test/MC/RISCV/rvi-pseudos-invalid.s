# RUN: llvm-mc --verify=expected,rv32 -triple=riscv32 < %s
# RUN: llvm-mc --verify=expected,rv64 -triple=riscv64 < %s

lga x1, 1234 # expected-error:9 {{operand must be a bare symbol name}}
lga x1, %pcrel_hi(1234) # expected-error:9 {{operand must be a bare symbol name}}
lga x1, %pcrel_lo(1234) # expected-error:9 {{operand must be a bare symbol name}}
lga x1, %pcrel_hi(foo) # expected-error:9 {{operand must be a bare symbol name}}
lga x1, %pcrel_lo(foo) # expected-error:9 {{operand must be a bare symbol name}}
lga x1, %hi(1234) # expected-error:9 {{operand must be a bare symbol name}}
lga x1, %lo(1234) # expected-error:9 {{operand must be a bare symbol name}}
lga x1, %hi(foo) # expected-error:9 {{operand must be a bare symbol name}}
lga x1, %lo(foo) # expected-error:9 {{operand must be a bare symbol name}}

sw a2, %hi(a_symbol), a3
# rv32-error@-1:8 {{operand must be a bare symbol name}}
# rv64-error@-2:8 {{operand must be a bare symbol name}}

sw a2, %lo(a_symbol), a3 # expected-error:8 {{operand must be a bare symbol name}}
sw a2, %lo(a_symbol)(a4), a3 # expected-error:27 {{expected '%' relocation specifier}}

# Too few operands must be rejected
sw a2, a_symbol
# expected-error@-1:16 {{too few operands for instruction}}

# Zero register as the destination/temporary for load/store pseudos is illegal
# since that would result in the auipc result being ignored.
lb x0, a_symbol # expected-error:4 {{register must be a GPR excluding zero (x0)}}
lbu x0, a_symbol # expected-error:5 {{register must be a GPR excluding zero (x0)}}
lh x0, a_symbol # expected-error:4 {{register must be a GPR excluding zero (x0)}}
lhu x0, a_symbol # expected-error:5 {{register must be a GPR excluding zero (x0)}}
lw x0, a_symbol # expected-error:4 {{register must be a GPR excluding zero (x0)}}
lwu x0, a_symbol
# rv32-error@-1:1 {{instruction requires the following: RV64I Base Instruction Set}}
# rv64-error@-2:5 {{register must be a GPR excluding zero (x0)}}
ld x0, a_symbol
# rv32-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# rv32-note@-2:1 {{instruction requires the following: RV64I Base Instruction Set}}
# rv32-note@-3:1 {{instruction requires the following: 'Zilsd' (Load/Store pair instructions)}}
# rv64-error@-4:4 {{register must be a GPR excluding zero (x0)}}

sb a2, a_symbol, x0 # expected-error:18 {{register must be a GPR excluding zero (x0)}}
sh a2, a_symbol, x0 # expected-error:18 {{register must be a GPR excluding zero (x0)}}
sw a2, a_symbol, x0 # expected-error:18 {{register must be a GPR excluding zero (x0)}}
sd a2, a_symbol, x0
# rv32-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# rv32-note@-2:1 {{instruction requires the following: RV64I Base Instruction Set}}
# rv32-note@-3:1 {{instruction requires the following: 'Zilsd' (Load/Store pair instructions)}}
# rv64-error@-4:18 {{register must be a GPR excluding zero (x0)}}
