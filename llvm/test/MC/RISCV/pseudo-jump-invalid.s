# RUN: llvm-mc --verify -triple riscv32 < %s

jump 1234, x31 # expected-error:6 {{operand must be a valid jump target}}
jump foo@plt, x31 # expected-error:9 {{unexpected token}}
jump %pcrel_lo(1234), x31 # expected-error:6 {{unknown token in expression}}
jump foo@xxx # expected-error:9 {{unexpected token}}
