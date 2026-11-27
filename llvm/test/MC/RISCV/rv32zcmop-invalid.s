# RUN: llvm-mc --verify -triple riscv32 -mattr=+zcmop < %s

c.mop.0 # expected-error:1 {{unrecognized instruction mnemonic, did you mean: c.mop.1, c.mop.3, c.mop.5, c.mop.7, c.mop.9?}}

c.mop.1 t0 # expected-error:9 {{unexpected extra operand for instruction}}

c.mop.1 0x0 # expected-error:9 {{unexpected extra operand for instruction}}
