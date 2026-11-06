# RUN: llvm-mc --verify -triple riscv32 -mattr=+a < %s
# RUN: llvm-mc --verify -triple riscv32 -mattr=+zaamo < %s

# Final operand must have parentheses
amoswap.w a1, a2, a3 # expected-error:19 {{expected '(' or optional integer offset}}
amomin.w a1, a2, 1 # expected-error:20 {{expected '(' after optional integer offset}}
amomin.w a1, a2, 1(a3) # expected-error:18 {{optional integer offset must be 0}}

# Only .aq, .rl, and .aqrl suffixes are valid
amoxor.w.rlqa a2, a3, (a4) # expected-error:1 {{unrecognized instruction mnemonic}}
amoor.w.aq.rl a4, a5, (a6) # expected-error:1 {{unrecognized instruction mnemonic}}
amoor.w. a4, a5, (a6) # expected-error:1 {{unrecognized instruction mnemonic}}
