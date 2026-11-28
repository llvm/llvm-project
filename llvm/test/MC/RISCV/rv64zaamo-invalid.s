# RUN: llvm-mc --verify -triple riscv64 -mattr=+a < %s
# RUN: llvm-mc --verify -triple riscv64 -mattr=+zaamo < %s

# Final operand must have parentheses
amoswap.d a1, a2, a3 # expected-error:19 {{expected '(' or optional integer offset}}
amomin.d a1, a2, 1 # expected-error:20 {{expected '(' after optional integer offset}}
amomin.d a1, a2, 1(a3) # expected-error:18 {{optional integer offset must be 0}}

# Only .aq, .rl, and .aqrl suffixes are valid
amoxor.d.rlqa a2, a3, (a4) # expected-error:1 {{unrecognized instruction mnemonic}}
amoor.d.aq.rl a4, a5, (a6) # expected-error:1 {{unrecognized instruction mnemonic}}
amoor.d. a4, a5, (a6) # expected-error:1 {{unrecognized instruction mnemonic}}
