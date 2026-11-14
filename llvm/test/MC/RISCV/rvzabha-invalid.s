# RUN: llvm-mc --verify -triple riscv32 -mattr=+a,+zabha < %s
# RUN: llvm-mc --verify -triple riscv64 -mattr=+a,+zabha < %s

# Final operand must have parentheses
amoswap.b a1, a2, a3 # expected-error:19 {{expected '(' or optional integer offset}}
amomin.b a1, a2, 1 # expected-error:20 {{expected '(' after optional integer offset}}
amomin.b a1, a2, 1(a3) # expected-error:18 {{optional integer offset must be 0}}

# Only .aq, .rl, and .aqrl suffixes are valid
amoxor.b.rlqa a2, a3, (a4) # expected-error:1 {{unrecognized instruction mnemonic, did you mean: amoxor.b.rl?}}
amoor.b.aq.rl a4, a5, (a6) # expected-error:1 {{unrecognized instruction mnemonic, did you mean: amoor.b.aqrl, amoxor.b.aqrl?}}
amoor.b. a4, a5, (a6) # expected-error:1 {{unrecognized instruction mnemonic, did you mean: amoor.b, amoor.b.aq, amoor.b.rl, amoxor.b?}}

# Non-zero offsets not supported for the third operand (rs1).
amocas.b a1, a3, 1(a5) # expected-error:18 {{optional integer offset must be 0}}
amocas.h a0, a2, 2(a5) # expected-error:18 {{optional integer offset must be 0}}
