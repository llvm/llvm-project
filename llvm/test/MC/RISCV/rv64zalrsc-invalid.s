# RUN: llvm-mc --verify -triple riscv64 -mattr=+a < %s
# RUN: llvm-mc --verify -triple riscv64 -mattr=+zalrsc < %s

# Final operand must have parentheses
lr.d a4, a5 # expected-error:10 {{expected '(' or optional integer offset}}

# lr only takes two operands
lr.d s0, (s1), s2 # expected-error:16 {{unexpected extra operand for instruction}}
