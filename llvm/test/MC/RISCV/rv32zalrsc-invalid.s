# RUN: llvm-mc --verify -triple riscv32 -mattr=+a < %s
# RUN: llvm-mc --verify -triple riscv32 -mattr=+zalrsc < %s

# Final operand must have parentheses
lr.w a4, a5 # expected-error:10 {{expected '(' or optional integer offset}}

# lr only takes two operands
lr.w s0, (s1), s2 # expected-error:16 {{unexpected extra operand for instruction}}
