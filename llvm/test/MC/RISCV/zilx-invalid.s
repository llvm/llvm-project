# RUN: llvm-mc --verify=expected,rv32-error -triple=riscv32 --mattr=+experimental-zilx %s
# RUN: llvm-mc --verify -triple=riscv64 --mattr=+experimental-zilx %s

# The base register must be written in parentheses; the bare three-register
# syntax is not accepted.
lxh a0, a1, a2 # expected-error:9 {{expected '(' or optional integer offset}}

# Only a zero offset may precede the parenthesized base register.
lxh a0, 1, a2 # expected-error:10 {{expected '(' after optional integer offset}}

# The base operand must be a register.
lxh a0, (1), a2 # expected-error:10 {{expected register}}

lxh a0, a1 # expected-error:9 {{expected '(' or optional integer offset}}

# Doubleword and unsigned-word forms are RV64-only.
lxd a0, (a1), a2 # rv32-error-error:1 {{instruction requires the following: RV64I Base Instruction Set}}

lxwu a0, (a1), a2 # rv32-error-error:1 {{instruction requires the following: RV64I Base Instruction Set}}

lxsd a0, (a1), a2 # rv32-error-error:1 {{instruction requires the following: RV64I Base Instruction Set}}

lxswu a0, (a1), a2 # rv32-error-error:1 {{instruction requires the following: RV64I Base Instruction Set}}

# The scaled unsigned-word-index loads are RV64-only.
lxsuwb a0, (a1), a2 # rv32-error-error:1 {{instruction requires the following: RV64I Base Instruction Set}}

lxsuwwu a0, (a1), a2 # rv32-error-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
