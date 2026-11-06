# RUN: llvm-mc --verify=expected,rv32-error -triple=riscv32 --mattr=+experimental-zilx %s
# RUN: llvm-mc --verify -triple=riscv64 --mattr=+experimental-zilx %s

# The base register must be written in parentheses; the bare three-register
# syntax is not accepted.
# expected-error@+1:9 {{expected '(' or optional integer offset}}
lxh a0, a1, a2

# Only a zero offset may precede the parenthesized base register.
# expected-error@+1:10 {{expected '(' after optional integer offset}}
lxh a0, 1, a2

# The base operand must be a register.
# expected-error@+1:10 {{expected register}}
lxh a0, (1), a2

# expected-error@+1:9 {{expected '(' or optional integer offset}}
lxh a0, a1

# Doubleword and unsigned-word forms are RV64-only.
# rv32-error-error@+1:1 {{instruction requires the following: RV64I Base Instruction Set}}
lxd a0, (a1), a2

# rv32-error-error@+1:1 {{instruction requires the following: RV64I Base Instruction Set}}
lxwu a0, (a1), a2

# rv32-error-error@+1:1 {{instruction requires the following: RV64I Base Instruction Set}}
lxsd a0, (a1), a2

# rv32-error-error@+1:1 {{instruction requires the following: RV64I Base Instruction Set}}
lxswu a0, (a1), a2

# The scaled unsigned-word-index loads are RV64-only.
# rv32-error-error@+1:1 {{instruction requires the following: RV64I Base Instruction Set}}
lxsuwb a0, (a1), a2

# rv32-error-error@+1:1 {{instruction requires the following: RV64I Base Instruction Set}}
lxsuwwu a0, (a1), a2
