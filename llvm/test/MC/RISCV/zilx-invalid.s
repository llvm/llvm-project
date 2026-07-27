# RUN: not llvm-mc -triple=riscv32 --mattr=+experimental-zilx %s 2>&1 \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ERROR,CHECK-RV32-ERROR
# RUN: not llvm-mc -triple=riscv64 --mattr=+experimental-zilx %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR

# The base register must be written in parentheses; the bare three-register
# syntax is not accepted.
# CHECK-ERROR: :[[@LINE+1]]:9: error: expected '(' or optional integer offset
lxh a0, a1, a2

# Only a zero offset may precede the parenthesized base register.
# CHECK-ERROR: :[[@LINE+1]]:10: error: expected '(' after optional integer offset
lxh a0, 1, a2

# The base operand must be a register.
# CHECK-ERROR: :[[@LINE+1]]:10: error: expected register
lxh a0, (1), a2

# CHECK-ERROR: :[[@LINE+1]]:9: error: expected '(' or optional integer offset
lxh a0, a1

# Doubleword and unsigned-word forms are RV64-only.
# CHECK-RV32-ERROR: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
lxd a0, (a1), a2

# CHECK-RV32-ERROR: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
lxwu a0, (a1), a2

# CHECK-RV32-ERROR: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
lxsd a0, (a1), a2

# CHECK-RV32-ERROR: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
lxswu a0, (a1), a2

# The scaled unsigned-word-index loads are RV64-only.
# CHECK-RV32-ERROR: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
lxsuwb a0, (a1), a2

# CHECK-RV32-ERROR: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set
lxsuwwu a0, (a1), a2
