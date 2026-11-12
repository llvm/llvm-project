# RUN: not llvm-mc -triple riscv32 < %s 2>&1 | FileCheck --match-full-lines %s

jump 1234, x31 # CHECK: <stdin>:[[@LINE]]:6: error: operand must be a valid jump target
jump foo@plt, x31 # CHECK: <stdin>:[[@LINE]]:9: error: unexpected token
jump %pcrel_lo(1234), x31 # CHECK: <stdin>:[[@LINE]]:6: error: unknown token in expression
jump foo@xxx # CHECK: <stdin>:[[@LINE]]:9: error: unexpected token
