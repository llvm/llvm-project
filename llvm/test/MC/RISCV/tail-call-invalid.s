# RUN: not llvm-mc -triple riscv32 < %s 2>&1 | FileCheck --match-full-lines %s
# RUN: not llvm-mc -triple riscv64 < %s 2>&1 | FileCheck --match-full-lines %s

tail 1234 # CHECK: <stdin>:[[@LINE]]:6: error: operand must be a bare symbol name
tail %pcrel_hi(1234) # CHECK: <stdin>:[[@LINE]]:6: error: operand must be a bare symbol name
tail %pcrel_lo(1234) # CHECK: <stdin>:[[@LINE]]:6: error: operand must be a bare symbol name
tail %pcrel_hi(foo) # CHECK: <stdin>:[[@LINE]]:6: error: operand must be a bare symbol name
tail %pcrel_lo(foo) # CHECK: <stdin>:[[@LINE]]:6: error: operand must be a bare symbol name
tail %hi(1234) # CHECK: <stdin>:[[@LINE]]:6: error: operand must be a bare symbol name
tail %lo(1234) # CHECK: <stdin>:[[@LINE]]:6: error: operand must be a bare symbol name
tail %hi(foo) # CHECK: <stdin>:[[@LINE]]:6: error: operand must be a bare symbol name
tail %lo(foo) # CHECK: <stdin>:[[@LINE]]:6: error: operand must be a bare symbol name
tail foo@pls # CHECK: <stdin>:[[@LINE]]:10: error: @ (except the deprecated/ignored @plt) is disallowed
