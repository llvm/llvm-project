# RUN: not llvm-mc -triple riscv32 < %s 2>&1 | FileCheck %s

call 1234 # CHECK: :[[@LINE]]:6: error: operand must be a bare symbol name
call %pcrel_hi(1234) # CHECK: :[[@LINE]]:6: error: operand must be a bare symbol name
call %pcrel_lo(1234) # CHECK: :[[@LINE]]:6: error: operand must be a bare symbol name
call %pcrel_hi(foo) # CHECK: :[[@LINE]]:6: error: operand must be a bare symbol name
call %pcrel_lo(foo) # CHECK: :[[@LINE]]:6: error: operand must be a bare symbol name
call %hi(1234) # CHECK: :[[@LINE]]:6: error: operand must be a bare symbol name
call %lo(1234) # CHECK: :[[@LINE]]:6: error: operand must be a bare symbol name
call %hi(foo) # CHECK: :[[@LINE]]:6: error: operand must be a bare symbol name
call %lo(foo) # CHECK: :[[@LINE]]:6: error: operand must be a bare symbol name
call foo, bar # CHECK: :[[@LINE]]:6: error: operand must be a bare symbol name
call foo@pls # CHECK: :[[@LINE]]:10: error: @ (except the deprecated/ignored @plt) is disallowed
call foo@3 # CHECK: :[[@LINE]]:10: error: @ (except the deprecated/ignored @plt) is disallowed
