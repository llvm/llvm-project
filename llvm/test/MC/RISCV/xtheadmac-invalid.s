# RUN: not llvm-mc -triple riscv32 -mattr=+xtheadmac < %s 2>&1 | FileCheck --check-prefixes=CHECK,CHECK-RV32 %s
# RUN: not llvm-mc -triple riscv64 -mattr=+xtheadmac < %s 2>&1 | FileCheck --check-prefixes=CHECK,CHECK-RV64 %s

th.mula  t0, t1, 0(t2)
# CHECK: :[[@LINE-1]]:18: error: register must be a GPR

th.muls  t0, 12, t2     
# CHECK: :[[@LINE-1]]:14: error: register must be a GPR

th.mulsh t0, t1, 34     
# CHECK: :[[@LINE-1]]:18: error: register must be a GPR

th.mulsw t0, -56, t2    
# CHECK-RV64: :[[@LINE-1]]:14: error: register must be a GPR
# CHECK-RV32: :[[@LINE-2]]:1: error: invalid instruction
