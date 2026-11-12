# RUN: not llvm-mc -triple riscv64 < %s 2>&1 | FileCheck --match-full-lines %s

hlv.wu a0, 0x10 # CHECK: <stdin>:[[@LINE]]:17: error: expected '(' after optional integer offset

hlv.wu a0, a1 # CHECK: <stdin>:[[@LINE]]:12: error: expected '(' or optional integer offset

hlv.wu a0, 1(a1) # CHECK: <stdin>:[[@LINE]]:12: error: optional integer offset must be 0

hlv.d a0, 0x10 # CHECK: <stdin>:[[@LINE]]:16: error: expected '(' after optional integer offset

hlv.d a0, a1 # CHECK: <stdin>:[[@LINE]]:11: error: expected '(' or optional integer offset

hlv.d a0, 1(a1) # CHECK: <stdin>:[[@LINE]]:11: error: optional integer offset must be 0

hsv.d a0, 0x10 # CHECK: <stdin>:[[@LINE]]:16: error: expected '(' after optional integer offset

hsv.d a0, a1 # CHECK: <stdin>:[[@LINE]]:11: error: expected '(' or optional integer offset

hsv.d a0, 1(a1) # CHECK: <stdin>:[[@LINE]]:11: error: optional integer offset must be 0
