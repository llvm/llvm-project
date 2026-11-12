# RUN: not llvm-mc -triple riscv32 -mattr=+zicbom < %s 2>&1 | FileCheck --match-full-lines %s
# RUN: not llvm-mc -triple riscv64 -mattr=+zicbom < %s 2>&1 | FileCheck --match-full-lines %s

# Must have a single register argument.
cbo.clean # CHECK: <stdin>:[[@LINE]]:1: error: too few operands for instruction
cbo.flush # CHECK: <stdin>:[[@LINE]]:1: error: too few operands for instruction
cbo.inval # CHECK: <stdin>:[[@LINE]]:1: error: too few operands for instruction

cbo.clean 1 # CHECK: <stdin>:[[@LINE]]:13: error: expected '(' after optional integer offset
cbo.flush 2 # CHECK: <stdin>:[[@LINE]]:13: error: expected '(' after optional integer offset
cbo.inval 3 # CHECK: <stdin>:[[@LINE]]:13: error: expected '(' after optional integer offset

cbo.clean t0, t1 # CHECK: <stdin>:[[@LINE]]:11: error: expected '(' or optional integer offset
cbo.flush t0, t1 # CHECK: <stdin>:[[@LINE]]:11: error: expected '(' or optional integer offset
cbo.inval t0, t1 # CHECK: <stdin>:[[@LINE]]:11: error: expected '(' or optional integer offset

# Non-zero offsets are not supported.
cbo.clean 1(t0) # CHECK: <stdin>:[[@LINE]]:11: error: optional integer offset must be 0
cbo.flush 2(t0) # CHECK: <stdin>:[[@LINE]]:11: error: optional integer offset must be 0
cbo.inval 3(t0) # CHECK: <stdin>:[[@LINE]]:11: error: optional integer offset must be 0

# Instructions from other zicbo* extensions aren't available without enabling
# the appropriate -mattr flag.
cbo.zero (t0) # CHECK: <stdin>:[[@LINE]]:1: error: instruction requires the following: 'Zicboz' (Cache-Block Zero Instructions)
