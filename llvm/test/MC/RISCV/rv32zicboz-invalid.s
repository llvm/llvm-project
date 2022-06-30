# RUN: not llvm-mc -triple riscv32 -mattr=+zicboz < %s 2>&1 | FileCheck %s
# RUN: not llvm-mc -triple riscv64 -mattr=+zicboz < %s 2>&1 | FileCheck %s

# Must have a single register argument.
cbo.zero # CHECK: :[[@LINE]]:1: error: too few operands for instruction
cbo.zero 1 # CHECK: :[[@LINE]]:12: error: expected '(' after optional integer offset
cbo.zero t0, t1 # CHECK: :[[@LINE]]:10: error: expected '(' or optional integer offset

# Non-zero offsets are not supported.
cbo.zero 1(t0) # CHECK: :[[@LINE]]:10: error: optional integer offset must be 0

# Instructions from other zicbo* extensions aren't available without enabling
# the appropriate -mattr flag.
cbo.clean (t0) # CHECK: :[[@LINE]]:1: error: instruction requires the following: 'Zicbom' (Cache-Block Management Instructions)
cbo.flush (t1) # CHECK: :[[@LINE]]:1: error: instruction requires the following: 'Zicbom' (Cache-Block Management Instructions)
cbo.inval (t2) # CHECK: :[[@LINE]]:1: error: instruction requires the following: 'Zicbom' (Cache-Block Management Instructions)
prefetch.i 0(t3) # CHECK: :[[@LINE]]:1: error: instruction requires the following: 'Zicbop' (Cache-Block Prefetch Instructions)
prefetch.r 0(t4) # CHECK: :[[@LINE]]:1: error: instruction requires the following: 'Zicbop' (Cache-Block Prefetch Instructions)
prefetch.w 0(t5) # CHECK: :[[@LINE]]:1: error: instruction requires the following: 'Zicbop' (Cache-Block Prefetch Instructions)
