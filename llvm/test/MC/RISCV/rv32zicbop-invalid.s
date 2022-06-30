# RUN: not llvm-mc -triple riscv32 -mattr=+zicbop < %s 2>&1 | FileCheck %s
# RUN: not llvm-mc -triple riscv64 -mattr=+zicbop < %s 2>&1 | FileCheck %s

# Memory operand not formatted correctly.
prefetch.i a0, 32 # CHECK: :[[@LINE]]:12: error: immediate must be a multiple of 32 bytes in the range [-2048, 2016]
prefetch.r 32, a0 # CHECK: :[[@LINE]]:16: error: invalid operand for instruction
prefetch.w a0(32) # CHECK: :[[@LINE]]:14: error: unexpected token

# Out of range offset.
prefetch.i -2080(a0) # CHECK: :[[@LINE]]:12: error: immediate must be a multiple of 32 bytes in the range [-2048, 2016]
prefetch.r 2048(a0) # CHECK: :[[@LINE]]:12: error: immediate must be a multiple of 32 bytes in the range [-2048, 2016]
prefetch.w 2050(a0) # CHECK: :[[@LINE]]:12: error: immediate must be a multiple of 32 bytes in the range [-2048, 2016]

# Offsets that aren't multiples of 32.
prefetch.i 31(a0) # CHECK: :[[@LINE]]:12: error: immediate must be a multiple of 32 bytes in the range [-2048, 2016]
prefetch.r -31(a0) # CHECK: :[[@LINE]]:12: error: immediate must be a multiple of 32 bytes in the range [-2048, 2016]
prefetch.w 2047(a0) # CHECK: :[[@LINE]]:12: error: immediate must be a multiple of 32 bytes in the range [-2048, 2016]

# Symbols should not be accepted.
prefetch.i foo(a0) # CHECK: :[[@LINE]]:12: error: immediate must be a multiple of 32 bytes in the range [-2048, 2016]
prefetch.r %lo(foo)(a0) # CHECK: :[[@LINE]]:12: error: immediate must be a multiple of 32 bytes in the range [-2048, 2016]
prefetch.w %pcrel_lo(foo)(a0) # CHECK: :[[@LINE]]:12: error: immediate must be a multiple of 32 bytes in the range [-2048, 2016]

# Instructions from other zicbo* extensions aren't available without enabling
# the appropriate -mattr flag.
cbo.clean (t0) # CHECK: :[[@LINE]]:1: error: instruction requires the following: 'Zicbom' (Cache-Block Management Instructions)
cbo.flush (t1) # CHECK: :[[@LINE]]:1: error: instruction requires the following: 'Zicbom' (Cache-Block Management Instructions)
cbo.inval (t2) # CHECK: :[[@LINE]]:1: error: instruction requires the following: 'Zicbom' (Cache-Block Management Instructions)
cbo.zero (t0) # CHECK: :[[@LINE]]:1: error: instruction requires the following: 'Zicboz' (Cache-Block Zero Instructions)
