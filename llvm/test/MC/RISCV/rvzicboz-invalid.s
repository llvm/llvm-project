# RUN: llvm-mc --verify -triple riscv32 -mattr=+zicboz < %s
# RUN: llvm-mc --verify -triple riscv64 -mattr=+zicboz < %s

# Must have a single register argument.
cbo.zero # expected-error:1 {{too few operands for instruction}}
cbo.zero 1 # expected-error:12 {{expected '(' after optional integer offset}}
cbo.zero t0, t1 # expected-error:10 {{expected '(' or optional integer offset}}

# Non-zero offsets are not supported.
cbo.zero 1(t0) # expected-error:10 {{optional integer offset must be 0}}

# Instructions from other zicbo* extensions aren't available without enabling
# the appropriate -mattr flag.
cbo.clean (t0) # expected-error:1 {{instruction requires the following: 'Zicbom' (Cache-Block Management Instructions)}}
cbo.flush (t1) # expected-error:1 {{instruction requires the following: 'Zicbom' (Cache-Block Management Instructions)}}
cbo.inval (t2) # expected-error:1 {{instruction requires the following: 'Zicbom' (Cache-Block Management Instructions)}}
