# RUN: llvm-mc --verify -triple riscv32 -mattr=+zicbom < %s
# RUN: llvm-mc --verify -triple riscv64 -mattr=+zicbom < %s

# Must have a single register argument.
cbo.clean # expected-error:1 {{too few operands for instruction}}
cbo.flush # expected-error:1 {{too few operands for instruction}}
cbo.inval # expected-error:1 {{too few operands for instruction}}

cbo.clean 1 # expected-error:13 {{expected '(' after optional integer offset}}
cbo.flush 2 # expected-error:13 {{expected '(' after optional integer offset}}
cbo.inval 3 # expected-error:13 {{expected '(' after optional integer offset}}

cbo.clean t0, t1 # expected-error:11 {{expected '(' or optional integer offset}}
cbo.flush t0, t1 # expected-error:11 {{expected '(' or optional integer offset}}
cbo.inval t0, t1 # expected-error:11 {{expected '(' or optional integer offset}}

# Non-zero offsets are not supported.
cbo.clean 1(t0) # expected-error:11 {{optional integer offset must be 0}}
cbo.flush 2(t0) # expected-error:11 {{optional integer offset must be 0}}
cbo.inval 3(t0) # expected-error:11 {{optional integer offset must be 0}}

# Instructions from other zicbo* extensions aren't available without enabling
# the appropriate -mattr flag.
cbo.zero (t0) # expected-error:1 {{instruction requires the following: 'Zicboz' (Cache-Block Zero Instructions)}}
