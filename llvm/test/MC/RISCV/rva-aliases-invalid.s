# RUN: llvm-mc --verify %s -triple=riscv32
# RUN: llvm-mc --verify %s -triple=riscv64

# The below tests for lr(.w), sc(.w) and amo*(.w), using `0(reg)` are actually
# implemented using a custom parser. These tests ensure the custom parser gives
# good error messages.

lr.w a1, a0     # expected-error:10 {{expected '(' or optional integer offset}}
lr.w a1, foo    # expected-error:10 {{expected '(' or optional integer offset}}
lr.w a1, 1(a0)  # expected-error:10 {{optional integer offset must be 0}}
lr.w a1, (foo)  # expected-error:11 {{expected register}}
lr.w a1, 0(foo) # expected-error:12 {{expected register}}
lr.w a1, (f0)   # expected-error:1 {{invalid instruction}}
lr.w a1, 0(f0)  # expected-error:1 {{invalid instruction}}
lr.w a1, 0(a0   # expected-error:17 {{expected ')'}}
lr.w a1, (a0    # expected-error:17 {{expected ')'}}

sc.w a2, a1, a0     # expected-error:14 {{expected '(' or optional integer offset}}
sc.w a2, a1, foo    # expected-error:14 {{expected '(' or optional integer offset}}
sc.w a2, a1, 1(a0)  # expected-error:14 {{optional integer offset must be 0}}
sc.w a2, a1, (foo)  # expected-error:15 {{expected register}}
sc.w a2, a1, 0(foo) # expected-error:16 {{expected register}}
sc.w a2, a1, (f0)   # expected-error:1 {{invalid instruction}}
sc.w a2, a1, 0(f0)  # expected-error:1 {{invalid instruction}}
sc.w a2, a1, 0(a0   # expected-error:21 {{expected ')'}}
sc.w a2, a1, (a0    # expected-error:21 {{expected ')'}}

amoswap.w a2, a1, a0     # expected-error:19 {{expected '(' or optional integer offset}}
amoswap.w a2, a1, foo    # expected-error:19 {{expected '(' or optional integer offset}}
amoswap.w a2, a1, 1(a0)  # expected-error:19 {{optional integer offset must be 0}}
amoswap.w a2, a1, (foo)  # expected-error:20 {{expected register}}
amoswap.w a2, a1, 0(foo) # expected-error:21 {{expected register}}
amoswap.w a2, a1, (f0)   # expected-error:1 {{invalid instruction}}
amoswap.w a2, a1, 0(f0)  # expected-error:1 {{invalid instruction}}
amoswap.w a2, a1, 0(a0   # expected-error:26 {{expected ')'}}
amoswap.w a2, a1, (a0    # expected-error:26 {{expected ')'}}

amoadd.w a2, a1, a0     # expected-error:18 {{expected '(' or optional integer offset}}
amoadd.w a2, a1, foo    # expected-error:18 {{expected '(' or optional integer offset}}
amoadd.w a2, a1, 1(a0)  # expected-error:18 {{optional integer offset must be 0}}
amoadd.w a2, a1, (foo)  # expected-error:19 {{expected register}}
amoadd.w a2, a1, 0(foo) # expected-error:20 {{expected register}}
amoadd.w a2, a1, (f0)   # expected-error:1 {{invalid instruction}}
amoadd.w a2, a1, 0(f0)  # expected-error:1 {{invalid instruction}}
amoadd.w a2, a1, 0(a0   # expected-error:25 {{expected ')'}}
amoadd.w a2, a1, (a0    # expected-error:25 {{expected ')'}}

amoxor.w a2, a1, a0     # expected-error:18 {{expected '(' or optional integer offset}}
amoxor.w a2, a1, foo    # expected-error:18 {{expected '(' or optional integer offset}}
amoxor.w a2, a1, 1(a0)  # expected-error:18 {{optional integer offset must be 0}}
amoxor.w a2, a1, (foo)  # expected-error:19 {{expected register}}
amoxor.w a2, a1, 0(foo) # expected-error:20 {{expected register}}
amoxor.w a2, a1, (f0)   # expected-error:1 {{invalid instruction}}
amoxor.w a2, a1, 0(f0)  # expected-error:1 {{invalid instruction}}
amoxor.w a2, a1, 0(a0   # expected-error:25 {{expected ')'}}
amoxor.w a2, a1, (a0    # expected-error:25 {{expected ')'}}

amoand.w a2, a1, a0     # expected-error:18 {{expected '(' or optional integer offset}}
amoand.w a2, a1, foo    # expected-error:18 {{expected '(' or optional integer offset}}
amoand.w a2, a1, 1(a0)  # expected-error:18 {{optional integer offset must be 0}}
amoand.w a2, a1, (foo)  # expected-error:19 {{expected register}}
amoand.w a2, a1, 0(foo) # expected-error:20 {{expected register}}
amoand.w a2, a1, (f0)   # expected-error:1 {{invalid instruction}}
amoand.w a2, a1, 0(f0)  # expected-error:1 {{invalid instruction}}
amoand.w a2, a1, 0(a0   # expected-error:25 {{expected ')'}}
amoand.w a2, a1, (a0    # expected-error:25 {{expected ')'}}

amoor.w a2, a1, a0     # expected-error:17 {{expected '(' or optional integer offset}}
amoor.w a2, a1, foo    # expected-error:17 {{expected '(' or optional integer offset}}
amoor.w a2, a1, 1(a0)  # expected-error:17 {{optional integer offset must be 0}}
amoor.w a2, a1, (foo)  # expected-error:18 {{expected register}}
amoor.w a2, a1, 0(foo) # expected-error:19 {{expected register}}
amoor.w a2, a1, (f0)   # expected-error:1 {{invalid instruction}}
amoor.w a2, a1, 0(f0)  # expected-error:1 {{invalid instruction}}
amoor.w a2, a1, 0(a0   # expected-error:24 {{expected ')'}}
amoor.w a2, a1, (a0    # expected-error:24 {{expected ')'}}

amomin.w a2, a1, a0     # expected-error:18 {{expected '(' or optional integer offset}}
amomin.w a2, a1, foo    # expected-error:18 {{expected '(' or optional integer offset}}
amomin.w a2, a1, 1(a0)  # expected-error:18 {{optional integer offset must be 0}}
amomin.w a2, a1, (foo)  # expected-error:19 {{expected register}}
amomin.w a2, a1, 0(foo) # expected-error:20 {{expected register}}
amomin.w a2, a1, (f0)   # expected-error:1 {{invalid instruction}}
amomin.w a2, a1, 0(f0)  # expected-error:1 {{invalid instruction}}
amomin.w a2, a1, 0(a0   # expected-error:25 {{expected ')'}}
amomin.w a2, a1, (a0    # expected-error:25 {{expected ')'}}

amomax.w a2, a1, a0     # expected-error:18 {{expected '(' or optional integer offset}}
amomax.w a2, a1, foo    # expected-error:18 {{expected '(' or optional integer offset}}
amomax.w a2, a1, 1(a0)  # expected-error:18 {{optional integer offset must be 0}}
amomax.w a2, a1, (foo)  # expected-error:19 {{expected register}}
amomax.w a2, a1, 0(foo) # expected-error:20 {{expected register}}
amomax.w a2, a1, (f0)   # expected-error:1 {{invalid instruction}}
amomax.w a2, a1, 0(f0)  # expected-error:1 {{invalid instruction}}
amomax.w a2, a1, 0(a0   # expected-error:25 {{expected ')'}}
amomax.w a2, a1, (a0    # expected-error:25 {{expected ')'}}

amominu.w a2, a1, a0     # expected-error:19 {{expected '(' or optional integer offset}}
amominu.w a2, a1, foo    # expected-error:19 {{expected '(' or optional integer offset}}
amominu.w a2, a1, 1(a0)  # expected-error:19 {{optional integer offset must be 0}}
amominu.w a2, a1, (foo)  # expected-error:20 {{expected register}}
amominu.w a2, a1, 0(foo) # expected-error:21 {{expected register}}
amominu.w a2, a1, (f0)   # expected-error:1 {{invalid instruction}}
amominu.w a2, a1, 0(f0)  # expected-error:1 {{invalid instruction}}
amominu.w a2, a1, 0(a0   # expected-error:26 {{expected ')'}}
amominu.w a2, a1, (a0    # expected-error:26 {{expected ')'}}

amomaxu.w a2, a1, a0     # expected-error:19 {{expected '(' or optional integer offset}}
amomaxu.w a2, a1, foo    # expected-error:19 {{expected '(' or optional integer offset}}
amomaxu.w a2, a1, 1(a0)  # expected-error:19 {{optional integer offset must be 0}}
amomaxu.w a2, a1, (foo)  # expected-error:20 {{expected register}}
amomaxu.w a2, a1, 0(foo) # expected-error:21 {{expected register}}
amomaxu.w a2, a1, (f0)   # expected-error:1 {{invalid instruction}}
amomaxu.w a2, a1, 0(f0)  # expected-error:1 {{invalid instruction}}
amomaxu.w a2, a1, 0(a0   # expected-error:26 {{expected ')'}}
amomaxu.w a2, a1, (a0    # expected-error:26 {{expected ')'}}
