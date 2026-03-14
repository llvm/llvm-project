# RUN: not llvm-mc %s -triple=riscv32 2>&1 | FileCheck %s
# RUN: not llvm-mc %s -triple=riscv64 2>&1 | FileCheck %s

# The below tests for lr(.w), sc(.w) and amo*(.w), using `0(reg)` are actually
# implemented using a custom parser. These tests ensure the custom parser gives
# good error messages.

lr.w a1, a0     # CHECK: :[[@LINE]]:10: error: expected '(' or optional integer offset
lr.w a1, foo    # CHECK: :[[@LINE]]:10: error: expected '(' or optional integer offset
lr.w a1, 1(a0)  # CHECK: :[[@LINE]]:10: error: optional integer offset must be 0
lr.w a1, (foo)  # CHECK: :[[@LINE]]:11: error: expected register
lr.w a1, 0(foo) # CHECK: :[[@LINE]]:12: error: expected register
lr.w a1, (f0)   # CHECK: :[[@LINE]]:11: error: invalid operand for instruction
lr.w a1, 0(f0)  # CHECK: :[[@LINE]]:12: error: invalid operand for instruction
lr.w a1, 0(a0   # CHECK: :[[@LINE]]:17: error: expected ')'
lr.w a1, (a0    # CHECK: :[[@LINE]]:17: error: expected ')'

sc.w a2, a1, a0     # CHECK: :[[@LINE]]:14: error: expected '(' or optional integer offset
sc.w a2, a1, foo    # CHECK: :[[@LINE]]:14: error: expected '(' or optional integer offset
sc.w a2, a1, 1(a0)  # CHECK: :[[@LINE]]:14: error: optional integer offset must be 0
sc.w a2, a1, (foo)  # CHECK: :[[@LINE]]:15: error: expected register
sc.w a2, a1, 0(foo) # CHECK: :[[@LINE]]:16: error: expected register
sc.w a2, a1, (f0)   # CHECK: :[[@LINE]]:15: error: invalid operand for instruction
sc.w a2, a1, 0(f0)  # CHECK: :[[@LINE]]:16: error: invalid operand for instruction
sc.w a2, a1, 0(a0   # CHECK: :[[@LINE]]:21: error: expected ')'
sc.w a2, a1, (a0    # CHECK: :[[@LINE]]:21: error: expected ')'

amoswap.w a2, a1, a0     # CHECK: :[[@LINE]]:19: error: expected '(' or optional integer offset
amoswap.w a2, a1, foo    # CHECK: :[[@LINE]]:19: error: expected '(' or optional integer offset
amoswap.w a2, a1, 1(a0)  # CHECK: :[[@LINE]]:19: error: optional integer offset must be 0
amoswap.w a2, a1, (foo)  # CHECK: :[[@LINE]]:20: error: expected register
amoswap.w a2, a1, 0(foo) # CHECK: :[[@LINE]]:21: error: expected register
amoswap.w a2, a1, (f0)   # CHECK: :[[@LINE]]:20: error: invalid operand for instruction
amoswap.w a2, a1, 0(f0)  # CHECK: :[[@LINE]]:21: error: invalid operand for instruction
amoswap.w a2, a1, 0(a0   # CHECK: :[[@LINE]]:26: error: expected ')'
amoswap.w a2, a1, (a0    # CHECK: :[[@LINE]]:26: error: expected ')'

amoadd.w a2, a1, a0     # CHECK: :[[@LINE]]:18: error: expected '(' or optional integer offset
amoadd.w a2, a1, foo    # CHECK: :[[@LINE]]:18: error: expected '(' or optional integer offset
amoadd.w a2, a1, 1(a0)  # CHECK: :[[@LINE]]:18: error: optional integer offset must be 0
amoadd.w a2, a1, (foo)  # CHECK: :[[@LINE]]:19: error: expected register
amoadd.w a2, a1, 0(foo) # CHECK: :[[@LINE]]:20: error: expected register
amoadd.w a2, a1, (f0)   # CHECK: :[[@LINE]]:19: error: invalid operand for instruction
amoadd.w a2, a1, 0(f0)  # CHECK: :[[@LINE]]:20: error: invalid operand for instruction
amoadd.w a2, a1, 0(a0   # CHECK: :[[@LINE]]:25: error: expected ')'
amoadd.w a2, a1, (a0    # CHECK: :[[@LINE]]:25: error: expected ')'

amoxor.w a2, a1, a0     # CHECK: :[[@LINE]]:18: error: expected '(' or optional integer offset
amoxor.w a2, a1, foo    # CHECK: :[[@LINE]]:18: error: expected '(' or optional integer offset
amoxor.w a2, a1, 1(a0)  # CHECK: :[[@LINE]]:18: error: optional integer offset must be 0
amoxor.w a2, a1, (foo)  # CHECK: :[[@LINE]]:19: error: expected register
amoxor.w a2, a1, 0(foo) # CHECK: :[[@LINE]]:20: error: expected register
amoxor.w a2, a1, (f0)   # CHECK: :[[@LINE]]:19: error: invalid operand for instruction
amoxor.w a2, a1, 0(f0)  # CHECK: :[[@LINE]]:20: error: invalid operand for instruction
amoxor.w a2, a1, 0(a0   # CHECK: :[[@LINE]]:25: error: expected ')'
amoxor.w a2, a1, (a0    # CHECK: :[[@LINE]]:25: error: expected ')'

amoand.w a2, a1, a0     # CHECK: :[[@LINE]]:18: error: expected '(' or optional integer offset
amoand.w a2, a1, foo    # CHECK: :[[@LINE]]:18: error: expected '(' or optional integer offset
amoand.w a2, a1, 1(a0)  # CHECK: :[[@LINE]]:18: error: optional integer offset must be 0
amoand.w a2, a1, (foo)  # CHECK: :[[@LINE]]:19: error: expected register
amoand.w a2, a1, 0(foo) # CHECK: :[[@LINE]]:20: error: expected register
amoand.w a2, a1, (f0)   # CHECK: :[[@LINE]]:19: error: invalid operand for instruction
amoand.w a2, a1, 0(f0)  # CHECK: :[[@LINE]]:20: error: invalid operand for instruction
amoand.w a2, a1, 0(a0   # CHECK: :[[@LINE]]:25: error: expected ')'
amoand.w a2, a1, (a0    # CHECK: :[[@LINE]]:25: error: expected ')'

amoor.w a2, a1, a0     # CHECK: :[[@LINE]]:17: error: expected '(' or optional integer offset
amoor.w a2, a1, foo    # CHECK: :[[@LINE]]:17: error: expected '(' or optional integer offset
amoor.w a2, a1, 1(a0)  # CHECK: :[[@LINE]]:17: error: optional integer offset must be 0
amoor.w a2, a1, (foo)  # CHECK: :[[@LINE]]:18: error: expected register
amoor.w a2, a1, 0(foo) # CHECK: :[[@LINE]]:19: error: expected register
amoor.w a2, a1, (f0)   # CHECK: :[[@LINE]]:18: error: invalid operand for instruction
amoor.w a2, a1, 0(f0)  # CHECK: :[[@LINE]]:19: error: invalid operand for instruction
amoor.w a2, a1, 0(a0   # CHECK: :[[@LINE]]:24: error: expected ')'
amoor.w a2, a1, (a0    # CHECK: :[[@LINE]]:24: error: expected ')'

amomin.w a2, a1, a0     # CHECK: :[[@LINE]]:18: error: expected '(' or optional integer offset
amomin.w a2, a1, foo    # CHECK: :[[@LINE]]:18: error: expected '(' or optional integer offset
amomin.w a2, a1, 1(a0)  # CHECK: :[[@LINE]]:18: error: optional integer offset must be 0
amomin.w a2, a1, (foo)  # CHECK: :[[@LINE]]:19: error: expected register
amomin.w a2, a1, 0(foo) # CHECK: :[[@LINE]]:20: error: expected register
amomin.w a2, a1, (f0)   # CHECK: :[[@LINE]]:19: error: invalid operand for instruction
amomin.w a2, a1, 0(f0)  # CHECK: :[[@LINE]]:20: error: invalid operand for instruction
amomin.w a2, a1, 0(a0   # CHECK: :[[@LINE]]:25: error: expected ')'
amomin.w a2, a1, (a0    # CHECK: :[[@LINE]]:25: error: expected ')'

amomax.w a2, a1, a0     # CHECK: :[[@LINE]]:18: error: expected '(' or optional integer offset
amomax.w a2, a1, foo    # CHECK: :[[@LINE]]:18: error: expected '(' or optional integer offset
amomax.w a2, a1, 1(a0)  # CHECK: :[[@LINE]]:18: error: optional integer offset must be 0
amomax.w a2, a1, (foo)  # CHECK: :[[@LINE]]:19: error: expected register
amomax.w a2, a1, 0(foo) # CHECK: :[[@LINE]]:20: error: expected register
amomax.w a2, a1, (f0)   # CHECK: :[[@LINE]]:19: error: invalid operand for instruction
amomax.w a2, a1, 0(f0)  # CHECK: :[[@LINE]]:20: error: invalid operand for instruction
amomax.w a2, a1, 0(a0   # CHECK: :[[@LINE]]:25: error: expected ')'
amomax.w a2, a1, (a0    # CHECK: :[[@LINE]]:25: error: expected ')'

amominu.w a2, a1, a0     # CHECK: :[[@LINE]]:19: error: expected '(' or optional integer offset
amominu.w a2, a1, foo    # CHECK: :[[@LINE]]:19: error: expected '(' or optional integer offset
amominu.w a2, a1, 1(a0)  # CHECK: :[[@LINE]]:19: error: optional integer offset must be 0
amominu.w a2, a1, (foo)  # CHECK: :[[@LINE]]:20: error: expected register
amominu.w a2, a1, 0(foo) # CHECK: :[[@LINE]]:21: error: expected register
amominu.w a2, a1, (f0)   # CHECK: :[[@LINE]]:20: error: invalid operand for instruction
amominu.w a2, a1, 0(f0)  # CHECK: :[[@LINE]]:21: error: invalid operand for instruction
amominu.w a2, a1, 0(a0   # CHECK: :[[@LINE]]:26: error: expected ')'
amominu.w a2, a1, (a0    # CHECK: :[[@LINE]]:26: error: expected ')'

amomaxu.w a2, a1, a0     # CHECK: :[[@LINE]]:19: error: expected '(' or optional integer offset
amomaxu.w a2, a1, foo    # CHECK: :[[@LINE]]:19: error: expected '(' or optional integer offset
amomaxu.w a2, a1, 1(a0)  # CHECK: :[[@LINE]]:19: error: optional integer offset must be 0
amomaxu.w a2, a1, (foo)  # CHECK: :[[@LINE]]:20: error: expected register
amomaxu.w a2, a1, 0(foo) # CHECK: :[[@LINE]]:21: error: expected register
amomaxu.w a2, a1, (f0)   # CHECK: :[[@LINE]]:20: error: invalid operand for instruction
amomaxu.w a2, a1, 0(f0)  # CHECK: :[[@LINE]]:21: error: invalid operand for instruction
amomaxu.w a2, a1, 0(a0   # CHECK: :[[@LINE]]:26: error: expected ')'
amomaxu.w a2, a1, (a0    # CHECK: :[[@LINE]]:26: error: expected ')'