# RUN: not llvm-mc -triple riscv32 -mattr=+xtheadcmo < %s 2>&1 | FileCheck --match-full-lines %s
# RUN: not llvm-mc -triple riscv64 -mattr=+xtheadcmo < %s 2>&1 | FileCheck --match-full-lines %s

th.dcache.csw # CHECK: <stdin>:[[@LINE]]:1: error: too few operands for instruction
th.dcache.isw # CHECK: <stdin>:[[@LINE]]:1: error: too few operands for instruction
th.dcache.cisw # CHECK: <stdin>:[[@LINE]]:1: error: too few operands for instruction
th.dcache.cval1 # CHECK: <stdin>:[[@LINE]]:1: error: too few operands for instruction
th.dcache.cva # CHECK: <stdin>:[[@LINE]]:1: error: too few operands for instruction
th.dcache.iva # CHECK: <stdin>:[[@LINE]]:1: error: too few operands for instruction
th.dcache.civa # CHECK: <stdin>:[[@LINE]]:1: error: too few operands for instruction
th.dcache.cpal1 # CHECK: <stdin>:[[@LINE]]:1: error: too few operands for instruction
th.dcache.cpa # CHECK: <stdin>:[[@LINE]]:1: error: too few operands for instruction
th.dcache.ipa # CHECK: <stdin>:[[@LINE]]:1: error: too few operands for instruction
th.dcache.cipa # CHECK: <stdin>:[[@LINE]]:1: error: too few operands for instruction
th.icache.iva # CHECK: <stdin>:[[@LINE]]:1: error: too few operands for instruction
th.icache.ipa # CHECK: <stdin>:[[@LINE]]:1: error: too few operands for instruction

th.dcache.call t0 # CHECK: <stdin>:[[@LINE]]:16: error: unexpected extra operand for instruction
th.dcache.iall t0 # CHECK: <stdin>:[[@LINE]]:16: error: unexpected extra operand for instruction
th.dcache.ciall t0 # CHECK: <stdin>:[[@LINE]]:17: error: unexpected extra operand for instruction
th.icache.iall t0 # CHECK: <stdin>:[[@LINE]]:16: error: unexpected extra operand for instruction
th.icache.ialls t0 # CHECK: <stdin>:[[@LINE]]:17: error: unexpected extra operand for instruction
th.l2cache.call t0 # CHECK: <stdin>:[[@LINE]]:17: error: unexpected extra operand for instruction
th.l2cache.iall t0 # CHECK: <stdin>:[[@LINE]]:17: error: unexpected extra operand for instruction
th.l2cache.ciall t0 # CHECK: <stdin>:[[@LINE]]:18: error: unexpected extra operand for instruction
