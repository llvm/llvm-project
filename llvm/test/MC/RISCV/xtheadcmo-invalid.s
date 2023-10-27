# RUN: not llvm-mc -triple riscv32 -mattr=+xtheadcmo < %s 2>&1 | FileCheck %s
# RUN: not llvm-mc -triple riscv64 -mattr=+xtheadcmo < %s 2>&1 | FileCheck %s

th.dcache.csw # CHECK: :[[@LINE]]:1: error: too few operands for instruction
th.dcache.isw # CHECK: :[[@LINE]]:1: error: too few operands for instruction
th.dcache.cisw # CHECK: :[[@LINE]]:1: error: too few operands for instruction
th.dcache.cval1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
th.dcache.cva # CHECK: :[[@LINE]]:1: error: too few operands for instruction
th.dcache.iva # CHECK: :[[@LINE]]:1: error: too few operands for instruction
th.dcache.civa # CHECK: :[[@LINE]]:1: error: too few operands for instruction
th.dcache.cpal1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
th.dcache.cpa # CHECK: :[[@LINE]]:1: error: too few operands for instruction
th.dcache.ipa # CHECK: :[[@LINE]]:1: error: too few operands for instruction
th.dcache.cipa # CHECK: :[[@LINE]]:1: error: too few operands for instruction
th.icache.iva # CHECK: :[[@LINE]]:1: error: too few operands for instruction
th.icache.ipa # CHECK: :[[@LINE]]:1: error: too few operands for instruction

th.dcache.call t0 # CHECK: :[[@LINE]]:16: error: invalid operand for instruction
th.dcache.iall t0 # CHECK: :[[@LINE]]:16: error: invalid operand for instruction
th.dcache.ciall t0 # CHECK: :[[@LINE]]:17: error: invalid operand for instruction
th.icache.iall t0 # CHECK: :[[@LINE]]:16: error: invalid operand for instruction
th.icache.ialls t0 # CHECK: :[[@LINE]]:17: error: invalid operand for instruction
th.l2cache.call t0 # CHECK: :[[@LINE]]:17: error: invalid operand for instruction
th.l2cache.iall t0 # CHECK: :[[@LINE]]:17: error: invalid operand for instruction
th.l2cache.ciall t0 # CHECK: :[[@LINE]]:18: error: invalid operand for instruction
