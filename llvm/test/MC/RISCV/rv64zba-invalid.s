# RUN: not llvm-mc -triple riscv64 -mattr=+zba < %s 2>&1 | FileCheck %s

# Too few operands
slli.uw t0, t1 # CHECK: :[[@LINE]]:15: error: too few operands for instruction
# Immediate operand out of range
slli.uw t0, t1, 64 # CHECK: :[[@LINE]]:17: error: immediate must be an integer in the range [0, 63]
slli.uw t0, t1, -1 # CHECK: :[[@LINE]]:17: error: immediate must be an integer in the range [0, 63]
# Too few operands
add.uw t0, t1 # CHECK: :[[@LINE]]:14: error: too few operands for instruction
# Too few operands
sh1add.uw t0, t1 # CHECK: :[[@LINE]]:17: error: too few operands for instruction
# Too few operands
sh2add.uw t0, t1 # CHECK: :[[@LINE]]:17: error: too few operands for instruction
# Too few operands
sh3add.uw t0, t1 # CHECK: :[[@LINE]]:17: error: too few operands for instruction

# Base+index symbol names require a %base_idx_add modifier.
sh1add a0, a0, a1, %hi(foo)
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:20: note: invalid operand for instruction
# CHECK: :[[@LINE-3]]:20: note: operand must be a symbol with %base_idx_add specifier

sh2add a0, a0, a1, %hi(foo)
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:20: note: invalid operand for instruction
# CHECK: :[[@LINE-3]]:20: note: operand must be a symbol with %base_idx_add specifier

sh3add a0, a0, a1, %hi(foo)
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:20: note: invalid operand for instruction
# CHECK: :[[@LINE-3]]:20: note: operand must be a symbol with %base_idx_add specifier

add.uw a0, a0, a1, %hi(foo)
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:20: note: invalid operand for instruction
# CHECK: :[[@LINE-3]]:20: note: operand must be a symbol with %base_idx_add specifier

sh1add.uw a0, a0, a1, %hi(foo)
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:23: note: invalid operand for instruction
# CHECK: :[[@LINE-3]]:23: note: operand must be a symbol with %base_idx_add specifier

sh2add.uw a0, a0, a1, %hi(foo)
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:23: note: invalid operand for instruction
# CHECK: :[[@LINE-3]]:23: note: operand must be a symbol with %base_idx_add specifier

sh3add.uw a0, a0, a1, %hi(foo)
# CHECK: :[[@LINE-1]]:1: error: invalid instruction, any one of the following would fix this:
# CHECK: :[[@LINE-2]]:23: note: invalid operand for instruction
# CHECK: :[[@LINE-3]]:23: note: operand must be a symbol with %base_idx_add specifier