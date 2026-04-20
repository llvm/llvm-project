# RUN: not llvm-mc -triple riscv64 -mattr=+e,+zcmp < %s 2>&1 | FileCheck %s

# Perform a simple check that registers x16-x31 (and the equivalent ABI names)
# are rejected for RV64E during assembly.


# CHECK: :[[@LINE+1]]:16: error: invalid register
cm.push {ra,s0-s2}, -32
# CHECK: :[[@LINE+1]]:18: error: invalid register
cm.popret {ra,s0-s2}, 32
# CHECK: :[[@LINE+1]]:20: error: invalid register
cm.pop {x1, x8-x9, x18}, 32
# CHECK: :[[@LINE+1]]:16: error: invalid register
cm.pop {ra, s0-s2}, 32
