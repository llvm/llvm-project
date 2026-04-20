# RUN: not llvm-mc -triple riscv32 -mattr=+e,+xqccmp < %s 2>&1 | FileCheck %s

# Perform a simple check that registers x16-x31 (and the equivalent ABI names)
# are rejected for RV32E during assembly.


# CHECK: :[[@LINE+1]]:19: error: invalid register
qc.cm.push {ra,s0-s2}, -16
# CHECK: :[[@LINE+1]]:21: error: invalid register
qc.cm.popret {ra,s0-s2}, 16
# CHECK: :[[@LINE+1]]:23: error: invalid register
qc.cm.pop {x1, x8-x9, x18}, 16
# CHECK: :[[@LINE+1]]:26: error: invalid register
qc.cm.pushfp {x1, x8-x9, x18}, -16
# CHECK: :[[@LINE+1]]:22: error: invalid register
qc.cm.pushfp {ra, s0-s2}, -16
