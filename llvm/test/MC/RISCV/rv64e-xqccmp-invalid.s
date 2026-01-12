# RUN: not llvm-mc -triple riscv64 -mattr=+e,+xqccmp < %s 2>&1 | FileCheck %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+xqccmp < %s \
# RUN:     | llvm-objdump --mattr=+e,+xqccmp -M no-aliases -d -r - \
# RUN:     | FileCheck -check-prefix=CHECK-DIS %s

# Perform a simple check that registers x16-x31 (and the equivalent ABI names)
# are rejected for RV64E, when both assembling and disassembling.


# CHECK-DIS: b872 <unknown>
# CHECK: :[[@LINE+1]]:19: error: invalid register
qc.cm.push {ra,s0-s2}, -32
# CHECK-DIS: be72 <unknown>
# CHECK: :[[@LINE+1]]:21: error: invalid register
qc.cm.popret {ra,s0-s2}, 32
# CHECK-DIS: ba72 <unknown>
# CHECK: :[[@LINE+1]]:23: error: invalid register
qc.cm.pop {x1, x8-x9, x18}, 32
# CHECK-DIS: b972 <unknown>
# CHECK: :[[@LINE+1]]:26: error: invalid register
qc.cm.pushfp {x1, x8-x9, x18}, -32
# CHECK-DIS: b972 <unknown>
# CHECK: :[[@LINE+1]]:22: error: invalid register
qc.cm.pushfp {ra, s0-s2}, -32
