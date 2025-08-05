# RUN: not llvm-mc -triple=riscv32 -mattr=+experimental-xqccmp -M no-aliases -show-encoding < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-ERROR %s

# CHECK-ERROR: :[[@LINE+1]]:14: error: invalid operand for instruction
qc.cm.mvsa01 a1, a2

# CHECK-ERROR: :[[@LINE+1]]:14: error: rs1 and rs2 must be different
qc.cm.mvsa01 s0, s0

# CHECK-ERROR: :[[@LINE+1]]:14: error: invalid operand for instruction
qc.cm.mva01s a1, a2

# CHECK-ERROR: :[[@LINE+1]]:15: error: invalid register list, '{ra, s0-s10}' or '{x1, x8-x9, x18-x26}' is not supported
qc.cm.popretz {ra, s0-s10}, 112

# CHECK-ERROR: :[[@LINE+1]]:28: error: stack adjustment for register list must be a multiple of 16 bytes in the range [16, 64]
qc.cm.popretz {ra, s0-s1}, 112

# CHECK-ERROR: :[[@LINE+1]]:18: error: stack adjustment for register list must be a multiple of 16 bytes in the range [-64, -16]
qc.cm.push {ra}, 16

# CHECK-ERROR: :[[@LINE+1]]:24: error: stack adjustment for register list must be a multiple of 16 bytes in the range [-64, -16]
qc.cm.pushfp {ra, s0}, 16

# CHECK-ERROR: :[[@LINE+1]]:24: error: stack adjustment for register list must be a multiple of 16 bytes in the range [16, 64]
qc.cm.pop {ra, s0-s1}, -32

# CHECK-ERROR: :[[@LINE+1]]:18: error: stack adjustment for register list must be a multiple of 16 bytes in the range [-64, -16]
qc.cm.push {ra}, -8

# CHECK-ERROR: :[[@LINE+1]]:24: error: stack adjustment for register list must be a multiple of 16 bytes in the range [-64, -16]
qc.cm.pushfp {ra, s0}, -12

# CHECK-ERROR: :[[@LINE+1]]:24: error: stack adjustment for register list must be a multiple of 16 bytes in the range [16, 64]
qc.cm.pop {ra, s0-s1}, -40

# CHECK-ERROR: :[[@LINE+1]]:14: error: register list must include 's0' or 'x8'
qc.cm.pushfp {ra}, -16

# CHECK-ERROR: :[[@LINE+1]]:12: error: operand must be {ra [, s0[-sN]]} or {x1 [, x8[-x9][, x18[-xN]]]}
qc.cm.push x1, -16

# CHECK-ERROR: :[[@LINE+1]]:14: error: operand must be {ra, s0[-sN]} or {x1, x8[-x9][, x18[-xN]]}
qc.cm.pushfp x1, -16
