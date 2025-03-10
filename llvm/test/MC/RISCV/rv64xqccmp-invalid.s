# RUN: not llvm-mc -triple=riscv64 -mattr=experimental-xqccmp -M no-aliases -show-encoding < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-ERROR %s

# CHECK-ERROR: error: invalid operand for instruction
qc.cm.mvsa01 a1, a2

# CHECK-ERROR: error: rs1 and rs2 must be different
qc.cm.mvsa01 s0, s0

# CHECK-ERROR: error: invalid operand for instruction
qc.cm.mva01s a1, a2

# CHECK-ERROR: error: invalid register list, {ra, s0-s10} or {x1, x8-x9, x18-x26} is not supported
qc.cm.popretz {ra, s0-s10}, 112

# CHECK-ERROR: error: stack adjustment for register list must be a multiple of 16 bytes in the range [32, 80]
qc.cm.popretz {ra, s0-s1}, 112

# CHECK-ERROR: error: stack adjustment for register list must be a multiple of 16 bytes in the range [-64, -16]
qc.cm.push {ra}, 16

# CHECK-ERROR: error: stack adjustment for register list must be a multiple of 16 bytes in the range [-64, -16]
qc.cm.pushfp {ra, s0}, 16

# CHECK-ERROR: error: stack adjustment for register list must be a multiple of 16 bytes in the range [32, 80]
qc.cm.pop {ra, s0-s1}, -32

# CHECK-ERROR: error: stack adjustment for register list must be a multiple of 16 bytes in the range [-64, -16]
qc.cm.push {ra}, -15

# CHECK-ERROR: error: stack adjustment for register list must be a multiple of 16 bytes in the range [-64, -16]
qc.cm.push {ra, s0}, -15

# CHECK-ERROR: error: stack adjustment for register list must be a multiple of 16 bytes in the range [32, 80]
qc.cm.pop {ra, s0-s1}, -33
