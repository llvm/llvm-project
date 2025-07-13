# RUN: not llvm-mc -triple=riscv32 -mattr=zcmp -M no-aliases -show-encoding < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-ERROR %s

# CHECK-ERROR: :[[@LINE+1]]:11: error: invalid operand for instruction
cm.mvsa01 a1, a2

# CHECK-ERROR: :[[@LINE+1]]:11: error: rs1 and rs2 must be different
cm.mvsa01 s0, s0

# CHECK-ERROR: :[[@LINE+1]]:11: error: invalid operand for instruction
cm.mva01s a1, a2

# CHECK-ERROR: :[[@LINE+1]]:12: error: invalid register list, '{ra, s0-s10}' or '{x1, x8-x9, x18-x26}' is not supported
cm.popretz {ra, s0-s10}, 112

# CHECK-ERROR: :[[@LINE+1]]:25: error: stack adjustment for register list must be a multiple of 16 bytes in the range [16, 64]
cm.popretz {ra, s0-s1}, 112

# CHECK-ERROR: :[[@LINE+1]]:15: error: stack adjustment for register list must be a multiple of 16 bytes in the range [-64, -16]
cm.push {ra}, 16

# CHECK-ERROR: :[[@LINE+1]]:21: error: stack adjustment for register list must be a multiple of 16 bytes in the range [16, 64]
cm.pop {ra, s0-s1}, -32

# CHECK-ERROR: :[[@LINE+1]]:15: error: stack adjustment for register list must be a multiple of 16 bytes in the range [-64, -16]
cm.push {ra}, -8

# CHECK-ERROR: :[[@LINE+1]]:9: error: register list must start from 'ra' or 'x1'
cm.pop {s0}, -40

# CHECK-ERROR: :[[@LINE+1]]:13: error: register must be 's0'
cm.pop {ra, t1}, -40

# CHECK-ERROR: :[[@LINE+1]]:16: error: register must be in the range 's1' to 's11'
cm.pop {ra, s0-t1}, -40

# CHECK-ERROR: :[[@LINE+1]]:20: error: register must be 'x18'
cm.pop {x1, x8-x9, x28}, -40

# CHECK-ERROR: :[[@LINE+1]]:24: error: register must be in the range 'x19' to 'x27'
cm.pop {x1, x8-x9, x18-x28}, -40

# CHECK-ERROR: :[[@LINE+1]]:24: error: register must be in the range 'x19' to 'x27'
cm.pop {x1, x8-x9, x18-x17}, -40

# CHECK-ERROR: :[[@LINE+1]]:16: error: register must be 'x9'
cm.pop {x1, x8-f8, x18-x17}, -40

# CHECK-ERROR: :[[@LINE+1]]:9: error: operand must be {ra [, s0[-sN]]} or {x1 [, x8[-x9][, x18[-xN]]]}
cm.push x1, -16

# CHECK-ERROR: :[[@LINE+1]]:16: error: register must be in the range 's1' to 's11'
cm.pop {ra, s0-f8}, -40
