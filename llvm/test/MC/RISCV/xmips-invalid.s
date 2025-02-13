# RUN: not llvm-mc -triple=riscv64 < %s 2>&1 | FileCheck %s -check-prefixes=CHECK-FEATURE
# RUN: not llvm-mc -triple=riscv64 -mattr=+xmipslsp,+xmipscmove < %s 2>&1 | FileCheck %s

mips.ccmov x0, x1, 0x10
# CHECK: error: invalid operand for instruction

mips.ccmov x10
# CHECK: error: too few operands for instruction

mips.ccmov	s0, s1, s2, s3
# CHECK-FEATURE: error: instruction requires the following: 'Xmipscmove' ('mips.ccmov' instruction)

mips.lwp x10, x11
# CHECK: error: too few operands for instruction

mips.ldp x9, 0x20
# CHECK: error: invalid operand for instruction

mips.lwp x11, x12, 0(x13)
# CHECK-FEATURE: error: instruction requires the following: 'Xmipslsp' (load and store pair instructions)

mips.swp x18, x19, 8(x2)
# CHECK-FEATURE: error: instruction requires the following: 'Xmipslsp' (load and store pair instructions)

mips.sdp 0x10, x3, 12(x4)
# CHECK: error: invalid operand for instruction
