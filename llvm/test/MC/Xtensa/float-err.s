# RUN: not llvm-mc %s -triple=xtensa -filetype=asm 2>&1 | FileCheck --implicit-check-not=error: %s

ceil.s	a2, f3, 17
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

const.s	f3, 18
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

float.s	f2, a3, 16
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

ufloat.s	f2, a3, 25
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: expected immediate in range [0, 15]

floor.s	a2, f3, 17
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

lsi f2, a3, 4099
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

lsip f2, a3, 4099
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

round.s	a2, f3, 20
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

ssi f2, a3, 5000
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

ssip f2, a3, 5001
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

trunc.s	a2, f3, 21
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: expected immediate in range [0, 15]

utrunc.s	a2, f3, 19
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: expected immediate in range [0, 15]
