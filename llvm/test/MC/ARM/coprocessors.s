@ RUN: llvm-mc -triple=armv7 < %s 2> %t | FileCheck --check-prefix=ACCEPT-01234567CD --check-prefix=ACCEPT-89 --check-prefix=ACCEPT-AB --check-prefix=ACCEPT-EF %s
@ RUN: llvm-mc -triple=thumbv7 < %s 2> %t | FileCheck --check-prefix=ACCEPT-01234567CD --check-prefix=ACCEPT-89 --check-prefix=ACCEPT-AB --check-prefix=ACCEPT-EF %s
@ RUN: not llvm-mc -triple=armv8 < %s 2> %t | FileCheck --check-prefix=ACCEPT-EF %s
@ RUN: FileCheck --check-prefix=REJECT-01234567CD --check-prefix=REJECT-89 --check-prefix=REJECT-AB < %t %s
@ RUN: not llvm-mc -triple=thumbv8 < %s 2> %t | FileCheck --check-prefix=ACCEPT-EF %s
@ RUN: FileCheck --check-prefix=REJECT-01234567CD --check-prefix=REJECT-89 --check-prefix=REJECT-AB < %t %s
@ RUN: not llvm-mc -triple=thumbv8.1m.main < %s 2> %t | FileCheck --check-prefix=ACCEPT-01234567CD --check-prefix=ACCEPT-AB %s
@ RUN: FileCheck --check-prefix=REJECT-89 --check-prefix=REJECT-EF < %t %s

mrc   p0, #1, r2, c3, c4, #5
@ ACCEPT-01234567CD: mrc   p0, #1, r2, c3, c4, #5
@ REJECT-01234567CD: [[@LINE-2]]:7: error: invalid operand for instruction

mrc   p1, #1, r2, c3, c4, #5
@ ACCEPT-01234567CD: mrc   p1, #1, r2, c3, c4, #5
@ REJECT-01234567CD: [[@LINE-2]]:7: error: invalid operand for instruction

mrc   p2, #1, r2, c3, c4, #5
@ ACCEPT-01234567CD: mrc   p2, #1, r2, c3, c4, #5
@ REJECT-01234567CD: [[@LINE-2]]:7: error: invalid operand for instruction

mrc   p3, #1, r2, c3, c4, #5
@ ACCEPT-01234567CD: mrc   p3, #1, r2, c3, c4, #5
@ REJECT-01234567CD: [[@LINE-2]]:7: error: invalid operand for instruction

mrc   p4, #1, r2, c3, c4, #5
@ ACCEPT-01234567CD: mrc   p4, #1, r2, c3, c4, #5
@ REJECT-01234567CD: [[@LINE-2]]:7: error: invalid operand for instruction

mrc   p5, #1, r2, c3, c4, #5
@ ACCEPT-01234567CD: mrc   p5, #1, r2, c3, c4, #5
@ REJECT-01234567CD: [[@LINE-2]]:7: error: invalid operand for instruction

mrc   p6, #1, r2, c3, c4, #5
@ ACCEPT-01234567CD: mrc   p6, #1, r2, c3, c4, #5
@ REJECT-01234567CD: [[@LINE-2]]:7: error: invalid operand for instruction

mrc   p7, #1, r2, c3, c4, #5
@ ACCEPT-01234567CD: mrc   p7, #1, r2, c3, c4, #5
@ REJECT-01234567CD: [[@LINE-2]]:7: error: invalid operand for instruction

mrc   p8, #1, r2, c3, c4, #5
@ ACCEPT-89: mrc   p8, #1, r2, c3, c4, #5
@ REJECT-89: [[@LINE-2]]:7: error: invalid operand for instruction

mrc   p9, #1, r2, c3, c4, #5
@ ACCEPT-89: mrc   p9, #1, r2, c3, c4, #5
@ REJECT-89: [[@LINE-2]]:7: error: invalid operand for instruction

mrc   p10, #1, r2, c3, c4, #5
@ ACCEPT-AB: mrc   p10, #1, r2, c3, c4, #5
@ REJECT-AB: [[@LINE-2]]:7: error: invalid operand for instruction

mrc   p11, #1, r2, c3, c4, #5
@ ACCEPT-AB: mrc   p11, #1, r2, c3, c4, #5
@ REJECT-AB: [[@LINE-2]]:7: error: invalid operand for instruction

mrc   p12, #1, r2, c3, c4, #5
@ ACCEPT-01234567CD: mrc   p12, #1, r2, c3, c4, #5
@ REJECT-01234567CD: [[@LINE-2]]:7: error: invalid operand for instruction

mrc   p13, #1, r2, c3, c4, #5
@ ACCEPT-01234567CD: mrc   p13, #1, r2, c3, c4, #5
@ REJECT-01234567CD: [[@LINE-2]]:7: error: invalid operand for instruction

mrc   p14, #1, r2, c3, c4, #5
@ ACCEPT-EF: mrc   p14, #1, r2, c3, c4, #5
@ REJECT-EF: [[@LINE-2]]:7: error: invalid operand for instruction

mrc   p15, #1, r2, c3, c4, #5
@ ACCEPT-EF: mrc   p15, #1, r2, c3, c4, #5
@ REJECT-EF: [[@LINE-2]]:7: error: invalid operand for instruction
