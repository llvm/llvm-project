@ RUN: llvm-mc -triple armv8 -show-encoding < %s 2>&1 | FileCheck %s --check-prefix=CHECK-ARMV8
@ RUN: llvm-mc -triple thumbv8 -show-encoding < %s 2>&1 | FileCheck %s --check-prefix=CHECK-THUMBV8
@ RUN: llvm-mc -triple armv7 -show-encoding < %s 2>&1 | FileCheck %s --check-prefix=CHECK-ARMV7
@ RUN: llvm-mc -triple thumbv7 -show-encoding < %s 2>&1 | FileCheck %s --check-prefix=CHECK-THUMBV7
@ RUN: llvm-mc -triple armv6 -show-encoding < %s 2>&1 | FileCheck %s --check-prefix=CHECK-ARMV6
@ RUN: llvm-mc -triple thumbv8 -show-encoding -no-deprecated-warn < %s 2>&1 | FileCheck %s --check-prefix=CHECK-NO-WARN
setend be
@ CHECK-ARMV8: warning: deprecated
@ CHECK-THUMBV8: warning: deprecated
@ CHECK-ARMV7-NOT: warning: deprecated
@ CHECK-THUMBV7-NOT: warning: deprecated
@ CHECK-NO-WARN-NOT: warning: deprecated
mcr p15, #0, r5, c7, c5, #4
@ CHECK-ARMV8: warning: deprecated since v7, use 'isb'
@ CHECK-THUMBV8: warning: deprecated since v7, use 'isb'
@ CHECK-ARMV7: warning: deprecated since v7, use 'isb'
@ CHECK-THUMBV7: warning: deprecated since v7, use 'isb'
@ CHECK-ARMV6-NOT: warning: deprecated since v7, use 'isb'
@ CHECK-NO-WARN-NOT: warning: deprecated since v7, use 'isb'
mcr p15, #0, r5, c7, c10, #4
@ CHECK-ARMV8: warning: deprecated since v7, use 'dsb'
@ CHECK-THUMBV8: warning: deprecated since v7, use 'dsb'
@ CHECK-ARMV7: warning: deprecated since v7, use 'dsb'
@ CHECK-THUMBV7: warning: deprecated since v7, use 'dsb'
@ CHECK-ARMV6-NOT: warning: deprecated since v7, use 'dsb'
@ CHECK-NO-WARN-NOT: warning: deprecated since v7, use 'dsb'
mcr p15, #0, r5, c7, c10, #5
@ CHECK-ARMV8: warning: deprecated since v7, use 'dmb'
@ CHECK-THUMBV8: warning: deprecated since v7, use 'dmb'
@ CHECK-ARMV7: warning: deprecated since v7, use 'dmb'
@ CHECK-THUMBV7: warning: deprecated since v7, use 'dmb'
@ CHECK-ARMV6-NOT: warning: deprecated since v7, use 'dmb'
@ CHECK-NO-WARN-NOT: warning: deprecated since v7, use 'dmb'
it ge
movge r0, #4096
@ CHECK-THUMBV8-NOT: warning
@ CHECK-THUMBV7-NOT: warning
@ CHECK-NO-WARN-NOT: warning
ite ge
addge r0, r1
addlt r0, r2
@ CHECK-ARMV8-NOT: warning
@ CHECK-THUMBV8-NOT: warning
@ CHECK-THUMBV7-NOT: warning
@ CHECK-NO-WARN-NOT: warning
it ge
movge r0, pc // invalid operand
@ CHECK-THUMBV8-NOT: warning
@ CHECK-THUMBV7-NOT: warning
@ CHECK-NO-WARN-NOT: warning
it ge
revge r0, r0 // invalid instruction
@ CHECK-THUMBV8-NOT: warning
@ CHECK-THUMBV7-NOT: warning
@ CHECK-NO-WARN-NOT: warning
it ge
clzge r0, r0 // only has 32-bit form
@ CHECK-THUMBV8-NOT: warning
@ CHECK-THUMBV7-NOT: warning
@ CHECK-NO-WARN-NOT: warning
