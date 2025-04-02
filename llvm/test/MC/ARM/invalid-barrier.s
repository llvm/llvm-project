@ RUN: not llvm-mc -triple=armv7   -show-encoding < %s 2>&1 | FileCheck %s
@ RUN: not llvm-mc -triple=thumbv7 -show-encoding < %s 2>&1 | FileCheck %s

@------------------------------------------------------------------------------
@ DMB
@------------------------------------------------------------------------------
        dmb #0x10
@ CHECK: [[@LINE-1]]:{{.*}}: error: immediate value out of range
        dmb imaginary_scope
@ CHECK: [[@LINE-1]]:{{.*}}: error: invalid operand for instruction
        dmb [r0]
@ CHECK: [[@LINE-1]]:{{.*}}: error: expected an immediate or barrier type
        dmb [], @, -=_+
@ CHECK: [[@LINE-1]]:{{.*}}: error: expected an immediate or barrier type
        dmb ,,,,,
@ CHECK: [[@LINE-1]]:{{.*}}: error: expected an immediate or barrier type
        dmb 3.141
@ CHECK: [[@LINE-1]]:{{.*}}: error: expected an immediate or barrier type

@------------------------------------------------------------------------------
@ DSB
@------------------------------------------------------------------------------
        dsb #0x10
@ CHECK: [[@LINE-1]]:{{.*}}: error: immediate value out of range
        dsb imaginary_scope
@ CHECK: [[@LINE-1]]:{{.*}}: error: invalid operand for instruction
        dsb [r0]
@ CHECK: [[@LINE-1]]:{{.*}}: error: expected an immediate or barrier type
        dsb [], @, -=_+
@ CHECK: [[@LINE-1]]:{{.*}}: error: expected an immediate or barrier type
        dsb ,,,,,
@ CHECK: [[@LINE-1]]:{{.*}}: error: expected an immediate or barrier type
        dsb 3.141
@ CHECK: [[@LINE-1]]:{{.*}}: error: expected an immediate or barrier type

@------------------------------------------------------------------------------
@ ISB
@------------------------------------------------------------------------------
        isb #0x1f
@ CHECK: [[@LINE-1]]:{{.*}}: error: immediate value out of range
        isb imaginary_domain
@ CHECK: [[@LINE-1]]:{{.*}}: error: invalid operand for instruction
        isb [r0]
@ CHECK: [[@LINE-1]]:{{.*}}: error: expected an immediate or barrier type
        isb [], @, -=_+
@ CHECK: [[@LINE-1]]:{{.*}}: error: expected an immediate or barrier type
        isb ,,,,,
@ CHECK: [[@LINE-1]]:{{.*}}: error: expected an immediate or barrier type
        isb 3.141
@ CHECK: [[@LINE-1]]:{{.*}}: error: expected an immediate or barrier type
