@ RUN: llvm-mc -triple=armv7 -mcpu=cortex-a15 -show-encoding < %s 2>&1 | FileCheck -check-prefix A15-ARM %s
@ RUN: llvm-mc -triple=thumbv7 -mcpu=cortex-a15 -show-encoding < %s 2>&1 | FileCheck -check-prefix A15-THUMB %s

@ RUN: llvm-mc -triple=armv7 -mcpu=cortex-a15 -mattr=-hwdiv -show-encoding < %s 2>&1 | FileCheck -check-prefix A15-ARM-NOTHUMBHWDIV %s
@ RUN: llvm-mc -triple=thumbv7 -mcpu=cortex-a15 -mattr=-hwdiv-arm -show-encoding < %s 2>&1 | FileCheck -check-prefix A15-THUMB-NOARMHWDIV %s

@ RUN: llvm-mc -triple=armv8 -show-encoding < %s 2>&1 | FileCheck -check-prefix ARMV8 %s
@ RUN: llvm-mc -triple=thumbv8 -show-encoding < %s 2>&1 | FileCheck -check-prefix THUMBV8 %s

@ RUN: llvm-mc -triple=armv8 -mattr=-hwdiv -show-encoding < %s 2>&1 | FileCheck -check-prefix ARMV8-NOTHUMBHWDIV %s
@ RUN: llvm-mc -triple=thumbv8 -mattr=-hwdiv-arm -show-encoding < %s 2>&1 | FileCheck -check-prefix THUMBV8-NOARMHWDIV %s

        sdiv  r1, r2, r3
        udiv  r3, r4, r5
@ A15-ARM:              sdiv   r1, r2, r3               @ encoding: [0x12,0xf3,0x11,0xe7]
@ A15-ARM:              udiv   r3, r4, r5               @ encoding: [0x14,0xf5,0x33,0xe7]
@ A15-THUMB:            sdiv   r1, r2, r3               @ encoding: [0x92,0xfb,0xf3,0xf1]
@ A15-THUMB:            udiv   r3, r4, r5               @ encoding: [0xb4,0xfb,0xf5,0xf3]

@ A15-ARM-NOTHUMBHWDIV: sdiv    r1, r2, r3              @ encoding: [0x12,0xf3,0x11,0xe7]
@ A15-ARM-NOTHUMBHWDIV: udiv    r3, r4, r5              @ encoding: [0x14,0xf5,0x33,0xe7]
@ A15-THUMB-NOARMHWDIV: sdiv    r1, r2, r3              @ encoding: [0x92,0xfb,0xf3,0xf1]
@ A15-THUMB-NOARMHWDIV: udiv    r3, r4, r5              @ encoding: [0xb4,0xfb,0xf5,0xf3]

@ ARMV8:                sdiv    r1, r2, r3              @ encoding: [0x12,0xf3,0x11,0xe7]
@ ARMV8:                udiv    r3, r4, r5              @ encoding: [0x14,0xf5,0x33,0xe7]
@ THUMBV8:              sdiv    r1, r2, r3              @ encoding: [0x92,0xfb,0xf3,0xf1]
@ THUMBV8:              udiv    r3, r4, r5              @ encoding: [0xb4,0xfb,0xf5,0xf3]

@ ARMV8-NOTHUMBHWDIV:   sdiv    r1, r2, r3              @ encoding: [0x12,0xf3,0x11,0xe7]
@ ARMV8-NOTHUMBHWDIV:   udiv    r3, r4, r5              @ encoding: [0x14,0xf5,0x33,0xe7]
@ THUMBV8-NOARMHWDIV:   sdiv    r1, r2, r3              @ encoding: [0x92,0xfb,0xf3,0xf1]
@ THUMBV8-NOARMHWDIV:   udiv    r3, r4, r5              @ encoding: [0xb4,0xfb,0xf5,0xf3]
