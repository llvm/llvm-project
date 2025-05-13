@ RUN: llvm-mc -triple=armv7 -mcpu=cortex-m3 -show-encoding < %s 2>&1 | FileCheck -check-prefix M3-ARM %s
@ RUN: llvm-mc -triple=thumbv7 -mcpu=cortex-m3 -show-encoding < %s 2>&1 | FileCheck -check-prefix M3-THUMB %s

@ RUN: llvm-mc -triple=thumbv7 -mcpu=cortex-a15 -show-encoding < %s 2>&1 | FileCheck -check-prefix A15-THUMB %s
@ RUN: llvm-mc -triple=thumbv7 -mcpu=cortex-a15 -show-encoding < %s 2>&1 | FileCheck -check-prefix A15-THUMB %s

@ RUN: llvm-mc -triple=armv7 -mcpu=cortex-a15 -mattr=-hwdiv -show-encoding < %s 2>&1 | FileCheck -check-prefix A15-ARM-NOTHUMBHWDIV %s
@ RUN: llvm-mc -triple=thumbv7 -mcpu=cortex-a15 -mattr=-hwdiv-arm -show-encoding < %s 2>&1 | FileCheck -check-prefix A15-THUMB-NOARMHWDIV %s

@ RUN: llvm-mc -triple=armv8 -show-encoding < %s 2>&1 | FileCheck -check-prefix ARMV8 %s
@ RUN: llvm-mc -triple=thumbv8 -show-encoding < %s 2>&1 | FileCheck -check-prefix THUMBV8 %s

@ RUN: llvm-mc -triple=armv8 -mattr=-hwdiv -show-encoding < %s 2>&1 | FileCheck -check-prefix ARMV8-NOTHUMBHWDIV %s
@ RUN: llvm-mc -triple=thumbv8 -mattr=-hwdiv-arm -show-encoding < %s 2>&1 | FileCheck -check-prefix THUMBV8-NOARMHWDIV %s

        sdiv  r1, r2
        udiv  r3, r4

@ M3-ARM:               sdiv   r1, r1, r2               @ encoding: [0x91,0xfb,0xf2,0xf1]
@ M3-ARM:               udiv   r3, r3, r4               @ encoding: [0xb3,0xfb,0xf4,0xf3]
@ M3-THUMB:             sdiv   r1, r1, r2               @ encoding: [0x91,0xfb,0xf2,0xf1]
@ M3-THUMB:             udiv   r3, r3, r4               @ encoding: [0xb3,0xfb,0xf4,0xf3]

@ A15-ARM:              sdiv   r1, r1, r2               @ encoding: [0x11,0xf2,0x11,0xe7]
@ A15-ARM:              udiv   r3, r3, r4               @ encoding: [0x13,0xf4,0x33,0xe7]
@ A15-THUMB:            sdiv   r1, r1, r2               @ encoding: [0x91,0xfb,0xf2,0xf1]
@ A15-THUMB:            udiv   r3, r3, r4               @ encoding: [0xb3,0xfb,0xf4,0xf3]

@ A15-ARM-NOTHUMBHWDIV: sdiv    r1, r1, r2              @ encoding: [0x11,0xf2,0x11,0xe7]
@ A15-ARM-NOTHUMBHWDIV: udiv    r3, r3, r4              @ encoding: [0x13,0xf4,0x33,0xe7] 
@ A15-THUMB-NOARMHWDIV: sdiv    r1, r1, r2              @ encoding: [0x91,0xfb,0xf2,0xf1]
@ A15-THUMB-NOARMHWDIV: udiv    r3, r3, r4              @ encoding: [0xb3,0xfb,0xf4,0xf3]

@ ARMV8:                sdiv    r1, r1, r2              @ encoding: [0x11,0xf2,0x11,0xe7]
@ ARMV8:                udiv    r3, r3, r4              @ encoding: [0x13,0xf4,0x33,0xe7]
@ THUMBV8:              sdiv    r1, r1, r2              @ encoding: [0x91,0xfb,0xf2,0xf1] 
@ THUMBV8:              udiv    r3, r3, r4              @ encoding: [0xb3,0xfb,0xf4,0xf3]

@ ARMV8-NOTHUMBHWDIV:   sdiv    r1, r1, r2              @ encoding: [0x11,0xf2,0x11,0xe7] 
@ ARMV8-NOTHUMBHWDIV:   udiv    r3, r3, r4              @ encoding: [0x13,0xf4,0x33,0xe7]
@ THUMBV8-NOARMHWDIV:   sdiv    r1, r1, r2              @ encoding: [0x91,0xfb,0xf2,0xf1]
@ THUMBV8-NOARMHWDIV:   udiv    r3, r3, r4              @ encoding: [0xb3,0xfb,0xf4,0xf3]
