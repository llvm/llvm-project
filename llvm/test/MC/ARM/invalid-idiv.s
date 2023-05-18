@ RUN: not llvm-mc -triple=armv7 -mcpu=cortex-a15 -mattr=-hwdiv-arm < %s 2> %t
@ RUN: FileCheck --check-prefix=ARM-A15 < %t %s
@ RUN: not llvm-mc -triple=thumbv7 -mcpu=cortex-a15 -mattr=-hwdiv < %s 2> %t
@ RUN: FileCheck --check-prefix=THUMB-A15 < %t %s
@ RUN: not llvm-mc -triple=armv7 < %s 2> %t
@ RUN: FileCheck --check-prefix=ARM < %t %s
@ RUN: not llvm-mc -triple=thumbv7 < %s 2> %t
@ RUN: FileCheck --check-prefix=THUMB < %t %s

        sdiv  r1, r2, r3
        udiv  r3, r4, r5
@ ARM-A15: note: instruction requires: divide in ARM
@ ARM-A15: note: instruction requires: thumb
@ ARM-A15: sdiv r1, r2, r3
@ ARM-A15: note: instruction requires: divide in ARM
@ ARM-A15: note: instruction requires: thumb
@ ARM-A15: udiv r3, r4, r5
@ THUMB-A15: note: instruction requires: arm-mode 
@ THUMB-A15: note: instruction requires: divide in THUMB
@ THUMB-A15: sdiv r1, r2, r3
@ THUMB-A15: note: instruction requires: arm-mode 
@ THUMB-A15: note: instruction requires: divide in THUMB
@ THUMB-A15: udiv r3, r4, r5

@ ARM: error: instruction requires: divide in ARM
@ ARM: sdiv r1, r2, r3
@ ARM: error: instruction requires: divide in ARM
@ ARM: udiv r3, r4, r5
@ THUMB: error: instruction requires: divide in THUMB
@ THUMB: sdiv r1, r2, r3
@ THUMB: error: instruction requires: divide in THUMB
@ THUMB: udiv r3, r4, r5
