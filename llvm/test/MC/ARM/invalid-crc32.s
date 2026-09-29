@ RUN: not llvm-mc -triple=armv8 -show-encoding < %s 2>&1 | FileCheck %s
@ RUN: not llvm-mc -triple=thumbv8 -show-encoding < %s 2>&1 | FileCheck %s

        crc32cbeq  r0, r1, r2
        crc32bne   r0, r1, r2
        crc32chcc  r0, r1, r2
        crc32hpl   r0, r1, r2
        crc32cwgt  r0, r1, r2
        crc32wle   r0, r1, r2

@ CHECK: error: instruction 'crc32cb' is not predicable, but condition code specified
@ CHECK: error: instruction 'crc32b' is not predicable, but condition code specified
@ CHECK: error: instruction 'crc32ch' is not predicable, but condition code specified
@ CHECK: error: instruction 'crc32h' is not predicable, but condition code specified
@ CHECK: error: instruction 'crc32cw' is not predicable, but condition code specified
@ CHECK: error: instruction 'crc32w' is not predicable, but condition code specified
