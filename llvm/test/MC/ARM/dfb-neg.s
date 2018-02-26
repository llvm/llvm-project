@ RUN: not llvm-mc -triple armv8-none-eabi -mcpu=cortex-r52 -mattr=-dfb -show-encoding < %s 2>&1 | FileCheck %s
@ RUN: not llvm-mc -triple thumbv8-none-eabi -mcpu=cortex-r52 -mattr=-dfb -show-encoding < %s 2>&1 | FileCheck %s

        dfb
@ CHECK: error: instruction requires: full-data-barrier

        dfb sy
        dfb #0
@ CHECK: error: invalid instruction
@ CHECK: error: invalid instruction
