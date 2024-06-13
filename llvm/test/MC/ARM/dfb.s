@ RUN: llvm-mc -triple armv8-none-eabi -mcpu=cortex-r52 -show-encoding < %s | FileCheck %s --check-prefix=CHECK-ARM
@ RUN: llvm-mc -triple thumbv8-none-eabi -mcpu=cortex-r52 -show-encoding < %s | FileCheck %s --check-prefix=CHECK-THUMB
@ RUN: llvm-mc -triple armv8-none-eabi -mcpu=cortex-r52plus -show-encoding < %s | FileCheck %s --check-prefix=CHECK-ARM
@ RUN: llvm-mc -triple thumbv8-none-eabi -mcpu=cortex-r52plus -show-encoding < %s | FileCheck %s --check-prefix=CHECK-THUMB

        dfb
@ CHECK-ARM:   dfb                             @ encoding: [0x4c,0xf0,0x7f,0xf5]
@ CHECK-THUMB: dfb                             @ encoding: [0xbf,0xf3,0x4c,0x8f]
