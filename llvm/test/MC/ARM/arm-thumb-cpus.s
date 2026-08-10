@ RUN: not llvm-mc -show-encoding -triple=arm-eabi < %s 2>&1 \
@ RUN:  | FileCheck %s --check-prefix=CHECK-ARM-ONLY

@ RUN: llvm-mc -show-encoding -triple=armv4t < %s 2>&1 \
@ RUN:  | FileCheck %s --check-prefix=CHECK-ARM-THUMB

@ RUN: llvm-mc -show-encoding -triple=arm-eabi -mcpu=cortex-a15 < %s 2>&1 \
@ RUN:  | FileCheck %s --check-prefix=CHECK-ARM-THUMB

@ RUN: not llvm-mc -show-encoding -triple=arm-eabi -mcpu=cortex-m3 < %s 2>&1 \
@ RUN:  | FileCheck %s --check-prefix=CHECK-THUMB-ONLY

@ RUN: not llvm-mc -show-encoding -triple=armv7m-eabi < %s 2>&1 \
@ RUN:  | FileCheck %s --check-prefix=CHECK-THUMB-ONLY

@ RUN: not llvm-mc -show-encoding -triple=armv6m-eabi < %s 2>&1 \
@ RUN:  | FileCheck %s --check-prefix=CHECK-THUMB-ONLY

@ RUN: not llvm-mc -show-encoding -triple=armv6sm-eabi < %s 2>&1 \
@ RUN:  | FileCheck %s --check-prefix=CHECK-THUMB-ONLY

@ RUN: not llvm-mc -show-encoding -triple=thumbv7-win32-gnu < %s 2>&1 \
@ RUN:  | FileCheck %s --check-prefix=CHECK-THUMB-ONLY

        @ Make sure correct diagnostics are given for CPUs without support for
        @ one or other of the execution states.
        .thumb
        .arm
        .code 16
        .code 32
@ CHECK-ARM-THUMB-NOT: target does not support

@ CHECK-ARM-ONLY: target does not support Thumb mode
@ CHECK-ARM-ONLY: target does not support Thumb mode

@ CHECK-THUMB-ONLY: target does not support ARM mode
@ CHECK-THUMB-ONLY: target does not support ARM mode
