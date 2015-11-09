@ RUN: not llvm-mc -triple arm-gnueabi-linux -filetype asm %s 2>&1 | FileCheck %s

        .arch	armv6
        dsb
@ CHECK: error: instruction requires: data-barriers

        .arch   armv7
        dsb
@ CHECK-NOT: error: instruction requires: data-barriers

        .arch   invalid_architecture_name
@ CHECK: error: Unknown arch name
