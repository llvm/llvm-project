@ RUN: not llvm-mc -triple=armv8r-linux-gnu -mcpu=cortex-r52 -show-encoding < %s 2>&1 | FileCheck %s -check-prefix=CHECK-NOTZ
@ RUN: not llvm-mc -triple=armv8r-linux-gnu -mcpu=cortex-r52plus     -show-encoding < %s 2>&1 | FileCheck %s -check-prefix=CHECK-NOTZ

    smc #0xf

@ CHECK-NOTZ: error: instruction requires: TrustZone
