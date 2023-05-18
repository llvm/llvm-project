@ RUN: not llvm-mc -triple=armv8 < %s 2>&1 | FileCheck %s

swp r0, r1, [r2]
@ CHECK: instruction requires: armv7 or earlier

swpb r0, r1, [r2]
@ CHECK: instruction requires: armv7 or earlier
