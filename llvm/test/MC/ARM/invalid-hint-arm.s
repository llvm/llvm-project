@ RUN: not llvm-mc -triple=armv7-apple-darwin -mcpu=cortex-a8 < %s 2>&1 | FileCheck %s

hint #240
hint #1000

@ CHECK: error: operand must be an immediate in the range [0,239]
@ CHECK: error: operand must be an immediate in the range [0,239]

