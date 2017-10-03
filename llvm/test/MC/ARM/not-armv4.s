@ RUN: not llvm-mc < %s -triple armv4-unknown-unknown -show-encoding 2>&1 | FileCheck %s

@ PR18524
@ CHECK: instruction requires: armv5t
clz r4,r9

@ CHECK: instruction requires: armv6t2
rbit r4,r9

@ CHECK: error: instruction requires: armv6t2
movw r4,#0x1234
@ CHECK: error: instruction requires: armv6t2
mov  r4,#0x1234
