# RUN: not llvm-mc -filetype=obj -triple=i386 %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error:

.intel_syntax noprefix

.code16

jmp foo+0x10001

# CHECK: <unknown>:0: error: value of 65536 is too large for field of 2 bytes
jmp foo+0x10002
