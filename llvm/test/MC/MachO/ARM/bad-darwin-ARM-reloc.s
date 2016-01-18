@ RUN: not llvm-mc -n -triple armv7-apple-darwin10 %s -filetype=obj -o - 2> %t.err > %t
@ RUN: FileCheck --check-prefix=CHECK-ERROR < %t.err %s
@ rdar://15586725
.text
    ldr r3, L___fcommon
.section myseg, mysect
L___fcommon: 
    .word 0
@ CHECK-ERROR: unsupported relocation on symbol
