## Check we do not support .eh_frame sections of versions greater than 1.

# RUN: llvm-mc -triple x86_64-unknown-linux %s -filetype=obj -o %t
# RUN: not llvm-dwarfdump -debug-frame %t 2>&1 | FileCheck %s

# CHECK: unsupported CIE version: 2

.section .eh_frame,"a",@unwind
 .long .Lend - .LCIEptr ## Length
.LCIEptr:
 .long 0x00000000       ## CIE ID
 .byte 2                ## Version
.Lend:
