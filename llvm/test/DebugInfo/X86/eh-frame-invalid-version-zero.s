## Check we do not support .eh_frame sections of version 0.

# RUN: llvm-mc -triple x86_64-unknown-linux %s -filetype=obj -o %t
# RUN: not llvm-dwarfdump -debug-frame %t 2>&1 | FileCheck %s

# CHECK: unsupported CIE version: 0

.section .eh_frame,"a",@unwind
 .long .Lend - .LCIEptr ## Length
.LCIEptr:
 .long 0x00000000       ## CIE ID
 .byte 0                ## Version
.Lend:
