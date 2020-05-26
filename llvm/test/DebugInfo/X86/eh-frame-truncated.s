## Check we report a proper error when the content
## of the .eh_frame section is truncated.

# RUN: llvm-mc -triple x86_64 %s -filetype=obj -o %t
# RUN: not llvm-dwarfdump -debug-frame %t 2>&1 | FileCheck %s

# CHECK: error: unexpected end of data at offset 0x4

.section .eh_frame,"a",@unwind
.long 0xFF ## Length
