# RUN: llvm-mc -g -filetype=obj -triple x86_64-pc-linux-gnu %s -o %t.64
# RUN: llvm-dwarfdump -debug-info %t.64 | FileCheck -check-prefix=DEFAULTABI %s

# RUN: llvm-mc -g -filetype=obj -triple x86_64-pc-linux-gnux32 %s -o %t.32
# RUN: llvm-dwarfdump -debug-info %t.32 | FileCheck -check-prefix=X32ABI %s

# This test checks the dwarf info section emitted to the output object by the
# assembler, looking at the difference between the x32 ABI and default x86-64
# ABI.

# DEFAULTABI: addr_size = 0x08
# X32ABI: addr_size = 0x04

.globl _bar
_bar:
        movl    $0, %eax
L1:     leave
        ret
_foo:
_baz:
        nop
.data
_x:     .long 1

