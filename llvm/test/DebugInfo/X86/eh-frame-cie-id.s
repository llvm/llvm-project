# RUN: llvm-mc -triple x86_64-unknown-linux %s -filetype=obj -o %t
# RUN: not llvm-dwarfdump -debug-frame %t 2>&1 | FileCheck %s

# CHECK: parsing FDE data at 0x0 failed due to missing CIE

        .section .eh_frame,"a",@unwind
## This FDE was formerly wrongly interpreted as a CIE because its CIE pointer
## is similar to CIE id of a .debug_frame FDE.
        .long .Lend - .LCIEptr  # Length
.LCIEptr:
        .long 0xffffffff        # CIE pointer
        .quad 0x1111abcd        # Initial location
        .quad 0x00010000        # Address range
.Lend:
