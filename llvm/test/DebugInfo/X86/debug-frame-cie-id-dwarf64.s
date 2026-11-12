# RUN: llvm-mc -triple x86_64-unknown-linux %s -filetype=obj -o - | \
# RUN:   not llvm-dwarfdump -debug-frame - 2>&1 | \
# RUN:   FileCheck %s

# CHECK: 00000000 {{.*}} FDE
# CHECK: error: decoding the FDE opcodes into rows failed
# CHECK: error: unable to get CIE for FDE at offset 0x0

        .section .debug_frame,"",@progbits
## This FDE was formerly wrongly interpreted as a CIE because its CIE pointer
## is similar to DWARF32 CIE id.
        .long 0xffffffff        # DWARF64 mark
        .quad .Lend - .LCIEptr  # Length
.LCIEptr:
        .quad 0xffffffff        # CIE pointer
        .quad 0x1111abcd        # Initial location
        .quad 0x00010000        # Address range
.Lend:
