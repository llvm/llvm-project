# RUN: llvm-mc -triple i386-unknown-linux %s -filetype=obj -o - | \
# RUN:  not llvm-dwarfdump -debug-frame - 2>&1 | \
# RUN:  FileCheck %s

# CHECK: .debug_frame contents:
# CHECK: 00000000 0000000c 12345678 FDE cie=<invalid offset> pc=00010000...00010010
# CHECK: error: decoding the FDE opcodes into rows failed
# CHECK: error: unable to get CIE for FDE at offset 0x0

    .section .debug_frame,"",@progbits
    .long   .LFDE0end-.LFDE0id  # Length
.LFDE0id:
    .long   0x12345678          # CIE pointer (invalid)
    .long   0x00010000          # Initial location
    .long   0x00000010          # Address range
.LFDE0end:
