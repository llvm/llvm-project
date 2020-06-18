# RUN: llvm-mc -triple i386-unknown-linux %s -filetype=obj -o - | \
# RUN:  llvm-dwarfdump -debug-frame - | \
# RUN:  FileCheck %s

# CHECK: .debug_frame contents:
# CHECK: 00000000 0000000c 12345678 FDE cie=<invalid offset> pc=00010000...00010010

    .section .debug_frame,"",@progbits
    .long   .LFDE0end-.LFDE0id  # Length
.LFDE0id:
    .long   0x12345678          # CIE pointer (invalid)
    .long   0x00010000          # Initial location
    .long   0x00000010          # Address range
.LFDE0end:
