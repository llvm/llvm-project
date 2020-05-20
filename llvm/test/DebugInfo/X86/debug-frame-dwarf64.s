# RUN: llvm-mc -triple x86_64 %s -filetype=obj -o - | \
# RUN:   llvm-dwarfdump -debug-frame - | \
# RUN:   FileCheck %s

# CHECK:      00000000 0000000000000010 ffffffffffffffff CIE
# CHECK-NEXT:   Version:               4
# CHECK-NEXT:   Augmentation:          ""
# CHECK-NEXT:   Address size:          8
# CHECK-NEXT:   Segment desc size:     0
# CHECK-NEXT:   Code alignment factor: 1
# CHECK-NEXT:   Data alignment factor: -8
# CHECK-NEXT:   Return address column: 16
# CHECK-EMPTY:
# CHECK-NEXT:   DW_CFA_nop:

# CHECK:      0000001c 0000000000000018 0000000000000000 FDE cie=00000000 pc=00112233...00122233

    .section .debug_frame, "", @progbits
.LCIE:
    .long 0xffffffff            # DWARF64 mark
    .quad .LCIEend-.LCIEid      # Length
.LCIEid:
    .quad 0xffffffffffffffff    # CIE id
    .byte 4                     # Version
    .asciz ""                   # Augmentation
    .byte 8                     # Address size
    .byte 0                     # Segment selector size
    .uleb128 1                  # Code alignment factor
    .sleb128 -8                 # Data alignment factor
    .uleb128 16                 # Return address register
    .byte 0                     # DW_CFA_nop
.LCIEend:
.LFDE:
    .long 0xffffffff            # DWARF64 mark
    .quad .LFDEend-.LFDEcieptr  # Length
.LFDEcieptr:
    .quad .LCIE-.debug_frame    # CIE pointer
    .quad 0x00112233            # Initial location
    .quad 0x00010000            # Address range
.LFDEend:
