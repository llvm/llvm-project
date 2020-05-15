# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: llvm-dwarfdump -debug-loc %t.o | FileCheck %s

# This checks that we do not try to interpret DW_OP_call_ref if it is
# encountered in a location table.

# CHECK: .debug_loc contents:
# CHECK-NEXT: 0x00000000:
# CHECK-NEXT:   (0x0000000000000000, 0x0000000000000015): <decoding error> 9a ff 00 00 00

    .section .debug_loc, "", @progbits
    .quad 0                         # Beginning address offset
    .quad 0x15                      # Ending address offset
    .short .LDescrEnd-.LDescrBegin  # Location description length
.LDescrBegin:
    .byte 0x9a                      # DW_OP_call_ref
    .long 0xff
.LDescrEnd:
    .quad 0, 0                      # EOL entry

# A dummy CU to provide the parser of .debug_loc with the address size.
    .section .debug_info,"",@progbits
    .long .LCUEnd-.LCUBegin         # Length of Unit
.LCUBegin:
    .short 4                        # DWARF version number
    .long 0                         # Offset Into Abbrev. Section
    .byte 8                         # Address Size
.LCUEnd:
