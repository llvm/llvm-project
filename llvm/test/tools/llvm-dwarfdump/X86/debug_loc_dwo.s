# RUN: llvm-mc %s -filetype obj -triple x86_64-pc-linux -o %t.o
# RUN: llvm-dwarfdump --debug-loc %t.o | FileCheck %s

# We make sure that llvm-dwarfdump can dump the .debug_loc.dwo section
# without requiring a compilation unit in the .debug_info.dwo section.

# CHECK:         .debug_loc.dwo contents:
# CHECK-NEXT:    0x00000000:
# CHECK-NEXT:    DW_LLE_startx_length (0x00000001, 0x00000010): DW_OP_reg5 RDI

.section .debug_loc.dwo,"",@progbits
# One location list. The pre-DWARF v5 implementation only recognizes
# DW_LLE_startx_length as an entry kind in .debug_loc.dwo (besides
# end_of_list), which is what llvm generates as well.
.byte 3          # DW_LLE_startx_length
.byte 0x01       # Index
.long 0x10       # Length
.short 1         # Loc expr size
.byte 85         # DW_OP_reg5
.byte 0          # DW_LLE_end_of_list
