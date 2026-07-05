# RUN: llvm-mc -triple x86_64-unknown-linux %s -filetype=obj -o %t.o
# RUN: not llvm-dwarfdump -debug-aranges %t.o 2>&1 | FileCheck %s
# RUN: not llvm-dwarfdump -lookup 10 %t.o 2>&1 | FileCheck %s

## This checks that llvm-dwarfdump shows parsing errors in .debug_aranges.
## For more error cases see unittests/DebugInfo/DWARF/DWARFDebugArangeSetTest.cpp.

# CHECK: the length of address range table at offset 0x0 exceeds section size

    .section .debug_aranges,"",@progbits
    .long   .Lend - .Lversion + 1   # The length exceeds the section boundaries
.Lversion:
    .short  2                       # Version
    .long   0                       # Debug Info Offset
    .byte   4                       # Address Size
    .byte   0                       # Segment Selector Size
    .space  4                       # Padding
.Ltuples:
    .long   0, 1                    # Address and length
    .long   0, 0                    # Termination tuple
.Lend:
