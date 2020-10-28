## The test checks that llvm-dwarfdump can handle a malformed input file without
## crashing.

# RUN: llvm-mc -triple x86_64 %s -filetype=obj -o %t
# RUN: not llvm-dwarfdump -debug-rnglists %t 2>&1 | FileCheck %s

# CHECK: error: .debug_rnglists table at offset 0x0 has too small length (0x4) to contain a complete header

## An assertion used to trigger in the debug build of the DebugInfo/DWARF 
## library if the unit length field in a range list table was 0.
    .section .debug_rnglists,"",@progbits
    .long 0
