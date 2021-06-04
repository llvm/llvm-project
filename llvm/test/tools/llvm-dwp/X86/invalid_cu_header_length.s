# RUN: llvm-mc -triple x86_64-unknown-linux %s -filetype=obj -o %t.o \
# RUN:         -split-dwarf-file=%t.dwo -dwarf-version=5
# RUN: not llvm-dwp %t.dwo -o %t.dwp 2>&1 | FileCheck %s

# CHECK: error: compile unit exceeds .debug_info section range: 20 >= 6
    .section	.debug_info.dwo,"e",@progbits
    .long 16      # Length of Unit
    .short 5      # Version