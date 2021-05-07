# RUN: llvm-mc -triple x86_64-unknown-linux %s -filetype=obj -o %t.o \
# RUN:         -split-dwarf-file=%t.dwo -dwarf-version=5
# RUN: not llvm-dwp %t.dwo -o %t.dwp 2>&1 | FileCheck %s

# CHECK: error: cannot parse compile unit length: unexpected end of data at offset 0x2 while reading [0x0, 0x4
    .section	.debug_info.dwo,"e",@progbits
    .short	0 # Length of Unit
