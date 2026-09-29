# RUN: llvm-mc %s -filetype obj -triple x86_64-apple-darwin -o - \
# RUN: | not llvm-dwarfdump --verify --debug-info - \
# RUN: | FileCheck %s
# CHECK: error: unable to decode LEB128 at offset 0x00000005: malformed uleb128, extends past end
# CHECK: error: Compilation unit without DIE.

        .section        __DWARF,__debug_info,regular,debug
.long 8  # CU length
.short 3 # Version
.long 0  # Abbrev offset
.byte 4  # AddrSize
.byte 1  # Abbrev 1
.long 7  # Unit lengthh...
.short 3
.long 0
.byte 4
        .section        __DWARF,__debug_abbrev,regular,debug
.byte 1    # Abbrev code
.byte 0x11 # TAG_compile_unit
.byte 0    # no children
.byte 0    # EOM(1)
.byte 0    # EOM(2)
           # Intentionally missing EOM(3)
