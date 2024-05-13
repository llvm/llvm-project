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
.byte 0x11 # DW_TAG_compile_unit
.byte 0    # DW_CHILDREN_no
.byte 0    # Terminating attribute
.byte 0    # Terminating form
.byte 0    # Terminating abbrev code
