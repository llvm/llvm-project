## This test checks that llvm-dwarfdump produces error message
## while parsing an *_strx form, if units contribution to
## debug_macro[.dwo] section is missing.

# RUN: llvm-mc -triple x86_64-unknown-linux -filetype=obj %s -o -| \
# RUN:   not llvm-dwarfdump -debug-macro - /dev/null 2>&1 | FileCheck %s

# CHECK: error: Macro contribution of the unit not found
       .section        .debug_abbrev,"",@progbits
       .byte   1                       # Abbreviation Code
       .byte   17                      # DW_TAG_compile_unit
       .byte   0                       # DW_CHILDREN_no
       .byte   0                       # EOM(1)
       .byte   0                       # EOM(2)
       .byte   0                       # EOM(3)

       .section        .debug_info,"",@progbits
.Lcu_begin0:
       .long   .Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
       .short  5                      # DWARF version number
       .byte   1                       # DWARF Unit Type
       .byte   8                       # Address Size (in bytes)
       .long   .debug_abbrev           # Offset Into Abbrev. Section
       .byte   1                       # Abbrev [1] 0xc:0x12 DW_TAG_compile_unit
.Ldebug_info_end0:

       .section        .debug_macro,"",@progbits
.Lcu_macro_begin0:
       .short  5                      # Macro information version
       .byte   0                       # Flags: 32 bit
       .byte   11                      # DW_MACRO_define_strx
