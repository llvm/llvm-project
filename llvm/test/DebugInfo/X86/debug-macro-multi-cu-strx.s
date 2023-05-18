## This test checks that llvm-dwarfdump can dump debug_macro
## section containing contributions from multiple CU's represented
## using DW_MACRO_define_strx form.

# RUN: llvm-mc -triple x86_64-unknown-linux -filetype=obj %s -o -| \
# RUN:   llvm-dwarfdump -debug-macro - | FileCheck -strict-whitespace -match-full-lines %s

#      CHECK:.debug_macro contents:
# CHECK-NEXT:0x00000000:
# CHECK-NEXT:macro header: version = 0x0005, flags = 0x02, format = DWARF32, debug_line_offset = 0x00000000
# CHECK-NEXT:DW_MACRO_start_file - lineno: 0 filenum: 0
# CHECK-NEXT:  DW_MACRO_define_strx - lineno: 1 macro: DWARF_VERSION 5
# CHECK-NEXT:  DW_MACRO_define_strx - lineno: 2 macro: COMPILE_UNIT 1
# CHECK-NEXT:  DW_MACRO_undef_strx - lineno: 3 macro: COMPILE_UNIT
# CHECK-NEXT:DW_MACRO_end_file

#      CHECK:0x00000015:
# CHECK-NEXT:macro header: version = 0x0005, flags = 0x02, format = DWARF32, debug_line_offset = 0x00000000
# CHECK-NEXT:DW_MACRO_start_file - lineno: 1 filenum: 3
# CHECK-NEXT:  DW_MACRO_define_strx - lineno: 2 macro: COMPILE_UNIT 2
# CHECK-NEXT:  DW_MACRO_undef_strx - lineno: 3 macro: COMPILE_UNIT
# CHECK-NEXT:DW_MACRO_end_file

       .section        .debug_abbrev,"",@progbits
       .byte   1                       # Abbreviation Code
       .byte   17                      # DW_TAG_compile_unit
       .byte   0                       # DW_CHILDREN_no
       .byte   114                     # DW_AT_str_offsets_base
       .byte   23                      # DW_FORM_sec_offset
       .byte   121                     # DW_AT_macros
       .byte   23                      # DW_FORM_sec_offset
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
       .long   .Lstr_offsets_base0     # DW_AT_str_offsets_base
       .long   .Lcu_macro_begin0       # DW_AT_macros
.Ldebug_info_end0:
.Lcu_begin1:
       .long   .Ldebug_info_end1-.Ldebug_info_start1 # Length of Unit
.Ldebug_info_start1:
       .short  5                      # DWARF version number
       .byte   1                       # DWARF Unit Type
       .byte   8                       # Address Size (in bytes)
       .long   .debug_abbrev           # Offset Into Abbrev. Section
       .byte   1                       # Abbrev [1] 0xc:0x12 DW_TAG_compile_unit
       .long   .Lstr_offsets_base1     # DW_AT_str_offsets_base
       .long   .Lcu_macro_begin1       # DW_AT_macros
.Ldebug_info_end1:

       .section        .debug_macro,"",@progbits
.Lcu_macro_begin0:
       .short  5                      # Macro information version
       .byte   2                       # Flags: 32 bit, debug_line_offset present
       .long   0                       # debug_line_offset
       .byte   3                       # DW_MACRO_start_file
       .byte   0                       # Line Number
       .byte   0                       # File Number
       .byte   11                      # DW_MACRO_define_strx
       .byte   1                       # Line Number
       .byte   0                       # Macro String
       .byte   11                      # DW_MACRO_define_strx
       .byte   2                       # Line Number
       .byte   1                       # Macro String
       .byte   12                      # DW_MACRO_undef_strx
       .byte   3                       # Line Number
       .byte   2                       # Macro String
       .byte   4                       # DW_MACRO_end_file
       .byte   0                       # End Of Macro List Mark
.Lcu_macro_begin1:
       .short  5                      # Macro information version
       .byte   2                       # Flags: 32 bit, debug_line_offset present
       .long   0                       # debug_line_offset
       .byte   3                       # DW_MACRO_start_file
       .byte   1                       # Line Number
       .byte   3                       # File Number
       .byte   11                      # DW_MACRO_define_strx
       .byte   2                       # Line Number
       .byte   0                       # Macro String
       .byte   12                      # DW_MACRO_undef_strx
       .byte   3                       # Line Number
       .byte   1                       # Macro String
       .byte   4                       # DW_MACRO_end_file

       .section        .debug_str_offsets,"",@progbits
       .long   16                      # Unit length
       .short  5                       # Version
       .short  0                       # Padding
.Lstr_offsets_base0:
       .long   .Linfo_string0
       .long   .Linfo_string1
       .long   .Linfo_string2
       .long   12                      # Unit length
       .short  5                       # Version
       .short  0                       # Padding
.Lstr_offsets_base1:
       .long   .Linfo_string3
       .long   .Linfo_string4

       .section        .debug_str,"MS",@progbits,1
.Linfo_string0:
       .asciz  "DWARF_VERSION 5"
.Linfo_string1:
       .asciz  "COMPILE_UNIT 1"
.Linfo_string2:
       .asciz  "COMPILE_UNIT"
.Linfo_string3:
       .asciz  "COMPILE_UNIT 2"
.Linfo_string4:
       .asciz  "COMPILE_UNIT"
