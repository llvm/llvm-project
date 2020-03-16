# This tests the scenario where a (split) compile unit contains reference to a
# type, but that type is defined in another compile unit in the main object
# file.

# REQUIRES: x86

# RUN: llvm-mc %s -o %t --triple=x86_64-pc-linux --filetype=obj --defsym MAIN=0
# RUN: llvm-mc %s -o %T/dwo-type-in-main-file-cu2.dwo --triple=x86_64-pc-linux --filetype=obj --defsym DWO=0
# RUN: cd %T
# RUN: %lldb %t -o "target var a" -b 2>&1 | FileCheck %s

# CHECK: (A) a = (b = 47)

.ifdef MAIN
        .type   a,@object               # @a
        .data
        .globl  a
        .quad   0 # padding
a:
        .long   47                      # 0x2f
        .size   a, 4

        .section        .debug_addr,"",@progbits
.Laddr_table_base0:
        .quad   a

        .section        .debug_info,"",@progbits
.Lcu_begin0:
        .long   .Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
        .short  4                       # DWARF version number
        .long   .debug_abbrev           # Offset Into Abbrev. Section
        .byte   8                       # Address Size (in bytes)
        .byte   1                       # Abbrev [1] 0xb:0x89 DW_TAG_compile_unit
        .asciz  "Hand-written DWARF"    # DW_AT_producer
        .asciz  "cu1.cc"                # DW_AT_name
        .byte   2                       # Abbrev [2] 0x2a:0x2d DW_TAG_structure_type
        .asciz  "A"                     # DW_AT_name
        .byte   4                       # DW_AT_byte_size
        .byte   3                       # Abbrev [3] 0x33:0xc DW_TAG_member
        .asciz  "b"                     # DW_AT_name
        .long   .Lint                   # DW_AT_type
        .byte   0                       # DW_AT_data_member_location
        .byte   0                       # End Of Children Mark
.Lint:
        .byte   7                       # Abbrev [7] 0x57:0x7 DW_TAG_base_type
        .asciz  "int"                   # DW_AT_name
        .byte   5                       # DW_AT_encoding
        .byte   4                       # DW_AT_byte_size
        .byte   0                       # End Of Children Mark
.Ldebug_info_end0:

.Lcu_begin1:
        .long   .Ldebug_info_end1-.Ldebug_info_start1 # Length of Unit
.Ldebug_info_start1:
        .short  4                       # DWARF version number
        .long   .debug_abbrev           # Offset Into Abbrev. Section
        .byte   8                       # Address Size (in bytes)
        .byte   8                       # Abbrev DW_TAG_compile_unit
        .asciz  "dwo-type-in-main-file-cu2.dwo" # DW_AT_GNU_dwo_name
        .quad   5578312047953902346     # DW_AT_GNU_dwo_id
        .long   .Laddr_table_base0      # DW_AT_GNU_addr_base
.Ldebug_info_end1:

        .section        .debug_abbrev,"",@progbits
        .byte   1                       # Abbreviation Code
        .byte   17                      # DW_TAG_compile_unit
        .byte   1                       # DW_CHILDREN_yes
        .byte   37                      # DW_AT_producer
        .byte   8                       # DW_FORM_string
        .byte   3                       # DW_AT_name
        .byte   8                       # DW_FORM_string
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   2                       # Abbreviation Code
        .byte   19                      # DW_TAG_structure_type
        .byte   1                       # DW_CHILDREN_yes
        .byte   3                       # DW_AT_name
        .byte   8                       # DW_FORM_string
        .byte   11                      # DW_AT_byte_size
        .byte   11                      # DW_FORM_data1
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   3                       # Abbreviation Code
        .byte   13                      # DW_TAG_member
        .byte   0                       # DW_CHILDREN_no
        .byte   3                       # DW_AT_name
        .byte   8                       # DW_FORM_string
        .byte   73                      # DW_AT_type
        .byte   19                      # DW_FORM_ref4
        .byte   56                      # DW_AT_data_member_location
        .byte   11                      # DW_FORM_data1
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   7                       # Abbreviation Code
        .byte   36                      # DW_TAG_base_type
        .byte   0                       # DW_CHILDREN_no
        .byte   3                       # DW_AT_name
        .byte   8                       # DW_FORM_string
        .byte   62                      # DW_AT_encoding
        .byte   11                      # DW_FORM_data1
        .byte   11                      # DW_AT_byte_size
        .byte   11                      # DW_FORM_data1
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   8                       # Abbreviation Code
        .byte   17                      # DW_TAG_compile_unit
        .byte   0                       # DW_CHILDREN_no
        .ascii  "\260B"                 # DW_AT_GNU_dwo_name
        .byte   8                       # DW_FORM_string
        .ascii  "\261B"                 # DW_AT_GNU_dwo_id
        .byte   7                       # DW_FORM_data8
        .ascii  "\263B"                 # DW_AT_GNU_addr_base
        .byte   23                      # DW_FORM_sec_offset
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   0                       # EOM(3)
.endif

.ifdef DWO
        .section        .debug_info.dwo,"e",@progbits
        .long   .Ldebug_info_dwo_end0-.Ldebug_info_dwo_start0 # Length of Unit
.Ldebug_info_dwo_start0:
        .short  4                       # DWARF version number
        .long   0                       # Offset Into Abbrev. Section
        .byte   8                       # Address Size (in bytes)
        .byte   1                       # Abbrev [1] 0xb:0x1c DW_TAG_compile_unit
        .asciz  "Hand-written DWARF"    # DW_AT_producer
        .asciz  "cu2.cc"                # DW_AT_name
        .asciz  "dwo-type-in-main-file-cu2.dwo" # DW_AT_GNU_dwo_name
        .quad   5578312047953902346     # DW_AT_GNU_dwo_id
        .byte   2                       # Abbrev [2] 0x19:0xb DW_TAG_variable
        .asciz  "a"                     # DW_AT_name
        .long   .LA_fwd-.debug_info.dwo # DW_AT_type
        .byte   2                       # DW_AT_location
        .byte   251
        .byte   0
.LA_fwd:
        .byte   3                       # Abbrev [3] 0x24:0x2 DW_TAG_structure_type
        .asciz  "A"                     # DW_AT_name
                                        # DW_AT_declaration
        .byte   0                       # End Of Children Mark
.Ldebug_info_dwo_end0:

        .section        .debug_abbrev.dwo,"e",@progbits
        .byte   1                       # Abbreviation Code
        .byte   17                      # DW_TAG_compile_unit
        .byte   1                       # DW_CHILDREN_yes
        .byte   37                      # DW_AT_producer
        .byte   8                       # DW_FORM_string
        .byte   3                       # DW_AT_name
        .byte   8                       # DW_FORM_string
        .ascii  "\260B"                 # DW_AT_GNU_dwo_name
        .byte   8                       # DW_FORM_string
        .ascii  "\261B"                 # DW_AT_GNU_dwo_id
        .byte   7                       # DW_FORM_data8
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   2                       # Abbreviation Code
        .byte   52                      # DW_TAG_variable
        .byte   0                       # DW_CHILDREN_no
        .byte   3                       # DW_AT_name
        .byte   8                       # DW_FORM_string
        .byte   73                      # DW_AT_type
        .byte   19                      # DW_FORM_ref4
        .byte   2                       # DW_AT_location
        .byte   24                      # DW_FORM_exprloc
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   3                       # Abbreviation Code
        .byte   19                      # DW_TAG_structure_type
        .byte   0                       # DW_CHILDREN_no
        .byte   3                       # DW_AT_name
        .byte   8                       # DW_FORM_string
        .byte   60                      # DW_AT_declaration
        .byte   25                      # DW_FORM_flag_present
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   0                       # EOM(3)
.endif
