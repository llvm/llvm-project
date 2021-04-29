# RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux %s -o %t
# RUN: lldb-test symbols %t | FileCheck %s

        .file   1 "/tmp" "b.cc"

        .section        .debug_types,"",@progbits

# CHECK: Types:
# Type unit one: "struct A" defined at b.cc:1
# CHECK-DAG: name = "A", size = 4, decl = b.cc:1
1:
        .long   4f-2f                   # Length of Unit
2:
        .short  4                       # DWARF version number
        .long   .debug_abbrev           # Offset Into Abbrev. Section
        .byte   8                       # Address Size (in bytes)
        .quad   5390450678491038984     # Type Signature
        .long   3f-1b                   # Type DIE Offset
        .byte   1                       # Abbrev [1] 0x17:0x1b DW_TAG_type_unit
        .short  4                       # DW_AT_language
        .long   .Lline_table_start0     # DW_AT_stmt_list
3:
        .byte   2                       # Abbrev [2] 0x1e:0xc DW_TAG_structure_type
        .long   .LA                     # DW_AT_name
        .byte   4                       # DW_AT_byte_size
        .byte   1                       # DW_AT_decl_file
        .byte   1                       # DW_AT_decl_line
        .byte   0                       # End Of Children Mark
4:

# Type unit two: "struct B" defined at b.cc:2
# It shares the same line table as unit one.
# CHECK-DAG: name = "B", size = 4, decl = b.cc:2
1:
        .long   4f-2f                   # Length of Unit
2:
        .short  4                       # DWARF version number
        .long   .debug_abbrev           # Offset Into Abbrev. Section
        .byte   8                       # Address Size (in bytes)
        .quad   5390450678491038985     # Type Signature
        .long   3f-1b                   # Type DIE Offset
        .byte   1                       # Abbrev [1] 0x17:0x1b DW_TAG_type_unit
        .short  4                       # DW_AT_language
        .long   .Lline_table_start0     # DW_AT_stmt_list
3:
        .byte   2                       # Abbrev [2] 0x1e:0xc DW_TAG_structure_type
        .long   .LB                     # DW_AT_name
        .byte   4                       # DW_AT_byte_size
        .byte   1                       # DW_AT_decl_file
        .byte   2                       # DW_AT_decl_line
        .byte   0                       # End Of Children Mark
4:

# Type unit three: "struct C".
# DW_AT_stmt_list missing
# CHECK-DAG: name = "C", size = 4, line = 3
1:
        .long   4f-2f                   # Length of Unit
2:
        .short  4                       # DWARF version number
        .long   .debug_abbrev           # Offset Into Abbrev. Section
        .byte   8                       # Address Size (in bytes)
        .quad   5390450678491038986     # Type Signature
        .long   3f-1b                   # Type DIE Offset
        .byte   4                       # Abbrev [4] 0x17:0x1b DW_TAG_type_unit
        .short  4                       # DW_AT_language
3:
        .byte   2                       # Abbrev [2] 0x1e:0xc DW_TAG_structure_type
        .long   .LC                     # DW_AT_name
        .byte   4                       # DW_AT_byte_size
        .byte   1                       # DW_AT_decl_file
        .byte   3                       # DW_AT_decl_line
        .byte   0                       # End Of Children Mark
4:

# Type unit four: "struct D".
# DW_AT_stmt_list invalid
# CHECK-DAG: name = "D", size = 4, line = 4
1:
        .long   4f-2f                   # Length of Unit
2:
        .short  4                       # DWARF version number
        .long   .debug_abbrev           # Offset Into Abbrev. Section
        .byte   8                       # Address Size (in bytes)
        .quad   5390450678491038987     # Type Signature
        .long   3f-1b                   # Type DIE Offset
        .byte   1                       # Abbrev [1] 0x17:0x1b DW_TAG_type_unit
        .short  4                       # DW_AT_language
        .long   .Lline_table_start0+47  # DW_AT_stmt_list
3:
        .byte   2                       # Abbrev [2] 0x1e:0xc DW_TAG_structure_type
        .long   .LD                     # DW_AT_name
        .byte   4                       # DW_AT_byte_size
        .byte   1                       # DW_AT_decl_file
        .byte   4                       # DW_AT_decl_line
        .byte   0                       # End Of Children Mark
4:

# Type unit five: "struct E".
# DW_AT_decl_file invalid
# CHECK-DAG: name = "E", size = 4, line = 5
1:
        .long   4f-2f                   # Length of Unit
2:
        .short  4                       # DWARF version number
        .long   .debug_abbrev           # Offset Into Abbrev. Section
        .byte   8                       # Address Size (in bytes)
        .quad   5390450678491038988     # Type Signature
        .long   3f-1b                   # Type DIE Offset
        .byte   1                       # Abbrev [1] 0x17:0x1b DW_TAG_type_unit
        .short  4                       # DW_AT_language
        .long   .Lline_table_start0     # DW_AT_stmt_list
3:
        .byte   2                       # Abbrev [2] 0x1e:0xc DW_TAG_structure_type
        .long   .LE                     # DW_AT_name
        .byte   4                       # DW_AT_byte_size
        .byte   47                      # DW_AT_decl_file
        .byte   5                       # DW_AT_decl_line
        .byte   0                       # End Of Children Mark
4:


        .section        .debug_str,"MS",@progbits,1
.LA:
        .asciz  "A"
.LB:
        .asciz  "B"
.LC:
        .asciz  "C"
.LD:
        .asciz  "D"
.LE:
        .asciz  "E"

        .section        .debug_abbrev,"",@progbits
        .byte   1                       # Abbreviation Code
        .byte   65                      # DW_TAG_type_unit
        .byte   1                       # DW_CHILDREN_yes
        .byte   19                      # DW_AT_language
        .byte   5                       # DW_FORM_data2
        .byte   16                      # DW_AT_stmt_list
        .byte   23                      # DW_FORM_sec_offset
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   2                       # Abbreviation Code
        .byte   19                      # DW_TAG_structure_type
        .byte   0                       # DW_CHILDREN_no
        .byte   3                       # DW_AT_name
        .byte   14                      # DW_FORM_strp
        .byte   11                      # DW_AT_byte_size
        .byte   11                      # DW_FORM_data1
        .byte   58                      # DW_AT_decl_file
        .byte   11                      # DW_FORM_data1
        .byte   59                      # DW_AT_decl_line
        .byte   11                      # DW_FORM_data1
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   3                       # Abbreviation Code
        .byte   65                      # DW_TAG_type_unit
        .byte   1                       # DW_CHILDREN_yes
        .byte   19                      # DW_AT_language
        .byte   5                       # DW_FORM_data2
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   4                       # Abbreviation Code
        .byte   17                      # DW_TAG_compile_unit
        .byte   1                       # DW_CHILDREN_yes
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   5                       # Abbreviation Code
        .byte   19                      # DW_TAG_structure_type
        .byte   0                       # DW_CHILDREN_no
        .byte   105                     # DW_AT_signature
        .byte   32                      # DW_FORM_ref_sig8
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   0                       # EOM(3)
        .section        .debug_info,"",@progbits
.Lcu_begin0:
        .long   .Ldebug_info_end1-.Ldebug_info_start1 # Length of Unit
.Ldebug_info_start1:
        .short  4                       # DWARF version number
        .long   .debug_abbrev           # Offset Into Abbrev. Section
        .byte   8                       # Address Size (in bytes)
        .byte   4                       # Abbrev [4] 0xb:0x32 DW_TAG_compile_unit
        .byte   5                       # Abbrev [5] DW_TAG_structure_type
        .quad   5390450678491038984     # DW_AT_signature
        .byte   5                       # Abbrev [5] DW_TAG_structure_type
        .quad   5390450678491038985     # DW_AT_signature
        .byte   5                       # Abbrev [5] DW_TAG_structure_type
        .quad   5390450678491038986     # DW_AT_signature
        .byte   5                       # Abbrev [5] DW_TAG_structure_type
        .quad   5390450678491038987     # DW_AT_signature
        .byte   5                       # Abbrev [5] DW_TAG_structure_type
        .quad   5390450678491038988     # DW_AT_signature
        .byte   0                       # End Of Children Mark
.Ldebug_info_end1:

        .section        .debug_line,"",@progbits
.Lline_table_start0:
