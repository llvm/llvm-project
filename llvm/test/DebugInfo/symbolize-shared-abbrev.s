# Show that multiple CUs can have a single common .debug_abbrev table. This can
# occur due to e.g. LTO.

# REQUIRES: x86-registered-target

# RUN: llvm-mc --filetype=obj --triple=x86_64-pc-linux %s -o %t.o
# RUN: llvm-symbolizer --obj=%t.o 0 16 --functions=short | FileCheck %s

# CHECK:      foo
# CHECK-NEXT: foo.c:1:0

# CHECK:      bar
# CHECK-NEXT: bar.c:2:0

.global foo
.type foo,@function
foo:
.Lfunc_begin0:
    .file   1 "." "foo.c"
    .loc    1 1 0
    ret
.Lfunc_end0:

.global bar
.p2align 4, 0x90
.type bar,@function
bar:
.Lfunc_begin1:
    .file   2 "." "bar.c"
    .loc    2 2 0
    ret
.Lfunc_end1:

    .section    .debug_str,"MS",@progbits,1
.Linfo_string1:
    .asciz  "foo.c"
.Linfo_string2:
    .asciz  "."
.Linfo_string3:
    .asciz  "foo"
.Linfo_string4:
    .asciz  "bar.c"
.Linfo_string5:
    .asciz  "bar"

    # Regular .debug_abbrev section with CU and subprogram, but duplicated, with second
    # half reordered slightly, to show that the correct abbrev is being referenced.
    .section    .debug_abbrev,"",@progbits
    .byte   1                       # Abbreviation Code
    .byte   17                      # DW_TAG_compile_unit
    .byte   1                       # DW_CHILDREN_yes
    .byte   3                       # DW_AT_name
    .byte   14                      # DW_FORM_strp
    .byte   16                      # DW_AT_stmt_list
    .byte   23                      # DW_FORM_sec_offset
    .byte   27                      # DW_AT_comp_dir
    .byte   14                      # DW_FORM_strp
    .byte   17                      # DW_AT_low_pc
    .byte   1                       # DW_FORM_addr
    .byte   18                      # DW_AT_high_pc
    .byte   6                       # DW_FORM_data4
    .byte   0                       # EOM(1)
    .byte   0                       # EOM(2)
    .byte   2                       # Abbreviation Code
    .byte   46                      # DW_TAG_subprogram
    .byte   0                       # DW_CHILDREN_no
    .byte   17                      # DW_AT_low_pc
    .byte   1                       # DW_FORM_addr
    .byte   18                      # DW_AT_high_pc
    .byte   6                       # DW_FORM_data4
    .byte   3                       # DW_AT_name
    .byte   14                      # DW_FORM_strp
    .byte   58                      # DW_AT_decl_file
    .byte   11                      # DW_FORM_data1
    .byte   59                      # DW_AT_decl_line
    .byte   11                      # DW_FORM_data1
    .byte   63                      # DW_AT_external
    .byte   25                      # DW_FORM_flag_present
    .byte   0                       # EOM(1)
    .byte   0                       # EOM(2)
    # Second-half starts here.
    .byte   3                       # Abbreviation Code
    .byte   17                      # DW_TAG_compile_unit
    .byte   1                       # DW_CHILDREN_yes
    .byte   16                      # DW_AT_stmt_list
    .byte   23                      # DW_FORM_sec_offset
    .byte   27                      # DW_AT_comp_dir
    .byte   14                      # DW_FORM_strp
    .byte   17                      # DW_AT_low_pc
    .byte   1                       # DW_FORM_addr
    .byte   18                      # DW_AT_high_pc
    .byte   6                       # DW_FORM_data4
    .byte   3                       # DW_AT_name
    .byte   14                      # DW_FORM_strp
    .byte   0                       # EOM(1)
    .byte   0                       # EOM(2)
    .byte   4                       # Abbreviation Code
    .byte   46                      # DW_TAG_subprogram
    .byte   0                       # DW_CHILDREN_no
    .byte   17                      # DW_AT_low_pc
    .byte   1                       # DW_FORM_addr
    .byte   18                      # DW_AT_high_pc
    .byte   6                       # DW_FORM_data4
    .byte   58                      # DW_AT_decl_file
    .byte   11                      # DW_FORM_data1
    .byte   59                      # DW_AT_decl_line
    .byte   11                      # DW_FORM_data1
    .byte   63                      # DW_AT_external
    .byte   25                      # DW_FORM_flag_present
    .byte   3                       # DW_AT_name
    .byte   14                      # DW_FORM_strp
    .byte   0                       # EOM(1)
    .byte   0                       # EOM(2)
    .byte   0                       # EOM(3)

    .section    .debug_info,"",@progbits
    # First CU table.
    .long   .Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
    .short  4                       # DWARF version number
    .long   .debug_abbrev           # Offset Into Abbrev. Section
    .byte   8                       # Address Size (in bytes)
    .byte   1                       # Abbrev [1] 0xb:0x35 DW_TAG_compile_unit
    .long   .Linfo_string1          # DW_AT_name
    .long   .Lline_table_start0     # DW_AT_stmt_list
    .long   .Linfo_string2          # DW_AT_comp_dir
    .quad   .Lfunc_begin0           # DW_AT_low_pc
    .long   .Lfunc_end0-.Lfunc_begin0 # DW_AT_high_pc
    .byte   2                       # Abbrev [2] 0x2a:0x15 DW_TAG_subprogram
    .quad   .Lfunc_begin0           # DW_AT_low_pc
    .long   .Lfunc_end0-.Lfunc_begin0 # DW_AT_high_pc
    .long   .Linfo_string3          # DW_AT_name
    .byte   1                       # DW_AT_decl_file
    .byte   1                       # DW_AT_decl_line
                                        # DW_AT_external
    .byte   0                       # End Of Children Mark
.Ldebug_info_end0:

    # Second CU table.
    .long   .Ldebug_info_end1-.Ldebug_info_start1 # Length of Unit
.Ldebug_info_start1:
    .short  4                       # DWARF version number
    .long   .debug_abbrev           # Offset Into Abbrev. Section
    .byte   8                       # Address Size (in bytes)
    .byte   3                       # Abbrev [1] 0xb:0x35 DW_TAG_compile_unit
    .long   .Lline_table_start0     # DW_AT_stmt_list
    .long   .Linfo_string2          # DW_AT_comp_dir
    .quad   .Lfunc_begin1           # DW_AT_low_pc
    .long   .Lfunc_end1-.Lfunc_begin1 # DW_AT_high_pc
    .long   .Linfo_string4          # DW_AT_name
    .byte   4                       # Abbrev [2] 0x2a:0x15 DW_TAG_subprogram
    .quad   .Lfunc_begin1           # DW_AT_low_pc
    .long   .Lfunc_end1-.Lfunc_begin1 # DW_AT_high_pc
    .byte   1                       # DW_AT_decl_file
    .byte   1                       # DW_AT_decl_line
                                        # DW_AT_external
    .long   .Linfo_string5          # DW_AT_name
    .byte   0                       # End Of Children Mark
.Ldebug_info_end1:

    .section    .debug_line,"",@progbits
.Lline_table_start0:
