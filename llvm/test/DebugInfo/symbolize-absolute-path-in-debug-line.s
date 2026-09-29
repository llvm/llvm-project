# Create .debug_line containing absolute path in filename. Show that the path is sensibly printed/found/etc.
# REQUIRES: x86-registered-target

# RUN: sed s!FILEPATH!%/s! %s > %t.s
# RUN: llvm-mc --filetype=obj --triple=x86_64-pc-linux %t.s -o %t.o
# RUN: llvm-symbolizer --obj=%t.o 0 | FileCheck %s -DPATH=%/s

# CHECK: {{^}}[[PATH]]:1:0

.type _start,@function
_start:
.Lfunc_begin0:
    ret
.Lfunc_end0:

.section .debug_str,"MS",@progbits,1
.Linfo_string1:
    .asciz "test.c"
.Linfo_string2:
    .asciz "/some/dir"
.Linfo_string3:
    .asciz "_start"

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
    .byte   0                       # EOM(3)
    .section    .debug_info,"",@progbits
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

# Hand-written .debug_line to allow replacing in the absolute path
# into the filename table at runtime.
.section .debug_line,"",@progbits
.Lline_table_start0:
    .long .Ltable_end - .Ltable_start   # unit length
.Ltable_start:
    .short 4                            # version
    .long .Lheader_end - .Lheader_start # header length
.Lheader_start:
    .byte 1                             # min instruction length
    .byte 1                             # max ops per instruction
    .byte 1                             # default is_stmt
    .byte -5                            # line base
    .byte 14                            # line range
    .byte 13                            # opcode base
    .byte 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1 # standard opcode lengths
    .byte 0                             # directory table
    .asciz "FILEPATH"                   # filename table
    .byte 0, 0, 0
    .byte 0
.Lheader_end:
    .byte 0, 9, 2                       # DW_LNE_set_address
    .quad .Lfunc_begin0
    .byte 1                             # DW_LNS_copy
    .byte 33                            # +1 address, +1 line
    .byte 0, 1, 1                       # DW_LNE_end_sequence
.Ltable_end:
