## Test that inline function resolution works when the function has been split
## into multiple discontinuous parts (and those parts are placed in different
## sections)

# RUN: llvm-mc -triple x86_64-pc-linux -filetype=obj %s -o %t
# RUN: %lldb %t -o "image lookup -v -n look_me_up" -o exit | FileCheck %s

# CHECK:      1 match found in {{.*}}
# CHECK:      Summary: {{.*}}`foo + 6 [inlined] foo_inl + 1
# CHECK-NEXT:          {{.*}}`foo + 5
# CHECK:      Blocks: id = {{.*}}, ranges = [0x00000000-0x00000003)[0x00000004-0x00000008)
# CHECK-NEXT:         id = {{.*}}, ranges = [0x00000001-0x00000002)[0x00000005-0x00000007), name = "foo_inl"

        .text

        .type   foo,@function
foo:
        nop
.Lfoo_inl:
        nop
.Lfoo_inl_end:
        nop
.Lfoo_end:
        .size   foo, .Lfoo_end-foo

bar:
        nop
.Lbar_end:
        .size   bar, .Lbar_end-bar

        .section        .text.__part1,"ax",@progbits
foo.__part.1:
        nop
.Lfoo_inl.__part.1:
        nop
        .type   look_me_up,@function
look_me_up:
        nop
.Lfoo_inl.__part.1_end:
        nop
.Lfoo.__part.1_end:
        .size   foo.__part.1, .Lfoo.__part.1_end-foo.__part.1


        .section        .debug_abbrev,"",@progbits
        .byte   1                               # Abbreviation Code
        .byte   17                              # DW_TAG_compile_unit
        .byte   1                               # DW_CHILDREN_yes
        .byte   37                              # DW_AT_producer
        .byte   8                               # DW_FORM_string
        .byte   19                              # DW_AT_language
        .byte   5                               # DW_FORM_data2
        .byte   17                              # DW_AT_low_pc
        .byte   1                               # DW_FORM_addr
        .byte   85                              # DW_AT_ranges
        .byte   35                              # DW_FORM_rnglistx
        .byte   116                             # DW_AT_rnglists_base
        .byte   23                              # DW_FORM_sec_offset
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   2                               # Abbreviation Code
        .byte   46                              # DW_TAG_subprogram
        .byte   0                               # DW_CHILDREN_no
        .byte   3                               # DW_AT_name
        .byte   8                               # DW_FORM_string
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   3                               # Abbreviation Code
        .byte   46                              # DW_TAG_subprogram
        .byte   1                               # DW_CHILDREN_yes
        .byte   85                              # DW_AT_ranges
        .byte   35                              # DW_FORM_rnglistx
        .byte   3                               # DW_AT_name
        .byte   8                               # DW_FORM_string
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   4                               # Abbreviation Code
        .byte   29                              # DW_TAG_inlined_subroutine
        .byte   0                               # DW_CHILDREN_no
        .byte   85                              # DW_AT_ranges
        .byte   35                              # DW_FORM_rnglistx
        .byte   49                              # DW_AT_abstract_origin
        .byte   19                              # DW_FORM_ref4
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   0                               # EOM(3)

        .section        .debug_info,"",@progbits
.Lcu_begin0:
        .long   .Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
        .short  5                               # DWARF version number
        .byte   1                               # DWARF Unit Type
        .byte   8                               # Address Size (in bytes)
        .long   .debug_abbrev                   # Offset Into Abbrev. Section
        .byte   1                               # Abbrev DW_TAG_compile_unit
        .asciz  "Hand-written DWARF"            # DW_AT_producer
        .short  29                              # DW_AT_language
        .quad   0                               # DW_AT_low_pc
        .byte   1                               # DW_AT_ranges
        .long   .Lrnglists_table_base0          # DW_AT_rnglists_base

        .byte   3                               # Abbrev DW_TAG_subprogram
        .byte   2                               # DW_AT_ranges
        .asciz  "bar"                           # DW_AT_name
        .byte   0                               # End Of Children Mark

.Lfoo_inl_die:
        .byte   2                               # Abbrev DW_TAG_subprogram
        .asciz  "foo_inl"                       # DW_AT_name

        .byte   3                               # Abbrev DW_TAG_subprogram
        .byte   0                               # DW_AT_ranges
        .asciz  "foo"                           # DW_AT_name
        .byte   4                               # Abbrev DW_TAG_inlined_subroutine
        .byte   3                               # DW_AT_ranges
        .long   .Lfoo_inl_die-.Lcu_begin0       # DW_AT_abstract_origin
        .byte   0                               # End Of Children Mark

        .byte   0                               # End Of Children Mark
.Ldebug_info_end0:

        .section        .debug_rnglists,"",@progbits
        .long   .Ldebug_list_header_end0-.Ldebug_list_header_start0 # Length
.Ldebug_list_header_start0:
        .short  5                               # Version
        .byte   8                               # Address size
        .byte   0                               # Segment selector size
        .long   4                               # Offset entry count
.Lrnglists_table_base0:
        .long   .Ldebug_ranges0-.Lrnglists_table_base0
        .long   .Ldebug_ranges1-.Lrnglists_table_base0
        .long   .Ldebug_ranges2-.Lrnglists_table_base0
        .long   .Ldebug_ranges3-.Lrnglists_table_base0
.Ldebug_ranges0:
        .byte   6                               # DW_RLE_start_end
        .quad   foo
        .quad   .Lfoo_end
        .byte   6                               # DW_RLE_start_end
        .quad   foo.__part.1
        .quad   .Lfoo.__part.1_end
        .byte   0                               # DW_RLE_end_of_list
.Ldebug_ranges1:
        .byte   6                               # DW_RLE_start_end
        .quad   bar
        .quad   .Lbar_end
        .byte   6                               # DW_RLE_start_end
        .quad   foo.__part.1
        .quad   .Lfoo.__part.1_end
        .byte   6                               # DW_RLE_start_end
        .quad   foo
        .quad   .Lfoo_end
        .byte   0                               # DW_RLE_end_of_list
.Ldebug_ranges2:
        .byte   6                               # DW_RLE_start_end
        .quad   bar
        .quad   .Lbar_end
        .byte   0                               # DW_RLE_end_of_list
.Ldebug_ranges3:
        .byte   6                               # DW_RLE_start_end
        .quad   .Lfoo_inl
        .quad   .Lfoo_inl_end
        .byte   6                               # DW_RLE_start_end
        .quad   .Lfoo_inl.__part.1
        .quad   .Lfoo_inl.__part.1_end
        .byte   0                               # DW_RLE_end_of_list
.Ldebug_list_header_end0:
