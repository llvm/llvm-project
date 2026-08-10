# Test handling of DWARF5 line tables. In particular, test that we handle files
# which are present in the line table more than once.

# RUN: llvm-mc -filetype=obj -o %t -triple x86_64-pc-linux %s
# RUN: %lldb %t -o "source info -f file0.c" -o "source info -f file1.c" \
# RUN:   -o "breakpoint set -f file0.c -l 42" \
# RUN:   -o "breakpoint set -f file0.c -l 47" \
# RUN:   -o exit | FileCheck %s

# CHECK-LABEL: source info -f file0.c
# CHECK: [0x0000000000000000-0x0000000000000001): /file0.c:42
# CHECK-LABEL: source info -f file1.c
# CHECK: [0x0000000000000001-0x0000000000000002): /file1.c:47
# CHECK-LABEL: breakpoint set -f file0.c -l 42
# CHECK: Breakpoint 1: {{.*}}`foo,
# CHECK-LABEL: breakpoint set -f file0.c -l 47
# CHECK: Breakpoint 2: {{.*}}`foo + 2,

        .text
        .globl  foo
foo:
        nop
        nop
        nop
.Lfoo_end:

        .section        .debug_abbrev,"",@progbits
        .byte   1                       # Abbreviation Code
        .byte   17                      # DW_TAG_compile_unit
        .byte   0                       # DW_CHILDREN_no
        .byte   37                      # DW_AT_producer
        .byte   8                       # DW_FORM_string
        .byte   19                      # DW_AT_language
        .byte   5                       # DW_FORM_data2
        .byte   3                       # DW_AT_name
        .byte   8                       # DW_FORM_string
        .byte   16                      # DW_AT_stmt_list
        .byte   23                      # DW_FORM_sec_offset
        .byte   27                      # DW_AT_comp_dir
        .byte   8                       # DW_FORM_string
        .byte   17                      # DW_AT_low_pc
        .byte   1                       # DW_FORM_addr
        .byte   18                      # DW_AT_high_pc
        .byte   6                       # DW_FORM_data4
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   0                       # EOM(3)

        .section        .debug_info,"",@progbits
.Lcu_begin0:
        .long   .Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
        .short  5                       # DWARF version number
        .byte   1                       # DWARF Unit Type
        .byte   8                       # Address Size (in bytes)
        .long   .debug_abbrev           # Offset Into Abbrev. Section
        .byte   1                       # Abbrev [1] 0xc:0x23 DW_TAG_compile_unit
        .asciz  "Hand-written DWARF"    # DW_AT_producer
        .short  12                      # DW_AT_language
        .asciz  "file0.c"               # DW_AT_name
        .long   .Lline_table_begin      # DW_AT_stmt_list
        .asciz  "/"                     # DW_AT_comp_dir
        .quad   foo                     # DW_AT_low_pc
        .long   .Lfoo_end-foo           # DW_AT_high_pc
.Ldebug_info_end0:

        .section        .debug_line,"",@progbits
.Lline_table_begin:
        .long .Lline_end-.Lline_start
.Lline_start:
        .short  5                       # DWARF version number
        .byte   8                       # Address Size (in bytes)
        .byte   0                       # Segment Selector Size
        .long   .Lheader_end-.Lheader_start
.Lheader_start:
        .byte   1                       # Minimum Instruction Length
        .byte   1                       # Maximum Operations per Instruction
        .byte   1                       # Default is_stmt
        .byte   0                       # Line Base
        .byte   0                       # Line Range
        .byte   5                       # Opcode Base
        .byte   0, 1, 1, 1              # Standard Opcode Lengths

        # Directory table format
        .byte   1                       # One element per directory entry
        .byte   1                       # DW_LNCT_path
        .byte   0x08                    # DW_FORM_string

        # Directory table entries
        .byte   1                       # 1 directory
        .asciz  "/"

        # File table format
        .byte   2                       # 2 elements per file entry
        .byte   1                       # DW_LNCT_path
        .byte   0x08                    # DW_FORM_string
        .byte   2                       # DW_LNCT_directory_index
        .byte   0x0b                    # DW_FORM_data1

        # File table entries
        .byte   3                       # 3 files
        .asciz  "file0.c"
        .byte   0
        .asciz  "file1.c"
        .byte   0
        .asciz  "file0.c"
        .byte   0
.Lheader_end:

        .byte   4, 0                    # DW_LNS_set_file 0
        .byte   0, 9, 2                 # DW_LNE_set_address
        .quad   foo
        .byte   3, 41                   # DW_LNS_advance_line 41
        .byte   1                       # DW_LNS_copy

        .byte   4, 1                    # DW_LNS_set_file 1
        .byte   2, 1                    # DW_LNS_advance_pc 1
        .byte   3, 5                    # DW_LNS_advance_line 5
        .byte   1                       # DW_LNS_copy

        .byte   4, 2                    # DW_LNS_set_file 2
        .byte   2, 1                    # DW_LNS_advance_pc 1
        .byte   1                       # DW_LNS_copy

        .byte   2, 1                    # DW_LNS_advance_pc 1
        .byte   0, 1, 1                 # DW_LNE_end_sequence
.Lline_end:
