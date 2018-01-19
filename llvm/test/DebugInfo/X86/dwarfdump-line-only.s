# Test object to verify dwarfdump handles dumping a DWARF v5 line table
# without an associated unit.
# FIXME: Support FORM_strp in this situation.
#
# RUN: llvm-mc -triple x86_64-unknown-linux %s -filetype=obj -o - | \
# RUN: llvm-dwarfdump -v - | FileCheck %s

        .section .text
        # Dummy function
foo:    ret

        .section .debug_line,"",@progbits
# CHECK-LABEL: .debug_line contents:

# DWARF v5 line-table header.
LH_5_start:
        .long   LH_5_end-LH_5_version   # Length of Unit (DWARF-32 format)
LH_5_version:
        .short  5               # DWARF version number
        .byte   8               # Address Size
        .byte   0               # Segment Selector Size
        .long   LH_5_header_end-LH_5_params     # Length of Prologue
LH_5_params:
        .byte   1               # Minimum Instruction Length
        .byte   1               # Maximum Operations per Instruction
        .byte   1               # Default is_stmt
        .byte   -5              # Line Base
        .byte   14              # Line Range
        .byte   13              # Opcode Base
        .byte   0               # Standard Opcode Lengths
        .byte   1
        .byte   1
        .byte   1
        .byte   1
        .byte   0
        .byte   0
        .byte   0
        .byte   1
        .byte   0
        .byte   0
        .byte   1
        # Directory table format
        .byte   1               # One element per directory entry
        .byte   1               # DW_LNCT_path
        .byte   0x08            # DW_FORM_string
        # Directory table entries
        .byte   2               # Two directory entries
        .asciz  "Directory1"
        .asciz  "Directory2"
        # File table format
        .byte   4               # Four elements per file entry
        .byte   2               # DW_LNCT_directory_index
        .byte   0x0b            # DW_FORM_data1
        .byte   1               # DW_LNCT_path
        .byte   0x08            # DW_FORM_string
        .byte   3               # DW_LNCT_timestamp
        .byte   0x0f            # DW_FORM_udata
        .byte   4               # DW_LNCT_size
        .byte   0x0f            # DW_FORM_udata
        # File table entries
        .byte   2               # Two file entries
        .byte   1
        .asciz "File1"
        .byte   0x51
        .byte   0x52
        .byte   0
        .asciz "File2"
        .byte   0x53
        .byte   0x54
LH_5_header_end:
        # Minimal line number program with an address in it, which shows
        # we picked up the address size from the line-table header.
        .byte   0
        .byte   9
        .byte   2               # DW_LNE_set_address
        .quad   .text
        .byte   0
        .byte   1
        .byte   1               # DW_LNE_end_sequence
LH_5_end:

# CHECK: Line table prologue:
# CHECK: version: 5
# CHECK: address_size: 8
# CHECK: seg_select_size: 0
# CHECK: max_ops_per_inst: 1
# CHECK: include_directories[  0] = 'Directory1'
# CHECK: include_directories[  1] = 'Directory2'
# CHECK-NOT: include_directories
# CHECK: file_names[  1]    1 0x00000051 0x00000052 File1{{$}}
# CHECK: file_names[  2]    0 0x00000053 0x00000054 File2{{$}}
# CHECK-NOT: file_names
# CHECK: 0x0000000000000000 {{.*}} is_stmt end_sequence
