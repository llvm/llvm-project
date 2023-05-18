# This test that we don't get confused by line tables containing a tombstone
# (-1) value, as produced by recent lld's. Line sequences with the tombstone
# value should be completely ignored. The tombstone sequence is deliberately
# longer so that any attempt at an address binary search will likely land inside
# the sequence.

# RUN: llvm-mc --filetype=obj --triple=x86_64-pc-linux %s -o %t
# RUN: %lldb -o "image lookup -n main -v" -o "image dump line-table main.cpp" \
# RUN:   -o exit %t | FileCheck %s

# CHECK-LABEL: image lookup -n main -v
# CHECK: LineEntry: [0x0000000000001000-0x0000000000001001): main.cpp:1
# CHECK-LABEL: image dump line-table main.cpp
# CHECK-NEXT: Line table for main.cpp
# CHECK-NEXT: 0x0000000000001000: main.cpp:1
# CHECK-NEXT: 0x0000000000001001: main.cpp:1
# CHECK-EMPTY:
# CHECK-NEXT: exit

        .text
.space 0x1000
main:
  nop
.Lmain_end:

        .section        .debug_abbrev,"",@progbits
        .byte   1                               # Abbreviation Code
        .byte   17                              # DW_TAG_compile_unit
        .byte   0                               # DW_CHILDREN_no
        .byte   37                              # DW_AT_producer
        .byte   8                               # DW_FORM_string
        .byte   3                               # DW_AT_name
        .byte   8                               # DW_FORM_string
        .byte   16                              # DW_AT_stmt_list
        .byte   23                              # DW_FORM_sec_offset
        .byte   17                              # DW_AT_low_pc
        .byte   1                               # DW_FORM_addr
        .byte   18                              # DW_AT_high_pc
        .byte   6                               # DW_FORM_data4
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   0                               # EOM(3)

        .section        .debug_info,"",@progbits
.Lcu_begin0:
        .long   .Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
        .short  4                               # DWARF version number
        .long   0                               # Offset Into Abbrev. Section
        .byte   8                               # Address Size (in bytes)
        .byte   1                               # Abbrev [1] 0xb:0xc4 DW_TAG_compile_unit
        .asciz  "Hand-written DWARF"            # DW_AT_producer
        .asciz  "main.cpp"                      # DW_AT_name
        .long   0                               # DW_AT_stmt_list
        .quad   main-.text                      # DW_AT_low_pc
        .long   .Lmain_end-main                 # DW_AT_high_pc
.Ldebug_info_end0:

.section .debug_line,"",@progbits
        .long   .Llt1_end - .Llt1_start # Length of Unit (DWARF-32 format)
.Llt1_start:
        .short  4               # DWARF version number
        .long   .Lprologue1_end-.Lprologue1_start # Length of Prologue
.Lprologue1_start:
        .byte   1               # Minimum Instruction Length
        .byte   1               # Maximum Operations per Instruction
        .byte   1               # Default is_stmt
        .byte   -5              # Line Base
        .byte   14              # Line Range
        .byte   13              # Opcode Base
        .byte   0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1 # Standard Opcode Lengths
        .byte   0
        .asciz "main.cpp"          # File table
        .byte   0, 0, 0
        .byte   0
.Lprologue1_end:
        .byte   0, 9, 2         # DW_LNE_set_address
        .quad   -1
        .byte   1               # DW_LNS_copy
        .byte   33              # address += 1,  line += 1
        .byte   33              # address += 1,  line += 1
        .byte   33              # address += 1,  line += 1
        .byte   33              # address += 1,  line += 1
        .byte   33              # address += 1,  line += 1
        .byte   33              # address += 1,  line += 1
        .byte   33              # address += 1,  line += 1
        .byte   33              # address += 1,  line += 1
        .byte   33              # address += 1,  line += 1
        .byte   33              # address += 1,  line += 1
        .byte   33              # address += 1,  line += 1
        .byte   33              # address += 1,  line += 1
        .byte   33              # address += 1,  line += 1
        .byte   33              # address += 1,  line += 1
        .byte   33              # address += 1,  line += 1
        .byte   2               # DW_LNS_advance_pc
        .uleb128 1
        .byte   0, 1, 1         # DW_LNE_end_sequence

        .byte   0, 9, 2         # DW_LNE_set_address
        .quad   main-.text
        .byte   18              # address += 0,  line += 0
        .byte   2               # DW_LNS_advance_pc
        .uleb128 1
        .byte   0, 1, 1         # DW_LNE_end_sequence
.Llt1_end:

