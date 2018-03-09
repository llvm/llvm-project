# Test object to verify dwarfdump handles a syntactically correct line-number
# program containing unrecognized extended opcodes.
# RUN: llvm-mc -triple x86_64-unknown-linux %s -filetype=obj -o %t.o
# RUN: llvm-dwarfdump -v %t.o | FileCheck %s
# RUN: llvm-dwarfdump -v %t.o 2>&1 | FileCheck %s --check-prefix=ERR

        .section .text
        # Dummy function
foo:    ret

# FIXME: When we can dump a line-table without a unit, we could remove
# the .debug_abbrev and .debug_info sections from this test.
        .section .debug_abbrev,"",@progbits
        .byte 0x01  # Abbrev code
        .byte 0x11  # DW_TAG_compile_unit
        .byte 0x00  # DW_CHILDREN_no
        .byte 0x10  # DW_AT_stmt_list
        .byte 0x17  # DW_FORM_sec_offset
        .byte 0x00  # EOM(1)
        .byte 0x00  # EOM(2)

        .section .debug_info,"",@progbits
        .long  CU_end-CU_version  # Length of Unit
CU_version:
        .short 4               # DWARF version number
        .long .debug_abbrev    # Offset Into Abbrev. Section
        .byte 8                # Address Size (in bytes)
# The compile-unit DIE, with DW_AT_stmt_list.
        .byte 1
        .long LT_start
        .byte 0 # NULL
CU_end:

        .long  CU2_end-CU2_version  # Length of Unit
CU2_version:
        .short 4               # DWARF version number
        .long .debug_abbrev    # Offset Into Abbrev. Section
        .byte 8                # Address Size (in bytes)
# The compile-unit DIE, with DW_AT_stmt_list.
        .byte 1
        .long LT2_start
        .byte 0 # NULL
CU2_end:

        .section .debug_line,"",@progbits
# CHECK-LABEL: .debug_line contents:

# DWARF v4 line-table header.
LT_start:
        .long   LT_end-LT_version   # Length of Unit (DWARF-32 format)
LT_version:
        .short  4               # DWARF version number
        .long   LT_header_end-LT_params     # Length of Prologue
LT_params:
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
        # No directories.
        .byte   0
        # No files.
        .byte   0
LT_header_end:
        # Bogus extended opcode with zero length.
        .byte   0               # Extended opcode indicator.
        .byte   0               # LEB length of extended opcode + operands.
        # Real opcode and operand.
        .byte   0
        .byte   9
        .byte   2               # DW_LNE_set_address
        .quad   .text
        # Bogus extended opcode with multibyte LEB length.
        .byte   0
        .byte   0x82            # Length of 2 but with additional length byte.
        .byte   0               # Additional length byte.
        .byte   0x47            # Unrecognized opcode...
        .byte   0               # with its 1-byte operand.
        # Proper end-sequence opcode.
        .byte   0
        .byte   1
        .byte   1               # DW_LNE_end_sequence
LT_end:

# CHECK:      Line table prologue:
# CHECK:      version: 4
# Exact prologue length isn't important but it tells us where to expect the
# line-number program to start, and we do want to verify those offsets.
# CHECK-NEXT: prologue_length: 0x00000014
# CHECK:      0x0000001e: 00 Badly formed extended line op
# CHECK-NEXT: 0x00000020: 00 DW_LNE_set_address
# CHECK-NEXT: 0x0000002b: 00 Unrecognized extended op 0x47 length 2
# CHECK-NEXT: 0x00000030: 00 DW_LNE_end_sequence
# CHECK-NEXT: 0x0000000000000000 {{.*}} is_stmt end_sequence


# DWARF v4 line-table header #2.
LT2_start:
        .long   LT2_end-LT2_version   # Length of Unit (DWARF-32 format)
LT2_version:
        .short  4               # DWARF version number
        .long   LT2_header_end-LT2_params   # Length of Prologue
LT2_params:
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
        # No directories.
        .byte   0
        # No files.
        .byte   0
LT2_header_end:
        # Real opcode and operand.
        .byte   0
        .byte   9
        .byte   2               # DW_LNE_set_address
        .quad   .text
        # Real opcode with incorrect length.
        .byte   0
        .byte   2               # Wrong length, should be 1.
        .byte   1               # DW_LNE_end_sequence
LT2_end:

# ERR:      Unexpected line op length at offset 0x0000005e
# ERR-SAME: expected 0x02 found 0x01

# The above parsing errors still let us move to the next unit.
# If the prologue is bogus, we need to bail out because we can't
# even find the next unit.

# DWARF v4 line-table header #3.
LT3_start:
        .long   LT3_end-LT3_version   # Length of Unit (DWARF-32 format)
LT3_version:
        .short  4               # DWARF version number
        .long   LT3_header_end-LT3_params   # Length of Prologue
LT3_params:
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
        # No directories.
        .byte   0
        # No files.
        .byte   0
        # Extra junk at the end of the prologue, so the length isn't right.
        .long   0
LT3_header_end:
        # Real opcode and operand.
        .byte   0
        .byte   9
        .byte   2               # DW_LNE_set_address
        .quad   .text
        # Real opcode with incorrect length.
        .byte   0
        .byte   2               # Wrong length, should be 1.
        .byte   1               # DW_LNE_end_sequence
LT3_end:

# We should have bailed out above, so never see this in the dump.
# DWARF v4 line-table header #4.
LT4_start:
        .long   LT4_end-LT4_version   # Length of Unit (DWARF-32 format)
LT4_version:
        .short  4               # DWARF version number
        .long   LT4_header_end-LT4_params   # Length of Prologue
LT4_params:
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
        # No directories.
        .byte   0
        # No files.
        .byte   0
LT4_header_end:
        # Real opcode and operand.
        .byte   0
        .byte   9
        .byte   2               # DW_LNE_set_address
        .quad   .text
        # Real opcode with correct length.
        .byte   0
        .byte   1
        .byte   1               # DW_LNE_end_sequence
LT4_end:

# Look for the dump of unit 3, and don't want unit 4.
# CHECK:     Line table prologue:
# CHECK-NOT: Line table prologue:

# And look for the error message.
# ERR:      warning: parsing line table prologue at 0x0000005f should have
# ERR-SAME: ended at 0x00000081 but it ended at 0x0000007d
