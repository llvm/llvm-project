## Show that llvm-dwarfdump dumps the whole .debug_line section when
## --debug-line is specified.

# RUN: llvm-mc %s -filetype obj -triple x86_64-pc-linux -o %t.o
# RUN: llvm-dwarfdump --debug-line %t.o | FileCheck %s --match-full-lines

# CHECK:      .debug_line contents:
# CHECK-NEXT: debug_line[0x00000000]
# CHECK-NEXT: Line table prologue:
# CHECK-NEXT:     total_length: 0x00000069
# CHECK-NEXT:          version: 5
# CHECK-NEXT:     address_size: 8
# CHECK-NEXT:  seg_select_size: 0
# CHECK-NEXT:  prologue_length: 0x0000004c
# CHECK-NEXT:  min_inst_length: 1
# CHECK-NEXT: max_ops_per_inst: 1
# CHECK-NEXT:  default_is_stmt: 1
# CHECK-NEXT:        line_base: -5
# CHECK-NEXT:       line_range: 7
# CHECK-NEXT:      opcode_base: 14
# CHECK-NEXT: standard_opcode_lengths[DW_LNS_copy] = 0
# CHECK-NEXT: standard_opcode_lengths[DW_LNS_advance_pc] = 1
# CHECK-NEXT: standard_opcode_lengths[DW_LNS_advance_line] = 1
# CHECK-NEXT: standard_opcode_lengths[DW_LNS_set_file] = 1
# CHECK-NEXT: standard_opcode_lengths[DW_LNS_set_column] = 1
# CHECK-NEXT: standard_opcode_lengths[DW_LNS_negate_stmt] = 0
# CHECK-NEXT: standard_opcode_lengths[DW_LNS_set_basic_block] = 0
# CHECK-NEXT: standard_opcode_lengths[DW_LNS_const_add_pc] = 0
# CHECK-NEXT: standard_opcode_lengths[DW_LNS_fixed_advance_pc] = 1
# CHECK-NEXT: standard_opcode_lengths[DW_LNS_set_prologue_end] = 0
# CHECK-NEXT: standard_opcode_lengths[DW_LNS_set_epilogue_begin] = 0
# CHECK-NEXT: standard_opcode_lengths[DW_LNS_set_isa] = 1
# CHECK-NEXT: standard_opcode_lengths[(null)] = 0
# CHECK-NEXT: include_directories[  0] = "dir1/dir2"
# CHECK-NEXT: file_names[  0]:
# CHECK-NEXT:            name: "file1.c"
# CHECK-NEXT:       dir_index: 2
# CHECK-NEXT:        mod_time: 0x12345678
# CHECK-NEXT:          length: 0x00000010
# CHECK-EMPTY:
# CHECK-NEXT: Address            Line   Column File   ISA Discriminator Flags
# CHECK-NEXT: ------------------ ------ ------ ------ --- ------------- -------------
# CHECK-NEXT: 0x0000000000000002      1      0      1   0             0  is_stmt
# CHECK-NEXT: 0x0000000000000002      1      4      3   0             0  is_stmt
# CHECK-NEXT: 0x0000000000000024      1      4      3   5             6  basic_block prologue_end epilogue_begin end_sequence
# CHECK-EMPTY:
# CHECK-NEXT: debug_line[0x0000006d]
# CHECK-NEXT: Line table prologue:
# CHECK-NEXT:     total_length: 0x0000001b
# CHECK-NEXT:          version: 4
# CHECK-NEXT:  prologue_length: 0x00000015
# CHECK-NEXT:  min_inst_length: 2
# CHECK-NEXT: max_ops_per_inst: 4
# CHECK-NEXT:  default_is_stmt: 0
# CHECK-NEXT:        line_base: 42
# CHECK-NEXT:       line_range: 10
# CHECK-NEXT:      opcode_base: 2
# CHECK-NEXT: standard_opcode_lengths[DW_LNS_copy] = 42
# CHECK-NEXT: include_directories[  1] = "baz"
# CHECK-NEXT: file_names[  1]:
# CHECK-NEXT:            name: "foo.c"
# CHECK-NEXT:       dir_index: 1
# CHECK-NEXT:        mod_time: 0x00000011
# CHECK-NEXT:          length: 0x00000022
# CHECK-EMPTY:
# CHECK-NEXT: debug_line[0x0000008c]

.section .debug_line,"",@progbits
    .long .Lunit0_end - .Lunit0_begin ## unit_length
.Lunit0_begin:
    .short 5 ## version
    .byte 8  ## address_size
    .byte 0  ## segment_selector_size
    .long .Lheader0_end - .Lheader0_begin ## header_length
.Lheader0_begin:
    .byte 1  ## minimum_instruction_length
    .byte 1  ## maximum_operations_per_instruction
    .byte 1  ## default_is_stmt
    .byte -5 ## line_base
    .byte 7  ## line_range
    ## Use an opcode_base > than the last standard opcode to show that unknown
    ## standard opcodes can be handled.
    .byte 14 ## opcode_base
    .byte 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0 ## standard_opcode_lengths
    .byte 2  ## directory_entry_format_count
    ## Use two formats to show that only the path is included in the output.
    .byte 0x1, 0x8 ## DW_LNCT_path, DW_FORM_string
    .byte 0x4, 0x5 ## DW_LNCT_size, DW_FORM_data2
    .byte 1  ## directories_count
    .asciz "dir1/dir2" ## directory entry 0
    .short 0x1234
    .byte 4  ## file_name_entry_format_count
    .byte 0x4, 0x1E ## DW_LNCT_MD5, DW_FORM_data16
    .byte 0x3, 0x6  ## DW_LNCT_timestamp, DW_FORM_data4
    .byte 0x2, 0xB  ## DW_LNCT_directory_index, DW_FORM_data1
    .byte 0x1, 0x8  ## DW_LNCT_path, DW_FORM_string
    .byte 1  ## file_names_count
    .quad 0x1111222233334444, 0x5555666677778888 ## file name entry 0
    .long 0x12345678
    .byte 2
    .asciz "file1.c"
.Lheader0_end:
    .byte 0x21 ## Special opcode - shows line printed after special opcode.
    .byte 0x4  ## DW_LNS_set_file - shows file can be changed to an arbitrary value.
    .byte 3
    .byte 0x5  ## DW_LNS_set_column - shows column can be changed.
    .byte 4
    .byte 0x1  ## DW_LNS_copy - shows line printed after copy opcode.
    .byte 0x8  ## DW_LNS_const_add_pc - shows address can be changed.
    .byte 0xC  ## DW_LNS_set_isa - shows isa register value can be changed.
    .byte 5
    .byte 0, 0x2, 0x4 ## DW_LNE_set_discriminator - shows discriminator can be changed.
    .byte 6
    ## These lines all show that the printed boolean register values can be changed.
    .byte 0x6  ## DW_LNS_negate_stmt
    .byte 0x7  ## DW_LNS_set_basic_block
    .byte 0xA  ## DW_LNS_set_prologue_end
    .byte 0xB  ## DW_LNS_set_epilogue_begin

    .byte 0xD  ## DW_LNS_unknown - shows that unknown opcodes do not affect state.
    .byte 0, 0x1, 0x1 ## DW_LNE_end_sequence
.Lunit0_end:

## Second line table program with version 4 and no sequences.
    .long .Lunit1_end - .Lunit1_begin ## unit_length
.Lunit1_begin:
    .short 4 ## version
    .long .Lheader1_end - .Lheader1_begin ## header_length
.Lheader1_begin:
    .byte 2  ## minimum_instruction_length
    .byte 4  ## maximum_operations_per_instruction
    .byte 0  ## default_is_stmt
    .byte 42 ## line_base
    .byte 10 ## line_range
    .byte 2  ## opcode_base - lower than normal to show this can be handled.
    .byte 42 ## standard_opcode_lengths - different to normal.
    ## include_directories
    .asciz "baz"
    .byte 0
    ## file_names
    .asciz "foo.c"
    .byte 1    ## Directory index
    .byte 0x11 ## Timestamp
    .byte 0x22 ## Length
.Lheader1_end:
.Lunit1_end:

## Third line table program needed to show that only a single blank line is
## printed after a program with no sequences. The values in this table are
## arbitrary.
    .long .Lunit2_end - .Lunit2_begin ## unit_length
.Lunit2_begin:
    .short 4 ## version
    .long .Lheader2_end - .Lheader2_begin ## header_length
.Lheader2_begin:
    .byte 1 ## minimum_instruction_length
    .byte 2 ## maximum_operations_per_instruction
    .byte 1 ## default_is_stmt
    .byte 1 ## line_base
    .byte 1 ## line_range
    .byte 2 ## opcode_base
    .byte 1 ## standard_opcode_lengths
    ## include_directories
    .byte 0
    ## file_names
    .asciz "bar.c"
    .byte 0 ## Directory index
    .byte 0 ## Timestamp
    .byte 0 ## Length
.Lheader2_end:
    .byte 0, 0x1, 0x1 ## DW_LNE_end_sequence
.Lunit2_end:
