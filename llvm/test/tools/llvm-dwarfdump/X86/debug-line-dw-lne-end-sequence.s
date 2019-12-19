## Show that the DW_LNE_end_sequence opcode resets the line state
## properly and the rows are printed correctly.

# RUN: llvm-mc -filetype obj -triple x86_64 %s -o %t.o
# RUN: llvm-dwarfdump --debug-line %t.o | FileCheck %s --check-prefixes=HEADER,ROWS
# RUN: llvm-dwarfdump --debug-line %t.o --verbose | FileCheck %s --check-prefix=ROWS

# HEADER:      Address            Line   Column File   ISA Discriminator Flags
# HEADER-NEXT: ------------------ ------ ------ ------ --- ------------- -------------
# ROWS:        0x0000000012345678      1      0      1   0             1  is_stmt basic_block prologue_end epilogue_begin end_sequence
# ROWS:        0x0000000000000001      2      0      1   0             0  is_stmt
# ROWS:        0x0000000000000001      2      0      1   0             0  is_stmt end_sequence

.section .debug_line,"",@progbits
.Line_table_start0:
  .long   .Line_table_end0-.Line_table_start0-4   # Length of Unit
  .short  5               # DWARF version number
  .byte   8               # Address Size
  .byte   0               # Segment Selector Size
  .long   .Line_table_header_end0-.Line_table_params0     # Length of Prologue
.Line_table_params0:
  .byte   1               # Minimum Instruction Length
  .byte   1               # Maximum Operations per Instruction
  .byte   1               # Default is_stmt
  .byte   -5              # Line Base
  .byte   14              # Line Range
  .byte   13              # Opcode Base
  .byte   0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1 # Standard Opcode Lengths
  # Directory table format
  .byte   1               # One element per directory entry
  .byte   1               # DW_LNCT_path
  .byte   0x08            # DW_FORM_string
  # Directory table entries
  .byte   1               # 1 directory
  .asciz  "/tmp"
  # File table format
  .byte   2               # 2 elements per file entry
  .byte   1               # DW_LNCT_path
  .byte   0x08            # DW_FORM_string
  .byte   2               # DW_LNCT_directory_index
  .byte   0x0b            # DW_FORM_data1
  # File table entries
  .byte   1               # 1 file
  .asciz  "a.c"
  .byte   0
.Line_table_header_end0:
  .byte   0,9,2           # DW_LNE_set_address
  .quad   0x12345678
  .byte   7               # DW_LNS_set_basic_block
  .byte   10              # DW_LNS_set_prologue_end
  .byte   11              # DW_LNS_set_epilogue_begin
  .byte   0,2,4,1         # DW_LNE_set_discriminator 1
  .byte   0,1,1           # DW_LNE_end_sequence
  .byte   33              # address += 1, line += 1
  .byte   0,1,1           # DW_LNE_end_sequence
.Line_table_end0:
