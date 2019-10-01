# RUN: llvm-mc -filetype obj -triple x86_64-pc-linux %s -o %t.o
# RUN: llvm-dwarfdump -debug-line %t.o | FileCheck %s

# CHECK:      Address            Line   Column File   ISA Discriminator Flags
# CHECK-NEXT: ------------------ ------ ------ ------ --- ------------- -------------
# CHECK-NEXT: 0x0000000000000000      1      0      1   0             1  is_stmt
# CHECK-NEXT: 0x0000000000000001      2      0      1   0             0  is_stmt
# CHECK-NEXT: 0x0000000000000001      2      0      1   0             0  is_stmt end_sequence

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
  .byte   0,2,4,1         # DW_LNE_set_discriminator 1
  .byte   1               # DW_LNS_copy
  .byte   33              # address += 1, line += 1
  .byte   0,1,1           # DW_LNE_end_sequence
.Line_table_end0:
