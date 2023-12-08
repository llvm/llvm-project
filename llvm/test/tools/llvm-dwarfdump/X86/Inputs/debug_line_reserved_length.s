.section .debug_line,"",@progbits
# Leading good section
.long   .Lunit1_end - .Lunit1_start # Length of Unit (DWARF-32 format)
.Lunit1_start:
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
.asciz "dir1"           # Include table
.asciz "dir2"
.byte   0
.asciz "file1"          # File table
.byte   0, 0, 0
.asciz "file2"
.byte   1, 0, 0
.byte   0
.Lprologue1_end:
.byte   0, 9, 2         # DW_LNE_set_address
.quad   0x0badbeef
.byte   0, 1, 1         # DW_LNE_end_sequence
.Lunit1_end:

# Malformed section
.long   0xfffffffe      # reserved unit length

# Trailing good section
.long   .Lunit3_end - .Lunit3_start # Length of Unit (DWARF-32 format)
.Lunit3_start:
.short  4               # DWARF version number
.long   .Lprologue3_end-.Lprologue3_start # Length of Prologue
.Lprologue3_start:
.byte   1               # Minimum Instruction Length
.byte   1               # Maximum Operations per Instruction
.byte   1               # Default is_stmt
.byte   -5              # Line Base
.byte   14              # Line Range
.byte   13              # Opcode Base
.byte   0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1 # Standard Opcode Lengths
.asciz "dir1"           # Include table
.asciz "dir2"
.byte   0
.asciz "file1"          # File table
.byte   0, 0, 0
.asciz "file2"
.byte   1, 0, 0
.byte   0
.Lprologue3_end:
.byte   0, 9, 2         # DW_LNE_set_address
.quad   0xcafebabe
.byte   0, 1, 1         # DW_LNE_end_sequence
.Lunit3_end:
