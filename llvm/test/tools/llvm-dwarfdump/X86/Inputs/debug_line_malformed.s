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

# version 0
.long   .Lunit_v0_end - .Lunit_v0_start # unit length
.Lunit_v0_start:
.short  0               # version
.Lunit_v0_end:

# version 1
.long   .Lunit_v1_end - .Lunit_v1_start # unit length
.Lunit_v1_start:
.short  1               # version
.Lunit_v1_end:

# version 5 malformed line/include table
.long   .Lunit_v5_end - .Lunit_v5_start # unit length
.Lunit_v5_start:
.short  5               # version
.byte   8               # address size
.byte   8               # segment selector
.long   .Lprologue_v5_end-.Lprologue_v5_start # Length of Prologue
.Lprologue_v5_start:
.byte   1               # Minimum Instruction Length
.byte   1               # Maximum Operations per Instruction
.byte   1               # Default is_stmt
.byte   -5              # Line Base
.byte   14              # Line Range
.byte   13              # Opcode Base
.byte   0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1 # Standard Opcode Lengths
.byte   0               # directory table (invalid)
.Lprologue_v5_end:
.Lunit_v5_end:

# Short prologue
.long   .Lunit_short_prologue_end - .Lunit_short_prologue_start # unit length
.Lunit_short_prologue_start:
.short  4               # version
.long   .Lprologue_short_prologue_end-.Lprologue_short_prologue_start - 2 # Length of Prologue
.Lprologue_short_prologue_start:
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
.Lprologue_short_prologue_end:
.Lunit_short_prologue_end:

# Over-long prologue
.long   .Lunit_long_prologue_end - .Lunit_long_prologue_start # unit length
.Lunit_long_prologue_start:
.short  4               # version
.long   .Lprologue_long_prologue_end-.Lprologue_long_prologue_start + 1 # Length of Prologue
.Lprologue_long_prologue_start:
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
.Lprologue_long_prologue_end:
.Lunit_long_prologue_end:

# Over-long extended opcode
.long   .Lunit_long_opcode_end - .Lunit_long_opcode_start # unit length
.Lunit_long_opcode_start:
.short  4               # version
.long   .Lprologue_long_opcode_end-.Lprologue_long_opcode_start # Length of Prologue
.Lprologue_long_opcode_start:
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
.Lprologue_long_opcode_end:
.byte   0, 9, 2        # DW_LNE_set_address
.quad   0xabbadaba
.byte   0, 2, 1         # DW_LNE_end_sequence (wrong length)
.byte   0, 9, 2        # DW_LNE_set_address
.quad   0xbabb1e45
.byte   0, 1, 1         # DW_LNE_end_sequence (wrong length)
.Lunit_long_opcode_end:

# No end of sequence
.long   .Lunit_no_eos_end - .Lunit_no_eos_start # unit length
.Lunit_no_eos_start:
.short  4               # version
.long   .Lprologue_no_eos_end-.Lprologue_no_eos_start # Length of Prologue
.Lprologue_no_eos_start:
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
.Lprologue_no_eos_end:
.byte   0, 9, 2        # DW_LNE_set_address
.quad   0xdeadfade
.byte   1              # DW_LNS_copy
.Lunit_no_eos_end:

# Invalid prologue length
.long   .Linvalid_description_end0-.Linvalid_description_start0   # Length of Unit
.Linvalid_description_start0:
.short  5               # DWARF version number
.byte   8               # Address Size
.byte   0               # Segment Selector Size
.long   15              # Length of Prologue (invalid)
.Linvalid_description_params0:
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
.Linvalid_description_header_end0:
.byte   0,2,4,1         # DW_LNE_set_discriminator 1
.byte   1               # DW_LNS_copy
.byte   33              # address += 1, line += 1
.byte   0,1,1           # DW_LNE_end_sequence
.Linvalid_description_end0:

# Invalid file entry
.long   .Linvalid_file_end0-.Linvalid_file_start0   # Length of Unit
.Linvalid_file_start0:
.short  5               # DWARF version number
.byte   8               # Address Size
.byte   0               # Segment Selector Size
.long   .Linvalid_file_header_end0-.Linvalid_file_params0-7     # Length of Prologue (invalid)
.Linvalid_file_params0:
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
.Linvalid_file_header_end0:
.byte   0,2,4,1         # DW_LNE_set_discriminator 1
.byte   1               # DW_LNS_copy
.byte   33              # address += 1, line += 1
.byte   0,1,1           # DW_LNE_end_sequence
.Linvalid_file_end0:

# Invalid directory entry
.long   .Linvalid_dir_end0-.Linvalid_dir_start0   # Length of Unit
.Linvalid_dir_start0:
.short  5               # DWARF version number
.byte   8               # Address Size
.byte   0               # Segment Selector Size
.long   .Linvalid_dir_header_end0-.Linvalid_dir_params0-16     # Length of Prologue (invalid)
.Linvalid_dir_params0:
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
.Linvalid_dir_header_end0:
.byte   0,2,4,1         # DW_LNE_set_discriminator 1
.byte   1               # DW_LNS_copy
.byte   33              # address += 1, line += 1
.byte   0,1,1           # DW_LNE_end_sequence
.Linvalid_dir_end0:

# Invalid MD5 hash
.long   .Linvalid_md5_end0-.Linvalid_md5_start0   # Length of Unit
.Linvalid_md5_start0:
.short  5               # DWARF version number
.byte   8               # Address Size
.byte   0               # Segment Selector Size
.long   .Linvalid_md5_header_end0-.Linvalid_md5_params0     # Length of Prologue (invalid)
.Linvalid_md5_params0:
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
.byte   3               # 2 elements per file entry
.byte   1               # DW_LNCT_path
.byte   0x08            # DW_FORM_string
.byte   2               # DW_LNCT_directory_index
.byte   0x0b            # DW_FORM_data1
.byte   5               # DW_LNCT_MD5
.byte   0x09            # DW_FORM_data1
# File table entries
.byte   1               # 1 file
.asciz  "a.c"
.byte   0
.Linvalid_md5_header_end0:
.byte   0,2,4,1         # DW_LNE_set_discriminator 1
.byte   1               # DW_LNS_copy
.byte   33              # address += 1, line += 1
.byte   0,1,1           # DW_LNE_end_sequence
.Linvalid_md5_end0:

# Trailing good section
.long   .Lunit_good_end - .Lunit_good_start # Length of Unit (DWARF-32 format)
.Lunit_good_start:
.short  4               # DWARF version number
.long   .Lprologue_good_end-.Lprologue_good_start # Length of Prologue
.Lprologue_good_start:
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
.Lprologue_good_end:
.byte   0, 9, 2         # DW_LNE_set_address
.quad   0xcafebabe
.byte   0, 1, 1         # DW_LNE_end_sequence
.Lunit_good_end:

