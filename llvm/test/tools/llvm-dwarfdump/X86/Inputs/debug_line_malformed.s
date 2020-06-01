.section .debug_line,"",@progbits
# Leading good section.
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

# Version 0.
.long   .Lunit_v0_end - .Lunit_v0_start # unit length
.Lunit_v0_start:
.short  0               # version
.Lunit_v0_end:

# Version 1.
.long   .Lunit_v1_end - .Lunit_v1_start # unit length
.Lunit_v1_start:
.short  1               # version
.Lunit_v1_end:

# Version 5 malformed line/include table.
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
.byte   0               # directory table (invalid as no path component)
.Lprologue_v5_end:
.byte   0, 9, 2         # DW_LNE_set_address
.quad   0x8877665544332211
.byte   0, 1, 1         # DW_LNE_end_sequence
.Lunit_v5_end:

# Short prologue.
.long   .Lunit_short_prologue_end - .Lunit_short_prologue_start # unit length
.Lunit_short_prologue_start:
.short  4               # version
.long   .Lprologue_short_prologue_end-.Lprologue_short_prologue_start # Length of Prologue
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
.byte   1, 2, 3
.asciz "file2"
.byte   1, 2
.Lprologue_short_prologue_end:
.byte   6               # Read as part of the prologue,
                        # then later again as DW_LNS_negate_stmt.
# Header end
.byte   0, 9, 2         # DW_LNE_set_address
.quad   0x1122334455667788
.byte   0, 1, 1         # DW_LNE_end_sequence
.Lunit_short_prologue_end:

# Over-long prologue.
.long   .Lunit_long_prologue_end - .Lunit_long_prologue_start # unit length
.Lunit_long_prologue_start:
.short  4               # version
.long   .Lprologue_long_prologue_end-.Lprologue_long_prologue_start # Length of Prologue
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
.byte   1, 2, 3
.byte   0
# Skipped byte (treated as part of prologue).
.byte   6
.Lprologue_long_prologue_end:
.byte   0, 9, 2        # DW_LNE_set_address
.quad   0x1111222233334444
.byte   0, 1, 1        # DW_LNE_end_sequence
.Lunit_long_prologue_end:

# Incorrect length extended opcodes.
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
.byte   0, 9, 2         # DW_LNE_set_address
.quad   0xabbadaba
.byte   0, 2, 1         # DW_LNE_end_sequence (too long)
.byte   6               # DW_LNS_negate_stmt (but will be consumed with the end sequence above).
.byte   0, 1, 4         # DW_LNE_set_discriminator (too short)
.byte   0xa             # Parsed as argument for set_discriminator and also DW_LNS_set_prologue_end.
.byte   0, 9, 2         # DW_LNE_set_address
.quad   0xbabb1e45
.byte   0, 1, 1         # DW_LNE_end_sequence
.Lunit_long_opcode_end:

# No end of sequence.
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

# V5 very short prologue length.
.long   .Linvalid_description_end0-.Linvalid_description_start0   # Length of Unit
.Linvalid_description_start0:
.short  5               # DWARF version number
.byte   8               # Address Size
.byte   0               # Segment Selector Size
.long   .Linvalid_description_header_end0 - .Linvalid_description_params0 # Length of Prologue (invalid)
.Linvalid_description_params0:
.byte   1               # Minimum Instruction Length
.byte   1               # Maximum Operations per Instruction
.byte   1               # Default is_stmt
.byte   -5              # Line Base
.byte   14              # Line Range
.byte   13              # Opcode Base
.byte   0, 1, 1, 1, 1, 0, 0, 0, 1, 0 # Standard Opcode Lengths
.Linvalid_description_header_end0:
# The bytes from here onwards will also be read as part of the main body.
                        # --- Prologue interpretation --- | --- Main body interpretation ---
.byte   0, 1            # More standard opcodes           | First part of DW_LNE_end_sequence
# Directory table format
.byte   1               # One element per directory entry | End of DW_LNE_end_sequence
.byte   1               # DW_LNCT_path                    | DW_LNS_copy
.byte   0x08            # DW_FORM_string                  | DW_LNS_const_add_pc
# Directory table entries
.byte   1               # 1 directory                     | DW_LNS_copy
.asciz  "/tmp"          # Directory name                  | four special opcodes + start of DW_LNE_end_sequence
# File table format
.byte   1               # 1 element per file entry        | DW_LNE_end_sequence length
.byte   1               # DW_LNCT_path                    | DW_LNE_end_sequence opcode
.byte   0x08            # DW_FORM_string                  | DW_LNS_const_add_pc
# File table entries
.byte   1               # 1 file                          | DW_LNS_copy
.asciz  "xyz"           # File name                       | three special opcodes + start of DW_LNE_set_address
# Header end
.byte   9, 2            # Remainder of DW_LNE_set_address
.quad   0xbabb1ebabb1e
.byte   0, 1, 1         # DW_LNE_end_sequence
.Linvalid_description_end0:

# V5 prologue ends during file table.
.long   .Linvalid_file_end0-.Linvalid_file_start0   # Length of Unit
.Linvalid_file_start0:
.short  5               # DWARF version number
.byte   8               # Address Size
.byte   0               # Segment Selector Size
.long   .Linvalid_file_header_end0 - .Linvalid_file_params0 # Length of Prologue (invalid)
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
.Linvalid_file_header_end0:
# The bytes from here onwards will also be read as part of the main body.
                        # --- Prologue interpretation --- | --- Main body interpretation ---
.byte   0x0b            # DW_FORM_data1                   | DW_LNS_set_epilogue_begin
# File table entries
.byte   1               # 1 file                          | DW_LNS_copy
.asciz  "xyz"           # File name                       | 3 special opcodes + start of DW_LNE_end_sequence
.byte   1               # Dir index                       | DW_LNE_end_sequence length
# Header end
.byte   1               # DW_LNE_end_sequence opcode
.byte   0, 9, 2         # DW_LNE_set_address
.quad   0xab4acadab4a
.byte   0, 1, 1         # DW_LNE_end_sequence
.Linvalid_file_end0:

# V5 prologue ends during directory table.
.long   .Linvalid_dir_end0-.Linvalid_dir_start0   # Length of Unit
.Linvalid_dir_start0:
.short  5               # DWARF version number
.byte   8               # Address Size
.byte   0               # Segment Selector Size
.long   .Linvalid_dir_header_end0 - .Linvalid_dir_params0 # Length of Prologue (invalid)
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
.Linvalid_dir_header_end0:
# The bytes from here onwards will also be read as part of the main body.
                        # --- Prologue interpretation --- | --- Main body interpretation ---
.asciz  "/tmp"          # Directory name                  | 4 special opcodes + start of DW_LNE_end_sequence
# File table format
.byte   1               # 1 element per file entry        | DW_LNE_end_sequence length
.byte   1               # DW_LNCT_path                    | DW_LNE_end_sequence length opcode
.byte   0x08            # DW_FORM_string                  | DW_LNS_const_add_pc
# File table entries
.byte   1               # 1 file                          | DW_LNS_copy
.asciz  "xyz"           # File name                       | start of DW_LNE_set_address
# Header end
.byte   9, 2            # DW_LNE_set_address length + opcode
.quad   0x4444333322221111
.byte   0, 1, 1         # DW_LNE_end_sequence
.Linvalid_dir_end0:

# Invalid MD5 hash, where there is data still to be read afterwards.
.long   .Linvalid_md5_end0-.Linvalid_md5_start0   # Length of Unit
.Linvalid_md5_start0:
.short  5               # DWARF version number
.byte   8               # Address Size
.byte   0               # Segment Selector Size
.long   .Linvalid_md5_header_end0-.Linvalid_md5_params0     # Length of Prologue
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
.byte   3               # 3 elements per file entry
.byte   1               # DW_LNCT_path
.byte   0x08            # DW_FORM_string
.byte   5               # DW_LNCT_MD5
.byte   0x0b            # DW_FORM_data1
.byte   2               # DW_LNCT_directory_index
.byte   0x0b            # DW_FORM_data1
# File table entries
.byte   1               # 1 file
.asciz  "a.c"
.byte   0
# Data to show that the rest of the prologue is skipped.
.byte   1
.Linvalid_md5_header_end0:
.byte   0, 9, 2         # DW_LNE_set_address
.quad   0x1234123412341234
.byte   0, 1, 1         # DW_LNE_end_sequence
.Linvalid_md5_end0:

# Invalid MD5 hash, when data beyond the prologue length has
# been read before the MD5 problem is identified.
.long   .Linvalid_md5_end1-.Linvalid_md5_start1   # Length of Unit
.Linvalid_md5_start1:
.short  5               # DWARF version number
.byte   8               # Address Size
.byte   0               # Segment Selector Size
.long   .Linvalid_md5_header_end1 - .Linvalid_md5_params1 # Length of Prologue
.Linvalid_md5_params1:
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
.byte   5               # DW_LNCT_MD5
.Linvalid_md5_header_end1:
# The bytes from here onwards will also be read as part of the main body.
                        # --- Prologue interpretation --- | --- Main body interpretation ---
.byte   0x0b            # DW_FORM_data1                   | DW_LNS_set_epilogue_begin
# File table entries
.byte   1               # 1 file                          | DW_LNS_copy
.asciz  "xyz"           # File name                       | 3 special opcodes + DW_LNE_set_address start
.byte   9               # MD5 hash value                  | DW_LNE_set_address length
# Header end
.byte   2               # DW_LNE_set_address opcode
.quad   0x4321432143214321
.byte   0, 1, 1         # DW_LNE_end_sequence
.Linvalid_md5_end1:

# V5 invalid directory content description has unsupported form.
.long   .Linvalid_dir_form_end0-.Linvalid_dir_form_start0   # Length of Unit
.Linvalid_dir_form_start0:
.short  5               # DWARF version number
.byte   8               # Address Size
.byte   0               # Segment Selector Size
.long   .Linvalid_dir_form_header_end0 - .Linvalid_dir_form_params0
.Linvalid_dir_form_params0:
.byte   1               # Minimum Instruction Length
.byte   1               # Maximum Operations per Instruction
.byte   1               # Default is_stmt
.byte   -5              # Line Base
.byte   14              # Line Range
.byte   13              # Opcode Base
.byte   0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1 # Standard Opcode Lengths
# Directory table format
.byte   2               # Two elements per directory entry
.byte   1               # DW_LNCT_path
.byte   0x08            # DW_FORM_string
.byte   2               # DW_LNCT_directory_index (ignored)
.byte   0x7f            # Unknown form
# Directory table entries
.byte   2               # 2 directories
.asciz  "/foo"          # Directory name
.byte   0xff            # Arbitrary data for unknown form
.asciz  "/bar"          # Directory name
.byte   0xff            # Arbitrary data for unknown form
# File table format
.byte   1               # 1 element per file entry
.byte   1               # DW_LNCT_path
.byte   0x08            # DW_FORM_string
# File table entries
.byte   1               # 1 file
.asciz  "xyz"           # File names
.Linvalid_dir_form_header_end0:
.byte   0, 9, 2         # DW_LNE_set_address
.quad   0xaaaabbbbccccdddd
.byte   0, 1, 1         # DW_LNE_end_sequence
.Linvalid_dir_form_end0:

# Zero opcode base.
.long   .Lzero_opcode_base_end - .Lzero_opcode_base_start # unit length
.Lzero_opcode_base_start:
.short  4               # version
.long   .Lzero_opcode_base_prologue_end-.Lzero_opcode_base_prologue_start # Length of Prologue
.Lzero_opcode_base_prologue_start:
.byte   1               # Minimum Instruction Length
.byte   1               # Maximum Operations per Instruction
.byte   1               # Default is_stmt
.byte   0               # Line Base
.byte   1               # Line Range
.byte   0               # Opcode Base
.asciz "dir1"           # Include table
.byte   0
.asciz "file1"
.byte   1, 2, 3
.byte   0
.Lzero_opcode_base_prologue_end:
.byte   0, 9, 2        # DW_LNE_set_address
.quad   0xffffeeeeddddcccc
.byte   0x1            # Special opcode
.byte   0, 1, 1        # DW_LNE_end_sequence
.Lzero_opcode_base_end:

# V4 table with unterminated include directory table.
.long   .Lunterminated_include_end - .Lunterminated_include_start # unit length
.Lunterminated_include_start:
.short  4               # version
.long   .Lunterminated_include_prologue_end-.Lunterminated_include_prologue_start # Length of Prologue
.Lunterminated_include_prologue_start:
.byte   1               # Minimum Instruction Length
.byte   1               # Maximum Operations per Instruction
.byte   1               # Default is_stmt
.byte   -5              # Line Base
.byte   14              # Line Range
.byte   13              # Opcode Base
.byte   0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1 # Standard Opcode Lengths
.asciz  "dir1"          # Include table
.Lunterminated_include_prologue_end:
.byte   0, 9, 2         # DW_LNE_set_address
.quad   0xabcdef0123456789
.byte   0, 1, 1         # DW_LNE_end_sequence
.Lunterminated_include_end:

# V4 table with unterminated file name table.
.long   .Lunterminated_files_end - .Lunterminated_files_start # unit length
.Lunterminated_files_start:
.short  4               # version
.long   .Lunterminated_files_prologue_end-.Lunterminated_files_prologue_start # Length of Prologue
.Lunterminated_files_prologue_start:
.byte   1               # Minimum Instruction Length
.byte   1               # Maximum Operations per Instruction
.byte   1               # Default is_stmt
.byte   -5              # Line Base
.byte   14              # Line Range
.byte   13              # Opcode Base
.byte   0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1 # Standard Opcode Lengths
.asciz  "dir1"          # Include table
.byte   0
.asciz  "foo.c"         # File table
.byte   1, 2, 3
.Lunterminated_files_prologue_end:
.byte   0, 9, 2         # DW_LNE_set_address
.quad   0xababcdcdefef0909
.byte   0, 1, 1         # DW_LNE_end_sequence
.Lunterminated_files_end:

# Opcode extends past the end of the table, as claimed by the unit length field.
.long   .Lextended_past_end_end - .Lextended_past_end_start # Length of Unit
.Lextended_past_end_start:
.short  4               # DWARF version number
.long   .Lprologue_extended_past_end_end-.Lprologue_extended_past_end_start # Length of Prologue
.Lprologue_extended_past_end_start:
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
.Lprologue_extended_past_end_end:
.byte   0, 9, 2         # DW_LNE_set_address
.quad   0xfeedfeed
.byte   1               # DW_LNS_copy
.byte   0, 9, 2         # DW_LNE_set_address
.long   0xf001f000      # Truncated address (should be 8 bytes)
.byte   0xf0, 0, 1
.Lextended_past_end_end:

# Trailing good section.
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
