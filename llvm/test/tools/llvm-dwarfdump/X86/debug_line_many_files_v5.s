## An object with many files and directories in a single debug_line contribution
## meant to test the handling of directory_count and file_name_count fields.

# RUN: llvm-mc -triple x86_64-pc-linux -filetype=obj %s -o %t
# RUN: llvm-dwarfdump -debug-line %t | FileCheck %s

# CHECK:      include_directories[  0] = "/d000"
# CHECK:      include_directories[299] = "/d299"
# CHECK:      file_names[  0]:
# CHECK-NEXT:            name: "000.c"
# CHECK-NEXT:       dir_index: 0
# CHECK:      file_names[299]:
# CHECK-NEXT:            name: "299.c"
# CHECK-NEXT:       dir_index: 299

.section .debug_line,"",@progbits
.long   .Lunit_end0-.Lunit_start0   # Length of Unit
.Lunit_start0:
.short  5               # DWARF version number
.byte   8               # Address Size
.byte   0               # Segment Selector Size
.long   .Lunit_header_end0 - .Lunit_params0 # Length of Prologue (invalid)
.Lunit_params0:
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
.uleb128 300            # 300 directories
.irpc a,012
.irpc b,0123456789
.irpc c,0123456789
.byte '/', 'd', '0'+\a, '0'+\b, '0'+\c, 0
.endr
.endr
.endr

# File table format
.byte   2               # 2 elements per file entry
.byte   1               # DW_LNCT_path
.byte   0x08            # DW_FORM_string
.byte   2               # DW_LNCT_directory_index
.byte   0x05            # DW_FORM_data2

# File table entries
.uleb128 300            # 300 files
.irpc a,012
.irpc b,0123456789
.irpc c,0123456789
.byte '0'+\a, '0'+\b, '0'+\c, '.', 'c', 0 # File name
.word \a*100+\b*10+\c   # Dir index
.endr
.endr
.endr

.Lunit_header_end0:
.byte   0, 1, 1         # DW_LNE_end_sequence
.Lunit_end0:
