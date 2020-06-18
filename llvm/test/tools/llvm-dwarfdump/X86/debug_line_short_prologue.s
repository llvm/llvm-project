## Test cases when we run into the end of section while parsing a line table
## prologue.

# RUN: llvm-mc -triple x86_64-pc-linux %s -filetype=obj --defsym CASE=0 -o %t0
# RUN: llvm-dwarfdump -debug-line %t0 2>&1 | FileCheck %s --check-prefixes=ALL,C0

# RUN: llvm-mc -triple x86_64-pc-linux %s -filetype=obj --defsym CASE=1 -o %t1
# RUN: llvm-dwarfdump -debug-line %t1 2>&1 | FileCheck %s --check-prefixes=ALL,C1

# RUN: llvm-mc -triple x86_64-pc-linux %s -filetype=obj --defsym CASE=2 -o %t2
# RUN: llvm-dwarfdump -debug-line %t2 2>&1 | FileCheck %s --check-prefixes=ALL,C2

# RUN: llvm-mc -triple x86_64-pc-linux %s -filetype=obj --defsym CASE=3 -o %t3
# RUN: llvm-dwarfdump -debug-line %t3 2>&1 | FileCheck %s --check-prefixes=ALL,OK

# ALL:      debug_line[0x00000000]
# C0-NEXT:  warning: parsing line table prologue at 0x00000000 found an invalid directory or file table description at 0x00000027
# C0-NEXT:  warning: failed to parse entry content descriptors: unexpected end of data at offset 0x27
# C1-NEXT:  warning: parsing line table prologue at 0x00000000 found an invalid directory or file table description at 0x0000002a
# C1-NEXT:  warning: failed to parse entry content descriptors: unable to decode LEB128 at offset 0x0000002a: malformed uleb128, extends past end
# C2-NEXT:  warning: parsing line table prologue at 0x00000000 found an invalid directory or file table description at 0x0000002b
# C2-NEXT:  warning: failed to parse entry content descriptors: unable to decode LEB128 at offset 0x0000002b: malformed uleb128, extends past end
# ALL:      include_directories[  0] = "/tmp"
# OK:       file_names[  0]:
# OK-NEXT:             name: "foo"
# OK-NEXT:        dir_index: 0

.section .debug_line,"",@progbits
.long   .Lend-.Lstart   # Length of Unit
.Lstart:
.short  5               # DWARF version number
.byte   8               # Address Size
.byte   0               # Segment Selector Size
.long   .Lprologue_end-.Lprologue_start  # Length of Prologue
.Lprologue_start:
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
.if CASE >= 1
.byte   2               # 2 elements per file entry
.byte   1               # DW_LNCT_path
.byte   8               # DW_FORM_string
.if CASE >= 2
.byte   2               # DW_LNCT_directory_index
.if CASE >= 3
.byte   11              # DW_FORM_data1
# File table entries
.byte   1
.asciz  "foo"
.byte   0
.endif
.endif
.endif

.Lprologue_end:
.Lend:
