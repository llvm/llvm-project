## Test cases when we run into the end of section while parsing a line table
## prologue.

# RUN: llvm-mc -triple x86_64-pc-linux %s -filetype=obj --defsym CASE=0 -o %t0
# RUN: llvm-dwarfdump -debug-line %t0 2>&1 | FileCheck %s --check-prefixes=ALL,C0

# RUN: llvm-mc -triple x86_64-pc-linux %s -filetype=obj --defsym CASE=1 -o %t1
# RUN: llvm-dwarfdump -debug-line %t1 2>&1 | FileCheck %s --check-prefixes=ALL,C1

# RUN: llvm-mc -triple x86_64-pc-linux %s -filetype=obj --defsym CASE=2 -o %t2
# RUN: llvm-dwarfdump -debug-line %t2 2>&1 | FileCheck %s --check-prefixes=ALL,C1

# RUN: llvm-mc -triple x86_64-pc-linux %s -filetype=obj --defsym CASE=3 -o %t3
# RUN: llvm-dwarfdump -debug-line %t3 2>&1 | FileCheck %s --check-prefixes=ALL,C1

# RUN: llvm-mc -triple x86_64-pc-linux %s -filetype=obj --defsym CASE=4 -o %t4
# RUN: llvm-dwarfdump -debug-line %t4 2>&1 | FileCheck %s --check-prefixes=ALL,C1

# RUN: llvm-mc -triple x86_64-pc-linux %s -filetype=obj --defsym CASE=5 -o %t5
# RUN: llvm-dwarfdump -debug-line %t5 2>&1 | FileCheck %s --check-prefixes=ALL,OK

# ALL:      debug_line[0x00000000]

# C0-NEXT:  warning: parsing line table prologue at 0x00000000 found an invalid directory or file table description at 0x00000021
# C0-NEXT:  warning: include directories table was not null terminated before the end of the prologue
# C0:       include_directories[  1] = "dir1"

# C1-NEXT:  warning: parsing line table prologue at 0x00000000 found an invalid directory or file table description
# C1-NEXT:  warning: file names table was not null terminated before the end of the prologue
# C1:       include_directories[  2] = "dir2"
# C1-NEXT:  file_names[  1]:
# C1-NEXT:             name: "file1"
# C1-NEXT:        dir_index: 1
# C1-NEXT:         mod_time: 0x00000002
# C1-NEXT:           length: 0x00000003

# OK:       file_names[  2]:
# OK-NEXT:             name: "file2"
# OK-NEXT:        dir_index: 1
# OK-NEXT:         mod_time: 0x00000005
# OK-NEXT:           length: 0x00000006

.section .debug_line,"",@progbits
.long   .Lend-.Lstart   # Length of Unit
.Lstart:
.short  4               # DWARF version number
.long   .Lprologue_end-.Lprologue_start  # Length of Prologue
.Lprologue_start:
.byte   1               # Minimum Instruction Length
.byte   1               # Maximum Operations per Instruction
.byte   1               # Default is_stmt
.byte   -5              # Line Base
.byte   14              # Line Range
.byte   13              # Opcode Base
.byte   0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1 # Standard Opcode Lengths
.asciz "dir1"           # Include table
.if CASE >= 1
.asciz "dir2"
.byte   0
.asciz "file1"          # File table
.byte   1, 2, 3
.if CASE >= 2
.asciz "file2"
.if CASE >= 3
.byte 1
.if CASE >= 4
.byte 5
.if CASE >= 5
.byte 6
.byte 0
.endif
.endif
.endif
.endif
.endif

.Lprologue_end:
.Lend:
