# Dump the complete .debug_line.dwo, then just one part.
#
# RUN: llvm-mc -triple x86_64-unknown-unknown -filetype=obj %s -o %t.o
# RUN: llvm-dwarfdump -debug-line %t.o | FileCheck %s --check-prefixes=PART1,PART2
# RUN: llvm-dwarfdump -debug-line=0x32 %t.o | FileCheck %s --check-prefix=PART2

        .section .debug_line.dwo,"e",@progbits
LH_1_start:
        .long   LH_1_end-LH_1_version   # Length of Unit
LH_1_version:
        .short  4               # DWARF version number
        .long   LH_1_header_end-LH_1_params     # Length of Prologue
LH_1_params:
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
        # Directory table
        .asciz  "Directory1"
        .byte   0
        # File table
        .asciz  "File1"         # File name
        .byte   1               # Directory index
        .byte   0x41            # Timestamp
        .byte   0x42            # File Size
        .byte   0               # End of list
LH_1_header_end:
        # Line number program, which is empty.
LH_1_end:

# PART1:      Line table prologue:
# PART1-NEXT: total_length: 0x0000002e
# PART1-NEXT: format: DWARF32
# PART1-NEXT: version: 4
# PART1-NEXT: prologue_length: 0x00000028
# PART1:      include_directories[  1] = "Directory1"
# PART1:      file_names[  1]
# PART1:      name: "File1"

# Second line table.
LH_2_start:
        .long   LH_2_end-LH_2_version   # Length of Unit
LH_2_version:
        .short  4               # DWARF version number
        .long   LH_2_header_end-LH_2_params     # Length of Prologue
LH_2_params:
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
        # Directory table
        .asciz  "Dir2"
        .byte   0
        # File table
        .asciz  "File2"         # File name
        .byte   1               # Directory index
        .byte   0x14            # Timestamp
        .byte   0x24            # File Size
        .byte   0               # End of list
LH_2_header_end:
        # Line number program, which is empty.
LH_2_end:

# PART2:      Line table prologue:
# PART2-NEXT: total_length: 0x00000028
# PART2-NEXT: format: DWARF32
# PART2-NEXT: version: 4
# PART2-NEXT: prologue_length: 0x00000022
# PART2-NOT:  prologue:
# PART2:      include_directories[  1] = "Dir2"
# PART2:      file_names[  1]
# PART2:      name: "File2"
# PART2-NOT:  prologue:
