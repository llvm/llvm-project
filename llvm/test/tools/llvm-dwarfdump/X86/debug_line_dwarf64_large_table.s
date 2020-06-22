## Test that we can dump the (intact) prologue of a large table which was
## truncated. Also, make sure we don't get confused by a DWARF64 length which
## matches one of the reserved initial length values.

# RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux %s >%t
# RUN: llvm-dwarfdump %t -debug-line 2>&1 | FileCheck %s

# CHECK: debug_line[0x00000000]
# CHECK-NEXT: Line table prologue:
# CHECK-NEXT:     total_length: 0x00000000fffffff0
# CHECK-NEXT:           format: DWARF64
# CHECK-NEXT:          version: 4
# CHECK-NEXT:  prologue_length: 0x0000000000000016
# CHECK:        file_names[ 1]:
# CHECK-NEXT:             name: "file1"
# CHECK-NEXT:        dir_index: 0
# CHECK-NEXT:         mod_time: 0x00000000
# CHECK-NEXT:           length: 0x00000000
# CHECK-NEXT: warning: line table program with offset 0x00000000 has length 0xfffffffc but only 0x0000003a bytes are available

# CHECK:      0x000000000badbeef      1      0      1   0             0  is_stmt end_sequence

.section .debug_line,"",@progbits
.long   0xffffffff      # Length of Unit (DWARF-64 format)
.quad   0xfffffff0
.short  4               # DWARF version number
.quad   .Lprologue1_end-.Lprologue1_start # Length of Prologue
.Lprologue1_start:
.byte   1               # Minimum Instruction Length
.byte   1               # Maximum Operations per Instruction
.byte   1               # Default is_stmt
.byte   -5              # Line Base
.byte   14              # Line Range
.byte   1               # Opcode Base
.asciz "dir1"           # Include table
.byte   0
.asciz "file1"          # File table
.byte   0, 0, 0
.byte   0
.Lprologue1_end:
.byte   0, 9, 2         # DW_LNE_set_address
.quad   0x0badbeef
.byte   0, 1, 1         # DW_LNE_end_sequence
