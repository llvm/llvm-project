# RUN: llvm-mc -triple x86_64-unknown-linux %s -filetype=obj -o %t
# RUN: llvm-dwarfdump -debug-line %t 2>&1 | FileCheck %s

## The file name entry descriptors use the unknown form 0x4f. The prologue
## parses the descriptors itself, without the skipValue() validation that
## attribute parsing does, so the form reaches DWARFFormValue::extractValue().
## It must report the bad form instead of hitting llvm_unreachable().

# CHECK: error: failed to parse file entry because extracting the form value failed

    .section .debug_line, "", @progbits
.LBegin:
    .long .LEnd-.LVer           # Length of unit
.LVer:
    .short 5                    # Version
    .byte 8                     # Address size
    .byte 0                     # Segment selector size
    .long .LProg-.LAfterHL      # Header length
.LAfterHL:
    .byte 1                     # Minimum instruction length
    .byte 1                     # Maximum operations per instruction
    .byte 1                     # Default is_stmt
    .byte 0xfb                  # Line base (-5)
    .byte 14                    # Line range
    .byte 13                    # Opcode base
    .byte 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1   # Standard opcode lengths
    .byte 1                     # Directory entry format count
    .byte 1                     # DW_LNCT_path
    .byte 0x08                  # DW_FORM_string
    .byte 1                     # Directory count
    .asciz "d"                  # Directory entry
    .byte 1                     # File name entry format count
    .byte 1                     # DW_LNCT_path
    .byte 0x4f                  # Invalid form
    .byte 1                     # File name count
.LProg:
    .byte 0                     # Extended opcode
    .byte 1                     # Length
    .byte 1                     # DW_LNE_end_sequence
.LEnd: