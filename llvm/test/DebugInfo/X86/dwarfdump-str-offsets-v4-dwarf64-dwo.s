## This tests dumping a .debug_str_offsets.dwo section which is referenced by
## DWARF64 pre-v5 units and dumping attributes in such units which use the
## DW_FORM_strx form.

# RUN: llvm-mc -triple x86_64 %s -filetype=obj -o - | \
# RUN:   llvm-dwarfdump -v - | \
# RUN:   FileCheck %s

# CHECK:      .debug_info.dwo contents:
# CHECK:      Compile Unit:
# CHECK:      DW_TAG_compile_unit [1]
# CHECK-NEXT:   DW_AT_name [DW_FORM_strx] (indexed (00000000) string = "Compilation Unit")

# CHECK:      .debug_types.dwo contents:
# CHECK:      Type Unit:
# CHECK:      DW_TAG_type_unit [2] *
# CHECK-NEXT:   DW_AT_name [DW_FORM_strx] (indexed (00000001) string = "Type Unit")
# CHECK:      DW_TAG_structure_type [3]  
# CHECK-NEXT:   DW_AT_name [DW_FORM_strx] (indexed (00000002) string = "Structure")

# CHECK:      .debug_str.dwo contents:
# CHECK-NEXT: 0x00000000: "Compilation Unit"
# CHECK-NEXT: 0x00000011: "Type Unit"
# CHECK-NEXT: 0x0000001b: "Structure"

# CHECK:      .debug_str_offsets.dwo contents:
# CHECK-NEXT: 0x00000000: Contribution size = 24, Format = DWARF64, Version = 4
# CHECK-NEXT: 0x00000000: 0000000000000000 "Compilation Unit"
# CHECK-NEXT: 0x00000008: 0000000000000011 "Type Unit"
# CHECK-NEXT: 0x00000010: 000000000000001b "Structure"

    .section .debug_str.dwo, "MSe", @progbits, 1
.LStr0:
    .asciz "Compilation Unit"
.LStr1:
    .asciz "Type Unit"
.LStr2:
    .asciz "Structure"

    .section .debug_str_offsets.dwo, "e", @progbits
    .quad .LStr0-.debug_str.dwo     # 0: "Compilation Unit"
    .quad .LStr1-.debug_str.dwo     # 1: "Type Unit"
    .quad .LStr2-.debug_str.dwo     # 2: "Structure"

    .section .debug_abbrev.dwo, "e", @progbits
    .uleb128 0x01                   # Abbrev code
    .uleb128 0x11                   # DW_TAG_compile_unit
    .byte 0x00                      # DW_CHILDREN_no
    .uleb128 0x03                   # DW_AT_name
    .uleb128 0x1a                   # DW_FORM_strx
    .byte 0x00                      # EOM(1)
    .byte 0x00                      # EOM(2)
    .uleb128 0x02                   # Abbrev code
    .uleb128 0x41                   # DW_TAG_type_unit
    .byte 0x01                      # DW_CHILDREN_yes
    .uleb128 0x03                   # DW_AT_name
    .uleb128 0x1a                   # DW_FORM_strx
    .byte 0x00                      # EOM(1)
    .byte 0x00                      # EOM(2)
    .uleb128 0x03                   # Abbrev code
    .uleb128 0x13                   # DW_TAG_structure_type
    .byte 0x00                      # DW_CHILDREN_no (no members)
    .uleb128 0x03                   # DW_AT_name
    .uleb128 0x1a                   # DW_FORM_strx
    .byte 0x00                      # EOM(1)
    .byte 0x00                      # EOM(2)
    .byte 0x00                      # EOM(3)

    .section .debug_info.dwo, "e", @progbits
    .long 0xffffffff                # DWARF64 mark
    .quad .LCUEnd-.LCUVer           # Length
.LCUVer:
    .short 4                        # Version
    .quad 0                         # Abbrev. offset
    .byte 8                         # Address size
    .uleb128 1                      # Abbrev [1] DW_TAG_compile_unit
    .uleb128 0                      # DW_AT_name ("Compilation Unit")
.LCUEnd:

    .section .debug_types.dwo, "e", @progbits
.LTU:
    .long 0xffffffff                # DWARF64 mark
    .quad .LTUEnd-.LTUVer           # Length
.LTUVer:
    .short 4                        # Version
    .quad 0                         # Abbrev. offset
    .byte 8                         # Address size
    .quad 0x11110022ffffffff        # Type Signature
    .quad .LTUType-.LTU             # Type offset
    .uleb128 2                      # Abbrev [2] DW_TAG_type_unit
    .uleb128 1                      # DW_AT_name ("Type Unit")
.LTUType:
    .uleb128 3                      # Abbrev [3] DW_TAG_structure_type
    .uleb128 2                      # DW_AT_name ("Structure")
.LTUEnd:
