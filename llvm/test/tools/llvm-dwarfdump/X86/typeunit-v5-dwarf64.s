# RUN: llvm-mc %s -filetype obj -triple x86_64-unknown-elf -o - | \
# RUN:   llvm-dwarfdump -debug-info - | \
# RUN:   FileCheck %s

        .section .debug_abbrev,"",@progbits
        .byte 0x01  # Abbrev code
        .byte 0x41  # DW_TAG_type_unit
        .byte 0x01  # DW_CHILDREN_yes
        .byte 0x17  # DW_AT_visibility
        .byte 0x0b  # DW_FORM_data1
        .byte 0x00  # EOM(1)
        .byte 0x00  # EOM(2)
        .byte 0x02  # Abbrev code
        .byte 0x13  # DW_TAG_structure_type
        .byte 0x00  # DW_CHILDREN_no (no members)
        .byte 0x17  # DW_AT_visibility
        .byte 0x0b  # DW_FORM_data1
        .byte 0x00  # EOM(1)
        .byte 0x00  # EOM(2)
        .byte 0x00  # EOM(3)

        .section .debug_info,"",@progbits
# CHECK: .debug_info contents:
# CHECK-NEXT: 0x00000000: Type Unit:
TU_5_64_start:
        .long 0xffffffff            # DWARF64 mark
        .quad TU_5_64_end-TU_5_64_version  # Length of Unit
# CHECK-SAME: length = 0x0000000000000022
TU_5_64_version:
        .short 5                    # DWARF version number
# CHECK-SAME: version = 0x0005
        .byte 2                     # DW_UT_type
        .byte 8                     # Address Size (in bytes)
        .quad .debug_abbrev         # Offset Into Abbrev. Section
# CHECK-SAME: abbr_offset = 0x0000
# CHECK-SAME: addr_size = 0x08
# CHECK-SAME: name = ''
        .quad 0x0011223344556677    # Type Signature
# CHECK-SAME: type_signature = 0x0011223344556677
        .quad TU_5_64_type-TU_5_64_start # Type offset
# CHECK-SAME: type_offset = 0x002a
# CHECK-SAME: (next unit at 0x0000002e)

        .byte 1                     # Abbreviation code
# CHECK: 0x00000028: DW_TAG_type_unit
        .byte 1                     # DW_VIS_local
# CHECK-NEXT: DW_AT_visibility (DW_VIS_local)

TU_5_64_type:
        .byte 2                     # Abbreviation code
# CHECK: 0x0000002a: DW_TAG_structure_type
        .byte 1                     # DW_VIS_local
# CHECK-NEXT: DW_AT_visibility (DW_VIS_local)

        .byte 0 # NULL
# CHECK: 0x0000002c: NULL
        .byte 0 # NULL
TU_5_64_end:
