# RUN: llvm-mc %s -filetype obj -triple x86_64-unknown-elf -o - | \
# RUN:   llvm-dwarfdump -debug-info - | \
# RUN:   FileCheck %s

        .section .debug_abbrev,"",@progbits
        .byte 0x01  # Abbrev code
        .byte 0x11  # DW_TAG_compile_unit
        .byte 0x00  # DW_CHILDREN_no
        .byte 0x13  # DW_AT_language
        .byte 0x05  # DW_FORM_data2
        .byte 0x00  # EOM(1)
        .byte 0x00  # EOM(2)
        .byte 0x00  # EOM(3)
        
        .section .debug_info,"",@progbits
# CHECK: .debug_info contents:
# CHECK-NEXT: 0x00000000: Compile Unit:
DI_4_64_start:
        .long 0xffffffff       # DWARF64 mark
        .quad DI_4_64_end - DI_4_64_version # Length of Unit
# CHECK-SAME: length = 0x000000000000000f
# CHECK-SAME: format = DWARF64
DI_4_64_version:
        .short 4               # DWARF version number
# CHECK-SAME: version = 0x0004
        .quad .debug_abbrev    # Offset Into Abbrev. Section
# CHECK-SAME: abbr_offset = 0x0000
        .byte 8                # Address Size (in bytes)
# CHECK-SAME: addr_size = 0x08
# CHECK-SAME: (next unit at 0x0000001b)

        .byte 1                # Abbreviation code
# CHECK: 0x00000017: DW_TAG_compile_unit
        .short 4               # DW_LANG_C_plus_plus
# CHECK-NEXT: DW_AT_language (DW_LANG_C_plus_plus)
        .byte 0                # NULL
DI_4_64_end:

