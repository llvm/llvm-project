# RUN: llvm-mc -triple x86_64-unknown-linux %s -filetype=obj -o %t.o
# RUN: llvm-dwarfdump -v %t.o 2>&1 | FileCheck %s
#
# Test object to verify that llvm-dwarfdump handles an invalid string offsets
# table.
#
# A rudimentary abbrev section.
        .section .debug_abbrev,"",@progbits
        .byte 0x01  # Abbrev code
        .byte 0x11  # DW_TAG_compile_unit
        .byte 0x00  # DW_CHILDREN_no
        .byte 0x72  # DW_AT_str_offsets_base
        .byte 0x17  # DW_FORM_sec_offset
        .byte 0x00  # EOM(1)
        .byte 0x00  # EOM(2)
        .byte 0x00  # EOM(3)

# A rudimentary compile unit to convince dwarfdump that we are dealing with a 
# DWARF v5 string offsets table.
        .section .debug_info,"",@progbits

# DWARF v5 32 bit CU header.
        .long  CU1_end-CU1_begin  # Length of Unit
CU1_begin:
        .short 5               # DWARF version number
        .byte 1                # DWARF Unit Type
        .byte 8                # Address Size (in bytes)
        .long .debug_abbrev    # Offset Into Abbrev. Section
        .byte 1                # Abbreviation code: DW_TAG_compile_unit
	.long 0                # DW_AT_str_offsets_base
CU1_end:

# DWARF v5 64 bit CU header.
	.long 0xffffffff
        .quad  CU2_end-CU2_begin  # Length of Unit
CU2_begin:
        .short 5               # DWARF version number
        .byte 1                # DWARF Unit Type
        .byte 8                # Address Size (in bytes)
        .quad .debug_abbrev    # Offset Into Abbrev. Section
        .byte 1                # Abbreviation code: DW_TAG_compile_unit
	.quad 0                # DW_AT_str_offsets_base
CU2_end:
        .long  CU3_end-CU3_begin  # Length of Unit
CU3_begin:
        .short 5               # DWARF version number
        .byte 1                # DWARF Unit Type
        .byte 8                # Address Size (in bytes)
        .long .debug_abbrev    # Offset Into Abbrev. Section
        .byte 1                # Abbreviation code: DW_TAG_compile_unit
	.quad .str_off0        # DW_AT_str_offsets_base
CU3_end:
        .long  CU4_end-CU4_begin  # Length of Unit
CU4_begin:
        .short 5               # DWARF version number
        .byte 1                # DWARF Unit Type
        .byte 8                # Address Size (in bytes)
        .long .debug_abbrev    # Offset Into Abbrev. Section
        .byte 1                # Abbreviation code: DW_TAG_compile_unit
	.quad .str_off1        # DW_AT_str_offsets_base
CU4_end:
        .long  CU5_end-CU5_begin  # Length of Unit
CU5_begin:
        .short 5               # DWARF version number
        .byte 1                # DWARF Unit Type
        .byte 8                # Address Size (in bytes)
        .long .debug_abbrev    # Offset Into Abbrev. Section
        .byte 1                # Abbreviation code: DW_TAG_compile_unit
        .long .str_off2_begin  # DW_AT_str_offsets_base
CU5_end:
        .long  CU6_end-CU6_begin  # Length of Unit
CU6_begin:
        .short 5               # DWARF version number
        .byte 1                # DWARF Unit Type
        .byte 8                # Address Size (in bytes)
        .long .debug_abbrev    # Offset Into Abbrev. Section
        .byte 1                # Abbreviation code: DW_TAG_compile_unit
        .long .str_off3_begin  # DW_AT_str_offsets_base
CU6_end:
	.long 0xffffffff
        .quad  CU7_end-CU7_begin  # Length of Unit
CU7_begin:
        .short 5               # DWARF version number
        .byte 1                # DWARF Unit Type
        .byte 8                # Address Size (in bytes)
        .quad .debug_abbrev    # Offset Into Abbrev. Section
        .byte 1                # Abbreviation code: DW_TAG_compile_unit
	.quad .str_off4_begin  # DW_AT_str_offsets_base
CU7_end:
        .long  CU8_end-CU8_begin  # Length of Unit
CU8_begin:
        .short 5               # DWARF version number
        .byte 1                # DWARF Unit Type
        .byte 8                # Address Size (in bytes)
        .long .debug_abbrev    # Offset Into Abbrev. Section
        .byte 1                # Abbreviation code: DW_TAG_compile_unit
        .long .str_off_end+16  # DW_AT_str_offsets_base
CU8_end:
	.long 0xffffffff
        .quad  CU9_end-CU9_begin  # Length of Unit
CU9_begin:
        .short 5               # DWARF version number
        .byte 1                # DWARF Unit Type
        .byte 8                # Address Size (in bytes)
        .quad .debug_abbrev    # Offset Into Abbrev. Section
        .byte 1                # Abbreviation code: DW_TAG_compile_unit
	.quad .str_off_end+8  # DW_AT_str_offsets_base
CU9_end:

        .section .debug_str_offsets,"",@progbits
# Invalid length
        .long 0xfffffff4
        .short 5    # DWARF version
        .short 0    # Padding
.str_off0:
        .long 0
# Length beyond section bounds
        .long .str_off_end-.str_off1+8
        .short 5    # DWARF version
        .short 0    # Padding
.str_off1:
        .long 0
# Length intrudes on following unit
        .long .str_off2_end-.str_off2_begin+8
        .short 5    # DWARF version
        .short 0    # Padding
.str_off2_begin:
        .long 0
.str_off2_end:
# Plain contribution, no errors here
        .long .str_off3_end-.str_off3_begin
        .short 5    # DWARF version
        .short 0    # Padding
.str_off3_begin:
        .long 0
.str_off3_end:
# 32 bit contribution referenced from a 64 bit unit
        .long .str_off4_end-.str_off4_begin
        .short 5    # DWARF version
        .short 0    # Padding
.str_off4_begin:
        .long 0
.str_off4_end:
.str_off_end:


# CHECK: error: invalid contribution to string offsets table in section .debug_str_offsets[.dwo]: insufficient space for 32 bit header prefix
# CHECK: error: invalid contribution to string offsets table in section .debug_str_offsets[.dwo]: insufficient space for 64 bit header prefix
# CHECK: error: invalid contribution to string offsets table in section .debug_str_offsets[.dwo]: invalid length
# CHECK: error: invalid contribution to string offsets table in section .debug_str_offsets[.dwo]: length exceeds section size
# CHECK: error: invalid contribution to string offsets table in section .debug_str_offsets[.dwo]: 32 bit contribution referenced from a 64 bit unit
# CHECK: error: invalid contribution to string offsets table in section .debug_str_offsets[.dwo]: section offset exceeds section size
# CHECK: error: invalid contribution to string offsets table in section .debug_str_offsets[.dwo]: section offset exceeds section size
# CHECK: error: overlapping contributions to string offsets table in section .debug_str_offsets.
