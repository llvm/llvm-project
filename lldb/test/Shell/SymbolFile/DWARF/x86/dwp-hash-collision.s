## Test that lldb handles (mainly, that it doesn't crash) the situation where
## two skeleton compile units have the same DWO ID (and try to claim the same
## split unit from the DWP file. This can sometimes happen when the compile unit
## is nearly empty (e.g. because LTO has optimized all of it away).

# RUN: llvm-mc -triple=x86_64-pc-linux -filetype=obj %s --defsym MAIN=0 > %t
# RUN: llvm-mc -triple=x86_64-pc-linux -filetype=obj %s > %t.dwp
# RUN: %lldb %t -o "image lookup -t my_enum_type" \
# RUN:   -o "image dump separate-debug-info" -o exit | FileCheck %s

## Check that we're able to access the type within the split unit (no matter
## which skeleton unit it ends up associated with). Completely ignoring the unit
## might also be reasonable.
# CHECK: image lookup -t my_enum_type
# CHECK: 1 match found
# CHECK:      name = "my_enum_type", byte-size = 4, compiler_type = "enum my_enum_type {
# CHECK-NEXT: }"
#
## Check that we get some indication of the error.
# CHECK: image dump separate-debug-info
# CHECK:      Dwo ID             Err Dwo Path
# CHECK: 0xdeadbeefbaadf00d E   multiple compile units with Dwo ID 0xdeadbeefbaadf00d

.set DWO_ID, 0xdeadbeefbaadf00d

## The main file.
.ifdef MAIN
        .section        .debug_abbrev,"",@progbits
        .byte   1                       # Abbreviation Code
        .byte   74                      # DW_TAG_compile_unit
        .byte   0                       # DW_CHILDREN_no
        .byte   0x76                    # DW_AT_dwo_name
        .byte   8                       # DW_FORM_string
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   0                       # EOM(3)


        .section        .debug_info,"",@progbits
.irpc I,01
.Lcu_begin\I:
        .long   .Ldebug_info_end\I-.Ldebug_info_start\I # Length of Unit
.Ldebug_info_start\I:
        .short  5                       # DWARF version number
        .byte   4                       # DWARF Unit Type
        .byte   8                       # Address Size (in bytes)
        .long   .debug_abbrev           # Offset Into Abbrev. Section
        .quad   DWO_ID                  # DWO id
        .byte   1                       # Abbrev [1] DW_TAG_compile_unit
        .ascii  "foo"
        .byte   '0' + \I
        .asciz  ".dwo\0"                # DW_AT_dwo_name
.Ldebug_info_end\I:
.endr

.else
## DWP file starts here.

        .section        .debug_abbrev.dwo,"e",@progbits
.LAbbrevBegin:
        .byte   1                       # Abbreviation Code
        .byte   17                      # DW_TAG_compile_unit
        .byte   1                       # DW_CHILDREN_yes
        .byte   37                      # DW_AT_producer
        .byte   8                       # DW_FORM_string
        .byte   19                      # DW_AT_language
        .byte   5                       # DW_FORM_data2
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   2                       # Abbreviation Code
        .byte   4                       # DW_TAG_enumeration_type
        .byte   0                       # DW_CHILDREN_no
        .byte   3                       # DW_AT_name
        .byte   8                       # DW_FORM_string
        .byte   73                      # DW_AT_type
        .byte   19                      # DW_FORM_ref4
        .byte   11                      # DW_AT_byte_size
        .byte   11                      # DW_FORM_data1
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   4                       # Abbreviation Code
        .byte   36                      # DW_TAG_base_type
        .byte   0                       # DW_CHILDREN_no
        .byte   3                       # DW_AT_name
        .byte   8                       # DW_FORM_string
        .byte   62                      # DW_AT_encoding
        .byte   11                      # DW_FORM_data1
        .byte   11                      # DW_AT_byte_size
        .byte   11                      # DW_FORM_data1
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   0                       # EOM(3)
.LAbbrevEnd:
        .section        .debug_info.dwo,"e",@progbits
.LCUBegin:
.Lcu_begin1:
        .long   .Ldebug_info_end1-.Ldebug_info_start1 # Length of Unit
.Ldebug_info_start1:
        .short  5                       # DWARF version number
        .byte   5                       # DWARF Unit Type
        .byte   8                       # Address Size (in bytes)
        .long   0                       # Offset Into Abbrev. Section
        .quad   DWO_ID                  # DWO id
        .byte   1                       # Abbrev [1] DW_TAG_compile_unit
        .asciz  "Hand-written DWARF"    # DW_AT_producer
        .short  12                      # DW_AT_language
        .byte   2                       # Abbrev [2] DW_TAG_enumeration_type
        .asciz  "my_enum_type"          # DW_AT_name
        .long   .Lint-.Lcu_begin1       # DW_AT_type
        .byte   4                       # DW_AT_byte_size
.Lint:
        .byte   4                       # Abbrev [4] DW_TAG_base_type
        .asciz  "int"                   # DW_AT_name
        .byte   5                       # DW_AT_encoding
        .byte   4                       # DW_AT_byte_size
        .byte   0                       # End Of Children Mark
.Ldebug_info_end1:
.LCUEnd:
        .section .debug_cu_index, "", @progbits
## Header:
        .short 5                        # Version
        .short 0                        # Padding
        .long 2                         # Section count
        .long 1                         # Unit count
        .long 2                         # Slot count
## Hash Table of Signatures:
        .quad 0
        .quad DWO_ID
## Parallel Table of Indexes:
        .long 0
        .long 1
## Table of Section Offsets:
## Row 0:
        .long 1                         # DW_SECT_INFO
        .long 3                         # DW_SECT_ABBREV
## Row 1:
        .long 0                         # Offset in .debug_info.dwo
        .long 0                         # Offset in .debug_abbrev.dwo
## Table of Section Sizes:
        .long .LCUEnd-.LCUBegin         # Size in .debug_info.dwo
        .long .LAbbrevEnd-.LAbbrevBegin # Size in .debug_abbrev.dwo
.endif
