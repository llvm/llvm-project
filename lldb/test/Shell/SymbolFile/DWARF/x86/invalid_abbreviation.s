# REQUIRES: x86

# RUN: llvm-mc -triple=x86_64-pc-linux -filetype=obj %s > %t
# RUN: %lldb %t \
# RUN:   -o exit 2>&1 | FileCheck %s

# CHECK-DAG: error: {{.*}} [0x0000000000000022]: abbreviation code 65536 too big, please file a bug and attach the file at the start of this error message
# CHECK-DAG: error: {{.*}} [0x0000000000000048]: invalid abbreviation code 47, please file a bug and attach the file at the start of this error message


        .section        .debug_abbrev,"",@progbits
        .uleb128 65535                  # Largest representable Abbreviation Code
        .byte   17                      # DW_TAG_compile_unit
        .byte   1                       # DW_CHILDREN_yes
        .byte   37                      # DW_AT_producer
        .byte   8                       # DW_FORM_string
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   0                       # EOM(3)

        .section        .debug_info,"",@progbits
.Lcu_begin0:
        .long   .Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
        .short  5                       # DWARF version number
        .byte   1                       # DWARF Unit Type
        .byte   8                       # Address Size (in bytes)
        .long   .debug_abbrev           # Offset Into Abbrev. Section
        .uleb128 65535                  # DW_TAG_compile_unit
        .asciz  "Hand-written DWARF"    # DW_AT_producer
        .uleb128 65536                  # Unrepresentable abbreviation
        .byte   0                       # End Of Children Mark
.Ldebug_info_end0:

        .section        .debug_info,"",@progbits
.Lcu_begin1:
        .long   .Ldebug_info_end1-.Ldebug_info_start1 # Length of Unit
.Ldebug_info_start1:
        .short  5                       # DWARF version number
        .byte   1                       # DWARF Unit Type
        .byte   8                       # Address Size (in bytes)
        .long   .debug_abbrev           # Offset Into Abbrev. Section
        .uleb128 65535                  # DW_TAG_compile_unit
        .asciz  "Hand-written DWARF"    # DW_AT_producer
        .byte   47                      # Missing abbreviation
        .byte   0                       # End Of Children Mark
.Ldebug_info_end1:
