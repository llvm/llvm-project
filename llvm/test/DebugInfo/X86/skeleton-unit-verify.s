# RUN: llvm-mc -triple x86_64-unknown-linux %s -filetype=obj -o %t.o
# RUN: not llvm-dwarfdump --error-display=details --verify %t.o | FileCheck %s

# CHECK: Verifying .debug_abbrev...
# CHECK-NEXT: Verifying .debug_info Unit Header Chain...
# CHECK-NEXT: Verifying .debug_types Unit Header Chain...
# CHECK-NEXT: Verifying non-dwo Units...
# CHECK-NEXT: Verifying unit: 1 / 2, "test.cpp"
# CHECK-NEXT: Verifying unit: 2 / 2{{$}}
# CHECK-NEXT: warning: DW_TAG_skeleton_unit has DW_CHILDREN_yes but DIE has no children
# CHECK-NEXT: DW_TAG_skeleton_unit
# CHECK-NEXT: error: Skeleton compilation unit has children.
# CHECK-NEXT: Verifying dwo Units...
# CHECK-NEXT: Verifying .debug_line...
# CHECK-NEXT: Verifying .debug_str_offsets...
# CHECK-NEXT: Errors detected.

        .section .debug_abbrev,"",@progbits
        .byte   1                       # Abbreviation Code
        .byte   74                      # DW_TAG_skeleton_unit
        .byte   0                       # DW_CHILDREN_no
        .byte   3                       # DW_AT_name
        .byte   8                       # DW_FORM_string
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   2                       # Abbreviation Code
        .byte   74                      # DW_TAG_skeleton_unit
        .byte   1                       # DW_CHILDREN_yes
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   0                       # EOM(3)

        .section        .debug_info,"",@progbits
.Lcu_begin0:
        .long   .Lcu_end0-.Lcu_start0 # Length of Unit
.Lcu_start0:
        .short  5                       # DWARF version number
        .byte   4                       # DWARF Unit Type
        .byte   8                       # Address Size (in bytes)
        .long   .debug_abbrev           # Offset Into Abbrev. Section
        .quad   -6573227469967412476
        .byte   1                       # Abbrev [1]
        .asciz  "test.cpp"
        .byte   0
.Lcu_end0:
        .long   .Lcu_end1-.Lcu_start1 # Length of Unit
.Lcu_start1:
        .short  5                       # DWARF version number
        .byte   4                       # DWARF Unit Type
        .byte   8                       # Address Size (in bytes)
        .long   .debug_abbrev           # Offset Into Abbrev. Section
        .quad   -6573227469967412476
        .byte   2                       # Abbrev [2]
        .byte   0
.Lcu_end1:
