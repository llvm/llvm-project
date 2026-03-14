# RUN: llvm-mc -triple x86_64 %s -filetype=obj -o %t.o
# RUN: llvm-dwarfdump -debug-info %t.o | FileCheck %s
# RUN: llvm-dwarfdump -verify %t.o | FileCheck %s --check-prefix=VERIFY

# CHECK: DW_TAG_variable
# CHECK-NEXT: DW_AT_location (DW_OP_call_ref 0x1100223344)

# VERIFY-NOT: error: DIE contains invalid DWARF expression:
# VERIFY: No errors.

    .section .debug_abbrev,"",@progbits
    .uleb128 1                      # Abbreviation Code
    .uleb128 17                     # DW_TAG_compile_unit
    .byte 1                         # DW_CHILDREN_yes
    .byte 0                         # EOM(1)
    .byte 0                         # EOM(2)
    .uleb128 5                      # Abbreviation Code
    .uleb128 52                     # DW_TAG_variable
    .byte 0                         # DW_CHILDREN_no
    .uleb128 2                      # DW_AT_location
    .uleb128 24                     # DW_FORM_exprloc
    .byte 0                         # EOM(1)
    .byte 0                         # EOM(2)
    .byte 0                         # EOM(3)

    .section .debug_info,"",@progbits
    .long 0xffffffff                # DWARF64 mark
    .quad .Lcu_end-.Lcu_start       # Length of Unit
.Lcu_start:
    .short 5                        # DWARF version number
    .byte 1                         # DW_UT_compile
    .byte 8                         # Address Size
    .quad .debug_abbrev             # Offset Into Abbrev. Section
    .uleb128 1                      # Abbrev [1] DW_TAG_compile_unit
    .uleb128 5                      # Abbrev [5] DW_TAG_variable
    .byte .Lloc_end-.Lloc_begin     # DW_AT_location
.Lloc_begin:
    .byte 154                       # DW_OP_call_ref
    .quad 0x1100223344              # Offset
.Lloc_end:
    .byte 0                         # End Of Children Mark
.Lcu_end:
