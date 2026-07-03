# Check that array type bounds are displayed correctly for DWARFv6 compile units.
# The presentation of array bounds is driven by the CU's language, which, starting in,
# DWARFv6 changed.

# RUN: llvm-mc -triple=x86_64--linux -filetype=obj < %s | \
# RUN:     llvm-dwarfdump --debug-info - | FileCheck %s

# CHECK-LABEL: DW_TAG_compile_unit
# CHECK:         DW_AT_language_name (DW_LNAME_C_plus_plus)
# CHECK:         DW_AT_language_version (C++17)
# CHECK:       DW_TAG_variable
# CHECK-NEXT:    DW_AT_name  ("arr")
# CHECK-NEXT:    DW_AT_type  ({{.*}} "int[3]")

        .section        .debug_abbrev,"",@progbits

        .byte   1
        .byte   17              # DW_TAG_compile_unit
        .byte   1               # DW_CHILDREN_yes
        .ascii  "\220\001"      # DW_AT_language_name (0x90)
        .byte   5               # DW_FORM_data2
        .ascii  "\221\001"      # DW_AT_language_version (0x91)
        .byte   6               # DW_FORM_data4
        .byte   0, 0            # end of attributes

        .byte   2
        .byte   52              # DW_TAG_variable
        .byte   0               # DW_CHILDREN_no
        .byte   3               # DW_AT_name
        .byte   8               # DW_FORM_string
        .byte   73              # DW_AT_type
        .byte   19              # DW_FORM_ref4
        .byte   0, 0

        .byte   3
        .byte   1               # DW_TAG_array_type
        .byte   1               # DW_CHILDREN_yes
        .byte   73              # DW_AT_type
        .byte   19              # DW_FORM_ref4
        .byte   0, 0

        .byte   4
        .byte   33              # DW_TAG_subrange_type
        .byte   0               # DW_CHILDREN_no
        .byte   73              # DW_AT_type
        .byte   19              # DW_FORM_ref4
        .byte   55              # DW_AT_count
        .byte   11              # DW_FORM_data1
        .byte   0, 0

        .byte   5
        .byte   36              # DW_TAG_base_type
        .byte   0               # DW_CHILDREN_no
        .byte   3               # DW_AT_name
        .byte   8               # DW_FORM_string
        .byte   62              # DW_AT_encoding
        .byte   11              # DW_FORM_data1
        .byte   11              # DW_AT_byte_size
        .byte   11              # DW_FORM_data1
        .byte   0, 0

        .byte   0               # end of abbreviation table

        .section        .debug_info,"",@progbits
        .long   .Ldebug_info_end0 - .Ldebug_info_start0 # unit length
.Ldebug_info_start0:
        .short  6               # DWARF version
        .byte   1               # DW_UT_compile
        .byte   8               # address size
        .long   0               # debug_abbrev_offset

        # DW_TAG_compile_unit
        .byte   1               # abbrev 1
        .short  4               # DW_AT_language_name = DW_LNAME_C_plus_plus
        .long   201703          # DW_AT_language_version = C++17

          # DW_TAG_variable
          .byte   2             # abbrev 2
          .ascii  "arr\0"       # DW_AT_name
          .long   .Larray_type - .Ldebug_info_start0 + 4  # DW_AT_type

.Larray_type:
          # DW_TAG_array_type
          .byte   3             # abbrev 3
          .long   .Lint_type - .Ldebug_info_start0 + 4   # DW_AT_type

            # DW_TAG_subrange_type
            .byte   4           # abbrev 4
            .long   .Lsize_type - .Ldebug_info_start0 + 4 # DW_AT_type
            .byte   3           # DW_AT_count = 3
            .byte   0           # end of children (array_type)

.Lint_type:
          # DW_TAG_base_type "int"
          .byte   5             # abbrev 5
          .ascii  "int\0"       # DW_AT_name
          .byte   5             # DW_AT_encoding = DW_ATE_signed
          .byte   4             # DW_AT_byte_size = 4

.Lsize_type:
          # DW_TAG_base_type "__ARRAY_SIZE_TYPE__"
          .byte   5             # abbrev 5
          .ascii  "__ARRAY_SIZE_TYPE__\0" # DW_AT_name
          .byte   7             # DW_AT_encoding = DW_ATE_unsigned
          .byte   8             # DW_AT_byte_size = 8

        .byte   0               # end of children (compile_unit)
.Ldebug_info_end0:
