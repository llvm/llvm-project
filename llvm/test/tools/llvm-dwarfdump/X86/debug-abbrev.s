## Show that llvm-dwarfdump prints a range of different structures in the
## .debug_abbrev section when requested. As well as basic cases, this test also
## shows how llvm-dwarfdump handles unknown tag, attribute and form values.

# RUN: llvm-mc %s -filetype obj -triple x86_64-pc-linux -o %t.o
# RUN: llvm-dwarfdump --debug-abbrev %t.o | FileCheck %s --match-full-lines

# CHECK:      .debug_abbrev contents:
# CHECK-NEXT: Abbrev table for offset: 0x00000000
# CHECK-NEXT: [66] DW_TAG_compile_unit        DW_CHILDREN_yes
# CHECK-NEXT:         DW_AT_name      DW_FORM_string
# CHECK-NEXT:         DW_AT_stmt_list DW_FORM_data2
# CHECK-EMPTY:
# CHECK-NEXT: [128] DW_TAG_unknown_ffff       DW_CHILDREN_no
# CHECK-NEXT:         DW_AT_unknown_5 DW_FORM_unknown_2
# CHECK-EMPTY:
# CHECK-NEXT: [66] DW_TAG_unknown_4080        DW_CHILDREN_no
# CHECK-EMPTY:
# CHECK-NEXT: [1] DW_TAG_unknown_4000 DW_CHILDREN_no
# CHECK-NEXT:         DW_AT_unknown_2000      DW_FORM_unknown_ffff
# CHECK-NEXT:         DW_AT_unknown_3fff      DW_FORM_unknown_7f
# CHECK-NEXT:         DW_AT_unknown_1fff      DW_FORM_block2
# CHECK-NEXT:         DW_AT_unknown_4000      DW_FORM_block4
# CHECK-EMPTY:
# CHECK-NEXT: [2] DW_TAG_unknown_407f DW_CHILDREN_no
# CHECK-EMPTY:
# CHECK-NEXT: Abbrev table for offset: 0x00000038
# CHECK-NEXT: [66] DW_TAG_entry_point DW_CHILDREN_no
# CHECK-NEXT:         DW_AT_elemental DW_FORM_flag_present

## All values in this assembly are intended to be arbitrary, except where
## specified.
.section .debug_abbrev,"",@progbits
    .byte 0x42 ## Abbreviation Code
    ## Known tag with children and known attributes and forms.
    .byte 0x11 ## DW_TAG_compile_unit
    .byte 0x01 ## DW_CHILDREN_yes
        .byte 0x03 ## DW_AT_name
        .byte 0x08 ## DW_FORM_string (valid form for DW_AT_name).
        .byte 0x10 ## DW_AT_stmt_list
        .byte 0x05 ## DW_FORM_data2 (invalid form for DW_AT_stmt_list).
        .byte 0, 0 ## End of attributes

    ## Tag without children and reserved form/attribute values.
    .byte 0x80, 0x01       ## Multi-byte Abbreviation Code
    .byte 0xFF, 0xFF, 0x03 ## 0xFFFF == DW_TAG_hi_user
    .byte 0x00             ## DW_CHILDREN_no
        ## Reserved attribute and form.
        .byte 0x05 ## Reserved DW_AT_*
        .byte 0x02 ## Reserved DW_FORM_*
        .byte 0, 0 ## End of attributes.

    ## Tag with no attributes.
    .byte 0x42             ## Abbreviation Code (duplicate)
    .byte 0x80, 0x81, 0x01 ## 0x4080 == DW_TAG_lo_user
    .byte 0x00             ## DW_CHILDREN_no
        .byte 0, 0         ## End of attributes.

    ## Tag with attributes/forms with unknown values/values in user-defined ranges.
    .byte 0x01                 ## Abbreviation Code
    ## FIXME: https://bugs.llvm.org/show_bug.cgi?id=44258 means that 0x80, 0x80, 0x04
    ## results in a failure to parse correctly, whilst the following tag value is
    ## interpreted incorrectly as 0x4000.
    .byte 0x80, 0x80, 0x05     ## 0x10001 == DW_TAG_hi_user + 2
    .byte 0x00                 ## DW_CHILDREN_no
        .byte 0x80, 0x40       ## 0x2000 == DW_AT_lo_user
        .byte 0xFF, 0xFF, 0x03 ## 0xFFFF == Unknown multi-byte form.
        .byte 0xFF, 0x7F       ## DW_AT_hi_user
        .byte 0x7F             ## Unknown single-byte form.
        .byte 0xFF, 0x3F       ## DW_AT_lo_user - 1
        .byte 0x03             ## DW_FORM_block2
        .byte 0x80, 0x80, 0x01 ## DW_AT_hi_user + 1
        .byte 0x04             ## DW_FORM_block4
        .byte 0, 0             ## End of attributes.

    ## Tag with invalid children encoding.
    .byte 0x02             ## Abbreviation Code
    .byte 0xFF, 0x80, 0x01 ## 0x407F == DW_TAG_lo_user - 1
    .byte 0x02 ## Invalid children encoding (interpreted as DW_CHILDREN_no).
        .byte 0, 0         ## End of attributes.

    .byte 0 ## End of first abbreviation set.

    ## Second .debug_abbrev set.
    .byte 0x42 ## Abbreviation Code (duplicate in different unit)
    .byte 0x03 ## DW_TAG_entry_point
    .byte 0x00 ## DW_CHILDREN_no
        .byte 0x66 ## DW_AT_elemental
        .byte 0x19 ## DW_FORM_flag_present
        .byte 0, 0 ## End of attributes.

    .byte 0 # End of second abbreviation set.
