## Check that lldb can locate a static constant variable when its declaration is
## referenced by a debug_names index. This is a non-conforming extension used by
## dsymutil.

# RUN: llvm-mc -triple=x86_64-pc-linux -filetype=obj %s > %t
# RUN: %lldb %t -o "target variable Class::constant" \
# RUN:   -o "expr -l c++ -- Class::constant" -o exit | FileCheck %s

# CHECK:      (lldb) target variable Class::constant
# CHECK-NEXT: (const int) Class::constant = 47
# CHECK:      (lldb) expr -l c++ -- Class::constant
# CHECK-NEXT: (const int) $0 = 47

        .section        .debug_abbrev,"",@progbits
        .byte   1                               # Abbreviation Code
        .byte   17                              # DW_TAG_compile_unit
        .byte   1                               # DW_CHILDREN_yes
        .byte   37                              # DW_AT_producer
        .byte   8                               # DW_FORM_string
        .byte   19                              # DW_AT_language
        .byte   5                               # DW_FORM_data2
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   3                               # Abbreviation Code
        .byte   2                               # DW_TAG_class_type
        .byte   1                               # DW_CHILDREN_yes
        .byte   54                              # DW_AT_calling_convention
        .byte   11                              # DW_FORM_data1
        .byte   3                               # DW_AT_name
        .byte   14                              # DW_FORM_strp
        .byte   11                              # DW_AT_byte_size
        .byte   11                              # DW_FORM_data1
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   4                               # Abbreviation Code
        .byte   52                              # DW_TAG_variable
        .byte   0                               # DW_CHILDREN_no
        .byte   3                               # DW_AT_name
        .byte   14                              # DW_FORM_strp
        .byte   73                              # DW_AT_type
        .byte   19                              # DW_FORM_ref4
        .byte   63                              # DW_AT_external
        .byte   25                              # DW_FORM_flag_present
        .byte   60                              # DW_AT_declaration
        .byte   25                              # DW_FORM_flag_present
        .byte   28                              # DW_AT_const_value
        .byte   13                              # DW_FORM_sdata
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   5                               # Abbreviation Code
        .byte   38                              # DW_TAG_const_type
        .byte   0                               # DW_CHILDREN_no
        .byte   73                              # DW_AT_type
        .byte   19                              # DW_FORM_ref4
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   6                               # Abbreviation Code
        .byte   36                              # DW_TAG_base_type
        .byte   0                               # DW_CHILDREN_no
        .byte   3                               # DW_AT_name
        .byte   14                              # DW_FORM_strp
        .byte   62                              # DW_AT_encoding
        .byte   11                              # DW_FORM_data1
        .byte   11                              # DW_AT_byte_size
        .byte   11                              # DW_FORM_data1
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   0                               # EOM(3)
        .section        .debug_info,"",@progbits
.Lcu_begin0:
        .long   .Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
        .short  5                               # DWARF version number
        .byte   1                               # DWARF Unit Type
        .byte   8                               # Address Size (in bytes)
        .long   .debug_abbrev                   # Offset Into Abbrev. Section
        .byte   1                               # Abbrev [1] 0xc:0x40 DW_TAG_compile_unit
        .asciz  "Hand-written DWARF"            # DW_AT_producer
        .short  33                              # DW_AT_language
.LClass:
        .byte   3                               # Abbrev [3] 0x29:0x10 DW_TAG_class_type
        .byte   5                               # DW_AT_calling_convention
        .long   .Linfo_string4                  # DW_AT_name
        .byte   1                               # DW_AT_byte_size
.Lvariable:
        .byte   4                               # Abbrev [4] 0x2f:0x9 DW_TAG_variable
        .long   .Linfo_string5                  # DW_AT_name
        .long   .Lconst_int-.Lcu_begin0         # DW_AT_type
                                                # DW_AT_external
                                                # DW_AT_declaration
        .byte   47                              # DW_AT_const_value
        .byte   0                               # End Of Children Mark
.Lconst_int:
        .byte   5                               # Abbrev [5] 0x39:0x5 DW_TAG_const_type
        .long   .Lint-.Lcu_begin0               # DW_AT_type
.Lint:
        .byte   6                               # Abbrev [6] 0x3e:0x4 DW_TAG_base_type
        .long   .Linfo_string6                  # DW_AT_name
        .byte   5                               # DW_AT_encoding
        .byte   4                               # DW_AT_byte_size
        .byte   0                               # End Of Children Mark
.Ldebug_info_end0:

        .section        .debug_str,"MS",@progbits,1
.Linfo_string4:
        .asciz  "Class"
.Linfo_string5:
        .asciz  "constant"
.Linfo_string6:
        .asciz  "int"

        .section        .debug_names,"",@progbits
        .long   .Lnames_end0-.Lnames_start0     # Header: unit length
.Lnames_start0:
        .short  5                               # Header: version
        .short  0                               # Header: padding
        .long   1                               # Header: compilation unit count
        .long   0                               # Header: local type unit count
        .long   0                               # Header: foreign type unit count
        .long   0                               # Header: bucket count
        .long   3                               # Header: name count
        .long   .Lnames_abbrev_end0-.Lnames_abbrev_start0 # Header: abbreviation table size
        .long   8                               # Header: augmentation string size
        .ascii  "LLVM0700"                      # Header: augmentation string
        .long   .Lcu_begin0                     # Compilation unit 0
        .long   .Linfo_string4                  # String: Class
        .long   .Linfo_string5                  # String: constant
        .long   .Linfo_string6                  # String: int
        .long   .Lnames0-.Lnames_entries0
        .long   .Lnames3-.Lnames_entries0
        .long   .Lnames1-.Lnames_entries0
.Lnames_abbrev_start0:
        .byte   1                               # Abbrev code
        .byte   2                               # DW_TAG_class_type
        .byte   3                               # DW_IDX_die_offset
        .byte   19                              # DW_FORM_ref4
        .byte   0                               # End of abbrev
        .byte   0                               # End of abbrev
        .byte   2                               # Abbrev code
        .byte   52                              # DW_TAG_variable
        .byte   3                               # DW_IDX_die_offset
        .byte   19                              # DW_FORM_ref4
        .byte   0                               # End of abbrev
        .byte   0                               # End of abbrev
        .byte   3                               # Abbrev code
        .byte   36                              # DW_TAG_base_type
        .byte   3                               # DW_IDX_die_offset
        .byte   19                              # DW_FORM_ref4
        .byte   0                               # End of abbrev
        .byte   0                               # End of abbrev
        .byte   0                               # End of abbrev list
.Lnames_abbrev_end0:
.Lnames_entries0:
.Lnames0:
        .byte   1                               # Abbreviation code
        .long   .LClass-.Lcu_begin0             # DW_IDX_die_offset
        .byte   0                               # DW_IDX_parent
                                        # End of list: Class
.Lnames3:
        .byte   2                               # Abbreviation code
        .long   .Lvariable-.Lcu_begin0          # DW_IDX_die_offset
        .byte   0                               # DW_IDX_parent
                                        # End of list: constant
.Lnames1:
        .byte   3                               # Abbreviation code
        .long   .Lint-.Lcu_begin0               # DW_IDX_die_offset
        .byte   0                               # DW_IDX_parent
                                        # End of list: int
.Lnames_end0:
