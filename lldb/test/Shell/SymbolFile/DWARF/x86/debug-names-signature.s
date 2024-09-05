## Test that we can correctly complete types even if the debug_names index
## contains entries referring to declaration dies (clang emitted entries like
## that until bd5c6367bd7).
##
## This test consists of two compile units and one type unit. CU1 has the
## definition of a variable, but only a forward-declaration of its type. When
## attempting to find a definition, the debug_names lookup will return the DIE
## in CU0, which is also a forward-declaration (with a reference to a type
## unit). LLDB needs to find the definition of the type within the type unit.

# RUN: llvm-mc -triple=x86_64-pc-linux -filetype=obj %s > %t
# RUN: %lldb %t -o "target variable s" -o exit | FileCheck %s

# CHECK:      (lldb) target variable s
# CHECK-NEXT: (Struct) s = (member = 47)

        .data
        .p2align        2, 0x0
        .long   0
s:
        .long   47                              # 0x2f

        .section        .debug_abbrev,"",@progbits
        .byte   1                               # Abbreviation Code
        .byte   65                              # DW_TAG_type_unit
        .byte   1                               # DW_CHILDREN_yes
        .byte   19                              # DW_AT_language
        .byte   5                               # DW_FORM_data2
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   2                               # Abbreviation Code
        .byte   19                              # DW_TAG_structure_type
        .byte   1                               # DW_CHILDREN_yes
        .byte   54                              # DW_AT_calling_convention
        .byte   11                              # DW_FORM_data1
        .byte   3                               # DW_AT_name
        .byte   14                              # DW_FORM_strp
        .byte   11                              # DW_AT_byte_size
        .byte   11                              # DW_FORM_data1
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   3                               # Abbreviation Code
        .byte   13                              # DW_TAG_member
        .byte   0                               # DW_CHILDREN_no
        .byte   3                               # DW_AT_name
        .byte   14                              # DW_FORM_strp
        .byte   73                              # DW_AT_type
        .byte   19                              # DW_FORM_ref4
        .byte   56                              # DW_AT_data_member_location
        .byte   11                              # DW_FORM_data1
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   4                               # Abbreviation Code
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
        .byte   5                               # Abbreviation Code
        .byte   17                              # DW_TAG_compile_unit
        .byte   1                               # DW_CHILDREN_yes
        .byte   37                              # DW_AT_producer
        .byte   8                               # DW_FORM_string
        .byte   19                              # DW_AT_language
        .byte   5                               # DW_FORM_data2
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   6                               # Abbreviation Code
        .byte   52                              # DW_TAG_variable
        .byte   0                               # DW_CHILDREN_no
        .byte   3                               # DW_AT_name
        .byte   14                              # DW_FORM_strp
        .byte   73                              # DW_AT_type
        .byte   19                              # DW_FORM_ref4
        .byte   2                               # DW_AT_location
        .byte   24                              # DW_FORM_exprloc
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   7                               # Abbreviation Code
        .byte   19                              # DW_TAG_structure_type
        .byte   0                               # DW_CHILDREN_no
        .byte   60                              # DW_AT_declaration
        .byte   25                              # DW_FORM_flag_present
        .byte   105                             # DW_AT_signature
        .byte   32                              # DW_FORM_ref_sig8
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   8                               # Abbreviation Code
        .byte   19                              # DW_TAG_structure_type
        .byte   0                               # DW_CHILDREN_no
        .byte   3                               # DW_AT_name
        .byte   14                              # DW_FORM_strp
        .byte   60                              # DW_AT_declaration
        .byte   25                              # DW_FORM_flag_present
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   0                               # EOM(3)

        .section        .debug_info,"",@progbits
.Ltu_begin0:
        .long   .Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
        .short  5                               # DWARF version number
        .byte   2                               # DWARF Unit Type
        .byte   8                               # Address Size (in bytes)
        .long   .debug_abbrev                   # Offset Into Abbrev. Section
        .quad   4878254330033667422             # Type Signature
        .long   .LStruct_def-.Ltu_begin0        # Type DIE Offset
        .byte   1                               # Abbrev [1] 0x18:0x20 DW_TAG_type_unit
        .short  33                              # DW_AT_language
.LStruct_def:
        .byte   2                               # Abbrev [2] 0x23:0x10 DW_TAG_structure_type
        .byte   5                               # DW_AT_calling_convention
        .long   .Linfo_string6                  # DW_AT_name
        .byte   4                               # DW_AT_byte_size
        .byte   3                               # Abbrev [3] 0x29:0x9 DW_TAG_member
        .long   .Linfo_string4                  # DW_AT_name
        .long   .Lint-.Ltu_begin0               # DW_AT_type
        .byte   0                               # DW_AT_data_member_location
        .byte   0                               # End Of Children Mark
.Lint:
        .byte   4                               # Abbrev [4] 0x33:0x4 DW_TAG_base_type
        .long   .Linfo_string5                  # DW_AT_name
        .byte   5                               # DW_AT_encoding
        .byte   4                               # DW_AT_byte_size
        .byte   0                               # End Of Children Mark
.Ldebug_info_end0:

.Lcu_begin0:
        .long   .Ldebug_info_end1-.Ldebug_info_start1 # Length of Unit
.Ldebug_info_start1:
        .short  5                               # DWARF version number
        .byte   1                               # DWARF Unit Type
        .byte   8                               # Address Size (in bytes)
        .long   .debug_abbrev                   # Offset Into Abbrev. Section
        .byte   5                               # Abbrev [5] 0xc:0x27 DW_TAG_compile_unit
        .asciz  "Hand-written DWARF"            # DW_AT_producer
        .short  33                              # DW_AT_language
.Ls:
        .byte   6                               # Abbrev [6] 0x1e:0xb DW_TAG_variable
        .long   .Linfo_string3                  # DW_AT_name
        .long   .LStruct_decl2-.Lcu_begin0       # DW_AT_type
        .byte   9                               # DW_AT_location
        .byte   3
        .quad   s
.LStruct_decl2:
        .byte   8                               # Abbrev [8] 0x29:0x9 DW_TAG_structure_type
        .long   .Linfo_string6                  # DW_AT_name
                                                # DW_AT_declaration
        .byte   0                               # End Of Children Mark
.Ldebug_info_end1:

.Lcu_begin1:
        .long   .Ldebug_info_end2-.Ldebug_info_start2 # Length of Unit
.Ldebug_info_start2:
        .short  5                               # DWARF version number
        .byte   1                               # DWARF Unit Type
        .byte   8                               # Address Size (in bytes)
        .long   .debug_abbrev                   # Offset Into Abbrev. Section
        .byte   5                               # Abbrev [5] 0xc:0x27 DW_TAG_compile_unit
        .asciz  "Hand-written DWARF"            # DW_AT_producer
        .short  33                              # DW_AT_language
.LStruct_decl:
        .byte   7                               # Abbrev [7] 0x29:0x9 DW_TAG_structure_type
                                                # DW_AT_declaration
        .quad   4878254330033667422             # DW_AT_signature
        .byte   0                               # End Of Children Mark
.Ldebug_info_end2:

        .section        .debug_str,"MS",@progbits,1
.Linfo_string3:
        .asciz  "s"                             # string offset=60
.Linfo_string4:
        .asciz  "member"                        # string offset=62
.Linfo_string5:
        .asciz  "int"                           # string offset=69
.Linfo_string6:
        .asciz  "Struct"                        # string offset=73

        .section        .debug_names,"",@progbits
        .long   .Lnames_end0-.Lnames_start0     # Header: unit length
.Lnames_start0:
        .short  5                               # Header: version
        .short  0                               # Header: padding
        .long   2                               # Header: compilation unit count
        .long   1                               # Header: local type unit count
        .long   0                               # Header: foreign type unit count
        .long   0                               # Header: bucket count
        .long   3                               # Header: name count
        .long   .Lnames_abbrev_end0-.Lnames_abbrev_start0 # Header: abbreviation table size
        .long   8                               # Header: augmentation string size
        .ascii  "LLVM0700"                      # Header: augmentation string
        .long   .Lcu_begin0                     # Compilation unit 0
        .long   .Lcu_begin1                     # Compilation unit 1
        .long   .Ltu_begin0                     # Type unit 0
        .long   .Linfo_string6                  # String in Bucket 0: Struct
        .long   .Linfo_string3                  # String in Bucket 1: s
        .long   .Linfo_string5                  # String in Bucket 2: int
        .long   .Lnames1-.Lnames_entries0       # Offset in Bucket 0
        .long   .Lnames2-.Lnames_entries0       # Offset in Bucket 1
        .long   .Lnames0-.Lnames_entries0       # Offset in Bucket 2
.Lnames_abbrev_start0:
        .byte   1                               # Abbrev code
        .byte   19                              # DW_TAG_structure_type
        .byte   2                               # DW_IDX_type_unit
        .byte   11                              # DW_FORM_data1
        .byte   3                               # DW_IDX_die_offset
        .byte   19                              # DW_FORM_ref4
        .byte   0                               # End of abbrev
        .byte   0                               # End of abbrev
        .byte   2                               # Abbrev code
        .byte   52                              # DW_TAG_variable
        .byte   1                               # DW_IDX_compile_unit
        .byte   11                              # DW_FORM_data1
        .byte   3                               # DW_IDX_die_offset
        .byte   19                              # DW_FORM_ref4
        .byte   0                               # End of abbrev
        .byte   0                               # End of abbrev
        .byte   3                               # Abbrev code
        .byte   36                              # DW_TAG_base_type
        .byte   2                               # DW_IDX_type_unit
        .byte   11                              # DW_FORM_data1
        .byte   3                               # DW_IDX_die_offset
        .byte   19                              # DW_FORM_ref4
        .byte   0                               # End of abbrev
        .byte   0                               # End of abbrev
        .byte   4                               # Abbrev code
        .byte   19                              # DW_TAG_structure_type
        .byte   1                               # DW_IDX_compile_unit
        .byte   11                              # DW_FORM_data1
        .byte   3                               # DW_IDX_die_offset
        .byte   19                              # DW_FORM_ref4
        .byte   0                               # End of abbrev
        .byte   0                               # End of abbrev
        .byte   0                               # End of abbrev list
.Lnames_abbrev_end0:
.Lnames_entries0:
.Lnames1:
        .byte   4                               # Abbreviation code
        .byte   1                               # DW_IDX_compile_unit
        .long   .LStruct_decl-.Lcu_begin1       # DW_IDX_die_offset
        .byte   1                               # Abbreviation code
        .byte   0                               # DW_IDX_type_unit
        .long   .LStruct_def-.Ltu_begin0        # DW_IDX_die_offset
        .byte   0
                                        # End of list: Struct
.Lnames2:
        .byte   2                               # Abbreviation code
        .byte   0                               # DW_IDX_compile_unit
        .long   .Ls-.Lcu_begin0                 # DW_IDX_die_offset
        .byte   0
                                        # End of list: s
.Lnames0:
        .byte   3                               # Abbreviation code
        .byte   0                               # DW_IDX_type_unit
        .long   .Lint-.Ltu_begin0               # DW_IDX_die_offset
        .byte   0
                                        # End of list: int
        .p2align        2, 0x0
.Lnames_end0:
