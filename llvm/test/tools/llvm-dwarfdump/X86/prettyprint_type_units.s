# RUN: llvm-mc < %s -filetype obj -triple x86_64 -o - \
# RUN:   | llvm-dwarfdump - | FileCheck %s

# Hand-written assembly roughly equivalent to this source code:
#
# struct t1 { };
# struct t2 { };
# template<typename ...T>
# struct S {};
# S<t1, t2, t1> s;
#
# To cover various scenarios, the test uses a mixture of DWARF v4 and v5 type
# units, and of llvm and gcc styles of referring to them.


# CHECK:      DW_TAG_variable
# CHECK-NEXT:   DW_AT_name ("s")
# CHECK-NEXT:   DW_AT_type ({{.*}} "S<t1, t2, t1>")
# CHECK:      DW_TAG_template_type_parameter
# CHECK-NEXT:   DW_AT_type ({{.*}} "t1")
# CHECK:      DW_TAG_template_type_parameter
# CHECK-NEXT:   DW_AT_type ({{.*}} "t2")
# CHECK:      DW_TAG_template_type_parameter
# CHECK-NEXT:   DW_AT_type (0xdeadbeef00000001 "t1")

.set S_sig,  0xdeadbeef00000000
.set t1_sig, 0xdeadbeef00000001
.set t2_sig, 0xdeadbeef00000002

        .section        .debug_types,"G",@progbits,t1_sig,comdat
.Ltu_begin0:
        .long   .Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
        .short  4                               # DWARF version number
        .long   .debug_abbrev                   # Offset Into Abbrev. Section
        .byte   8                               # Address Size (in bytes)
        .quad   t1_sig                          # Type Signature
        .long   .Lt1_def-.Ltu_begin0            # Type DIE Offset
        .byte   10                              # Abbrev [10] DW_TAG_type_unit
        .short  33                              # DW_AT_language
.Lt1_def:
        .byte   11                              # Abbrev [11] DW_TAG_structure_type
        .long   .Linfo_string6                  # DW_AT_name
        .byte   1                               # DW_AT_byte_size
        .byte   0                               # End Of Children Mark
.Ldebug_info_end0:
        .section        .debug_info,"G",@progbits,t2_sig,comdat
.Ltu_begin1:
        .long   .Ldebug_info_end1-.Ldebug_info_start1 # Length of Unit
.Ldebug_info_start1:
        .short  5                               # DWARF version number
        .byte   2                               # DWARF Unit Type
        .byte   8                               # Address Size (in bytes)
        .long   .debug_abbrev                   # Offset Into Abbrev. Section
        .quad   t2_sig                          # Type Signature
        .long   .Lt2_def-.Ltu_begin1            # Type DIE Offset
        .byte   1                               # Abbrev [1] 0x18:0x12 DW_TAG_type_unit
        .short  33                              # DW_AT_language
        .long   .Lstr_offsets_base0             # DW_AT_str_offsets_base
.Lt2_def:
        .byte   2                               # Abbrev [2] 0x23:0x6 DW_TAG_structure_type
        .byte   7                               # DW_AT_name
        .byte   1                               # DW_AT_byte_size
        .byte   0                               # End Of Children Mark
.Ldebug_info_end1:
        .section        .debug_abbrev,"",@progbits
        .byte   1                               # Abbreviation Code
        .byte   65                              # DW_TAG_type_unit
        .byte   1                               # DW_CHILDREN_yes
        .byte   19                              # DW_AT_language
        .byte   5                               # DW_FORM_data2
        .byte   114                             # DW_AT_str_offsets_base
        .byte   23                              # DW_FORM_sec_offset
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   2                               # Abbreviation Code
        .byte   19                              # DW_TAG_structure_type
        .byte   0                               # DW_CHILDREN_no
        .byte   3                               # DW_AT_name
        .byte   37                              # DW_FORM_strx1
        .byte   11                              # DW_AT_byte_size
        .byte   11                              # DW_FORM_data1
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   3                               # Abbreviation Code
        .byte   17                              # DW_TAG_compile_unit
        .byte   1                               # DW_CHILDREN_yes
        .byte   37                              # DW_AT_producer
        .byte   37                              # DW_FORM_strx1
        .byte   19                              # DW_AT_language
        .byte   5                               # DW_FORM_data2
        .byte   3                               # DW_AT_name
        .byte   37                              # DW_FORM_strx1
        .byte   114                             # DW_AT_str_offsets_base
        .byte   23                              # DW_FORM_sec_offset
        .byte   27                              # DW_AT_comp_dir
        .byte   37                              # DW_FORM_strx1
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   4                               # Abbreviation Code
        .byte   52                              # DW_TAG_variable
        .byte   0                               # DW_CHILDREN_no
        .byte   3                               # DW_AT_name
        .byte   8                               # DW_FORM_string
        .byte   73                              # DW_AT_type
        .byte   19                              # DW_FORM_ref4
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   6                               # Abbreviation Code
        .ascii  "\207\202\001"                  # DW_TAG_GNU_template_parameter_pack
        .byte   1                               # DW_CHILDREN_yes
        .byte   3                               # DW_AT_name
        .byte   37                              # DW_FORM_strx1
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   7                               # Abbreviation Code
        .byte   47                              # DW_TAG_template_type_parameter
        .byte   0                               # DW_CHILDREN_no
        .byte   73                              # DW_AT_type
        .byte   19                              # DW_FORM_ref4
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   8                               # Abbreviation Code
        .byte   36                              # DW_TAG_base_type
        .byte   0                               # DW_CHILDREN_no
        .byte   3                               # DW_AT_name
        .byte   37                              # DW_FORM_strx1
        .byte   62                              # DW_AT_encoding
        .byte   11                              # DW_FORM_data1
        .byte   11                              # DW_AT_byte_size
        .byte   11                              # DW_FORM_data1
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   9                               # Abbreviation Code
        .byte   19                              # DW_TAG_structure_type
        .byte   0                               # DW_CHILDREN_no
        .byte   60                              # DW_AT_declaration
        .byte   25                              # DW_FORM_flag_present
        .byte   105                             # DW_AT_signature
        .byte   32                              # DW_FORM_ref_sig8
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   10                              # Abbreviation Code
        .byte   65                              # DW_TAG_type_unit
        .byte   1                               # DW_CHILDREN_yes
        .byte   19                              # DW_AT_language
        .byte   5                               # DW_FORM_data2
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   11                              # Abbreviation Code
        .byte   19                              # DW_TAG_structure_type
        .byte   0                               # DW_CHILDREN_no
        .byte   3                               # DW_AT_name
        .byte   14                              # DW_FORM_strp
        .byte   11                              # DW_AT_byte_size
        .byte   11                              # DW_FORM_data1
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   12                              # Abbreviation Code
        .byte   47                              # DW_TAG_template_type_parameter
        .byte   0                               # DW_CHILDREN_no
        .byte   73                              # DW_AT_type
        .byte   32                              # DW_FORM_ref_sig8
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   13                              # Abbreviation Code
        .byte   19                              # DW_TAG_structure_type
        .byte   1                               # DW_CHILDREN_yes
        .byte   3                               # DW_AT_name
        .byte   8                               # DW_FORM_string
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   0                               # EOM(3)
        .section        .debug_info,"",@progbits
.Lcu_begin0:
        .long   .Ldebug_info_end2-.Ldebug_info_start2 # Length of Unit
.Ldebug_info_start2:
        .short  5                               # DWARF version number
        .byte   1                               # DWARF Unit Type
        .byte   8                               # Address Size (in bytes)
        .long   .debug_abbrev                   # Offset Into Abbrev. Section
        .byte   3                               # Abbrev [3] 0xc:0x5f DW_TAG_compile_unit
        .byte   0                               # DW_AT_producer
        .short  33                              # DW_AT_language
        .byte   1                               # DW_AT_name
        .long   .Lstr_offsets_base0             # DW_AT_str_offsets_base
        .byte   2                               # DW_AT_comp_dir
        .byte   4                               # Abbrev [4] DW_TAG_variable
        .asciz  "s"                             # DW_AT_name
        .long   .LS_decl-.Lcu_begin0            # DW_AT_type
.LS_decl:
        .byte   9                               # Abbrev [9] DW_TAG_structure_type
        .quad   S_sig                           # DW_AT_signature
        .byte   8                               # Abbrev [8] 0x54:0x4 DW_TAG_base_type
        .byte   4                               # DW_AT_name
        .byte   5                               # DW_AT_encoding
        .byte   4                               # DW_AT_byte_size
        .byte   0                               # End Of Children Mark
.Ldebug_info_end2:
        .section        .debug_info,"G",@progbits,S_sig,comdat
.Ltu_begin2:
        .long   .Ldebug_info_end3-.Ldebug_info_start3 # Length of Unit
.Ldebug_info_start3:
        .short  5                               # DWARF version number
        .byte   2                               # DWARF Unit Type
        .byte   8                               # Address Size (in bytes)
        .long   .debug_abbrev                   # Offset Into Abbrev. Section
        .quad   S_sig                           # Type Signature
        .long   .LS_def-.Ltu_begin2             # Type DIE Offset
        .byte   1                               # Abbrev [1] DW_TAG_type_unit
        .short  33                              # DW_AT_language
        .long   .Lstr_offsets_base0             # DW_AT_str_offsets_base
.LS_def:
        .byte   13                              # Abbrev [13] DW_TAG_structure_type
        .asciz  "S"                             # DW_AT_name (simplified template name)
        .byte   6                               # Abbrev [6] 0x46:0xd DW_TAG_GNU_template_parameter_pack
        .byte   5                               # DW_AT_name
        .byte   7                               # Abbrev [7] 0x48:0x5 DW_TAG_template_type_parameter
        .long   .Lt1_decl-.Ltu_begin2           # DW_AT_type
        .byte   7                               # Abbrev [7] 0x4d:0x5 DW_TAG_template_type_parameter
        .long   .Lt2_decl-.Ltu_begin2           # DW_AT_type
        # Simulate DWARF emitted by GCC where the signature is directly in the type attribute.
        .byte   12                              # Abbrev [12] DW_TAG_template_type_parameter
        .quad   t1_sig                          # DW_AT_type
        .byte   0                               # End Of Children Mark
        .byte   0                               # End Of Children Mark
.Lt1_decl:
        .byte   9                               # Abbrev [9] 0x58:0x9 DW_TAG_structure_type
                                        # DW_AT_declaration
        .quad   t1_sig                          # DW_AT_signature
.Lt2_decl:
        .byte   9                               # Abbrev [9] 0x61:0x9 DW_TAG_structure_type
                                        # DW_AT_declaration
        .quad   t2_sig                          # DW_AT_signature
        .byte   0                               # End Of Children Mark
.Ldebug_info_end3:
        .section        .debug_str_offsets,"",@progbits
        .long   .Lstr_offsets_end-.Lstr_offsets_start # Length of String Offsets Set
.Lstr_offsets_start:
        .short  5
        .short  0
.Lstr_offsets_base0:
        .section        .debug_str,"MS",@progbits,1
.Linfo_string0:
        .asciz  "hand-written DWARF"
.Linfo_string1:
        .asciz  "test.cpp"
.Linfo_string2:
        .asciz  "/tmp"
.Linfo_string3:
        .asciz  "S"
.Linfo_string4:
        .asciz  "int"
.Linfo_string5:
        .asciz  "T"
.Linfo_string6:
        .asciz  "t1"
.Linfo_string7:
        .asciz  "t2"
        .section        .debug_str_offsets,"",@progbits
        .long   .Linfo_string0
        .long   .Linfo_string1
        .long   .Linfo_string2
        .long   .Linfo_string3
        .long   .Linfo_string4
        .long   .Linfo_string5
        .long   .Linfo_string6
        .long   .Linfo_string7
.Lstr_offsets_end:
