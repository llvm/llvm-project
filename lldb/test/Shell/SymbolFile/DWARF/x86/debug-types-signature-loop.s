# REQUIRES: lld

# RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux -o %t.o %s
# RUN: ld.lld %t.o -o %t
# RUN: %lldb %t -o "target variable e" -b | FileCheck %s

# CHECK: e = <could not resolve type>

        .type   e,@object               # @e
        .section        .rodata,"a",@progbits
        .globl  e
        .p2align        2
e:
        .long   0                       # 0x0
        .size   e, 4

.Lstr_offsets_base0:
        .section        .debug_str,"MS",@progbits,1
.Linfo_string0:
        .asciz  "Hand-written DWARF"
.Linfo_string1:
        .asciz  "a.cpp"            
.Linfo_string3:
        .asciz  "e"           
.Linfo_string4:
        .asciz  "unsigned int"
.Linfo_string5:
        .asciz  "e1"          
.Linfo_string6:
        .asciz  "E"           

        .section        .debug_abbrev,"",@progbits
        .byte   1                       # Abbreviation Code
        .byte   65                      # DW_TAG_type_unit
        .byte   1                       # DW_CHILDREN_yes
        .byte   19                      # DW_AT_language
        .byte   5                       # DW_FORM_data2
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   5                       # Abbreviation Code
        .byte   17                      # DW_TAG_compile_unit
        .byte   1                       # DW_CHILDREN_yes
        .byte   37                      # DW_AT_producer
        .byte   14                      # DW_FORM_strp
        .byte   19                      # DW_AT_language
        .byte   5                       # DW_FORM_data2
        .byte   3                       # DW_AT_name
        .byte   14                      # DW_FORM_strp
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   6                       # Abbreviation Code
        .byte   52                      # DW_TAG_variable
        .byte   0                       # DW_CHILDREN_no
        .byte   3                       # DW_AT_name
        .byte   14                      # DW_FORM_strp
        .byte   73                      # DW_AT_type
        .byte   19                      # DW_FORM_ref4
        .byte   2                       # DW_AT_location
        .byte   24                      # DW_FORM_exprloc
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   8                       # Abbreviation Code
        .byte   4                       # DW_TAG_enumeration_type
        .byte   0                       # DW_CHILDREN_no
        .byte   60                      # DW_AT_declaration
        .byte   25                      # DW_FORM_flag_present
        .byte   105                     # DW_AT_signature
        .byte   32                      # DW_FORM_ref_sig8
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   0                       # EOM(3)

        .section        .debug_info,"",@progbits
.Ltu_begin0:
        .long   .Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
        .short  5                       # DWARF version number
        .byte   2                       # DWARF Unit Type
        .byte   8                       # Address Size (in bytes)
        .long   .debug_abbrev           # Offset Into Abbrev. Section
        .quad   5390450678491038984     # Type Signature
        .long   .LE-.Ltu_begin0         # Type DIE Offset
        .byte   1                       # Abbrev [1] 0x18:0x1d DW_TAG_type_unit
        .short  4                       # DW_AT_language
.LE:
        .byte   8                       # Abbrev [8] 0x23:0xd DW_TAG_enumeration_type
                                        # DW_AT_declaration
        .quad   5390450678491038984     # DW_AT_signature
.Lbase:
        .byte   0                       # End Of Children Mark
.Ldebug_info_end0:

.Lcu_begin0:
        .long   .Ldebug_info_end1-.Ldebug_info_start1 # Length of Unit
.Ldebug_info_start1:
        .short  5                       # DWARF version number
        .byte   1                       # DWARF Unit Type
        .byte   8                       # Address Size (in bytes)
        .long   .debug_abbrev           # Offset Into Abbrev. Section
        .byte   5                       # Abbrev [5] 0xc:0x2c DW_TAG_compile_unit
        .long   .Linfo_string0          # DW_AT_producer
        .short  4                       # DW_AT_language
        .long   .Linfo_string1          # DW_AT_name
        .byte   6                       # Abbrev [6] 0x1e:0xb DW_TAG_variable
        .long   .Linfo_string3          # DW_AT_name
        .long   .LE_sig-.Lcu_begin0     # DW_AT_type
        .byte   9                       # DW_AT_location
        .byte   3
        .quad   e
.LE_sig:
        .byte   8                       # Abbrev [8] 0x2e:0x9 DW_TAG_enumeration_type
                                        # DW_AT_declaration
        .quad   5390450678491038984     # DW_AT_signature
        .byte   0                       # End Of Children Mark
.Ldebug_info_end1:
