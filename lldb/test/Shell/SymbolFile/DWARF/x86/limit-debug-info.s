# REQUIRES: lld
# RUN: llvm-mc --triple=x86_64-pc-windows --filetype=obj --defsym DLL=0 %s >%t.dll.o
# RUN: llvm-mc --triple=x86_64-pc-windows --filetype=obj --defsym EXE=0 %s >%t.exe.o
# RUN: lld-link /OUT:%t.dll %t.dll.o /SUBSYSTEM:console /dll /noentry /debug
# RUN: lld-link /OUT:%t.exe %t.exe.o /SUBSYSTEM:console /debug /force
# RUN: %lldb %t.exe -o "target modules add %t.dll" -o "p var" \
# RUN:   -o exit 2>&1 | FileCheck %s

# CHECK: (lldb) p var
# CHECK: (A) $0 = (member = 47)

        .section        .debug_abbrev,"dr"
.Lsection_abbrev:
        .byte   1                       # Abbreviation Code
        .byte   17                      # DW_TAG_compile_unit
        .byte   1                       # DW_CHILDREN_yes
        .byte   37                      # DW_AT_producer
        .byte   8                       # DW_FORM_string
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   2                       # Abbreviation Code
        .byte   52                      # DW_TAG_variable
        .byte   0                       # DW_CHILDREN_no
        .byte   3                       # DW_AT_name
        .byte   8                       # DW_FORM_string
        .byte   73                      # DW_AT_type
        .byte   19                      # DW_FORM_ref4
        .byte   2                       # DW_AT_location
        .byte   24                      # DW_FORM_exprloc
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   3                       # Abbreviation Code
        .byte   19                      # DW_TAG_structure_type
        .byte   1                       # DW_CHILDREN_yes
        .byte   3                       # DW_AT_name
        .byte   8                       # DW_FORM_string
        .byte   11                      # DW_AT_byte_size
        .byte   11                      # DW_FORM_data1
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   4                       # Abbreviation Code
        .byte   13                      # DW_TAG_member
        .byte   0                       # DW_CHILDREN_no
        .byte   3                       # DW_AT_name
        .byte   8                       # DW_FORM_string
        .byte   73                      # DW_AT_type
        .byte   19                      # DW_FORM_ref4
        .byte   56                      # DW_AT_data_member_location
        .byte   11                      # DW_FORM_data1
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   5                       # Abbreviation Code
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
        .byte   6                       # Abbreviation Code
        .byte   19                      # DW_TAG_structure_type
        .byte   0                       # DW_CHILDREN_no
        .byte   3                       # DW_AT_name
        .byte   8                       # DW_FORM_string
        .byte   60                      # DW_AT_declaration
        .byte   25                      # DW_FORM_flag_present
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   0                       # EOM(3)

.ifdef DLL
        .section        .debug_info,"dr"
.Lsection_info:
.Lcu_begin0:
        .long   .Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
        .short  4                       # DWARF version number
        .secrel32       .Lsection_abbrev # Offset Into Abbrev. Section
        .byte   8                       # Address Size (in bytes)
        .byte   1                       # Abbrev [1] 0xb:0x4a DW_TAG_compile_unit
        .asciz  "Hand-written DWARF"    # DW_AT_producer
        .byte   3                       # Abbrev [3] 0x37:0x16 DW_TAG_structure_type
        .asciz  "A"                     # DW_AT_name
        .byte   4                       # DW_AT_byte_size
        .byte   4                       # Abbrev [4] 0x40:0xc DW_TAG_member
        .asciz  "member"                # DW_AT_name
        .long   .Lint-.Lsection_info    # DW_AT_type
        .byte   0                       # DW_AT_data_member_location
        .byte   0                       # End Of Children Mark
.Lint:
        .byte   5                       # Abbrev [5] 0x4d:0x7 DW_TAG_base_type
        .asciz  "int"                   # DW_AT_name
        .byte   5                       # DW_AT_encoding
        .byte   4                       # DW_AT_byte_size
        .byte   0                       # End Of Children Mark
.Ldebug_info_end0:
.endif

.ifdef EXE
        .data
        .globl  var
        .p2align        2
var:
        .long   47

        .section        .debug_info,"dr"
.Lsection_info:
.Lcu_begin0:
        .long   .Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
        .short  4                       # DWARF version number
        .secrel32       .Lsection_abbrev # Offset Into Abbrev. Section
        .byte   8                       # Address Size (in bytes)
        .byte   1                       # Abbrev [1] 0xb:0x4a DW_TAG_compile_unit
        .asciz  "Hand-written DWARF"    # DW_AT_producer
        .byte   2                       # Abbrev [2] 0x1e:0x19 DW_TAG_variable
        .asciz  "var"                   # DW_AT_name
        .long   .LA-.Lsection_info      # DW_AT_type
        .byte   9                       # DW_AT_location
        .byte   3
        .quad   var
.LA:
        .byte   6                       # Abbrev [6] 0x37:0x16 DW_TAG_structure_type
        .asciz  "A"                     # DW_AT_name
                                        # DW_AT_declaration
        .byte   0                       # End Of Children Mark
.Ldebug_info_end0:
.endif
