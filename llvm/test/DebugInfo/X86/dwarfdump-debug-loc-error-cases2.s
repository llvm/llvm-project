# RUN: llvm-mc -triple=x86_64-pc-linux -filetype=obj %s > %t
# RUN: not llvm-dwarfdump %t 2>%t.err | FileCheck %s
# RUN: FileCheck %s < %t.err -check-prefix=ERR

# CHECK:      DW_AT_name        ("x0")
# CHECK-NEXT: DW_AT_location    (0x00000000
# CHECK-NEXT:    [0x0000000000000000,  0x0000000000000002): DW_OP_reg5 RDI
# CHECK-NEXT:    [0x0000000000000002,  0x0000000000000003): DW_OP_reg0 RAX)

# CHECK:      DW_AT_name        ("x1")
# CHECK-NEXT: DW_AT_location    (0xdeadbeef: )
# ERR:    error: offset 0xdeadbeef is beyond the end of data at 0x48

# CHECK:      DW_AT_name        ("x2")
# CHECK-NEXT: DW_AT_location    (0x00000036: )
# ERR:    error: unexpected end of data at offset 0x48 while reading [0x48, 0xdef5)


        .type   f,@function
f:                                      # @f
.Lfunc_begin0:
        movl    %edi, %eax
.Ltmp0:
        retq
.Ltmp1:
.Lfunc_end0:
        .size   f, .Lfunc_end0-f

        .section        .debug_str,"MS",@progbits,1
.Linfo_string0:
        .asciz  "Hand-written DWARF"
.Linfo_string3:
        .asciz  "f"
.Linfo_string4:
        .asciz  "int"
.Lx0:
        .asciz  "x0"
.Lx1:
        .asciz  "x1"
.Lx2:
        .asciz  "x2"

        .section        .debug_loc,"",@progbits
.Ldebug_loc0:
        .quad   .Lfunc_begin0-.Lfunc_begin0
        .quad   .Ltmp0-.Lfunc_begin0
        .short  1                       # Loc expr size
        .byte   85                      # super-register DW_OP_reg5
        .quad   .Ltmp0-.Lfunc_begin0
        .quad   .Lfunc_end0-.Lfunc_begin0
        .short  1                       # Loc expr size
        .byte   80                      # super-register DW_OP_reg0
        .quad   0
        .quad   0
.Ldebug_loc2:
        .quad   .Lfunc_begin0-.Lfunc_begin0
        .quad   .Lfunc_end0-.Lfunc_begin0
        .short  0xdead                  # Loc expr size

        .section        .debug_abbrev,"",@progbits
        .byte   1                       # Abbreviation Code
        .byte   17                      # DW_TAG_compile_unit
        .byte   1                       # DW_CHILDREN_yes
        .byte   37                      # DW_AT_producer
        .byte   14                      # DW_FORM_strp
        .byte   19                      # DW_AT_language
        .byte   5                       # DW_FORM_data2
        .byte   17                      # DW_AT_low_pc
        .byte   1                       # DW_FORM_addr
        .byte   18                      # DW_AT_high_pc
        .byte   6                       # DW_FORM_data4
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   2                       # Abbreviation Code
        .byte   46                      # DW_TAG_subprogram
        .byte   1                       # DW_CHILDREN_yes
        .byte   3                       # DW_AT_name
        .byte   14                      # DW_FORM_strp
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   3                       # Abbreviation Code
        .byte   5                       # DW_TAG_formal_parameter
        .byte   0                       # DW_CHILDREN_no
        .byte   3                       # DW_AT_name
        .byte   14                      # DW_FORM_strp
        .byte   2                       # DW_AT_location
        .byte   23                      # DW_FORM_sec_offset
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   4                       # Abbreviation Code
        .byte   36                      # DW_TAG_base_type
        .byte   0                       # DW_CHILDREN_no
        .byte   3                       # DW_AT_name
        .byte   14                      # DW_FORM_strp
        .byte   62                      # DW_AT_encoding
        .byte   11                      # DW_FORM_data1
        .byte   11                      # DW_AT_byte_size
        .byte   11                      # DW_FORM_data1
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   0                       # EOM(3)

        .section        .debug_info,"",@progbits
.Lcu_begin0:
        .long   .Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
        .short  4                       # DWARF version number
        .long   .debug_abbrev           # Offset Into Abbrev. Section
        .byte   8                       # Address Size (in bytes)
        .byte   1                       # Abbrev [1] 0xb:0x50 DW_TAG_compile_unit
        .long   .Linfo_string0          # DW_AT_producer
        .short  12                      # DW_AT_language
        .quad   0                       # DW_AT_low_pc
        .long   0x100                   # DW_AT_high_pc
        .byte   2                       # Abbrev [2] 0x2a:0x29 DW_TAG_subprogram
        .long   .Linfo_string3          # DW_AT_name
        .byte   3                       # Abbrev [3] DW_TAG_formal_parameter
        .long   .Lx0                    # DW_AT_name
        .long   .Ldebug_loc0            # DW_AT_location
        .byte   3                       # Abbrev [3] DW_TAG_formal_parameter
        .long   .Lx1                    # DW_AT_name
        .long   0xdeadbeef              # DW_AT_location
        .byte   3                       # Abbrev [3] DW_TAG_formal_parameter
        .long   .Lx2                    # DW_AT_name
        .long   .Ldebug_loc2            # DW_AT_location
        .byte   0                       # End Of Children Mark
        .byte   0                       # End Of Children Mark
.Ldebug_info_end0:
