# Test handling of (optimized-out/location-less) variables whose value is
# specified by DW_AT_const_value

# RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux %s -o %t
# RUN: %lldb %t \
# RUN:   -o "target variable udata data1 data2 data4 data8 string strp ref4 udata_ptr" \
# RUN:   -o exit | FileCheck %s

# CHECK-LABEL: target variable
## Variable specified via DW_FORM_udata. This is typical for clang (10).
# CHECK: (unsigned long) udata = 4742474247424742
## Variables specified via fixed-size forms. This is typical for gcc (9).
# CHECK: (unsigned long) data1 = 47
# CHECK: (unsigned long) data2 = 4742
# CHECK: (unsigned long) data4 = 47424742
# CHECK: (unsigned long) data8 = 4742474247424742
## Variables specified using string forms. This behavior purely speculative -- I
## don't know of any compiler that would represent character strings this way.
# CHECK: (char[7]) string = "string"
# CHECK: (char[7]) strp = "strp"
## Bogus attribute form. Let's make sure we don't crash at least.
# CHECK: (char[7]) ref4 = <empty constant data>
## A variable of pointer type.
# CHECK: (unsigned long *) udata_ptr = 0xdeadbeefbaadf00d

        .section        .debug_abbrev,"",@progbits
        .byte   1                       # Abbreviation Code
        .byte   17                      # DW_TAG_compile_unit
        .byte   1                       # DW_CHILDREN_yes
        .byte   37                      # DW_AT_producer
        .byte   8                       # DW_FORM_string
        .byte   3                       # DW_AT_name
        .byte   8                       # DW_FORM_string
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   2                       # Abbreviation Code
        .byte   15                      # DW_TAG_pointer_type
        .byte   0                       # DW_CHILDREN_no
        .byte   73                      # DW_AT_type
        .byte   19                      # DW_FORM_ref4
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   4                       # Abbreviation Code
        .byte   1                       # DW_TAG_array_type
        .byte   1                       # DW_CHILDREN_yes
        .byte   73                      # DW_AT_type
        .byte   19                      # DW_FORM_ref4
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   5                       # Abbreviation Code
        .byte   33                      # DW_TAG_subrange_type
        .byte   0                       # DW_CHILDREN_no
        .byte   73                      # DW_AT_type
        .byte   19                      # DW_FORM_ref4
        .byte   55                      # DW_AT_count
        .byte   11                      # DW_FORM_data1
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   6                       # Abbreviation Code
        .byte   36                      # DW_TAG_base_type
        .byte   0                       # DW_CHILDREN_no
        .byte   3                       # DW_AT_name
        .byte   8                       # DW_FORM_string
        .byte   11                      # DW_AT_byte_size
        .byte   11                      # DW_FORM_data1
        .byte   62                      # DW_AT_encoding
        .byte   11                      # DW_FORM_data1
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
.macro var code, form
        .byte   \code                   # Abbreviation Code
        .byte   52                      # DW_TAG_variable
        .byte   0                       # DW_CHILDREN_no
        .byte   3                       # DW_AT_name
        .byte   8                       # DW_FORM_string
        .byte   73                      # DW_AT_type
        .byte   19                      # DW_FORM_ref4
        .byte   28                      # DW_AT_const_value
        .byte   \form
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
.endm
        var 10, 0xf                     # DW_FORM_udata
        var 11, 0xb                     # DW_FORM_data1
        var 12, 0x5                     # DW_FORM_data2
        var 13, 0x6                     # DW_FORM_data4
        var 14, 0x7                     # DW_FORM_data8
        var 15, 0x8                     # DW_FORM_string
        var 16, 0xe                     # DW_FORM_strp
        var 17, 0x13                    # DW_FORM_ref4
        .byte   0                       # EOM(3)
        .section        .debug_info,"",@progbits
.Lcu_begin0:
        .long   .Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
        .short  4                       # DWARF version number
        .long   .debug_abbrev           # Offset Into Abbrev. Section
        .byte   8                       # Address Size (in bytes)
        .byte   1                       # Abbrev DW_TAG_compile_unit
        .asciz  "Hand-written DWARF"    # DW_AT_producer
        .asciz  "const.c"               # DW_AT_name
.Lchar_arr:
        .byte   4                       # Abbrev DW_TAG_array_type
        .long   .Lchar-.Lcu_begin0      # DW_AT_type
        .byte   5                       # Abbrev DW_TAG_subrange_type
        .long   .Lulong-.Lcu_begin0     # DW_AT_type
        .byte   7                       # DW_AT_count
        .byte   0                       # End Of Children Mark
.Lchar:
        .byte   6                       # Abbrev DW_TAG_base_type
        .asciz  "char"                  # DW_AT_name
        .byte   1                       # DW_AT_byte_size
        .byte   6                       # DW_AT_encoding
.Lulong:
        .byte   6                       # Abbrev DW_TAG_base_type
        .asciz  "unsigned long"         # DW_AT_name
        .byte   8                       # DW_AT_byte_size
        .byte   7                       # DW_AT_encoding
.Lulong_ptr:
        .byte   2                       # Abbrev DW_TAG_pointer_type
        .long   .Lulong-.Lcu_begin0     # DW_AT_type

        .byte   10                      # Abbrev DW_TAG_variable
        .asciz  "udata"                 # DW_AT_name
        .long   .Lulong-.Lcu_begin0     # DW_AT_type
        .uleb128 4742474247424742       # DW_AT_const_value

        .byte   11                      # Abbrev DW_TAG_variable
        .asciz  "data1"                 # DW_AT_name
        .long   .Lulong-.Lcu_begin0     # DW_AT_type
        .byte   47                      # DW_AT_const_value

        .byte   12                      # Abbrev DW_TAG_variable
        .asciz  "data2"                 # DW_AT_name
        .long   .Lulong-.Lcu_begin0     # DW_AT_type
        .word   4742                    # DW_AT_const_value

        .byte   13                      # Abbrev DW_TAG_variable
        .asciz  "data4"                 # DW_AT_name
        .long   .Lulong-.Lcu_begin0     # DW_AT_type
        .long   47424742                # DW_AT_const_value

        .byte   14                      # Abbrev DW_TAG_variable
        .asciz  "data8"                 # DW_AT_name
        .long   .Lulong-.Lcu_begin0     # DW_AT_type
        .quad   4742474247424742        # DW_AT_const_value

        .byte   15                      # Abbrev DW_TAG_variable
        .asciz  "string"                # DW_AT_name
        .long   .Lchar_arr-.Lcu_begin0  # DW_AT_type
        .asciz  "string"                # DW_AT_const_value

        .byte   16                      # Abbrev DW_TAG_variable
        .asciz  "strp"                  # DW_AT_name
        .long   .Lchar_arr-.Lcu_begin0  # DW_AT_type
        .long   .Lstrp                  # DW_AT_const_value

        .byte   17                      # Abbrev DW_TAG_variable
        .asciz  "ref4"                  # DW_AT_name
        .long   .Lchar_arr-.Lcu_begin0  # DW_AT_type
        .long   .Lulong-.Lcu_begin0     # DW_AT_const_value

        .byte   10                      # Abbrev DW_TAG_variable
        .asciz  "udata_ptr"             # DW_AT_name
        .long   .Lulong_ptr-.Lcu_begin0 # DW_AT_type
        .uleb128 0xdeadbeefbaadf00d     # DW_AT_const_value

        .byte   0                       # End Of Children Mark
.Ldebug_info_end0:

        .section        .debug_str,"MS",@progbits,1
.Lstrp:
        .asciz "strp"
