# Test location list handling, including the cases of invalid input. The exact
# behavior in the invalid cases is not particularly important, but it should be
# "reasonable".

# UNSUPPORTED: lldb-repro

# RUN: llvm-mc -triple=x86_64-pc-linux -filetype=obj %s --defsym LOC=0 > %t
# RUN: %lldb %t -o "image lookup -v -a 0" -o "image lookup -v -a 2" \
# RUN:   -o "image dump symfile" -o exit | FileCheck %s

# RUN: %lldb %t -o "image lookup -v -a 0 -show-variable-ranges" -o \
# RUN: "image lookup -v -a 2 -show-variable-ranges" \
# RUN: -o exit | FileCheck %s --check-prefix=ALL-RANGES

# RUN: llvm-mc -triple=x86_64-pc-linux -filetype=obj %s --defsym LOCLISTS=0 > %t
# RUN: %lldb %t -o "image lookup -v -a 0" -o "image lookup -v -a 2" \
# RUN:   -o "image dump symfile" -o exit | FileCheck %s --check-prefix=CHECK --check-prefix=LOCLISTS

# ALL-RANGES-LABEL: image lookup -v -a 0 -show-variable-ranges
# ALL-RANGES: Variable: id = {{.*}}, name = "x0", type = "int", valid ranges = <block>, location = [0x0000000000000000, 0x0000000000000001) -> DW_OP_reg5 RDI, [0x0000000000000001, 0x0000000000000006) -> DW_OP_reg0 RAX
# ALL-RANGES: Variable: id = {{.*}}, name = "x1", type = "int", valid ranges = <block>, location = <empty>
# ALL-RANGES-LABEL: image lookup -v -a 2 -show-variable-ranges
# ALL-RANGES:  Variable: id = {{.*}}, name = "x0", type = "int", valid ranges = <block>, location = [0x0000000000000000, 0x0000000000000001) -> DW_OP_reg5 RDI, [0x0000000000000001, 0x0000000000000006) -> DW_OP_reg0 RAX
# ALL-RANGES: Variable: id = {{.*}}, name = "x1", type = "int", valid ranges = <block>, location = <empty>
# ALL-RANGES: Variable: id = {{.*}}, name = "x3", type = "int", valid ranges = <block>, location = [0x0000000000000002, 0x0000000000000003) -> DW_OP_reg1 RDX

# CHECK-LABEL: image lookup -v -a 0
# CHECK: Variable: {{.*}}, name = "x0", type = "int", valid ranges = <block>, location = [0x0000000000000000, 0x0000000000000001) -> DW_OP_reg5 RDI
# CHECK: Variable: {{.*}}, name = "x1", type = "int", valid ranges = <block>, location = <empty>,

# CHECK-LABEL: image lookup -v -a 2
# CHECK: Variable: {{.*}}, name = "x0", type = "int", valid ranges = <block>, location = [0x0000000000000001, 0x0000000000000006) -> DW_OP_reg0 RAX
# CHECK: Variable: {{.*}}, name = "x1", type = "int", valid ranges = <block>, location = <empty>,
# CHECK: Variable: {{.*}}, name = "x3", type = "int", valid ranges = <block>, location = [0x0000000000000002, 0x0000000000000003) -> DW_OP_reg1 RDX

# CHECK-LABEL: image dump symfile
# CHECK: CompileUnit{0x00000000}
# CHECK:   Function{
# CHECK:     Variable{{.*}}, name = "x0", {{.*}}, scope = parameter, location =
# CHECK-NEXT:  [0x0000000000000000, 0x0000000000000001): DW_OP_reg5 RDI
# CHECK-NEXT:  [0x0000000000000001, 0x0000000000000006): DW_OP_reg0 RAX
# CHECK:     Variable{{.*}}, name = "x1", {{.*}}, scope = parameter
# CHECK:     Variable{{.*}}, name = "x2", {{.*}}, scope = parameter, location = 0x00000000: error: unexpected end of data
# CHECK:     Variable{{.*}}, name = "x3", {{.*}}, scope = parameter, location =
# CHECK-NEXT:  [0x0000000000000002, 0x0000000000000003): DW_OP_reg1 RDX
# LOCLISTS:  Variable{{.*}}, name = "x4", {{.*}}, scope = parameter, location =
# LOCLISTS-NEXT: DW_LLE_startx_length   (0x000000000000dead, 0x0000000000000001): DW_OP_reg2 RCX

.ifdef LOC
.macro OFFSET_PAIR lo hi
        .quad \lo
        .quad \hi
.endm

.macro BASE_ADDRESS base
        .quad -1
        .quad \base
.endm

.macro EXPR_SIZE sz
        .short \sz
.endm

.macro END_OF_LIST
        .quad 0
        .quad 0
.endm
.endif

.ifdef LOCLISTS
.macro OFFSET_PAIR lo hi
        .byte   4                       # DW_LLE_offset_pair
        .uleb128 \lo
        .uleb128 \hi
.endm

.macro BASE_ADDRESS base
        .byte   6                       # DW_LLE_base_address
        .quad \base
.endm

.macro EXPR_SIZE sz
        .uleb128 \sz
.endm

.macro END_OF_LIST
        .byte   0                       # DW_LLE_end_of_list
.endm
.endif

        .type   f,@function
f:                                      # @f
.Lfunc_begin0:
        nop
.Ltmp0:
        nop
.Ltmp1:
        nop
.Ltmp2:
        nop
.Ltmp3:
        nop
.Ltmp4:
        nop
.Lfunc_end0:
        .size   f, .Lfunc_end0-f

        .section        .debug_str,"MS",@progbits,1
.Linfo_string0:
        .asciz  "Hand-written DWARF"
.Linfo_string3:
        .asciz  "f"
.Linfo_string4:
        .asciz  "int"

.ifdef LOC
        .section        .debug_loc,"",@progbits
.endif
.ifdef LOCLISTS
        .section        .debug_loclists,"",@progbits
        .long   .Ldebug_loclist_table_end0-.Ldebug_loclist_table_start0 # Length
.Ldebug_loclist_table_start0:
        .short  5                       # Version
        .byte   8                       # Address size
        .byte   0                       # Segment selector size
        .long   0                       # Offset entry count
.endif
.Ldebug_loc0:
        OFFSET_PAIR .Lfunc_begin0-.Lfunc_begin0, .Ltmp0-.Lfunc_begin0
        EXPR_SIZE 1
        .byte   85                      # super-register DW_OP_reg5
        OFFSET_PAIR   .Ltmp0-.Lfunc_begin0, .Lfunc_end0-.Lfunc_begin0
        EXPR_SIZE 1
        .byte   80                      # super-register DW_OP_reg0
        END_OF_LIST

.Ldebug_loc3:
        BASE_ADDRESS .Ltmp1
        OFFSET_PAIR .Ltmp1-.Ltmp1, .Ltmp2-.Ltmp1
        EXPR_SIZE 1
        .byte   81                      # super-register DW_OP_reg1
        END_OF_LIST

.ifdef LOCLISTS
.Ldebug_loc4:
        .byte   3                       # DW_LLE_startx_length
        .uleb128 0xdead
        .uleb128 1
        EXPR_SIZE 1
        .byte   82                      # super-register DW_OP_reg2
        END_OF_LIST
.endif

.Ldebug_loc2:
        OFFSET_PAIR .Lfunc_begin0-.Lfunc_begin0, .Lfunc_end0-.Lfunc_begin0
        EXPR_SIZE  0xdead
.Ldebug_loclist_table_end0:

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
        .byte   17                      # DW_AT_low_pc
        .byte   1                       # DW_FORM_addr
        .byte   18                      # DW_AT_high_pc
        .byte   6                       # DW_FORM_data4
        .byte   3                       # DW_AT_name
        .byte   14                      # DW_FORM_strp
        .byte   73                      # DW_AT_type
        .byte   19                      # DW_FORM_ref4
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   3                       # Abbreviation Code
        .byte   5                       # DW_TAG_formal_parameter
        .byte   0                       # DW_CHILDREN_no
        .byte   2                       # DW_AT_location
        .byte   23                      # DW_FORM_sec_offset
        .byte   3                       # DW_AT_name
        .byte   8                       # DW_FORM_string
        .byte   73                      # DW_AT_type
        .byte   19                      # DW_FORM_ref4
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
.ifdef LOC
        .short  4                       # DWARF version number
        .long   .debug_abbrev           # Offset Into Abbrev. Section
        .byte   8                       # Address Size (in bytes)
.endif
.ifdef LOCLISTS
        .short  5                       # DWARF version number
        .byte   1                       # DWARF Unit Type
        .byte   8                       # Address Size (in bytes)
        .long   .debug_abbrev           # Offset Into Abbrev. Section
.endif
        .byte   1                       # Abbrev [1] 0xb:0x50 DW_TAG_compile_unit
        .long   .Linfo_string0          # DW_AT_producer
        .short  12                      # DW_AT_language
        .quad   .Lfunc_begin0           # DW_AT_low_pc
        .long   .Lfunc_end0-.Lfunc_begin0 # DW_AT_high_pc
        .byte   2                       # Abbrev [2] 0x2a:0x29 DW_TAG_subprogram
        .quad   .Lfunc_begin0           # DW_AT_low_pc
        .long   .Lfunc_end0-.Lfunc_begin0 # DW_AT_high_pc
        .long   .Linfo_string3          # DW_AT_name
        .long   .Lint                   # DW_AT_type
        .byte   3                       # Abbrev [3] DW_TAG_formal_parameter
        .long   .Ldebug_loc0            # DW_AT_location
        .asciz  "x0"                    # DW_AT_name
        .long   .Lint-.Lcu_begin0       # DW_AT_type
        .byte   3                       # Abbrev [3] DW_TAG_formal_parameter
        .long   0xdeadbeef              # DW_AT_location
        .asciz  "x1"                    # DW_AT_name
        .long   .Lint-.Lcu_begin0       # DW_AT_type
        .byte   3                       # Abbrev [3] DW_TAG_formal_parameter
        .long   .Ldebug_loc2            # DW_AT_location
        .asciz  "x2"                    # DW_AT_name
        .long   .Lint-.Lcu_begin0       # DW_AT_type
        .byte   3                       # Abbrev [3] DW_TAG_formal_parameter
        .long   .Ldebug_loc3            # DW_AT_location
        .asciz  "x3"                    # DW_AT_name
        .long   .Lint-.Lcu_begin0       # DW_AT_type
.ifdef LOCLISTS
        .byte   3                       # Abbrev [3] DW_TAG_formal_parameter
        .long   .Ldebug_loc4            # DW_AT_location
        .asciz  "x4"                    # DW_AT_name
        .long   .Lint-.Lcu_begin0       # DW_AT_type
.endif
        .byte   0                       # End Of Children Mark
.Lint:
        .byte   4                       # Abbrev [4] 0x53:0x7 DW_TAG_base_type
        .long   .Linfo_string4          # DW_AT_name
        .byte   5                       # DW_AT_encoding
        .byte   4                       # DW_AT_byte_size
        .byte   0                       # End Of Children Mark
.Ldebug_info_end0:
