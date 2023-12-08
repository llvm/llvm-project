# The below program is roughly derived from the following C program.
# To see the annotated debug info, look for the section 
# '.section    .debug_info' below.
#
# __attribute__((always_inline))
# void f(void* unused1, int used, int unused2, int partial, int unused3) {
#   used += partial;
#   printf("f %i", partial);
#   printf("f %i", used);   // |partial| is not live at this line.
# }
#
# void g(int unused) {
#   printf("Hello");
# }
#
# __attribute__((noinline))
# void other() {
#   f(nullptr, 1, 0, 2, 0);
# }
#
# int main(int argc, char** argv) {
#   f(argv, 42, 1, argc, 2);
#   g(1);
#   other();
#   return 0;
# }

    .text
    .file    "unused-inlined-params.c"

.Lcu_begin:

    .globl    other
other:
    nop
.Linlined_f_in_other:
break_at_inlined_f_in_other:
    callq    printf        # Omitted the setup of arguments.
.Linlined_f_in_other_between_printfs:
    callq    printf        # Omitted the setup of arguments.
.Linlined_f_in_other_end:
    retq
.Lother_end:
    .size    other, .Lother_end-other

    .globl    main
main:
    .file    1 "/example" "unused-inlined-params.c"
    movl    $1, %esi
.Linlined_f:
break_at_inlined_f_in_main:
    leal    42(%rsi), %ebx
.Linlined_f_before_printf:
    callq    printf        # Omitted the setup of arguments.
.Linlined_f_between_printfs:
break_at_inlined_f_in_main_between_printfs:
    callq    printf        # Omitted the setup of arguments.
.Linlined_f_end:
.Linlined_g:
break_at_inlined_g_in_main:
    callq    printf        # Omitted the setup of arguments.
.Linlined_g_end:
    callq    other
    retq
.Lmain_end:
    .size    main, .Lmain_end-main

# Dummy printf implementation.
printf:
    retq

# Simple entry point to make the linker happy.
    .globl  _start
_start:
    jmp     main

.Lcu_end:


    .section    .debug_loc,"",@progbits
.Ldebug_loc_partial:
    .quad    .Linlined_f-.Lcu_begin
    .quad    .Linlined_f_between_printfs-.Lcu_begin
    .short   1                               # Loc expr size
    .byte    84                              # super-register DW_OP_reg4
    .quad    0
    .quad    0
.Ldebug_loc_used:
    .quad    .Linlined_f-.Lcu_begin
    .quad    .Linlined_f_before_printf-.Lcu_begin
    .short   3                               # Loc expr size
    .byte    17                              # DW_OP_consts
    .byte    42                              # value
    .byte    159                             # DW_OP_stack_value
    .quad    .Linlined_f_before_printf-.Lcu_begin
    .quad    .Linlined_f_end-.Lcu_begin
    .short   1                               # Loc expr size
    .byte    83                              # super-register DW_OP_reg3
    .quad    0
    .quad    0
.Ldebug_loc_partial_in_other:
    .quad    .Linlined_f_in_other-.Lcu_begin
    .quad    .Linlined_f_in_other_between_printfs-.Lcu_begin
    .short   3                               # Loc expr size
    .byte    17                              # DW_OP_consts
    .byte    2                               # value
    .byte    159                             # DW_OP_stack_value
    .quad    0
    .quad    0
.Ldebug_loc_used_in_other:
    .quad    .Linlined_f_in_other-.Lcu_begin
    .quad    .Linlined_f_in_other_end-.Lcu_begin
    .short   3                               # Loc expr size
    .byte    17                              # DW_OP_consts
    .byte    1                               # value
    .byte    159                             # DW_OP_stack_value
    .quad    0
    .quad    0

    .section    .debug_abbrev,"",@progbits
    .byte    1                               # Abbreviation Code
    .byte    17                              # DW_TAG_compile_unit
    .byte    1                               # DW_CHILDREN_yes
    .byte    3                               # DW_AT_name
    .byte    14                              # DW_FORM_strp
    .byte    16                              # DW_AT_stmt_list
    .byte    23                              # DW_FORM_sec_offset
    .byte    17                              # DW_AT_low_pc
    .byte    1                               # DW_FORM_addr
    .byte    18                              # DW_AT_high_pc
    .byte    6                               # DW_FORM_data4
    .byte    0                               # EOM(1)
    .byte    0                               # EOM(2)

    .byte    4                               # Abbreviation Code
    .byte    5                               # DW_TAG_formal_parameter
    .byte    0                               # DW_CHILDREN_no
    .byte    2                               # DW_AT_location
    .byte    23                              # DW_FORM_sec_offset
    .byte    49                              # DW_AT_abstract_origin
    .byte    19                              # DW_FORM_ref4
    .byte    0                               # EOM(1)
    .byte    0                               # EOM(2)

    .byte    5                               # Abbreviation Code
    .byte    46                              # DW_TAG_subprogram
    .byte    1                               # DW_CHILDREN_yes
    .byte    3                               # DW_AT_name
    .byte    14                              # DW_FORM_strp
    .byte    58                              # DW_AT_decl_file
    .byte    11                              # DW_FORM_data1
    .byte    59                              # DW_AT_decl_line
    .byte    11                              # DW_FORM_data1
    .byte    39                              # DW_AT_prototyped
    .byte    25                              # DW_FORM_flag_present
    .byte    63                              # DW_AT_external
    .byte    25                              # DW_FORM_flag_present
    .byte    32                              # DW_AT_inline
    .byte    11                              # DW_FORM_data1
    .byte    0                               # EOM(1)
    .byte    0                               # EOM(2)

    .byte    6                               # Abbreviation Code
    .byte    5                               # DW_TAG_formal_parameter
    .byte    0                               # DW_CHILDREN_no
    .byte    3                               # DW_AT_name
    .byte    14                              # DW_FORM_strp
    .byte    58                              # DW_AT_decl_file
    .byte    11                              # DW_FORM_data1
    .byte    59                              # DW_AT_decl_line
    .byte    11                              # DW_FORM_data1
    .byte    73                              # DW_AT_type
    .byte    19                              # DW_FORM_ref4
    .byte    0                               # EOM(1)
    .byte    0                               # EOM(2)

    .byte    7                               # Abbreviation Code
    .byte    15                              # DW_TAG_pointer_type
    .byte    0                               # DW_CHILDREN_no
    .byte    0                               # EOM(1)
    .byte    0                               # EOM(2)

    .byte    8                               # Abbreviation Code
    .byte    36                              # DW_TAG_base_type
    .byte    0                               # DW_CHILDREN_no
    .byte    3                               # DW_AT_name
    .byte    14                              # DW_FORM_strp
    .byte    62                              # DW_AT_encoding
    .byte    11                              # DW_FORM_data1
    .byte    11                              # DW_AT_byte_size
    .byte    11                              # DW_FORM_data1
    .byte    0                               # EOM(1)
    .byte    0                               # EOM(2)

    .byte    9                               # Abbreviation Code
    .byte    46                              # DW_TAG_subprogram
    .byte    1                               # DW_CHILDREN_yes
    .byte    17                              # DW_AT_low_pc
    .byte    1                               # DW_FORM_addr
    .byte    18                              # DW_AT_high_pc
    .byte    6                               # DW_FORM_data4
    .byte    64                              # DW_AT_frame_base
    .byte    24                              # DW_FORM_exprloc
    .byte    3                               # DW_AT_name
    .byte    14                              # DW_FORM_strp
    .byte    58                              # DW_AT_decl_file
    .byte    11                              # DW_FORM_data1
    .byte    59                              # DW_AT_decl_line
    .byte    11                              # DW_FORM_data1
    .byte    39                              # DW_AT_prototyped
    .byte    25                              # DW_FORM_flag_present
    .byte    73                              # DW_AT_type
    .byte    19                              # DW_FORM_ref4
    .byte    63                              # DW_AT_external
    .byte    25                              # DW_FORM_flag_present
    .byte    0                               # EOM(1)
    .byte    0                               # EOM(2)

    .byte    10                              # Abbreviation Code
    .byte    5                               # DW_TAG_formal_parameter
    .byte    0                               # DW_CHILDREN_no
    .byte    2                               # DW_AT_location
    .byte    23                              # DW_FORM_sec_offset
    .byte    3                               # DW_AT_name
    .byte    14                              # DW_FORM_strp
    .byte    58                              # DW_AT_decl_file
    .byte    11                              # DW_FORM_data1
    .byte    59                              # DW_AT_decl_line
    .byte    11                              # DW_FORM_data1
    .byte    73                              # DW_AT_type
    .byte    19                              # DW_FORM_ref4
    .byte    0                               # EOM(1)
    .byte    0                               # EOM(2)

    .byte    11                              # Abbreviation Code
    .byte    29                              # DW_TAG_inlined_subroutine
    .byte    1                               # DW_CHILDREN_yes
    .byte    49                              # DW_AT_abstract_origin
    .byte    19                              # DW_FORM_ref4
    .byte    17                              # DW_AT_low_pc
    .byte    1                               # DW_FORM_addr
    .byte    18                              # DW_AT_high_pc
    .byte    6                               # DW_FORM_data4
    .byte    88                              # DW_AT_call_file
    .byte    11                              # DW_FORM_data1
    .byte    89                              # DW_AT_call_line
    .byte    11                              # DW_FORM_data1
    .byte    87                              # DW_AT_call_column
    .byte    11                              # DW_FORM_data1
    .byte    0                               # EOM(1)
    .byte    0                               # EOM(2)

    .byte    12                              # Abbreviation Code
    .byte    15                              # DW_TAG_pointer_type
    .byte    0                               # DW_CHILDREN_no
    .byte    73                              # DW_AT_type
    .byte    19                              # DW_FORM_ref4
    .byte    0                               # EOM(1)
    .byte    0                               # EOM(2)
    .byte    0                               # EOM(3)

    .section    .debug_info,"",@progbits
.Ldi_cu_begin:
    .long    .Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
    .short    4                              # DWARF version number
    .long    .debug_abbrev                   # Offset Into Abbrev. Section
    .byte    8                               # Address Size (in bytes)
    .byte    1                               # Abbrev [1] DW_TAG_compile_unit
    .long    .Linfo_string_fname             #   DW_AT_name
    .long    .Lline_table_start0             #   DW_AT_stmt_list
    .quad    .Lcu_begin                      #   DW_AT_low_pc
    .long    .Lcu_end-.Lcu_begin             #   DW_AT_high_pc

# Debug info for |f| (abstract version with all parameters).

.Ldebug_info_f:
    .byte    5                               #   Abbrev [5] DW_TAG_subprogram
    .long    .Linfo_string_f                 #     DW_AT_name
    .byte    1                               #     DW_AT_decl_file
    .byte    4                               #     DW_AT_decl_line
                                             #     DW_AT_prototyped
                                             #     DW_AT_external
    .byte    1                               #     DW_AT_inline
.Ldebug_info_param1:
    .byte    6                               #     Abbrev [6] DW_TAG_formal_parameter
    .long    .Linfo_string_unused1           #       DW_AT_name
    .byte    1                               #       DW_AT_decl_file
    .byte    4                               #       DW_AT_decl_line
    .long    .Ldebug_info_void_ptr-.Ldi_cu_begin
                                             #       DW_AT_type
.Ldebug_info_param2:
    .byte    6                               #     Abbrev [6] DW_TAG_formal_parameter
    .long    .Linfo_string_used              #       DW_AT_name
    .byte    1                               #       DW_AT_decl_file
    .byte    4                               #       DW_AT_decl_line
    .long    .Ldebug_info_int-.Ldi_cu_begin  #       DW_AT_type
.Ldebug_info_param3:
    .byte    6                               #     Abbrev [6] DW_TAG_formal_parameter
    .long    .Linfo_string_unused2           #       DW_AT_name
    .byte    1                               #       DW_AT_decl_file
    .byte    4                               #       DW_AT_decl_line
    .long    .Ldebug_info_int-.Ldi_cu_begin  #       DW_AT_type
.Ldebug_info_param4:
    .byte    6                               #     Abbrev [6] DW_TAG_formal_parameter
    .long    .Linfo_string_partial           #       DW_AT_name
    .byte    1                               #       DW_AT_decl_file
    .byte    4                               #       DW_AT_decl_line
    .long    .Ldebug_info_int-.Ldi_cu_begin  #       DW_AT_type
.Ldebug_info_param5:
    .byte    6                               #     Abbrev [6] DW_TAG_formal_parameter
    .long    .Linfo_string_unused3           #       DW_AT_name
    .byte    1                               #       DW_AT_decl_file
    .byte    4                               #       DW_AT_decl_line
    .long    .Ldebug_info_int-.Ldi_cu_begin  #       DW_AT_type
    .byte    0                               #   End Of Children Mark (DW_TAG_subprogram)

# Debug info for |g| (abstract version with all parameters).

.Ldebug_info_g:
    .byte    5                               #   Abbrev [5] DW_TAG_subprogram
    .long    .Linfo_string_g                 #     DW_AT_name
    .byte    1                               #     DW_AT_decl_file
    .byte    4                               #     DW_AT_decl_line
                                             #     DW_AT_prototyped
                                             #     DW_AT_external
    .byte    1                               #     DW_AT_inline
.Ldebug_info_g_param1:
    .byte    6                               #     Abbrev [6] DW_TAG_formal_parameter
    .long    .Linfo_string_unused            #       DW_AT_name
    .byte    1                               #       DW_AT_decl_file
    .byte    10                              #       DW_AT_decl_line
    .long    .Ldebug_info_int-.Ldi_cu_begin
    .byte    0                               #   End Of Children Mark (DW_TAG_subprogram)

# Debug info for |main|.

    .byte    9                               #   Abbrev [9] DW_TAG_subprogram
    .quad    main                            #     DW_AT_low_pc
    .long    .Lmain_end-main                 #     DW_AT_high_pc
    .byte    1                               #     DW_AT_frame_base
    .byte    87
    .long    .Linfo_string_main              #     DW_AT_name
    .byte    1                               #     DW_AT_decl_file
    .byte    18                              #     DW_AT_decl_line
                                             #     DW_AT_prototyped
    .long    .Ldebug_info_int-.Ldi_cu_begin  #     DW_AT_type
                                             #     DW_AT_external

#   Debug info for concrete |f| inlined into |main|.

    .byte    11                              #     Abbrev [11] DW_TAG_inlined_subroutine
    .long    .Ldebug_info_f-.Ldi_cu_begin
                                             #       DW_AT_abstract_origin
    .quad    .Linlined_f                     #       DW_AT_low_pc
    .long    .Linlined_f_end-.Linlined_f     #       DW_AT_high_pc
    .byte    1                               #       DW_AT_call_file
    .byte    20                              #       DW_AT_call_line
    .byte    3                               #       DW_AT_call_column
    .byte    4                               #       Abbrev [4] DW_TAG_formal_parameter
    .long    .Ldebug_loc_used                #         DW_AT_location
    .long    .Ldebug_info_param2-.Ldi_cu_begin
                                             #         DW_AT_abstract_origin
    .byte    4                               #       Abbrev [4] DW_TAG_formal_parameter
    .long    .Ldebug_loc_partial             #         DW_AT_location
    .long    .Ldebug_info_param4-.Ldi_cu_begin
                                             #         DW_AT_abstract_origin
    .byte    0                               #     End Of Children Mark (DW_TAG_inlined_subroutine)

#   Debug info for concrete |g| inlined into |main|.

    .byte    11                              #     Abbrev [11] DW_TAG_inlined_subroutine
    .long    .Ldebug_info_g-.Ldi_cu_begin
                                             #       DW_AT_abstract_origin
    .quad    .Linlined_g                     #       DW_AT_low_pc
    .long    .Linlined_g_end-.Linlined_g     #       DW_AT_high_pc
    .byte    1                               #       DW_AT_call_file
    .byte    21                              #       DW_AT_call_line
    .byte    3                               #       DW_AT_call_column
    .byte    0                               #     End Of Children Mark (DW_TAG_inlined_subroutine)

    .byte    0                               #   End Of Children Mark (DW_TAG_subprogram)

# Debug info for |other|.

    .byte    9                               #   Abbrev [9] DW_TAG_subprogram
    .quad    other                           #     DW_AT_low_pc
    .long    .Lother_end-other               #     DW_AT_high_pc
    .byte    1                               #     DW_AT_frame_base
    .byte    87
    .long    .Linfo_string_other             #     DW_AT_name
    .byte    1                               #     DW_AT_decl_file
    .byte    15                              #     DW_AT_decl_line
                                             #     DW_AT_prototyped
    .long    .Ldebug_info_int-.Ldi_cu_begin  #     DW_AT_type
                                             #     DW_AT_external

#   Debug info for concrete |f| inlined into |other|.

    .byte    11                              #     Abbrev [11] DW_TAG_inlined_subroutine
    .long    .Ldebug_info_f-.Ldi_cu_begin
                                             #       DW_AT_abstract_origin
    .quad    .Linlined_f_in_other            #       DW_AT_low_pc
    .long    .Linlined_f_in_other_end-.Linlined_f_in_other
                                             #       DW_AT_high_pc
    .byte    1                               #       DW_AT_call_file
    .byte    16                              #       DW_AT_call_line
    .byte    3                               #       DW_AT_call_column
    .byte    4                               #       Abbrev [4] DW_TAG_formal_parameter
    .long    .Ldebug_loc_used_in_other       #         DW_AT_location
    .long    .Ldebug_info_param2-.Ldi_cu_begin
                                             #         DW_AT_abstract_origin
    .byte    4                               #       Abbrev [4] DW_TAG_formal_parameter
    .long    .Ldebug_loc_partial_in_other    #         DW_AT_location
    .long    .Ldebug_info_param4-.Ldi_cu_begin
                                             #         DW_AT_abstract_origin
    .byte    0                               #     End Of Children Mark (DW_TAG_inlined_subroutine)
    .byte    0                               #   End Of Children Mark (DW_TAG_subprogram)

.Ldebug_info_void_ptr:
    .byte    7                               #   Abbrev [7] DW_TAG_pointer_type
.Ldebug_info_int:
    .byte    8                               #   Abbrev [8] DW_TAG_base_type
    .long    .Linfo_string_int               #     DW_AT_name
    .byte    5                               #     DW_AT_encoding
    .byte    4                               #     DW_AT_byte_size

    .byte    0                               # End Of Children Mark (DW_TAG_compile_unit)
.Ldebug_info_end0:
    .section    .debug_str,"MS",@progbits,1
.Linfo_string_fname:
    .asciz    "unused-inlined-params.c"
.Linfo_string_f:
    .asciz    "f"
.Linfo_string_unused1:
    .asciz    "unused1"
.Linfo_string_used:
    .asciz    "used"
.Linfo_string_int:
    .asciz    "int"
.Linfo_string_unused2:
    .asciz    "unused2"
.Linfo_string_partial:
    .asciz    "partial"
.Linfo_string_unused3:
    .asciz    "unused3"
.Linfo_string_main:
    .asciz    "main"
.Linfo_string_g:
    .asciz    "g"
.Linfo_string_unused:
    .asciz    "unused"
.Linfo_string_other:
    .asciz    "other"
    .section    ".note.GNU-stack","",@progbits
    .addrsig
    .section    .debug_line,"",@progbits
.Lline_table_start0:
