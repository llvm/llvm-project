# Check that DW_AT_decl_file inherited by DW_AT_specification from a different
# DW_TAG_compile_unit is using the DW_TAG_compile_unit->DW_AT_stmt_list where the
# DW_AT_decl_file is located (and not where the DW_AT_specification is located).

# RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux %s -o %t
# RUN: %lldb %t \
# RUN:   -o "image lookup -v -n main" \
# RUN:   -o exit | FileCheck %s

# CHECK: FuncType: id = {0x0000002a}, byte-size = 0, decl = filename1:1, compiler_type = "int (void)"

	.text
	.globl	main                            # -- Begin function main
	.type	main,@function
main:                                           # @main
.Lfunc_begin0:
        .byte 0
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
                                                # -- End function
	.section	.debug_abbrev,"",@progbits
	.byte	1                               # Abbreviation Code
	.byte	17                              # DW_TAG_compile_unit
	.byte	1                               # DW_CHILDREN_yes
	.byte	37                              # DW_AT_producer
	.byte	14                              # DW_FORM_strp
	.byte	19                              # DW_AT_language
	.byte	5                               # DW_FORM_data2
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	16                              # DW_AT_stmt_list
	.byte	23                              # DW_FORM_sec_offset
	.byte	27                              # DW_AT_comp_dir
	.byte	14                              # DW_FORM_strp
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	2                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.byte	71                              # DW_AT_specification
	.byte	16                              # DW_FORM_ref_addr
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	3                               # Abbreviation Code
	.byte	36                              # DW_TAG_base_type
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	62                              # DW_AT_encoding
	.byte	11                              # DW_FORM_data1
	.byte	11                              # DW_AT_byte_size
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	4                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	0                               # DW_CHILDREN_no
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	39                              # DW_AT_prototyped
	.byte	25                              # DW_FORM_flag_present
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	60                              # DW_AT_declaration
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	0                               # EOM(3)

	.section	.debug_info,"",@progbits
.Ldebug_cu0:

	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	4                               # DWARF version number
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.byte	1                               # Abbrev [1] 0xb:0x40 DW_TAG_compile_unit
	.long	.Linfo_string_producer          # DW_AT_producer
	.short	12                              # DW_AT_language
	.long	.Linfo_string_source            # DW_AT_name
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.long	.Linfo_string_directory         # DW_AT_comp_dir
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	2                               # Abbrev [2] 0x2a:0x19 DW_TAG_subprogram
	.long	.Linfo_string_main              # DW_AT_name
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Ldebug_info1_main - .Ldebug_cu0 # DW_AT_specification
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:

.Ldebug_info1:
	.long	.Ldebug_info_end1-.Ldebug_info_start1 # Length of Unit
.Ldebug_info_start1:
	.short	4                               # DWARF version number
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.byte	1                               # Abbrev [1] 0xb:0x40 DW_TAG_compile_unit
	.long	.Linfo_string_producer          # DW_AT_producer
	.short	12                              # DW_AT_language
	.long	.Linfo_string_source            # DW_AT_name
	.long	.Lline_table_start1             # DW_AT_stmt_list
	.long	.Linfo_string_directory         # DW_AT_comp_dir
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
.Ldebug_info1_main:
	.byte	4                               # Abbrev [2] 0x2a:0x19 DW_TAG_subprogram
	.byte	1                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
                                                # DW_AT_prototyped
	.long	.Ldebug_info1_int - .Ldebug_info1 # DW_AT_type
                                                # DW_AT_external
.Ldebug_info1_int:
	.byte	3                               # Abbrev [3] 0x43:0x7 DW_TAG_base_type
	.long	.Linfo_string_int               # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	0                               # End Of Children Mark
.Ldebug_info_end1:

	.section	.debug_str,"MS",@progbits,1
.Linfo_string_producer:
	.asciz	"clang version 12.0.0" # string offset=0
.Linfo_string_source:
	.asciz	"source.c" # string offset=130
.Linfo_string_directory:
	.asciz	"/directory" # string offset=196
.Linfo_string_main:
	.asciz	"main"
.Linfo_string_int:
	.asciz	"int"
	.ident	"clang version 12.0.0"
	.section	".note.GNU-stack","",@progbits
	.addrsig

	.section	.debug_line,"",@progbits

.Lline_table_start0:
        .long .Lline_table_end0 - 1f  # length from next byte
1:
        .short 4                      # version
        .long .Lline_table_end0 - 2f  # prologue length from next byte
2:
        .byte 1                       # minimum instruction length
        .byte 1                       # maximum ops per instruction
        .byte 1                       # initial value of is_stmt
        .byte -5                      # line base
        .byte 14                      # line range
        .byte 1                       # opcode base
                                      # no standard opcodes
        .asciz "directory0"
        .byte 0                       # last directory
        .asciz "filename0"
        .uleb128 1                    # directory entry
        .uleb128 0                    # time
        .uleb128 0                    # file length
        .byte 0                       # last filename
.Lline_table_end0:

.Lline_table_start1:
        .long .Lline_table_end1 - 1f  # length from next byte
1:
        .short 4                      # version
        .long .Lline_table_end1 - 2f  # prologue length from next byte
2:
        .byte 1                       # minimum instruction length
        .byte 1                       # maximum ops per instruction
        .byte 1                       # initial value of is_stmt
        .byte -5                      # line base
        .byte 14                      # line range
        .byte 1                       # opcode base
                                      # no standard opcodes
        .asciz "directory1"
        .byte 0                       # last directory
        .asciz "filename1"
        .uleb128 1                    # directory entry
        .uleb128 0                    # time
        .uleb128 0                    # file length
        .byte 0                       # last filename
.Lline_table_end1:
