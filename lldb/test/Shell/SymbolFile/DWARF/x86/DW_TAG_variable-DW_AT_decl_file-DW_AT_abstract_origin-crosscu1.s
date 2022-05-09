# Check that DW_AT_decl_file of DW_AT_variable which is inherited by
# DW_AT_abstract_origin from a different DW_TAG_compile_unit is using the
# DW_TAG_compile_unit->DW_AT_stmt_list where the DW_AT_decl_file is located (and
# not where the DW_AT_abstract_origin is located).
# DW_TAG_variable in CU 1 is using DW_AT_decl_file 3.
# CU 1 has files:
# file_names[  1]: name: "inlinevarother.h"
# file_names[  2]: name: "inlinevar1.c"
# file_names[  3]: name: "inlinevar.h"
# CU 2 has files:
# file_names[  1]: name: "inlinevar2.c"
# file_names[  2]: name: "inlinevar.h"

# UNSUPPORTED: system-darwin, system-windows
# REQUIRES: target-x86_64

# RUN: %clang_host -gdwarf-4 -o %t %s \
# RUN:   %S/Inputs/DW_TAG_variable-DW_AT_decl_file-DW_AT_abstract_origin-crosscu2.s

# RUN: %lldb %t \
# RUN:   -o 'b other' -o r -o disas -o 'frame variable --show-declaration' \
# RUN:   -o exit | FileCheck %s

# CHECK: inlinevar.h:2: (int) var = {{.*}}
# Unfixed LLDB did show only: (int) var = {{.*}}

	.text
	.file	"inlinevar1.c"
	.file	1 "" "./inlinevarother.h"
	.globl	main                            # -- Begin function main
	.type	main,@function
main:                                   # @main
.Lfunc_begin1:
	.file	2 "" "inlinevar1.c"
	.loc	2 4 0                           # inlinevar1.c:4:0
.Ltmp2:
	.file	3 "" "./inlinevar.h"
	.loc	3 2 16 prologue_end             # ./inlinevar.h:2:16
	movl	$42, %eax
	pushq	%rax
	.loc	3 3 10                          # ./inlinevar.h:3:10
.Ltmp3:
	.loc	2 5 20                          # inlinevar1.c:5:20
	callq	other
	popq	%rcx
	.loc	2 5 19                          # inlinevar1.c:5:19
	addl	%ecx, %eax
	.loc	2 5 3                           # inlinevar1.c:5:3
	retq
.Ltmp4:
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
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
	.byte	3                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	32                              # DW_AT_inline
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	4                               # Abbreviation Code
	.byte	52                              # DW_TAG_variable
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	5                               # Abbreviation Code
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
	.byte	0                               # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	4                               # DWARF version number
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.byte	1                               # Abbrev [1] DW_TAG_compile_unit
	.long	.Linfo_string0                  # DW_AT_producer
	.short	0xc                             # DW_AT_language
	.long	.Linfo_string1                  # DW_AT_name
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.long	.Linfo_string2                  # DW_AT_comp_dir
	.quad	.Lfunc_begin1                   # DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin1       # DW_AT_high_pc
	.globl	debuginfo_func_inlined
debuginfo_func_inlined:
.Lfunc_inlined:
	.byte	3                               # Abbrev [3] DW_TAG_subprogram
	.long	.Linfo_string4                  # DW_AT_name
	.byte	3                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.long	.Ltype_int-.Lcu_begin0          # DW_AT_type
	.byte	1                               # DW_AT_inline
	.globl	debuginfo_var_var
debuginfo_var_var:
.Lvar_var:
	.byte	4                               # Abbrev [4] DW_TAG_variable
	.long	.Linfo_string6                  # DW_AT_name
	.byte	3                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.long	.Ltype_int-.Lcu_begin0          # DW_AT_type
	.byte	0                               # End Of Children Mark
.Ltype_int:
	.byte	5                               # Abbrev [5] DW_TAG_base_type
	.long	.Linfo_string5                  # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 11.0.0 + hand coding"
.Linfo_string1:
	.asciz	"inlinevar1.c"
.Linfo_string2:
	.asciz	""
.Linfo_string4:
	.asciz	"inlined"
.Linfo_string5:
	.asciz	"int"
.Linfo_string6:
	.asciz	"var"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym other
	.section	.debug_line,"",@progbits
.Lline_table_start0:
