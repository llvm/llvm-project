# Note: This file is compiled from the following code, for 
# 		the purpose of creating an overflowed dwo section.
#       After being compiled from source, section `.debug_info.dwo`
#       is changed to have length (2^32 - 30) Bytes, and added 
#       padding with `.fill` directive, so it is likely to 
#       overflow when packed with other files.
# 
# clang -g -S -gsplit-dwarf -gdwarf-4 hello.c
#
# #include <stdio.h>
# void hello() {
#     printf("hello\n");
# }

	.text
	.file	"hello.c"
	.globl	hello                           # -- Begin function hello
	.p2align	4, 0x90
	.type	hello,@function
hello:                                  # @hello
.Lfunc_begin0:
	.file	1 "/xxxxxx/xxxx/xxxxxxxxxx/xxxxxxxx/hello" "hello.c"
	.loc	1 3 0                           # hello.c:3:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp0:
	.loc	1 4 5 prologue_end              # hello.c:4:5
	movabsq	$.L.str, %rdi
	movb	$0, %al
	callq	printf
	.loc	1 5 1                           # hello.c:5:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp1:
.Lfunc_end0:
	.size	hello, .Lfunc_end0-hello
	.cfi_endproc
                                        # -- End function
	.type	.L.str,@object                  # @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	"hello\n"
	.size	.L.str, 7

	.section	.debug_abbrev,"",@progbits
	.byte	1                               # Abbreviation Code
	.byte	17                              # DW_TAG_compile_unit
	.byte	0                               # DW_CHILDREN_no
	.byte	16                              # DW_AT_stmt_list
	.byte	23                              # DW_FORM_sec_offset
	.byte	27                              # DW_AT_comp_dir
	.byte	14                              # DW_FORM_strp
	.ascii	"\264B"                         # DW_AT_GNU_pubnames
	.byte	25                              # DW_FORM_flag_present
	.ascii	"\260B"                         # DW_AT_GNU_dwo_name
	.byte	14                              # DW_FORM_strp
	.ascii	"\261B"                         # DW_AT_GNU_dwo_id
	.byte	7                               # DW_FORM_data8
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.ascii	"\263B"                         # DW_AT_GNU_addr_base
	.byte	23                              # DW_FORM_sec_offset
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	0                               # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	# .long	4294967295   # 2^32 - 1    #44                              # Length of Unit
	.long	44                              # Length of Unit
	.short	4                               # DWARF version number
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.byte	1                               # Abbrev [1] 0xb:0x25 DW_TAG_compile_unit
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.long	.Lskel_string0                  # DW_AT_comp_dir
                                        # DW_AT_GNU_pubnames
	.long	.Lskel_string1                  # DW_AT_GNU_dwo_name
	.quad	-94954012350180462              # DW_AT_GNU_dwo_id
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.long	.Laddr_table_base0              # DW_AT_GNU_addr_base
	# .fill   4294967251   # = 2^32 - 1 - 44
	.section	.debug_str,"MS",@progbits,1
.Lskel_string0:
	.asciz	"/xxxxxx/xxxx/xxxxxxxxxx/xxxxxxxx/hello" # string offset=0
.Lskel_string1:
	.asciz	"hello.dwo"                     # string offset=39
	.section	.debug_str.dwo,"eMS",@progbits,1
.Linfo_string0:
	.asciz	"hello"                         # string offset=0
.Linfo_string1:
	.asciz	"clang version 11.1.0 (https://github.com/llvm/llvm-project.git 173544ee3d09cdce8665f2097f677c31e1f1a9a1)" # string offset=6
.Linfo_string2:
	.asciz	"hello.c"                       # string offset=111
.Linfo_string3:
	.asciz	"hello.dwo"                     # string offset=119
	.section	.debug_str_offsets.dwo,"e",@progbits
	.long	0
	.long	6
	.long	111
	.long	119
	.section	.debug_info.dwo,"e",@progbits
	.long	4294967266   # 2^32 - 30        #33                              # Length of Unit
	.short	4                               # DWARF version number
	.long	0                               # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.byte	1                               # Abbrev [1] 0xb:0x1a DW_TAG_compile_unit
	.byte	1                               # DW_AT_producer
	.short	12                              # DW_AT_language
	.byte	2                               # DW_AT_name
	.byte	3                               # DW_AT_GNU_dwo_name
	.quad	-94954012350180462              # DW_AT_GNU_dwo_id
	.byte	2                               # Abbrev [2] 0x19:0xb DW_TAG_subprogram
	.byte	0                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	0                               # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
                                        # DW_AT_external
	.byte	0                               # End Of Children Mark
	.fill   4294967233  # 2^32 - 30 - 33
	.section	.debug_abbrev.dwo,"e",@progbits
	.byte	1                               # Abbreviation Code
	.byte	17                              # DW_TAG_compile_unit
	.byte	1                               # DW_CHILDREN_yes
	.byte	37                              # DW_AT_producer
	.ascii	"\202>"                         # DW_FORM_GNU_str_index
	.byte	19                              # DW_AT_language
	.byte	5                               # DW_FORM_data2
	.byte	3                               # DW_AT_name
	.ascii	"\202>"                         # DW_FORM_GNU_str_index
	.ascii	"\260B"                         # DW_AT_GNU_dwo_name
	.ascii	"\202>"                         # DW_FORM_GNU_str_index
	.ascii	"\261B"                         # DW_AT_GNU_dwo_id
	.byte	7                               # DW_FORM_data8
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	2                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	0                               # DW_CHILDREN_no
	.byte	17                              # DW_AT_low_pc
	.ascii	"\201>"                         # DW_FORM_GNU_addr_index
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.byte	3                               # DW_AT_name
	.ascii	"\202>"                         # DW_FORM_GNU_str_index
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	0                               # EOM(3)
	.section	.debug_addr,"",@progbits
.Laddr_table_base0:
	.quad	.Lfunc_begin0
	.section	.debug_gnu_pubnames,"",@progbits
	.long	.LpubNames_end0-.LpubNames_begin0 # Length of Public Names Info
.LpubNames_begin0:
	.short	2                               # DWARF Version
	.long	.Lcu_begin0                     # Offset of Compilation Unit Info
	.long	48                              # Compilation Unit Length
	.long	25                              # DIE offset
	.byte	48                              # Attributes: FUNCTION, EXTERNAL
	.asciz	"hello"                         # External Name
	.long	0                               # End Mark
.LpubNames_end0:
	.section	.debug_gnu_pubtypes,"",@progbits
	.long	.LpubTypes_end0-.LpubTypes_begin0 # Length of Public Types Info
.LpubTypes_begin0:
	.short	2                               # DWARF Version
	.long	.Lcu_begin0                     # Offset of Compilation Unit Info
	.long	48                              # Compilation Unit Length
	.long	0                               # End Mark
.LpubTypes_end0:
	.ident	"clang version 11.1.0 (https://github.com/llvm/llvm-project.git 173544ee3d09cdce8665f2097f677c31e1f1a9a1)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym printf
	.section	.debug_line,"",@progbits
.Lline_table_start0: