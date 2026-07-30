# RUN: llvm-mc --dwarf-version=5 < %s -filetype obj -triple x86_64-pc-linux -o - | \ 
# RUN:   llvm-dwarfdump - | FileCheck %s

# CHECK: DW_AT_low_pc (DW_OP_const4u 0x0)

	.text
	.file	"test.cpp"
	.globl	_Z2f1v                  # -- Begin function _Z2f1v
	.p2align	4, 0x90
	.type	_Z2f1v,@function
_Z2f1v:                                 # @_Z2f1v
.Lfunc_begin0:
	.file	0 "/usr/local/google/home/blaikie/dev/scratch" "test.cpp" md5 0x74f7c574cd1ba04403967d02e757afeb
	.loc	0 1 0                   # test.cpp:1:0
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
.Ltmp0:
	.loc	0 2 1 prologue_end      # test.cpp:2:1
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp1:
.Lfunc_end0:
	.size	_Z2f1v, .Lfunc_end0-_Z2f1v
	.cfi_endproc
                                        # -- End function
	.section	.debug_str_offsets,"",@progbits
	.long	24
	.short	5
	.short	0
.Lstr_offsets_base0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 10.0.0 (git@github.com:llvm/llvm-project.git bf71564a526196f94cdde813063c8b1ff665fde7)" # string offset=0
.Linfo_string1:
	.asciz	"test.cpp"              # string offset=101
.Linfo_string2:
	.asciz	"/usr/local/google/home/blaikie/dev/scratch" # string offset=110
.Linfo_string3:
	.asciz	"_Z2f1v"                # string offset=153
.Linfo_string4:
	.asciz	"f1"                    # string offset=160
	.section	.debug_str_offsets,"",@progbits
	.long	.Linfo_string0
	.long	.Linfo_string1
	.long	.Linfo_string2
	.long	.Linfo_string3
	.long	.Linfo_string4
	.section	.debug_abbrev,"",@progbits
	.byte	1                       # Abbreviation Code
	.byte	17                      # DW_TAG_compile_unit
	.byte	0                       # DW_CHILDREN_no
	.byte	37                      # DW_AT_producer
	.byte	37                      # DW_FORM_strx1
	.byte	19                      # DW_AT_language
	.byte	5                       # DW_FORM_data2
	.byte	3                       # DW_AT_name
	.byte	37                      # DW_FORM_strx1
	.byte	114                     # DW_AT_str_offsets_base
	.byte	23                      # DW_FORM_sec_offset
	.byte	16                      # DW_AT_stmt_list
	.byte	23                      # DW_FORM_sec_offset
	.byte	27                      # DW_AT_comp_dir
	.byte	37                      # DW_FORM_strx1
	.byte	17                      # DW_AT_low_pc
	.byte	24                      # DW_FORM_exprloc
	.byte	18                      # DW_AT_high_pc
	.byte	6                       # DW_FORM_data4
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	0                       # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	5                       # DWARF version number
	.byte	1                       # DWARF Unit Type
	.byte	8                       # Address Size (in bytes)
	.long	.debug_abbrev           # Offset Into Abbrev. Section
	.byte	1                       # Abbrev [1] 0xc:0x38 DW_TAG_compile_unit
	.byte	0                       # DW_AT_producer
	.short	33                      # DW_AT_language
	.byte	1                       # DW_AT_name
	.long	.Lstr_offsets_base0     # DW_AT_str_offsets_base
	.long	.Lline_table_start0     # DW_AT_stmt_list
	.byte	2                       # DW_AT_comp_dir
	.byte	5                       # DW_AT_low_pc
	.byte	12
	.long	.Lfunc_begin0
	.long	.Lfunc_end0-.Lfunc_begin0 # DW_AT_high_pc
.Ldebug_info_end0:
	.ident	"clang version 10.0.0 (git@github.com:llvm/llvm-project.git bf71564a526196f94cdde813063c8b1ff665fde7)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.section	.debug_line,"",@progbits
.Lline_table_start0:
