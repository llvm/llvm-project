# REQUIRES: x86-registered-target

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: llvm-symbolizer --obj=%t.o 0x30 0x20 0x6c 0xf | FileCheck %s

# CHECK:      main
# CHECK-NEXT: llvm-symbolizer-bbsections-test.c:13
# CHECK:      g
# CHECK-NEXT: llvm-symbolizer-bbsections-test.c:5
# CHECK:      main
# CHECK-NEXT: llvm-symbolizer-bbsections-test.c:15
# CHECK:      main
# CHECK-NEXT: llvm-symbolizer-bbsections-test.c:18
# How to generate this file:
#  int f(int a) {
#    return a + 1;
#  }
#
#  int g(int a) {
#    return a + 2;
#  }
#
#  int h(int a) {
#    return a + 3;
#  }
#
#  // Use simple control flow to generate lots of basic block sections.
#  int main(int argc, char *argv[]) {
#    if (argc > 10)
#      return f(argc);
#    else if (argc > 8)
#      return g(argc);
#    else if (argc > 4)
#      return h(argc);
#    return 0;
#  }
#
#  $ clang -S -fbasic-block-sections=all llvm-symbolizer-bbsections-test.cc
#  Manually reororder the sections to place them in this order:
#   _Z1fi
#   main.__part.4
#   _Z1gi
#   main
#   _Z1hi
#   main.__part.5
#   (rest)
#  Strip the .section .text directives to have all the functions in the same
#  section.
#  This ensures the basic blocks are reordered non-contiguous exactly like
#  how a linker would do it.
	.text
	.file	"llvm-symbolizer-bbsections-test.c"
	#.section .text._Z1fi,"ax",@progbits
	.globl	_Z1fi                           # -- Begin function _Z1fi
	.p2align	4, 0x90
	.type	_Z1fi,@function
_Z1fi:                                  # @_Z1fi
.Lfunc_begin0:
	.file	1 "Examples" "llvm-symbolizer-bbsections-test.c"
	.loc	1 1 0                           # llvm-symbolizer-bbsections-test.c:1:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movl	%edi, -4(%rbp)
.Ltmp0:
	.loc	1 2 10 prologue_end             # llvm-symbolizer-bbsections-test.c:2:10
	movl	-4(%rbp), %eax
	.loc	1 2 12 is_stmt 0                # llvm-symbolizer-bbsections-test.c:2:12
	addl	$1, %eax
	.loc	1 2 3                           # llvm-symbolizer-bbsections-test.c:2:3
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp1:
	.cfi_endproc
.Lfunc_end0:
	.size	_Z1fi, .Lfunc_end0-_Z1fi
                                        # -- End function
	#.section text.main,"ax",@progbits,unique,4
main.__part.4:
	.cfi_startproc
	.cfi_def_cfa %rbp, 16
	.cfi_offset %rbp, -16
	.loc	1 18 17 is_stmt 1               # llvm-symbolizer-bbsections-test.c:18:17
	cmpl	$4, -8(%rbp)
.Ltmp9:
	.loc	1 18 12 is_stmt 0               # llvm-symbolizer-bbsections-test.c:18:12
	jle	main.__part.6
	jmp	main.__part.5
.LBB_END3_4:
	.size	main.__part.4, .LBB_END3_4-main.__part.4
	.cfi_endproc

	#.section .text._Z1gi,"ax",@progbits
	.globl	_Z1gi                           # -- Begin function _Z1gi
	.p2align	4, 0x90
	.type	_Z1gi,@function
_Z1gi:                                  # @_Z1gi
.Lfunc_begin1:
	.loc	1 5 0 is_stmt 1                 # llvm-symbolizer-bbsections-test.c:5:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movl	%edi, -4(%rbp)
.Ltmp2:
	.loc	1 6 10 prologue_end             # llvm-symbolizer-bbsections-test.c:6:10
	movl	-4(%rbp), %eax
	.loc	1 6 12 is_stmt 0                # llvm-symbolizer-bbsections-test.c:6:12
	addl	$2, %eax
	.loc	1 6 3                           # llvm-symbolizer-bbsections-test.c:6:3
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp3:
	.cfi_endproc
.Lfunc_end1:
	.size	_Z1gi, .Lfunc_end1-_Z1gi
                                        # -- End function
	#.section .text.main,"ax",@progbits
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
.Lfunc_begin3:
	.loc	1 13 0 is_stmt 1                # llvm-symbolizer-bbsections-test.c:13:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movl	$0, -4(%rbp)
	movl	%edi, -8(%rbp)
	movq	%rsi, -16(%rbp)
.Ltmp6:
	.loc	1 14 12 prologue_end            # llvm-symbolizer-bbsections-test.c:14:12
	cmpl	$10, -8(%rbp)
.Ltmp7:
	.loc	1 14 7 is_stmt 0                # llvm-symbolizer-bbsections-test.c:14:7
	jle	main.__part.2
	jmp	main.__part.1
	.cfi_endproc
        #.section .text._Z1hi,"ax",@progbits
	.globl	_Z1hi                           # -- Begin function _Z1hi
	.p2align	4, 0x90
	.type	_Z1hi,@function
_Z1hi:                                  # @_Z1hi
.Lfunc_begin2:
	.loc	1 9 0 is_stmt 1                 # llvm-symbolizer-bbsections-test.c:9:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movl	%edi, -4(%rbp)
.Ltmp4:
	.loc	1 10 10 prologue_end            # llvm-symbolizer-bbsections-test.c:10:10
	movl	-4(%rbp), %eax
	.loc	1 10 12 is_stmt 0               # llvm-symbolizer-bbsections-test.c:10:12
	addl	$3, %eax
	.loc	1 10 3                          # llvm-symbolizer-bbsections-test.c:10:3
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp5:
	.cfi_endproc
.Lfunc_end2:
	.size	_Z1hi, .Lfunc_end2-_Z1hi
                                        # -- End function
	#.section .text.main,"ax",@progbits,unique,5
main.__part.5:
	.cfi_startproc
	.cfi_def_cfa %rbp, 16
	.cfi_offset %rbp, -16
	.loc	1 19 14 is_stmt 1               # llvm-symbolizer-bbsections-test.c:19:14
	movl	-8(%rbp), %edi
	.loc	1 19 12 is_stmt 0               # llvm-symbolizer-bbsections-test.c:19:12
	callq	_Z1hi
	.loc	1 19 5                          # llvm-symbolizer-bbsections-test.c:19:5
	movl	%eax, -4(%rbp)
	jmp	main.__part.9
.Ltmp10:
.LBB_END3_5:
	.size	main.__part.5, .LBB_END3_5-main.__part.5
	.cfi_endproc
	#.section .text.main,"ax",@progbits,unique,1
main.__part.1:
	.cfi_startproc
	.cfi_def_cfa %rbp, 16
	.cfi_offset %rbp, -16
	.loc	1 15 14 is_stmt 1               # llvm-symbolizer-bbsections-test.c:15:14
	movl	-8(%rbp), %edi
	.loc	1 15 12 is_stmt 0               # llvm-symbolizer-bbsections-test.c:15:12
	callq	_Z1fi
	.loc	1 15 5                          # llvm-symbolizer-bbsections-test.c:15:5
	movl	%eax, -4(%rbp)
	jmp	main.__part.9
.LBB_END3_1:
	.size	main.__part.1, .LBB_END3_1-main.__part.1
	.cfi_endproc
	#.section .text.main,"ax",@progbits,unique,2
main.__part.2:
	.cfi_startproc
	.cfi_def_cfa %rbp, 16
	.cfi_offset %rbp, -16
	.loc	1 16 17 is_stmt 1               # llvm-symbolizer-bbsections-test.c:16:17
	cmpl	$8, -8(%rbp)
.Ltmp8:
	.loc	1 16 12 is_stmt 0               # llvm-symbolizer-bbsections-test.c:16:12
	jle	main.__part.4
	jmp	main.__part.3
.LBB_END3_2:
	.size	main.__part.2, .LBB_END3_2-main.__part.2
	.cfi_endproc
	#.section .text.main,"ax",@progbits,unique,3
main.__part.3:
	.cfi_startproc
	.cfi_def_cfa %rbp, 16
	.cfi_offset %rbp, -16
	.loc	1 17 14 is_stmt 1               # llvm-symbolizer-bbsections-test.c:17:14
	movl	-8(%rbp), %edi
	.loc	1 17 12 is_stmt 0               # llvm-symbolizer-bbsections-test.c:17:12
	callq	_Z1gi
	.loc	1 17 5                          # llvm-symbolizer-bbsections-test.c:17:5
	movl	%eax, -4(%rbp)
	jmp	main.__part.9
.LBB_END3_3:
	.size	main.__part.3, .LBB_END3_3-main.__part.3
	.cfi_endproc
	#.section .text.main,"ax",@progbits,unique,6
main.__part.6:
	.cfi_startproc
	.cfi_def_cfa %rbp, 16
	.cfi_offset %rbp, -16
	.loc	1 0 5                           # llvm-symbolizer-bbsections-test.c:0:5
	jmp	main.__part.7
	jmp	main.__part.7
.LBB_END3_6:
	.size	main.__part.6, .LBB_END3_6-main.__part.6
	.cfi_endproc
	#.section .text.main,"ax",@progbits,unique,7
main.__part.7:
	.cfi_startproc
	.cfi_def_cfa %rbp, 16
	.cfi_offset %rbp, -16
	jmp	main.__part.8
	jmp	main.__part.8
.LBB_END3_7:
	.size	main.__part.7, .LBB_END3_7-main.__part.7
	.cfi_endproc
	#.section .text.main,"ax",@progbits,unique,8
main.__part.8:
	.cfi_startproc
	.cfi_def_cfa %rbp, 16
	.cfi_offset %rbp, -16
	.loc	1 20 3 is_stmt 1                # llvm-symbolizer-bbsections-test.c:20:3
	movl	$0, -4(%rbp)
	jmp	main.__part.9
.LBB_END3_8:
	.size	main.__part.8, .LBB_END3_8-main.__part.8
	.cfi_endproc
	#.section .text.main,"ax",@progbits,unique,9
main.__part.9:
	.cfi_startproc
	.cfi_def_cfa %rbp, 16
	.cfi_offset %rbp, -16
	.loc	1 21 1                          # llvm-symbolizer-bbsections-test.c:21:1
	movl	-4(%rbp), %eax
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp11:
.LBB_END3_9:
	.size	main.__part.9, .LBB_END3_9-main.__part.9
	.cfi_endproc
	#.section .text.main,"ax",@progbits
.Lfunc_end3:
	.size	main, .Lfunc_end3-main
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
	.byte	85                              # DW_AT_ranges
	.byte	23                              # DW_FORM_sec_offset
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	2                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.byte	110                             # DW_AT_linkage_name
	.byte	14                              # DW_FORM_strp
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	3                               # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
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
	.byte	4                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	85                              # DW_AT_ranges
	.byte	23                              # DW_FORM_sec_offset
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
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
	.byte	6                               # Abbreviation Code
	.byte	15                              # DW_TAG_pointer_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
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
	.byte	1                               # Abbrev [1] 0xb:0xea DW_TAG_compile_unit
	.long	.Linfo_string0                  # DW_AT_producer
	.short	33                              # DW_AT_language
	.long	.Linfo_string1                  # DW_AT_name
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.long	.Linfo_string2                  # DW_AT_comp_dir
	.quad	0                               # DW_AT_low_pc
	.long	.Ldebug_ranges1                 # DW_AT_ranges
	.byte	2                               # Abbrev [2] 0x2a:0x2c DW_TAG_subprogram
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string3                  # DW_AT_linkage_name
	.long	.Linfo_string4                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.long	220                             # DW_AT_type
                                        # DW_AT_external
	.byte	3                               # Abbrev [3] 0x47:0xe DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	124
	.long	.Linfo_string11                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.long	220                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	2                               # Abbrev [2] 0x56:0x2c DW_TAG_subprogram
	.quad	.Lfunc_begin1                   # DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin1       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string6                  # DW_AT_linkage_name
	.long	.Linfo_string7                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.long	220                             # DW_AT_type
                                        # DW_AT_external
	.byte	3                               # Abbrev [3] 0x73:0xe DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	124
	.long	.Linfo_string11                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.long	220                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	2                               # Abbrev [2] 0x82:0x2c DW_TAG_subprogram
	.quad	.Lfunc_begin2                   # DW_AT_low_pc
	.long	.Lfunc_end2-.Lfunc_begin2       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string8                  # DW_AT_linkage_name
	.long	.Linfo_string9                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	9                               # DW_AT_decl_line
	.long	220                             # DW_AT_type
                                        # DW_AT_external
	.byte	3                               # Abbrev [3] 0x9f:0xe DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	124
	.long	.Linfo_string11                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	9                               # DW_AT_decl_line
	.long	220                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	4                               # Abbrev [4] 0xae:0x2e DW_TAG_subprogram
	.long	.Ldebug_ranges0                 # DW_AT_ranges
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string10                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	13                              # DW_AT_decl_line
	.long	220                             # DW_AT_type
                                        # DW_AT_external
	.byte	3                               # Abbrev [3] 0xbf:0xe DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string12                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	13                              # DW_AT_decl_line
	.long	220                             # DW_AT_type
	.byte	3                               # Abbrev [3] 0xcd:0xe DW_TAG_formal_parameter
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string13                 # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	13                              # DW_AT_decl_line
	.long	227                             # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	5                               # Abbrev [5] 0xdc:0x7 DW_TAG_base_type
	.long	.Linfo_string5                  # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	6                               # Abbrev [6] 0xe3:0x5 DW_TAG_pointer_type
	.long	232                             # DW_AT_type
	.byte	6                               # Abbrev [6] 0xe8:0x5 DW_TAG_pointer_type
	.long	237                             # DW_AT_type
	.byte	5                               # Abbrev [5] 0xed:0x7 DW_TAG_base_type
	.long	.Linfo_string14                 # DW_AT_name
	.byte	6                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_ranges,"",@progbits
.Ldebug_ranges0:
	.quad	main.__part.1
	.quad	.LBB_END3_1
	.quad	main.__part.2
	.quad	.LBB_END3_2
	.quad	main.__part.3
	.quad	.LBB_END3_3
	.quad	main.__part.4
	.quad	.LBB_END3_4
	.quad	main.__part.5
	.quad	.LBB_END3_5
	.quad	main.__part.6
	.quad	.LBB_END3_6
	.quad	main.__part.7
	.quad	.LBB_END3_7
	.quad	main.__part.8
	.quad	.LBB_END3_8
	.quad	main.__part.9
	.quad	.LBB_END3_9
	.quad	.Lfunc_begin3
	.quad	.Lfunc_end3
	.quad	0
	.quad	0
.Ldebug_ranges1:
	.quad	.Lfunc_begin0
	.quad	.Lfunc_end0
	.quad	.Lfunc_begin1
	.quad	.Lfunc_end1
	.quad	.Lfunc_begin2
	.quad	.Lfunc_end2
	.quad	main.__part.1
	.quad	.LBB_END3_1
	.quad	main.__part.2
	.quad	.LBB_END3_2
	.quad	main.__part.3
	.quad	.LBB_END3_3
	.quad	main.__part.4
	.quad	.LBB_END3_4
	.quad	main.__part.5
	.quad	.LBB_END3_5
	.quad	main.__part.6
	.quad	.LBB_END3_6
	.quad	main.__part.7
	.quad	.LBB_END3_7
	.quad	main.__part.8
	.quad	.LBB_END3_8
	.quad	main.__part.9
	.quad	.LBB_END3_9
	.quad	.Lfunc_begin3
	.quad	.Lfunc_end3
	.quad	0
	.quad	0
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 12.0.0 (git@github.com:llvm/llvm-project.git bfa6ca07a8cda0ab889b7fee0b914907ce594e11)" # string offset=0
.Linfo_string1:
	.asciz	"llvm-symbolizer-bbsections-test.c" # string offset=101
.Linfo_string2:
	.asciz	"Examples" # string offset=135
.Linfo_string3:
	.asciz	"_Z1fi"                         # string offset=182
.Linfo_string4:
	.asciz	"f"                             # string offset=188
.Linfo_string5:
	.asciz	"int"                           # string offset=190
.Linfo_string6:
	.asciz	"_Z1gi"                         # string offset=194
.Linfo_string7:
	.asciz	"g"                             # string offset=200
.Linfo_string8:
	.asciz	"_Z1hi"                         # string offset=202
.Linfo_string9:
	.asciz	"h"                             # string offset=208
.Linfo_string10:
	.asciz	"main"                          # string offset=210
.Linfo_string11:
	.asciz	"a"                             # string offset=215
.Linfo_string12:
	.asciz	"argc"                          # string offset=217
.Linfo_string13:
	.asciz	"argv"                          # string offset=222
.Linfo_string14:
	.asciz	"char"                          # string offset=227
	.ident	"clang version 12.0.0 (git@github.com:llvm/llvm-project.git bfa6ca07a8cda0ab889b7fee0b914907ce594e11)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym _Z1fi
	.addrsig_sym _Z1gi
	.addrsig_sym _Z1hi
	.section	.debug_line,"",@progbits
.Lline_table_start0:
