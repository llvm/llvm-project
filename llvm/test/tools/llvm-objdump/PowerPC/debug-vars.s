## Check that the --debug-vars option works for simple register locations, when
## using DWARF4 debug info, with multiple functions in one section.

## Generated with this compile command and source code:
## clang --target=powerpc64-unknown-linux -c debug.c -O1 -S -o -

## The unicode characters in this test cause test failures on Windows.
# UNSUPPORTED: system-windows

## int foo(int a, int b, int c) {
##   int x = a + b;
##   int y = x + c;
##   return y;
## }
##
## int bar(int a) {
##   a++;
##   return a;
## }

# RUN: llvm-mc -triple powerpc64-unknown-linux < %s -filetype=obj | \
# RUN:     llvm-objdump - -d --debug-vars --no-show-raw-insn | \
# RUN:     FileCheck %s

# CHECK: Disassembly of section .text:
# CHECK-EMPTY:
# CHECK-NEXT: 0000000000000000 <.text>:
# CHECK-NEXT:                                                                   ┠─ a = S3
# CHECK-NEXT:                                                                   ┃ ┠─ b = S4
# CHECK-NEXT:                                                                   ┃ ┃ ┠─ c = S5
# CHECK-NEXT:                                                                   ┃ ┃ ┃ ┌─ x = S3
# CHECK-NEXT:        0:       add 3, 4, 3                                       ┻ ┃ ┃ ╈
# CHECK-NEXT:                                                                   ┌─ y = S3
# CHECK-NEXT:        4:       add 3, 3, 5                                       ╈ ┃ ┃ ┻
# CHECK-NEXT:        8:       extsw 3, 3                                        ┻ ┃ ┃
# CHECK-NEXT:        c:       blr                                                 ┃ ┃
# CHECK-NEXT:                 ...
# CHECK-NEXT:                                                                   ┠─ a = S3
# CHECK-NEXT:       1c:       addi 3, 3, 1                                      ┃
# CHECK-NEXT:       20:       extsw 3, 3                                        ┻
# CHECK-NEXT:       24:       blr
# CHECK-NEXT:                 ...

	.text
	.file	"debug.c"
	.globl	foo                     # -- Begin function foo
	.p2align	2
	.type	foo,@function
	.section	.opd,"aw",@progbits
foo:                                    # @foo
	.p2align	3
	.quad	.Lfunc_begin0
	.quad	.TOC.@tocbase
	.quad	0
	.text
.Lfunc_begin0:
	.file	1 "/work" "llvm/src/llvm/test/tools/llvm-objdump/ARM/Inputs/debug.c"
	.loc	1 1 0                   # llvm/src/llvm/test/tools/llvm-objdump/ARM/Inputs/debug.c:1:0
	.cfi_sections .debug_frame
	.cfi_startproc
# %bb.0:                                # %entry
	#DEBUG_VALUE: foo:a <- $x3
	#DEBUG_VALUE: foo:a <- $r3
	#DEBUG_VALUE: foo:b <- $x4
	#DEBUG_VALUE: foo:b <- $x4
	#DEBUG_VALUE: foo:b <- $r4
	#DEBUG_VALUE: foo:c <- $x5
	#DEBUG_VALUE: foo:c <- $x5
	#DEBUG_VALUE: foo:c <- $r5
	.loc	1 2 13 prologue_end     # llvm/src/llvm/test/tools/llvm-objdump/ARM/Inputs/debug.c:2:13
	add 3, 4, 3
.Ltmp0:
	#DEBUG_VALUE: foo:x <- $r3
	.loc	1 3 13                  # llvm/src/llvm/test/tools/llvm-objdump/ARM/Inputs/debug.c:3:13
	add 3, 3, 5
.Ltmp1:
	#DEBUG_VALUE: foo:y <- $r3
	.loc	1 4 3                   # llvm/src/llvm/test/tools/llvm-objdump/ARM/Inputs/debug.c:4:3
	extsw 3, 3
.Ltmp2:
	blr
.Ltmp3:
	.long	0
	.quad	0
.Lfunc_end0:
	.size	foo, .Lfunc_end0-.Lfunc_begin0
	.cfi_endproc
                                        # -- End function
	.globl	bar                     # -- Begin function bar
	.p2align	2
	.type	bar,@function
	.section	.opd,"aw",@progbits
bar:                                    # @bar
	.p2align	3
	.quad	.Lfunc_begin1
	.quad	.TOC.@tocbase
	.quad	0
	.text
.Lfunc_begin1:
	.loc	1 7 0                   # llvm/src/llvm/test/tools/llvm-objdump/ARM/Inputs/debug.c:7:0
	.cfi_startproc
# %bb.0:                                # %entry
	#DEBUG_VALUE: bar:a <- $x3
	#DEBUG_VALUE: bar:a <- $r3
	.loc	1 8 4 prologue_end      # llvm/src/llvm/test/tools/llvm-objdump/ARM/Inputs/debug.c:8:4
	addi 3, 3, 1
.Ltmp4:
	#DEBUG_VALUE: bar:a <- $r3
	.loc	1 9 3                   # llvm/src/llvm/test/tools/llvm-objdump/ARM/Inputs/debug.c:9:3
	extsw 3, 3
.Ltmp5:
	blr
.Ltmp6:
	.long	0
	.quad	0
.Lfunc_end1:
	.size	bar, .Lfunc_end1-.Lfunc_begin1
	.cfi_endproc
                                        # -- End function
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 10.0.0 (git@github.com:llvm/llvm-project.git e73f78acd34360f7450b81167d9dc858ccddc262)" # string offset=0
.Linfo_string1:
	.asciz	"/work/llvm/src/llvm/test/tools/llvm-objdump/ARM/Inputs/debug.c" # string offset=101
.Linfo_string2:
	.asciz	"/work/scratch"         # string offset=164
.Linfo_string3:
	.asciz	"foo"                   # string offset=178
.Linfo_string4:
	.asciz	"int"                   # string offset=182
.Linfo_string5:
	.asciz	"bar"                   # string offset=186
.Linfo_string6:
	.asciz	"a"                     # string offset=190
.Linfo_string7:
	.asciz	"b"                     # string offset=192
.Linfo_string8:
	.asciz	"c"                     # string offset=194
.Linfo_string9:
	.asciz	"x"                     # string offset=196
.Linfo_string10:
	.asciz	"y"                     # string offset=198
	.section	.debug_loc,"",@progbits
.Ldebug_loc0:
	.quad	.Lfunc_begin0-.Lfunc_begin0
	.quad	.Ltmp0-.Lfunc_begin0
	.short	3                       # Loc expr size
	.byte	144                     # super-register DW_OP_regx
	.byte	179                     # 1203
	.byte	9                       #
	.quad	0
	.quad	0
.Ldebug_loc1:
	.quad	.Ltmp0-.Lfunc_begin0
	.quad	.Ltmp1-.Lfunc_begin0
	.short	3                       # Loc expr size
	.byte	144                     # super-register DW_OP_regx
	.byte	179                     # 1203
	.byte	9                       #
	.quad	0
	.quad	0
.Ldebug_loc2:
	.quad	.Ltmp1-.Lfunc_begin0
	.quad	.Ltmp2-.Lfunc_begin0
	.short	3                       # Loc expr size
	.byte	144                     # super-register DW_OP_regx
	.byte	179                     # 1203
	.byte	9                       #
	.quad	0
	.quad	0
.Ldebug_loc3:
	.quad	.Lfunc_begin1-.Lfunc_begin0
	.quad	.Ltmp5-.Lfunc_begin0
	.short	3                       # Loc expr size
	.byte	144                     # super-register DW_OP_regx
	.byte	179                     # 1203
	.byte	9                       #
	.quad	0
	.quad	0
	.section	.debug_abbrev,"",@progbits
	.byte	1                       # Abbreviation Code
	.byte	17                      # DW_TAG_compile_unit
	.byte	1                       # DW_CHILDREN_yes
	.byte	37                      # DW_AT_producer
	.byte	14                      # DW_FORM_strp
	.byte	19                      # DW_AT_language
	.byte	5                       # DW_FORM_data2
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	16                      # DW_AT_stmt_list
	.byte	23                      # DW_FORM_sec_offset
	.byte	27                      # DW_AT_comp_dir
	.byte	14                      # DW_FORM_strp
	.byte	17                      # DW_AT_low_pc
	.byte	1                       # DW_FORM_addr
	.byte	18                      # DW_AT_high_pc
	.byte	6                       # DW_FORM_data4
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	2                       # Abbreviation Code
	.byte	46                      # DW_TAG_subprogram
	.byte	1                       # DW_CHILDREN_yes
	.byte	17                      # DW_AT_low_pc
	.byte	1                       # DW_FORM_addr
	.byte	18                      # DW_AT_high_pc
	.byte	6                       # DW_FORM_data4
	.byte	64                      # DW_AT_frame_base
	.byte	24                      # DW_FORM_exprloc
	.ascii	"\227B"                 # DW_AT_GNU_all_call_sites
	.byte	25                      # DW_FORM_flag_present
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	58                      # DW_AT_decl_file
	.byte	11                      # DW_FORM_data1
	.byte	59                      # DW_AT_decl_line
	.byte	11                      # DW_FORM_data1
	.byte	39                      # DW_AT_prototyped
	.byte	25                      # DW_FORM_flag_present
	.byte	73                      # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	63                      # DW_AT_external
	.byte	25                      # DW_FORM_flag_present
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	3                       # Abbreviation Code
	.byte	5                       # DW_TAG_formal_parameter
	.byte	0                       # DW_CHILDREN_no
	.byte	2                       # DW_AT_location
	.byte	23                      # DW_FORM_sec_offset
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	58                      # DW_AT_decl_file
	.byte	11                      # DW_FORM_data1
	.byte	59                      # DW_AT_decl_line
	.byte	11                      # DW_FORM_data1
	.byte	73                      # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	4                       # Abbreviation Code
	.byte	5                       # DW_TAG_formal_parameter
	.byte	0                       # DW_CHILDREN_no
	.byte	2                       # DW_AT_location
	.byte	24                      # DW_FORM_exprloc
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	58                      # DW_AT_decl_file
	.byte	11                      # DW_FORM_data1
	.byte	59                      # DW_AT_decl_line
	.byte	11                      # DW_FORM_data1
	.byte	73                      # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	5                       # Abbreviation Code
	.byte	52                      # DW_TAG_variable
	.byte	0                       # DW_CHILDREN_no
	.byte	2                       # DW_AT_location
	.byte	23                      # DW_FORM_sec_offset
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	58                      # DW_AT_decl_file
	.byte	11                      # DW_FORM_data1
	.byte	59                      # DW_AT_decl_line
	.byte	11                      # DW_FORM_data1
	.byte	73                      # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	6                       # Abbreviation Code
	.byte	36                      # DW_TAG_base_type
	.byte	0                       # DW_CHILDREN_no
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	62                      # DW_AT_encoding
	.byte	11                      # DW_FORM_data1
	.byte	11                      # DW_AT_byte_size
	.byte	11                      # DW_FORM_data1
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	0                       # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	4                       # DWARF version number
	.long	.debug_abbrev           # Offset Into Abbrev. Section
	.byte	8                       # Address Size (in bytes)
	.byte	1                       # Abbrev [1] 0xb:0xb5 DW_TAG_compile_unit
	.long	.Linfo_string0          # DW_AT_producer
	.short	12                      # DW_AT_language
	.long	.Linfo_string1          # DW_AT_name
	.long	.Lline_table_start0     # DW_AT_stmt_list
	.long	.Linfo_string2          # DW_AT_comp_dir
	.quad	.Lfunc_begin0           # DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin0 # DW_AT_high_pc
	.byte	2                       # Abbrev [2] 0x2a:0x65 DW_TAG_subprogram
	.quad	.Lfunc_begin0           # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0 # DW_AT_high_pc
	.byte	1                       # DW_AT_frame_base
	.byte	81
                                        # DW_AT_GNU_all_call_sites
	.long	.Linfo_string3          # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	1                       # DW_AT_decl_line
                                        # DW_AT_prototyped
	.long	184                     # DW_AT_type
                                        # DW_AT_external
	.byte	3                       # Abbrev [3] 0x43:0xf DW_TAG_formal_parameter
	.long	.Ldebug_loc0            # DW_AT_location
	.long	.Linfo_string6          # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	1                       # DW_AT_decl_line
	.long	184                     # DW_AT_type
	.byte	4                       # Abbrev [4] 0x52:0xf DW_TAG_formal_parameter
	.byte	3                       # DW_AT_location
	.byte	144
	.ascii	"\264\t"
	.long	.Linfo_string7          # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	1                       # DW_AT_decl_line
	.long	184                     # DW_AT_type
	.byte	4                       # Abbrev [4] 0x61:0xf DW_TAG_formal_parameter
	.byte	3                       # DW_AT_location
	.byte	144
	.ascii	"\265\t"
	.long	.Linfo_string8          # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	1                       # DW_AT_decl_line
	.long	184                     # DW_AT_type
	.byte	5                       # Abbrev [5] 0x70:0xf DW_TAG_variable
	.long	.Ldebug_loc1            # DW_AT_location
	.long	.Linfo_string9          # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	2                       # DW_AT_decl_line
	.long	184                     # DW_AT_type
	.byte	5                       # Abbrev [5] 0x7f:0xf DW_TAG_variable
	.long	.Ldebug_loc2            # DW_AT_location
	.long	.Linfo_string10         # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	3                       # DW_AT_decl_line
	.long	184                     # DW_AT_type
	.byte	0                       # End Of Children Mark
	.byte	2                       # Abbrev [2] 0x8f:0x29 DW_TAG_subprogram
	.quad	.Lfunc_begin1           # DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin1 # DW_AT_high_pc
	.byte	1                       # DW_AT_frame_base
	.byte	81
                                        # DW_AT_GNU_all_call_sites
	.long	.Linfo_string5          # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	7                       # DW_AT_decl_line
                                        # DW_AT_prototyped
	.long	184                     # DW_AT_type
                                        # DW_AT_external
	.byte	3                       # Abbrev [3] 0xa8:0xf DW_TAG_formal_parameter
	.long	.Ldebug_loc3            # DW_AT_location
	.long	.Linfo_string6          # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	7                       # DW_AT_decl_line
	.long	184                     # DW_AT_type
	.byte	0                       # End Of Children Mark
	.byte	6                       # Abbrev [6] 0xb8:0x7 DW_TAG_base_type
	.long	.Linfo_string4          # DW_AT_name
	.byte	5                       # DW_AT_encoding
	.byte	4                       # DW_AT_byte_size
	.byte	0                       # End Of Children Mark
.Ldebug_info_end0:
	.ident	"clang version 10.0.0 (git@github.com:llvm/llvm-project.git e73f78acd34360f7450b81167d9dc858ccddc262)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.section	.debug_line,"",@progbits
.Lline_table_start0:
