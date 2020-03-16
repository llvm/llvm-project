## Check that the --debug-vars option works for simple register locations, when
## using DWARF4 debug info, with functions in multiple sections.

## Generated with this compile command, with the source code in Inputs/debug.c:
## clang --target=arm--none-eabi -march=armv7-a -c debug.c -O1 -gdwarf-5 -S -o - -ffunction-sections

## The unicode characters in this test cause test failures on Windows.
# UNSUPPORTED: system-windows

# RUN: llvm-mc -triple armv8a--none-eabi < %s -filetype=obj --dwarf-version=5 | \
# RUN:     llvm-objdump - -d --debug-vars --no-show-raw-insn | \
# RUN:     FileCheck %s

# CHECK: Disassembly of section .text.foo:
# CHECK-EMPTY:
# CHECK-NEXT: 00000000 <foo>:
# CHECK-NEXT:                                                                   ┠─ a = R0
# CHECK-NEXT:                                                                   ┃ ┠─ b = R1
# CHECK-NEXT:                                                                   ┃ ┃ ┠─ c = R2
# CHECK-NEXT:                                                                   ┃ ┃ ┃ ┌─ x = R0
# CHECK-NEXT:        0:       add     r0, r1, r0                                ┻ ┃ ┃ ╈
# CHECK-NEXT:                                                                   ┌─ y = R0
# CHECK-NEXT:        4:       add     r0, r0, r2                                ╈ ┃ ┃ ┻
# CHECK-NEXT:        8:       bx      lr                                        ┻ ┻ ┻
# CHECK-EMPTY:
# CHECK-NEXT: Disassembly of section .text.bar:
# CHECK-EMPTY:
# CHECK-NEXT: 00000000 <bar>:
# CHECK-NEXT:                                                                   ┠─ a = R0
# CHECK-NEXT:        0:       add     r0, r0, #1                                ┃
# CHECK-NEXT:        4:       bx      lr                                        ┻

	.text
	.syntax unified
	.eabi_attribute	67, "2.09"
	.eabi_attribute	6, 10
	.eabi_attribute	7, 65
	.eabi_attribute	8, 1
	.eabi_attribute	9, 2
	.fpu	neon
	.eabi_attribute	34, 0
	.eabi_attribute	17, 1
	.eabi_attribute	20, 1
	.eabi_attribute	21, 1
	.eabi_attribute	23, 3
	.eabi_attribute	24, 1
	.eabi_attribute	25, 1
	.eabi_attribute	38, 1
	.eabi_attribute	18, 4
	.eabi_attribute	26, 2
	.eabi_attribute	14, 0
	.file	"debug.c"
	.section	.text.foo,"ax",%progbits
	.globl	foo
	.p2align	2
	.type	foo,%function
	.code	32
foo:
.Lfunc_begin0:
	.file	0 "/work/scratch" "/work/llvm/src/llvm/test/tools/llvm-objdump/ARM/Inputs/debug.c" md5 0x07374f01ab24ec7c07db73bc13bd778e
	.file	1 "/work" "llvm/src/llvm/test/tools/llvm-objdump/ARM/Inputs/debug.c" md5 0x07374f01ab24ec7c07db73bc13bd778e
	.loc	1 1 0
	.fnstart
	.cfi_sections .debug_frame
	.cfi_startproc
	.loc	1 2 13 prologue_end
	add	r0, r1, r0
.Ltmp0:
	.loc	1 3 13
	add	r0, r0, r2
.Ltmp1:
	.loc	1 4 3
	bx	lr
.Ltmp2:
.Lfunc_end0:
	.size	foo, .Lfunc_end0-foo
	.cfi_endproc
	.cantunwind
	.fnend

	.section	.text.bar,"ax",%progbits
	.globl	bar
	.p2align	2
	.type	bar,%function
	.code	32
bar:
.Lfunc_begin1:
	.loc	1 7 0
	.fnstart
	.cfi_startproc
	.loc	1 8 4 prologue_end
	add	r0, r0, #1
.Ltmp3:
	.loc	1 9 3
	bx	lr
.Ltmp4:
.Lfunc_end1:
	.size	bar, .Lfunc_end1-bar
	.cfi_endproc
	.cantunwind
	.fnend

	.section	.debug_str_offsets,"",%progbits
	.long	48
	.short	5
	.short	0
.Lstr_offsets_base0:
	.section	.debug_str,"MS",%progbits,1
.Linfo_string0:
	.asciz	"clang version 10.0.0 (git@github.com:llvm/llvm-project.git e73f78acd34360f7450b81167d9dc858ccddc262)"
.Linfo_string1:
	.asciz	"/work/llvm/src/llvm/test/tools/llvm-objdump/ARM/Inputs/debug.c"
.Linfo_string2:
	.asciz	"/work/scratch"
.Linfo_string3:
	.asciz	"foo"
.Linfo_string4:
	.asciz	"int"
.Linfo_string5:
	.asciz	"bar"
.Linfo_string6:
	.asciz	"a"
.Linfo_string7:
	.asciz	"b"
.Linfo_string8:
	.asciz	"c"
.Linfo_string9:
	.asciz	"x"
.Linfo_string10:
	.asciz	"y"
	.section	.debug_str_offsets,"",%progbits
	.long	.Linfo_string0
	.long	.Linfo_string1
	.long	.Linfo_string2
	.long	.Linfo_string3
	.long	.Linfo_string4
	.long	.Linfo_string5
	.long	.Linfo_string6
	.long	.Linfo_string7
	.long	.Linfo_string8
	.long	.Linfo_string9
	.long	.Linfo_string10
	.section	.debug_loclists,"",%progbits
	.long	.Ldebug_loclist_table_end0-.Ldebug_loclist_table_start0
.Ldebug_loclist_table_start0:
	.short	5
	.byte	4
	.byte	0
	.long	3
.Lloclists_table_base0:
	.long	.Ldebug_loc0-.Lloclists_table_base0
	.long	.Ldebug_loc1-.Lloclists_table_base0
	.long	.Ldebug_loc2-.Lloclists_table_base0
.Ldebug_loc0:
	.byte	3
	.byte	0
	.uleb128 .Ltmp0-.Lfunc_begin0
	.byte	1
	.byte	80
	.byte	0
.Ldebug_loc1:
	.byte	1
	.byte	0
	.byte	4
	.uleb128 .Ltmp0-.Lfunc_begin0
	.uleb128 .Ltmp1-.Lfunc_begin0
	.byte	1
	.byte	80
	.byte	0
.Ldebug_loc2:
	.byte	1
	.byte	0
	.byte	4
	.uleb128 .Ltmp1-.Lfunc_begin0
	.uleb128 .Lfunc_end0-.Lfunc_begin0
	.byte	1
	.byte	80
	.byte	0
.Ldebug_loclist_table_end0:
	.section	.debug_abbrev,"",%progbits
	.byte	1
	.byte	17
	.byte	1
	.byte	37
	.byte	37
	.byte	19
	.byte	5
	.byte	3
	.byte	37
	.byte	114
	.byte	23
	.byte	16
	.byte	23
	.byte	27
	.byte	37
	.byte	17
	.byte	1
	.byte	85
	.byte	35
	.byte	115
	.byte	23
	.byte	116
	.byte	23
	.ascii	"\214\001"
	.byte	23
	.byte	0
	.byte	0
	.byte	2
	.byte	46
	.byte	1
	.byte	17
	.byte	27
	.byte	18
	.byte	6
	.byte	64
	.byte	24
	.byte	122
	.byte	25
	.byte	3
	.byte	37
	.byte	58
	.byte	11
	.byte	59
	.byte	11
	.byte	39
	.byte	25
	.byte	73
	.byte	19
	.byte	63
	.byte	25
	.byte	0
	.byte	0
	.byte	3
	.byte	5
	.byte	0
	.byte	2
	.byte	34
	.byte	3
	.byte	37
	.byte	58
	.byte	11
	.byte	59
	.byte	11
	.byte	73
	.byte	19
	.byte	0
	.byte	0
	.byte	4
	.byte	5
	.byte	0
	.byte	2
	.byte	24
	.byte	3
	.byte	37
	.byte	58
	.byte	11
	.byte	59
	.byte	11
	.byte	73
	.byte	19
	.byte	0
	.byte	0
	.byte	5
	.byte	52
	.byte	0
	.byte	2
	.byte	34
	.byte	3
	.byte	37
	.byte	58
	.byte	11
	.byte	59
	.byte	11
	.byte	73
	.byte	19
	.byte	0
	.byte	0
	.byte	6
	.byte	36
	.byte	0
	.byte	3
	.byte	37
	.byte	62
	.byte	11
	.byte	11
	.byte	11
	.byte	0
	.byte	0
	.byte	0
	.section	.debug_info,"",%progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0
.Ldebug_info_start0:
	.short	5
	.byte	1
	.byte	4
	.long	.debug_abbrev
	.byte	1
	.byte	0
	.short	12
	.byte	1
	.long	.Lstr_offsets_base0
	.long	.Lline_table_start0
	.byte	2
	.long	0
	.byte	0
	.long	.Laddr_table_base0
	.long	.Lrnglists_table_base0
	.long	.Lloclists_table_base0
	.byte	2
	.byte	0
	.long	.Lfunc_end0-.Lfunc_begin0
	.byte	1
	.byte	91

	.byte	3
	.byte	1
	.byte	1

	.long	132

	.byte	3
	.byte	0
	.byte	6
	.byte	1
	.byte	1
	.long	132
	.byte	4
	.byte	1
	.byte	81
	.byte	7
	.byte	1
	.byte	1
	.long	132
	.byte	4
	.byte	1
	.byte	82
	.byte	8
	.byte	1
	.byte	1
	.long	132
	.byte	5
	.byte	1
	.byte	9
	.byte	1
	.byte	2
	.long	132
	.byte	5
	.byte	2
	.byte	10
	.byte	1
	.byte	3
	.long	132
	.byte	0
	.byte	2
	.byte	1
	.long	.Lfunc_end1-.Lfunc_begin1
	.byte	1
	.byte	91

	.byte	5
	.byte	1
	.byte	7

	.long	132

	.byte	4
	.byte	1
	.byte	80
	.byte	6
	.byte	1
	.byte	7
	.long	132
	.byte	0
	.byte	6
	.byte	4
	.byte	5
	.byte	4
	.byte	0
.Ldebug_info_end0:
	.section	.debug_rnglists,"",%progbits
	.long	.Ldebug_rnglist_table_end0-.Ldebug_rnglist_table_start0
.Ldebug_rnglist_table_start0:
	.short	5
	.byte	4
	.byte	0
	.long	1
.Lrnglists_table_base0:
	.long	.Ldebug_ranges0-.Lrnglists_table_base0
.Ldebug_ranges0:
	.byte	3
	.byte	0
	.uleb128 .Lfunc_end0-.Lfunc_begin0
	.byte	3
	.byte	1
	.uleb128 .Lfunc_end1-.Lfunc_begin1
	.byte	0
.Ldebug_rnglist_table_end0:
	.section	.debug_addr,"",%progbits
	.long	.Ldebug_addr_end0-.Ldebug_addr_start0
.Ldebug_addr_start0:
	.short	5
	.byte	4
	.byte	0
.Laddr_table_base0:
	.long	.Lfunc_begin0
	.long	.Lfunc_begin1
.Ldebug_addr_end0:
	.ident	"clang version 10.0.0 (git@github.com:llvm/llvm-project.git e73f78acd34360f7450b81167d9dc858ccddc262)"
	.section	".note.GNU-stack","",%progbits
	.addrsig
	.eabi_attribute	30, 1
	.section	.debug_line,"",%progbits
.Lline_table_start0:
