# RUN: mkdir -p %t/a
# RUN: cp %p/Inputs/wide-char.c %t/a/wide-char.c
# RUN: sed -e "s,SRC_COMPDIR,%/t/a,g" %s > %t.s
# RUN: llvm-mc -triple armv8a--none-eabi < %t.s -filetype=obj | \
# RUN:     llvm-objdump - -d --debug-vars --source | \
# RUN:     FileCheck %s --strict-whitespace

## The Chinese character in the source does not print correctly on Windows.
# UNSUPPORTED: system-windows

## Check that the --debug-vars option correctly aligns the variable display when
## the source code (printed by the -S option) includes East Asian wide
## characters.

# CHECK: 00000000 <foo>:
# CHECK-NEXT: ;   return *喵;                                                             ┠─ 喵 = R0
# CHECK-NEXT:        0: 00 00 90 e5  	ldr	r0, [r0]                                    ┻   
# CHECK-NEXT:        4: 1e ff 2f e1  	bx	lr                                              

	.text
	.syntax unified
	.eabi_attribute	67, "2.09"
	.eabi_attribute	6, 10
	.eabi_attribute	7, 65
	.eabi_attribute	8, 1
	.eabi_attribute	9, 2
	.fpu	vfpv3
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
	.file	"wide.c"
	.globl	foo
	.p2align	2
	.type	foo,%function
	.code	32
foo:
.Lfunc_begin0:
	.file	1 "SRC_COMPDIR/wide-char.c"
	.loc	1 1 0
	.fnstart
	.cfi_sections .debug_frame
	.cfi_startproc
	.loc	1 2 10 prologue_end
	ldr	r0, [r0]
.Ltmp0:
	.loc	1 2 3 is_stmt 0
	bx	lr
.Ltmp1:
.Lfunc_end0:
	.size	foo, .Lfunc_end0-foo
	.cfi_endproc
	.cantunwind
	.fnend

	.section	.debug_str,"MS",%progbits,1
.Linfo_string0:
	.asciz	"clang version 6.0.0-1ubuntu2 (tags/RELEASE_600/final)"
.Linfo_string1:
	.asciz	"wide-char.c"
.Linfo_string2:
	.asciz	"SRC_COMPDIR"
.Linfo_string3:
	.asciz	"foo"
.Linfo_string4:
	.asciz	"int"
.Linfo_string5:
	.asciz	"\345\226\265"
	.section	.debug_loc,"",%progbits
.Ldebug_loc0:
	.long	.Lfunc_begin0-.Lfunc_begin0
	.long	.Ltmp0-.Lfunc_begin0
	.short	1
	.byte	80
	.long	0
	.long	0
	.section	.debug_abbrev,"",%progbits
	.byte	1
	.byte	17
	.byte	1
	.byte	37
	.byte	14
	.byte	19
	.byte	5
	.byte	3
	.byte	14
	.byte	16
	.byte	23
	.byte	27
	.byte	14
	.ascii	"\264B"
	.byte	25
	.byte	17
	.byte	1
	.byte	18
	.byte	6
	.byte	0
	.byte	0
	.byte	2
	.byte	46
	.byte	1
	.byte	17
	.byte	1
	.byte	18
	.byte	6
	.byte	64
	.byte	24
	.byte	3
	.byte	14
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
	.byte	23
	.byte	3
	.byte	14
	.byte	58
	.byte	11
	.byte	59
	.byte	11
	.byte	73
	.byte	19
	.byte	0
	.byte	0
	.byte	4
	.byte	36
	.byte	0
	.byte	3
	.byte	14
	.byte	62
	.byte	11
	.byte	11
	.byte	11
	.byte	0
	.byte	0
	.byte	5
	.byte	15
	.byte	0
	.byte	73
	.byte	19
	.byte	0
	.byte	0
	.byte	0
	.section	.debug_info,"",%progbits
.Lcu_begin0:
	.long	84
	.short	4
	.long	.debug_abbrev
	.byte	4
	.byte	1
	.long	.Linfo_string0
	.short	12
	.long	.Linfo_string1
	.long	.Lline_table_start0
	.long	.Linfo_string2

	.long	.Lfunc_begin0
	.long	.Lfunc_end0-.Lfunc_begin0
	.byte	2
	.long	.Lfunc_begin0
	.long	.Lfunc_end0-.Lfunc_begin0
	.byte	1
	.byte	91
	.long	.Linfo_string3
	.byte	1
	.byte	1

	.long	75

	.byte	3
	.long	.Ldebug_loc0
	.long	.Linfo_string5
	.byte	1
	.byte	1
	.long	82
	.byte	0
	.byte	4
	.long	.Linfo_string4
	.byte	5
	.byte	4
	.byte	5
	.long	75
	.byte	0
	.section	.debug_ranges,"",%progbits
	.section	.debug_macinfo,"",%progbits
.Lcu_macro_begin0:
	.byte	0
	.section	.debug_pubnames,"",%progbits
	.long	.LpubNames_end0-.LpubNames_begin0
.LpubNames_begin0:
	.short	2
	.long	.Lcu_begin0
	.long	88
	.long	38
	.asciz	"foo"
	.long	0
.LpubNames_end0:
	.section	.debug_pubtypes,"",%progbits
	.long	.LpubTypes_end0-.LpubTypes_begin0
.LpubTypes_begin0:
	.short	2
	.long	.Lcu_begin0
	.long	88
	.long	75
	.asciz	"int"
	.long	0
.LpubTypes_end0:

	.ident	"clang version 6.0.0-1ubuntu2 (tags/RELEASE_600/final)"
	.section	".note.GNU-stack","",%progbits
	.eabi_attribute	30, 1
	.section	.debug_line,"",%progbits
.Lline_table_start0:
