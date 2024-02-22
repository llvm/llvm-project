// REQUIRES: aarch64-registered-target
// RUN: rm -rf %t && mkdir -p %t
// RUN: llvm-mc --filetype=obj --cas-backend --cas=%t/cas --mccas-native %s --mccas-emit-casid-file -triple arm64-apple-macosx14.0.0 -o %t/test.o
// RUN: cat %t/test.o.casid | FileCheck %s --check-prefix=NATIVE_FILENAME
// NATIVE_FILENAME: CASID:Jllvmcas://{{.*}}
//
// RUN: rm -rf %t && mkdir -p %t
// RUN: llvm-mc --filetype=obj --cas-backend --cas=%t/cas --mccas-verify %s --mccas-emit-casid-file -triple arm64-apple-macosx14.0.0 -o %t/test.o
// RUN: cat %t/test.o.casid | FileCheck %s --check-prefix=VERIFY_FILENAME
// VERIFY_FILENAME: CASID:Jllvmcas://{{.*}}
//
// RUN: rm -rf %t && mkdir -p %t
// RUN: llvm-mc --filetype=obj --cas-backend --cas=%t/cas --mccas-casid %s --mccas-emit-casid-file -triple arm64-apple-macosx14.0.0 -o %t/test.o
// RUN: not cat %t/test.o.casid
//
// RUN: rm -rf %t && mkdir -p %t
// RUN: llvm-mc --filetype=obj --cas-backend --cas=%t/cas --mccas-native %s --mccas-emit-casid-file -triple arm64-apple-macosx14.0.0 -o -
// RUN: not cat %t/test.o.casid
//
// RUN: rm -rf %t && mkdir -p %t
// RUN: llvm-mc --filetype=obj --cas-backend --cas=%t/cas --mccas-verify %s --mccas-emit-casid-file -triple arm64-apple-macosx14.0.0 -o -
// RUN: not cat %t/test.o.casid
//
// RUN: rm -rf %t && mkdir -p %t
// RUN: llvm-mc --filetype=obj --cas-backend --cas=%t/cas --mccas-casid %s --mccas-emit-casid-file -triple arm64-apple-macosx14.0.0 -o -
// RUN: not cat %t/test.o.casid

	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 14, 0
	.globl	__Z3fooi
	.p2align	2
__Z3fooi:
Lfunc_begin0:
	.file	1 "/Users/shubham/Development" "test109275485/a.cpp"
	.loc	1 1 0
	.cfi_startproc

	sub	sp, sp, #16
	.cfi_def_cfa_offset 16
	str	w0, [sp, #12]
Ltmp1:
	.loc	1 2 12 prologue_end
	ldr	w8, [sp, #12]
	.loc	1 2 13 is_stmt 0
	add	w0, w8, #2
	.loc	1 2 5 epilogue_begin
	add	sp, sp, #16
	ret
Ltmp2:
Lfunc_end0:
	.cfi_endproc

	.section	__DWARF,__debug_abbrev,regular,debug
Lsection_abbrev:
	.byte	1
	.byte	17
	.byte	1
	.byte	37
	.byte	14
	.byte	19
	.byte	5
	.byte	3
	.byte	14
	.ascii	"\202|"
	.byte	14
	.byte	16
	.byte	23
	.byte	27
	.byte	14
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
	.ascii	"\347\177"
	.byte	25
	.byte	64
	.byte	24
	.byte	110
	.byte	14
	.byte	3
	.byte	14
	.byte	58
	.byte	11
	.byte	59
	.byte	11
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
	.byte	24
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
	.byte	0
	.section	__DWARF,__debug_info,regular,debug
Lsection_info:
Lcu_begin0:
.set Lset0, Ldebug_info_end0-Ldebug_info_start0
	.long	Lset0
Ldebug_info_start0:
	.short	4
.set Lset1, Lsection_abbrev-Lsection_abbrev
	.long	Lset1
	.byte	8
	.byte	1
	.long	0
	.short	33
	.long	102
	.long	149
.set Lset2, Lline_table_start0-Lsection_line
	.long	Lset2
	.long	151
	.quad	Lfunc_begin0
.set Lset3, Lfunc_end0-Lfunc_begin0
	.long	Lset3
	.byte	2
	.quad	Lfunc_begin0
.set Lset4, Lfunc_end0-Lfunc_begin0
	.long	Lset4

	.byte	1
	.byte	111
	.long	224
	.long	220
	.byte	1
	.byte	1
	.long	90

	.byte	3
	.byte	2
	.byte	145
	.byte	12
	.long	236
	.byte	1
	.byte	1
	.long	90
	.byte	0
	.byte	4
	.long	232
	.byte	5
	.byte	4
	.byte	0
Ldebug_info_end0:
	.section	__DWARF,__debug_str,regular,debug
Linfo_string:
	.asciz	"clang version 18.0.0 (git@github.com:apple/llvm-project.git 7f16558429157c87cdbe8802086dd04b0deed7f7)"
	.asciz	"/Users/shubham/Development/test109275485/a.cpp"
	.asciz	"/"
	.asciz	"/Users/shubham/Development/llvm-project-cas/llvm-project/build_ninja"
	.asciz	"foo"
	.asciz	"_Z3fooi"
	.asciz	"int"
	.asciz	"x"
	.section	__DWARF,__apple_names,regular,debug
Lnames_begin:
	.long	1212240712
	.short	1
	.short	0
	.long	2
	.long	2
	.long	12
	.long	0
	.long	1
	.short	1
	.short	6
	.long	0
	.long	1
	.long	1784752350
	.long	193491849
.set Lset5, LNames1-Lnames_begin
	.long	Lset5
.set Lset6, LNames0-Lnames_begin
	.long	Lset6
LNames1:
	.long	224
	.long	1
	.long	46
	.long	0
LNames0:
	.long	220
	.long	1
	.long	46
	.long	0
	.section	__DWARF,__apple_objc,regular,debug
Lobjc_begin:
	.long	1212240712
	.short	1
	.short	0
	.long	1
	.long	0
	.long	12
	.long	0
	.long	1
	.short	1
	.short	6
	.long	-1
	.section	__DWARF,__apple_namespac,regular,debug
Lnamespac_begin:
	.long	1212240712
	.short	1
	.short	0
	.long	1
	.long	0
	.long	12
	.long	0
	.long	1
	.short	1
	.short	6
	.long	-1
	.section	__DWARF,__apple_types,regular,debug
Ltypes_begin:
	.long	1212240712
	.short	1
	.short	0
	.long	1
	.long	1
	.long	20
	.long	0
	.long	3
	.short	1
	.short	6
	.short	3
	.short	5
	.short	4
	.short	11
	.long	0
	.long	193495088
.set Lset7, Ltypes0-Ltypes_begin
	.long	Lset7
Ltypes0:
	.long	232
	.long	1
	.long	90
	.short	36
	.byte	0
	.long	0
.subsections_via_symbols
	.section	__DWARF,__debug_line,regular,debug
Lsection_line:
Lline_table_start0:
