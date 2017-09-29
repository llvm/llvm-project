# RUN: llvm-mc %s -filetype obj -triple x86_64-apple-darwin -o - \
# RUN: | not llvm-dwarfdump -verify - \
# RUN: | FileCheck %s

# CHECK: Verifying .apple_names...
# CHECK-NEXT:	error: Unsupported form: failed to read HashData.

# This test is meant to verify that the -verify option
# in llvm-dwarfdump, correctly identifies that Atom[0].form is unsupported.
# As a result, the hashdata cannot be read.

	.section	__TEXT,__text,regular,pure_instructions
	.file	1 "basic.c"
	.comm	_i,4,2                  ## @i
	.comm	_j,4,2                  ## @j
	.section	__DWARF,__debug_str,regular,debug
Linfo_string:
	.asciz	"Apple LLVM version 8.1.0 (clang-802.0.35)" ## string offset=0
	.asciz	"basic.c"               ## string offset=42
	.asciz	"/Users/sgravani/Development/tests" ## string offset=50
	.asciz	"i"                     ## string offset=84
	.asciz	"int"                   ## string offset=86
	.asciz	"j"                     ## string offset=90
	.section	__DWARF,__debug_info,regular,debug
Lsection_info:
	.section	__DWARF,__apple_names,regular,debug
Lnames_begin:
	.long	1212240712              ## Header Magic
	.short	1                       ## Header Version
	.short	0                       ## Header Hash Function
	.long	2                       ## Header Bucket Count
	.long	2                       ## Header Hash Count
	.long	12                      ## Header Data Length
	.long	0                       ## HeaderData Die Offset Base
	.long	1                       ## HeaderData Atom Count
	.short	1                       ## DW_ATOM_die_offset
	.short	400                     ## DW_FORM_data4 -- error: unsupported form; failed to read HashData.
	.long	0                       ## Bucket 0
	.long	1                       ## Bucket 1
	.long	177678                  ## Hash in Bucket 0
	.long	177679                  ## Hash in Bucket 1
	.long	LNames0-Lnames_begin    ## Offset in Bucket 0
	.long	LNames1-Lnames_begin    ## Offset in Bucket 1
LNames0:
	.long	84                      ## i
	.long	1                       ## Num DIEs
	.long	30
	.long	0
LNames1:
	.long	90                      ## j
	.long	1                       ## Num DIEs
	.long	58
	.long	0

.subsections_via_symbols
	.section	__DWARF,__debug_line,regular,debug
Lsection_line:
Lline_table_start0:
