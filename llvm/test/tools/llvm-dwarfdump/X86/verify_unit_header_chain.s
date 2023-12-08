# RUN: llvm-mc %s -filetype obj -triple x86_64-apple-darwin -o - \
# RUN: | not llvm-dwarfdump -verify - \
# RUN: | FileCheck %s

# CHECK: Verifying .debug_info Unit Header Chain...
# CHECK-NEXT: error: Units[1] - start offset: 0x0000000d
# CHECK-NEXT: note: The unit type encoding is not valid.
# CHECK-NEXT: note: The address size is unsupported.
# CHECK-NEXT: error: Units[2] - start offset: 0x00000026
# CHECK-NEXT: note: The 16 bit unit header version is not valid.
# CHECK-NEXT: note: The offset into the .debug_abbrev section is not valid.
# CHECK-NEXT: error: Units[4] - start offset: 0x00000041
# CHECK-NEXT: note: The length for this unit is too large for the .debug_info provided.

	.section	__TEXT,__text,regular,pure_instructions
	.file	1 "basic.c"
	.comm	_i,4,2                  ## @i
	.comm	_j,4,2                  ## @j
	.section	__DWARF,__debug_str,regular,debug
Linfo_string:
	.asciz	"clang version 5.0.0 (trunk 307232) (llvm/trunk 307042)" ## string offset=0
	.asciz	"basic.c"               ## string offset=55
	.asciz	"/Users/sgravani/Development/tests" ## string offset=63
	.asciz	"i"                     ## string offset=97
	.asciz	"int"                   ## string offset=99
	.asciz	"j"                     ## string offset=103
	.section	__DWARF,__debug_abbrev,regular,debug
Lsection_abbrev:
	.byte	1                       ## Abbreviation Code
	.byte	17                      ## DW_TAG_compile_unit
        .byte   0                       ## DW_CHILDREN_no
	.byte	0                       ## EOM(1)
	.byte	0                       ## EOM(2)
	.byte	0                       ## EOM(3)
	.section	__DWARF,__debug_info,regular,debug
Lsection_info:
Lcu_begin0:
	.long	9                      ## Length of Unit
	.short	4                       ## DWARF version number
Lset0 = Lsection_abbrev-Lsection_abbrev ## Offset Into Abbrev. Section
	.long	Lset0
	.byte	4                       ## Address Size (in bytes)
	.byte	1                       ## Abbrev [1] 0xc:0x45 DW_TAG_compile_unit
	.byte	0                       ## End Of Children Mark
Ltu_begin0:
	.long	21                      ## Length of Unit
	.short	5                       ## DWARF version number
	.byte	0                       ## DWARF Unit Type -- Error: The unit type encoding is not valid.
	.byte	3                       ## Address Size (in bytes) -- Error: The address size is unsupported.
	.long	0
	.quad	0
	.long   0
	.byte 	0
Lcu_begin1:
	.long	10                      ## Length of Unit
	.short	6                       ## DWARF version number -- Error: The 16 bit unit header version is not valid.
	.byte	1                       ## DWARF Unit Type
	.byte	4                       ## Address Size (in bytes) -- The offset into the .debug_abbrev section is not valid.
	.long	Lline_table_start0
	.byte	1                       ## Abbrev [1] 0xc:0x45 DW_TAG_compile_unit
	.byte	0                       ## End Of Children Mark
Lcu_begin2:
	.long	9                      ## Length of Unit
	.short	5                       ## DWARF version number
	.byte	1                       ## DWARF Unit Type
	.byte	4                       ## Address Size (in bytes)
	.long	0						## Abbrev offset
	.byte 	0
Ltu_begin1:
	.long	26                      ## Length of Unit -- Error: The length for this unit is too large for the .debug_info provided.
	.short	5                       ## DWARF version number
	.byte	2                       ## DWARF Unit Type
	.byte	4                       ## Address Size (in bytes)
	.long	0
	.quad	0
	.long   0
	.byte 	0

.subsections_via_symbols
	.section	__DWARF,__debug_line,regular,debug
Lsection_line:
Lline_table_start0:
