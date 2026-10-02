; The DWARF 2 form of subprogram-high-pc-past-symbol.s, where DW_AT_high_pc is
; an address rather than a length.

	.text
	.globl	_a
	.p2align 2
_a:
	ret
_filler:
	nop
	.globl	_b
	.p2align 2
_b:
	ret

	.section __DWARF,__debug_abbrev,regular,debug
	.byte	1                       ; abbrev 1: DW_TAG_compile_unit
	.byte	0x11
	.byte	1                       ; DW_CHILDREN_yes
	.byte	0x25, 0x08              ; DW_AT_producer,  DW_FORM_string
	.byte	0x13, 0x0b              ; DW_AT_language,  DW_FORM_data1
	.byte	0x03, 0x08              ; DW_AT_name,      DW_FORM_string
	.byte	0x11, 0x01              ; DW_AT_low_pc,    DW_FORM_addr
	.byte	0x12, 0x01              ; DW_AT_high_pc,   DW_FORM_addr
	.byte	0, 0
	.byte	2                       ; abbrev 2: DW_TAG_subprogram
	.byte	0x2e
	.byte	1                       ; DW_CHILDREN_yes
	.byte	0x03, 0x08              ; DW_AT_name,      DW_FORM_string
	.byte	0x11, 0x01              ; DW_AT_low_pc,    DW_FORM_addr
	.byte	0x12, 0x01              ; DW_AT_high_pc,   DW_FORM_addr
	.byte	0x3f, 0x0c              ; DW_AT_external,  DW_FORM_flag
	.byte	0, 0
	.byte	3                       ; abbrev 3: DW_TAG_lexical_block
	.byte	0x0b
	.byte	0                       ; DW_CHILDREN_no
	.byte	0x11, 0x01              ; DW_AT_low_pc,    DW_FORM_addr
	.byte	0x12, 0x01              ; DW_AT_high_pc,   DW_FORM_addr
	.byte	0, 0
	.byte	0

	.section __DWARF,__debug_info,regular,debug
Lcu_begin:
	.long	Lcu_end-Lcu_version
Lcu_version:
	.short	2
	.long	0
	.byte	8
	.byte	1                       ; DW_TAG_compile_unit
	.asciz	"hand-written"
	.byte	0x0c                    ; DW_LANG_C99
	.asciz	"t.c"
	.quad	_a
	.quad	0xc                     ; _a, _filler and _b together
	.byte	2                       ; DW_TAG_subprogram "a"
	.asciz	"a"
	.quad	_a
	.quad	0x8                     ; four bytes past the end of _a
	.byte	1
	.byte	3                       ; DW_TAG_lexical_block in "a"
	.quad	_a
	.quad	0x8                     ; reaches as far as its parent
	.byte	0                       ; end of "a"'s children
	.byte	2                       ; DW_TAG_subprogram "b"
	.asciz	"b"
	.quad	_b
	.quad	0xc
	.byte	1
	.byte	0                       ; end of "b"'s children
	.byte	0                       ; end of the compile unit's children
Lcu_end:
	.subsections_via_symbols
