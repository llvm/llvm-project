;; Demonstrate dumping DW_AT_APPLE_enum_kind.
; RUN: llvm-mc -triple=aarch64--darwin -filetype=obj < %s | \
; RUN:     llvm-dwarfdump -v - | FileCheck %s

; CHECK: .debug_abbrev contents:
; CHECK: DW_AT_APPLE_enum_kind DW_FORM_data1
; CHECK: .debug_info contents:
; CHECK: DW_AT_APPLE_enum_kind [DW_FORM_data1] (DW_APPLE_ENUM_KIND_Closed)
; CHECK: DW_AT_APPLE_enum_kind [DW_FORM_data1] (DW_APPLE_ENUM_KIND_Open)

	.section	__DWARF,__debug_abbrev,regular,debug
Lsection_abbrev:
	.byte	1                               ; Abbreviation Code
	.byte	17                              ; DW_TAG_compile_unit
	.byte	1                               ; DW_CHILDREN_yes
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	2                               ; Abbreviation Code
	.byte	4                               ; DW_TAG_enumeration_type
	.byte	0                               ; DW_CHILDREN_no
	.ascii	"\361\177"                      ; DW_AT_APPLE_enum_kind
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	0                               ; EOM(3)
	.section	__DWARF,__debug_info,regular,debug
Lsection_info:
Lcu_begin0:
.set Lset0, Ldebug_info_end0-Ldebug_info_start0 ; Length of Unit
	.long	Lset0
Ldebug_info_start0:
	.short	5                               ; DWARF version number
	.byte	1                               ; DWARF Unit Type
	.byte	8                               ; Address Size (in bytes)
.set Lset1, Lsection_abbrev-Lsection_abbrev ; Offset Into Abbrev. Section
	.long	Lset1
	.byte	1                               ; Abbrev [1] 0xc:0x40 DW_TAG_compile_unit
	.byte	2                               ; Abbrev [3] 0x2a:0x9 DW_TAG_enumeration_type
	.byte	0                               ; DW_APPLE_ENUM_KIND_Closed
	.byte	2                               ; Abbrev [3] 0x42:0x9 DW_TAG_enumeration_type
	.byte	1                               ; DW_APPLE_ENUM_KIND_Open
	.byte	0                               ; End Of Children Mark
Ldebug_info_end0:

