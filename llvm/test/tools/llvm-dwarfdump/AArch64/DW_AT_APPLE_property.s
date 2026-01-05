# Checks that we correctly display the DW_AT_APPLE_property_name of a
# referenced DW_TAG_APPLE_property.
#
# RUN: llvm-mc -triple=aarch64--darwin -filetype=obj -o %t.o < %s
# RUN: not llvm-dwarfdump %t.o 2> %t.errs.txt | FileCheck %s
# RUN: FileCheck %s --check-prefix=ERRORS < %t.errs.txt 

# CHECK: 0x[[PROP_REF:[0-9a-f]+]]: DW_TAG_APPLE_property
# CHECK-NEXT: DW_AT_APPLE_property_name ("autoSynthProp")
#
# CHECK: 0x[[NO_NAME_PROP:[0-9a-f]+]]: DW_TAG_APPLE_property
# CHECK-NOT: DW_AT_APPLE_property_name
#
# CHECK: 0x[[INVALID_STRP:[0-9a-f]+]]: DW_TAG_APPLE_property
# CHECK-NEXT: DW_AT_APPLE_property_name
#
# CHECK: DW_TAG_member
# CHECK:   DW_AT_APPLE_property  (0x[[PROP_REF]] "autoSynthProp")
# CHECK:   DW_AT_APPLE_property  (0x[[NO_NAME_PROP]] "")
# CHECK:   DW_AT_APPLE_property  (0x{{.*}})
# CHECK:   DW_AT_APPLE_property  (0x{{.*}})
# CHECK:   DW_AT_APPLE_property  (0x[[INVALID_STRP]])

# ERRORS: error: decoding DW_AT_APPLE_property_name: not referencing a DW_TAG_APPLE_property
# ERRORS: error: decoding DW_AT_APPLE_property_name: invalid DIE
# ERRORS: error: decoding DW_AT_APPLE_property_name: DW_FORM_strp offset 102 is beyond .debug_str bounds

	.section	__DWARF,__debug_abbrev,regular,debug
Lsection_abbrev:
	.byte	1                               ; Abbreviation Code
	.byte	17                              ; DW_TAG_compile_unit
	.byte	1                               ; DW_CHILDREN_yes
	.byte	114                             ; DW_AT_str_offsets_base
	.byte	23                              ; DW_FORM_sec_offset
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	2                               ; Abbreviation Code
	.byte	19                              ; DW_TAG_structure_type
	.byte	1                               ; DW_CHILDREN_yes
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	3                               ; Abbreviation Code
	.ascii	"\200\204\001"                  ; DW_TAG_APPLE_property
	.byte	0                               ; DW_CHILDREN_no
	.ascii	"\350\177"                      ; DW_AT_APPLE_property_name
	.byte	37                              ; DW_FORM_strx1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	4                               ; Abbreviation Code
	.ascii	"\200\204\001"                  ; DW_TAG_APPLE_property
	.byte	0                               ; DW_CHILDREN_no
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	5                               ; Abbreviation Code
	.ascii	"\200\204\001"                  ; DW_TAG_APPLE_property
	.byte	0                               ; DW_CHILDREN_no
	.ascii	"\350\177"                      ; DW_AT_APPLE_property_name
	.byte	14                              ; DW_FORM_strp
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	6                               ; Abbreviation Code
	.byte	13                              ; DW_TAG_member
	.byte	0                               ; DW_CHILDREN_no
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.ascii	"\355\177"                      ; DW_AT_APPLE_property
	.byte	19                              ; DW_FORM_ref4
	.ascii	"\355\177"                      ; DW_AT_APPLE_property
	.byte	19                              ; DW_FORM_ref4
	.ascii	"\355\177"                      ; DW_AT_APPLE_property
	.byte	19                              ; DW_FORM_ref4
	.ascii	"\355\177"                      ; DW_AT_APPLE_property
	.byte	19                              ; DW_FORM_ref4
	.ascii	"\355\177"                      ; DW_AT_APPLE_property
	.byte	19                              ; DW_FORM_ref4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	0                               ; EOM(3)
	.section	__DWARF,__debug_info,regular,debug
Lsection_info:
Lcu_begin0:
Lset0 = Ldebug_info_end0-Ldebug_info_start0 ; Length of Unit
	.long	Lset0
Ldebug_info_start0:
	.short	5                               ; DWARF version number
	.byte	1                               ; DWARF Unit Type
	.byte	8                               ; Address Size (in bytes)
Lset1 = Lsection_abbrev-Lsection_abbrev ; Offset Into Abbrev. Section
	.long	Lset1
	.byte	1                               ; Abbrev [1] DW_TAG_compile_unit
Lset2 = Lstr_offsets_base0-Lsection_str_off ; DW_AT_str_offsets_base
	.long	Lset2
	.byte	2                               ; Abbrev [2] DW_TAG_structure_type
	.byte	2                               ; DW_AT_name
	.byte	3                               ; Abbrev [3] DW_TAG_APPLE_property
	.byte	0                               ; DW_AT_APPLE_property_name
	.byte	4                               ; Abbrev [4] DW_TAG_APPLE_property
	.byte	5                               ; Abbrev [5] DW_TAG_APPLE_property
	.long	102                             ; DW_AT_APPLE_property_name
	.byte	6                               ; Abbrev [6] DW_TAG_member
	.byte	1                               ; DW_AT_name
	.long	19                              ; DW_AT_APPLE_property
	.long	21                              ; DW_AT_APPLE_property
	.long	17                              ; DW_AT_APPLE_property
	.long	0                               ; DW_AT_APPLE_property
	.long	22                              ; DW_AT_APPLE_property
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
Ldebug_info_end0:
	.section	__DWARF,__debug_str_offs,regular,debug
Lsection_str_off:
	.long	16                              ; Length of String Offsets Set
	.short	5
	.short	0
Lstr_offsets_base0:
	.section	__DWARF,__debug_str,regular,debug
Linfo_string:
	.asciz	"autoSynthProp"                 ; string offset=0
	.asciz	"_var"                          ; string offset=14
	.asciz	"Foo"                           ; string offset=19
	.section	__DWARF,__debug_str_offs,regular,debug
	.long	0
	.long	14
	.long	19
