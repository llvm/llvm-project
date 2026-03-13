# RUN: llvm-mc -triple x86_64-pc-linux %s -filetype=obj | \
# RUN:   not llvm-dwarfdump -verify - | FileCheck %s

# CHECK: error: NameIndex @ 0x0: Abbreviation 0x2: DW_IDX_compile_unit uses an unexpected form DW_FORM_ref1 (expected form class constant).
# CHECK: error: NameIndex @ 0x0: Abbreviation 0x2: DW_IDX_type_unit uses an unexpected form DW_FORM_ref1 (expected form class constant).
# CHECK: error: NameIndex @ 0x0: Abbreviation 0x2: DW_IDX_type_hash uses an unexpected form DW_FORM_data4 (should be DW_FORM_data8).
# CHECK: warning: NameIndex @ 0x0: Abbreviation 0x2 contains an unknown index attribute: DW_IDX_unknown_2020.
# CHECK: error: NameIndex @ 0x0: Abbreviation 0x4 contains multiple DW_IDX_die_offset attributes.
# CHECK: NameIndex @ 0x0: Abbreviation 0x1: DW_IDX_die_offset uses an unknown form: DW_FORM_unknown_1fff.
# CHECK: warning: NameIndex @ 0x0: Abbreviation 0x3 references an unknown tag: DW_TAG_unknown_8080.
# CHECK: error: NameIndex @ 0x0: Abbreviation 0x5 has no DW_IDX_die_offset attribute.
# CHECK: error: NameIndex @ 0x55: Indexing multiple compile units and abbreviation 0x1 has no DW_IDX_compile_unit or DW_IDX_type_unit attribute.

	.section	.debug_str,"MS",@progbits,1
.Lstring_producer:
	.asciz	"Hand-written dwarf"

	.section	.debug_abbrev,"",@progbits
.Lsection_abbrev:
	.byte	1                       # Abbreviation Code
	.byte	17                      # DW_TAG_compile_unit
	.byte	1                       # DW_CHILDREN_yes
	.byte	37                      # DW_AT_producer
	.byte	14                      # DW_FORM_strp
	.byte	19                      # DW_AT_language
	.byte	5                       # DW_FORM_data2
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	0                       # EOM(3)

	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Lcu_end0-.Lcu_start0   # Length of Unit
.Lcu_start0:
	.short	4                       # DWARF version number
	.long	.Lsection_abbrev        # Offset Into Abbrev. Section
	.byte	8                       # Address Size (in bytes)
	.byte	1                       # Abbrev [1] DW_TAG_compile_unit
	.long	.Lstring_producer       # DW_AT_producer
	.short	12                      # DW_AT_language
	.byte	0                       # End Of Children Mark
.Lcu_end0:

.Lcu_begin1:
	.long	.Lcu_end1-.Lcu_start1   # Length of Unit
.Lcu_start1:
	.short	4                       # DWARF version number
	.long	.Lsection_abbrev        # Offset Into Abbrev. Section
	.byte	8                       # Address Size (in bytes)
	.byte	1                       # Abbrev [1] DW_TAG_compile_unit
	.long	.Lstring_producer       # DW_AT_producer
	.short	12                      # DW_AT_language
	.byte	0                       # End Of Children Mark
.Lcu_end1:

.Lcu_begin2:
	.long	.Lcu_end2-.Lcu_start2   # Length of Unit
.Lcu_start2:
	.short	4                       # DWARF version number
	.long	.Lsection_abbrev        # Offset Into Abbrev. Section
	.byte	8                       # Address Size (in bytes)
	.byte	1                       # Abbrev [1] DW_TAG_compile_unit
	.long	.Lstring_producer       # DW_AT_producer
	.short	12                      # DW_AT_language
	.byte	0                       # End Of Children Mark
.Lcu_end2:

	.section	.debug_names,"",@progbits
	.long	.Lnames_end0-.Lnames_start0 # Header: contribution length
.Lnames_start0:
	.short	5                       # Header: version
	.short	0                       # Header: padding
	.long	1                       # Header: compilation unit count
	.long	0                       # Header: local type unit count
	.long	0                       # Header: foreign type unit count
	.long	0                       # Header: bucket count
	.long	0                       # Header: name count
	.long	.Lnames_abbrev_end0-.Lnames_abbrev_start0 # Header: abbreviation table size
	.long	0                       # Header: augmentation length
	.long	.Lcu_begin0             # Compilation unit 0
.Lnames_abbrev_start0:
	.byte	1                       # Abbrev code
	.byte	46                      # DW_TAG_subprogram
	.byte	3                       # DW_IDX_die_offset
	.uleb128 0x1fff                 # DW_FORM_unknown_1fff
	.byte	0                       # End of abbrev
	.byte	0                       # End of abbrev
	.byte	2                       # Abbrev code
	.byte	46                      # DW_TAG_subprogram
	.byte	1                       # DW_IDX_compile_unit
	.byte   17                      # DW_FORM_ref1
	.byte	2                       # DW_IDX_type_unit
	.byte   17                      # DW_FORM_ref1
	.byte	2                       # DW_IDX_type_unit
	.byte   5                       # DW_FORM_data2
	.byte	5                       # DW_IDX_type_hash
	.byte   6                       # DW_FORM_data4
	.uleb128 0x2020                 # DW_IDX_unknown_2020
	.byte   6                       # DW_FORM_data4
	.byte	0                       # End of abbrev
	.byte	0                       # End of abbrev
	.byte	3                       # Abbrev code
	.uleb128 0x8080                 # DW_TAG_unknown_8080
	.byte	3                       # DW_IDX_die_offset
	.byte   17                      # DW_FORM_ref1
	.byte	0                       # End of abbrev
	.byte	0                       # End of abbrev
	.byte	4                       # Abbrev code
	.byte	46                      # DW_TAG_subprogram
	.byte	3                       # DW_IDX_die_offset
	.byte   17                      # DW_FORM_ref1
	.byte	3                       # DW_IDX_die_offset
	.byte   17                      # DW_FORM_ref1
	.byte	0                       # End of abbrev
	.byte	0                       # End of abbrev
	.byte	5                       # Abbrev code
	.byte	46                      # DW_TAG_subprogram
	.byte	4                       # DW_IDX_parent
	.byte   5                       # DW_FORM_data2
	.byte	0                       # End of abbrev
	.byte	0                       # End of abbrev
	.byte	0                       # End of abbrev list
.Lnames_abbrev_end0:
.Lnames_end0:

	.long	.Lnames_end1-.Lnames_start1 # Header: contribution length
.Lnames_start1:
	.short	5                       # Header: version
	.short	0                       # Header: padding
	.long	2                       # Header: compilation unit count
	.long	0                       # Header: local type unit count
	.long	0                       # Header: foreign type unit count
	.long	0                       # Header: bucket count
	.long	0                       # Header: name count
	.long	.Lnames_abbrev_end1-.Lnames_abbrev_start1 # Header: abbreviation table size
	.long	0                       # Header: augmentation length
	.long	.Lcu_begin1             # Compilation unit 0
	.long	.Lcu_begin2             # Compilation unit 1
.Lnames_abbrev_start1:
	.byte	1                       # Abbrev code
	.byte	46                      # DW_TAG_subprogram
	.byte	3                       # DW_IDX_die_offset
	.byte   17                      # DW_FORM_ref1
	.byte	0                       # End of abbrev
	.byte	0                       # End of abbrev
	.byte	0                       # End of abbrev list
.Lnames_abbrev_end1:
.Lnames_end1:
