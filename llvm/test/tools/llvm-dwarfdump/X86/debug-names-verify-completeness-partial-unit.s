# The .debug_names completeness check treats the root of a DW_TAG_partial_unit
# the way it already treats the root of a DW_TAG_compile_unit: its DW_AT_name is
# the path of a source file rather than a named subprogram, label, variable,
# type or namespace, so the index owes it no entry.
#
# One index covers two units. U01 is the DW_TAG_partial_unit that dwz produces
# and U02 is an ordinary compilation unit, the control. Neither root appears in
# the index and both subprograms do.

# RUN: llvm-mc -triple x86_64-pc-linux %s -filetype=obj -o %t.o
# RUN: llvm-dwarfdump --verify %t.o | FileCheck %s --implicit-check-not=error:

# CHECK: No errors.

# Excluding the root does not excuse the rest of the unit. A subprogram under
# the partial unit root that the index leaves out is still reported, and the
# aggregate count pins it as the only error, so the root is still not demanded.

# RUN: llvm-mc -triple x86_64-pc-linux --defsym UNINDEXED=1 %s -filetype=obj -o %t.unindexed.o
# RUN: not llvm-dwarfdump --verify %t.unindexed.o | FileCheck %s --check-prefix=UNINDEXED \
# RUN:   --implicit-check-not="dwz-common.h missing"

# UNINDEXED:      error: Name Index @ 0x0: Entry for DIE @ {{.*}} (DW_TAG_subprogram) with name unindexed missing.
# UNINDEXED:      error: Aggregated error counts:
# UNINDEXED-NEXT: error: Name Index DIE entry missing name occurred 1 time(s).

	.section	.debug_str,"MS",@progbits,1
.Lstring_producer:
	.asciz	"Hand-written DWARF"
.Lstring_dwz_common:
	.asciz	"dwz-common.h"
.Lstring_main:
	.asciz	"main.c"
.Lstring_foo:
	.asciz	"foo"
.Lstring_bar:
	.asciz	"bar"
.Lstring_unindexed:
	.asciz	"unindexed"

	.section	.debug_abbrev,"",@progbits
.Lsection_abbrev:
	.byte	1                       # Abbreviation Code
	.byte	60                      # DW_TAG_partial_unit
	.byte	1                       # DW_CHILDREN_yes
	.byte	37                      # DW_AT_producer
	.byte	14                      # DW_FORM_strp
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	19                      # DW_AT_language
	.byte	5                       # DW_FORM_data2
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)

	.byte	2                       # Abbreviation Code
	.byte	46                      # DW_TAG_subprogram
	.byte	0                       # DW_CHILDREN_no
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	17                      # DW_AT_low_pc
	.byte	1                       # DW_FORM_addr
	.byte	18                      # DW_AT_high_pc
	.byte	6                       # DW_FORM_data4
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)

	.byte	3                       # Abbreviation Code
	.byte	17                      # DW_TAG_compile_unit
	.byte	1                       # DW_CHILDREN_yes
	.byte	37                      # DW_AT_producer
	.byte	14                      # DW_FORM_strp
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	19                      # DW_AT_language
	.byte	5                       # DW_FORM_data2
	.byte	17                      # DW_AT_low_pc
	.byte	1                       # DW_FORM_addr
	.byte	18                      # DW_AT_high_pc
	.byte	6                       # DW_FORM_data4
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)

	.byte	0                       # EOM(3)

	.section	.debug_info,"",@progbits
# U01, the partial unit. Its root carries no address range, which is the shape
# dwz emits: a partial unit holds no code of its own.
.Lcu_begin0:
	.long	.Lcu_end0-.Lcu_start0   # Length of Unit
.Lcu_start0:
	.short	5                       # DWARF version number
	.byte	3                       # DW_UT_partial
	.byte	8                       # Address Size (in bytes)
	.long	.Lsection_abbrev        # Offset Into Abbrev. Section
	.byte	1                       # Abbrev [1] DW_TAG_partial_unit
	.long	.Lstring_producer       # DW_AT_producer
	.long	.Lstring_dwz_common     # DW_AT_name
	.short	12                      # DW_AT_language
.Ldie_foo:
	.byte	2                       # Abbrev [2] DW_TAG_subprogram
	.long	.Lstring_foo            # DW_AT_name
	.quad	0x1000                  # DW_AT_low_pc
	.long	0x10                    # DW_AT_high_pc
.ifdef UNINDEXED
.Ldie_unindexed:
	.byte	2                       # Abbrev [2] DW_TAG_subprogram
	.long	.Lstring_unindexed      # DW_AT_name
	.quad	0x1010                  # DW_AT_low_pc
	.long	0x10                    # DW_AT_high_pc
.endif
	.byte	0                       # End Of Children Mark
.Lcu_end0:

# U02, the control.
.Lcu_begin1:
	.long	.Lcu_end1-.Lcu_start1   # Length of Unit
.Lcu_start1:
	.short	5                       # DWARF version number
	.byte	1                       # DW_UT_compile
	.byte	8                       # Address Size (in bytes)
	.long	.Lsection_abbrev        # Offset Into Abbrev. Section
	.byte	3                       # Abbrev [3] DW_TAG_compile_unit
	.long	.Lstring_producer       # DW_AT_producer
	.long	.Lstring_main           # DW_AT_name
	.short	12                      # DW_AT_language
	.quad	0x2000                  # DW_AT_low_pc
	.long	0x100                   # DW_AT_high_pc
.Ldie_bar:
	.byte	2                       # Abbrev [2] DW_TAG_subprogram
	.long	.Lstring_bar            # DW_AT_name
	.quad	0x2000                  # DW_AT_low_pc
	.long	0x10                    # DW_AT_high_pc
	.byte	0                       # End Of Children Mark
.Lcu_end1:

	.section	.debug_names,"",@progbits
	.long	.Lnames_end0-.Lnames_start0 # Header: contribution length
.Lnames_start0:
	.short	5                       # Header: version
	.short	0                       # Header: padding
	.long	2                       # Header: compilation unit count
	.long	0                       # Header: local type unit count
	.long	0                       # Header: foreign type unit count
	.long	2                       # Header: bucket count
	.long	2                       # Header: name count
	.long	.Lnames_abbrev_end0-.Lnames_abbrev_start0 # Header: abbreviation table size
	.long	0                       # Header: augmentation length
	.long	.Lcu_begin0             # Compilation unit 0
	.long	.Lcu_begin1             # Compilation unit 1
	.long	1                       # Bucket 0
	.long	2                       # Bucket 1
	.long	193487034               # Hash in Bucket 0: bar
	.long	193491849               # Hash in Bucket 1: foo
	.long	.Lstring_bar            # String in Bucket 0: bar
	.long	.Lstring_foo            # String in Bucket 1: foo
	.long	.Lnames_bar-.Lnames_entries0 # Offset in Bucket 0
	.long	.Lnames_foo-.Lnames_entries0 # Offset in Bucket 1
.Lnames_abbrev_start0:
	.byte	46                      # Abbrev code
	.byte	46                      # DW_TAG_subprogram
	.byte	1                       # DW_IDX_compile_unit
	.byte	11                      # DW_FORM_data1
	.byte	3                       # DW_IDX_die_offset
	.byte	19                      # DW_FORM_ref4
	.byte	0                       # End of abbrev
	.byte	0                       # End of abbrev
	.byte	0                       # End of abbrev list
.Lnames_abbrev_end0:
.Lnames_entries0:
.Lnames_bar:
	.byte	46                      # Abbrev code
	.byte	1                       # DW_IDX_compile_unit
	.long	.Ldie_bar-.Lcu_begin1   # DW_IDX_die_offset
	.long	0                       # End of list: bar
.Lnames_foo:
	.byte	46                      # Abbrev code
	.byte	0                       # DW_IDX_compile_unit
	.long	.Ldie_foo-.Lcu_begin0   # DW_IDX_die_offset
	.long	0                       # End of list: foo
	.p2align	2
.Lnames_end0:
