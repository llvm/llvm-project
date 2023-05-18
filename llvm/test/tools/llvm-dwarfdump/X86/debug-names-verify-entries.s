# RUN: llvm-mc -triple x86_64-pc-linux %s -filetype=obj -o - | not llvm-dwarfdump -verify - | FileCheck %s

# CHECK: error: Name Index @ 0x0: Unable to get string associated with name 1.
# CHECK: error: Name Index @ 0x0: Entry @ 0x73 contains an invalid CU index (47).
# CHECK: error: Name Index @ 0x0: Entry @ 0x79 references a non-existing DIE @ 0x3fa.
# CHECK: error: Name Index @ 0x0: Entry @ 0x85: mismatched CU of DIE @ 0x30: index - 0x0; debug_info - 0x1e.
# CHECK: error: Name Index @ 0x0: Entry @ 0x8b: mismatched Tag of DIE @ 0x17: index - DW_TAG_subprogram; debug_info - DW_TAG_variable.
# CHECK: error: Name Index @ 0x0: Entry @ 0x91: mismatched Name of DIE @ 0x35: index - foo; debug_info - bar, _Z3bar.
# CHECK: error: Name Index @ 0x0: Name 2 (foo): Incorrectly terminated entry list.
# CHECK: error: Name Index @ 0x0: Name 3 (bar) is not associated with any entries.
# CHECK: error: Name Index @ 0x0: Entry @ 0x69: mismatched Name of DIE @ 0x1c: index - (pseudonymous namespace); debug_info - (anonymous namespace).

	.section	.debug_str,"MS",@progbits,1
.Lstring_foo:
	.asciz	"foo"
.Lstring_bar:
	.asciz	"bar"
.Lstring_bar_mangled:
	.asciz	"_Z3bar"
.Lstring_pseudo_namespace:
	.asciz	"(pseudonymous namespace)"
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
	.byte	2                       # Abbreviation Code
	.byte	46                      # DW_TAG_subprogram
	.byte	0                       # DW_CHILDREN_no
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	63                      # DW_AT_external
	.byte	25                      # DW_FORM_flag_present
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	3                       # Abbreviation Code
	.byte	52                      # DW_TAG_variable
	.byte	0                       # DW_CHILDREN_no
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	4                       # Abbreviation Code
	.byte	57                      # DW_TAG_namespace
	.byte	0                       # DW_CHILDREN_no
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	5                       # Abbreviation Code
	.byte	52                      # DW_TAG_variable
	.byte	0                       # DW_CHILDREN_no
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	110                     # DW_AT_linkage_name
	.byte	14                      # DW_FORM_strp
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
.Ldie_foo:
	.byte	2                       # Abbrev [2] DW_TAG_subprogram
	.long	.Lstring_foo            # DW_AT_name
                                        # DW_AT_external
.Ldie_foo_var:
	.byte	3                       # Abbrev [3] DW_TAG_variable
	.long	.Lstring_foo            # DW_AT_name
.Ldie_namespace:
	.byte	4                       # Abbrev [3] DW_TAG_namespace
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
.Ldie_foo2:
	.byte	2                       # Abbrev [2] DW_TAG_subprogram
	.long	.Lstring_foo            # DW_AT_name
                                        # DW_AT_external
.Ldie_bar_linkage:
	.byte	5                       # Abbrev [2] DW_TAG_variable
	.long	.Lstring_bar            # DW_AT_name
	.long	.Lstring_bar_mangled    # DW_AT_linkage_name
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
	.long	0                       # Header: bucket count
	.long	4                       # Header: name count
	.long	.Lnames_abbrev_end0-.Lnames_abbrev_start0 # Header: abbreviation table size
	.long	0                       # Header: augmentation length
	.long	.Lcu_begin0             # Compilation unit 0
	.long	.Lcu_begin1             # Compilation unit 1
	.long	.Lstring_foo+1000       # String 1: <broken>
	.long	.Lstring_foo            # String 2: foo
	.long	.Lstring_bar            # String 3: bar
	.long	.Lstring_pseudo_namespace # String 4: (pseudonymous namespace)
	.long	.Lnames0-.Lnames_entries0 # Offset 1
	.long	.Lnames0-.Lnames_entries0 # Offset 2
	.long	.Lnames1-.Lnames_entries0 # Offset 3
	.long	.Lnames2-.Lnames_entries0 # Offset 4
.Lnames_abbrev_start0:
	.byte	46                      # Abbrev code
	.byte	46                      # DW_TAG_subprogram
	.byte	1                       # DW_IDX_compile_unit
	.byte	11                      # DW_FORM_data1
	.byte	3                       # DW_IDX_die_offset
	.byte	19                      # DW_FORM_ref4
	.byte	0                       # End of abbrev
	.byte	0                       # End of abbrev
	.byte	57                      # Abbrev code
	.byte	57                      # DW_TAG_namespace
	.byte	1                       # DW_IDX_compile_unit
	.byte	11                      # DW_FORM_data1
	.byte	3                       # DW_IDX_die_offset
	.byte	19                      # DW_FORM_ref4
	.byte	0                       # End of abbrev
	.byte	0                       # End of abbrev
	.byte	52                      # Abbrev code
	.byte	52                      # DW_TAG_variable
	.byte	1                       # DW_IDX_compile_unit
	.byte	11                      # DW_FORM_data1
	.byte	3                       # DW_IDX_die_offset
	.byte	19                      # DW_FORM_ref4
	.byte	0                       # End of abbrev
	.byte	0                       # End of abbrev
	.byte	0                       # End of abbrev list
.Lnames_abbrev_end0:
.Lnames_entries0:
.Lnames1:
	.long	0                       # End of list: bar
.Lnames2:
	.byte	57                      # Abbrev code
	.byte	0                       # DW_IDX_compile_unit
	.long	.Ldie_namespace-.Lcu_begin0 # DW_IDX_die_offset
	.long	0                       # End of list: (pseudonymous namespace)
.Lnames0:
	.byte	46                      # Abbrev code
	.byte	47                      # DW_IDX_compile_unit
	.long	.Ldie_foo-.Lcu_begin0   # DW_IDX_die_offset
	.byte	46                      # Abbrev code
	.byte	0                       # DW_IDX_compile_unit
	.long	.Ldie_foo-.Lcu_begin0+1000 # DW_IDX_die_offset
	.byte	46                      # Abbrev code
	.byte	0                       # DW_IDX_compile_unit
	.long	.Ldie_foo-.Lcu_begin0   # DW_IDX_die_offset
	.byte	46                      # Abbrev code
	.byte	0                       # DW_IDX_compile_unit
	.long	.Ldie_foo2-.Lcu_begin0  # DW_IDX_die_offset
	.byte	46                      # Abbrev code
	.byte	0                       # DW_IDX_compile_unit
	.long	.Ldie_foo_var-.Lcu_begin0 # DW_IDX_die_offset
	.byte	52                      # Abbrev code
	.byte	1                       # DW_IDX_compile_unit
	.long	.Ldie_bar_linkage-.Lcu_begin1 # DW_IDX_die_offset
	#.long	0                       # End of list deliberately missing
.Lnames_end0:
