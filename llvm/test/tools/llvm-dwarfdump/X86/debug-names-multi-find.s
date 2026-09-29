# RUN: llvm-mc -triple x86_64-pc-linux %s -filetype=obj -o %t
# RUN: llvm-dwarfdump -find=foo - <%t | FileCheck %s

# CHECK: 12: DW_TAG_subprogram
# CHECK-NEXT: DW_AT_name ("foo")
# CHECK-NEXT: DW_AT_external
# CHECK: 17: DW_TAG_subprogram
# CHECK-NEXT: DW_AT_name ("foo")
# CHECK-NEXT: DW_AT_external

	.section	.debug_str,"MS",@progbits,1
.Lstring_foo:
	.asciz	"foo"
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
.Ldie_foo2:
	.byte	2                       # Abbrev [2] DW_TAG_subprogram
	.long	.Lstring_foo            # DW_AT_name
                                        # DW_AT_external
	.byte	0                       # End Of Children Mark
.Lcu_end0:

	.section	.debug_names,"",@progbits
	.long	.Lnames_end0-.Lnames_start0 # Header: contribution length
.Lnames_start0:
	.short	5                       # Header: version
	.short	0                       # Header: padding
	.long	1                       # Header: compilation unit count
	.long	0                       # Header: local type unit count
	.long	0                       # Header: foreign type unit count
	.long	0                       # Header: bucket count
	.long	1                       # Header: name count
	.long	.Lnames_abbrev_end0-.Lnames_abbrev_start0 # Header: abbreviation table size
	.long	0                       # Header: augmentation length
	.long	.Lcu_begin0             # Compilation unit 0
	.long	.Lstring_foo            # String in Bucket 0: foo
	.long	.Lnames0-.Lnames_entries0 # Offset in Bucket 0
.Lnames_abbrev_start0:
	.byte	46                      # Abbrev code
	.byte	46                      # DW_TAG_subprogram
	.byte	3                       # DW_IDX_die_offset
	.byte	19                      # DW_FORM_ref4
	.byte	0                       # End of abbrev
	.byte	0                       # End of abbrev
	.byte	0                       # End of abbrev list
.Lnames_abbrev_end0:
.Lnames_entries0:
.Lnames0:
	.byte	46                      # Abbrev code
	.long	.Ldie_foo-.Lcu_begin0   # DW_IDX_die_offset
	.byte	46                      # Abbrev code
	.long	.Ldie_foo2-.Lcu_begin0  # DW_IDX_die_offset
	.long	0                       # End of list: foo
	.p2align	2
.Lnames_end0:
