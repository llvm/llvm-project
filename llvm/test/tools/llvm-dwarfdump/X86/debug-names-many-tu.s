# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t
# RUN: llvm-dwarfdump -debug-names %t | FileCheck %s

# CHECK:      LocalTU[0]: 0x0000000d
# CHECK-NEXT: LocalTU[1]: 0x00000028
# CHECK:      ForeignTU[0]: 0x0011223344556677
# CHECK-NEXT: ForeignTU[1]: 0x1122334455667788

	.section	.debug_abbrev,"",@progbits
	.byte	1                       # Abbreviation Code
	.byte	17                      # DW_TAG_compile_unit
	.byte	0                       # DW_CHILDREN_no
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	2                       # Abbreviation Code
	.byte	19                      # DW_TAG_structure_type
	.byte	0                       # DW_CHILDREN_no
	.byte	3                       # DW_AT_name
	.byte	8                       # DW_FORM_string
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	0

	.section	.debug_names,"",@progbits
	.long	.Lnames_end0-.Lnames_start0 # Header: unit length
.Lnames_start0:
	.short	5                       # Header: version
	.short	0                       # Header: padding
	.long	1                       # Header: compilation unit count
	.long	2                       # Header: local type unit count
	.long	2                       # Header: foreign type unit count
	.long	0                       # Header: bucket count
	.long	2                       # Header: name count
	.long	.Lnames_abbrev_end0-.Lnames_abbrev_start0 # Header: abbreviation table size
	.long	0                       # Header: augmentation string size
	.long	.Lcu_begin0             # Compilation unit 0
	.long	.Ltu_begin0             # Local TU 0
	.long	.Ltu_begin1             # Local TU 1
	.quad	0x0011223344556677      # Foreign TU 0
	.quad	0x1122334455667788      # Foreign TU 1
	.long	.Lstring0               # String in Bucket 0: A
	.long	.Lstring1               # String in Bucket 1: B
	.long	.Lnames0-.Lnames_entries0 # Offset in Bucket 0
	.long	.Lnames1-.Lnames_entries0 # Offset in Bucket 1

.Lnames_abbrev_start0:
	.byte	1                       # Abbrev code
	.byte	19                      # DW_TAG_structure_type
	.byte	3                       # DW_IDX_die_offset
	.byte	19                      # DW_FORM_ref4
	.byte	0                       # End of abbrev
	.byte	0                       # End of abbrev
	.byte	0                       # End of abbrev list
.Lnames_abbrev_end0:
.Lnames_entries0:
.Lnames0:
	.byte	1                       # Abbreviation code
	.long	.Ltu_die0-.Lcu_begin0   # DW_IDX_die_offset
	.long	0                       # End of list: A
.Lnames1:
	.byte	1                       # Abbreviation code
	.long	.Ltu_die1-.Lcu_begin0   # DW_IDX_die_offset
	.long	0                       # End of list: B
	.p2align 2
.Lnames_end0:

	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Lcu_end0-.Lcu_begin0-4 # Length of Unit
	.short	5                       # DWARF version number
	.byte	1                       # DW_UT_compile
	.byte	8                       # Address Size (in bytes)
	.long	.debug_abbrev           # Offset Into Abbrev. Section
	.byte	1                       # Abbrev [1] DW_TAG_compile_unit
.Lcu_end0:

.Ltu_begin0:
	.long	.Ltu_end0-.Ltu_begin0-4 # Length of Unit
	.short	5                       # DWARF version number
	.byte	2                       # DW_UT_type
	.byte	8                       # Address Size (in bytes)
	.long	.debug_abbrev           # Offset Into Abbrev. Section
	.quad	0x0011223344556677      # Type Signature
	.long	.Ltu_die0-.Ltu_begin0   # Type Offset
.Ltu_die0:
	.byte	2                       # Abbrev [2] DW_TAG_structure_type
	.asciz "A"
.Ltu_end0:

.Ltu_begin1:
	.long	.Ltu_end1-.Ltu_begin1-4 # Length of Unit
	.short	5                       # DWARF version number
	.byte	2                       # DW_UT_type
	.byte	8                       # Address Size (in bytes)
	.long	.debug_abbrev           # Offset Into Abbrev. Section
	.quad	0x1122334455667788      # Type Signature
	.long	.Ltu_die1-.Ltu_begin1   # Type Offset
.Ltu_die1:
	.byte	2                       # Abbrev [1] 0xc:0x48 DW_TAG_structure_type
	.asciz "B"
.Ltu_end1:

	.section	.debug_str,"MS",@progbits,1
.Lstring0:
	.asciz	"A"
.Lstring1:
	.asciz	"B"
