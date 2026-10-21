# DWARF 2 encodes DW_AT_data_member_location as a location expression rather
# than a constant, and DW_AT_vtable_elem_location is an expression at every
# DWARF version. Both must survive placement in the artificial type unit.

# RUN: llvm-mc -triple x86_64-apple-darwin -filetype=obj %s -o %t.o
# RUN: llvm-dwarfdump --verify %t.o

# RUN: echo '---' > %t.map
# RUN: echo "triple:          'x86_64-apple-darwin'" >> %t.map
# RUN: echo 'objects:'  >> %t.map
# RUN: echo " -  filename: '%t.o'" >> %t.map
# RUN: echo '    symbols:' >> %t.map
# RUN: echo '      - { sym: __Z4keepP7Derived, objAddr: 0x0, binAddr: 0x10000, size: 0x1 }' >> %t.map
# RUN: echo '...' >> %t.map

# RUN: dsymutil --linker=parallel -y %t.map -f -o %t.parallel.dSYM
# RUN: llvm-dwarfdump --verify %t.parallel.dSYM
# RUN: llvm-dwarfdump -debug-info %t.parallel.dSYM \
# RUN:   | FileCheck %s --check-prefix=PARALLEL

# RUN: dsymutil --linker=classic -y %t.map -f -o %t.classic.dSYM
# RUN: llvm-dwarfdump --verify %t.classic.dSYM
# RUN: llvm-dwarfdump -debug-info %t.classic.dSYM \
# RUN:   | FileCheck %s --check-prefix=CLASSIC

# The parallel linker moves the deduplicated types into the artificial type unit
# and the classic linker keeps them in the compile unit, so the two emit them in
# different orders.

# PARALLEL: DW_AT_name{{.*}}"__artificial_type_unit"
# PARALLEL: DW_TAG_structure_type
# PARALLEL:   DW_AT_name{{.*}}"Base"
# PARALLEL:   DW_TAG_member
# PARALLEL:     DW_AT_name{{.*}}"first"
# PARALLEL:     DW_AT_data_member_location{{.*}}(DW_OP_plus_uconst 0x0)
# PARALLEL:   DW_TAG_member
# PARALLEL:     DW_AT_name{{.*}}"second"
# PARALLEL:     DW_AT_data_member_location{{.*}}(DW_OP_plus_uconst 0x4)
# PARALLEL: DW_TAG_structure_type
# PARALLEL:   DW_AT_name{{.*}}"Derived"
# PARALLEL:   DW_TAG_inheritance
# PARALLEL:     DW_AT_data_member_location{{.*}}(DW_OP_plus_uconst 0x8)
# PARALLEL:   DW_TAG_member
# PARALLEL:     DW_AT_name{{.*}}"third"
# PARALLEL:     DW_AT_data_member_location{{.*}}(DW_OP_plus_uconst 0x10)
# PARALLEL:   DW_TAG_subprogram
# PARALLEL:     DW_AT_name{{.*}}"virt"
# PARALLEL:     DW_AT_vtable_elem_location{{.*}}(DW_OP_constu 0x0)

# CLASSIC: DW_TAG_structure_type
# CLASSIC:   DW_AT_name{{.*}}"Derived"
# CLASSIC:   DW_TAG_inheritance
# CLASSIC:     DW_AT_data_member_location{{.*}}(DW_OP_plus_uconst 0x8)
# CLASSIC:   DW_TAG_member
# CLASSIC:     DW_AT_name{{.*}}"third"
# CLASSIC:     DW_AT_data_member_location{{.*}}(DW_OP_plus_uconst 0x10)
# CLASSIC:   DW_TAG_subprogram
# CLASSIC:     DW_AT_name{{.*}}"virt"
# CLASSIC:     DW_AT_vtable_elem_location{{.*}}(DW_OP_constu 0x0)
# CLASSIC: DW_TAG_structure_type
# CLASSIC:   DW_AT_name{{.*}}"Base"
# CLASSIC:   DW_TAG_member
# CLASSIC:     DW_AT_name{{.*}}"first"
# CLASSIC:     DW_AT_data_member_location{{.*}}(DW_OP_plus_uconst 0x0)
# CLASSIC:   DW_TAG_member
# CLASSIC:     DW_AT_name{{.*}}"second"
# CLASSIC:     DW_AT_data_member_location{{.*}}(DW_OP_plus_uconst 0x4)

	.section	__TEXT,__text,regular,pure_instructions
	.globl	__Z4keepP7Derived
__Z4keepP7Derived:
Lfunc_begin0:
	retq
Lfunc_end0:

	.section	__DWARF,__debug_abbrev,regular,debug
Lsection_abbrev:
	.byte	1                       ## Abbreviation Code
	.byte	17                      ## DW_TAG_compile_unit
	.byte	1                       ## DW_CHILDREN_yes
	.byte	37                      ## DW_AT_producer
	.byte	8                       ## DW_FORM_string
	.byte	19                      ## DW_AT_language
	.byte	5                       ## DW_FORM_data2
	.byte	3                       ## DW_AT_name
	.byte	8                       ## DW_FORM_string
	.byte	0, 0

	.byte	2                       ## Abbreviation Code
	.byte	46                      ## DW_TAG_subprogram
	.byte	1                       ## DW_CHILDREN_yes
	.byte	3                       ## DW_AT_name
	.byte	8                       ## DW_FORM_string
	.byte	0x87, 0x40              ## DW_AT_MIPS_linkage_name (0x2007)
	.byte	8                       ## DW_FORM_string
	.byte	17                      ## DW_AT_low_pc
	.byte	1                       ## DW_FORM_addr
	.byte	18                      ## DW_AT_high_pc
	.byte	1                       ## DW_FORM_addr
	.byte	63                      ## DW_AT_external
	.byte	12                      ## DW_FORM_flag
	.byte	0, 0

	.byte	3                       ## Abbreviation Code
	.byte	5                       ## DW_TAG_formal_parameter
	.byte	0                       ## DW_CHILDREN_no
	.byte	73                      ## DW_AT_type
	.byte	19                      ## DW_FORM_ref4
	.byte	0, 0

	.byte	4                       ## Abbreviation Code
	.byte	19                      ## DW_TAG_structure_type
	.byte	1                       ## DW_CHILDREN_yes
	.byte	3                       ## DW_AT_name
	.byte	8                       ## DW_FORM_string
	.byte	11                      ## DW_AT_byte_size
	.byte	11                      ## DW_FORM_data1
	.byte	0, 0

	.byte	5                       ## Abbreviation Code
	.byte	28                      ## DW_TAG_inheritance
	.byte	0                       ## DW_CHILDREN_no
	.byte	73                      ## DW_AT_type
	.byte	19                      ## DW_FORM_ref4
	.byte	56                      ## DW_AT_data_member_location
	.byte	10                      ## DW_FORM_block1
	.byte	0, 0

	.byte	6                       ## Abbreviation Code
	.byte	13                      ## DW_TAG_member
	.byte	0                       ## DW_CHILDREN_no
	.byte	3                       ## DW_AT_name
	.byte	8                       ## DW_FORM_string
	.byte	73                      ## DW_AT_type
	.byte	19                      ## DW_FORM_ref4
	.byte	56                      ## DW_AT_data_member_location
	.byte	10                      ## DW_FORM_block1
	.byte	0, 0

	.byte	7                       ## Abbreviation Code
	.byte	46                      ## DW_TAG_subprogram
	.byte	0                       ## DW_CHILDREN_no
	.byte	3                       ## DW_AT_name
	.byte	8                       ## DW_FORM_string
	.byte	0x87, 0x40              ## DW_AT_MIPS_linkage_name (0x2007)
	.byte	8                       ## DW_FORM_string
	.byte	76                      ## DW_AT_virtuality
	.byte	11                      ## DW_FORM_data1
	.byte	77                      ## DW_AT_vtable_elem_location
	.byte	10                      ## DW_FORM_block1
	.byte	60                      ## DW_AT_declaration
	.byte	12                      ## DW_FORM_flag
	.byte	0, 0

	.byte	8                       ## Abbreviation Code
	.byte	36                      ## DW_TAG_base_type
	.byte	0                       ## DW_CHILDREN_no
	.byte	3                       ## DW_AT_name
	.byte	8                       ## DW_FORM_string
	.byte	11                      ## DW_AT_byte_size
	.byte	11                      ## DW_FORM_data1
	.byte	62                      ## DW_AT_encoding
	.byte	11                      ## DW_FORM_data1
	.byte	0, 0

	.byte	9                       ## Abbreviation Code
	.byte	15                      ## DW_TAG_pointer_type
	.byte	0                       ## DW_CHILDREN_no
	.byte	73                      ## DW_AT_type
	.byte	19                      ## DW_FORM_ref4
	.byte	0, 0

	.byte	0                       ## EOM(3)

	.section	__DWARF,__debug_info,regular,debug
Lsection_info:
	.long	Lcu_end - Lcu_start     ## Length of Unit
Lcu_start:
	.short	2                       ## DWARF version number
	.long	0                       ## Offset Into Abbrev. Section
	.byte	8                       ## Address Size (in bytes)

	.byte	1                       ## Abbrev [1] DW_TAG_compile_unit
	.asciz	"hand-written"          ## DW_AT_producer
	.short	0x0004                  ## DW_AT_language (DW_LANG_C_plus_plus)
	.asciz	"dwarf2-member-location.cpp" ## DW_AT_name

	.byte	2                       ## Abbrev [2] DW_TAG_subprogram
	.asciz	"keep"                  ## DW_AT_name
	.asciz	"__Z4keepP7Derived"     ## DW_AT_MIPS_linkage_name
	.quad	Lfunc_begin0            ## DW_AT_low_pc
	.quad	Lfunc_end0              ## DW_AT_high_pc
	.byte	1                       ## DW_AT_external

	.byte	3                       ## Abbrev [3] DW_TAG_formal_parameter
	.long	Lderived_ptr - Lsection_info ## DW_AT_type

	.byte	0                       ## End Of Children Mark (keep)

Lderived:
	.byte	4                       ## Abbrev [4] DW_TAG_structure_type
	.asciz	"Derived"               ## DW_AT_name
	.byte	0x18                    ## DW_AT_byte_size

	.byte	5                       ## Abbrev [5] DW_TAG_inheritance
	.long	Lbase - Lsection_info   ## DW_AT_type
	.byte	2                       ## DW_AT_data_member_location length
	.byte	0x23                    ##   DW_OP_plus_uconst
	.byte	0x08                    ##   0x8

	.byte	6                       ## Abbrev [6] DW_TAG_member
	.asciz	"third"                 ## DW_AT_name
	.long	Lint - Lsection_info    ## DW_AT_type
	.byte	2                       ## DW_AT_data_member_location length
	.byte	0x23                    ##   DW_OP_plus_uconst
	.byte	0x10                    ##   0x10

	.byte	7                       ## Abbrev [7] DW_TAG_subprogram
	.asciz	"virt"                  ## DW_AT_name
	.asciz	"__ZN7Derived4virtEv"   ## DW_AT_MIPS_linkage_name
	.byte	1                       ## DW_AT_virtuality (DW_VIRTUALITY_virtual)
	.byte	2                       ## DW_AT_vtable_elem_location length
	.byte	0x10                    ##   DW_OP_constu
	.byte	0x00                    ##   0x0
	.byte	1                       ## DW_AT_declaration

	.byte	0                       ## End Of Children Mark (Derived)

Lbase:
	.byte	4                       ## Abbrev [4] DW_TAG_structure_type
	.asciz	"Base"                  ## DW_AT_name
	.byte	0x08                    ## DW_AT_byte_size

	.byte	6                       ## Abbrev [6] DW_TAG_member
	.asciz	"first"                 ## DW_AT_name
	.long	Lint - Lsection_info    ## DW_AT_type
	.byte	2                       ## DW_AT_data_member_location length
	.byte	0x23                    ##   DW_OP_plus_uconst
	.byte	0x00                    ##   0x0

	.byte	6                       ## Abbrev [6] DW_TAG_member
	.asciz	"second"                ## DW_AT_name
	.long	Lint - Lsection_info    ## DW_AT_type
	.byte	2                       ## DW_AT_data_member_location length
	.byte	0x23                    ##   DW_OP_plus_uconst
	.byte	0x04                    ##   0x4

	.byte	0                       ## End Of Children Mark (Base)

Lint:
	.byte	8                       ## Abbrev [8] DW_TAG_base_type
	.asciz	"int"                   ## DW_AT_name
	.byte	4                       ## DW_AT_byte_size
	.byte	5                       ## DW_AT_encoding (DW_ATE_signed)

Lderived_ptr:
	.byte	9                       ## Abbrev [9] DW_TAG_pointer_type
	.long	Lderived - Lsection_info ## DW_AT_type

	.byte	0                       ## End Of Children Mark (CU)
Lcu_end:
