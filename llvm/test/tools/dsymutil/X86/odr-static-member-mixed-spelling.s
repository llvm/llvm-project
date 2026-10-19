## DWARF 4 producers spell a C++ static data member as a DW_TAG_member with
## DW_AT_declaration, DWARF 5 producers spell it as a DW_TAG_variable, so both
## spellings show up whenever objects built at different DWARF versions, or by
## different producers, are linked together.
##
## The parallel linker derives the name of a data member from its position among
## the record's data members. A static data member has no storage in the record
## and therefore no such position, and it has to derive the same name from either
## spelling. Otherwise units disagreeing on the spelling derive different names
## for the same entity, and the deduplicated record gains a member per name.
##
## The hand-written DWARF below describes
##   struct S {
##     static const int kMask = 1;
##     int field;
##   };
## referenced from useA() in a.cpp (DWARF 4) and useB() in b.cpp (DWARF 5).

# RUN: llvm-mc -triple x86_64-apple-darwin -filetype=obj %s -o %t.o
# RUN: llvm-dwarfdump --verify %t.o

# RUN: echo '---' > %t.map
# RUN: echo "triple:          'x86_64-apple-darwin'" >> %t.map
# RUN: echo 'objects:'  >> %t.map
# RUN: echo " -  filename: '%t.o'" >> %t.map
# RUN: echo '    symbols:' >> %t.map
# RUN: echo '      - { sym: __Z4useAP1S, objAddr: 0x0, binAddr: 0x10000, size: 0x4 }' >> %t.map
# RUN: echo '      - { sym: __Z4useBP1S, objAddr: 0x4, binAddr: 0x10010, size: 0x4 }' >> %t.map
# RUN: echo '...' >> %t.map

# RUN: dsymutil --linker=parallel -y %t.map -f -o %t.parallel.dSYM
# RUN: llvm-dwarfdump --verify %t.parallel.dSYM
# RUN: llvm-dwarfdump --debug-info %t.parallel.dSYM > %t.parallel.txt
# RUN: FileCheck %s --check-prefix=PARALLEL --input-file %t.parallel.txt
# RUN: FileCheck %s --check-prefix=MASK --input-file %t.parallel.txt \
# RUN:   --implicit-check-not='DW_AT_name{{.*}}"kMask"'
# RUN: FileCheck %s --check-prefix=FIELD --input-file %t.parallel.txt \
# RUN:   --implicit-check-not='DW_AT_name{{.*}}"field"'

## The parallel linker moves S into the artificial type unit, where the static
## declaration survives alongside the real member. Both spellings are valid
## output, and the surviving one comes from the unit that wins the type slot.

# PARALLEL: DW_AT_name{{.*}}"__artificial_type_unit"
# PARALLEL: DW_TAG_structure_type
# PARALLEL-NEXT: DW_AT_name{{.*}}"S"
# PARALLEL: DW_TAG_member
# PARALLEL-NEXT: DW_AT_name{{.*}}"kMask"
# PARALLEL-NEXT: DW_AT_type
# PARALLEL-NEXT: DW_AT_external
# PARALLEL-NEXT: DW_AT_declaration

## Each implicit-check-not lets its member be named exactly once in the whole
## output, so the two spellings collapsed into a single DIE and neither member
## displaced the other.

# MASK: DW_AT_name{{.*}}"kMask"
# MASK-NEXT: DW_AT_type
# MASK-NEXT: DW_AT_external
# MASK-NEXT: DW_AT_declaration

# FIELD: DW_AT_name{{.*}}"field"
# FIELD-NEXT: DW_AT_type
# FIELD-NEXT: DW_AT_data_member_location{{.*}}(0x00)

## The classic linker derives no names from child positions, so it is
## unaffected. It runs here to guard against regressions and to confirm the
## inputs aren't pathological.

# RUN: dsymutil --linker=classic -y %t.map -f -o %t.classic.dSYM
# RUN: llvm-dwarfdump --verify %t.classic.dSYM
# RUN: llvm-dwarfdump --debug-info %t.classic.dSYM | FileCheck %s --check-prefix=CLASSIC

# CLASSIC: DW_TAG_structure_type
# CLASSIC-NEXT: DW_AT_name{{.*}}"S"
# CLASSIC: DW_AT_name{{.*}}"kMask"
# CLASSIC: DW_AT_name{{.*}}"field"
# CLASSIC-NEXT: DW_AT_type
# CLASSIC-NEXT: DW_AT_data_member_location{{.*}}(0x00)

	.section	__TEXT,__text,regular,pure_instructions
	.globl	__Z4useAP1S
__Z4useAP1S:
LfuncA_begin:
	retq
	nop
	nop
	nop
LfuncA_end:

	.globl	__Z4useBP1S
__Z4useBP1S:
LfuncB_begin:
	retq
	nop
	nop
	nop
LfuncB_end:

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
	.byte	110                     ## DW_AT_linkage_name
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
	.byte	13                      ## DW_TAG_member
	.byte	0                       ## DW_CHILDREN_no
	.byte	3                       ## DW_AT_name
	.byte	8                       ## DW_FORM_string
	.byte	73                      ## DW_AT_type
	.byte	19                      ## DW_FORM_ref4
	.byte	56                      ## DW_AT_data_member_location
	.byte	11                      ## DW_FORM_data1
	.byte	0, 0

	.byte	6                       ## Abbreviation Code
	.byte	13                      ## DW_TAG_member (static data member, DWARF 4)
	.byte	0                       ## DW_CHILDREN_no
	.byte	3                       ## DW_AT_name
	.byte	8                       ## DW_FORM_string
	.byte	73                      ## DW_AT_type
	.byte	19                      ## DW_FORM_ref4
	.byte	63                      ## DW_AT_external
	.byte	12                      ## DW_FORM_flag
	.byte	60                      ## DW_AT_declaration
	.byte	12                      ## DW_FORM_flag
	.byte	28                      ## DW_AT_const_value
	.byte	11                      ## DW_FORM_data1
	.byte	0, 0

	.byte	7                       ## Abbreviation Code
	.byte	52                      ## DW_TAG_variable (static data member, DWARF 5)
	.byte	0                       ## DW_CHILDREN_no
	.byte	3                       ## DW_AT_name
	.byte	8                       ## DW_FORM_string
	.byte	73                      ## DW_AT_type
	.byte	19                      ## DW_FORM_ref4
	.byte	63                      ## DW_AT_external
	.byte	12                      ## DW_FORM_flag
	.byte	60                      ## DW_AT_declaration
	.byte	12                      ## DW_FORM_flag
	.byte	28                      ## DW_AT_const_value
	.byte	11                      ## DW_FORM_data1
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
## DWARF 4 compile unit: the static data member is a DW_TAG_member.
Lcu1_begin:
	.long	Lcu1_end - Lcu1_start   ## Length of Unit
Lcu1_start:
	.short	4                       ## DWARF version number
	.long	0                       ## Offset Into Abbrev. Section
	.byte	8                       ## Address Size (in bytes)

	.byte	1                       ## Abbrev [1] DW_TAG_compile_unit
	.asciz	"hand-written"          ## DW_AT_producer
	.short	0x0004                  ## DW_AT_language (DW_LANG_C_plus_plus)
	.asciz	"a.cpp"                 ## DW_AT_name

	.byte	2                       ## Abbrev [2] DW_TAG_subprogram
	.asciz	"useA"                  ## DW_AT_name
	.asciz	"__Z4useAP1S"           ## DW_AT_linkage_name
	.quad	LfuncA_begin            ## DW_AT_low_pc
	.quad	LfuncA_end              ## DW_AT_high_pc
	.byte	1                       ## DW_AT_external

	.byte	3                       ## Abbrev [3] DW_TAG_formal_parameter
	.long	Lcu1_s_ptr - Lcu1_begin ## DW_AT_type

	.byte	0                       ## End Of Children Mark (useA)

Lcu1_s:
	.byte	4                       ## Abbrev [4] DW_TAG_structure_type
	.asciz	"S"                     ## DW_AT_name
	.byte	4                       ## DW_AT_byte_size

	.byte	6                       ## Abbrev [6] DW_TAG_member
	.asciz	"kMask"                 ## DW_AT_name
	.long	Lcu1_int - Lcu1_begin   ## DW_AT_type
	.byte	1                       ## DW_AT_external
	.byte	1                       ## DW_AT_declaration
	.byte	1                       ## DW_AT_const_value

	.byte	5                       ## Abbrev [5] DW_TAG_member
	.asciz	"field"                 ## DW_AT_name
	.long	Lcu1_int - Lcu1_begin   ## DW_AT_type
	.byte	0                       ## DW_AT_data_member_location

	.byte	0                       ## End Of Children Mark (S)

Lcu1_int:
	.byte	8                       ## Abbrev [8] DW_TAG_base_type
	.asciz	"int"                   ## DW_AT_name
	.byte	4                       ## DW_AT_byte_size
	.byte	5                       ## DW_AT_encoding (DW_ATE_signed)

Lcu1_s_ptr:
	.byte	9                       ## Abbrev [9] DW_TAG_pointer_type
	.long	Lcu1_s - Lcu1_begin     ## DW_AT_type

	.byte	0                       ## End Of Children Mark (CU)
Lcu1_end:

## DWARF 5 compile unit: the static data member is a DW_TAG_variable.
Lcu2_begin:
	.long	Lcu2_end - Lcu2_start   ## Length of Unit
Lcu2_start:
	.short	5                       ## DWARF version number
	.byte	1                       ## DW_UT_compile
	.byte	8                       ## Address Size (in bytes)
	.long	0                       ## Offset Into Abbrev. Section

	.byte	1                       ## Abbrev [1] DW_TAG_compile_unit
	.asciz	"hand-written"          ## DW_AT_producer
	.short	0x0004                  ## DW_AT_language (DW_LANG_C_plus_plus)
	.asciz	"b.cpp"                 ## DW_AT_name

	.byte	2                       ## Abbrev [2] DW_TAG_subprogram
	.asciz	"useB"                  ## DW_AT_name
	.asciz	"__Z4useBP1S"           ## DW_AT_linkage_name
	.quad	LfuncB_begin            ## DW_AT_low_pc
	.quad	LfuncB_end              ## DW_AT_high_pc
	.byte	1                       ## DW_AT_external

	.byte	3                       ## Abbrev [3] DW_TAG_formal_parameter
	.long	Lcu2_s_ptr - Lcu2_begin ## DW_AT_type

	.byte	0                       ## End Of Children Mark (useB)

Lcu2_s:
	.byte	4                       ## Abbrev [4] DW_TAG_structure_type
	.asciz	"S"                     ## DW_AT_name
	.byte	4                       ## DW_AT_byte_size

	.byte	7                       ## Abbrev [7] DW_TAG_variable
	.asciz	"kMask"                 ## DW_AT_name
	.long	Lcu2_int - Lcu2_begin   ## DW_AT_type
	.byte	1                       ## DW_AT_external
	.byte	1                       ## DW_AT_declaration
	.byte	1                       ## DW_AT_const_value

	.byte	5                       ## Abbrev [5] DW_TAG_member
	.asciz	"field"                 ## DW_AT_name
	.long	Lcu2_int - Lcu2_begin   ## DW_AT_type
	.byte	0                       ## DW_AT_data_member_location

	.byte	0                       ## End Of Children Mark (S)

Lcu2_int:
	.byte	8                       ## Abbrev [8] DW_TAG_base_type
	.asciz	"int"                   ## DW_AT_name
	.byte	4                       ## DW_AT_byte_size
	.byte	5                       ## DW_AT_encoding (DW_ATE_signed)

Lcu2_s_ptr:
	.byte	9                       ## Abbrev [9] DW_TAG_pointer_type
	.long	Lcu2_s - Lcu2_begin     ## DW_AT_type

	.byte	0                       ## End Of Children Mark (CU)
Lcu2_end:
