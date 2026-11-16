# Two compile units reference the same pointer type, so whichever unit marks it
# first leaves the other holding a reference to an already-marked subtree. The
# dependencies of such a subtree are summarized once and applied to every unit
# that references it. This exercises that path: the pointer names a nested type
# that has to leave the artificial type unit, because its member has a type from
# an anonymous namespace and so cannot be ODR deduplicated, while the type
# enclosing it stays in the type unit.
#
# A type-unit DIE may only reference DIEs that are themselves in the type unit,
# so leaving the pointer behind in the type unit produces a dangling type-unit
# reference.

# RUN: llvm-mc -triple x86_64-apple-darwin -filetype=obj %s -o %t.o
# RUN: llvm-dwarfdump --verify %t.o

# RUN: echo '---' > %t.map
# RUN: echo "triple:          'x86_64-apple-darwin'" >> %t.map
# RUN: echo 'objects:'  >> %t.map
# RUN: echo " -  filename: '%t.o'" >> %t.map
# RUN: echo '    symbols:' >> %t.map
# RUN: echo '      - { sym: __Z2f1v, objAddr: 0x0, binAddr: 0x10000, size: 0x1 }' >> %t.map
# RUN: echo '      - { sym: __Z2f2v, objAddr: 0x1, binAddr: 0x10010, size: 0x1 }' >> %t.map
# RUN: echo '...' >> %t.map

# RUN: dsymutil --linker=parallel -y %t.map -f -o %t.dSYM
# RUN: llvm-dwarfdump --verify %t.dSYM
# RUN: llvm-dwarfdump -debug-info %t.dSYM | FileCheck %s

## Only the deduplicated base type reaches the type unit. Neither the nested
## type, nor the type enclosing it, nor the pointer to it may be placed there.
# CHECK:          DW_AT_name{{.*}}"__artificial_type_unit"
# CHECK-NOT:      DW_TAG_structure_type
# CHECK-NOT:      DW_TAG_pointer_type

# CHECK:          DW_AT_name{{.*}}"CU1"
# CHECK:          DW_TAG_namespace
# CHECK:            DW_AT_name{{.*}}"Hidden"
# CHECK:          DW_TAG_structure_type
# CHECK:            DW_AT_name{{.*}}"Outer"
# CHECK:            DW_TAG_structure_type
# CHECK:              DW_AT_name{{.*}}"Inner"
# CHECK:          DW_TAG_pointer_type
# CHECK-NEXT:       DW_AT_type{{.*}}"Outer::Inner"

## The unit that found the pointer already marked has to reach the same
## placement, so its reference resolves into the compile unit as well.
# CHECK:          DW_AT_name{{.*}}"CU2"
# CHECK:            DW_AT_name{{.*}}"f2"
# CHECK:            DW_TAG_formal_parameter
# CHECK-NEXT:         DW_AT_type{{.*}}"Outer::Inner *"

## Placement must not depend on how the units interleave.
# RUN: dsymutil --linker=parallel -y %t.map -f -o %t.1.dSYM --num-threads 1
# RUN: dsymutil --linker=parallel -y %t.map -f -o %t.4.dSYM --num-threads 4
# RUN: dsymutil --linker=parallel -y %t.map -f -o %t.4b.dSYM --num-threads 4
## The first line of the dump names the input file, which differs by design.
# RUN: llvm-dwarfdump -debug-info %t.1.dSYM | sed 1d > %t.1.txt
# RUN: llvm-dwarfdump -debug-info %t.4.dSYM | sed 1d > %t.4.txt
# RUN: llvm-dwarfdump -debug-info %t.4b.dSYM | sed 1d > %t.4b.txt
# RUN: diff %t.1.txt %t.4.txt
# RUN: diff %t.4.txt %t.4b.txt

	.section	__TEXT,__text,regular,pure_instructions
	.globl	__Z2f1v
__Z2f1v:
Lfunc_begin0:
	retq
Lfunc_end0:
	.globl	__Z2f2v
__Z2f2v:
Lfunc_begin1:
	retq
Lfunc_end1:

	.section	__DWARF,__debug_abbrev,regular,debug
Lsection_abbrev:
## Abbreviations for CU1.
Labbrev_cu1:
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
	.byte	57                      ## DW_TAG_namespace
	.byte	1                       ## DW_CHILDREN_yes
	.byte	0, 0

	.byte	5                       ## Abbreviation Code
	.byte	19                      ## DW_TAG_structure_type
	.byte	1                       ## DW_CHILDREN_yes
	.byte	3                       ## DW_AT_name
	.byte	8                       ## DW_FORM_string
	.byte	11                      ## DW_AT_byte_size
	.byte	11                      ## DW_FORM_data1
	.byte	0, 0

	.byte	6                       ## Abbreviation Code
	.byte	13                      ## DW_TAG_member
	.byte	0                       ## DW_CHILDREN_no
	.byte	3                       ## DW_AT_name
	.byte	8                       ## DW_FORM_string
	.byte	73                      ## DW_AT_type
	.byte	19                      ## DW_FORM_ref4
	.byte	56                      ## DW_AT_data_member_location
	.byte	11                      ## DW_FORM_data1
	.byte	0, 0

	.byte	7                       ## Abbreviation Code
	.byte	15                      ## DW_TAG_pointer_type
	.byte	0                       ## DW_CHILDREN_no
	.byte	73                      ## DW_AT_type
	.byte	19                      ## DW_FORM_ref4
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

	.byte	0                       ## EOM(3)

## Abbreviations for CU2. The formal parameter references a DIE in CU1, so it
## uses DW_FORM_ref_addr rather than the unit-relative DW_FORM_ref4.
Labbrev_cu2:
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
	.byte	16                      ## DW_FORM_ref_addr
	.byte	0, 0

	.byte	0                       ## EOM(3)

	.section	__DWARF,__debug_info,regular,debug
Lsection_info:
Lcu1_begin:
	.long	Lcu1_end - Lcu1_start   ## Length of Unit
Lcu1_start:
	.short	4                       ## DWARF version number
	.long	Labbrev_cu1 - Lsection_abbrev ## Offset Into Abbrev. Section
	.byte	8                       ## Address Size (in bytes)

	.byte	1                       ## Abbrev [1] DW_TAG_compile_unit
	.asciz	"hand-written"          ## DW_AT_producer
	.short	0x0004                  ## DW_AT_language (DW_LANG_C_plus_plus)
	.asciz	"CU1"                   ## DW_AT_name

	.byte	2                       ## Abbrev [2] DW_TAG_subprogram
	.asciz	"f1"                    ## DW_AT_name
	.asciz	"__Z2f1v"               ## DW_AT_MIPS_linkage_name
	.quad	Lfunc_begin0            ## DW_AT_low_pc
	.quad	Lfunc_end0              ## DW_AT_high_pc
	.byte	1                       ## DW_AT_external

	.byte	3                       ## Abbrev [3] DW_TAG_formal_parameter
	.long	Linner_ptr - Lcu1_begin ## DW_AT_type

	.byte	0                       ## End Of Children Mark (f1)

	.byte	4                       ## Abbrev [4] DW_TAG_namespace (anonymous)

Lhidden:
	.byte	5                       ## Abbrev [5] DW_TAG_structure_type
	.asciz	"Hidden"                ## DW_AT_name
	.byte	4                       ## DW_AT_byte_size

	.byte	6                       ## Abbrev [6] DW_TAG_member
	.asciz	"x"                     ## DW_AT_name
	.long	Lint - Lcu1_begin       ## DW_AT_type
	.byte	0                       ## DW_AT_data_member_location

	.byte	0                       ## End Of Children Mark (Hidden)

	.byte	0                       ## End Of Children Mark (namespace)

## Outer is ODR deduplicated into the artificial type unit, but Inner has a
## member of a type from the anonymous namespace, which cannot be deduplicated.
## Inner is therefore demoted to plain DWARF independently of Outer, and the
## pointer to Inner has to follow it.
Louter:
	.byte	5                       ## Abbrev [5] DW_TAG_structure_type
	.asciz	"Outer"                 ## DW_AT_name
	.byte	4                       ## DW_AT_byte_size

Linner:
	.byte	5                       ## Abbrev [5] DW_TAG_structure_type
	.asciz	"Inner"                 ## DW_AT_name
	.byte	4                       ## DW_AT_byte_size

	.byte	6                       ## Abbrev [6] DW_TAG_member
	.asciz	"h"                     ## DW_AT_name
	.long	Lhidden - Lcu1_begin    ## DW_AT_type
	.byte	0                       ## DW_AT_data_member_location

	.byte	0                       ## End Of Children Mark (Inner)

	.byte	0                       ## End Of Children Mark (Outer)

Linner_ptr:
	.byte	7                       ## Abbrev [7] DW_TAG_pointer_type
	.long	Linner - Lcu1_begin     ## DW_AT_type

Lint:
	.byte	8                       ## Abbrev [8] DW_TAG_base_type
	.asciz	"int"                   ## DW_AT_name
	.byte	4                       ## DW_AT_byte_size
	.byte	5                       ## DW_AT_encoding (DW_ATE_signed)

	.byte	0                       ## End Of Children Mark (CU1)
Lcu1_end:

Lcu2_begin:
	.long	Lcu2_end - Lcu2_start   ## Length of Unit
Lcu2_start:
	.short	4                       ## DWARF version number
	.long	Labbrev_cu2 - Lsection_abbrev ## Offset Into Abbrev. Section
	.byte	8                       ## Address Size (in bytes)

	.byte	1                       ## Abbrev [1] DW_TAG_compile_unit
	.asciz	"hand-written"          ## DW_AT_producer
	.short	0x0004                  ## DW_AT_language (DW_LANG_C_plus_plus)
	.asciz	"CU2"                   ## DW_AT_name

	.byte	2                       ## Abbrev [2] DW_TAG_subprogram
	.asciz	"f2"                    ## DW_AT_name
	.asciz	"__Z2f2v"               ## DW_AT_MIPS_linkage_name
	.quad	Lfunc_begin1            ## DW_AT_low_pc
	.quad	Lfunc_end1              ## DW_AT_high_pc
	.byte	1                       ## DW_AT_external

	.byte	3                       ## Abbrev [3] DW_TAG_formal_parameter
	.long	Linner_ptr - Lsection_info ## DW_AT_type

	.byte	0                       ## End Of Children Mark (f2)

	.byte	0                       ## End Of Children Mark (CU2)
Lcu2_end:
