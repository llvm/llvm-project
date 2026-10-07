# A type reachable only through DW_AT_LLVM_alloc_type must still be kept and the
# attribute must resolve to a real type DIE. The allocated type here is a
# function-local typedef inside a template subprogram. As an ODR/type-table
# candidate it is skipped while the subprogram is marked live, so it survives
# only when the linker treats alloc_type as a type reference.

# RUN: llvm-mc -triple x86_64-apple-darwin -filetype=obj %s -o %t.o
# RUN: llvm-dwarfdump --verify %t.o

# RUN: echo '---' > %t.map
# RUN: echo "triple:          'x86_64-apple-darwin'" >> %t.map
# RUN: echo 'objects:'  >> %t.map
# RUN: echo " -  filename: '%t.o'" >> %t.map
# RUN: echo '    symbols:' >> %t.map
# RUN: echo '      - { sym: __Z13createAdaptorIiEvv, objAddr: 0x0, binAddr: 0x10000, size: 0x1 }' >> %t.map
# RUN: echo '...' >> %t.map

# RUN: dsymutil --linker=parallel -y %t.map -f -o %t.dSYM
# RUN: llvm-dwarfdump --verify %t.dSYM
# RUN: llvm-dwarfdump -debug-info %t.dSYM | FileCheck %s

# RUN: dsymutil --linker=classic -y %t.map -f -o %t.dSYM
# RUN: llvm-dwarfdump --verify %t.dSYM
# RUN: llvm-dwarfdump -debug-info %t.dSYM | FileCheck %s

# The allocated type must be emitted and alloc_type must resolve to it. The
# reference and the DIE offset are matched with leading zeros stripped because
# the two linkers encode the reference with different-width forms.

# CHECK: 0x{{0*}}[[TYPEDEF:[1-9a-f][0-9a-f]*]]: DW_TAG_typedef
# CHECK-NEXT: DW_AT_name{{.*}}"PassModelT"
# CHECK: DW_TAG_call_site
# CHECK: DW_AT_LLVM_alloc_type{{.*}}(0x{{0*}}[[TYPEDEF]])

	.section	__TEXT,__text,regular,pure_instructions
	.globl	__Z13createAdaptorIiEvv
__Z13createAdaptorIiEvv:
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
	.byte	110                     ## DW_AT_linkage_name
	.byte	8                       ## DW_FORM_string
	.byte	73                      ## DW_AT_type
	.byte	19                      ## DW_FORM_ref4
	.byte	17                      ## DW_AT_low_pc
	.byte	1                       ## DW_FORM_addr
	.byte	18                      ## DW_AT_high_pc
	.byte	6                       ## DW_FORM_data4
	.byte	122                     ## DW_AT_call_all_calls
	.byte	25                      ## DW_FORM_flag_present
	.byte	0, 0

	.byte	3                       ## Abbreviation Code
	.byte	5                       ## DW_TAG_formal_parameter
	.byte	0                       ## DW_CHILDREN_no
	.byte	73                      ## DW_AT_type
	.byte	19                      ## DW_FORM_ref4
	.byte	0, 0

	.byte	4                       ## Abbreviation Code
	.byte	22                      ## DW_TAG_typedef
	.byte	0                       ## DW_CHILDREN_no
	.byte	3                       ## DW_AT_name
	.byte	8                       ## DW_FORM_string
	.byte	73                      ## DW_AT_type
	.byte	19                      ## DW_FORM_ref4
	.byte	0, 0

	.byte	5                       ## Abbreviation Code
	.byte	72                      ## DW_TAG_call_site
	.byte	0                       ## DW_CHILDREN_no
	.byte	127                     ## DW_AT_call_origin
	.byte	19                      ## DW_FORM_ref4
	.byte	0x7d                    ## DW_AT_call_return_pc
	.byte	1                       ## DW_FORM_addr
	.byte	0x8e, 0x7c              ## DW_AT_LLVM_alloc_type (0x3e0e)
	.byte	19                      ## DW_FORM_ref4
	.byte	0, 0

	.byte	6                       ## Abbreviation Code
	.byte	36                      ## DW_TAG_base_type
	.byte	0                       ## DW_CHILDREN_no
	.byte	3                       ## DW_AT_name
	.byte	8                       ## DW_FORM_string
	.byte	0, 0

	.byte	7                       ## Abbreviation Code
	.byte	46                      ## DW_TAG_subprogram
	.byte	0                       ## DW_CHILDREN_no
	.byte	3                       ## DW_AT_name
	.byte	8                       ## DW_FORM_string
	.byte	110                     ## DW_AT_linkage_name
	.byte	8                       ## DW_FORM_string
	.byte	60                      ## DW_AT_declaration
	.byte	25                      ## DW_FORM_flag_present
	.byte	0, 0

	.byte	0                       ## EOM(3)

	.section	__DWARF,__debug_info,regular,debug
Lsection_info:
	.long	Lcu_end - Lcu_start     ## Length of Unit
Lcu_start:
	.short	4                       ## DWARF version number
	.long	0                       ## Offset Into Abbrev. Section
	.byte	8                       ## Address Size (in bytes)

	.byte	1                       ## Abbrev [1] DW_TAG_compile_unit
	.asciz	"hand-written"          ## DW_AT_producer
	.short	0x0021                  ## DW_AT_language (DW_LANG_C_plus_plus_14)
	.asciz	"alloc_type.cpp"        ## DW_AT_name

Lfoo:
	.byte	2                       ## Abbrev [2] DW_TAG_subprogram
	.asciz	"createAdaptor<int>"    ## DW_AT_name
	.asciz	"__Z13createAdaptorIiEvv" ## DW_AT_linkage_name
	.long	Lint - Lsection_info    ## DW_AT_type
	.quad	Lfunc_begin0            ## DW_AT_low_pc
	.long	Lfunc_end0 - Lfunc_begin0 ## DW_AT_high_pc

	.byte	3                       ## Abbrev [3] DW_TAG_formal_parameter
	.long	Lint - Lsection_info    ## DW_AT_type

Ltypedef:
	.byte	4                       ## Abbrev [4] DW_TAG_typedef
	.asciz	"PassModelT"            ## DW_AT_name
	.long	Lint - Lsection_info    ## DW_AT_type

	.byte	5                       ## Abbrev [5] DW_TAG_call_site
	.long	Lraw_alloc - Lsection_info ## DW_AT_call_origin
	.quad	Lfunc_begin0            ## DW_AT_call_return_pc
	.long	Ltypedef - Lsection_info ## DW_AT_LLVM_alloc_type

	.byte	0                       ## End Of Children Mark (foo)

Lint:
	.byte	6                       ## Abbrev [6] DW_TAG_base_type
	.asciz	"int"                   ## DW_AT_name

Lraw_alloc:
	.byte	7                       ## Abbrev [7] DW_TAG_subprogram
	.asciz	"raw_alloc"             ## DW_AT_name
	.asciz	"__Z9raw_allocv"        ## DW_AT_linkage_name

	.byte	0                       ## End Of Children Mark (CU)
Lcu_end:
