# This test sets non-zero Die Offset Base field in the accelerator table header,
# and makes sure it is *not* added to the DW_FORM_data*** forms.

# RUN: llvm-mc -triple x86_64-apple-darwin %s -filetype=obj -o %t
# RUN: llvm-dwarfdump -find=main %t | FileCheck %s

# CHECK: DW_TAG_subprogram
# CHECK-NEXT: DW_AT_name ("main")
# CHECK-NEXT: DW_AT_external

	.section	__DWARF,__debug_str,regular,debug
Ldebug_str:
Lstring_producer:
	.asciz	"Hand-written dwarf"
Lstring_main:
	.asciz	"main"

	.section	__DWARF,__debug_abbrev,regular,debug
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

	.section	__DWARF,__debug_info,regular,debug
Ldebug_info:
	.long	Lcu_end0-Lcu_start0   # Length of Unit
Lcu_start0:
	.short	4                       # DWARF version number
	.long	0                       # Offset Into Abbrev. Section
	.byte	8                       # Address Size (in bytes)
	.byte	1                       # Abbrev [1] DW_TAG_compile_unit
	.long	Lstring_producer-Ldebug_str # DW_AT_producer
	.short	12                      # DW_AT_language
Ldie_main:
	.byte	2                       # Abbrev [2] DW_TAG_subprogram
	.long	Lstring_main-Ldebug_str # DW_AT_name
                                        # DW_AT_external
	.byte	0                       # End Of Children Mark
Lcu_end0:

	.section	__DWARF,__apple_names,regular,debug
Lnames_begin:
	.long	1212240712              ## Header Magic
	.short	1                       ## Header Version
	.short	0                       ## Header Hash Function
	.long	1                       ## Header Bucket Count
	.long	1                       ## Header Hash Count
	.long	12                      ## Header Data Length
	.long	1                       ## HeaderData Die Offset Base
	.long	1                       ## HeaderData Atom Count
	.short	1                       ## DW_ATOM_die_offset
	.short	6                       ## DW_FORM_data4
	.long	0                       ## Bucket 0
	.long	2090499946              ## Hash in Bucket 0
	.long	LNames0-Lnames_begin    ## Offset in Bucket 0
LNames0:
	.long	Lstring_main-Ldebug_str ## main
	.long	1                       ## Num DIEs
	.long	Ldie_main-Ldebug_info
	.long	0
