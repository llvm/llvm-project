# A compile unit which is linked, and then found to be inter-connected with a
# unit linked after it, is linked again. Its string offsets must be computed
# again too: the DIEs kept the second time are not the same, so stale indexes
# would make its strings depend on how the units interleave.
#
# CU1 is linked first, alone, without the structure type only CU2 refers to.
# Linking CU1 again keeps the type, which must get the index of its position.

# RUN: llvm-mc -triple x86_64-apple-darwin -filetype=obj %s -o %t.o

# RUN: echo '---' > %t.map
# RUN: echo "triple:          'x86_64-apple-darwin'" >> %t.map
# RUN: echo 'objects:'  >> %t.map
# RUN: echo " -  filename: '%t.o'" >> %t.map
# RUN: echo '    symbols:' >> %t.map
# RUN: echo '      - { sym: _f1, objAddr: 0x0, binAddr: 0x10000, size: 0x1 }' >> %t.map
# RUN: echo '      - { sym: _f2, objAddr: 0x1, binAddr: 0x10010, size: 0x1 }' >> %t.map
# RUN: echo '...' >> %t.map

# RUN: dsymutil --linker=classic -y %t.map -f -o %t.classic.dSYM
# RUN: llvm-dwarfdump --verify %t.classic.dSYM
# RUN: llvm-dwarfdump -v -debug-info %t.classic.dSYM \
# RUN:   | FileCheck %s --check-prefixes=CHECK,CLASSIC

# RUN: dsymutil --linker=parallel -y %t.map -f -o %t.dSYM --num-threads 1
# RUN: llvm-dwarfdump --verify %t.dSYM
# RUN: llvm-dwarfdump -v -debug-info %t.dSYM \
# RUN:   | FileCheck %s --check-prefixes=CHECK,PARALLEL

## The classic linker emits one .debug_str_offsets table shared by all units,
## the parallel linker one table per unit.
# CHECK:         DW_TAG_compile_unit
# CHECK:           DW_AT_name [DW_FORM_strx] (indexed (00000000) string = "CU1")
# CHECK:         DW_TAG_structure_type
# CHECK-NEXT:      DW_AT_name [DW_FORM_strx] (indexed (00000001) string = "S")
# CHECK:         DW_TAG_subprogram
# CHECK-NEXT:      DW_AT_name [DW_FORM_strx] (indexed (00000002) string = "f1")

# CHECK:         DW_TAG_compile_unit
# CLASSIC:         DW_AT_name [DW_FORM_strx] (indexed (00000003) string = "CU2")
# PARALLEL:        DW_AT_name [DW_FORM_strx] (indexed (00000000) string = "CU2")
# CHECK:         DW_TAG_formal_parameter
# CLASSIC-NEXT:    DW_AT_name [DW_FORM_strx] (indexed (00000005) string = "p")
# PARALLEL-NEXT:   DW_AT_name [DW_FORM_strx] (indexed (00000002) string = "p")
# CHECK-NEXT:      DW_AT_type [DW_FORM_ref_addr] {{.*}} "S"

## The output of the parallel linker does not depend on the number of threads.
# RUN: dsymutil --linker=parallel -y %t.map -f -o %t.4.dSYM --num-threads 4
# RUN: llvm-dwarfdump -debug-info %t.dSYM | sed 1d > %t.1.txt
# RUN: llvm-dwarfdump -debug-info %t.4.dSYM | sed 1d > %t.4.txt
# RUN: diff %t.1.txt %t.4.txt

	.section	__TEXT,__text,regular,pure_instructions
	.globl	_f1
_f1:
	retq
	.globl	_f2
_f2:
	retq

	.section	__DWARF,__debug_abbrev,regular,debug
Lsection_abbrev:
	.byte	1                       ## Abbreviation Code
	.byte	17                      ## DW_TAG_compile_unit
	.byte	1                       ## DW_CHILDREN_yes
	.byte	3                       ## DW_AT_name
	.byte	37                      ## DW_FORM_strx1
	.byte	19                      ## DW_AT_language
	.byte	5                       ## DW_FORM_data2
	.byte	114                     ## DW_AT_str_offsets_base
	.byte	23                      ## DW_FORM_sec_offset
	.byte	17                      ## DW_AT_low_pc
	.byte	1                       ## DW_FORM_addr
	.byte	18                      ## DW_AT_high_pc
	.byte	6                       ## DW_FORM_data4
	.byte	0                       ## EOM(1)
	.byte	0                       ## EOM(2)
	.byte	2                       ## Abbreviation Code
	.byte	19                      ## DW_TAG_structure_type
	.byte	0                       ## DW_CHILDREN_no
	.byte	3                       ## DW_AT_name
	.byte	37                      ## DW_FORM_strx1
	.byte	11                      ## DW_AT_byte_size
	.byte	11                      ## DW_FORM_data1
	.byte	0                       ## EOM(1)
	.byte	0                       ## EOM(2)
	.byte	3                       ## Abbreviation Code
	.byte	46                      ## DW_TAG_subprogram
	.byte	0                       ## DW_CHILDREN_no
	.byte	3                       ## DW_AT_name
	.byte	37                      ## DW_FORM_strx1
	.byte	17                      ## DW_AT_low_pc
	.byte	1                       ## DW_FORM_addr
	.byte	18                      ## DW_AT_high_pc
	.byte	6                       ## DW_FORM_data4
	.byte	0                       ## EOM(1)
	.byte	0                       ## EOM(2)
	.byte	4                       ## Abbreviation Code
	.byte	46                      ## DW_TAG_subprogram
	.byte	1                       ## DW_CHILDREN_yes
	.byte	3                       ## DW_AT_name
	.byte	37                      ## DW_FORM_strx1
	.byte	17                      ## DW_AT_low_pc
	.byte	1                       ## DW_FORM_addr
	.byte	18                      ## DW_AT_high_pc
	.byte	6                       ## DW_FORM_data4
	.byte	0                       ## EOM(1)
	.byte	0                       ## EOM(2)
	.byte	5                       ## Abbreviation Code
	.byte	5                       ## DW_TAG_formal_parameter
	.byte	0                       ## DW_CHILDREN_no
	.byte	3                       ## DW_AT_name
	.byte	37                      ## DW_FORM_strx1
	.byte	73                      ## DW_AT_type
	.byte	16                      ## DW_FORM_ref_addr
	.byte	0                       ## EOM(1)
	.byte	0                       ## EOM(2)
	.byte	0                       ## EOM(3)

	.section	__DWARF,__debug_info,regular,debug
Lsection_info:
Lcu_begin0:
	.long	Lcu_end0-Lcu_start0     ## Length of Unit
Lcu_start0:
	.short	5                       ## DWARF version number
	.byte	1                       ## DWARF Unit Type
	.byte	8                       ## Address Size (in bytes)
	.long	Lsection_abbrev-Lsection_abbrev ## Offset Into Abbrev. Section
	.byte	1                       ## Abbrev [1] DW_TAG_compile_unit
	.byte	0                       ## DW_AT_name "CU1"
	.short	12                      ## DW_AT_language DW_LANG_C99
	.long	Lstr_offsets_base-Lsection_str_off ## DW_AT_str_offsets_base
	.quad	_f1                     ## DW_AT_low_pc
	.long	1                       ## DW_AT_high_pc
Ltype_S:
	.byte	2                       ## Abbrev [2] DW_TAG_structure_type
	.byte	1                       ## DW_AT_name "S"
	.byte	4                       ## DW_AT_byte_size
	.byte	3                       ## Abbrev [3] DW_TAG_subprogram
	.byte	2                       ## DW_AT_name "f1"
	.quad	_f1                     ## DW_AT_low_pc
	.long	1                       ## DW_AT_high_pc
	.byte	0                       ## End Of Children Mark
Lcu_end0:
Lcu_begin1:
	.long	Lcu_end1-Lcu_start1     ## Length of Unit
Lcu_start1:
	.short	5                       ## DWARF version number
	.byte	1                       ## DWARF Unit Type
	.byte	8                       ## Address Size (in bytes)
	.long	Lsection_abbrev-Lsection_abbrev ## Offset Into Abbrev. Section
	.byte	1                       ## Abbrev [1] DW_TAG_compile_unit
	.byte	3                       ## DW_AT_name "CU2"
	.short	12                      ## DW_AT_language DW_LANG_C99
	.long	Lstr_offsets_base-Lsection_str_off ## DW_AT_str_offsets_base
	.quad	_f2                     ## DW_AT_low_pc
	.long	1                       ## DW_AT_high_pc
	.byte	4                       ## Abbrev [4] DW_TAG_subprogram
	.byte	4                       ## DW_AT_name "f2"
	.quad	_f2                     ## DW_AT_low_pc
	.long	1                       ## DW_AT_high_pc
	.byte	5                       ## Abbrev [5] DW_TAG_formal_parameter
	.byte	5                       ## DW_AT_name "p"
	.long	Ltype_S-Lsection_info   ## DW_AT_type
	.byte	0                       ## End Of Children Mark
	.byte	0                       ## End Of Children Mark
Lcu_end1:

	.section	__DWARF,__debug_str_offs,regular,debug
Lsection_str_off:
	.long	28                      ## Length of String Offsets Set
	.short	5
	.short	0
Lstr_offsets_base:
	.long	Linfo_string0-Lsection_str
	.long	Linfo_string1-Lsection_str
	.long	Linfo_string2-Lsection_str
	.long	Linfo_string3-Lsection_str
	.long	Linfo_string4-Lsection_str
	.long	Linfo_string5-Lsection_str

	.section	__DWARF,__debug_str,regular,debug
Lsection_str:
Linfo_string0:
	.asciz	"CU1"
Linfo_string1:
	.asciz	"S"
Linfo_string2:
	.asciz	"f1"
Linfo_string3:
	.asciz	"CU2"
Linfo_string4:
	.asciz	"f2"
Linfo_string5:
	.asciz	"p"

.subsections_via_symbols
