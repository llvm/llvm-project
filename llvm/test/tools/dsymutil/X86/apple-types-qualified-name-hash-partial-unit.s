## Check that both linkers compute the same DW_ATOM_qual_name_hash for a type
## whether the unit root is DW_TAG_compile_unit or DW_TAG_partial_unit. A
## partial unit is not the containing scope of its declarations (DWARFv5 section
## 3.1.1), so the root's DW_AT_name, a source file path, must not be hashed.

# RUN: llvm-mc -triple x86_64-apple-darwin -filetype=obj --defsym ROOT=17 --defsym UNITTYPE=1 %s -o %t.cu.o
# RUN: llvm-mc -triple x86_64-apple-darwin -filetype=obj --defsym ROOT=60 --defsym UNITTYPE=3 %s -o %t.pu.o

# RUN: echo '---' > %t.cu.map
# RUN: echo "triple:          'x86_64-apple-darwin'" >> %t.cu.map
# RUN: echo 'objects:'  >> %t.cu.map
# RUN: echo " -  filename: '%t.cu.o'" >> %t.cu.map
# RUN: echo '    symbols:' >> %t.cu.map
# RUN: echo '      - { sym: _foo, objAddr: 0x0, binAddr: 0x10000, size: 0x1 }' >> %t.cu.map
# RUN: echo '...' >> %t.cu.map

# RUN: echo '---' > %t.pu.map
# RUN: echo "triple:          'x86_64-apple-darwin'" >> %t.pu.map
# RUN: echo 'objects:'  >> %t.pu.map
# RUN: echo " -  filename: '%t.pu.o'" >> %t.pu.map
# RUN: echo '    symbols:' >> %t.pu.map
# RUN: echo '      - { sym: _foo, objAddr: 0x0, binAddr: 0x10000, size: 0x1 }' >> %t.pu.map
# RUN: echo '...' >> %t.pu.map

# RUN: dsymutil --linker=parallel -accelerator=Apple --verify-dwarf=all -y %t.cu.map -f -o %t.cu.parallel.dSYM
# RUN: llvm-dwarfdump --apple-types %t.cu.parallel.dSYM | FileCheck %s --implicit-check-not="Data 1"

# RUN: dsymutil --linker=parallel -accelerator=Apple --verify-dwarf=none -y %t.pu.map -f -o %t.pu.parallel.dSYM
# RUN: llvm-dwarfdump --apple-types %t.pu.parallel.dSYM | FileCheck %s --implicit-check-not="Data 1"

# RUN: dsymutil --linker=classic -accelerator=Apple --verify-dwarf=all -y %t.cu.map -f -o %t.cu.classic.dSYM
# RUN: llvm-dwarfdump --apple-types %t.cu.classic.dSYM | FileCheck %s --implicit-check-not="Data 1"

# RUN: dsymutil --linker=classic -accelerator=Apple --verify-dwarf=none -y %t.pu.map -f -o %t.pu.classic.dSYM
# RUN: llvm-dwarfdump --apple-types %t.pu.classic.dSYM | FileCheck %s --implicit-check-not="Data 1"

## FIXME: Verify the partial unit output once #219365 is fixed. dsymutil
## verifies by default only in assertions builds, so the control runs ask.

## Atom[0], the output DIE offset, differs between the linkers. Bar, in
## namespace ns, catches a walk that stops before the unit root.

# CHECK:      String: {{.*}} "Bar"
# CHECK-NEXT: Data 0 [
# CHECK-NEXT:   Atom[0]:
# CHECK-NEXT:   Atom[1]: 0x0013 (DW_TAG_structure_type)
# CHECK-NEXT:   Atom[2]: 0x00
# CHECK-NEXT:   Atom[3]: 0x27c1798f

# CHECK:      String: {{.*}} "Foo"
# CHECK-NEXT: Data 0 [
# CHECK-NEXT:   Atom[0]:
# CHECK-NEXT:   Atom[1]: 0x0013 (DW_TAG_structure_type)
# CHECK-NEXT:   Atom[2]: 0x00
# CHECK-NEXT:   Atom[3]: 0x0c3993dd

	.section	__TEXT,__text,regular,pure_instructions
	.globl	_foo
_foo:
Lfunc_begin0:
	retq
Lfunc_end0:

	.section	__DWARF,__debug_abbrev,regular,debug
Lsection_abbrev:
	.byte	1                       ## Abbreviation Code
	.byte	ROOT                    ## DW_TAG_compile_unit / DW_TAG_partial_unit
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
	.byte	73                      ## DW_AT_type
	.byte	19                      ## DW_FORM_ref4
	.byte	17                      ## DW_AT_low_pc
	.byte	1                       ## DW_FORM_addr
	.byte	18                      ## DW_AT_high_pc
	.byte	6                       ## DW_FORM_data4
	.byte	0, 0

	.byte	3                       ## Abbreviation Code
	.byte	5                       ## DW_TAG_formal_parameter
	.byte	0                       ## DW_CHILDREN_no
	.byte	3                       ## DW_AT_name
	.byte	8                       ## DW_FORM_string
	.byte	73                      ## DW_AT_type
	.byte	19                      ## DW_FORM_ref4
	.byte	0, 0

	.byte	4                       ## Abbreviation Code
	.byte	19                      ## DW_TAG_structure_type
	.byte	0                       ## DW_CHILDREN_no
	.byte	3                       ## DW_AT_name
	.byte	8                       ## DW_FORM_string
	.byte	11                      ## DW_AT_byte_size
	.byte	11                      ## DW_FORM_data1
	.byte	0, 0

	.byte	5                       ## Abbreviation Code
	.byte	57                      ## DW_TAG_namespace
	.byte	1                       ## DW_CHILDREN_yes
	.byte	3                       ## DW_AT_name
	.byte	8                       ## DW_FORM_string
	.byte	0, 0

	.byte	0                       ## EOM(3)

	.section	__DWARF,__debug_info,regular,debug
Lsection_info:
	.long	Lcu_end - Lcu_start     ## Length of Unit
Lcu_start:
	.short	5                       ## DWARF version number
	.byte	UNITTYPE                ## DW_UT_compile / DW_UT_partial
	.byte	8                       ## Address Size (in bytes)
	.long	0                       ## Offset Into Abbrev. Section

	.byte	1                       ## Abbrev [1] root
	.asciz	"hand-written"          ## DW_AT_producer
	.short	0x0004                  ## DW_AT_language (DW_LANG_C_plus_plus)
	.asciz	"dwz-common.h"          ## DW_AT_name

	.byte	2                       ## Abbrev [2] DW_TAG_subprogram
	.asciz	"foo"                   ## DW_AT_name
	.long	Lfootype - Lsection_info ## DW_AT_type
	.quad	Lfunc_begin0            ## DW_AT_low_pc
	.long	Lfunc_end0 - Lfunc_begin0 ## DW_AT_high_pc

	.byte	3                       ## Abbrev [3] DW_TAG_formal_parameter
	.asciz	"b"                     ## DW_AT_name
	.long	Lbartype - Lsection_info ## DW_AT_type

	.byte	0                       ## End Of Children Mark (subprogram)

Lfootype:
	.byte	4                       ## Abbrev [4] DW_TAG_structure_type
	.asciz	"Foo"                   ## DW_AT_name
	.byte	8                       ## DW_AT_byte_size

	.byte	5                       ## Abbrev [5] DW_TAG_namespace
	.asciz	"ns"                    ## DW_AT_name

Lbartype:
	.byte	4                       ## Abbrev [4] DW_TAG_structure_type
	.asciz	"Bar"                   ## DW_AT_name
	.byte	4                       ## DW_AT_byte_size

	.byte	0                       ## End Of Children Mark (namespace)

	.byte	0                       ## End Of Children Mark (root)
Lcu_end:
