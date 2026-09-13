# A Swift type's mangled name, from DW_AT_linkage_name, is indexed as a type
# name alongside the short DW_AT_name by both linkers.

# RUN: llvm-mc -triple x86_64-apple-darwin -filetype=obj %s -o %t.o
# RUN: llvm-dwarfdump --verify %t.o

# RUN: echo '---' > %t.map
# RUN: echo "triple:          'x86_64-apple-darwin'" >> %t.map
# RUN: echo 'objects:'  >> %t.map
# RUN: echo " -  filename: '%t.o'" >> %t.map
# RUN: echo '    symbols:' >> %t.map
# RUN: echo '      - { sym: _foo, objAddr: 0x0, binAddr: 0x10000, size: 0x1 }' >> %t.map
# RUN: echo '...' >> %t.map

# RUN: dsymutil --linker=parallel -accelerator=Dwarf -y %t.map -f -o %t.parallel.dSYM
# RUN: llvm-dwarfdump --verify %t.parallel.dSYM
# RUN: llvm-dwarfdump --debug-names %t.parallel.dSYM > %t.parallel.names

# RUN: dsymutil --linker=classic -accelerator=Dwarf -y %t.map -f -o %t.classic.dSYM
# RUN: llvm-dwarfdump --verify %t.classic.dSYM
# RUN: llvm-dwarfdump --debug-names %t.classic.dSYM > %t.classic.names

# The qualified name hash reaches only the Apple tables.

# RUN: dsymutil --linker=parallel -accelerator=Apple -y %t.map -f -o %t.parallel.apple.dSYM
# RUN: llvm-dwarfdump --apple-types %t.parallel.apple.dSYM > %t.parallel.apple

# RUN: dsymutil --linker=classic -accelerator=Apple -y %t.map -f -o %t.classic.apple.dSYM
# RUN: llvm-dwarfdump --apple-types %t.classic.apple.dSYM > %t.classic.apple

# One FileCheck run per name, so that no prefix depends on the bucket a linker
# assigns to the other names.

# RUN: FileCheck %s --check-prefix=MANGLED --input-file %t.parallel.names
# RUN: FileCheck %s --check-prefix=MANGLED --input-file %t.classic.names
# RUN: FileCheck %s --check-prefix=SHORT --input-file %t.parallel.names
# RUN: FileCheck %s --check-prefix=SHORT --input-file %t.classic.names
# RUN: FileCheck %s --check-prefix=SPEC --input-file %t.parallel.names
# RUN: FileCheck %s --check-prefix=SPEC --input-file %t.classic.names
# RUN: FileCheck %s --check-prefix=DEDUP --input-file %t.parallel.names
# RUN: FileCheck %s --check-prefix=DEDUP --input-file %t.classic.names
# RUN: FileCheck %s --check-prefix=APPLE --input-file %t.parallel.apple
# RUN: FileCheck %s --check-prefix=APPLE --input-file %t.classic.apple

# The tag is checked so that the entry is known to describe the type rather
# than another DIE carrying the same string.

# MANGLED:      String: {{.*}} "$sSSD"
# MANGLED-NEXT: Entry @
# MANGLED-NEXT:   Abbrev:
# MANGLED-NEXT:   Tag: DW_TAG_structure_type

# SHORT:        String: {{.*}} "String"
# SHORT-NEXT:   Entry @
# SHORT-NEXT:     Abbrev:
# SHORT-NEXT:     Tag: DW_TAG_structure_type

# A mangled name reachable only through DW_AT_specification.

# SPEC:         String: {{.*}} "$sSpecD"
# SPEC-NEXT:    Entry @
# SPEC-NEXT:      Abbrev:
# SPEC-NEXT:      Tag: DW_TAG_structure_type

# A type whose two names are equal is indexed once.

# DEDUP:        String: {{.*}} "Same"
# DEDUP-NOT:    String: {{.*}} "Same"

# Atom[3] is djbHash of the mangled name, not the qualified name hash that the
# short name records.

# APPLE:        String: {{.*}} "$sSSD"
# APPLE-NEXT:   Data 0 [
# APPLE-NEXT:     Atom[0]:
# APPLE-NEXT:     Atom[1]: 0x0013 (DW_TAG_structure_type)
# APPLE-NEXT:     Atom[2]: 0x00
# APPLE-NEXT:     Atom[3]: 0x0acaede6

	.section	__TEXT,__text,regular,pure_instructions
	.globl	_foo
_foo:
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
	.byte	73                      ## DW_AT_type
	.byte	19                      ## DW_FORM_ref4
	.byte	17                      ## DW_AT_low_pc
	.byte	1                       ## DW_FORM_addr
	.byte	18                      ## DW_AT_high_pc
	.byte	6                       ## DW_FORM_data4
	.byte	0, 0

	.byte	3                       ## Abbreviation Code
	.byte	19                      ## DW_TAG_structure_type
	.byte	0                       ## DW_CHILDREN_no
	.byte	3                       ## DW_AT_name
	.byte	8                       ## DW_FORM_string
	.byte	110                     ## DW_AT_linkage_name
	.byte	8                       ## DW_FORM_string
	.byte	11                      ## DW_AT_byte_size
	.byte	11                      ## DW_FORM_data1
	.byte	0xe6, 0x7f              ## DW_AT_APPLE_runtime_class (0x3fe6)
	.byte	5                       ## DW_FORM_data2
	.byte	0, 0

	.byte	4                       ## Abbreviation Code
	.byte	5                       ## DW_TAG_formal_parameter
	.byte	0                       ## DW_CHILDREN_no
	.byte	73                      ## DW_AT_type
	.byte	19                      ## DW_FORM_ref4
	.byte	0, 0

	.byte	5                       ## Abbreviation Code
	.byte	19                      ## DW_TAG_structure_type
	.byte	0                       ## DW_CHILDREN_no
	.byte	3                       ## DW_AT_name
	.byte	8                       ## DW_FORM_string
	.byte	110                     ## DW_AT_linkage_name
	.byte	8                       ## DW_FORM_string
	.byte	60                      ## DW_AT_declaration
	.byte	25                      ## DW_FORM_flag_present
	.byte	0, 0

	.byte	6                       ## Abbreviation Code
	.byte	19                      ## DW_TAG_structure_type
	.byte	0                       ## DW_CHILDREN_no
	.byte	71                      ## DW_AT_specification
	.byte	19                      ## DW_FORM_ref4
	.byte	11                      ## DW_AT_byte_size
	.byte	11                      ## DW_FORM_data1
	.byte	0xe6, 0x7f              ## DW_AT_APPLE_runtime_class (0x3fe6)
	.byte	5                       ## DW_FORM_data2
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
	.short	0x001e                  ## DW_AT_language (DW_LANG_Swift)
	.asciz	"swift-mangled-type.swift" ## DW_AT_name

	.byte	2                       ## Abbrev [2] DW_TAG_subprogram
	.asciz	"foo"                   ## DW_AT_name
	.long	Lstring - Lsection_info ## DW_AT_type
	.quad	Lfunc_begin0            ## DW_AT_low_pc
	.long	Lfunc_end0 - Lfunc_begin0 ## DW_AT_high_pc

	.byte	4                       ## Abbrev [4] DW_TAG_formal_parameter
	.long	Lsame - Lsection_info   ## DW_AT_type

	.byte	4                       ## Abbrev [4] DW_TAG_formal_parameter
	.long	Lspec_defn - Lsection_info ## DW_AT_type

	.byte	0                       ## End Of Children Mark (subprogram)

Lstring:
	.byte	3                       ## Abbrev [3] DW_TAG_structure_type
	.asciz	"String"                ## DW_AT_name
	.asciz	"$sSSD"                 ## DW_AT_linkage_name
	.byte	16                      ## DW_AT_byte_size
	.short	0x001e                  ## DW_AT_APPLE_runtime_class (DW_LANG_Swift)

# DW_AT_name and DW_AT_linkage_name are the same string.
Lsame:
	.byte	3                       ## Abbrev [3] DW_TAG_structure_type
	.asciz	"Same"                  ## DW_AT_name
	.asciz	"Same"                  ## DW_AT_linkage_name
	.byte	8                       ## DW_AT_byte_size
	.short	0x001e                  ## DW_AT_APPLE_runtime_class (DW_LANG_Swift)

# Neither name is on the definition, so both come from the declaration.
Lspec_decl:
	.byte	5                       ## Abbrev [5] DW_TAG_structure_type
	.asciz	"Spec"                  ## DW_AT_name
	.asciz	"$sSpecD"               ## DW_AT_linkage_name
                                ## DW_AT_declaration

Lspec_defn:
	.byte	6                       ## Abbrev [6] DW_TAG_structure_type
	.long	Lspec_decl - Lsection_info ## DW_AT_specification
	.byte	8                       ## DW_AT_byte_size
	.short	0x001e                  ## DW_AT_APPLE_runtime_class (DW_LANG_Swift)

	.byte	0                       ## End Of Children Mark (CU)
Lcu_end:
