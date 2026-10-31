# Check that the verifier warns about local types and imported entities that
# are placed into an out-of-line instance of a subprogram (a subprogram with
# DW_AT_abstract_origin) but are missing from its abstract origin. Entities that
# are present in both, or that refer to their abstract counterpart, are fine.

# RUN: llvm-mc %s -filetype=obj -triple=x86_64-pc-linux -o %t.o
# RUN: llvm-dwarfdump -verify %t.o | FileCheck %s \
# RUN:   --implicit-check-not=warning: --implicit-check-not=error:

# CHECK:      warning: DW_TAG_structure_type in out-of-line instance 0x00000044 of a subprogram is missing from its abstract origin 0x00000016:
# CHECK-EMPTY:
# CHECK-NEXT: 0x00000051: DW_TAG_structure_type
# CHECK-NEXT:               DW_AT_name {{.*}}("OnlyHere")

# CHECK:      warning: DW_TAG_enumeration_type in out-of-line instance 0x00000044 of a subprogram is missing from its abstract origin 0x00000016:
# CHECK-EMPTY:
# CHECK-NEXT: 0x0000005b: DW_TAG_enumeration_type
# CHECK-NEXT:               DW_AT_name {{.*}}("OnlyHereEnum")

# CHECK:      warning: DW_TAG_imported_module in out-of-line instance 0x00000044 of a subprogram is missing from its abstract origin 0x00000016:
# CHECK-EMPTY:
# CHECK-NEXT: 0x00000074: DW_TAG_imported_module
# CHECK-NEXT:               DW_AT_import {{.*}} "ns"
# CHECK-NEXT:               DW_AT_decl_line {{.*}} (7)

# CHECK:      warning: DW_TAG_union_type in out-of-line instance 0x00000044 of a subprogram is missing from its abstract origin 0x00000016:
# CHECK-EMPTY:
# CHECK-NEXT: 0x0000008c: DW_TAG_union_type
# CHECK-NEXT:               DW_AT_name {{.*}}("BlockOnlyHere")

# CHECK: No errors.

	.section	.debug_abbrev,"",@progbits
	.uleb128 1                      # Abbrev 1
	.uleb128 0x11                   # DW_TAG_compile_unit
	.byte	1                       # DW_CHILDREN_yes
	.uleb128 0x03                   # DW_AT_name
	.uleb128 0x08                   # DW_FORM_string
	.byte	0, 0
	.uleb128 2                      # Abbrev 2
	.uleb128 0x39                   # DW_TAG_namespace
	.byte	0                       # DW_CHILDREN_no
	.uleb128 0x03                   # DW_AT_name
	.uleb128 0x08                   # DW_FORM_string
	.byte	0, 0
	.uleb128 3                      # Abbrev 3
	.uleb128 0x2e                   # DW_TAG_subprogram
	.byte	1                       # DW_CHILDREN_yes
	.uleb128 0x03                   # DW_AT_name
	.uleb128 0x08                   # DW_FORM_string
	.uleb128 0x20                   # DW_AT_inline
	.uleb128 0x0b                   # DW_FORM_data1
	.byte	0, 0
	.uleb128 4                      # Abbrev 4
	.uleb128 0x13                   # DW_TAG_structure_type
	.byte	0                       # DW_CHILDREN_no
	.uleb128 0x03                   # DW_AT_name
	.uleb128 0x08                   # DW_FORM_string
	.byte	0, 0
	.uleb128 5                      # Abbrev 5
	.uleb128 0x3a                   # DW_TAG_imported_module
	.byte	0                       # DW_CHILDREN_no
	.uleb128 0x18                   # DW_AT_import
	.uleb128 0x13                   # DW_FORM_ref4
	.uleb128 0x3b                   # DW_AT_decl_line
	.uleb128 0x0b                   # DW_FORM_data1
	.byte	0, 0
	.uleb128 6                      # Abbrev 6
	.uleb128 0x0b                   # DW_TAG_lexical_block
	.byte	1                       # DW_CHILDREN_yes
	.byte	0, 0
	.uleb128 7                      # Abbrev 7
	.uleb128 0x16                   # DW_TAG_typedef
	.byte	0                       # DW_CHILDREN_no
	.uleb128 0x03                   # DW_AT_name
	.uleb128 0x08                   # DW_FORM_string
	.byte	0, 0
	.uleb128 8                      # Abbrev 8
	.uleb128 0x2e                   # DW_TAG_subprogram
	.byte	1                       # DW_CHILDREN_yes
	.uleb128 0x31                   # DW_AT_abstract_origin
	.uleb128 0x13                   # DW_FORM_ref4
	.byte	0, 0
	.uleb128 9                      # Abbrev 9
	.uleb128 0x04                   # DW_TAG_enumeration_type
	.byte	0                       # DW_CHILDREN_no
	.uleb128 0x03                   # DW_AT_name
	.uleb128 0x08                   # DW_FORM_string
	.byte	0, 0
	.uleb128 10                     # Abbrev 10
	.uleb128 0x0b                   # DW_TAG_lexical_block
	.byte	1                       # DW_CHILDREN_yes
	.uleb128 0x31                   # DW_AT_abstract_origin
	.uleb128 0x13                   # DW_FORM_ref4
	.byte	0, 0
	.uleb128 11                     # Abbrev 11
	.uleb128 0x17                   # DW_TAG_union_type
	.byte	0                       # DW_CHILDREN_no
	.uleb128 0x03                   # DW_AT_name
	.uleb128 0x08                   # DW_FORM_string
	.byte	0, 0
	.uleb128 12                     # Abbrev 12
	.uleb128 0x13                   # DW_TAG_structure_type
	.byte	0                       # DW_CHILDREN_no
	.uleb128 0x31                   # DW_AT_abstract_origin
	.uleb128 0x13                   # DW_FORM_ref4
	.byte	0, 0
	.uleb128 13                     # Abbrev 13
	.uleb128 0x2e                   # DW_TAG_subprogram
	.byte	1                       # DW_CHILDREN_yes
	.uleb128 0x03                   # DW_AT_name
	.uleb128 0x08                   # DW_FORM_string
	.byte	0, 0
	.byte	0                       # End of abbreviations

	.section	.debug_info,"",@progbits
.Lcu_begin:
	.long	.Lcu_end - .Lcu_version # Unit length
.Lcu_version:
	.short	4                       # DWARF version
	.long	.debug_abbrev           # Abbrev offset
	.byte	8                       # Address size
	.uleb128 1                      # DW_TAG_compile_unit
	.asciz	"t.cpp"

.Lns:
	.uleb128 2                      # DW_TAG_namespace
	.asciz	"ns"

## The abstract origin.
.Lf:
	.uleb128 3                      # DW_TAG_subprogram
	.asciz	"f"
	.byte	1                       # DW_INL_inlined
	.uleb128 4                      # DW_TAG_structure_type
	.asciz	"InBoth"
.LInAbstract:
	.uleb128 4                      # DW_TAG_structure_type
	.asciz	"InAbstract"
	.uleb128 5                      # DW_TAG_imported_module
	.long	.Lns - .Lcu_begin
	.byte	3                       # DW_AT_decl_line
.Lf_block:
	.uleb128 6                      # DW_TAG_lexical_block
	.uleb128 7                      # DW_TAG_typedef
	.asciz	"BlockInBoth"
	.byte	0                       # End of lexical block
	.byte	0                       # End of f

## The out-of-line instance.
	.uleb128 8                      # DW_TAG_subprogram
	.long	.Lf - .Lcu_begin
	.uleb128 4                      # DW_TAG_structure_type
	.asciz	"InBoth"
	.uleb128 4                      # DW_TAG_structure_type
	.asciz	"OnlyHere"
	.uleb128 9                      # DW_TAG_enumeration_type
	.asciz	"OnlyHereEnum"
	.uleb128 12                     # DW_TAG_structure_type
	.long	.LInAbstract - .Lcu_begin
	.uleb128 5                      # DW_TAG_imported_module
	.long	.Lns - .Lcu_begin
	.byte	3                       # DW_AT_decl_line
	.uleb128 5                      # DW_TAG_imported_module
	.long	.Lns - .Lcu_begin
	.byte	7                       # DW_AT_decl_line
	.uleb128 10                     # DW_TAG_lexical_block
	.long	.Lf_block - .Lcu_begin
	.uleb128 7                      # DW_TAG_typedef
	.asciz	"BlockInBoth"
	.uleb128 11                     # DW_TAG_union_type
	.asciz	"BlockOnlyHere"
	.byte	0                       # End of lexical block
	.byte	0                       # End of the out-of-line instance

## A subprogram that is not an instance of another one is not checked.
	.uleb128 13                     # DW_TAG_subprogram
	.asciz	"g"
	.uleb128 4                      # DW_TAG_structure_type
	.asciz	"NotAnInstance"
	.byte	0                       # End of g

	.byte	0                       # End of compile unit
.Lcu_end:
