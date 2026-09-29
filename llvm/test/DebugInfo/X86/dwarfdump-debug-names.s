# RUN: llvm-mc -triple x86_64-pc-linux %s -filetype=obj -o - | llvm-dwarfdump -debug-names - | FileCheck %s
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"foo"
.Linfo_string1:
	.asciz	"_Z3foov"
.Linfo_string2:
	.asciz	"bar"

# Fake .debug_info. We just need it for the offsets to two "compile units" and
# two "DIEs"
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.byte	0
.Ldie0:
	.byte	0
.Lcu_begin1:
	.byte	0
.Ldie1:
	.byte	0

	.section	.debug_names,"",@progbits
	.long	.Lnames_end0-.Lnames_start0 # Header: contribution length
.Lnames_start0:
	.short	5                       # Header: version
	.short	0                       # Header: padding
	.long	1                       # Header: compilation unit count
	.long	0                       # Header: local type unit count
	.long	0                       # Header: foreign type unit count
	.long	2                       # Header: bucket count
	.long	2                       # Header: name count
	.long	.Lnames_abbrev_end0-.Lnames_abbrev_start0 # Header: abbreviation table size
	.long	0                       # Header: augmentation length
	.long	.Lcu_begin0             # Compilation unit 0
	.long	0                       # Bucket 0
	.long	1                       # Bucket 1
	.long	193491849               # Hash in Bucket 1
	.long	-1257882357             # Hash in Bucket 1
	.long	.Linfo_string0          # String in Bucket 1: foo
	.long	.Linfo_string1          # String in Bucket 1: _Z3foov
	.long	.Lnames0-.Lnames_entries0 # Offset in Bucket 1
	.long	.Lnames1-.Lnames_entries0 # Offset in Bucket 1
.Lnames_abbrev_start0:
	.byte	46                      # Abbrev code
	.byte	46                      # DW_TAG_subprogram
	.byte	3                       # DW_IDX_die_offset
	.byte	19                      # DW_FORM_ref4
	.byte	0                       # End of abbrev
	.byte	0                       # End of abbrev
	.byte	0                       # End of abbrev list
.Lnames_abbrev_end0:
.Lnames_entries0:
.Lnames0:
	.byte	46                      # Abbrev code
	.long	.Ldie0-.Lcu_begin0      # DW_IDX_die_offset
	.long	0                       # End of list: foo
.Lnames1:
	.byte	46                      # Abbrev code
	.long	.Ldie0-.Lcu_begin0      # DW_IDX_die_offset
	.long	0                       # End of list: _Z3foov
	.p2align	2
.Lnames_end0:

	.long	.Lnames_end1-.Lnames_start1 # Header: contribution length
.Lnames_start1:
	.short	5                       # Header: version
	.short	0                       # Header: padding
	.long	1                       # Header: compilation unit count
	.long	0                       # Header: local type unit count
	.long	0                       # Header: foreign type unit count
	.long	1                       # Header: bucket count
	.long	1                       # Header: name count
	.long	.Lnames_abbrev_end1-.Lnames_abbrev_start1 # Header: abbreviation table size
	.long	0                       # Header: augmentation length
	.long	.Lcu_begin1             # Compilation unit 0
	.long	1                       # Bucket 0
	.long	193487034               # Hash in Bucket 0
	.long	.Linfo_string2          # String in Bucket 0: bar
	.long	.Lnames2-.Lnames_entries1 # Offset in Bucket 0
.Lnames_abbrev_start1:
	.byte	52                      # Abbrev code
	.byte	52                      # DW_TAG_variable
	.byte	3                       # DW_IDX_die_offset
	.byte	19                      # DW_FORM_ref4
	.byte	0                       # End of abbrev
	.byte	0                       # End of abbrev
	.byte	0                       # End of abbrev list
.Lnames_abbrev_end1:
.Lnames_entries1:
.Lnames2:
	.byte	52                      # Abbrev code
	.long	.Ldie1-.Lcu_begin1      # DW_IDX_die_offset
	.long	0                       # End of list: bar
	.p2align	2
.Lnames_end1:

	.long	0xffffffff                  # DWARF64 mark
	.quad	.Lnames_end2-.Lnames_start2 # Length
.Lnames_start2:
	.short	5                           # Version
	.space	2                           # Padding
	.long	1                           # CU count
	.long	1                           # Local TU count
	.long	1                           # Foreign TU count
	.long	1                           # Bucket count
	.long	1                           # Name count
	.long	.Lnames_abbrev_end2-.Lnames_abbrev_start2   # Abbreviations table size
	.long	0                           # Augmentation string size
	.quad	0xcc00cccccccc              # CU0 offset
	.quad	0xaa00aaaaaaaa              # Local TU0 offset
	.quad	0xffffff00ffffffff          # Foreign TU2 signature
	.long	1                           # Bucket 0
	.long	0xb887389                   # Hash in Bucket 0
	.quad	.Linfo_string0              # String in Bucket 0: foo
	.quad	.Lnames3-.Lnames_entries2   # Offset in Bucket 0
.Lnames_abbrev_start2:
	.byte	0x01                        # Abbrev code
	.byte	0x24                        # DW_TAG_base_type
	.byte	0x02                        # DW_IDX_type_unit
	.byte	0x06                        # DW_FORM_data4
	.byte	0x05                        # DW_IDX_type_hash
	.byte	0x07                        # DW_FORM_data8
	.byte	0x00                        # End of abbrev
	.byte	0x00                        # End of abbrev
	.byte	0x00                        # End of abbrev list
.Lnames_abbrev_end2:
.Lnames_entries2:
.Lnames3:
	.byte	0x01                        # Abbrev code
	.long	1                           # DW_IDX_type_unit
	.quad	0xff03ffffffff              # DW_IDX_type_hash
	.byte   0x00                        # End of list: foo
	.p2align	2
.Lnames_end2:

# CHECK: .debug_names contents:
# CHECK-NEXT: Name Index @ 0x0 {
# CHECK-NEXT:   Header {
# CHECK-NEXT:     Length: 0x60
# CHECK-NEXT:     Format: DWARF32
# CHECK-NEXT:     Version: 5
# CHECK-NEXT:     CU count: 1
# CHECK-NEXT:     Local TU count: 0
# CHECK-NEXT:     Foreign TU count: 0
# CHECK-NEXT:     Bucket count: 2
# CHECK-NEXT:     Name count: 2
# CHECK-NEXT:     Abbreviations table size: 0x7
# CHECK-NEXT:     Augmentation: ''
# CHECK-NEXT:   }
# CHECK-NEXT:   Compilation Unit offsets [
# CHECK-NEXT:     CU[0]: 0x00000000
# CHECK-NEXT:   ]
# CHECK-NEXT:   Abbreviations [
# CHECK-NEXT:     Abbreviation [[ABBREV:0x[0-9a-f]*]] {
# CHECK-NEXT:       Tag: DW_TAG_subprogram
# CHECK-NEXT:       DW_IDX_die_offset: DW_FORM_ref4
# CHECK-NEXT:     }
# CHECK-NEXT:   ]
# CHECK-NEXT:   Bucket 0 [
# CHECK-NEXT:     EMPTY
# CHECK-NEXT:   ]
# CHECK-NEXT:   Bucket 1 [
# CHECK-NEXT:     Name 1 {
# CHECK-NEXT:       Hash: 0xB887389
# CHECK-NEXT:       String: 0x00000000 "foo"
# CHECK-NEXT:       Entry @ 0x4f {
# CHECK-NEXT:         Abbrev: [[ABBREV]]
# CHECK-NEXT:         Tag: DW_TAG_subprogram
# CHECK-NEXT:         DW_IDX_die_offset: 0x00000001
# CHECK-NEXT:       }
# CHECK-NEXT:     }
# CHECK-NEXT:     Name 2 {
# CHECK-NEXT:       Hash: 0xB5063D0B
# CHECK-NEXT:       String: 0x00000004 "_Z3foov"
# CHECK-NEXT:       Entry @ 0x58 {
# CHECK-NEXT:         Abbrev: [[ABBREV]]
# CHECK-NEXT:         Tag: DW_TAG_subprogram
# CHECK-NEXT:         DW_IDX_die_offset: 0x00000001
# CHECK-NEXT:       }
# CHECK-NEXT:     }
# CHECK-NEXT:   ]
# CHECK-NEXT: }
# CHECK-NEXT: Name Index @ 0x64 {
# CHECK-NEXT:   Header {
# CHECK-NEXT:     Length: 0x44
# CHECK-NEXT:     Format: DWARF32
# CHECK-NEXT:     Version: 5
# CHECK-NEXT:     CU count: 1
# CHECK-NEXT:     Local TU count: 0
# CHECK-NEXT:     Foreign TU count: 0
# CHECK-NEXT:     Bucket count: 1
# CHECK-NEXT:     Name count: 1
# CHECK-NEXT:     Abbreviations table size: 0x7
# CHECK-NEXT:     Augmentation: ''
# CHECK-NEXT:   }
# CHECK-NEXT:   Compilation Unit offsets [
# CHECK-NEXT:     CU[0]: 0x00000002
# CHECK-NEXT:   ]
# CHECK-NEXT:   Abbreviations [
# CHECK-NEXT:     Abbreviation [[ABBREV1:0x[0-9a-f]*]] {
# CHECK-NEXT:       Tag: DW_TAG_variable
# CHECK-NEXT:       DW_IDX_die_offset: DW_FORM_ref4
# CHECK-NEXT:     }
# CHECK-NEXT:   ]
# CHECK-NEXT:   Bucket 0 [
# CHECK-NEXT:     Name 1 {
# CHECK-NEXT:       Hash: 0xB8860BA
# CHECK-NEXT:       String: 0x0000000c "bar"
# CHECK-NEXT:       Entry @ 0xa3 {
# CHECK-NEXT:         Abbrev: [[ABBREV1]]
# CHECK-NEXT:         Tag: DW_TAG_variable
# CHECK-NEXT:         DW_IDX_die_offset: 0x00000001
# CHECK-NEXT:       }
# CHECK-NEXT:     }
# CHECK-NEXT:   ]
# CHECK-NEXT: }
# CHECK-NEXT: Name Index @ 0xac {
# CHECK-NEXT:   Header {
# CHECK-NEXT:     Length: 0x68
# CHECK-NEXT:     Format: DWARF64
# CHECK-NEXT:     Version: 5
# CHECK-NEXT:     CU count: 1
# CHECK-NEXT:     Local TU count: 1
# CHECK-NEXT:     Foreign TU count: 1
# CHECK-NEXT:     Bucket count: 1
# CHECK-NEXT:     Name count: 1
# CHECK-NEXT:     Abbreviations table size: 0x9
# CHECK-NEXT:     Augmentation: ''
# CHECK-NEXT:   }
# CHECK-NEXT:   Compilation Unit offsets [
# CHECK-NEXT:     CU[0]: 0xcc00cccccccc
# CHECK-NEXT:   ]
# CHECK-NEXT:   Local Type Unit offsets [
# CHECK-NEXT:     LocalTU[0]: 0xaa00aaaaaaaa
# CHECK-NEXT:   ]
# CHECK-NEXT:   Foreign Type Unit signatures [
# CHECK-NEXT:     ForeignTU[0]: 0xffffff00ffffffff
# CHECK-NEXT:   ]
# CHECK-NEXT:   Abbreviations [
# CHECK-NEXT:     Abbreviation [[ABBREV2:0x[0-9a-f]*]] {
# CHECK-NEXT:       Tag: DW_TAG_base_type
# CHECK-NEXT:       DW_IDX_type_unit: DW_FORM_data4
# CHECK-NEXT:       DW_IDX_type_hash: DW_FORM_data8
# CHECK-NEXT:     }
# CHECK-NEXT:   ]
# CHECK-NEXT:   Bucket 0 [
# CHECK-NEXT:     Name 1 {
# CHECK-NEXT:       Hash: 0xB887389
# CHECK-NEXT:       String: 0x00000000 "foo"
# CHECK-NEXT:       Entry @ 0x111 {
# CHECK-NEXT:         Abbrev: [[ABBREV2]]
# CHECK-NEXT:         Tag: DW_TAG_base_type
# CHECK-NEXT:         DW_IDX_type_unit: 0x00000001
# CHECK-NEXT:         DW_IDX_type_hash: 0x0000ff03ffffffff
# CHECK-NEXT:       }
# CHECK-NEXT:     }
# CHECK-NEXT:   ]
# CHECK-NEXT: }
