# RUN: llvm-mc -triple x86_64-pc-linux %s -filetype=obj -o - | llvm-dwarfdump -debug-names - | FileCheck %s
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"foo"
.Linfo_string1:
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
	.long	0                       # Header: bucket count
	.long	1                       # Header: name count
	.long	.Lnames_abbrev_end0-.Lnames_abbrev_start0 # Header: abbreviation table size
	.long	0                       # Header: augmentation length
	.long	.Lcu_begin0             # Compilation unit 0
	.long	.Linfo_string0          # String 1: foo
	.long	.Lnames0-.Lnames_entries0 # Offset 1
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
	.p2align	2
        .byte   42                      # Deliberately misalign the next contribution
.Lnames_end0:

	.long	.Lnames_end1-.Lnames_start1 # Header: contribution length
.Lnames_start1:
	.short	5                       # Header: version
	.short	0                       # Header: padding
	.long	1                       # Header: compilation unit count
	.long	0                       # Header: local type unit count
	.long	0                       # Header: foreign type unit count
	.long	0                       # Header: bucket count
	.long	1                       # Header: name count
	.long	.Lnames_abbrev_end1-.Lnames_abbrev_start1 # Header: abbreviation table size
	.long	0                       # Header: augmentation length
	.long	.Lcu_begin1             # Compilation unit 0
	.long	.Linfo_string1          # String 1: bar
	.long	.Lnames1-.Lnames_entries1 # Offset 1
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
.Lnames1:
	.byte	52                      # Abbrev code
	.long	.Ldie1-.Lcu_begin1      # DW_IDX_die_offset
	.long	0                       # End of list: bar
	.p2align	2
.Lnames_end1:
# CHECK:     Name Index @ 0x0
# CHECK:       Name 1 {
# CHECK-NEXT:    String: 0x00000000 "foo"
# CHECK-NEXT:    Entry @ 0x37 {
# CHECK-NEXT:      Abbrev: 0x2e
# CHECK-NEXT:      Tag: DW_TAG_subprogram
# CHECK-NEXT:      DW_IDX_die_offset: 0x00000001
# CHECK-NEXT:    }
# CHECK-NEXT:  }

# CHECK:     Name Index @ 0x41
# CHECK:       Name 1 {
# CHECK-NEXT:    String: 0x00000004 "bar"
# CHECK-NEXT:    Entry @ 0x78 {
# CHECK-NEXT:      Abbrev: 0x34
# CHECK-NEXT:      Tag: DW_TAG_variable
# CHECK-NEXT:      DW_IDX_die_offset: 0x00000001
# CHECK-NEXT:    }
# CHECK-NEXT:  }
