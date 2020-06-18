# RUN: llvm-mc -triple x86_64-pc-linux %s -filetype=obj | \
# RUN:   not llvm-dwarfdump -verify - | FileCheck %s

# CHECK: parsing .debug_names header at 0x0: cannot read header augmentation

	.section	.debug_str,"MS",@progbits,1
.Lstring_producer:
	.asciz	"Hand-written dwarf"

	.section	.debug_abbrev,"",@progbits
.Lsection_abbrev:
	.byte	1                       # Abbreviation Code
	.byte	17                      # DW_TAG_compile_unit
	.byte	1                       # DW_CHILDREN_yes
	.byte	37                      # DW_AT_producer
	.byte	14                      # DW_FORM_strp
	.byte	19                      # DW_AT_language
	.byte	5                       # DW_FORM_data2
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	0                       # EOM(3)

	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Lcu_end0-.Lcu_start0   # Length of Unit
.Lcu_start0:
	.short	4                       # DWARF version number
	.long	.Lsection_abbrev        # Offset Into Abbrev. Section
	.byte	8                       # Address Size (in bytes)
	.byte	1                       # Abbrev [1] DW_TAG_compile_unit
	.long	.Lstring_producer       # DW_AT_producer
	.short	12                      # DW_AT_language
	.byte	0                       # End Of Children Mark
.Lcu_end0:

	.section	.debug_names,"",@progbits
	.long	.Lnames_end0-.Lnames_start0 # Header: contribution length
.Lnames_start0:
	.short	5                       # Header: version
	.short	0                       # Header: padding
	.long	1                       # Header: compilation unit count
	.long	0                       # Header: local type unit count
	.long	0                       # Header: foreign type unit count
	.long	2                       # Header: bucket count
	.long	3                       # Header: name count
	.long	47                      # Header: abbreviation table size
	.long	8                       # Header: augmentation length
        .asciz "short"                  # Header: augmentation
.Lnames_end0:
