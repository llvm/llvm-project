# Test that we don't crash when parsing slightly invalid DWARF.  The compile
# unit in this file sets DW_CHILDREN_no, but it still includes an
# end-of-children marker in its contribution.

# RUN: llvm-mc -triple x86_64-pc-linux %s -filetype=obj > %t.o
# RUN: lldb-test symbols %t.o

	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"Hand-written DWARF"
.Linfo_string1:
	.asciz	"-"
.Linfo_string2:
	.asciz	"/tmp"

	.section	.debug_abbrev,"",@progbits
	.byte	1                       # Abbreviation Code
	.byte	17                      # DW_TAG_compile_unit
	.byte	0                       # DW_CHILDREN_no
	.byte	37                      # DW_AT_producer
	.byte	14                      # DW_FORM_strp
	.byte	19                      # DW_AT_language
	.byte	5                       # DW_FORM_data2
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	27                      # DW_AT_comp_dir
	.byte	14                      # DW_FORM_strp
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	0                       # EOM(3)

	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Lcu_length_end-.Lcu_length_start # Length of Unit
.Lcu_length_start:
	.short	4                       # DWARF version number
	.long	.debug_abbrev           # Offset Into Abbrev. Section
	.byte	8                       # Address Size (in bytes)
	.byte	1                       # Abbrev [1] 0xb:0x30 DW_TAG_compile_unit
	.long	.Linfo_string0          # DW_AT_producer
	.short	12                      # DW_AT_language
	.long	.Linfo_string1          # DW_AT_name
	.long	.Linfo_string2          # DW_AT_comp_dir
	.byte	0                       # Bogus End Of Children Mark
.Lcu_length_end:
