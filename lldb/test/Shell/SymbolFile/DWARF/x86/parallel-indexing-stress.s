# Stress-test the parallel indexing of compile units.

# RUN: llvm-mc -triple x86_64-pc-linux %s -o %t -filetype=obj
# RUN: %lldb %t -o "target variable A" -b | FileCheck %s

# CHECK-COUNT-256: A = 47

	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"Hand-written DWARF"
.Lname:
	.asciz	"A"
.Linfo_string4:
	.asciz	"int"                   # string offset=95

	.section	.debug_abbrev,"",@progbits
	.byte	1                       # Abbreviation Code
	.byte	17                      # DW_TAG_compile_unit
	.byte	1                       # DW_CHILDREN_yes
	.byte	37                      # DW_AT_producer
	.byte	14                      # DW_FORM_strp
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	2                       # Abbreviation Code
	.byte	52                      # DW_TAG_variable
	.byte	0                       # DW_CHILDREN_no
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	73                      # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	2                       # DW_AT_location
	.byte	24                      # DW_FORM_exprloc
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	3                       # Abbreviation Code
	.byte	36                      # DW_TAG_base_type
	.byte	0                       # DW_CHILDREN_no
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	62                      # DW_AT_encoding
	.byte	11                      # DW_FORM_data1
	.byte	11                      # DW_AT_byte_size
	.byte	11                      # DW_FORM_data1
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	0                       # EOM(3)

.macro generate_unit
	.data
A\@:
	.long	47

	.section	.debug_str,"MS",@progbits,1

	.section	.debug_info,"",@progbits
.Lcu_begin\@:
	.long	.Ldebug_info_end\@-.Ldebug_info_start\@ # Length of Unit
.Ldebug_info_start\@:
	.short	4                       # DWARF version number
	.long	.debug_abbrev           # Offset Into Abbrev. Section
	.byte	8                       # Address Size (in bytes)
	.byte	1                       # Abbrev [1] 0xb:0x30 DW_TAG_compile_unit
	.long	.Linfo_string0          # DW_AT_producer
	.byte	2                       # Abbrev [2] 0x1e:0x15 DW_TAG_variable
	.long	.Lname                  # DW_AT_name
	.long	.Ltype\@-.Lcu_begin\@   # DW_AT_type
	.byte	9                       # DW_AT_location
	.byte	3
	.quad	A\@
.Ltype\@:
	.byte	3                       # Abbrev [3] 0x33:0x7 DW_TAG_base_type
	.long	.Linfo_string4          # DW_AT_name
	.byte	5                       # DW_AT_encoding
	.byte	4                       # DW_AT_byte_size
	.byte	0                       # End Of Children Mark
.Ldebug_info_end\@:

.endm

.rept 256
generate_unit
.endr
