# Test handling of DWARF5 DW_FORM_implicit_const as used by GCC.

# RUN: llvm-mc -filetype=obj -o %t -triple x86_64-pc-linux %s
# RUN: %lldb %t -o "expression -T -- variable_implicit_const" \
# RUN:   -o exit | FileCheck %s

# Failing case was:
# error: need to add support for DW_TAG_base_type 'int' encoded with DW_ATE = 0x5, bit_size = 0
# CHECK: (int) $0 = 0

	.bss
	.globl	variable_implicit_const
	.type	variable_implicit_const, @object
	.size	variable_implicit_const, 4
variable_implicit_const:
	.long	0
.Lvariable_implicit_const_end:
	.section	.debug_info,"",@progbits
.Ldebug_info0:
	.long	.Ldebug_info0_end - .Ldebug_info0_start	# Length of Compilation Unit Info
.Ldebug_info0_start:
	.value	0x5	# DWARF version number
	.byte	0x1	# DW_UT_compile
	.byte	0x8	# Pointer Size (in bytes)
	.long	.Ldebug_abbrev0	# Offset Into Abbrev. Section
	.uleb128 0x1	# (DIE DW_TAG_compile_unit)
			# DW_AT_producer
	.string	"GNU C17 11.0.0 20210210 (Red Hat 11.0.0-0) -mtune=generic -march=x86-64 -g"
	.byte	0x1d	# DW_AT_language
			# DW_AT_name
	.string	"var4.c"
	.uleb128 0x2	# (DIE DW_TAG_variable)
			# DW_AT_name
	.string	"variable_implicit_const"
	.long	.Ltype_int - .Ldebug_info0	# DW_AT_type
			# DW_AT_external
	.uleb128 0x9	# DW_AT_location
	.byte	0x3	# DW_OP_addr
	.quad	variable_implicit_const
.Ltype_int:
	.uleb128 0x3	# (DIE DW_TAG_base_type)
			# DW_AT_byte_size
	.byte	0x5	# DW_AT_encoding
	.ascii "int\0"	# DW_AT_name
	.byte	0	# end of children of DIE DW_TAG_compile_unit
.Ldebug_info0_end:
	.section	.debug_abbrev,"",@progbits
.Ldebug_abbrev0:
	.uleb128 0x1	# (abbrev code)
	.uleb128 0x11	# (TAG: DW_TAG_compile_unit)
	.byte	0x1	# DW_children_yes
	.uleb128 0x25	# (DW_AT_producer)
	.uleb128 0x8	# (DW_FORM_string)
	.uleb128 0x13	# (DW_AT_language)
	.uleb128 0xb	# (DW_FORM_data1)
	.uleb128 0x3	# (DW_AT_name)
	.uleb128 0x8	# (DW_FORM_string)
	.byte	0
	.byte	0
	.uleb128 0x2	# (abbrev code)
	.uleb128 0x34	# (TAG: DW_TAG_variable)
	.byte	0	# DW_children_no
	.uleb128 0x3	# (DW_AT_name)
	.uleb128 0x8	# (DW_FORM_string)
	.uleb128 0x49	# (DW_AT_type)
	.uleb128 0x13	# (DW_FORM_ref4)
	.uleb128 0x3f	# (DW_AT_external)
	.uleb128 0x19	# (DW_FORM_flag_present)
	.uleb128 0x2	# (DW_AT_location)
	.uleb128 0x18	# (DW_FORM_exprloc)
	.byte	0
	.byte	0
	.uleb128 0x3	# (abbrev code)
	.uleb128 0x24	# (TAG: DW_TAG_base_type)
	.byte	0	# DW_children_no
	.uleb128 0xb	# (DW_AT_byte_size)
	.uleb128 0x21	# (DW_FORM_implicit_const)
	.sleb128 .Lvariable_implicit_const_end - variable_implicit_const
	.uleb128 0x3e	# (DW_AT_encoding)
	.uleb128 0xb	# (DW_FORM_data1)
	.uleb128 0x3	# (DW_AT_name)
	.uleb128 0x8	# (DW_FORM_string)
	.byte	0
	.byte	0
	.byte	0
