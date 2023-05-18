# REQUIRES: x86-registered-target

# RUN: llvm-mc --filetype=obj --triple=x86_64-pc-linux %s -o %t.o -g

# RUN: llvm-dwarfdump -v %t.o | FileCheck %s

# CHECK:      [1] DW_TAG_compile_unit	DW_CHILDREN_no
# CHECK-NEXT: DW_AT_language	DW_FORM_implicit_const	29

# CHECK:      0x0000000c: DW_TAG_compile_unit [1]  
# CHECK-NEXT: DW_AT_language [DW_FORM_implicit_const]	(DW_LANG_C11)

	.section	.debug_info,"",@progbits
.Ldebug_info0:
	.long	.Ldebug_info0_end - .Ldebug_info0_start	# Length of Compilation Unit Info
.Ldebug_info0_start:
	.value	0x5	# DWARF version number
	.byte	0x1	# DW_UT_compile
	.byte	0x8	# Pointer Size (in bytes)
	.long	.Ldebug_abbrev0	# Offset Into Abbrev. Section
	.uleb128 0x1	# (DIE DW_TAG_compile_unit)
			# DW_AT_language
.Ldebug_info0_end:
	.section	.debug_abbrev,"",@progbits
.Ldebug_abbrev0:
	.uleb128 0x1	# (abbrev code)
	.uleb128 0x11	# (TAG: DW_TAG_compile_unit)
	.byte	0x0	# DW_children_no
	.uleb128 0x13	# (DW_AT_language)
	.uleb128 0x21   # (DW_FORM_implicit_const)
	.sleb128 0x1d
	.byte	0
	.byte	0
	.byte	0
