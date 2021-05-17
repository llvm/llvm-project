# Test handling of DWARF5 DW_FORM_line_strp from .debug_info as used by GCC.

# UNSUPPORTED: system-darwin, system-windows

# RUN: llvm-mc -filetype=obj -o %t -triple x86_64-pc-linux %s
# RUN: %lldb %t -o "p main" \
# RUN:   -o exit | FileCheck %s

# CHECK: (void (*)()) $0 = 0x0000000000000000

	.text
.Ltext0:
	.globl	main
	.type	main, @function
main:
.LFB0:
.LM1:
        .long	0
.LM2:
        .long	0
.LFE0:
	.size	main, .-main
.Letext0:
	.section	.debug_info,"",@progbits
.Ldebug_info0:
	.long	.Ldebug_info0_end - .Ldebug_info0_start	# Length of Compilation Unit Info
.Ldebug_info0_start:
	.value	0x5	# DWARF version number
	.byte	0x1	# DW_UT_compile
	.byte	0x8	# Pointer Size (in bytes)
	.long	.Ldebug_abbrev0	# Offset Into Abbrev. Section
	.uleb128 0x1	# (DIE DW_TAG_compile_unit)
	.long	.LASF2	# DW_AT_producer: "GNU C17 11.0.0 20210210 (Red Hat 11.0.0-0) -mtune=generic -march=x86-64 -gdwarf-5 -gno-as-loc-support"
	.byte	0x1d	# DW_AT_language
	.long	.LASF0	# DW_AT_name: "main.c"
	.long	.LASF1	# DW_AT_comp_dir: ""
	.quad	.Ltext0	# DW_AT_low_pc
	.quad	.Letext0-.Ltext0	# DW_AT_high_pc
	.long	.Ldebug_line0	# DW_AT_stmt_list
	.uleb128 0x2	# (DIE DW_TAG_subprogram)
			# DW_AT_external
	.long	.LASF3	# DW_AT_name: "main"
	.byte	0x1	# DW_AT_decl_file (main.c)
	.byte	0x1	# DW_AT_decl_line
	.quad	.LFB0	# DW_AT_low_pc
	.quad	.LFE0-.LFB0	# DW_AT_high_pc
	.byte	0	# end of children of DIE DW_TAG_compile_unit
.Ldebug_info0_end:
	.section	.debug_abbrev,"",@progbits
.Ldebug_abbrev0:
	.uleb128 0x1	# (abbrev code)
	.uleb128 0x11	# (TAG: DW_TAG_compile_unit)
	.byte	0x1	# DW_children_yes
	.uleb128 0x25	# (DW_AT_producer)
	.uleb128 0xe	# (DW_FORM_strp)
	.uleb128 0x13	# (DW_AT_language)
	.uleb128 0xb	# (DW_FORM_data1)
	.uleb128 0x3	# (DW_AT_name)
	.uleb128 0x1f	# (DW_FORM_line_strp)
	.uleb128 0x1b	# (DW_AT_comp_dir)
	.uleb128 0x1f	# (DW_FORM_line_strp)
	.uleb128 0x11	# (DW_AT_low_pc)
	.uleb128 0x1	# (DW_FORM_addr)
	.uleb128 0x12	# (DW_AT_high_pc)
	.uleb128 0x7	# (DW_FORM_data8)
	.uleb128 0x10	# (DW_AT_stmt_list)
	.uleb128 0x17	# (DW_FORM_sec_offset)
	.byte	0
	.byte	0
	.uleb128 0x2	# (abbrev code)
	.uleb128 0x2e	# (TAG: DW_TAG_subprogram)
	.byte	0	# DW_children_no
	.uleb128 0x3f	# (DW_AT_external)
	.uleb128 0x19	# (DW_FORM_flag_present)
	.uleb128 0x3	# (DW_AT_name)
	.uleb128 0xe	# (DW_FORM_strp)
	.uleb128 0x3a	# (DW_AT_decl_file)
	.uleb128 0xb	# (DW_FORM_data1)
	.uleb128 0x3b	# (DW_AT_decl_line)
	.uleb128 0xb	# (DW_FORM_data1)
	.uleb128 0x11	# (DW_AT_low_pc)
	.uleb128 0x1	# (DW_FORM_addr)
	.uleb128 0x12	# (DW_AT_high_pc)
	.uleb128 0x7	# (DW_FORM_data8)
	.byte	0
	.byte	0
	.byte	0
	.section	.debug_line,"",@progbits
.Ldebug_line0:
	.long	.LELT0-.LSLT0	# Length of Source Line Info
.LSLT0:
	.value	0x5	# DWARF version number
	.byte	0x8	# Address Size
	.byte	0	# Segment Size
	.long	.LELTP0-.LASLTP0	# Prolog Length
.LASLTP0:
	.byte	0x1	# Minimum Instruction Length
	.byte	0x1	# Maximum Operations Per Instruction
	.byte	0x1	# Default is_stmt_start flag
	.byte	0xf6	# Line Base Value (Special Opcodes)
	.byte	0xf2	# Line Range Value (Special Opcodes)
	.byte	0xd	# Special Opcode Base
	.byte	0	# opcode: 0x1 has 0 args
	.byte	0x1	# opcode: 0x2 has 1 args
	.byte	0x1	# opcode: 0x3 has 1 args
	.byte	0x1	# opcode: 0x4 has 1 args
	.byte	0x1	# opcode: 0x5 has 1 args
	.byte	0	# opcode: 0x6 has 0 args
	.byte	0	# opcode: 0x7 has 0 args
	.byte	0	# opcode: 0x8 has 0 args
	.byte	0x1	# opcode: 0x9 has 1 args
	.byte	0	# opcode: 0xa has 0 args
	.byte	0	# opcode: 0xb has 0 args
	.byte	0x1	# opcode: 0xc has 1 args
	.byte	0x1	# Directory entry format count
	.uleb128 0x1	# DW_LNCT_path
	.uleb128 0x1f	# DW_FORM_line_strp
	.uleb128 0x2	# Directories count
	.long	.LASF1	# Directory Entry: 0: ""
	.long	.LASF4	# Directory Entry: 0: ""
	.byte	0x2	# File name entry format count
	.uleb128 0x1	# DW_LNCT_path
	.uleb128 0x1f	# DW_FORM_line_strp
	.uleb128 0x2	# DW_LNCT_directory_index
	.uleb128 0xb	# DW_FORM_data1
	.uleb128 0x2	# File names count
	.long	.LASF0	# File Entry: 0: "main.c"
	.byte	0
	.long	.LASF5	# File Entry: 0: "main.c"
	.byte	0x1
.LELTP0:
	.byte	0	# set address *.LM1
	.uleb128 0x9
	.byte	0x2
	.quad	.LM1
	.byte	0x1	# copy line 1
	.byte	0x5	# column 12
	.uleb128 0xc	# 12
	.byte	0	# set address *.LM2
	.uleb128 0x9
	.byte	0x2
	.quad	.LM2
	.byte	0x1	# copy line 1
	.byte	0x5	# column 13
	.uleb128 0xd	# 13
	.byte	0	# set address *.Letext0
	.uleb128 0x9
	.byte	0x2
	.quad	.Letext0
	.byte	0	# end sequence
	.uleb128 0x1
	.byte	0x1
.LELT0:
	.section	.debug_str,"MS",@progbits,1
.LASF2:
	.string	"GNU C17 11.0.0 20210210 (Red Hat 11.0.0-0) -mtune=generic -march=x86-64 -gdwarf-5 -gno-as-loc-support"
.LASF3:
	.string	"main"
	.section	.debug_line_str,"MS",@progbits,1
.LASF1:
	.string	""
.LASF4:
	.string	""
.LASF0:
	.string	"main.c"
.LASF5:
	.string	"main.c"
	.ident	"GCC: (GNU) 11.0.0 20210210 (Red Hat 11.0.0-0)"
	.section	.note.GNU-stack,"",@progbits
