## This test checks that lldb reads debug info from a dwp file when the dwo_id
## is in a DWARF5 CU header instead of a DW_AT_GNU_dwo_id attribute.

# RUN: llvm-mc --filetype=obj --triple x86_64 %s -o %t --defsym MAIN=1
# RUN: llvm-mc --filetype=obj --triple x86_64 %s -o %t.dwp
# RUN: %lldb %t -o "target variable i" -b | FileCheck %s
# CHECK: (int) i = 42

.ifdef MAIN
## Main file
	.text
	.globl	main                            # -- Begin function main
main:                                   # @main
.Lfunc_begin0:
	pushq	%rbp
	movq	%rsp, %rbp
	xorl	%eax, %eax
	popq	%rbp
	retq
.Lfunc_end0:
                                        # -- End function
	.data
i:
	.long	42                              # 0x2a


	.section	.debug_abbrev,"",@progbits
	.byte	1                               # Abbreviation Code
	.byte	74                              # DW_TAG_skeleton_unit
	.byte	0                               # DW_CHILDREN_no
	.byte	118                             # DW_AT_dwo_name
	.byte	8                               # DW_FORM_string
	.byte	17                              # DW_AT_low_pc
	.byte	27                              # DW_FORM_addrx
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	115                             # DW_AT_addr_base
	.byte	23                              # DW_FORM_sec_offset
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	0                               # EOM(3)


	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	5                             # DWARF version number
	.byte	4                               # DWARF Unit Type
	.byte	8                               # Address Size (in bytes)
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.quad	1026699901672188186             # dwo_id
	.byte	1                               # Abbrev [1] 0x14:0x14 DW_TAG_skeleton_unit
	.asciz "hello.dwo"                    # DW_AT_dwo_name
	.byte	1                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.long	.Laddr_table_base0              # DW_AT_addr_base
.Ldebug_info_end0:

	.section	.debug_addr,"",@progbits
	.long	.Ldebug_addr_end0-.Ldebug_addr_start0 # Length of contribution
.Ldebug_addr_start0:
	.short	5                               # DWARF version number
	.byte	8                               # Address size
	.byte	0                               # Segment selector size
.Laddr_table_base0:
	.quad	i
	.quad	.Lfunc_begin0
.Ldebug_addr_end0:

.else
## DWP file
	.section	.debug_str_offsets.dwo,"e",@progbits
	.long	28
	.short	5
	.short	0


	.section	.debug_str.dwo,"eMS",@progbits,1
.Linfo_string0:
	.asciz	"i"
.Linfo_string1:
	.asciz	"int"
.Linfo_string2:
	.asciz	"main"
.Linfo_string3:
	.asciz	"hand-tuned clang output"
.Linfo_string4:
	.asciz	"hello.c"
.Linfo_string5:
	.asciz	"hello.dwo"


	.section	.debug_str_offsets.dwo,"e",@progbits
	.long	.Linfo_string0-.debug_str.dwo
	.long	.Linfo_string1-.debug_str.dwo
	.long	.Linfo_string2-.debug_str.dwo
	.long	.Linfo_string3-.debug_str.dwo
	.long	.Linfo_string4-.debug_str.dwo
	.long	.Linfo_string5-.debug_str.dwo
.Lstr_offsets_end:


	.section	.debug_info.dwo,"e",@progbits
	.long	.Ldebug_info_dwo_end0-.Ldebug_info_dwo_start0 # Length of Unit
.Ldebug_info_dwo_start0:
	.short	5                               # DWARF version number
	.byte	5                               # DWARF Unit Type
	.byte	8                               # Address Size (in bytes)
	.long	0                               # Offset Into Abbrev. Section
	.quad	1026699901672188186             # dwo_id
	.byte	1                               # Abbrev [1] 0x14:0x25 DW_TAG_compile_unit
	.byte	3                               # DW_AT_producer
	.short	12                              # DW_AT_language
	.byte	4                               # DW_AT_name
	.byte	5                               # DW_AT_dwo_name
	.byte	2                               # Abbrev [2] 0x1a:0xb DW_TAG_variable
	.byte	0                               # DW_AT_name
	.long	37                              # DW_AT_type
                                        # DW_AT_external
	.byte	0                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.byte	2                               # DW_AT_location
	.byte	161
	.byte	0
	.byte	3                               # Abbrev [3] 0x25:0x4 DW_TAG_base_type
	.byte	1                               # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	4                               # Abbrev [4] 0x29:0xf DW_TAG_subprogram
	.byte	1                               # DW_AT_low_pc
	.long	8       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	2                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.long	37                              # DW_AT_type
                                        # DW_AT_external
	.byte	0                               # End Of Children Mark
.Ldebug_info_dwo_end0:


	.section	.debug_abbrev.dwo,"e",@progbits
	.byte	1                               # Abbreviation Code
	.byte	17                              # DW_TAG_compile_unit
	.byte	1                               # DW_CHILDREN_yes
	.byte	37                              # DW_AT_producer
	.byte	37                              # DW_FORM_strx1
	.byte	19                              # DW_AT_language
	.byte	5                               # DW_FORM_data2
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	118                             # DW_AT_dwo_name
	.byte	37                              # DW_FORM_strx1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	2                               # Abbreviation Code
	.byte	52                              # DW_TAG_variable
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	3                               # Abbreviation Code
	.byte	36                              # DW_TAG_base_type
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	62                              # DW_AT_encoding
	.byte	11                              # DW_FORM_data1
	.byte	11                              # DW_AT_byte_size
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	4                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	0                               # DW_CHILDREN_no
	.byte	17                              # DW_AT_low_pc
	.byte	27                              # DW_FORM_addrx
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	0                               # EOM(3)
.Ldebug_abbrev_end:

.section .debug_cu_index,"",@progbits
.short 5  # DWARF version number
.short 0  # Reserved
.long  3  # Section count
.long  1  # Unit count
.long  2 # Slot count

.quad  1026699901672188186, 0 # Hash table
.long  1, 0 # Index table

## Table header
.long 1   # DW_SECT_INFO
.long 3   # DW_SECT_ABBREV
.long 6   # DW_SECT_STR_OFFSETS

## Offsets
.long 0   # offset into .debug_info.dwo
.long 0   # offset into .debug_abbrev.dwo
.long 0   # offset into .debug_str_offsets.dwo

## Sizes
.long .Ldebug_info_dwo_end0 - .debug_info.dwo
.long .Ldebug_abbrev_end - .debug_abbrev.dwo
.long .Lstr_offsets_end - .debug_str_offsets.dwo

.endif
