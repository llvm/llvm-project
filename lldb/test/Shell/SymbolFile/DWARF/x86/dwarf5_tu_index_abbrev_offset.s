## This test checks that lldb uses the abbrev_offset from .debug_tu_index when
## reading TUs in the .debug_info section of a DWARF v5 DWP.
##
## The assembly here is, essentially, slightly hand-reduced output from
## `clang -gsplit-dwarf -gdwarf-5 -fdebug-types-section`, with a manually-added
## .debug_cu_index and a .debug_tu_index to create a DWP, and a twist: abbrevs
## from the TU are listed *AFTER* abbrevs from the CU so that they don't begin
## at offset 0.

# RUN: llvm-mc --filetype=obj --triple x86_64 %s -o %t --defsym MAIN=1
# RUN: llvm-mc --filetype=obj --triple x86_64 %s -o %t.dwp
# RUN: %lldb %t -o "image lookup -t t1" -b | FileCheck %s
# CHECK: struct t1

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
	.quad	-8218585293556409984            # dwo_id
	.byte	1                               # Abbrev [1] 0x14:0x14 DW_TAG_skeleton_unit
	.asciz "hello.dwo"                    # DW_AT_dwo_name
	.byte	0                               # DW_AT_low_pc
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
	.quad	.Lfunc_begin0
.Ldebug_addr_end0:


.else
## DWP file
	.section	.debug_info.dwo,"e",@progbits
	.long	.Ldebug_info_dwo_end0-.Ldebug_info_dwo_start0 # Length of Unit
.Ldebug_info_dwo_start0:
	.short	5                               # DWARF version number
	.byte	6                               # DWARF Unit Type
	.byte	8                               # Address Size (in bytes)
	.long	0                               # Offset Into Abbrev. Section
	.quad	-4149699470930386446            # Type Signature
	.long	31                              # Type DIE Offset
	.byte	1                               # Abbrev [1] 0x18:0xe DW_TAG_type_unit
	.short	33                              # DW_AT_language
	.long	0                               # DW_AT_stmt_list
	.byte	2                               # Abbrev [2] 0x1f:0x6 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.byte	3                               # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.byte	0                               # End Of Children Mark
.Ldebug_info_dwo_end0:
	.long	.Ldebug_info_dwo_end1-.Ldebug_info_dwo_start1 # Length of Unit
.Ldebug_info_dwo_start1:
	.short	5                               # DWARF version number
	.byte	5                               # DWARF Unit Type
	.byte	8                               # Address Size (in bytes)
	.long	0                               # Offset Into Abbrev. Section
	.quad	-8218585293556409984
	.byte	3                               # Abbrev [3] 0x14:0x2f DW_TAG_compile_unit
	.byte	4                               # DW_AT_producer
	.short	33                              # DW_AT_language
	.byte	5                               # DW_AT_name
	.byte	6                               # DW_AT_dwo_name
	.byte	4                               # Abbrev [4] 0x1a:0x1b DW_TAG_subprogram
	.byte	0                               # DW_AT_low_pc
	.long	8                               # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	86
	.byte	0                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.long	53                              # DW_AT_type
                                        # DW_AT_external
	.byte	5                               # Abbrev [5] 0x29:0xb DW_TAG_variable
	.byte	2                               # DW_AT_location
	.byte	145
	.byte	120
	.byte	2                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	3                               # DW_AT_decl_line
	.long	57                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	6                               # Abbrev [6] 0x35:0x4 DW_TAG_base_type
	.byte	1                               # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	7                               # Abbrev [7] 0x39:0x9 DW_TAG_structure_type
                                        # DW_AT_declaration
	.quad	-4149699470930386446            # DW_AT_signature
	.byte	0                               # End Of Children Mark
.Ldebug_info_dwo_end1:



	.section	.debug_str.dwo,"eMS",@progbits,1
.Linfo_string0:
	.asciz	"main"
.Linfo_string1:
	.asciz	"int"
.Linfo_string2:
	.asciz	"v1"
.Linfo_string3:
	.asciz	"t1"
.Linfo_string4:
	.asciz	"hand-tuned clang output"
.Linfo_string5:
	.asciz	"test.cc"
.Linfo_string6:
	.asciz	"test.dwo"


	.section	.debug_str_offsets.dwo,"e",@progbits
	.long	32                              # Length of String Offsets Set
	.short	5
	.short	0

	.long	.Linfo_string0-.debug_str.dwo
	.long	.Linfo_string1-.debug_str.dwo
	.long	.Linfo_string2-.debug_str.dwo
	.long	.Linfo_string3-.debug_str.dwo
	.long	.Linfo_string4-.debug_str.dwo
	.long	.Linfo_string5-.debug_str.dwo
	.long	.Linfo_string6-.debug_str.dwo
.Lstr_offsets_end:

	.section	.debug_abbrev.dwo,"e",@progbits
.Ldebug_cu_abbrev_begin:
	.byte	3                               # Abbreviation Code
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
	.byte	4                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
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
	.byte	5                               # Abbreviation Code
	.byte	52                              # DW_TAG_variable
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	6                               # Abbreviation Code
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
	.byte	7                               # Abbreviation Code
	.byte	19                              # DW_TAG_structure_type
	.byte	0                               # DW_CHILDREN_no
	.byte	60                              # DW_AT_declaration
	.byte	25                              # DW_FORM_flag_present
	.byte	105                             # DW_AT_signature
	.byte	32                              # DW_FORM_ref_sig8
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	0                               # EOM(3)
.Ldebug_cu_abbrev_end:
.Ldebug_tu_abbrev_begin:
  .byte	1                               # Abbreviation Code
	.byte	65                              # DW_TAG_type_unit
	.byte	1                               # DW_CHILDREN_yes
	.byte	19                              # DW_AT_language
	.byte	5                               # DW_FORM_data2
	.byte	16                              # DW_AT_stmt_list
	.byte	23                              # DW_FORM_sec_offset
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	2                               # Abbreviation Code
	.byte	19                              # DW_TAG_structure_type
	.byte	0                               # DW_CHILDREN_no
	.byte	54                              # DW_AT_calling_convention
	.byte	11                              # DW_FORM_data1
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	11                              # DW_AT_byte_size
	.byte	11                              # DW_FORM_data1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	0                               # EOM(3)
.Ldebug_tu_abbrev_end:
.Ldebug_abbrev_dwo_end:

.section .debug_tu_index,"",@progbits
.short 5  # DWARF version number
.short 0  # Reserved
.long  3  # Section count
.long  1  # Unit count
.long  2 # Slot count

.quad  -4149699470930386446, 0 # Hash table
.long  1, 0 # Index table

## Table header
.long 1   # DW_SECT_INFO
.long 3   # DW_SECT_ABBREV
.long 6   # DW_SECT_STR_OFFSETS

## Offsets
.long 0   # offset into .debug_info.dwo
.long .Ldebug_tu_abbrev_begin - .debug_abbrev.dwo   # offset into .debug_abbrev.dwo
.long 0   # offset into .debug_str_offsets.dwo

## Sizes
.long .Ldebug_info_dwo_end0 - .debug_info.dwo
.long .Ldebug_tu_abbrev_end - .Ldebug_tu_abbrev_begin
.long .Lstr_offsets_end - .debug_str_offsets.dwo


.section .debug_cu_index,"",@progbits
.short 5  # DWARF version number
.short 0  # Reserved
.long  3  # Section count
.long  1  # Unit count
.long  2 # Slot count

.quad  -8218585293556409984, 0 # Hash table
.long  1, 0 # Index table

## Table header
.long 1   # DW_SECT_INFO
.long 3   # DW_SECT_ABBREV
.long 6   # DW_SECT_STR_OFFSETS

## Offsets
.long .Ldebug_info_dwo_end0 - .debug_info.dwo   # offset into .debug_info.dwo
.long .Ldebug_cu_abbrev_begin - .debug_abbrev.dwo  # offset into .debug_abbrev.dwo
.long 0   # offset into .debug_str_offsets.dwo

## Sizes
.long .Ldebug_info_dwo_end1 - .Ldebug_info_dwo_end0
.long .Ldebug_cu_abbrev_end - .Ldebug_cu_abbrev_begin
.long .Lstr_offsets_end - .debug_str_offsets.dwo


.endif
