# RUN: llvm-mc %s -filetype obj -triple x86_64-pc-linux -o %t.o
# RUN: llvm-dwarfdump -debug-info %t.o | FileCheck %s
# RUN: llvm-dwarfdump -debug-info %t.o -v | FileCheck --check-prefix=VERBOSE %s

# CHECK: DW_TAG_compile_unit
# CHECK:   DW_AT_low_pc                                              (0x0000000000000000)
# VERBOSE: DW_AT_low_pc [DW_FORM_addrx] (indexed (00000000) address = 0x0000000000000000 ".text")
# FIXME: Improve the error message from "unresolved" to describe the specific
#        issue (in this case, the index is outside the bounds of the debug_addr
#        contribution/debug_addr section)
# CHECK:   DW_AT_low_pc                 (indexed (00000001) address = <unresolved>)
# VERBOSE: DW_AT_low_pc [DW_FORM_addrx] (indexed (00000001) address = <unresolved>)

# CHECK: DW_TAG_compile_unit
# FIXME: Should error "no debug_addr contribution" - rather than parsing debug_addr
#        from the start, incorrectly interpreting the header bytes as an address.
# CHECK:   DW_AT_low_pc                 (indexed (00000000) address = <unresolved>)
# VERBOSE: DW_AT_low_pc [DW_FORM_addrx] (indexed (00000000) address = <unresolved>)

	.globl	foo                     # -- Begin function foo
foo:                                    # @foo
.Lfunc_begin0:
	retq
.Lfunc_end0:
	.section	.debug_abbrev,"",@progbits
	.byte	1                       # Abbreviation Code
	.byte	17                      # DW_TAG_compile_unit
	.byte	0                       # DW_CHILDREN_no
	.byte	115                     # DW_AT_addr_base
	.byte	23                      # DW_FORM_sec_offset
	.byte	17                      # DW_AT_low_pc
	.byte	27                      # DW_FORM_addrx
	.byte	17                      # DW_AT_low_pc
	.byte	27                      # DW_FORM_addrx
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	2                       # Abbreviation Code
	.byte	17                      # DW_TAG_compile_unit
	.byte	0                       # DW_CHILDREN_no
	.byte	17                      # DW_AT_low_pc
	.byte	27                      # DW_FORM_addrx
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	0                       # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	5                       # DWARF version number
	.byte	1                       # DWARF Unit Type
	.byte	8                       # Address Size (in bytes)
	.long	.debug_abbrev           # Offset Into Abbrev. Section
	.byte	1                       # Abbrev [1] 0xc:0x23 DW_TAG_compile_unit
	.long	.Laddr_table_base0      # DW_AT_addr_base
	.byte	0                       # DW_AT_low_pc
	.byte	1                       # DW_AT_low_pc
.Ldebug_info_end0:
.Lcu_begin1:
	.long	.Ldebug_info_end1-.Ldebug_info_start1 # Length of Unit
.Ldebug_info_start1:
	.short	5                       # DWARF version number
	.byte	1                       # DWARF Unit Type
	.byte	8                       # Address Size (in bytes)
	.long	.debug_abbrev           # Offset Into Abbrev. Section
	.byte	2                       # Abbrev [2] 0xc:0x23 DW_TAG_compile_unit
	.long	.Laddr_table_base0      # DW_AT_addr_base
	.byte	0                       # DW_AT_low_pc
.Ldebug_info_end1:
	.section	.debug_macinfo,"",@progbits
	.byte	0                       # End Of Macro List Mark
	.section	.debug_addr,"",@progbits
	.long	12                      # Length of Pool
	.short	5                       # DWARF version number
	.byte	8                       # Address Size (in bytes)
	.byte	0                       # Segment selector size
.Laddr_table_base0:
	.quad	.Lfunc_begin0
