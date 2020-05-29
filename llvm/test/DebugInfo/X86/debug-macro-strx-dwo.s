## This test checks that llvm-dwarfdump can dump debug_macro.dwo
## section containing DW_MACRO_*_strx forms present in a dwo object.

# RUN: llvm-mc -triple x86_64-unknown-linux -filetype=obj %s -o -| \
# RUN:   llvm-dwarfdump -debug-macro - | FileCheck -strict-whitespace -match-full-lines %s

#      CHECK:.debug_macro.dwo contents:
# CHECK-NEXT:0x00000000:
# CHECK-NEXT:macro header: version = 0x0005, flags = 0x02, debug_line_offset = 0x0000
# CHECK-NEXT:DW_MACRO_start_file - lineno: 0 filenum: 0
# CHECK-NEXT:  DW_MACRO_define_strx - lineno: 1 macro: DWARF_VERSION 5
# CHECK-NEXT:  DW_MACRO_undef_strx - lineno: 4 macro: DWARF_VERSION
# CHECK-NEXT:DW_MACRO_end_file

	.section	.debug_macro.dwo,"e",@progbits
.Lcu_macro_begin0:
	.short	5                      # Macro information version
	.byte	2                       # Flags: 32 bit, debug_line_offset present
	.long	0                       # debug_line_offset
	.byte	3                       # DW_MACRO_start_file
	.byte	0                       # Line Number
	.byte	0                       # File Number
	.byte	11                      # DW_MACRO_define_strx
	.byte	1                       # Line Number
	.byte	0                       # Macro String Index
	.byte	12                      # DW_MACRO_undef_strx
	.byte	4                       # Line Number
	.byte	1                       # Macro String Index
	.byte	4                       # DW_MACRO_end_file
	.byte	0                       # End Of Macro List Mark

	.section	.debug_str_offsets.dwo,"e",@progbits
	.long   .Lcu_str_off_end0-.Lcu_str_off_start0 # Unit length
	.short	5                                     # Version
	.short	0                                     # Padding
.Lcu_str_off_start0:
	.long	.Linfo_string0-.debug_str.dwo
	.long	.Linfo_string1-.debug_str.dwo
.Lcu_str_off_end0:

	.section	.debug_str.dwo,"eMS",@progbits,1
.Linfo_string0:
	.asciz	"DWARF_VERSION 5"
.Linfo_string1:
	.asciz	"DWARF_VERSION"

	.section	.debug_info.dwo,"e",@progbits
	.long	.Ldebug_info_dwo_end0-.Ldebug_info_dwo_start0 # Length of Unit
.Ldebug_info_dwo_start0:
	.short	5                      # DWARF version number
	.byte	5                       # DWARF Unit Type
	.byte	8                       # Address Size (in bytes)
	.long	0                       # Offset Into Abbrev. Section
	.quad	1536875774479801980
	.byte	1                       # Abbrev [1] 0x14:0x1a DW_TAG_compile_unit
	.long   .Lcu_macro_begin0-.debug_macro.dwo # DW_AT_macros
	.byte	0                       # End Of Children Mark
.Ldebug_info_dwo_end0:

	.section	.debug_abbrev.dwo,"e",@progbits
	.byte	1                       # Abbreviation Code
	.byte	17                      # DW_TAG_compile_unit
	.byte	0                       # DW_CHILDREN_no
	.byte	121                     # DW_AT_macros
	.byte	23                      # DW_FORM_sec_offset
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	0                       # EOM(3)
