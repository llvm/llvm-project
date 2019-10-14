# Test that an empty .debug_aranges section doesn't confuse (or crash) us.

# REQUIRES: x86

# RUN: llvm-mc %s -triple x86_64-pc-linux -filetype=obj >%t
# RUN: lldb %t -o "breakpoint set -n f" -b | FileCheck %s

# CHECK: Breakpoint 1: where = {{.*}}`f, address = 0x0000000000000047

	.text
	.globl	f                       # -- Begin function f
	.type	f,@function
        .rept 0x47
        nop
        .endr
f:                                      # @f
.Lfunc_begin0:
	retq
.Lfunc_end0:
	.size	f, .Lfunc_end0-f
                                        # -- End function
	.section	.debug_str,"MS",@progbits,1
.Linfo_string3:
	.asciz	"f"                     # string offset=83
	.section	.debug_abbrev,"",@progbits
	.byte	1                       # Abbreviation Code
	.byte	17                      # DW_TAG_compile_unit
	.byte	1                       # DW_CHILDREN_yes
	.byte	17                      # DW_AT_low_pc
	.byte	1                       # DW_FORM_addr
	.byte	18                      # DW_AT_high_pc
	.byte	6                       # DW_FORM_data4
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	2                       # Abbreviation Code
	.byte	46                      # DW_TAG_subprogram
	.byte	0                       # DW_CHILDREN_no
	.byte	17                      # DW_AT_low_pc
	.byte	1                       # DW_FORM_addr
	.byte	18                      # DW_AT_high_pc
	.byte	6                       # DW_FORM_data4
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	0                       # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	4                       # DWARF version number
	.long	.debug_abbrev           # Offset Into Abbrev. Section
	.byte	8                       # Address Size (in bytes)
	.byte	1                       # Abbrev [1] 0xb:0x35 DW_TAG_compile_unit
	.quad	.Lfunc_begin0           # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0 # DW_AT_high_pc
	.byte	2                       # Abbrev [2] 0x2a:0x15 DW_TAG_subprogram
	.quad	.Lfunc_begin0           # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0 # DW_AT_high_pc
	.long	.Linfo_string3          # DW_AT_name
	.byte	0                       # End Of Children Mark
.Ldebug_info_end0:

	.section	".note.GNU-stack","",@progbits
	.section	.debug_aranges,"",@progbits
