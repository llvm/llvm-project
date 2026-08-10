# This tests that lldb is able to print DW_TAG_variable using DW_AT_const_value.

# RUN: llvm-mc -triple x86_64-unknown-linux-gnu %s -filetype=obj > %t.o
# RUN: %lldb %t.o -o "expression/x magic64" -o exit | FileCheck %s

# CHECK: (const long) $0 = 0xed9a924c00011151

# The DW_TAG_variable using DW_AT_const_value can be compiled from:
# static const long magic64 = 0xed9a924c00011151;
# int main(void) { return magic64; }

	.text
	.globl	main                    # -- Begin function main
	.type	main,@function
main:                                   # @main
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
                                        # -- End function
	.section	.debug_abbrev,"",@progbits
	.byte	1                       # Abbreviation Code
	.byte	17                      # DW_TAG_compile_unit
	.byte	1                       # DW_CHILDREN_yes
	.byte	37                      # DW_AT_producer
	.byte	8                       # DW_FORM_string
	.byte	19                      # DW_AT_language
	.byte	5                       # DW_FORM_data2
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	2                       # Abbreviation Code
	.byte	52                      # DW_TAG_variable
	.byte	0                       # DW_CHILDREN_no
	.byte	3                       # DW_AT_name
	.byte	8                       # DW_FORM_string
	.byte	73                      # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	28                      # DW_AT_const_value
#	.byte	15                      # DW_FORM_udata
	.byte	13                      # DW_FORM_sdata
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	3                       # Abbreviation Code
	.byte	38                      # DW_TAG_const_type
	.byte	0                       # DW_CHILDREN_no
	.byte	73                      # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	4                       # Abbreviation Code
	.byte	36                      # DW_TAG_base_type
	.byte	0                       # DW_CHILDREN_no
	.byte	3                       # DW_AT_name
	.byte	8                       # DW_FORM_string
	.byte	62                      # DW_AT_encoding
	.byte	11                      # DW_FORM_data1
	.byte	11                      # DW_AT_byte_size
	.byte	11                      # DW_FORM_data1
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
	.byte	1                       # Abbrev [1] 0xb:0x61 DW_TAG_compile_unit
	.asciz	"clang version 10.0.0"  # DW_AT_producer
	.short	12                      # DW_AT_language
	.byte	2                       # Abbrev [2] 0x2a:0x15 DW_TAG_variable
	.asciz	"magic64"               # DW_AT_name
	.long	.Lconst                 # DW_AT_type
        .sleb128 0xed9a924c00011151     # DW_AT_const_value
.Lconst:
	.byte	3                       # Abbrev [3] 0x3f:0x5 DW_TAG_const_type
	.long	.Lint64                 # DW_AT_type
.Lint64:
	.byte	4                       # Abbrev [4] 0x44:0x7 DW_TAG_base_type
	.asciz	"long int"              # DW_AT_name
	.byte	5                       # DW_AT_encoding
	.byte	8                       # DW_AT_byte_size
	.byte	0                       # End Of Children Mark
.Ldebug_info_end0:
