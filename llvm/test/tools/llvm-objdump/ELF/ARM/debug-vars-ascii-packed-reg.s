## NVPTX packs virtual register names into DWARF register numbers, but its text
## PTX output does not exercise llvm-objdump's disassembler. Put the packed
## value for "%r1" in an ARM object to make the target lookup miss. Compact
## printing must produce "%r1" without first emitting
## "<unknown register 2454065>".
##
## Keep both instructions; llvm-objdump does not render this live range with
## only one.

# RUN: llvm-mc -triple armv8a--none-eabi < %s -filetype=obj -o %t.o
# RUN: llvm-objdump %t.o -d --debug-vars=ascii | FileCheck %s \
# RUN:   --implicit-check-not="<unknown register"

# CHECK: x = %r1

	.text
	.arch	armv8-a
foo:
.Lfunc_begin0:
	ldr	r0, [r0]
	bx	lr
.Lfunc_end0:

	.section	.debug_abbrev,"",%progbits
	@ DW_TAG_compile_unit with children and no attributes.
	.byte	1, 17, 1, 0, 0
	@ DW_TAG_subprogram with DW_AT_low_pc and DW_AT_high_pc.
	.byte	2, 46, 1, 17, 1, 18, 6, 0, 0
	@ DW_TAG_variable with DW_AT_location and DW_AT_name.
	.byte	3, 52, 0, 2, 24, 3, 8, 0, 0
	.byte	0                  @ end of abbreviation table

	.section	.debug_info,"",%progbits
	.long	.Lcu_end0-.Lcu_post_length0 @ unit length
.Lcu_post_length0:
	.short	4                  @ DWARF version
	.long	.debug_abbrev      @ abbreviation table offset
	.byte	4                  @ address size

	.byte	1                  @ DW_TAG_compile_unit
	.byte	2                  @ DW_TAG_subprogram
	.long	.Lfunc_begin0      @ DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0 @ DW_AT_high_pc
	.byte	3                  @ DW_TAG_variable
	.byte	5, 0x90            @ exprloc length, DW_OP_regx
	.uleb128 0x257231      @ ASCII-packed "%r1"
	.asciz	"x"                @ DW_AT_name
	.byte	0                  @ end of subprogram children
	.byte	0                  @ end of CU
.Lcu_end0:
