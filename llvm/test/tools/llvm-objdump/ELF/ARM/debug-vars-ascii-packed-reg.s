## Regression test for the llvm-objdump LiveVariable::print buffer-and-flush
## fix. PR /#192353 added an ASCII-packed virtual-register decode fallback to
## the compact DWARF expression printer. llvm-objdump's GetRegName lambda
## used to write "<unknown register N>" to the output stream as a side
## effect before returning empty; when the new fallback then succeeded, the
## decoded name was appended, producing output like
## "<unknown register 2454065>%r1".
##
## This test exercises the buffer-and-flush fix end-to-end with four
## variables, each with a different DWARF location, covering the
## interesting interactions between the lambda's miss and the compact
## printer's per-op pass:
##
##   x: DW_OP_regx ULEB128(0x257231)
##      Single reg, lambda misses, ASCII fallback rescues -> printer
##      returns true, buffer discarded, renders "%r1". (Target fix.)
##
##   y: DW_OP_regx ULEB128(0x257231), DW_OP_plus
##      Lambda misses on the regx (ASCII fallback rescues inside the
##      printer's stack), but DW_OP_plus is not handled by the compact
##      printer so it bails with "<unknown op DW_OP_plus (34)>" written
##      directly to OS and returns false. The buffer is then flushed,
##      producing the rescued reg's miss marker as a trailing false
##      alarm. This is the multi-op pile-up case: noisy but not the
##      original corruption.
##
##   z: DW_OP_regx ULEB128(100)
##      Reg num 100 is below the ASCII validator floor, so the fallback
##      also rejects. Printer returns false from the reg-failure path
##      with empty OS; the buffer flush surfaces the single
##      "<unknown register 100>" marker -- same content as pre-fix.
##
##   w: DW_OP_regx ULEB128(0x257231), DW_OP_regx ULEB128(0x257232)
##      Two regs, both ASCII-rescued -> Stack ends with size 2 instead
##      of 1, printer writes "<stack of size 2, expected 1>" to OS and
##      returns false. Buffer flush appends both false-alarm reg
##      markers behind that, producing the multi-marker pile-up the
##      buffer-and-flush model is least graceful about.

## Empty/single instruction function do not render variable-liveness columns
## and need at least two instructions for (ThisAddr, NextAddr) transition to
## draw live-in/live-out markers.

# RUN: llvm-mc -triple armv8a--none-eabi < %s -filetype=obj -o %t.o
# RUN: llvm-objdump %t.o -d --debug-vars=ascii | FileCheck %s

# CHECK: 00000000 <foo>:
# CHECK: x = %r1
# CHECK: y = <unknown op DW_OP_plus (34)><unknown register 2454065>
# CHECK: z = <unknown register 100>
# CHECK: w = <stack of size 2, expected 1><unknown register 2454065><unknown register 2454066>
# CHECK-NOT: <unknown register 2454065>%r1

	.text
	.arch	armv8-a
foo:
.Lfunc_begin0:
	ldr	r0, [r0]
	bx	lr
.Lfunc_end0:

	.section	.debug_str,"MS",%progbits,1
.Lstr_test:
	.asciz	"test.c"
.Lstr_foo:
	.asciz	"foo"
.Lstr_x:
	.asciz	"x"
.Lstr_y:
	.asciz	"y"
.Lstr_z:
	.asciz	"z"
.Lstr_w:
	.asciz	"w"

	.section	.debug_loc,"",%progbits
.Ldebug_loc0:
	.long	0
	.long	.Lfunc_end0-.Lfunc_begin0
	.short	5
	.byte	0x90              @ DW_OP_regx
	.byte	0xb1, 0xe4, 0x95, 0x01  @ ULEB128(0x257231) = ASCII-packed "%r1"
	.long	0
	.long	0
.Ldebug_loc1:
	.long	0
	.long	.Lfunc_end0-.Lfunc_begin0
	.short	6
	.byte	0x90              @ DW_OP_regx
	.byte	0xb1, 0xe4, 0x95, 0x01  @ ULEB128(0x257231)
	.byte	0x22              @ DW_OP_plus (unhandled by compact printer)
	.long	0
	.long	0
.Ldebug_loc2:
	.long	0
	.long	.Lfunc_end0-.Lfunc_begin0
	.short	2
	.byte	0x90              @ DW_OP_regx
	.byte	0x64              @ ULEB128(100) -- below ASCII validator floor
	.long	0
	.long	0
.Ldebug_loc3:
	.long	0
	.long	.Lfunc_end0-.Lfunc_begin0
	.short	10
	.byte	0x90              @ DW_OP_regx
	.byte	0xb1, 0xe4, 0x95, 0x01  @ ULEB128(0x257231) = ASCII-packed "%r1"
	.byte	0x90              @ DW_OP_regx
	.byte	0xb2, 0xe4, 0x95, 0x01  @ ULEB128(0x257232) = ASCII-packed "%r2"
	.long	0
	.long	0

	.section	.debug_abbrev,"",%progbits
	@ abbrev 1: DW_TAG_compile_unit, children=yes
	.byte	1, 17, 1
	.byte	3, 14              @ DW_AT_name, DW_FORM_strp
	.byte	17, 1              @ DW_AT_low_pc, DW_FORM_addr
	.byte	18, 6              @ DW_AT_high_pc, DW_FORM_data4
	.byte	0, 0
	@ abbrev 2: DW_TAG_subprogram, children=yes
	.byte	2, 46, 1
	.byte	17, 1              @ DW_AT_low_pc, DW_FORM_addr
	.byte	18, 6              @ DW_AT_high_pc, DW_FORM_data4
	.byte	3, 14              @ DW_AT_name, DW_FORM_strp
	.byte	0, 0
	@ abbrev 3: DW_TAG_variable, no children
	.byte	3, 52, 0
	.byte	2, 23              @ DW_AT_location, DW_FORM_sec_offset
	.byte	3, 14              @ DW_AT_name, DW_FORM_strp
	.byte	0, 0
	.byte	0                  @ end of abbrev table

	.section	.debug_info,"",%progbits
.Lcu_begin0:
	.long	.Lcu_end0 - .Lcu_post_length0
.Lcu_post_length0:
	.short	4
	.long	.debug_abbrev
	.byte	4

	@ abbrev 1: DW_TAG_compile_unit
	.byte	1
	.long	.Lstr_test
	.long	.Lfunc_begin0
	.long	.Lfunc_end0-.Lfunc_begin0

	@ abbrev 2: DW_TAG_subprogram "foo"
	.byte	2
	.long	.Lfunc_begin0
	.long	.Lfunc_end0-.Lfunc_begin0
	.long	.Lstr_foo

	@ abbrev 3: DW_TAG_variable "x"
	.byte	3
	.long	.Ldebug_loc0
	.long	.Lstr_x

	@ abbrev 3: DW_TAG_variable "y"
	.byte	3
	.long	.Ldebug_loc1
	.long	.Lstr_y

	@ abbrev 3: DW_TAG_variable "z"
	.byte	3
	.long	.Ldebug_loc2
	.long	.Lstr_z

	@ abbrev 3: DW_TAG_variable "w"
	.byte	3
	.long	.Ldebug_loc3
	.long	.Lstr_w

	.byte	0                  @ end of subprogram children
	.byte	0                  @ end of CU
.Lcu_end0:
