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

# RUN: llvm-mc -triple armv8a--none-eabi < %s -filetype=obj -o %t.o
# RUN: llvm-objdump %t.o -d --debug-vars=ascii | FileCheck %s

# CHECK: 00000000 <foo>:
# CHECK: x = %r1
# CHECK: y = <unknown op DW_OP_plus (34)><unknown register 2454065>
# CHECK: z = <unknown register 100>
# CHECK: w = <stack of size 2, expected 1><unknown register 2454065><unknown register 2454066>
# CHECK-NOT: <unknown register 2454065>%r1

	.text
	.syntax unified
	.eabi_attribute	67, "2.09"
	.eabi_attribute	6, 10
	.eabi_attribute	7, 65
	.eabi_attribute	8, 1
	.eabi_attribute	9, 2
	.fpu	vfpv3
	.eabi_attribute	34, 0
	.eabi_attribute	17, 1
	.eabi_attribute	20, 1
	.eabi_attribute	21, 1
	.eabi_attribute	23, 3
	.eabi_attribute	24, 1
	.eabi_attribute	25, 1
	.eabi_attribute	38, 1
	.eabi_attribute	18, 4
	.eabi_attribute	26, 2
	.eabi_attribute	14, 0
	.file	"test.c"
	.globl	foo
	.p2align	2
	.type	foo,%function
	.code	32
foo:
.Lfunc_begin0:
	.file	1 "test.c"
	.loc	1 1 0
	.fnstart
	.cfi_sections .debug_frame
	.cfi_startproc
	.loc	1 2 10 prologue_end
	ldr	r0, [r0]
.Ltmp0:
	.loc	1 2 3 is_stmt 0
	bx	lr
.Ltmp1:
.Lfunc_end0:
	.size	foo, .Lfunc_end0-foo
	.cfi_endproc
	.cantunwind
	.fnend

	.section	.debug_str,"MS",%progbits,1
.Linfo_string0:
	.asciz	"clang"
.Linfo_string1:
	.asciz	"test.c"
.Linfo_string2:
	.asciz	"."
.Linfo_string3:
	.asciz	"foo"
.Linfo_string4:
	.asciz	"int"
.Linfo_string5:
	.asciz	"x"
.Linfo_string6:
	.asciz	"y"
.Linfo_string7:
	.asciz	"z"
.Linfo_string8:
	.asciz	"w"

	.section	.debug_loc,"",%progbits
.Ldebug_loc0:
	.long	.Lfunc_begin0-.Lfunc_begin0
	.long	.Lfunc_end0-.Lfunc_begin0
	.short	5
	.byte	0x90              @ DW_OP_regx
	.byte	0xb1, 0xe4, 0x95, 0x01  @ ULEB128(0x257231) = ASCII-packed "%r1"
	.long	0
	.long	0
.Ldebug_loc1:
	.long	.Lfunc_begin0-.Lfunc_begin0
	.long	.Lfunc_end0-.Lfunc_begin0
	.short	6
	.byte	0x90              @ DW_OP_regx
	.byte	0xb1, 0xe4, 0x95, 0x01  @ ULEB128(0x257231)
	.byte	0x22              @ DW_OP_plus (unhandled by compact printer)
	.long	0
	.long	0
.Ldebug_loc2:
	.long	.Lfunc_begin0-.Lfunc_begin0
	.long	.Lfunc_end0-.Lfunc_begin0
	.short	2
	.byte	0x90              @ DW_OP_regx
	.byte	0x64              @ ULEB128(100) -- below ASCII validator floor
	.long	0
	.long	0
.Ldebug_loc3:
	.long	.Lfunc_begin0-.Lfunc_begin0
	.long	.Lfunc_end0-.Lfunc_begin0
	.short	10
	.byte	0x90              @ DW_OP_regx
	.byte	0xb1, 0xe4, 0x95, 0x01  @ ULEB128(0x257231) = ASCII-packed "%r1"
	.byte	0x90              @ DW_OP_regx
	.byte	0xb2, 0xe4, 0x95, 0x01  @ ULEB128(0x257232) = ASCII-packed "%r2"
	.long	0
	.long	0

	.section	.debug_abbrev,"",%progbits
	.byte	1
	.byte	17
	.byte	1
	.byte	37
	.byte	14
	.byte	19
	.byte	5
	.byte	3
	.byte	14
	.byte	16
	.byte	23
	.byte	27
	.byte	14
	.ascii	"\264B"
	.byte	25
	.byte	17
	.byte	1
	.byte	18
	.byte	6
	.byte	0
	.byte	0
	.byte	2
	.byte	46
	.byte	1
	.byte	17
	.byte	1
	.byte	18
	.byte	6
	.byte	64
	.byte	24
	.byte	3
	.byte	14
	.byte	58
	.byte	11
	.byte	59
	.byte	11
	.byte	39
	.byte	25
	.byte	73
	.byte	19
	.byte	63
	.byte	25
	.byte	0
	.byte	0
	.byte	3
	.byte	5
	.byte	0
	.byte	2
	.byte	23
	.byte	3
	.byte	14
	.byte	58
	.byte	11
	.byte	59
	.byte	11
	.byte	73
	.byte	19
	.byte	0
	.byte	0
	.byte	4
	.byte	36
	.byte	0
	.byte	3
	.byte	14
	.byte	62
	.byte	11
	.byte	11
	.byte	11
	.byte	0
	.byte	0
	.byte	0

	.section	.debug_info,"",%progbits
.Lcu_begin0:
	.long	.Lcu_end0 - .Lcu_post_length0
.Lcu_post_length0:
	.short	4
	.long	.debug_abbrev
	.byte	4

	@ abbrev 1: DW_TAG_compile_unit
	.byte	1
	.long	.Linfo_string0
	.short	12
	.long	.Linfo_string1
	.long	.Lline_table_start0
	.long	.Linfo_string2
	.long	.Lfunc_begin0
	.long	.Lfunc_end0-.Lfunc_begin0

	@ abbrev 2: DW_TAG_subprogram "foo"
	.byte	2
	.long	.Lfunc_begin0
	.long	.Lfunc_end0-.Lfunc_begin0
	.byte	1
	.byte	91
	.long	.Linfo_string3
	.byte	1
	.byte	1
	.long	.Lint_die - .Lcu_begin0

	@ abbrev 3: DW_TAG_variable "x"
	.byte	3
	.long	.Ldebug_loc0
	.long	.Linfo_string5
	.byte	1
	.byte	1
	.long	.Lint_die - .Lcu_begin0

	@ abbrev 3: DW_TAG_variable "y"
	.byte	3
	.long	.Ldebug_loc1
	.long	.Linfo_string6
	.byte	1
	.byte	1
	.long	.Lint_die - .Lcu_begin0

	@ abbrev 3: DW_TAG_variable "z"
	.byte	3
	.long	.Ldebug_loc2
	.long	.Linfo_string7
	.byte	1
	.byte	1
	.long	.Lint_die - .Lcu_begin0

	@ abbrev 3: DW_TAG_variable "w"
	.byte	3
	.long	.Ldebug_loc3
	.long	.Linfo_string8
	.byte	1
	.byte	1
	.long	.Lint_die - .Lcu_begin0

	.byte	0                 @ end of subprogram children

.Lint_die:
	@ abbrev 4: DW_TAG_base_type "int"
	.byte	4
	.long	.Linfo_string4
	.byte	5
	.byte	4

	.byte	0                 @ end of CU
.Lcu_end0:

	.section	.debug_ranges,"",%progbits
	.section	.debug_macinfo,"",%progbits
.Lcu_macro_begin0:
	.byte	0

	.section	.debug_line,"",%progbits
.Lline_table_start0:
