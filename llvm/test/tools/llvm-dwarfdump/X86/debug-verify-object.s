# RUN: llvm-mc -triple x86_64-unknown-linux-gnu -filetype obj -o - %s | llvm-dwarfdump --verify -

	.text

	.section	.text.f,"ax",@progbits
	.globl	f
	.type	f,@function
f:
.Lfunc_begin0:
	pushq	$32
	popq	%rax
	retq
.Lfunc_end0:
	.size	f, .Lfunc_end0-f

	.section	.text.g,"ax",@progbits
	.globl	g
	.type	g,@function
g:
.Lfunc_begin1:
	pushq   $64
	popq    %rax
	retq
.Lfunc_end1:
	.size	g, .Lfunc_end1-g

	.section	.debug_abbrev,"",@progbits
	.byte	1                       # Abbreviation Code
	.byte	17                      # DW_TAG_compile_unit
	.byte	0                       # DW_CHILDREN_no
	.byte	17                      # DW_AT_low_pc
	.byte	1                       # DW_FORM_addr
	.byte	85                      # DW_AT_ranges
	.byte	23                      # DW_FORM_sec_offset
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	0                       # EOM(3)

	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	20                      # Length of Unit
	.short	4                       # DWARF version number
	.long	.debug_abbrev           # Offset Into Abbrev. Section
	.byte	8                       # Address Size (in bytes)
	.byte	1                       # Abbrev [1] 0xb:0x1f DW_TAG_compile_unit
	.quad	0                       # DW_AT_low_pc
	.long	.Ldebug_ranges0         # DW_AT_ranges

	.section        .debug_ranges,"",@progbits
.Ldebug_ranges0:
	.quad	.Lfunc_begin0
	.quad	.Lfunc_end0
	.quad	.Lfunc_begin1
	.quad	.Lfunc_end1
	.quad	0
	.quad	0

