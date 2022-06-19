## Check that the --debug-vars option works for simple register locations, when
## using DWARF4 debug info, with multiple functions in one section. Check that
## the live-range lines are rendered correctly when using the --no-show-raw-insn,
## --line-numbers and --source options. These do not affect the DWARF parsing
## used by --debug-vars, but do add extra lines or columns to the output, so we
## test to make sure the live ranges are still displayed correctly.

## Generated with this compile command, with the source code in Inputs/debug.c:
## clang --target=arm--none-eabi -march=armv7-a -c debug.c -O1 -gdwarf-4 -S -o -

# RUN: llvm-mc -triple armv8a--none-eabi < %s -filetype=obj -o %t.o

# RUN: llvm-objdump %t.o -d --debug-vars | \
# RUN:     FileCheck %s --check-prefix=RAW --strict-whitespace

## Check that passing the default value for --debug-vars-indent (52) makes no
## change to the output.
# RUN: llvm-objdump %t.o -d --debug-vars --debug-vars-indent=52 | \
# RUN:     FileCheck %s --check-prefix=RAW --strict-whitespace

# RUN: llvm-objdump %t.o -d --debug-vars --debug-vars-indent=30 | \
# RUN:     FileCheck %s --check-prefix=INDENT --strict-whitespace

# RUN: llvm-objdump %t.o -d --debug-vars --no-show-raw-insn | \
# RUN:     FileCheck %s --check-prefix=NO-RAW --strict-whitespace

# RUN: llvm-objdump %t.o -d --debug-vars --no-show-raw-insn --line-numbers | \
# RUN:     FileCheck %s --check-prefix=LINE-NUMS --strict-whitespace

# RUN: mkdir -p %t/a
# RUN: cp %p/Inputs/debug.c %t/a/debug.c
# RUN: sed -e "s,SRC_COMPDIR,%/t/a,g" %s > %t.s
# RUN: llvm-mc -triple armv8a--none-eabi < %t.s -filetype=obj | \
# RUN:     llvm-objdump - -d --debug-vars --no-show-raw-insn --source | \
# RUN:     FileCheck %s --check-prefix=SOURCE --strict-whitespace

## An optional argument to the --debug-vars= option can be used to switch
## between unicode and ascii output (with unicode being the default).
# RUN: llvm-objdump %t.o -d --debug-vars=unicode | \
# RUN:     FileCheck %s --check-prefix=RAW --strict-whitespace
# RUN: llvm-objdump %t.o -d --debug-vars=ascii | \
# RUN:     FileCheck %s --check-prefix=ASCII --strict-whitespace
# RUN: not llvm-objdump %t.o -d --debug-vars=bad_value 2>&1 | \
# RUN: FileCheck %s --check-prefix=ERROR

## Note that llvm-objdump emits tab characters in the disassembly, assuming an
## 8-byte tab stop, so these might not look aligned in a text editor.

# RAW: 00000000 <foo>:
# RAW-NEXT:                                                                             ┠─ a = R0 
# RAW-NEXT:                                                                             ┃ ┠─ b = R1 
# RAW-NEXT:                                                                             ┃ ┃ ┠─ c = R2 
# RAW-NEXT:                                                                             ┃ ┃ ┃ ┌─ x = R0 
# RAW-NEXT:        0: 00 00 81 e0  	add	r0, r1, r0                                  ┻ ┃ ┃ ╈   
# RAW-NEXT:                                                                             ┌─ y = R0 
# RAW-NEXT:        4: 02 00 80 e0  	add	r0, r0, r2                                  ╈ ┃ ┃ ┻   
# RAW-NEXT:        8: 1e ff 2f e1  	bx	lr                                          ┻ ┻ ┻     
# RAW-EMPTY:
# RAW-NEXT: 0000000c <bar>:
# RAW-NEXT:                                                                             ┠─ a = R0 
# RAW-NEXT:        c: 01 00 80 e2  	add	r0, r0, #1                                  ┃         
# RAW-NEXT:       10: 1e ff 2f e1  	bx	lr                                          ┻         


# INDENT: 00000000 <foo>:
# INDENT-NEXT:                                                       ┠─ a = R0 
# INDENT-NEXT:                                                       ┃ ┠─ b = R1 
# INDENT-NEXT:                                                       ┃ ┃ ┠─ c = R2 
# INDENT-NEXT:                                                       ┃ ┃ ┃ ┌─ x = R0 
# INDENT-NEXT:        0: 00 00 81 e0  	add	r0, r1, r0            ┻ ┃ ┃ ╈   
# INDENT-NEXT:                                                       ┌─ y = R0 
# INDENT-NEXT:        4: 02 00 80 e0  	add	r0, r0, r2            ╈ ┃ ┃ ┻   
# INDENT-NEXT:        8: 1e ff 2f e1  	bx	lr                    ┻ ┻ ┻     
# INDENT-EMPTY:
# INDENT-NEXT: 0000000c <bar>:
# INDENT-NEXT:                                                       ┠─ a = R0 
# INDENT-NEXT:        c: 01 00 80 e2  	add	r0, r0, #1            ┃         
# INDENT-NEXT:       10: 1e ff 2f e1  	bx	lr                    ┻         

# NO-RAW: 00000000 <foo>:
# NO-RAW-NEXT:                                                                     ┠─ a = R0
# NO-RAW-NEXT:                                                                     ┃ ┠─ b = R1
# NO-RAW-NEXT:                                                                     ┃ ┃ ┠─ c = R2
# NO-RAW-NEXT:                                                                     ┃ ┃ ┃ ┌─ x = R0
# NO-RAW-NEXT:        0:      	add	r0, r1, r0                                  ┻ ┃ ┃ ╈
# NO-RAW-NEXT:                                                                     ┌─ y = R0
# NO-RAW-NEXT:        4:      	add	r0, r0, r2                                  ╈ ┃ ┃ ┻
# NO-RAW-NEXT:        8:      	bx	lr                                          ┻ ┻ ┻
# NO-RAW-EMPTY:
# NO-RAW-NEXT: 0000000c <bar>:
# NO-RAW-NEXT:                                                                     ┠─ a = R0
# NO-RAW-NEXT:        c:      	add	r0, r0, #1                                  ┃
# NO-RAW-NEXT:       10:      	bx	lr                                          ┻

# LINE-NUMS: 00000000 <foo>:
# LINE-NUMS-NEXT: ; foo():
# LINE-NUMS-NEXT: ; SRC_COMPDIR{{[\\/]}}debug.c:2                                             ┠─ a = R0
# LINE-NUMS-NEXT:                                                                     ┃ ┠─ b = R1
# LINE-NUMS-NEXT:                                                                     ┃ ┃ ┠─ c = R2
# LINE-NUMS-NEXT:                                                                     ┃ ┃ ┃ ┌─ x = R0
# LINE-NUMS-NEXT:        0:      	add	r0, r1, r0                                  ┻ ┃ ┃ ╈
# LINE-NUMS-NEXT: ; SRC_COMPDIR{{[\\/]}}debug.c:3                                             ┌─ y = R0
# LINE-NUMS-NEXT:        4:      	add	r0, r0, r2                                  ╈ ┃ ┃ ┻
# LINE-NUMS-NEXT: ; SRC_COMPDIR{{[\\/]}}debug.c:4                                             ┃ ┃ ┃
# LINE-NUMS-NEXT:        8:      	bx	lr                                          ┻ ┻ ┻
# LINE-NUMS-EMPTY:
# LINE-NUMS-NEXT: 0000000c <bar>:
# LINE-NUMS-NEXT: ; bar():
# LINE-NUMS-NEXT: ; SRC_COMPDIR{{[\\/]}}debug.c:8                                             ┠─ a = R0
# LINE-NUMS-NEXT:        c:      	add	r0, r0, #1                                  ┃
# LINE-NUMS-NEXT: ; SRC_COMPDIR{{[\\/]}}debug.c:9                                             ┃
# LINE-NUMS-NEXT:       10:      	bx	lr                                          ┻

# SOURCE: 00000000 <foo>:
# SOURCE-NEXT: ;   int x = a + b;                                                  ┠─ a = R0
# SOURCE-NEXT:                                                                     ┃ ┠─ b = R1
# SOURCE-NEXT:                                                                     ┃ ┃ ┠─ c = R2
# SOURCE-NEXT:                                                                     ┃ ┃ ┃ ┌─ x = R0
# SOURCE-NEXT:        0:      	add	r0, r1, r0                                  ┻ ┃ ┃ ╈
# SOURCE-NEXT: ;   int y = x + c;                                                  ┌─ y = R0
# SOURCE-NEXT:        4:      	add	r0, r0, r2                                  ╈ ┃ ┃ ┻
# SOURCE-NEXT: ;   return y;                                                       ┃ ┃ ┃
# SOURCE-NEXT:        8:      	bx	lr                                          ┻ ┻ ┻
# SOURCE-EMPTY:
# SOURCE-NEXT: 0000000c <bar>:
# SOURCE-NEXT: ;   a++;                                                            ┠─ a = R0
# SOURCE-NEXT:        c:      	add	r0, r0, #1                                  ┃
# SOURCE-NEXT: ;   return a;                                                       ┃
# SOURCE-NEXT:       10:      	bx	lr                                          ┻

# ASCII: 00000000 <foo>:
# ASCII-NEXT:                                                                             |- a = R0 
# ASCII-NEXT:                                                                             | |- b = R1 
# ASCII-NEXT:                                                                             | | |- c = R2 
# ASCII-NEXT:                                                                             | | | /- x = R0 
# ASCII-NEXT:        0: 00 00 81 e0  	add	r0, r1, r0                                  v | | ^   
# ASCII-NEXT:                                                                             /- y = R0 
# ASCII-NEXT:        4: 02 00 80 e0  	add	r0, r0, r2                                  ^ | | v   
# ASCII-NEXT:        8: 1e ff 2f e1  	bx	lr                                          v v v     
# ASCII-EMPTY:
# ASCII-NEXT: 0000000c <bar>:
# ASCII-NEXT:                                                                             |- a = R0 
# ASCII-NEXT:        c: 01 00 80 e2  	add	r0, r0, #1                                  |         
# ASCII-NEXT:       10: 1e ff 2f e1  	bx	lr                                          v         

# ERROR: error: 'bad_value' is not a valid value for '--debug-vars='

	.text
	.syntax unified
	.eabi_attribute	67, "2.09"
	.eabi_attribute	6, 10
	.eabi_attribute	7, 65
	.eabi_attribute	8, 1
	.eabi_attribute	9, 2
	.fpu	neon
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
	.file	"debug.c"
	.globl	foo
	.p2align	2
	.type	foo,%function
	.code	32
foo:
.Lfunc_begin0:
	.file	1 "" "SRC_COMPDIR/debug.c"
	.loc	1 1 0
	.fnstart
	.cfi_sections .debug_frame
	.cfi_startproc
	.loc	1 2 13 prologue_end
	add	r0, r1, r0
.Ltmp0:
	.loc	1 3 13
	add	r0, r0, r2
.Ltmp1:
	.loc	1 4 3
	bx	lr
.Ltmp2:
.Lfunc_end0:
	.size	foo, .Lfunc_end0-foo
	.cfi_endproc
	.cantunwind
	.fnend

	.globl	bar
	.p2align	2
	.type	bar,%function
	.code	32
bar:
.Lfunc_begin1:
	.loc	1 7 0
	.fnstart
	.cfi_startproc
	.loc	1 8 4 prologue_end
	add	r0, r0, #1
.Ltmp3:
	.loc	1 9 3
	bx	lr
.Ltmp4:
.Lfunc_end1:
	.size	bar, .Lfunc_end1-bar
	.cfi_endproc
	.cantunwind
	.fnend

	.section	.debug_str,"MS",%progbits,1
.Linfo_string0:
	.asciz	"clang version 10.0.0 (git@github.com:llvm/llvm-project.git e73f78acd34360f7450b81167d9dc858ccddc262)"
.Linfo_string1:
	.asciz	"SRC_COMPDIR/debug.c"
.Linfo_string2:
	.asciz	""
.Linfo_string3:
	.asciz	"foo"
.Linfo_string4:
	.asciz	"int"
.Linfo_string5:
	.asciz	"bar"
.Linfo_string6:
	.asciz	"a"
.Linfo_string7:
	.asciz	"b"
.Linfo_string8:
	.asciz	"c"
.Linfo_string9:
	.asciz	"x"
.Linfo_string10:
	.asciz	"y"
	.section	.debug_loc,"",%progbits
.Ldebug_loc0:
	.long	.Lfunc_begin0-.Lfunc_begin0
	.long	.Ltmp0-.Lfunc_begin0
	.short	1
	.byte	80
	.long	0
	.long	0
.Ldebug_loc1:
	.long	.Ltmp0-.Lfunc_begin0
	.long	.Ltmp1-.Lfunc_begin0
	.short	1
	.byte	80
	.long	0
	.long	0
.Ldebug_loc2:
	.long	.Ltmp1-.Lfunc_begin0
	.long	.Lfunc_end0-.Lfunc_begin0
	.short	1
	.byte	80
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
	.ascii	"\227B"
	.byte	25
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
	.byte	5
	.byte	0
	.byte	2
	.byte	24
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
	.byte	5
	.byte	52
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
	.byte	6
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
	.long	.Ldebug_info_end0-.Ldebug_info_start0
.Ldebug_info_start0:
	.short	4
	.long	.debug_abbrev
	.byte	4
	.byte	1
	.long	.Linfo_string0
	.short	12
	.long	.Linfo_string1
	.long	.Lline_table_start0
	.long	.Linfo_string2
	.long	.Lfunc_begin0
	.long	.Lfunc_end1-.Lfunc_begin0
	.byte	2
	.long	.Lfunc_begin0
	.long	.Lfunc_end0-.Lfunc_begin0
	.byte	1
	.byte	91

	.long	.Linfo_string3
	.byte	1
	.byte	1

	.long	166

	.byte	3
	.long	.Ldebug_loc0
	.long	.Linfo_string6
	.byte	1
	.byte	1
	.long	166
	.byte	4
	.byte	1
	.byte	81
	.long	.Linfo_string7
	.byte	1
	.byte	1
	.long	166
	.byte	4
	.byte	1
	.byte	82
	.long	.Linfo_string8
	.byte	1
	.byte	1
	.long	166
	.byte	5
	.long	.Ldebug_loc1
	.long	.Linfo_string9
	.byte	1
	.byte	2
	.long	166
	.byte	5
	.long	.Ldebug_loc2
	.long	.Linfo_string10
	.byte	1
	.byte	3
	.long	166
	.byte	0
	.byte	2
	.long	.Lfunc_begin1
	.long	.Lfunc_end1-.Lfunc_begin1
	.byte	1
	.byte	91

	.long	.Linfo_string5
	.byte	1
	.byte	7

	.long	166

	.byte	4
	.byte	1
	.byte	80
	.long	.Linfo_string6
	.byte	1
	.byte	7
	.long	166
	.byte	0
	.byte	6
	.long	.Linfo_string4
	.byte	5
	.byte	4
	.byte	0
.Ldebug_info_end0:
	.ident	"clang version 10.0.0 (git@github.com:llvm/llvm-project.git e73f78acd34360f7450b81167d9dc858ccddc262)"
	.section	".note.GNU-stack","",%progbits
	.addrsig
	.eabi_attribute	30, 1
	.section	.debug_line,"",%progbits
.Lline_table_start0:
