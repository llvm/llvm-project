# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=arm64-apple-darwin -filetype=obj -o %t/crash.o %s
# RUN: not --crash llvm-jitlink -debugger-support=false \
# RUN:     -write-symtab %t/crash.symtab.txt %t/crash.o \
# RUN:     > %t/backtrace.txt 2>&1
# RUN: llvm-jitlink -symbolicate-with %t/crash.symtab.txt %t/backtrace.txt \
# RUN:     | FileCheck %s

# Deliberately crash by dereferencing an environment variable that should never
# be defined, then symbolicate the backtrace using the dumped symbol table.

# REQUIRES: system-darwin && native

# CHECK: this_should_crash {{.*}} ({{.*}}crash.o)

	.build_version macos, 26, 0
	.section	__TEXT,__text,regular,pure_instructions
	.globl	_this_should_crash
	.p2align	2
_this_should_crash:
	stp	x29, x30, [sp, #-16]!
	adrp	x0, l_.str@PAGE
	add	x0, x0, l_.str@PAGEOFF
	bl	_getenv
	ldrsb	w0, [x0]
	ldp	x29, x30, [sp], #16
	ret


	.globl	_main
	.p2align	2
_main:
	stp	x29, x30, [sp, #-16]!
	bl	_this_should_crash
	ldp	x29, x30, [sp], #16
	ret

	.section	__TEXT,__const
l_.str:
	.asciz	"a thousand curses upon anyone who dares define this"

.subsections_via_symbols
