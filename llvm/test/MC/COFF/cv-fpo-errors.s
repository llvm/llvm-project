# RUN: not llvm-mc < %s -triple i686-windows-msvc -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error:

.globl _foo
_foo:
	.cv_fpo_proc
	# CHECK: :[[#@LINE-1]]:{{[0-9]+}}: error: expected symbol name
	.cv_fpo_proc 1
	# CHECK: :[[#@LINE-1]]:{{[0-9]+}}: error: expected symbol name
	.cv_fpo_proc _foo extra
	# CHECK: :[[#@LINE-1]]:{{[0-9]+}}: error: expected parameter byte count
	.cv_fpo_proc _foo 4 extra
	# CHECK: :[[#@LINE-1]]:22: error: expected newline
	.cv_fpo_proc _foo 4

	pushl	%ebp
	.cv_fpo_pushreg 1
	# CHECK: :[[#@LINE-1]]:{{[0-9]+}}: error: invalid register name
	.cv_fpo_pushreg ebp

	subl $20, %esp
	.cv_fpo_stackalloc asdf
	# CHECK: :[[#@LINE-1]]:{{[0-9]+}}: error: expected offset
	.cv_fpo_stackalloc 20 asdf
	# CHECK: :[[#@LINE-1]]:24: error: expected newline
	.cv_fpo_stackalloc 20
	.cv_fpo_endprologue asdf
	# CHECK: :[[#@LINE-1]]:22: error: expected newline
	.cv_fpo_endprologue

	addl $20, %esp
	popl %ebp
	retl
	.cv_fpo_endproc asdf
	# CHECK: :[[#@LINE-1]]:18: error: expected newline
	.cv_fpo_endproc

	.section	.debug$S,"dr"
	.p2align	2
	.long	4                       # Debug section magic
	.cv_fpo_data
	# CHECK: :[[#@LINE-1]]:{{[0-9]+}}: error: expected symbol name
	.cv_fpo_data 1
	# CHECK: :[[#@LINE-1]]:{{[0-9]+}}: error: expected symbol name
	.cv_fpo_data _foo asdf
	# CHECK: :[[#@LINE-1]]:{{[0-9]+}}: error: expected newline
	.cv_fpo_data _foo
	.long 0
