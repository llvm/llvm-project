# RUN: not llvm-mc < %s -o /dev/null 2>&1 | FileCheck %s

	.text
	.global baz
baz:
.Lfunc_begin0:
	.cv_file 1 "t.cpp"
	.cv_func_id 0
	.cv_loc 0 1 1 1
	pushq %rbp
	movq %rsp, %rbp
	.cv_loc 0 1 2 1

	.data # Switching sections raises an error.

	incl    x(%rip)
	.cv_loc 0 1 3 1
# CHECK: error: all .cv_loc directives for a function must be in the same section
	popq %rbp
	retq
.Lfunc_end0:

	.section	.debug$S,"dr"
	.cv_linetable 0 .Lfunc_begin0 .Lfunc_end0
	.short 2   # Record length
	.short 2   # Record kind: S_INLINESITE_END
