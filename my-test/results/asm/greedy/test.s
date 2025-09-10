	.file	"test.c"
	.text
	.globl	f                               // -- Begin function f
	.p2align	2
	.type	f,@function
f:                                      // @f
// %bb.0:
	sub	sp, sp, #32
	stp	x29, x30, [sp, #16]             // 16-byte Folded Spill
	add	x29, sp, #16
	str	x0, [sp, #8]
	ldr	x8, [sp, #8]
	ldr	w8, [x8]
	str	w8, [sp, #4]
	ldr	w8, [sp, #4]
	add	w8, w8, #1
	str	w8, [sp, #4]
	ldr	w8, [sp, #4]
	ldr	x9, [sp, #8]
	str	w8, [x9]
	ldr	w0, [sp, #4]
	ldp	x29, x30, [sp, #16]             // 16-byte Folded Reload
	add	sp, sp, #32
	ret
.Lfunc_end0:
	.size	f, .Lfunc_end0-f
                                        // -- End function
	.ident	"clang version 22.0.0git (https://github.com/llvm/llvm-project.git 34109cd26ae1b317d91c061500d9828fe6ebab0b)"
	.section	".note.GNU-stack","",@progbits
