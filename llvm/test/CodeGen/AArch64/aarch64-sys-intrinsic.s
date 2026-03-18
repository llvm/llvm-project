	.file	"aarch64-sys-intrinsic.ll"
	.text
	.globl	sys_cgdsw                       // -- Begin function sys_cgdsw
	.p2align	2
	.type	sys_cgdsw,@function
sys_cgdsw:                              // @sys_cgdsw
	.cfi_startproc
// %bb.0:                               // %entry
	sys	#0, c7, c10, #6, x0
	ret
.Lfunc_end0:
	.size	sys_cgdsw, .Lfunc_end0-sys_cgdsw
	.cfi_endproc
                                        // -- End function
	.globl	sys_s1e2w                       // -- Begin function sys_s1e2w
	.p2align	2
	.type	sys_s1e2w,@function
sys_s1e2w:                              // @sys_s1e2w
	.cfi_startproc
// %bb.0:                               // %entry
	at	s1e2w, x0
	ret
.Lfunc_end1:
	.size	sys_s1e2w, .Lfunc_end1-sys_s1e2w
	.cfi_endproc
                                        // -- End function
	.globl	sys_vmalle1                     // -- Begin function sys_vmalle1
	.p2align	2
	.type	sys_vmalle1,@function
sys_vmalle1:                            // @sys_vmalle1
	.cfi_startproc
// %bb.0:                               // %entry
	sys	#0, c8, c7, #0, x0
	ret
.Lfunc_end2:
	.size	sys_vmalle1, .Lfunc_end2-sys_vmalle1
	.cfi_endproc
                                        // -- End function
	.section	".note.GNU-stack","",@progbits
