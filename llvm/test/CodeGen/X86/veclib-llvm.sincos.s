	.att_syntax
	.file	"veclib-llvm.sincos.ll"
	.text
	.globl	test_sincos_v4f32               # -- Begin function test_sincos_v4f32
	.p2align	4
	.type	test_sincos_v4f32,@function
test_sincos_v4f32:                      # @test_sincos_v4f32
	.cfi_startproc
# %bb.0:
	pushq	%rbx
	.cfi_def_cfa_offset 16
	subq	$16, %rsp
	.cfi_def_cfa_offset 32
	.cfi_offset %rbx, -16
	movq	%rsi, %rbx
	movq	%rsp, %rsi
	callq	amd_vrs4_sincosf@PLT
	movaps	(%rsp), %xmm0
	movaps	%xmm0, (%rbx)
	addq	$16, %rsp
	.cfi_def_cfa_offset 16
	popq	%rbx
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end0:
	.size	test_sincos_v4f32, .Lfunc_end0-test_sincos_v4f32
	.cfi_endproc
                                        # -- End function
	.globl	test_sincos_v8f32               # -- Begin function test_sincos_v8f32
	.p2align	4
	.type	test_sincos_v8f32,@function
test_sincos_v8f32:                      # @test_sincos_v8f32
	.cfi_startproc
# %bb.0:
	pushq	%r14
	.cfi_def_cfa_offset 16
	pushq	%rbx
	.cfi_def_cfa_offset 24
	subq	$56, %rsp
	.cfi_def_cfa_offset 80
	.cfi_offset %rbx, -24
	.cfi_offset %r14, -16
	movq	%rsi, %rbx
	movq	%rdi, %r14
	movaps	%xmm0, (%rsp)                   # 16-byte Spill
	addq	$16, %rdi
	leaq	16(%rsp), %rsi
	movaps	%xmm1, %xmm0
	callq	amd_vrs4_sincosf@PLT
	leaq	32(%rsp), %rsi
	movaps	(%rsp), %xmm0                   # 16-byte Reload
	movq	%r14, %rdi
	callq	amd_vrs4_sincosf@PLT
	movaps	16(%rsp), %xmm0
	movaps	32(%rsp), %xmm1
	movaps	%xmm1, (%rbx)
	movaps	%xmm0, 16(%rbx)
	addq	$56, %rsp
	.cfi_def_cfa_offset 24
	popq	%rbx
	.cfi_def_cfa_offset 16
	popq	%r14
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end1:
	.size	test_sincos_v8f32, .Lfunc_end1-test_sincos_v8f32
	.cfi_endproc
                                        # -- End function
	.globl	test_sincos_v16f32              # -- Begin function test_sincos_v16f32
	.p2align	4
	.type	test_sincos_v16f32,@function
test_sincos_v16f32:                     # @test_sincos_v16f32
	.cfi_startproc
# %bb.0:
	pushq	%r14
	.cfi_def_cfa_offset 16
	pushq	%rbx
	.cfi_def_cfa_offset 24
	subq	$120, %rsp
	.cfi_def_cfa_offset 144
	.cfi_offset %rbx, -24
	.cfi_offset %r14, -16
	movq	%rsi, %rbx
	movq	%rdi, %r14
	movaps	%xmm3, (%rsp)                   # 16-byte Spill
	movaps	%xmm2, 16(%rsp)                 # 16-byte Spill
	movaps	%xmm0, 32(%rsp)                 # 16-byte Spill
	addq	$16, %rdi
	leaq	80(%rsp), %rsi
	movaps	%xmm1, %xmm0
	callq	amd_vrs4_sincosf@PLT
	leaq	48(%r14), %rdi
	leaq	48(%rsp), %rsi
	movaps	(%rsp), %xmm0                   # 16-byte Reload
	callq	amd_vrs4_sincosf@PLT
	leaq	32(%r14), %rdi
	leaq	64(%rsp), %rsi
	movaps	16(%rsp), %xmm0                 # 16-byte Reload
	callq	amd_vrs4_sincosf@PLT
	leaq	96(%rsp), %rsi
	movaps	32(%rsp), %xmm0                 # 16-byte Reload
	movq	%r14, %rdi
	callq	amd_vrs4_sincosf@PLT
	movaps	80(%rsp), %xmm0
	movaps	48(%rsp), %xmm1
	movaps	64(%rsp), %xmm2
	movaps	96(%rsp), %xmm3
	movaps	%xmm3, (%rbx)
	movaps	%xmm2, 32(%rbx)
	movaps	%xmm1, 48(%rbx)
	movaps	%xmm0, 16(%rbx)
	addq	$120, %rsp
	.cfi_def_cfa_offset 24
	popq	%rbx
	.cfi_def_cfa_offset 16
	popq	%r14
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end2:
	.size	test_sincos_v16f32, .Lfunc_end2-test_sincos_v16f32
	.cfi_endproc
                                        # -- End function
	.globl	test_sincos_v2f64               # -- Begin function test_sincos_v2f64
	.p2align	4
	.type	test_sincos_v2f64,@function
test_sincos_v2f64:                      # @test_sincos_v2f64
	.cfi_startproc
# %bb.0:
	pushq	%rbx
	.cfi_def_cfa_offset 16
	subq	$16, %rsp
	.cfi_def_cfa_offset 32
	.cfi_offset %rbx, -16
	movq	%rsi, %rbx
	movq	%rsp, %rsi
	callq	amd_vrd2_sincos@PLT
	movaps	(%rsp), %xmm0
	movaps	%xmm0, (%rbx)
	addq	$16, %rsp
	.cfi_def_cfa_offset 16
	popq	%rbx
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end3:
	.size	test_sincos_v2f64, .Lfunc_end3-test_sincos_v2f64
	.cfi_endproc
                                        # -- End function
	.globl	test_sincos_v4f64               # -- Begin function test_sincos_v4f64
	.p2align	4
	.type	test_sincos_v4f64,@function
test_sincos_v4f64:                      # @test_sincos_v4f64
	.cfi_startproc
# %bb.0:
	pushq	%r14
	.cfi_def_cfa_offset 16
	pushq	%rbx
	.cfi_def_cfa_offset 24
	subq	$56, %rsp
	.cfi_def_cfa_offset 80
	.cfi_offset %rbx, -24
	.cfi_offset %r14, -16
	movq	%rsi, %rbx
	movq	%rdi, %r14
	movaps	%xmm0, (%rsp)                   # 16-byte Spill
	addq	$16, %rdi
	leaq	16(%rsp), %rsi
	movaps	%xmm1, %xmm0
	callq	amd_vrd2_sincos@PLT
	leaq	32(%rsp), %rsi
	movaps	(%rsp), %xmm0                   # 16-byte Reload
	movq	%r14, %rdi
	callq	amd_vrd2_sincos@PLT
	movaps	16(%rsp), %xmm0
	movaps	32(%rsp), %xmm1
	movaps	%xmm1, (%rbx)
	movaps	%xmm0, 16(%rbx)
	addq	$56, %rsp
	.cfi_def_cfa_offset 24
	popq	%rbx
	.cfi_def_cfa_offset 16
	popq	%r14
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end4:
	.size	test_sincos_v4f64, .Lfunc_end4-test_sincos_v4f64
	.cfi_endproc
                                        # -- End function
	.globl	test_sincos_v8f64               # -- Begin function test_sincos_v8f64
	.p2align	4
	.type	test_sincos_v8f64,@function
test_sincos_v8f64:                      # @test_sincos_v8f64
	.cfi_startproc
# %bb.0:
	pushq	%r14
	.cfi_def_cfa_offset 16
	pushq	%rbx
	.cfi_def_cfa_offset 24
	subq	$120, %rsp
	.cfi_def_cfa_offset 144
	.cfi_offset %rbx, -24
	.cfi_offset %r14, -16
	movq	%rsi, %rbx
	movq	%rdi, %r14
	movaps	%xmm3, (%rsp)                   # 16-byte Spill
	movaps	%xmm2, 16(%rsp)                 # 16-byte Spill
	movaps	%xmm0, 32(%rsp)                 # 16-byte Spill
	addq	$16, %rdi
	leaq	80(%rsp), %rsi
	movaps	%xmm1, %xmm0
	callq	amd_vrd2_sincos@PLT
	leaq	48(%r14), %rdi
	leaq	48(%rsp), %rsi
	movaps	(%rsp), %xmm0                   # 16-byte Reload
	callq	amd_vrd2_sincos@PLT
	leaq	32(%r14), %rdi
	leaq	64(%rsp), %rsi
	movaps	16(%rsp), %xmm0                 # 16-byte Reload
	callq	amd_vrd2_sincos@PLT
	leaq	96(%rsp), %rsi
	movaps	32(%rsp), %xmm0                 # 16-byte Reload
	movq	%r14, %rdi
	callq	amd_vrd2_sincos@PLT
	movaps	80(%rsp), %xmm0
	movaps	48(%rsp), %xmm1
	movaps	64(%rsp), %xmm2
	movaps	96(%rsp), %xmm3
	movaps	%xmm3, (%rbx)
	movaps	%xmm2, 32(%rbx)
	movaps	%xmm1, 48(%rbx)
	movaps	%xmm0, 16(%rbx)
	addq	$120, %rsp
	.cfi_def_cfa_offset 24
	popq	%rbx
	.cfi_def_cfa_offset 16
	popq	%r14
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end5:
	.size	test_sincos_v8f64, .Lfunc_end5-test_sincos_v8f64
	.cfi_endproc
                                        # -- End function
	.globl	test_sincos_v4f32_void          # -- Begin function test_sincos_v4f32_void
	.p2align	4
	.type	test_sincos_v4f32_void,@function
test_sincos_v4f32_void:                 # @test_sincos_v4f32_void
	.cfi_startproc
# %bb.0:
	pushq	%rax
	.cfi_def_cfa_offset 16
	callq	sincosf@PLT
	popq	%rax
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end6:
	.size	test_sincos_v4f32_void, .Lfunc_end6-test_sincos_v4f32_void
	.cfi_endproc
                                        # -- End function
	.section	".note.GNU-stack","",@progbits
