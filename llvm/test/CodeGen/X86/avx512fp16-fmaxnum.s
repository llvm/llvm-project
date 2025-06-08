	.file	"avx512fp16-fmaxnum.ll"
	.text
	.globl	test_intrinsic_fmaxh            # -- Begin function test_intrinsic_fmaxh
	.p2align	4
	.type	test_intrinsic_fmaxh,@function
test_intrinsic_fmaxh:                   # @test_intrinsic_fmaxh
	.cfi_startproc
# %bb.0:
	vmaxsh	%xmm0, %xmm1, %xmm2             # encoding: [0x62,0xf5,0x76,0x08,0x5f,0xd0]
	vcmpunordsh	%xmm0, %xmm0, %k1       # encoding: [0x62,0xf3,0x7e,0x08,0xc2,0xc8,0x03]
	vmovsh	%xmm1, %xmm0, %xmm2 {%k1}       # encoding: [0x62,0xf5,0x7e,0x09,0x10,0xd1]
	vmovaps	%xmm2, %xmm0                    # encoding: [0xc5,0xf8,0x28,0xc2]
	retq                                    # encoding: [0xc3]
.Lfunc_end0:
	.size	test_intrinsic_fmaxh, .Lfunc_end0-test_intrinsic_fmaxh
	.cfi_endproc
                                        # -- End function
	.globl	test_intrinsic_fmax_v2f16       # -- Begin function test_intrinsic_fmax_v2f16
	.p2align	4
	.type	test_intrinsic_fmax_v2f16,@function
test_intrinsic_fmax_v2f16:              # @test_intrinsic_fmax_v2f16
	.cfi_startproc
# %bb.0:
                                        # kill: def $xmm1 killed $xmm1 def $zmm1
                                        # kill: def $xmm0 killed $xmm0 def $zmm0
	vmaxph	%zmm0, %zmm1, %zmm2             # encoding: [0x62,0xf5,0x74,0x48,0x5f,0xd0]
	vcmpunordph	%zmm0, %zmm0, %k1       # encoding: [0x62,0xf3,0x7c,0x48,0xc2,0xc8,0x03]
	vmovdqu16	%zmm1, %zmm2 {%k1}      # encoding: [0x62,0xf1,0xff,0x49,0x6f,0xd1]
	vmovdqa	%xmm2, %xmm0                    # encoding: [0xc5,0xf9,0x6f,0xc2]
	vzeroupper                              # encoding: [0xc5,0xf8,0x77]
	retq                                    # encoding: [0xc3]
.Lfunc_end1:
	.size	test_intrinsic_fmax_v2f16, .Lfunc_end1-test_intrinsic_fmax_v2f16
	.cfi_endproc
                                        # -- End function
	.globl	test_intrinsic_fmax_v4f16       # -- Begin function test_intrinsic_fmax_v4f16
	.p2align	4
	.type	test_intrinsic_fmax_v4f16,@function
test_intrinsic_fmax_v4f16:              # @test_intrinsic_fmax_v4f16
	.cfi_startproc
# %bb.0:
                                        # kill: def $xmm1 killed $xmm1 def $zmm1
                                        # kill: def $xmm0 killed $xmm0 def $zmm0
	vmaxph	%zmm0, %zmm1, %zmm2             # encoding: [0x62,0xf5,0x74,0x48,0x5f,0xd0]
	vcmpunordph	%zmm0, %zmm0, %k1       # encoding: [0x62,0xf3,0x7c,0x48,0xc2,0xc8,0x03]
	vmovdqu16	%zmm1, %zmm2 {%k1}      # encoding: [0x62,0xf1,0xff,0x49,0x6f,0xd1]
	vmovdqa	%xmm2, %xmm0                    # encoding: [0xc5,0xf9,0x6f,0xc2]
	vzeroupper                              # encoding: [0xc5,0xf8,0x77]
	retq                                    # encoding: [0xc3]
.Lfunc_end2:
	.size	test_intrinsic_fmax_v4f16, .Lfunc_end2-test_intrinsic_fmax_v4f16
	.cfi_endproc
                                        # -- End function
	.globl	test_intrinsic_fmax_v8f16       # -- Begin function test_intrinsic_fmax_v8f16
	.p2align	4
	.type	test_intrinsic_fmax_v8f16,@function
test_intrinsic_fmax_v8f16:              # @test_intrinsic_fmax_v8f16
	.cfi_startproc
# %bb.0:
                                        # kill: def $xmm1 killed $xmm1 def $zmm1
                                        # kill: def $xmm0 killed $xmm0 def $zmm0
	vmaxph	%zmm0, %zmm1, %zmm2             # encoding: [0x62,0xf5,0x74,0x48,0x5f,0xd0]
	vcmpunordph	%zmm0, %zmm0, %k1       # encoding: [0x62,0xf3,0x7c,0x48,0xc2,0xc8,0x03]
	vmovdqu16	%zmm1, %zmm2 {%k1}      # encoding: [0x62,0xf1,0xff,0x49,0x6f,0xd1]
	vmovdqa	%xmm2, %xmm0                    # encoding: [0xc5,0xf9,0x6f,0xc2]
	vzeroupper                              # encoding: [0xc5,0xf8,0x77]
	retq                                    # encoding: [0xc3]
.Lfunc_end3:
	.size	test_intrinsic_fmax_v8f16, .Lfunc_end3-test_intrinsic_fmax_v8f16
	.cfi_endproc
                                        # -- End function
	.globl	test_intrinsic_fmax_v16f16      # -- Begin function test_intrinsic_fmax_v16f16
	.p2align	4
	.type	test_intrinsic_fmax_v16f16,@function
test_intrinsic_fmax_v16f16:             # @test_intrinsic_fmax_v16f16
	.cfi_startproc
# %bb.0:
                                        # kill: def $ymm1 killed $ymm1 def $zmm1
                                        # kill: def $ymm0 killed $ymm0 def $zmm0
	vmaxph	%zmm0, %zmm1, %zmm2             # encoding: [0x62,0xf5,0x74,0x48,0x5f,0xd0]
	vcmpunordph	%zmm0, %zmm0, %k1       # encoding: [0x62,0xf3,0x7c,0x48,0xc2,0xc8,0x03]
	vmovdqu16	%zmm1, %zmm2 {%k1}      # encoding: [0x62,0xf1,0xff,0x49,0x6f,0xd1]
	vmovdqa	%ymm2, %ymm0                    # encoding: [0xc5,0xfd,0x6f,0xc2]
	retq                                    # encoding: [0xc3]
.Lfunc_end4:
	.size	test_intrinsic_fmax_v16f16, .Lfunc_end4-test_intrinsic_fmax_v16f16
	.cfi_endproc
                                        # -- End function
	.globl	test_intrinsic_fmax_v32f16      # -- Begin function test_intrinsic_fmax_v32f16
	.p2align	4
	.type	test_intrinsic_fmax_v32f16,@function
test_intrinsic_fmax_v32f16:             # @test_intrinsic_fmax_v32f16
	.cfi_startproc
# %bb.0:
	vmaxph	%zmm0, %zmm1, %zmm2             # encoding: [0x62,0xf5,0x74,0x48,0x5f,0xd0]
	vcmpunordph	%zmm0, %zmm0, %k1       # encoding: [0x62,0xf3,0x7c,0x48,0xc2,0xc8,0x03]
	vmovdqu16	%zmm1, %zmm2 {%k1}      # encoding: [0x62,0xf1,0xff,0x49,0x6f,0xd1]
	vmovdqa64	%zmm2, %zmm0            # encoding: [0x62,0xf1,0xfd,0x48,0x6f,0xc2]
	retq                                    # encoding: [0xc3]
.Lfunc_end5:
	.size	test_intrinsic_fmax_v32f16, .Lfunc_end5-test_intrinsic_fmax_v32f16
	.cfi_endproc
                                        # -- End function
	.globl	maxnum_intrinsic_nnan_fmf_f432  # -- Begin function maxnum_intrinsic_nnan_fmf_f432
	.p2align	4
	.type	maxnum_intrinsic_nnan_fmf_f432,@function
maxnum_intrinsic_nnan_fmf_f432:         # @maxnum_intrinsic_nnan_fmf_f432
	.cfi_startproc
# %bb.0:
                                        # kill: def $xmm1 killed $xmm1 def $zmm1
                                        # kill: def $xmm0 killed $xmm0 def $zmm0
	vmaxph	%zmm1, %zmm0, %zmm0             # encoding: [0x62,0xf5,0x7c,0x48,0x5f,0xc1]
                                        # kill: def $xmm0 killed $xmm0 killed $zmm0
	vzeroupper                              # encoding: [0xc5,0xf8,0x77]
	retq                                    # encoding: [0xc3]
.Lfunc_end6:
	.size	maxnum_intrinsic_nnan_fmf_f432, .Lfunc_end6-maxnum_intrinsic_nnan_fmf_f432
	.cfi_endproc
                                        # -- End function
	.globl	maxnum_intrinsic_nnan_attr_f16  # -- Begin function maxnum_intrinsic_nnan_attr_f16
	.p2align	4
	.type	maxnum_intrinsic_nnan_attr_f16,@function
maxnum_intrinsic_nnan_attr_f16:         # @maxnum_intrinsic_nnan_attr_f16
	.cfi_startproc
# %bb.0:
	vmaxsh	%xmm1, %xmm0, %xmm0             # encoding: [0x62,0xf5,0x7e,0x08,0x5f,0xc1]
	retq                                    # encoding: [0xc3]
.Lfunc_end7:
	.size	maxnum_intrinsic_nnan_attr_f16, .Lfunc_end7-maxnum_intrinsic_nnan_attr_f16
	.cfi_endproc
                                        # -- End function
	.section	.rodata,"a",@progbits
	.p2align	1, 0x0                          # -- Begin function test_maxnum_const_op1
.LCPI8_0:
	.short	0x3c00                          # half 1
	.text
	.globl	test_maxnum_const_op1
	.p2align	4
	.type	test_maxnum_const_op1,@function
test_maxnum_const_op1:                  # @test_maxnum_const_op1
	.cfi_startproc
# %bb.0:
	vmaxsh	.LCPI8_0(%rip), %xmm0, %xmm0    # encoding: [0x62,0xf5,0x7e,0x08,0x5f,0x05,A,A,A,A]
                                        #   fixup A - offset: 6, value: .LCPI8_0-4, kind: reloc_riprel_4byte
	retq                                    # encoding: [0xc3]
.Lfunc_end8:
	.size	test_maxnum_const_op1, .Lfunc_end8-test_maxnum_const_op1
	.cfi_endproc
                                        # -- End function
	.section	.rodata,"a",@progbits
	.p2align	1, 0x0                          # -- Begin function test_maxnum_const_op2
.LCPI9_0:
	.short	0x3c00                          # half 1
	.text
	.globl	test_maxnum_const_op2
	.p2align	4
	.type	test_maxnum_const_op2,@function
test_maxnum_const_op2:                  # @test_maxnum_const_op2
	.cfi_startproc
# %bb.0:
	vmaxsh	.LCPI9_0(%rip), %xmm0, %xmm0    # encoding: [0x62,0xf5,0x7e,0x08,0x5f,0x05,A,A,A,A]
                                        #   fixup A - offset: 6, value: .LCPI9_0-4, kind: reloc_riprel_4byte
	retq                                    # encoding: [0xc3]
.Lfunc_end9:
	.size	test_maxnum_const_op2, .Lfunc_end9-test_maxnum_const_op2
	.cfi_endproc
                                        # -- End function
	.globl	test_maxnum_const_nan           # -- Begin function test_maxnum_const_nan
	.p2align	4
	.type	test_maxnum_const_nan,@function
test_maxnum_const_nan:                  # @test_maxnum_const_nan
	.cfi_startproc
# %bb.0:
	retq                                    # encoding: [0xc3]
.Lfunc_end10:
	.size	test_maxnum_const_nan, .Lfunc_end10-test_maxnum_const_nan
	.cfi_endproc
                                        # -- End function
	.section	".note.GNU-stack","",@progbits
