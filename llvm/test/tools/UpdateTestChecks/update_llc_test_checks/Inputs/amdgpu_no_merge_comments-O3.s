	.text
	.section	.AMDGPU.config
	.long	47176
	.long	0
	.long	47180
	.long	0
	.long	47200
	.long	0
	.long	4
	.long	0
	.long	8
	.long	0
	.text
	.hidden	main                            ; -- Begin function main
	.globl	main
	.p2align	2
	.type	main,@function
main:                                   ; @main
; %bb.0:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	v_add_u32_e32 v1, v0, v0
	v_mul_lo_u32 v0, v1, v0
	v_sub_u32_e32 v0, v0, v1
	s_setpc_b64 s[30:31]
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
                                        ; -- End function
	.section	.AMDGPU.csdata
; Function info:
; codeLenInByte = 24
; NumSgprs: 36
; NumVgprs: 2
; ScratchSize: 0
; MemoryBound: 0
	.section	".note.GNU-stack"
	.amd_amdgpu_isa "amdgcn-unknown---gfx900"
