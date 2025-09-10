	.attribute	4, 16
	.attribute	5, "rv64i2p1"
	.file	"icmp_ult_imm.ll"
	.text
	.globl	icmp_ult_imm                    # -- Begin function icmp_ult_imm
	.p2align	2
	.type	icmp_ult_imm,@function
icmp_ult_imm:                           # @icmp_ult_imm
	.cfi_startproc
# %bb.0:                                # %entry
	sext.w	a0, a0
	sltiu	a0, a0, 15
	ret
.Lfunc_end0:
	.size	icmp_ult_imm, .Lfunc_end0-icmp_ult_imm
	.cfi_endproc
                                        # -- End function
	.section	".note.GNU-stack","",@progbits
