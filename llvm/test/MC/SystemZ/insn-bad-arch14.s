# For arch14 only.
# RUN: not llvm-mc -triple s390x-linux-gnu -mcpu=arch14 < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: lbear	-1
#CHECK: error: invalid operand
#CHECK: lbear	4096
#CHECK: error: invalid use of indexed addressing
#CHECK: lbear	0(%r1,%r2)

	lbear	-1
	lbear	4096
	lbear	0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: lpswey	-524289
#CHECK: error: invalid operand
#CHECK: lpswey	524288
#CHECK: error: invalid use of indexed addressing
#CHECK: lpswey	0(%r1,%r2)

	lpswey	-524289
	lpswey	524288
	lpswey	0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: qpaci	-1
#CHECK: error: invalid operand
#CHECK: qpaci	4096
#CHECK: error: invalid use of indexed addressing
#CHECK: qpaci	0(%r1,%r2)

	qpaci	-1
	qpaci	4096
	qpaci	0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: rdp	%r0, %r0, %r0, -1
#CHECK: error: invalid operand
#CHECK: rdp	%r0, %r0, %r0, 16

	rdp	%r0, %r0, %r0, -1
	rdp	%r0, %r0, %r0, 16

#CHECK: error: invalid operand
#CHECK: stbear	-1
#CHECK: error: invalid operand
#CHECK: stbear	4096
#CHECK: error: invalid use of indexed addressing
#CHECK: stbear	0(%r1,%r2)

	stbear	-1
	stbear	4096
	stbear	0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: vcfn	%v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vcfn	%v0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: vcfn	%v0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vcfn	%v0, %v0, 16, 0

	vcfn	%v0, %v0, 0, -1
	vcfn	%v0, %v0, 0, 16
	vcfn	%v0, %v0, -1, 0
	vcfn	%v0, %v0, 16, 0

#CHECK: error: invalid operand
#CHECK: vclfnl	%v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vclfnl	%v0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: vclfnl	%v0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vclfnl	%v0, %v0, 16, 0

	vclfnl	%v0, %v0, 0, -1
	vclfnl	%v0, %v0, 0, 16
	vclfnl	%v0, %v0, -1, 0
	vclfnl	%v0, %v0, 16, 0

#CHECK: error: invalid operand
#CHECK: vclfnh	%v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vclfnh	%v0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: vclfnh	%v0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vclfnh	%v0, %v0, 16, 0

	vclfnh	%v0, %v0, 0, -1
	vclfnh	%v0, %v0, 0, 16
	vclfnh	%v0, %v0, -1, 0
	vclfnh	%v0, %v0, 16, 0

#CHECK: error: invalid operand
#CHECK: vcnf	%v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vcnf	%v0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: vcnf	%v0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vcnf	%v0, %v0, 16, 0

	vcnf	%v0, %v0, 0, -1
	vcnf	%v0, %v0, 0, 16
	vcnf	%v0, %v0, -1, 0
	vcnf	%v0, %v0, 16, 0

#CHECK: error: invalid operand
#CHECK: vcrnf	%v0, %v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vcrnf	%v0, %v0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: vcrnf	%v0, %v0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vcrnf	%v0, %v0, %v0, 16, 0

	vcrnf	%v0, %v0, %v0, 0, -1
	vcrnf	%v0, %v0, %v0, 0, 16
	vcrnf	%v0, %v0, %v0, -1, 0
	vcrnf	%v0, %v0, %v0, 16, 0

#CHECK: error: invalid operand
#CHECK: vclzdp	%v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vclzdp	%v0, %v0, 16

	vclzdp	%v0, %v0, -1
	vclzdp	%v0, %v0, 16

#CHECK: error: invalid operand
#CHECK: vcsph	%v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vcsph	%v0, %v0, %v0, 16

	vcsph	%v0, %v0, %v0, -1
	vcsph	%v0, %v0, %v0, 16

#CHECK: error: invalid operand
#CHECK: vpkzr	%v0, %v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vpkzr	%v0, %v0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: vpkzr	%v0, %v0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vpkzr	%v0, %v0, %v0, 256, 0

	vpkzr	%v0, %v0, %v0, 0, -1
	vpkzr	%v0, %v0, %v0, 0, 16
	vpkzr	%v0, %v0, %v0, -1, 0
	vpkzr	%v0, %v0, %v0, 256, 0

#CHECK: error: invalid operand
#CHECK: vschp	%v0, %v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vschp	%v0, %v0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: vschp	%v0, %v0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vschp	%v0, %v0, %v0, 16, 0

	vschp	%v0, %v0, %v0, 0, -1
	vschp	%v0, %v0, %v0, 0, 16
	vschp	%v0, %v0, %v0, -1, 0
	vschp	%v0, %v0, %v0, 16, 0

#CHECK: error: invalid operand
#CHECK: vschsp	%v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vschsp	%v0, %v0, %v0, 16

	vschsp	%v0, %v0, %v0, -1
	vschsp	%v0, %v0, %v0, 16

#CHECK: error: invalid operand
#CHECK: vschdp	%v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vschdp	%v0, %v0, %v0, 16

	vschdp	%v0, %v0, %v0, -1
	vschdp	%v0, %v0, %v0, 16

#CHECK: error: invalid operand
#CHECK: vschxp	%v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vschxp	%v0, %v0, %v0, 16

	vschxp	%v0, %v0, %v0, -1
	vschxp	%v0, %v0, %v0, 16

#CHECK: error: invalid operand
#CHECK: vsrpr	%v0, %v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vsrpr	%v0, %v0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: vsrpr	%v0, %v0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vsrpr	%v0, %v0, %v0, 256, 0

	vsrpr	%v0, %v0, %v0, 0, -1
	vsrpr	%v0, %v0, %v0, 0, 16
	vsrpr	%v0, %v0, %v0, -1, 0
	vsrpr	%v0, %v0, %v0, 256, 0

#CHECK: error: invalid operand
#CHECK: vupkzh	%v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vupkzh	%v0, %v0, 16

	vupkzh	%v0, %v0, -1
	vupkzh	%v0, %v0, 16

#CHECK: error: invalid operand
#CHECK: vupkzl	%v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vupkzl	%v0, %v0, 16

	vupkzl	%v0, %v0, -1
	vupkzl	%v0, %v0, 16
