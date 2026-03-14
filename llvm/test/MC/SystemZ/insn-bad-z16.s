# For z16 only.
# RUN: not llvm-mc -triple s390x-linux-gnu -mcpu=z16 < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: instruction requires: miscellaneous-extensions-4
#CHECK: bdepg	%r0, %r0, %r0

	bdepg	%r0, %r0, %r0

#CHECK: error: instruction requires: miscellaneous-extensions-4
#CHECK: bextg	%r0, %r0, %r0

	bextg	%r0, %r0, %r0

#CHECK: error: instruction requires: concurrent-functions
#CHECK: cal	%r0, 0, 0

	cal	%r0, 0, 0

#CHECK: error: instruction requires: concurrent-functions
#CHECK: calg	%r0, 0, 0

	calg	%r0, 0, 0

#CHECK: error: instruction requires: concurrent-functions
#CHECK: calgf	%r0, 0, 0

	calgf	%r0, 0, 0

#CHECK: error: instruction requires: miscellaneous-extensions-4
#CHECK: clzg	%r0, %r0

	clzg	%r0, %r0

#CHECK: error: instruction requires: miscellaneous-extensions-4
#CHECK: ctzg	%r0, %r0

	ctzg	%r0, %r0

#CHECK: error: instruction requires: message-security-assist-extension12
#CHECK: kimd	%r0, %r0, 0

	kimd	%r0, %r0, 0

#CHECK: error: instruction requires: message-security-assist-extension12
#CHECK: klmd	%r0, %r0, 0

	klmd	%r0, %r0, 0

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

#CHECK: error: instruction requires: miscellaneous-extensions-4
#CHECK: lxab	%r0, 0

        lxab    %r0, 0

#CHECK: error: instruction requires: miscellaneous-extensions-4
#CHECK: lxah	%r0, 0

        lxah    %r0, 0

#CHECK: error: instruction requires: miscellaneous-extensions-4
#CHECK: lxaf	%r0, 0

        lxaf    %r0, 0

#CHECK: error: instruction requires: miscellaneous-extensions-4
#CHECK: lxag	%r0, 0

        lxag    %r0, 0

#CHECK: error: instruction requires: miscellaneous-extensions-4
#CHECK: lxaq	%r0, 0

        lxaq    %r0, 0

#CHECK: error: instruction requires: miscellaneous-extensions-4
#CHECK: llxab	%r0, 0

        llxab    %r0, 0

#CHECK: error: instruction requires: miscellaneous-extensions-4
#CHECK: llxah	%r0, 0

        llxah    %r0, 0

#CHECK: error: instruction requires: miscellaneous-extensions-4
#CHECK: llxaf	%r0, 0

        llxaf    %r0, 0

#CHECK: error: instruction requires: miscellaneous-extensions-4
#CHECK: llxag	%r0, 0

        llxag    %r0, 0

#CHECK: error: instruction requires: miscellaneous-extensions-4
#CHECK: llxaq	%r0, 0

        llxaq    %r0, 0

#CHECK: error: instruction requires: concurrent-functions
#CHECK: pfcr	%r0, %r0, 0

	pfcr	%r0, %r0, 0

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

#CHECK: error: instruction requires: vector-enhancements-3
#CHECK: vavgq	%v0, %v0, %v0

	vavgq	%v0, %v0, %v0

#CHECK: error: instruction requires: vector-enhancements-3
#CHECK: vavglq	%v0, %v0, %v0

	vavglq	%v0, %v0, %v0

#CHECK: error: instruction requires: vector-enhancements-3
#CHECK: vblend	%v0, %v0, %v0, %v0, 0
#CHECK: error: instruction requires: vector-enhancements-3
#CHECK: vblendb	%v0, %v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-3
#CHECK: vblendh	%v0, %v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-3
#CHECK: vblendf	%v0, %v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-3
#CHECK: vblendg	%v0, %v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-3
#CHECK: vblendq	%v0, %v0, %v0, %v0

	vblend	%v0, %v0, %v0, %v0, 0
	vblendb	%v0, %v0, %v0, %v0
	vblendh	%v0, %v0, %v0, %v0
	vblendf	%v0, %v0, %v0, %v0
	vblendg	%v0, %v0, %v0, %v0
	vblendq	%v0, %v0, %v0, %v0

#CHECK: error: instruction requires: vector-enhancements-3
#CHECK: vceqq	%v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-3
#CHECK: vceqqs	%v0, %v0, %v0

	vceqq	%v0, %v0, %v0
	vceqqs	%v0, %v0, %v0

#CHECK: error: instruction requires: vector-enhancements-3
#CHECK: vchq	%v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-3
#CHECK: vchqs	%v0, %v0, %v0

	vchq	%v0, %v0, %v0
	vchqs	%v0, %v0, %v0

#CHECK: error: instruction requires: vector-enhancements-3
#CHECK: vchlq	%v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-3
#CHECK: vchlqs	%v0, %v0, %v0

	vchlq	%v0, %v0, %v0
	vchlqs	%v0, %v0, %v0

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

#CHECK: error: instruction requires: vector-enhancements-3
#CHECK: vclzq	%v0, %v0

        vclzq	%v0, %v0

#CHECK: error: instruction requires: vector-enhancements-3
#CHECK: vctzq	%v0, %v0

        vctzq	%v0, %v0

#CHECK: error: invalid operand
#CHECK: vcsph	%v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vcsph	%v0, %v0, %v0, 16

	vcsph	%v0, %v0, %v0, -1
	vcsph	%v0, %v0, %v0, 16

#CHECK: error: instruction requires: vector-packed-decimal-enhancement-3
#CHECK: vcvbq	%v0, %v0, 0

	vcvbq	%v0, %v0, 0

#CHECK: error: instruction requires: vector-packed-decimal-enhancement-3
#CHECK: vcvdq	%v0, %v0, 0, 0

	vcvdq	%v0, %v0, 0, 0

#CHECK: error: instruction requires: vector-enhancements-3
#CHECK: vd	%v0, %v0, %v0, 0, 0
#CHECK: vdf	%v0, %v0, %v0, 0
#CHECK: vdg	%v0, %v0, %v0, 0
#CHECK: vdq	%v0, %v0, %v0, 0

	vd	%v0, %v0, %v0, 0, 0
	vdf	%v0, %v0, %v0, 0
	vdg	%v0, %v0, %v0, 0
	vdq	%v0, %v0, %v0, 0

#CHECK: error: instruction requires: vector-enhancements-3
#CHECK: vdl	%v0, %v0, %v0, 0, 0
#CHECK: vdlf	%v0, %v0, %v0, 0
#CHECK: vdlg	%v0, %v0, %v0, 0
#CHECK: vdlq	%v0, %v0, %v0, 0

	vdl	%v0, %v0, %v0, 0, 0
	vdlf	%v0, %v0, %v0, 0
	vdlg	%v0, %v0, %v0, 0
	vdlq	%v0, %v0, %v0, 0

#CHECK: error: instruction requires: vector-enhancements-3
#CHECK: veval	%v0, %v0, %v0, %v0, 0

	veval	%v0, %v0, %v0, %v0, 0

#CHECK: error: instruction requires: vector-enhancements-3
#CHECK: vecq	%v0, %v0

	vecq	%v0, %v0

#CHECK: error: instruction requires: vector-enhancements-3
#CHECK: veclq	%v0, %v0

	veclq	%v0, %v0

#CHECK: error: instruction requires: vector-enhancements-3
#CHECK: vgem	%v0, %v0, 0
#CHECK: error: instruction requires: vector-enhancements-3
#CHECK: vgemb	%v0, %v0
#CHECK: error: instruction requires: vector-enhancements-3
#CHECK: vgemh	%v0, %v0
#CHECK: error: instruction requires: vector-enhancements-3
#CHECK: vgemf	%v0, %v0
#CHECK: error: instruction requires: vector-enhancements-3
#CHECK: vgemg	%v0, %v0
#CHECK: error: instruction requires: vector-enhancements-3
#CHECK: vgemq	%v0, %v0

	vgem	%v0, %v0, 0
	vgemb	%v0, %v0
	vgemh	%v0, %v0
	vgemf	%v0, %v0
	vgemg	%v0, %v0
	vgemq	%v0, %v0

#CHECK: error: instruction requires: vector-enhancements-3
#CHECK: vlcq	%v0, %v0

        vlcq	%v0, %v0

#CHECK: error: instruction requires: vector-enhancements-3
#CHECK: vlpq	%v0, %v0

        vlpq	%v0, %v0

#CHECK: error: instruction requires: vector-enhancements-3
#CHECK: vmalg	%v0, %v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-3
#CHECK: vmalq	%v0, %v0, %v0, %v0

	vmalg	%v0, %v0, %v0, %v0
	vmalq	%v0, %v0, %v0, %v0

#CHECK: error: instruction requires: vector-enhancements-3
#CHECK: vmahg	%v0, %v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-3
#CHECK: vmahq	%v0, %v0, %v0, %v0

	vmahg	%v0, %v0, %v0, %v0
	vmahq	%v0, %v0, %v0, %v0

#CHECK: error: instruction requires: vector-enhancements-3
#CHECK: vmalhg	%v0, %v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-3
#CHECK: vmalhq	%v0, %v0, %v0, %v0

	vmalhg	%v0, %v0, %v0, %v0
	vmalhq	%v0, %v0, %v0, %v0

#CHECK: error: instruction requires: vector-enhancements-3
#CHECK: vmaeg	%v0, %v0, %v0, %v0

	vmaeg	%v0, %v0, %v0, %v0

#CHECK: error: instruction requires: vector-enhancements-3
#CHECK: vmaleg	%v0, %v0, %v0, %v0

	vmaleg	%v0, %v0, %v0, %v0

#CHECK: error: instruction requires: vector-enhancements-3
#CHECK: vmaog	%v0, %v0, %v0, %v0

	vmaog	%v0, %v0, %v0, %v0

#CHECK: error: instruction requires: vector-enhancements-3
#CHECK: vmalog	%v0, %v0, %v0, %v0

	vmalog	%v0, %v0, %v0, %v0

#CHECK: error: instruction requires: vector-enhancements-3
#CHECK: vmlg	%v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-3
#CHECK: vmlq	%v0, %v0, %v0

	vmlg	%v0, %v0, %v0
	vmlq	%v0, %v0, %v0

#CHECK: error: instruction requires: vector-enhancements-3
#CHECK: vmhg	%v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-3
#CHECK: vmhq	%v0, %v0, %v0

	vmhg	%v0, %v0, %v0
	vmhq	%v0, %v0, %v0

#CHECK: error: instruction requires: vector-enhancements-3
#CHECK: vmlhg	%v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-3
#CHECK: vmlhq	%v0, %v0, %v0

	vmlhg	%v0, %v0, %v0
	vmlhq	%v0, %v0, %v0

#CHECK: error: instruction requires: vector-enhancements-3
#CHECK: vmeg	%v0, %v0, %v0

	vmeg	%v0, %v0, %v0

#CHECK: error: instruction requires: vector-enhancements-3
#CHECK: vmleg	%v0, %v0, %v0

	vmleg	%v0, %v0, %v0

#CHECK: error: instruction requires: vector-enhancements-3
#CHECK: vmog	%v0, %v0, %v0

	vmog	%v0, %v0, %v0

#CHECK: error: instruction requires: vector-enhancements-3
#CHECK: vmlog	%v0, %v0, %v0

	vmlog	%v0, %v0, %v0

#CHECK: error: instruction requires: vector-enhancements-3
#CHECK: vmnq	%v0, %v0, %v0

	vmnq	%v0, %v0, %v0

#CHECK: error: instruction requires: vector-enhancements-3
#CHECK: vmnlq	%v0, %v0, %v0

	vmnlq	%v0, %v0, %v0

#CHECK: error: instruction requires: vector-enhancements-3
#CHECK: vmxq	%v0, %v0, %v0

	vmxq	%v0, %v0, %v0

#CHECK: error: instruction requires: vector-enhancements-3
#CHECK: vmxlq	%v0, %v0, %v0

	vmxlq	%v0, %v0, %v0

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

#CHECK: error: instruction requires: vector-enhancements-3
#CHECK: vr	%v0, %v0, %v0, 0, 0
#CHECK: vrf	%v0, %v0, %v0, 0
#CHECK: vrg	%v0, %v0, %v0, 0
#CHECK: vrq	%v0, %v0, %v0, 0

	vr	%v0, %v0, %v0, 0, 0
	vrf	%v0, %v0, %v0, 0
	vrg	%v0, %v0, %v0, 0
	vrq	%v0, %v0, %v0, 0

#CHECK: error: instruction requires: vector-enhancements-3
#CHECK: vrl	%v0, %v0, %v0, 0, 0
#CHECK: vrlf	%v0, %v0, %v0, 0
#CHECK: vrlg	%v0, %v0, %v0, 0
#CHECK: vrlq	%v0, %v0, %v0, 0

	vrl	%v0, %v0, %v0, 0, 0
	vrlf	%v0, %v0, %v0, 0
	vrlg	%v0, %v0, %v0, 0
	vrlq	%v0, %v0, %v0, 0

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

#CHECK: error: instruction requires: vector-packed-decimal-enhancement-3
#CHECK: vtp     %v0, 0

	vtp     %v0, 0

#CHECK: error: instruction requires: vector-packed-decimal-enhancement-3
#CHECK: vtz     %v0, %v0, 0

	vtz     %v0, %v0, 0

#CHECK: error: instruction requires: vector-enhancements-3
#CHECK: vuphg	%v0, %v0

	vuphg	%v0, %v0

#CHECK: error: instruction requires: vector-enhancements-3
#CHECK: vuplg	%v0, %v0

	vuplg	%v0, %v0

#CHECK: error: instruction requires: vector-enhancements-3
#CHECK: vuplhg	%v0, %v0

	vuplhg	%v0, %v0

#CHECK: error: instruction requires: vector-enhancements-3
#CHECK: vupllg	%v0, %v0

	vupllg	%v0, %v0

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
