# For z13 only.
# RUN: not llvm-mc -triple s390x-linux-gnu -mcpu=z13 < %s 2> %t
# RUN: FileCheck < %t %s
# RUN: not llvm-mc -triple s390x-linux-gnu -mcpu=arch11 < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: instruction requires: miscellaneous-extensions-2
#CHECK: agh	%r0, 0

	agh	%r0, 0

#CHECK: error: instruction requires: miscellaneous-extensions-2
#CHECK: bi	0
#CHECK: error: instruction requires: miscellaneous-extensions-2
#CHECK: bic	0, 0

	bi	0
	bic	0, 0

#CHECK: error: invalid operand
#CHECK: cdpt	%f0, 0(1), -1
#CHECK: error: invalid operand
#CHECK: cdpt	%f0, 0(1), 16
#CHECK: error: missing length in address
#CHECK: cdpt	%f0, 0, 0
#CHECK: error: missing length in address
#CHECK: cdpt	%f0, 0(%r1), 0
#CHECK: error: invalid operand
#CHECK: cdpt	%f0, 0(0,%r1), 0
#CHECK: error: invalid operand
#CHECK: cdpt	%f0, 0(257,%r1), 0
#CHECK: error: invalid operand
#CHECK: cdpt	%f0, -1(1,%r1), 0
#CHECK: error: invalid operand
#CHECK: cdpt	%f0, 4096(1,%r1), 0
#CHECK: error: invalid use of indexed addressing
#CHECK: cdpt	%f0, 0(%r1,%r2), 0
#CHECK: error: unknown token in expression
#CHECK: cdpt	%f0, 0(-), 0

	cdpt	%f0, 0(1), -1
	cdpt	%f0, 0(1), 16
	cdpt	%f0, 0, 0
	cdpt	%f0, 0(%r1), 0
	cdpt	%f0, 0(0,%r1), 0
	cdpt	%f0, 0(257,%r1), 0
	cdpt	%f0, -1(1,%r1), 0
	cdpt	%f0, 4096(1,%r1), 0
	cdpt	%f0, 0(%r1,%r2), 0
	cdpt	%f0, 0(-), 0

#CHECK: error: invalid operand
#CHECK: cpdt	%f0, 0(1), -1
#CHECK: error: invalid operand
#CHECK: cpdt	%f0, 0(1), 16
#CHECK: error: missing length in address
#CHECK: cpdt	%f0, 0, 0
#CHECK: error: missing length in address
#CHECK: cpdt	%f0, 0(%r1), 0
#CHECK: error: invalid operand
#CHECK: cpdt	%f0, 0(0,%r1), 0
#CHECK: error: invalid operand
#CHECK: cpdt	%f0, 0(257,%r1), 0
#CHECK: error: invalid operand
#CHECK: cpdt	%f0, -1(1,%r1), 0
#CHECK: error: invalid operand
#CHECK: cpdt	%f0, 4096(1,%r1), 0
#CHECK: error: invalid use of indexed addressing
#CHECK: cpdt	%f0, 0(%r1,%r2), 0
#CHECK: error: unknown token in expression
#CHECK: cpdt	%f0, 0(-), 0

	cpdt	%f0, 0(1), -1
	cpdt	%f0, 0(1), 16
	cpdt	%f0, 0, 0
	cpdt	%f0, 0(%r1), 0
	cpdt	%f0, 0(0,%r1), 0
	cpdt	%f0, 0(257,%r1), 0
	cpdt	%f0, -1(1,%r1), 0
	cpdt	%f0, 4096(1,%r1), 0
	cpdt	%f0, 0(%r1,%r2), 0
	cpdt	%f0, 0(-), 0

#CHECK: error: invalid operand
#CHECK: cpxt	%f0, 0(1), -1
#CHECK: error: invalid operand
#CHECK: cpxt	%f0, 0(1), 16
#CHECK: error: missing length in address
#CHECK: cpxt	%f0, 0, 0
#CHECK: error: missing length in address
#CHECK: cpxt	%f0, 0(%r1), 0
#CHECK: error: invalid operand
#CHECK: cpxt	%f0, 0(0,%r1), 0
#CHECK: error: invalid operand
#CHECK: cpxt	%f0, 0(257,%r1), 0
#CHECK: error: invalid operand
#CHECK: cpxt	%f0, -1(1,%r1), 0
#CHECK: error: invalid operand
#CHECK: cpxt	%f0, 4096(1,%r1), 0
#CHECK: error: invalid use of indexed addressing
#CHECK: cpxt	%f0, 0(%r1,%r2), 0
#CHECK: error: unknown token in expression
#CHECK: cpxt	%f0, 0(-), 0
#CHECK: error: invalid register pair
#CHECK: cpxt	%f15, 0(1), 0

	cpxt	%f0, 0(1), -1
	cpxt	%f0, 0(1), 16
	cpxt	%f0, 0, 0
	cpxt	%f0, 0(%r1), 0
	cpxt	%f0, 0(0,%r1), 0
	cpxt	%f0, 0(257,%r1), 0
	cpxt	%f0, -1(1,%r1), 0
	cpxt	%f0, 4096(1,%r1), 0
	cpxt	%f0, 0(%r1,%r2), 0
	cpxt	%f0, 0(-), 0
	cpxt	%f15, 0(1), 0

#CHECK: error: invalid operand
#CHECK: cxpt	%f0, 0(1), -1
#CHECK: error: invalid operand
#CHECK: cxpt	%f0, 0(1), 16
#CHECK: error: missing length in address
#CHECK: cxpt	%f0, 0, 0
#CHECK: error: missing length in address
#CHECK: cxpt	%f0, 0(%r1), 0
#CHECK: error: invalid operand
#CHECK: cxpt	%f0, 0(0,%r1), 0
#CHECK: error: invalid operand
#CHECK: cxpt	%f0, 0(257,%r1), 0
#CHECK: error: invalid operand
#CHECK: cxpt	%f0, -1(1,%r1), 0
#CHECK: error: invalid operand
#CHECK: cxpt	%f0, 4096(1,%r1), 0
#CHECK: error: invalid use of indexed addressing
#CHECK: cxpt	%f0, 0(%r1,%r2), 0
#CHECK: error: unknown token in expression
#CHECK: cxpt	%f0, 0(-), 0
#CHECK: error: invalid register pair
#CHECK: cxpt	%f15, 0(1), 0

	cxpt	%f0, 0(1), -1
	cxpt	%f0, 0(1), 16
	cxpt	%f0, 0, 0
	cxpt	%f0, 0(%r1), 0
	cxpt	%f0, 0(0,%r1), 0
	cxpt	%f0, 0(257,%r1), 0
	cxpt	%f0, -1(1,%r1), 0
	cxpt	%f0, 4096(1,%r1), 0
	cxpt	%f0, 0(%r1,%r2), 0
	cxpt	%f0, 0(-), 0
	cxpt	%f15, 0(1), 0

#CHECK: error: instruction requires: insert-reference-bits-multiple
#CHECK: irbm	%r0, %r0

	irbm	%r0, %r0

#CHECK: error: instruction requires: message-security-assist-extension8
#CHECK: kma	%r2, %r4, %r6

	kma	%r2, %r4, %r6

#CHECK: error: invalid operand
#CHECK: lcbb	%r0, 0, -1
#CHECK: error: invalid operand
#CHECK: lcbb	%r0, 0, 16
#CHECK: error: invalid operand
#CHECK: lcbb	%r0, -1, 0
#CHECK: error: invalid operand
#CHECK: lcbb	%r0, 4096, 0
#CHECK: error: invalid use of vector addressing
#CHECK: lcbb	%r0, 0(%v1,%r2), 0

	lcbb	%r0, 0, -1
	lcbb	%r0, 0, 16
	lcbb	%r0, -1, 0
	lcbb	%r0, 4096, 0
	lcbb	%r0, 0(%v1,%r2), 0

#CHECK: error: instruction requires: guarded-storage
#CHECK: lgg	%r0, 0

	lgg	%r0, 0

#CHECK: error: instruction requires: guarded-storage
#CHECK: lgsc	%r0, 0

	lgsc	%r0, 0

#CHECK: error: instruction requires: guarded-storage
#CHECK: llgfsg	%r0, 0

	llgfsg	%r0, 0

#CHECK: error: invalid operand
#CHECK: llzrgf	%r0, -524289
#CHECK: error: invalid operand
#CHECK: llzrgf	%r0, 524288

	llzrgf	%r0, -524289
	llzrgf	%r0, 524288

#CHECK: error: invalid operand
#CHECK: locfh	%r0, 0, -1
#CHECK: error: invalid operand
#CHECK: locfh	%r0, 0, 16
#CHECK: error: invalid operand
#CHECK: locfh	%r0, -524289, 1
#CHECK: error: invalid operand
#CHECK: locfh	%r0, 524288, 1
#CHECK: error: invalid use of indexed addressing
#CHECK: locfh	%r0, 0(%r1,%r2), 1

	locfh	%r0, 0, -1
	locfh	%r0, 0, 16
	locfh	%r0, -524289, 1
	locfh	%r0, 524288, 1
	locfh	%r0, 0(%r1,%r2), 1

#CHECK: error: invalid operand
#CHECK: locfhr	%r0, %r0, -1
#CHECK: error: invalid operand
#CHECK: locfhr	%r0, %r0, 16

	locfhr	%r0, %r0, -1
	locfhr	%r0, %r0, 16

#CHECK: error: invalid operand
#CHECK: locghie	%r0, 66000
#CHECK: error: invalid operand
#CHECK: locghie	%f0, 0
#CHECK: error: invalid operand
#CHECK: locghie	0, %r0

	locghie	%r0, 66000
	locghie	%f0, 0
	locghie	0, %r0

#CHECK: error: invalid operand
#CHECK: lochhie	%r0, 66000
#CHECK: error: invalid operand
#CHECK: lochhie	%f0, 0
#CHECK: error: invalid operand
#CHECK: lochhie	0, %r0

	lochhie	%r0, 66000
	lochhie	%f0, 0
	lochhie	0, %r0

#CHECK: error: invalid operand
#CHECK: lochie	%r0, 66000
#CHECK: error: invalid operand
#CHECK: lochie	%f0, 0
#CHECK: error: invalid operand
#CHECK: lochie	0, %r0

	lochie	%r0, 66000
	lochie	%f0, 0
	lochie	0, %r0

#CHECK: error: invalid operand
#CHECK: lzrf	%r0, -524289
#CHECK: error: invalid operand
#CHECK: lzrf	%r0, 524288

	lzrf	%r0, -524289
	lzrf	%r0, 524288

#CHECK: error: invalid operand
#CHECK: lzrg	%r0, -524289
#CHECK: error: invalid operand
#CHECK: lzrg	%r0, 524288

	lzrg	%r0, -524289
	lzrg	%r0, 524288

#CHECK: error: instruction requires: miscellaneous-extensions-2
#CHECK: mg	%r0, 0

	mg	%r0, 0

#CHECK: error: instruction requires: miscellaneous-extensions-2
#CHECK: mgh	%r0, 0

	mgh	%r0, 0

#CHECK: error: instruction requires: miscellaneous-extensions-2
#CHECK: mgrk	%r0, %r0, %r0

	mgrk	%r0, %r0, %r0

#CHECK: error: instruction requires: miscellaneous-extensions-2
#CHECK: msc	%r0, 0

	msc	%r0, 0

#CHECK: error: instruction requires: miscellaneous-extensions-2
#CHECK: msgc	%r0, 0

	msgc	%r0, 0

#CHECK: error: instruction requires: miscellaneous-extensions-2
#CHECK: msrkc	%r0, %r0, %r0

	msrkc	%r0, %r0, %r0

#CHECK: error: instruction requires: miscellaneous-extensions-2
#CHECK: msgrkc	%r0, %r0, %r0

	msgrkc	%r0, %r0, %r0

#CHECK: error: invalid register pair
#CHECK: ppno	%r1, %r2
#CHECK: error: invalid register pair
#CHECK: ppno	%r2, %r1

	ppno	%r1, %r2
	ppno	%r2, %r1

#CHECK: error: instruction requires: message-security-assist-extension7
#CHECK: prno	%r2, %r4

	prno	%r2, %r4

#CHECK: error: instruction requires: miscellaneous-extensions-2
#CHECK: sgh	%r0, 0

	sgh	%r0, 0

#CHECK: error: instruction requires: guarded-storage
#CHECK: stgsc	%r0, 0

	stgsc	%r0, 0

#CHECK: error: invalid operand
#CHECK: stocfh	%r0, 0, -1
#CHECK: error: invalid operand
#CHECK: stocfh	%r0, 0, 16
#CHECK: error: invalid operand
#CHECK: stocfh	%r0, -524289, 1
#CHECK: error: invalid operand
#CHECK: stocfh	%r0, 524288, 1
#CHECK: error: invalid use of indexed addressing
#CHECK: stocfh	%r0, 0(%r1,%r2), 1

	stocfh	%r0, 0, -1
	stocfh	%r0, 0, 16
	stocfh	%r0, -524289, 1
	stocfh	%r0, 524288, 1
	stocfh	%r0, 0(%r1,%r2), 1

#CHECK: error: instruction requires: vector-packed-decimal
#CHECK: vap	%v0, %v0, %v0, 0, 0

	vap	%v0, %v0, %v0, 0, 0

#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: vbperm	%v0, %v0, %v0

	vbperm	%v0, %v0, %v0

#CHECK: error: invalid operand
#CHECK: vcdg	%v0, %v0, 0, 0, -1
#CHECK: error: invalid operand
#CHECK: vcdg	%v0, %v0, 0, 0, 16
#CHECK: error: invalid operand
#CHECK: vcdg	%v0, %v0, 0, -1, 0
#CHECK: error: invalid operand
#CHECK: vcdg	%v0, %v0, 0, 16, 0
#CHECK: error: invalid operand
#CHECK: vcdg	%v0, %v0, -1, 0, 0
#CHECK: error: invalid operand
#CHECK: vcdg	%v0, %v0, 16, 0, 0

	vcdg	%v0, %v0, 0, 0, -1
	vcdg	%v0, %v0, 0, 0, 16
	vcdg	%v0, %v0, 0, -1, 0
	vcdg	%v0, %v0, 0, 16, 0
	vcdg	%v0, %v0, -1, 0, 0
	vcdg	%v0, %v0, 16, 0, 0

#CHECK: error: invalid operand
#CHECK: vcdgb	%v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vcdgb	%v0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: vcdgb	%v0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vcdgb	%v0, %v0, 16, 0

	vcdgb	%v0, %v0, 0, -1
	vcdgb	%v0, %v0, 0, 16
	vcdgb	%v0, %v0, -1, 0
	vcdgb	%v0, %v0, 16, 0

#CHECK: error: invalid operand
#CHECK: vcdlg	%v0, %v0, 0, 0, -1
#CHECK: error: invalid operand
#CHECK: vcdlg	%v0, %v0, 0, 0, 16
#CHECK: error: invalid operand
#CHECK: vcdlg	%v0, %v0, 0, -1, 0
#CHECK: error: invalid operand
#CHECK: vcdlg	%v0, %v0, 0, 16, 0
#CHECK: error: invalid operand
#CHECK: vcdlg	%v0, %v0, -1, 0, 0
#CHECK: error: invalid operand
#CHECK: vcdlg	%v0, %v0, 16, 0, 0

	vcdlg	%v0, %v0, 0, 0, -1
	vcdlg	%v0, %v0, 0, 0, 16
	vcdlg	%v0, %v0, 0, -1, 0
	vcdlg	%v0, %v0, 0, 16, 0
	vcdlg	%v0, %v0, -1, 0, 0
	vcdlg	%v0, %v0, 16, 0, 0

#CHECK: error: invalid operand
#CHECK: vcdlgb	%v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vcdlgb	%v0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: vcdlgb	%v0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vcdlgb	%v0, %v0, 16, 0

	vcdlgb	%v0, %v0, 0, -1
	vcdlgb	%v0, %v0, 0, 16
	vcdlgb	%v0, %v0, -1, 0
	vcdlgb	%v0, %v0, 16, 0

#CHECK: error: invalid operand
#CHECK: vcgd	%v0, %v0, 0, 0, -1
#CHECK: error: invalid operand
#CHECK: vcgd	%v0, %v0, 0, 0, 16
#CHECK: error: invalid operand
#CHECK: vcgd	%v0, %v0, 0, -1, 0
#CHECK: error: invalid operand
#CHECK: vcgd	%v0, %v0, 0, 16, 0
#CHECK: error: invalid operand
#CHECK: vcgd	%v0, %v0, -1, 0, 0
#CHECK: error: invalid operand
#CHECK: vcgd	%v0, %v0, 16, 0, 0

	vcgd	%v0, %v0, 0, 0, -1
	vcgd	%v0, %v0, 0, 0, 16
	vcgd	%v0, %v0, 0, -1, 0
	vcgd	%v0, %v0, 0, 16, 0
	vcgd	%v0, %v0, -1, 0, 0
	vcgd	%v0, %v0, 16, 0, 0

#CHECK: error: invalid operand
#CHECK: vcgdb	%v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vcgdb	%v0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: vcgdb	%v0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vcgdb	%v0, %v0, 16, 0

	vcgdb	%v0, %v0, 0, -1
	vcgdb	%v0, %v0, 0, 16
	vcgdb	%v0, %v0, -1, 0
	vcgdb	%v0, %v0, 16, 0

#CHECK: error: invalid operand
#CHECK: vclgd	%v0, %v0, 0, 0, -1
#CHECK: error: invalid operand
#CHECK: vclgd	%v0, %v0, 0, 0, 16
#CHECK: error: invalid operand
#CHECK: vclgd	%v0, %v0, 0, -1, 0
#CHECK: error: invalid operand
#CHECK: vclgd	%v0, %v0, 0, 16, 0
#CHECK: error: invalid operand
#CHECK: vclgd	%v0, %v0, -1, 0, 0
#CHECK: error: invalid operand
#CHECK: vclgd	%v0, %v0, 16, 0, 0

	vclgd	%v0, %v0, 0, 0, -1
	vclgd	%v0, %v0, 0, 0, 16
	vclgd	%v0, %v0, 0, -1, 0
	vclgd	%v0, %v0, 0, 16, 0
	vclgd	%v0, %v0, -1, 0, 0
	vclgd	%v0, %v0, 16, 0, 0

#CHECK: error: invalid operand
#CHECK: vclgdb	%v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vclgdb	%v0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: vclgdb	%v0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vclgdb	%v0, %v0, 16, 0

	vclgdb	%v0, %v0, 0, -1
	vclgdb	%v0, %v0, 0, 16
	vclgdb	%v0, %v0, -1, 0
	vclgdb	%v0, %v0, 16, 0

#CHECK: error: instruction requires: vector-packed-decimal
#CHECK: vcp	%v0, %v0, 0

	vcp	%v0, %v0, 0

#CHECK: vcvb	%r0, %v0, 0

	vcvb	%r0, %v0, 0

#CHECK: error: instruction requires: vector-packed-decimal
#CHECK: vcvbg	%r0, %v0, 0

	vcvbg	%r0, %v0, 0

#CHECK: error: instruction requires: vector-packed-decimal
#CHECK: vcvd	%v0, %r0, 0, 0

	vcvd	%v0, %r0, 0, 0

#CHECK: error: instruction requires: vector-packed-decimal
#CHECK: vcvdg	%v0, %r0, 0, 0

	vcvdg	%v0, %r0, 0, 0

#CHECK: error: instruction requires: vector-packed-decimal
#CHECK: vdp	%v0, %v0, %v0, 0, 0

	vdp	%v0, %v0, %v0, 0, 0

#CHECK: error: invalid operand
#CHECK: verim	%v0, %v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: verim	%v0, %v0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: verim	%v0, %v0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: verim	%v0, %v0, %v0, 256, 0

	verim	%v0, %v0, %v0, 0, -1
	verim	%v0, %v0, %v0, 0, 16
	verim	%v0, %v0, %v0, -1, 0
	verim	%v0, %v0, %v0, 256, 0

#CHECK: error: invalid operand
#CHECK: verimb	%v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: verimb	%v0, %v0, %v0, 256

	verimb	%v0, %v0, %v0, -1
	verimb	%v0, %v0, %v0, 256

#CHECK: error: invalid operand
#CHECK: verimf	%v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: verimf	%v0, %v0, %v0, 256

	verimf	%v0, %v0, %v0, -1
	verimf	%v0, %v0, %v0, 256

#CHECK: error: invalid operand
#CHECK: verimg	%v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: verimg	%v0, %v0, %v0, 256

	verimg	%v0, %v0, %v0, -1
	verimg	%v0, %v0, %v0, 256

#CHECK: error: invalid operand
#CHECK: verimh	%v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: verimh	%v0, %v0, %v0, 256

	verimh	%v0, %v0, %v0, -1
	verimh	%v0, %v0, %v0, 256

#CHECK: error: invalid operand
#CHECK: verll	%v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: verll	%v0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: verll	%v0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: verll	%v0, %v0, 4096, 0

	verll	%v0, %v0, 0, -1
	verll	%v0, %v0, 0, 16
	verll	%v0, %v0, -1, 0
	verll	%v0, %v0, 4096, 0

#CHECK: error: invalid operand
#CHECK: verllb	%v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: verllb	%v0, %v0, 4096

	verllb	%v0, %v0, -1
	verllb	%v0, %v0, 4096

#CHECK: error: invalid operand
#CHECK: verllf	%v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: verllf	%v0, %v0, 4096

	verllf	%v0, %v0, -1
	verllf	%v0, %v0, 4096

#CHECK: error: invalid operand
#CHECK: verllg	%v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: verllg	%v0, %v0, 4096

	verllg	%v0, %v0, -1
	verllg	%v0, %v0, 4096

#CHECK: error: invalid operand
#CHECK: verllh	%v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: verllh	%v0, %v0, 4096

	verllh	%v0, %v0, -1
	verllh	%v0, %v0, 4096

#CHECK: error: invalid operand
#CHECK: vesl	%v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vesl	%v0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: vesl	%v0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vesl	%v0, %v0, 4096, 0

	vesl	%v0, %v0, 0, -1
	vesl	%v0, %v0, 0, 16
	vesl	%v0, %v0, -1, 0
	vesl	%v0, %v0, 4096, 0

#CHECK: error: invalid operand
#CHECK: veslb	%v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: veslb	%v0, %v0, 4096

	veslb	%v0, %v0, -1
	veslb	%v0, %v0, 4096

#CHECK: error: invalid operand
#CHECK: veslf	%v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: veslf	%v0, %v0, 4096

	veslf	%v0, %v0, -1
	veslf	%v0, %v0, 4096

#CHECK: error: invalid operand
#CHECK: veslg	%v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: veslg	%v0, %v0, 4096

	veslg	%v0, %v0, -1
	veslg	%v0, %v0, 4096

#CHECK: error: invalid operand
#CHECK: veslh	%v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: veslh	%v0, %v0, 4096

	veslh	%v0, %v0, -1
	veslh	%v0, %v0, 4096

#CHECK: error: invalid operand
#CHECK: vesra	%v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vesra	%v0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: vesra	%v0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vesra	%v0, %v0, 4096, 0

	vesra	%v0, %v0, 0, -1
	vesra	%v0, %v0, 0, 16
	vesra	%v0, %v0, -1, 0
	vesra	%v0, %v0, 4096, 0

#CHECK: error: invalid operand
#CHECK: vesrab	%v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vesrab	%v0, %v0, 4096

	vesrab	%v0, %v0, -1
	vesrab	%v0, %v0, 4096

#CHECK: error: invalid operand
#CHECK: vesraf	%v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vesraf	%v0, %v0, 4096

	vesraf	%v0, %v0, -1
	vesraf	%v0, %v0, 4096

#CHECK: error: invalid operand
#CHECK: vesrag	%v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vesrag	%v0, %v0, 4096

	vesrag	%v0, %v0, -1
	vesrag	%v0, %v0, 4096

#CHECK: error: invalid operand
#CHECK: vesrah	%v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vesrah	%v0, %v0, 4096

	vesrah	%v0, %v0, -1
	vesrah	%v0, %v0, 4096

#CHECK: error: invalid operand
#CHECK: vesrl	%v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vesrl	%v0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: vesrl	%v0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vesrl	%v0, %v0, 4096, 0

	vesrl	%v0, %v0, 0, -1
	vesrl	%v0, %v0, 0, 16
	vesrl	%v0, %v0, -1, 0
	vesrl	%v0, %v0, 4096, 0

#CHECK: error: invalid operand
#CHECK: vesrlb	%v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vesrlb	%v0, %v0, 4096

	vesrlb	%v0, %v0, -1
	vesrlb	%v0, %v0, 4096

#CHECK: error: invalid operand
#CHECK: vesrlf	%v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vesrlf	%v0, %v0, 4096

	vesrlf	%v0, %v0, -1
	vesrlf	%v0, %v0, 4096

#CHECK: error: invalid operand
#CHECK: vesrlg	%v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vesrlg	%v0, %v0, 4096

	vesrlg	%v0, %v0, -1
	vesrlg	%v0, %v0, 4096

#CHECK: error: invalid operand
#CHECK: vesrlh	%v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vesrlh	%v0, %v0, 4096

	vesrlh	%v0, %v0, -1
	vesrlh	%v0, %v0, 4096

#CHECK: error: invalid operand
#CHECK: vfae	%v0, %v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vfae	%v0, %v0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: vfae	%v0, %v0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vfae	%v0, %v0, %v0, 16, 0
#CHECK: error: too few operands
#CHECK: vfae	%v0, %v0, %v0
#CHECK: error: invalid operand
#CHECK: vfae	%v0, %v0, %v0, 0, 0, 0

	vfae	%v0, %v0, %v0, 0, -1
	vfae	%v0, %v0, %v0, 0, 16
	vfae	%v0, %v0, %v0, -1, 0
	vfae	%v0, %v0, %v0, 16, 0
	vfae	%v0, %v0, %v0
	vfae	%v0, %v0, %v0, 0, 0, 0

#CHECK: error: invalid operand
#CHECK: vfaeb	%v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vfaeb	%v0, %v0, %v0, 16
#CHECK: error: too few operands
#CHECK: vfaeb	%v0, %v0
#CHECK: error: invalid operand
#CHECK: vfaeb	%v0, %v0, %v0, 0, 0

	vfaeb	%v0, %v0, %v0, -1
	vfaeb	%v0, %v0, %v0, 16
	vfaeb	%v0, %v0
	vfaeb	%v0, %v0, %v0, 0, 0

#CHECK: error: invalid operand
#CHECK: vfaebs	%v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vfaebs	%v0, %v0, %v0, 16
#CHECK: error: too few operands
#CHECK: vfaebs	%v0, %v0
#CHECK: error: invalid operand
#CHECK: vfaebs	%v0, %v0, %v0, 0, 0

	vfaebs	%v0, %v0, %v0, -1
	vfaebs	%v0, %v0, %v0, 16
	vfaebs	%v0, %v0
	vfaebs	%v0, %v0, %v0, 0, 0

#CHECK: error: invalid operand
#CHECK: vfaef	%v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vfaef	%v0, %v0, %v0, 16
#CHECK: error: too few operands
#CHECK: vfaef	%v0, %v0
#CHECK: error: invalid operand
#CHECK: vfaef	%v0, %v0, %v0, 0, 0

	vfaef	%v0, %v0, %v0, -1
	vfaef	%v0, %v0, %v0, 16
	vfaef	%v0, %v0
	vfaef	%v0, %v0, %v0, 0, 0

#CHECK: error: invalid operand
#CHECK: vfaefs	%v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vfaefs	%v0, %v0, %v0, 16
#CHECK: error: too few operands
#CHECK: vfaefs	%v0, %v0
#CHECK: error: invalid operand
#CHECK: vfaefs	%v0, %v0, %v0, 0, 0

	vfaefs	%v0, %v0, %v0, -1
	vfaefs	%v0, %v0, %v0, 16
	vfaefs	%v0, %v0
	vfaefs	%v0, %v0, %v0, 0, 0

#CHECK: error: invalid operand
#CHECK: vfaeh	%v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vfaeh	%v0, %v0, %v0, 16
#CHECK: error: too few operands
#CHECK: vfaeh	%v0, %v0
#CHECK: error: invalid operand
#CHECK: vfaeh	%v0, %v0, %v0, 0, 0

	vfaeh	%v0, %v0, %v0, -1
	vfaeh	%v0, %v0, %v0, 16
	vfaeh	%v0, %v0
	vfaeh	%v0, %v0, %v0, 0, 0

#CHECK: error: invalid operand
#CHECK: vfaehs	%v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vfaehs	%v0, %v0, %v0, 16
#CHECK: error: too few operands
#CHECK: vfaehs	%v0, %v0
#CHECK: error: invalid operand
#CHECK: vfaehs	%v0, %v0, %v0, 0, 0

	vfaehs	%v0, %v0, %v0, -1
	vfaehs	%v0, %v0, %v0, 16
	vfaehs	%v0, %v0
	vfaehs	%v0, %v0, %v0, 0, 0

#CHECK: error: invalid operand
#CHECK: vfaezb	%v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vfaezb	%v0, %v0, %v0, 16
#CHECK: error: too few operands
#CHECK: vfaezb	%v0, %v0
#CHECK: error: invalid operand
#CHECK: vfaezb	%v0, %v0, %v0, 0, 0

	vfaezb	%v0, %v0, %v0, -1
	vfaezb	%v0, %v0, %v0, 16
	vfaezb	%v0, %v0
	vfaezb	%v0, %v0, %v0, 0, 0

#CHECK: error: invalid operand
#CHECK: vfaezbs	%v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vfaezbs	%v0, %v0, %v0, 16
#CHECK: error: too few operands
#CHECK: vfaezbs	%v0, %v0
#CHECK: error: invalid operand
#CHECK: vfaezbs	%v0, %v0, %v0, 0, 0

	vfaezbs	%v0, %v0, %v0, -1
	vfaezbs	%v0, %v0, %v0, 16
	vfaezbs	%v0, %v0
	vfaezbs	%v0, %v0, %v0, 0, 0

#CHECK: error: invalid operand
#CHECK: vfaezf	%v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vfaezf	%v0, %v0, %v0, 16
#CHECK: error: too few operands
#CHECK: vfaezf	%v0, %v0
#CHECK: error: invalid operand
#CHECK: vfaezf	%v0, %v0, %v0, 0, 0

	vfaezf	%v0, %v0, %v0, -1
	vfaezf	%v0, %v0, %v0, 16
	vfaezf	%v0, %v0
	vfaezf	%v0, %v0, %v0, 0, 0

#CHECK: error: invalid operand
#CHECK: vfaezfs	%v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vfaezfs	%v0, %v0, %v0, 16
#CHECK: error: too few operands
#CHECK: vfaezfs	%v0, %v0
#CHECK: error: invalid operand
#CHECK: vfaezfs	%v0, %v0, %v0, 0, 0

	vfaezfs	%v0, %v0, %v0, -1
	vfaezfs	%v0, %v0, %v0, 16
	vfaezfs	%v0, %v0
	vfaezfs	%v0, %v0, %v0, 0, 0

#CHECK: error: invalid operand
#CHECK: vfaezh	%v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vfaezh	%v0, %v0, %v0, 16
#CHECK: error: too few operands
#CHECK: vfaezh	%v0, %v0
#CHECK: error: invalid operand
#CHECK: vfaezh	%v0, %v0, %v0, 0, 0

	vfaezh	%v0, %v0, %v0, -1
	vfaezh	%v0, %v0, %v0, 16
	vfaezh	%v0, %v0
	vfaezh	%v0, %v0, %v0, 0, 0

#CHECK: error: invalid operand
#CHECK: vfaezhs	%v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vfaezhs	%v0, %v0, %v0, 16
#CHECK: error: too few operands
#CHECK: vfaezhs	%v0, %v0
#CHECK: error: invalid operand
#CHECK: vfaezhs	%v0, %v0, %v0, 0, 0

	vfaezhs	%v0, %v0, %v0, -1
	vfaezhs	%v0, %v0, %v0, 16
	vfaezhs	%v0, %v0
	vfaezhs	%v0, %v0, %v0, 0, 0

#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: vfasb	%v0, %v0, %v0

	vfasb	%v0, %v0, %v0

#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: vfcesb	%v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: vfcesbs	%v0, %v0, %v0

	vfcesb	%v0, %v0, %v0
	vfcesbs	%v0, %v0, %v0

#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: vfchsb	%v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: vfchsbs	%v0, %v0, %v0

	vfchsb	%v0, %v0, %v0
	vfchsbs	%v0, %v0, %v0

#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: vfchesb	%v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: vfchesbs %v0, %v0, %v0

	vfchesb	%v0, %v0, %v0
	vfchesbs %v0, %v0, %v0

#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: vfdsb	%v0, %v0, %v0

	vfdsb	%v0, %v0, %v0

#CHECK: error: invalid operand
#CHECK: vfee	%v0, %v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vfee	%v0, %v0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: vfee	%v0, %v0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vfee	%v0, %v0, %v0, 16, 0
#CHECK: error: too few operands
#CHECK: vfee	%v0, %v0, %v0
#CHECK: error: invalid operand
#CHECK: vfee	%v0, %v0, %v0, 0, 0, 0

	vfee	%v0, %v0, %v0, 0, -1
	vfee	%v0, %v0, %v0, 0, 16
	vfee	%v0, %v0, %v0, -1, 0
	vfee	%v0, %v0, %v0, 16, 0
	vfee	%v0, %v0, %v0
	vfee	%v0, %v0, %v0, 0, 0, 0

#CHECK: error: invalid operand
#CHECK: vfeeb	%v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vfeeb	%v0, %v0, %v0, 16
#CHECK: error: too few operands
#CHECK: vfeeb	%v0, %v0
#CHECK: error: invalid operand
#CHECK: vfeeb	%v0, %v0, %v0, 0, 0

	vfeeb	%v0, %v0, %v0, -1
	vfeeb	%v0, %v0, %v0, 16
	vfeeb	%v0, %v0
	vfeeb	%v0, %v0, %v0, 0, 0

#CHECK: error: too few operands
#CHECK: vfeebs	%v0, %v0
#CHECK: error: invalid operand
#CHECK: vfeebs	%v0, %v0, %v0, 0

	vfeebs	%v0, %v0
	vfeebs	%v0, %v0, %v0, 0

#CHECK: error: invalid operand
#CHECK: vfeef	%v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vfeef	%v0, %v0, %v0, 16
#CHECK: error: too few operands
#CHECK: vfeef	%v0, %v0
#CHECK: error: invalid operand
#CHECK: vfeef	%v0, %v0, %v0, 0, 0

	vfeef	%v0, %v0, %v0, -1
	vfeef	%v0, %v0, %v0, 16
	vfeef	%v0, %v0
	vfeef	%v0, %v0, %v0, 0, 0

#CHECK: error: too few operands
#CHECK: vfeefs	%v0, %v0
#CHECK: error: invalid operand
#CHECK: vfeefs	%v0, %v0, %v0, 0

	vfeefs	%v0, %v0
	vfeefs	%v0, %v0, %v0, 0

#CHECK: error: invalid operand
#CHECK: vfeeh	%v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vfeeh	%v0, %v0, %v0, 16
#CHECK: error: too few operands
#CHECK: vfeeh	%v0, %v0
#CHECK: error: invalid operand
#CHECK: vfeeh	%v0, %v0, %v0, 0, 0

	vfeeh	%v0, %v0, %v0, -1
	vfeeh	%v0, %v0, %v0, 16
	vfeeh	%v0, %v0
	vfeeh	%v0, %v0, %v0, 0, 0

#CHECK: error: too few operands
#CHECK: vfeehs	%v0, %v0
#CHECK: error: invalid operand
#CHECK: vfeehs	%v0, %v0, %v0, 0

	vfeehs	%v0, %v0
	vfeehs	%v0, %v0, %v0, 0

#CHECK: error: too few operands
#CHECK: vfeezb	%v0, %v0
#CHECK: error: invalid operand
#CHECK: vfeezb	%v0, %v0, %v0, 0

	vfeezb	%v0, %v0
	vfeezb	%v0, %v0, %v0, 0

#CHECK: error: too few operands
#CHECK: vfeezbs	%v0, %v0
#CHECK: error: invalid operand
#CHECK: vfeezbs	%v0, %v0, %v0, 0

	vfeezbs	%v0, %v0
	vfeezbs	%v0, %v0, %v0, 0

#CHECK: error: too few operands
#CHECK: vfeezf	%v0, %v0
#CHECK: error: invalid operand
#CHECK: vfeezf	%v0, %v0, %v0, 0

	vfeezf	%v0, %v0
	vfeezf	%v0, %v0, %v0, 0

#CHECK: error: too few operands
#CHECK: vfeezfs	%v0, %v0
#CHECK: error: invalid operand
#CHECK: vfeezfs	%v0, %v0, %v0, 0

	vfeezfs	%v0, %v0
	vfeezfs	%v0, %v0, %v0, 0

#CHECK: error: too few operands
#CHECK: vfeezh	%v0, %v0
#CHECK: error: invalid operand
#CHECK: vfeezh	%v0, %v0, %v0, 0

	vfeezh	%v0, %v0
	vfeezh	%v0, %v0, %v0, 0

#CHECK: error: too few operands
#CHECK: vfeezhs	%v0, %v0
#CHECK: error: invalid operand
#CHECK: vfeezhs	%v0, %v0, %v0, 0

	vfeezhs	%v0, %v0
	vfeezhs	%v0, %v0, %v0, 0

#CHECK: error: invalid operand
#CHECK: vfene	%v0, %v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vfene	%v0, %v0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: vfene	%v0, %v0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vfene	%v0, %v0, %v0, 16, 0
#CHECK: error: too few operands
#CHECK: vfene	%v0, %v0, %v0
#CHECK: error: invalid operand
#CHECK: vfene	%v0, %v0, %v0, 0, 0, 0

	vfene	%v0, %v0, %v0, 0, -1
	vfene	%v0, %v0, %v0, 0, 16
	vfene	%v0, %v0, %v0, -1, 0
	vfene	%v0, %v0, %v0, 16, 0
	vfene	%v0, %v0, %v0
	vfene	%v0, %v0, %v0, 0, 0, 0

#CHECK: error: invalid operand
#CHECK: vfeneb	%v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vfeneb	%v0, %v0, %v0, 16
#CHECK: error: too few operands
#CHECK: vfeneb	%v0, %v0
#CHECK: error: invalid operand
#CHECK: vfeneb	%v0, %v0, %v0, 0, 0

	vfeneb	%v0, %v0, %v0, -1
	vfeneb	%v0, %v0, %v0, 16
	vfeneb	%v0, %v0
	vfeneb	%v0, %v0, %v0, 0, 0

#CHECK: error: too few operands
#CHECK: vfenebs	%v0, %v0
#CHECK: error: invalid operand
#CHECK: vfenebs	%v0, %v0, %v0, 0

	vfenebs	%v0, %v0
	vfenebs	%v0, %v0, %v0, 0

#CHECK: error: invalid operand
#CHECK: vfenef	%v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vfenef	%v0, %v0, %v0, 16
#CHECK: error: too few operands
#CHECK: vfenef	%v0, %v0
#CHECK: error: invalid operand
#CHECK: vfenef	%v0, %v0, %v0, 0, 0

	vfenef	%v0, %v0, %v0, -1
	vfenef	%v0, %v0, %v0, 16
	vfenef	%v0, %v0
	vfenef	%v0, %v0, %v0, 0, 0

#CHECK: error: too few operands
#CHECK: vfenefs	%v0, %v0
#CHECK: error: invalid operand
#CHECK: vfenefs	%v0, %v0, %v0, 0

	vfenefs	%v0, %v0
	vfenefs	%v0, %v0, %v0, 0

#CHECK: error: invalid operand
#CHECK: vfeneh	%v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vfeneh	%v0, %v0, %v0, 16
#CHECK: error: too few operands
#CHECK: vfeneh	%v0, %v0
#CHECK: error: invalid operand
#CHECK: vfeneh	%v0, %v0, %v0, 0, 0

	vfeneh	%v0, %v0, %v0, -1
	vfeneh	%v0, %v0, %v0, 16
	vfeneh	%v0, %v0
	vfeneh	%v0, %v0, %v0, 0, 0

#CHECK: error: too few operands
#CHECK: vfenehs	%v0, %v0
#CHECK: error: invalid operand
#CHECK: vfenehs	%v0, %v0, %v0, 0

	vfenehs	%v0, %v0
	vfenehs	%v0, %v0, %v0, 0

#CHECK: error: too few operands
#CHECK: vfenezb	%v0, %v0
#CHECK: error: invalid operand
#CHECK: vfenezb	%v0, %v0, %v0, 0

	vfenezb	%v0, %v0
	vfenezb	%v0, %v0, %v0, 0

#CHECK: error: too few operands
#CHECK: vfenezbs %v0, %v0
#CHECK: error: invalid operand
#CHECK: vfenezbs %v0, %v0, %v0, 0

	vfenezbs %v0, %v0
	vfenezbs %v0, %v0, %v0, 0

#CHECK: error: too few operands
#CHECK: vfenezf	%v0, %v0
#CHECK: error: invalid operand
#CHECK: vfenezf	%v0, %v0, %v0, 0

	vfenezf	%v0, %v0
	vfenezf	%v0, %v0, %v0, 0

#CHECK: error: too few operands
#CHECK: vfenezfs %v0, %v0
#CHECK: error: invalid operand
#CHECK: vfenezfs %v0, %v0, %v0, 0

	vfenezfs %v0, %v0
	vfenezfs %v0, %v0, %v0, 0

#CHECK: error: too few operands
#CHECK: vfenezh	%v0, %v0
#CHECK: error: invalid operand
#CHECK: vfenezh	%v0, %v0, %v0, 0

	vfenezh	%v0, %v0
	vfenezh	%v0, %v0, %v0, 0

#CHECK: error: too few operands
#CHECK: vfenezhs %v0, %v0
#CHECK: error: invalid operand
#CHECK: vfenezhs %v0, %v0, %v0, 0

	vfenezhs %v0, %v0
	vfenezhs %v0, %v0, %v0, 0

#CHECK: error: invalid operand
#CHECK: vfi	%v0, %v0, 0, 0, -1
#CHECK: error: invalid operand
#CHECK: vfi	%v0, %v0, 0, 0, 16
#CHECK: error: invalid operand
#CHECK: vfi	%v0, %v0, 0, -1, 0
#CHECK: error: invalid operand
#CHECK: vfi	%v0, %v0, 0, 16, 0
#CHECK: error: invalid operand
#CHECK: vfi	%v0, %v0, -1, 0, 0
#CHECK: error: invalid operand
#CHECK: vfi	%v0, %v0, 16, 0, 0

	vfi	%v0, %v0, 0, 0, -1
	vfi	%v0, %v0, 0, 0, 16
	vfi	%v0, %v0, 0, -1, 0
	vfi	%v0, %v0, 0, 16, 0
	vfi	%v0, %v0, -1, 0, 0
	vfi	%v0, %v0, 16, 0, 0

#CHECK: error: invalid operand
#CHECK: vfidb	%v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vfidb	%v0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: vfidb	%v0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vfidb	%v0, %v0, 16, 0

	vfidb	%v0, %v0, 0, -1
	vfidb	%v0, %v0, 0, 16
	vfidb	%v0, %v0, -1, 0
	vfidb	%v0, %v0, 16, 0

#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: vfisb	%v0, %v0, 0, 0

	vfisb	%v0, %v0, 0, 0

#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: vfkedb	%v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: vfkedbs	%v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: vfkesb	%v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: vfkesbs	%v0, %v0, %v0

	vfkedb	%v0, %v0, %v0
	vfkedbs	%v0, %v0, %v0
	vfkesb	%v0, %v0, %v0
	vfkesbs	%v0, %v0, %v0

#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: vfkhdb	%v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: vfkhdbs	%v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: vfkhsb	%v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: vfkhsbs	%v0, %v0, %v0

	vfkhdb	%v0, %v0, %v0
	vfkhdbs	%v0, %v0, %v0
	vfkhsb	%v0, %v0, %v0
	vfkhsbs	%v0, %v0, %v0

#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: vfkhedb	%v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: vfkhedbs %v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: vfkhesb	%v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: vfkhesbs %v0, %v0, %v0

	vfkhedb	%v0, %v0, %v0
	vfkhedbs %v0, %v0, %v0
	vfkhesb	%v0, %v0, %v0
	vfkhesbs %v0, %v0, %v0

#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: vfpsosb	%v0, %v0, 0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: vflcsb	%v0, %v0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: vflnsb	%v0, %v0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: vflpsb	%v0, %v0

	vfpsosb	%v0, %v0, 0
	vflcsb	%v0, %v0
	vflnsb	%v0, %v0
	vflpsb	%v0, %v0

#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: vfll	%v0, %v0, 0, 0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: vflls	%v0, %v0

	vfll	%v0, %v0, 0, 0
	vflls	%v0, %v0

#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: vflr	%v0, %v0, 0, 0, 0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: vflrd	%v0, %v0, 0, 0

	vflr	%v0, %v0, 0, 0, 0
	vflrd	%v0, %v0, 0, 0

#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: vfmax	%v0, %v0, %v0, 0, 0, 0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: vfmaxdb	%v0, %v0, %v0, 0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: vfmaxsb	%v0, %v0, %v0, 0

	vfmax	%v0, %v0, %v0, 0, 0, 0
	vfmaxdb	%v0, %v0, %v0, 0
	vfmaxsb	%v0, %v0, %v0, 0

#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: vfmin	%v0, %v0, %v0, 0, 0, 0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: vfmindb	%v0, %v0, %v0, 0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: vfminsb	%v0, %v0, %v0, 0

	vfmin	%v0, %v0, %v0, 0, 0, 0
	vfmindb	%v0, %v0, %v0, 0
	vfminsb	%v0, %v0, %v0, 0

#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: vfmasb	%v0, %v0, %v0, %v0

	vfmasb	%v0, %v0, %v0, %v0

#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: vfmsb	%v0, %v0, %v0

	vfmsb	%v0, %v0, %v0

#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: vfmssb	%v0, %v0, %v0, %v0

	vfmssb	%v0, %v0, %v0, %v0

#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: vfnma	%v0, %v0, %v0, %v0, 0, 0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: vfnmadb	%v0, %v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: vfnmasb	%v0, %v0, %v0, %v0

	vfnma	%v0, %v0, %v0, %v0, 0, 0
	vfnmadb	%v0, %v0, %v0, %v0
	vfnmasb	%v0, %v0, %v0, %v0

#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: vfnms	%v0, %v0, %v0, %v0, 0, 0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: vfnmsdb	%v0, %v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: vfnmssb	%v0, %v0, %v0, %v0

	vfnms	%v0, %v0, %v0, %v0, 0, 0
	vfnmsdb	%v0, %v0, %v0, %v0
	vfnmssb	%v0, %v0, %v0, %v0

#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: vfssb	%v0, %v0, %v0

	vfssb	%v0, %v0, %v0

#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: vfsqsb	%v0, %v0

	vfsqsb	%v0, %v0

#CHECK: error: invalid operand
#CHECK: vftci	%v0, %v0, 0, 0, -1
#CHECK: error: invalid operand
#CHECK: vftci	%v0, %v0, 0, 0, 16
#CHECK: error: invalid operand
#CHECK: vftci	%v0, %v0, 0, -1, 0
#CHECK: error: invalid operand
#CHECK: vftci	%v0, %v0, 0, 16, 0
#CHECK: error: invalid operand
#CHECK: vftci	%v0, %v0, -1, 0, 0
#CHECK: error: invalid operand
#CHECK: vftci	%v0, %v0, 4096, 0, 0

	vftci	%v0, %v0, 0, 0, -1
	vftci	%v0, %v0, 0, 0, 16
	vftci	%v0, %v0, 0, -1, 0
	vftci	%v0, %v0, 0, 16, 0
	vftci	%v0, %v0, -1, 0, 0
	vftci	%v0, %v0, 4096, 0, 0

#CHECK: error: invalid operand
#CHECK: vftcidb	%v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vftcidb	%v0, %v0, 4096

	vftcidb	%v0, %v0, -1
	vftcidb	%v0, %v0, 4096

#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: vftcisb	%v0, %v0, 0

	vftcisb	%v0, %v0, 0

#CHECK: error: invalid operand
#CHECK: vgbm	%v0, -1
#CHECK: error: invalid operand
#CHECK: vgbm	%v0, 0x10000

	vgbm	%v0, -1
	vgbm	%v0, 0x10000

#CHECK: error: vector index required
#CHECK: vgef	%v0, 0(%r1), 0
#CHECK: error: vector index required
#CHECK: vgef	%v0, 0(%r2,%r1), 0
#CHECK: error: invalid operand
#CHECK: vgef	%v0, 0(%v0,%r1), -1
#CHECK: error: invalid operand
#CHECK: vgef	%v0, 0(%v0,%r1), 4
#CHECK: error: invalid operand
#CHECK: vgef	%v0, -1(%v0,%r1), 0
#CHECK: error: invalid operand
#CHECK: vgef	%v0, 4096(%v0,%r1), 0

	vgef	%v0, 0(%r1), 0
	vgef	%v0, 0(%r2,%r1), 0
	vgef	%v0, 0(%v0,%r1), -1
	vgef	%v0, 0(%v0,%r1), 4
	vgef	%v0, -1(%v0,%r1), 0
	vgef	%v0, 4096(%v0,%r1), 0

#CHECK: error: vector index required
#CHECK: vgeg	%v0, 0(%r1), 0
#CHECK: error: vector index required
#CHECK: vgeg	%v0, 0(%r2,%r1), 0
#CHECK: error: invalid operand
#CHECK: vgeg	%v0, 0(%v0,%r1), -1
#CHECK: error: invalid operand
#CHECK: vgeg	%v0, 0(%v0,%r1), 2
#CHECK: error: invalid operand
#CHECK: vgeg	%v0, -1(%v0,%r1), 0
#CHECK: error: invalid operand
#CHECK: vgeg	%v0, 4096(%v0,%r1), 0

	vgeg	%v0, 0(%r1), 0
	vgeg	%v0, 0(%r2,%r1), 0
	vgeg	%v0, 0(%v0,%r1), -1
	vgeg	%v0, 0(%v0,%r1), 2
	vgeg	%v0, -1(%v0,%r1), 0
	vgeg	%v0, 4096(%v0,%r1), 0

#CHECK: error: invalid operand
#CHECK: vgm	%v0, 0, 0, -1
#CHECK: error: invalid operand
#CHECK: vgm	%v0, 0, 0, 16
#CHECK: error: invalid operand
#CHECK: vgm	%v0, 0, -1, 0
#CHECK: error: invalid operand
#CHECK: vgm	%v0, 0, 256, 0
#CHECK: error: invalid operand
#CHECK: vgm	%v0, -1, 0, 0
#CHECK: error: invalid operand
#CHECK: vgm	%v0, 256, 0, 0

	vgm	%v0, 0, 0, -1
	vgm	%v0, 0, 0, 16
	vgm	%v0, 0, -1, 0
	vgm	%v0, 0, 256, 0
	vgm	%v0, -1, 0, 0
	vgm	%v0, 256, 0, 0

#CHECK: error: invalid operand
#CHECK: vgmb	%v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vgmb	%v0, 0, 256
#CHECK: error: invalid operand
#CHECK: vgmb	%v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vgmb	%v0, 256, 0

	vgmb	%v0, 0, -1
	vgmb	%v0, 0, 256
	vgmb	%v0, -1, 0
	vgmb	%v0, 256, 0

#CHECK: error: invalid operand
#CHECK: vgmf	%v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vgmf	%v0, 0, 256
#CHECK: error: invalid operand
#CHECK: vgmf	%v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vgmf	%v0, 256, 0

	vgmf	%v0, 0, -1
	vgmf	%v0, 0, 256
	vgmf	%v0, -1, 0
	vgmf	%v0, 256, 0

#CHECK: error: invalid operand
#CHECK: vgmg	%v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vgmg	%v0, 0, 256
#CHECK: error: invalid operand
#CHECK: vgmg	%v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vgmg	%v0, 256, 0

	vgmg	%v0, 0, -1
	vgmg	%v0, 0, 256
	vgmg	%v0, -1, 0
	vgmg	%v0, 256, 0

#CHECK: error: invalid operand
#CHECK: vgmh	%v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vgmh	%v0, 0, 256
#CHECK: error: invalid operand
#CHECK: vgmh	%v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vgmh	%v0, 256, 0

	vgmh	%v0, 0, -1
	vgmh	%v0, 0, 256
	vgmh	%v0, -1, 0
	vgmh	%v0, 256, 0

#CHECK: error: invalid operand
#CHECK: vistr	%v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vistr	%v0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: vistr	%v0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vistr	%v0, %v0, 16, 0
#CHECK: error: too few operands
#CHECK: vistr	%v0, %v0
#CHECK: error: invalid operand
#CHECK: vistr	%v0, %v0, 0, 0, 0

	vistr	%v0, %v0, 0, -1
	vistr	%v0, %v0, 0, 16
	vistr	%v0, %v0, -1, 0
	vistr	%v0, %v0, 16, 0
	vistr	%v0, %v0
	vistr	%v0, %v0, 0, 0, 0

#CHECK: error: invalid operand
#CHECK: vistrb	%v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vistrb	%v0, %v0, 16
#CHECK: error: too few operands
#CHECK: vistrb	%v0
#CHECK: error: invalid operand
#CHECK: vistrb	%v0, %v0, 0, 0

	vistrb	%v0, %v0, -1
	vistrb	%v0, %v0, 16
	vistrb	%v0
	vistrb	%v0, %v0, 0, 0

#CHECK: error: too few operands
#CHECK: vistrbs	%v0
#CHECK: error: invalid operand
#CHECK: vistrbs	%v0, %v0, 0

	vistrbs	%v0
	vistrbs	%v0, %v0, 0

#CHECK: error: invalid operand
#CHECK: vistrf	%v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vistrf	%v0, %v0, 16
#CHECK: error: too few operands
#CHECK: vistrf	%v0
#CHECK: error: invalid operand
#CHECK: vistrf	%v0, %v0, 0, 0

	vistrf	%v0, %v0, -1
	vistrf	%v0, %v0, 16
	vistrf	%v0
	vistrf	%v0, %v0, 0, 0

#CHECK: error: too few operands
#CHECK: vistrfs	%v0
#CHECK: error: invalid operand
#CHECK: vistrfs	%v0, %v0, 0

	vistrfs	%v0
	vistrfs	%v0, %v0, 0

#CHECK: error: invalid operand
#CHECK: vistrh	%v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vistrh	%v0, %v0, 16
#CHECK: error: too few operands
#CHECK: vistrh	%v0
#CHECK: error: invalid operand
#CHECK: vistrh	%v0, %v0, 0, 0

	vistrh	%v0, %v0, -1
	vistrh	%v0, %v0, 16
	vistrh	%v0
	vistrh	%v0, %v0, 0, 0

#CHECK: error: too few operands
#CHECK: vistrhs	%v0
#CHECK: error: invalid operand
#CHECK: vistrhs	%v0, %v0, 0

	vistrhs	%v0
	vistrhs	%v0, %v0, 0

#CHECK: error: invalid operand
#CHECK: vl	%v0, -1
#CHECK: error: invalid operand
#CHECK: vl	%v0, 4096
#CHECK: error: invalid use of vector addressing
#CHECK: vl	%v0, 0(%v1,%r2)
#CHECK: error: invalid operand
#CHECK: vl	%v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vl	%v0, 0, 16

	vl	%v0, -1
	vl	%v0, 4096
	vl	%v0, 0(%v1,%r2)
	vl	%v0, 0, -1
	vl	%v0, 0, 16

#CHECK: error: invalid operand
#CHECK: vlbb	%v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vlbb	%v0, 0, 16
#CHECK: error: invalid operand
#CHECK: vlbb	%v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vlbb	%v0, 4096, 0
#CHECK: error: invalid use of vector addressing
#CHECK: vlbb	%v0, 0(%v1,%r2), 0

	vlbb	%v0, 0, -1
	vlbb	%v0, 0, 16
	vlbb	%v0, -1, 0
	vlbb	%v0, 4096, 0
	vlbb	%v0, 0(%v1,%r2), 0

#CHECK: error: invalid operand
#CHECK: vleb	%v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vleb	%v0, 0, 16
#CHECK: error: invalid operand
#CHECK: vleb	%v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vleb	%v0, 4096, 0
#CHECK: error: invalid use of vector addressing
#CHECK: vleb	%v0, 0(%v1,%r2), 0

	vleb	%v0, 0, -1
	vleb	%v0, 0, 16
	vleb	%v0, -1, 0
	vleb	%v0, 4096, 0
	vleb	%v0, 0(%v1,%r2), 0

#CHECK: error: invalid operand
#CHECK: vled	%v0, %v0, 0, 0, -1
#CHECK: error: invalid operand
#CHECK: vled	%v0, %v0, 0, 0, 16
#CHECK: error: invalid operand
#CHECK: vled	%v0, %v0, 0, -1, 0
#CHECK: error: invalid operand
#CHECK: vled	%v0, %v0, 0, 16, 0
#CHECK: error: invalid operand
#CHECK: vled	%v0, %v0, -1, 0, 0
#CHECK: error: invalid operand
#CHECK: vled	%v0, %v0, 16, 0, 0

	vled	%v0, %v0, 0, 0, -1
	vled	%v0, %v0, 0, 0, 16
	vled	%v0, %v0, 0, -1, 0
	vled	%v0, %v0, 0, 16, 0
	vled	%v0, %v0, -1, 0, 0
	vled	%v0, %v0, 16, 0, 0

#CHECK: error: invalid operand
#CHECK: vledb	%v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vledb	%v0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: vledb	%v0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vledb	%v0, %v0, 16, 0

	vledb	%v0, %v0, 0, -1
	vledb	%v0, %v0, 0, 16
	vledb	%v0, %v0, -1, 0
	vledb	%v0, %v0, 16, 0

#CHECK: error: invalid operand
#CHECK: vlef	%v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vlef	%v0, 0, 4
#CHECK: error: invalid operand
#CHECK: vlef	%v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vlef	%v0, 4096, 0
#CHECK: error: invalid use of vector addressing
#CHECK: vlef	%v0, 0(%v1,%r2), 0

	vlef	%v0, 0, -1
	vlef	%v0, 0, 4
	vlef	%v0, -1, 0
	vlef	%v0, 4096, 0
	vlef	%v0, 0(%v1,%r2), 0

#CHECK: error: invalid operand
#CHECK: vleg	%v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vleg	%v0, 0, 2
#CHECK: error: invalid operand
#CHECK: vleg	%v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vleg	%v0, 4096, 0
#CHECK: error: invalid use of vector addressing
#CHECK: vleg	%v0, 0(%v1,%r2), 0

	vleg	%v0, 0, -1
	vleg	%v0, 0, 2
	vleg	%v0, -1, 0
	vleg	%v0, 4096, 0
	vleg	%v0, 0(%v1,%r2), 0

#CHECK: error: invalid operand
#CHECK: vleh	%v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vleh	%v0, 0, 8
#CHECK: error: invalid operand
#CHECK: vleh	%v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vleh	%v0, 4096, 0
#CHECK: error: invalid use of vector addressing
#CHECK: vleh	%v0, 0(%v1,%r2), 0

	vleh	%v0, 0, -1
	vleh	%v0, 0, 8
	vleh	%v0, -1, 0
	vleh	%v0, 4096, 0
	vleh	%v0, 0(%v1,%r2), 0

#CHECK: error: invalid operand
#CHECK: vleib	%v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vleib	%v0, 0, 16
#CHECK: error: invalid operand
#CHECK: vleib	%v0, -32769, 0
#CHECK: error: invalid operand
#CHECK: vleib	%v0, 32768, 0

	vleib	%v0, 0, -1
	vleib	%v0, 0, 16
	vleib	%v0, -32769, 0
	vleib	%v0, 32768, 0

#CHECK: error: invalid operand
#CHECK: vleif	%v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vleif	%v0, 0, 4
#CHECK: error: invalid operand
#CHECK: vleif	%v0, -32769, 0
#CHECK: error: invalid operand
#CHECK: vleif	%v0, 32768, 0

	vleif	%v0, 0, -1
	vleif	%v0, 0, 4
	vleif	%v0, -32769, 0
	vleif	%v0, 32768, 0

#CHECK: error: invalid operand
#CHECK: vleig	%v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vleig	%v0, 0, 2
#CHECK: error: invalid operand
#CHECK: vleig	%v0, -32769, 0
#CHECK: error: invalid operand
#CHECK: vleig	%v0, 32768, 0

	vleig	%v0, 0, -1
	vleig	%v0, 0, 2
	vleig	%v0, -32769, 0
	vleig	%v0, 32768, 0

#CHECK: error: invalid operand
#CHECK: vleih	%v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vleih	%v0, 0, 8
#CHECK: error: invalid operand
#CHECK: vleih	%v0, -32769, 0
#CHECK: error: invalid operand
#CHECK: vleih	%v0, 32768, 0

	vleih	%v0, 0, -1
	vleih	%v0, 0, 8
	vleih	%v0, -32769, 0
	vleih	%v0, 32768, 0

#CHECK: error: invalid operand
#CHECK: vlgv	%r0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vlgv	%r0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: vlgv	%r0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vlgv	%r0, %v0, 4096, 0

	vlgv	%r0, %v0, 0, -1
	vlgv	%r0, %v0, 0, 16
	vlgv	%r0, %v0, -1, 0
	vlgv	%r0, %v0, 4096, 0

#CHECK: error: invalid operand
#CHECK: vlgvb	%r0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vlgvb	%r0, %v0, 4096

	vlgvb	%r0, %v0, -1
	vlgvb	%r0, %v0, 4096

#CHECK: error: invalid operand
#CHECK: vlgvf	%r0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vlgvf	%r0, %v0, 4096

	vlgvf	%r0, %v0, -1
	vlgvf	%r0, %v0, 4096

#CHECK: error: invalid operand
#CHECK: vlgvg	%r0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vlgvg	%r0, %v0, 4096

	vlgvg	%r0, %v0, -1
	vlgvg	%r0, %v0, 4096

#CHECK: error: invalid operand
#CHECK: vlgvh	%r0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vlgvh	%r0, %v0, 4096

	vlgvh	%r0, %v0, -1
	vlgvh	%r0, %v0, 4096

#CHECK: error: instruction requires: vector-packed-decimal
#CHECK: vlip	%v0, 0, 0

	vlip	%v0, 0, 0

#CHECK: error: invalid operand
#CHECK: vll	%v0, %r0, -1
#CHECK: error: invalid operand
#CHECK: vll	%v0, %r0, 4096

	vll	%v0, %r0, -1
	vll	%v0, %r0, 4096

#CHECK: error: invalid operand
#CHECK: vllez	%v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vllez	%v0, 0, 16
#CHECK: error: invalid operand
#CHECK: vllez	%v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vllez	%v0, 4096, 0
#CHECK: error: invalid use of vector addressing
#CHECK: vllez	%v0, 0(%v1,%r2), 0

	vllez	%v0, 0, -1
	vllez	%v0, 0, 16
	vllez	%v0, -1, 0
	vllez	%v0, 4096, 0
	vllez	%v0, 0(%v1,%r2), 0

#CHECK: error: invalid operand
#CHECK: vllezb	%v0, -1
#CHECK: error: invalid operand
#CHECK: vllezb	%v0, 4096
#CHECK: error: invalid use of vector addressing
#CHECK: vllezb	%v0, 0(%v1,%r2)

	vllezb	%v0, -1
	vllezb	%v0, 4096
	vllezb	%v0, 0(%v1,%r2)

#CHECK: error: invalid operand
#CHECK: vllezf	%v0, -1
#CHECK: error: invalid operand
#CHECK: vllezf	%v0, 4096
#CHECK: error: invalid use of vector addressing
#CHECK: vllezf	%v0, 0(%v1,%r2)

	vllezf	%v0, -1
	vllezf	%v0, 4096
	vllezf	%v0, 0(%v1,%r2)

#CHECK: error: invalid operand
#CHECK: vllezg	%v0, -1
#CHECK: error: invalid operand
#CHECK: vllezg	%v0, 4096
#CHECK: error: invalid use of vector addressing
#CHECK: vllezg	%v0, 0(%v1,%r2)

	vllezg	%v0, -1
	vllezg	%v0, 4096
	vllezg	%v0, 0(%v1,%r2)

#CHECK: error: invalid operand
#CHECK: vllezh	%v0, -1
#CHECK: error: invalid operand
#CHECK: vllezh	%v0, 4096
#CHECK: error: invalid use of vector addressing
#CHECK: vllezh	%v0, 0(%v1,%r2)

	vllezh	%v0, -1
	vllezh	%v0, 4096
	vllezh	%v0, 0(%v1,%r2)

#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: vllezlf	%v0, 0

	vllezlf	%v0, 0

#CHECK: error: invalid operand
#CHECK: vlm	%v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vlm	%v0, %v0, 4096
#CHECK: error: invalid operand
#CHECK: vlm	%v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vlm	%v0, %v0, 0, 16

	vlm	%v0, %v0, -1
	vlm	%v0, %v0, 4096
	vlm	%v0, %v0, 0, -1
	vlm	%v0, %v0, 0, 16

#CHECK: error: invalid operand
#CHECK: vlrep	%v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vlrep	%v0, 0, 16
#CHECK: error: invalid operand
#CHECK: vlrep	%v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vlrep	%v0, 4096, 0
#CHECK: error: invalid use of vector addressing
#CHECK: vlrep	%v0, 0(%v1,%r2), 0

	vlrep	%v0, 0, -1
	vlrep	%v0, 0, 16
	vlrep	%v0, -1, 0
	vlrep	%v0, 4096, 0
	vlrep	%v0, 0(%v1,%r2), 0

#CHECK: error: invalid operand
#CHECK: vlrepb	%v0, -1
#CHECK: error: invalid operand
#CHECK: vlrepb	%v0, 4096
#CHECK: error: invalid use of vector addressing
#CHECK: vlrepb	%v0, 0(%v1,%r2)

	vlrepb	%v0, -1
	vlrepb	%v0, 4096
	vlrepb	%v0, 0(%v1,%r2)

#CHECK: error: invalid operand
#CHECK: vlrepf	%v0, -1
#CHECK: error: invalid operand
#CHECK: vlrepf	%v0, 4096
#CHECK: error: invalid use of vector addressing
#CHECK: vlrepf	%v0, 0(%v1,%r2)

	vlrepf	%v0, -1
	vlrepf	%v0, 4096
	vlrepf	%v0, 0(%v1,%r2)

#CHECK: error: invalid operand
#CHECK: vlrepg	%v0, -1
#CHECK: error: invalid operand
#CHECK: vlrepg	%v0, 4096
#CHECK: error: invalid use of vector addressing
#CHECK: vlrepg	%v0, 0(%v1,%r2)

	vlrepg	%v0, -1
	vlrepg	%v0, 4096
	vlrepg	%v0, 0(%v1,%r2)

#CHECK: error: invalid operand
#CHECK: vlreph	%v0, -1
#CHECK: error: invalid operand
#CHECK: vlreph	%v0, 4096
#CHECK: error: invalid use of vector addressing
#CHECK: vlreph	%v0, 0(%v1,%r2)

	vlreph	%v0, -1
	vlreph	%v0, 4096
	vlreph	%v0, 0(%v1,%r2)

#CHECK: error: instruction requires: vector-packed-decimal
#CHECK: vlrl	%v0, 0, 0

	vlrl	%v0, 0, 0

#CHECK: error: instruction requires: vector-packed-decimal
#CHECK: vlrlr	%v0, %r0, 0

	vlrlr	%v0, %r0, 0

#CHECK: error: invalid operand
#CHECK: vlvg	%v0, %r0, 0, -1
#CHECK: error: invalid operand
#CHECK: vlvg	%v0, %r0, 0, 16
#CHECK: error: invalid operand
#CHECK: vlvg	%v0, %r0, -1, 0
#CHECK: error: invalid operand
#CHECK: vlvg	%v0, %r0, 4096, 0

	vlvg	%v0, %r0, 0, -1
	vlvg	%v0, %r0, 0, 16
	vlvg	%v0, %r0, -1, 0
	vlvg	%v0, %r0, 4096, 0

#CHECK: error: invalid operand
#CHECK: vlvgb	%v0, %r0, -1
#CHECK: error: invalid operand
#CHECK: vlvgb	%v0, %r0, 4096

	vlvgb	%v0, %r0, -1
	vlvgb	%v0, %r0, 4096

#CHECK: error: invalid operand
#CHECK: vlvgf	%v0, %r0, -1
#CHECK: error: invalid operand
#CHECK: vlvgf	%v0, %r0, 4096

	vlvgf	%v0, %r0, -1
	vlvgf	%v0, %r0, 4096

#CHECK: error: invalid operand
#CHECK: vlvgg	%v0, %r0, -1
#CHECK: error: invalid operand
#CHECK: vlvgg	%v0, %r0, 4096

	vlvgg	%v0, %r0, -1
	vlvgg	%v0, %r0, 4096

#CHECK: error: invalid operand
#CHECK: vlvgh	%v0, %r0, -1
#CHECK: error: invalid operand
#CHECK: vlvgh	%v0, %r0, 4096

	vlvgh	%v0, %r0, -1
	vlvgh	%v0, %r0, 4096

#CHECK: error: instruction requires: vector-packed-decimal
#CHECK: vmp	%v0, %v0, %v0, 0, 0

	vmp	%v0, %v0, %v0, 0, 0

#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: vmsl	%v0, %v0, %v0, %v0, 0, 0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: vmslg	%v0, %v0, %v0, %v0, 0

	vmsl	%v0, %v0, %v0, %v0, 0, 0
	vmslg	%v0, %v0, %v0, %v0, 0

#CHECK: error: instruction requires: vector-packed-decimal
#CHECK: vmsp	%v0, %v0, %v0, 0, 0

	vmsp	%v0, %v0, %v0, 0, 0

#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: vnn	%v0, %v0, %v0

	vnn	%v0, %v0, %v0

#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: vnx	%v0, %v0, %v0

	vnx	%v0, %v0, %v0

#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: voc	%v0, %v0, %v0

	voc	%v0, %v0, %v0

#CHECK: error: invalid operand
#CHECK: vpdi	%v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vpdi	%v0, %v0, %v0, 16

	vpdi	%v0, %v0, %v0, -1
	vpdi	%v0, %v0, %v0, 16

#CHECK: error: instruction requires: vector-packed-decimal
#CHECK: vpkz	%v0, 0, 0

	vpkz	%v0, 0, 0

#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: vpopctb	%v0, %v0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: vpopctf	%v0, %v0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: vpopctg	%v0, %v0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: vpopcth	%v0, %v0

	vpopctb	%v0, %v0
	vpopctf	%v0, %v0
	vpopctg	%v0, %v0
	vpopcth	%v0, %v0

#CHECK: error: instruction requires: vector-packed-decimal
#CHECK: vpsop	%v0, %v0, 0, 0, 0

	vpsop	%v0, %v0, 0, 0, 0

#CHECK: error: invalid operand
#CHECK: vrep	%v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vrep	%v0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: vrep	%v0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vrep	%v0, %v0, 65536, 0

	vrep	%v0, %v0, 0, -1
	vrep	%v0, %v0, 0, 16
	vrep	%v0, %v0, -1, 0
	vrep	%v0, %v0, 65536, 0

#CHECK: error: invalid operand
#CHECK: vrepb	%v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vrepb	%v0, %v0, 65536

	vrepb	%v0, %v0, -1
	vrepb	%v0, %v0, 65536

#CHECK: error: invalid operand
#CHECK: vrepf	%v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vrepf	%v0, %v0, 65536

	vrepf	%v0, %v0, -1
	vrepf	%v0, %v0, 65536

#CHECK: error: invalid operand
#CHECK: vrepg	%v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vrepg	%v0, %v0, 65536

	vrepg	%v0, %v0, -1
	vrepg	%v0, %v0, 65536

#CHECK: error: invalid operand
#CHECK: vreph	%v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vreph	%v0, %v0, 65536

	vreph	%v0, %v0, -1
	vreph	%v0, %v0, 65536

#CHECK: error: invalid operand
#CHECK: vrepi	%v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vrepi	%v0, 0, 16
#CHECK: error: invalid operand
#CHECK: vrepi	%v0, -32769, 0
#CHECK: error: invalid operand
#CHECK: vrepi	%v0, 32768, 0

	vrepi	%v0, 0, -1
	vrepi	%v0, 0, 16
	vrepi	%v0, -32769, 0
	vrepi	%v0, 32768, 0

#CHECK: error: invalid operand
#CHECK: vrepib	%v0, -32769
#CHECK: error: invalid operand
#CHECK: vrepib	%v0, 32768

	vrepib	%v0, -32769
	vrepib	%v0, 32768

#CHECK: error: invalid operand
#CHECK: vrepif	%v0, -32769
#CHECK: error: invalid operand
#CHECK: vrepif	%v0, 32768

	vrepif	%v0, -32769
	vrepif	%v0, 32768

#CHECK: error: invalid operand
#CHECK: vrepig	%v0, -32769
#CHECK: error: invalid operand
#CHECK: vrepig	%v0, 32768

	vrepig	%v0, -32769
	vrepig	%v0, 32768

#CHECK: error: invalid operand
#CHECK: vrepih	%v0, -32769
#CHECK: error: invalid operand
#CHECK: vrepih	%v0, 32768

	vrepih	%v0, -32769
	vrepih	%v0, 32768

#CHECK: error: instruction requires: vector-packed-decimal
#CHECK: vrp	%v0, %v0, %v0, 0, 0

	vrp	%v0, %v0, %v0, 0, 0

#CHECK: error: vector index required
#CHECK: vscef	%v0, 0(%r1), 0
#CHECK: error: vector index required
#CHECK: vscef	%v0, 0(%r2,%r1), 0
#CHECK: error: invalid operand
#CHECK: vscef	%v0, 0(%v0,%r1), -1
#CHECK: error: invalid operand
#CHECK: vscef	%v0, 0(%v0,%r1), 4
#CHECK: error: invalid operand
#CHECK: vscef	%v0, -1(%v0,%r1), 0
#CHECK: error: invalid operand
#CHECK: vscef	%v0, 4096(%v0,%r1), 0

	vscef	%v0, 0(%r1), 0
	vscef	%v0, 0(%r2,%r1), 0
	vscef	%v0, 0(%v0,%r1), -1
	vscef	%v0, 0(%v0,%r1), 4
	vscef	%v0, -1(%v0,%r1), 0
	vscef	%v0, 4096(%v0,%r1), 0

#CHECK: error: vector index required
#CHECK: vsceg	%v0, 0(%r1), 0
#CHECK: error: vector index required
#CHECK: vsceg	%v0, 0(%r2,%r1), 0
#CHECK: error: invalid operand
#CHECK: vsceg	%v0, 0(%v0,%r1), -1
#CHECK: error: invalid operand
#CHECK: vsceg	%v0, 0(%v0,%r1), 2
#CHECK: error: invalid operand
#CHECK: vsceg	%v0, -1(%v0,%r1), 0
#CHECK: error: invalid operand
#CHECK: vsceg	%v0, 4096(%v0,%r1), 0

	vsceg	%v0, 0(%r1), 0
	vsceg	%v0, 0(%r2,%r1), 0
	vsceg	%v0, 0(%v0,%r1), -1
	vsceg	%v0, 0(%v0,%r1), 2
	vsceg	%v0, -1(%v0,%r1), 0
	vsceg	%v0, 4096(%v0,%r1), 0

#CHECK: error: instruction requires: vector-packed-decimal
#CHECK: vsdp	%v0, %v0, %v0, 0, 0

	vsdp	%v0, %v0, %v0, 0, 0

#CHECK: error: invalid operand
#CHECK: vsldb	%v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vsldb	%v0, %v0, %v0, 256

	vsldb	%v0, %v0, %v0, -1
	vsldb	%v0, %v0, %v0, 256

#CHECK: error: instruction requires: vector-packed-decimal
#CHECK: vsp	%v0, %v0, %v0, 0, 0

	vsp	%v0, %v0, %v0, 0, 0

#CHECK: error: instruction requires: vector-packed-decimal
#CHECK: vsrp	%v0, %v0, 0, 0, 0

	vsrp	%v0, %v0, 0, 0, 0

#CHECK: error: invalid operand
#CHECK: vst	%v0, -1
#CHECK: error: invalid operand
#CHECK: vst	%v0, 4096
#CHECK: error: invalid use of vector addressing
#CHECK: vst	%v0, 0(%v1,%r2)
#CHECK: error: invalid operand
#CHECK: vst	%v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vst	%v0, 0, 16

	vst	%v0, -1
	vst	%v0, 4096
	vst	%v0, 0(%v1,%r2)
	vst	%v0, 0, -1
	vst	%v0, 0, 16

#CHECK: error: invalid operand
#CHECK: vsteb	%v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vsteb	%v0, 0, 16
#CHECK: error: invalid operand
#CHECK: vsteb	%v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vsteb	%v0, 4096, 0
#CHECK: error: invalid use of vector addressing
#CHECK: vsteb	%v0, 0(%v1,%r2), 0

	vsteb	%v0, 0, -1
	vsteb	%v0, 0, 16
	vsteb	%v0, -1, 0
	vsteb	%v0, 4096, 0
	vsteb	%v0, 0(%v1,%r2), 0

#CHECK: error: invalid operand
#CHECK: vstef	%v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vstef	%v0, 0, 4
#CHECK: error: invalid operand
#CHECK: vstef	%v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vstef	%v0, 4096, 0
#CHECK: error: invalid use of vector addressing
#CHECK: vstef	%v0, 0(%v1,%r2), 0

	vstef	%v0, 0, -1
	vstef	%v0, 0, 4
	vstef	%v0, -1, 0
	vstef	%v0, 4096, 0
	vstef	%v0, 0(%v1,%r2), 0

#CHECK: error: invalid operand
#CHECK: vsteg	%v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vsteg	%v0, 0, 2
#CHECK: error: invalid operand
#CHECK: vsteg	%v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vsteg	%v0, 4096, 0
#CHECK: error: invalid use of vector addressing
#CHECK: vsteg	%v0, 0(%v1,%r2), 0

	vsteg	%v0, 0, -1
	vsteg	%v0, 0, 2
	vsteg	%v0, -1, 0
	vsteg	%v0, 4096, 0
	vsteg	%v0, 0(%v1,%r2), 0

#CHECK: error: invalid operand
#CHECK: vsteh	%v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vsteh	%v0, 0, 8
#CHECK: error: invalid operand
#CHECK: vsteh	%v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vsteh	%v0, 4096, 0
#CHECK: error: invalid use of vector addressing
#CHECK: vsteh	%v0, 0(%v1,%r2), 0

	vsteh	%v0, 0, -1
	vsteh	%v0, 0, 8
	vsteh	%v0, -1, 0
	vsteh	%v0, 4096, 0
	vsteh	%v0, 0(%v1,%r2), 0

#CHECK: error: invalid operand
#CHECK: vstl	%v0, %r0, -1
#CHECK: error: invalid operand
#CHECK: vstl	%v0, %r0, 4096

	vstl	%v0, %r0, -1
	vstl	%v0, %r0, 4096

#CHECK: error: invalid operand
#CHECK: vstm	%v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vstm	%v0, %v0, 4096
#CHECK: error: invalid operand
#CHECK: vstm	%v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vstm	%v0, %v0, 0, 16

	vstm	%v0, %v0, -1
	vstm	%v0, %v0, 4096
	vstm	%v0, %v0, 0, -1
	vstm	%v0, %v0, 0, 16

#CHECK: error: invalid operand
#CHECK: vstrc    %v0, %v0, %v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vstrc    %v0, %v0, %v0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: vstrc    %v0, %v0, %v0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vstrc    %v0, %v0, %v0, %v0, 16, 0
#CHECK: error: too few operands
#CHECK: vstrc    %v0, %v0, %v0, %v0
#CHECK: error: invalid operand
#CHECK: vstrc    %v0, %v0, %v0, %v0, 0, 0, 0

	vstrc    %v0, %v0, %v0, %v0, 0, -1
	vstrc    %v0, %v0, %v0, %v0, 0, 16
	vstrc    %v0, %v0, %v0, %v0, -1, 0
	vstrc    %v0, %v0, %v0, %v0, 16, 0
	vstrc    %v0, %v0, %v0, %v0
	vstrc    %v0, %v0, %v0, %v0, 0, 0, 0

#CHECK: error: invalid operand
#CHECK: vstrcb   %v0, %v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vstrcb   %v0, %v0, %v0, %v0, 16
#CHECK: error: too few operands
#CHECK: vstrcb   %v0, %v0, %v0
#CHECK: error: invalid operand
#CHECK: vstrcb   %v0, %v0, %v0, %v0, 0, 0

	vstrcb   %v0, %v0, %v0, %v0, -1
	vstrcb   %v0, %v0, %v0, %v0, 16
	vstrcb   %v0, %v0, %v0
	vstrcb   %v0, %v0, %v0, %v0, 0, 0

#CHECK: error: invalid operand
#CHECK: vstrcbs  %v0, %v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vstrcbs  %v0, %v0, %v0, %v0, 16
#CHECK: error: too few operands
#CHECK: vstrcbs  %v0, %v0, %v0
#CHECK: error: invalid operand
#CHECK: vstrcbs  %v0, %v0, %v0, %v0, 0, 0

	vstrcbs  %v0, %v0, %v0, %v0, -1
	vstrcbs  %v0, %v0, %v0, %v0, 16
	vstrcbs  %v0, %v0, %v0
	vstrcbs  %v0, %v0, %v0, %v0, 0, 0

#CHECK: error: invalid operand
#CHECK: vstrcf   %v0, %v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vstrcf   %v0, %v0, %v0, %v0, 16
#CHECK: error: too few operands
#CHECK: vstrcf   %v0, %v0, %v0
#CHECK: error: invalid operand
#CHECK: vstrcf   %v0, %v0, %v0, %v0, 0, 0

	vstrcf   %v0, %v0, %v0, %v0, -1
	vstrcf   %v0, %v0, %v0, %v0, 16
	vstrcf   %v0, %v0, %v0
	vstrcf   %v0, %v0, %v0, %v0, 0, 0

#CHECK: error: invalid operand
#CHECK: vstrcfs  %v0, %v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vstrcfs  %v0, %v0, %v0, %v0, 16
#CHECK: error: too few operands
#CHECK: vstrcfs  %v0, %v0, %v0
#CHECK: error: invalid operand
#CHECK: vstrcfs  %v0, %v0, %v0, %v0, 0, 0

	vstrcfs  %v0, %v0, %v0, %v0, -1
	vstrcfs  %v0, %v0, %v0, %v0, 16
	vstrcfs  %v0, %v0, %v0
	vstrcfs  %v0, %v0, %v0, %v0, 0, 0

#CHECK: error: invalid operand
#CHECK: vstrch   %v0, %v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vstrch   %v0, %v0, %v0, %v0, 16
#CHECK: error: too few operands
#CHECK: vstrch   %v0, %v0, %v0
#CHECK: error: invalid operand
#CHECK: vstrch   %v0, %v0, %v0, %v0, 0, 0

	vstrch   %v0, %v0, %v0, %v0, -1
	vstrch   %v0, %v0, %v0, %v0, 16
	vstrch   %v0, %v0, %v0
	vstrch   %v0, %v0, %v0, %v0, 0, 0

#CHECK: error: invalid operand
#CHECK: vstrchs  %v0, %v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vstrchs  %v0, %v0, %v0, %v0, 16
#CHECK: error: too few operands
#CHECK: vstrchs  %v0, %v0, %v0
#CHECK: error: invalid operand
#CHECK: vstrchs  %v0, %v0, %v0, %v0, 0, 0

	vstrchs  %v0, %v0, %v0, %v0, -1
	vstrchs  %v0, %v0, %v0, %v0, 16
	vstrchs  %v0, %v0, %v0
	vstrchs  %v0, %v0, %v0, %v0, 0, 0

#CHECK: error: invalid operand
#CHECK: vstrczb  %v0, %v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vstrczb  %v0, %v0, %v0, %v0, 16
#CHECK: error: too few operands
#CHECK: vstrczb  %v0, %v0, %v0
#CHECK: error: invalid operand
#CHECK: vstrczb  %v0, %v0, %v0, %v0, 0, 0

	vstrczb  %v0, %v0, %v0, %v0, -1
	vstrczb  %v0, %v0, %v0, %v0, 16
	vstrczb  %v0, %v0, %v0
	vstrczb  %v0, %v0, %v0, %v0, 0, 0

#CHECK: error: invalid operand
#CHECK: vstrczbs %v0, %v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vstrczbs %v0, %v0, %v0, %v0, 16
#CHECK: error: too few operands
#CHECK: vstrczbs %v0, %v0, %v0
#CHECK: error: invalid operand
#CHECK: vstrczbs %v0, %v0, %v0, %v0, 0, 0

	vstrczbs %v0, %v0, %v0, %v0, -1
	vstrczbs %v0, %v0, %v0, %v0, 16
	vstrczbs %v0, %v0, %v0
	vstrczbs %v0, %v0, %v0, %v0, 0, 0

#CHECK: error: invalid operand
#CHECK: vstrczf  %v0, %v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vstrczf  %v0, %v0, %v0, %v0, 16
#CHECK: error: too few operands
#CHECK: vstrczf  %v0, %v0, %v0
#CHECK: error: invalid operand
#CHECK: vstrczf  %v0, %v0, %v0, %v0, 0, 0

	vstrczf  %v0, %v0, %v0, %v0, -1
	vstrczf  %v0, %v0, %v0, %v0, 16
	vstrczf  %v0, %v0, %v0
	vstrczf  %v0, %v0, %v0, %v0, 0, 0

#CHECK: error: invalid operand
#CHECK: vstrczfs %v0, %v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vstrczfs %v0, %v0, %v0, %v0, 16
#CHECK: error: too few operands
#CHECK: vstrczfs %v0, %v0, %v0
#CHECK: error: invalid operand
#CHECK: vstrczfs %v0, %v0, %v0, %v0, 0, 0

	vstrczfs %v0, %v0, %v0, %v0, -1
	vstrczfs %v0, %v0, %v0, %v0, 16
	vstrczfs %v0, %v0, %v0
	vstrczfs %v0, %v0, %v0, %v0, 0, 0

#CHECK: error: invalid operand
#CHECK: vstrczh  %v0, %v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vstrczh  %v0, %v0, %v0, %v0, 16
#CHECK: error: too few operands
#CHECK: vstrczh  %v0, %v0, %v0
#CHECK: error: invalid operand
#CHECK: vstrczh  %v0, %v0, %v0, %v0, 0, 0

	vstrczh  %v0, %v0, %v0, %v0, -1
	vstrczh  %v0, %v0, %v0, %v0, 16
	vstrczh  %v0, %v0, %v0
	vstrczh  %v0, %v0, %v0, %v0, 0, 0

#CHECK: error: invalid operand
#CHECK: vstrczhs %v0, %v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vstrczhs %v0, %v0, %v0, %v0, 16
#CHECK: error: too few operands
#CHECK: vstrczhs %v0, %v0, %v0
#CHECK: error: invalid operand
#CHECK: vstrczhs %v0, %v0, %v0, %v0, 0, 0

	vstrczhs %v0, %v0, %v0, %v0, -1
	vstrczhs %v0, %v0, %v0, %v0, 16
	vstrczhs %v0, %v0, %v0
	vstrczhs %v0, %v0, %v0, %v0, 0, 0

#CHECK: error: instruction requires: vector-packed-decimal
#CHECK: vstrl	%v0, 0, 0

	vstrl	%v0, 0, 0

#CHECK: error: instruction requires: vector-packed-decimal
#CHECK: vstrlr	%v0, %r0, 0

	vstrlr	%v0, %r0, 0

#CHECK: error: instruction requires: vector-packed-decimal
#CHECK: vtp	%v0

	vtp	%v0

#CHECK: error: instruction requires: vector-packed-decimal
#CHECK: vupkz	%v0, 0, 0

	vupkz	%v0, 0, 0

#CHECK: error: invalid operand
#CHECK: wcdgb	%v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: wcdgb	%v0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: wcdgb	%v0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: wcdgb	%v0, %v0, 16, 0

	wcdgb	%v0, %v0, 0, -1
	wcdgb	%v0, %v0, 0, 16
	wcdgb	%v0, %v0, -1, 0
	wcdgb	%v0, %v0, 16, 0

#CHECK: error: invalid operand
#CHECK: wcdlgb	%v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: wcdlgb	%v0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: wcdlgb	%v0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: wcdlgb	%v0, %v0, 16, 0

	wcdlgb	%v0, %v0, 0, -1
	wcdlgb	%v0, %v0, 0, 16
	wcdlgb	%v0, %v0, -1, 0
	wcdlgb	%v0, %v0, 16, 0

#CHECK: error: invalid operand
#CHECK: wcgdb	%v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: wcgdb	%v0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: wcgdb	%v0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: wcgdb	%v0, %v0, 16, 0

	wcgdb	%v0, %v0, 0, -1
	wcgdb	%v0, %v0, 0, 16
	wcgdb	%v0, %v0, -1, 0
	wcgdb	%v0, %v0, 16, 0

#CHECK: error: invalid operand
#CHECK: wclgdb	%v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: wclgdb	%v0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: wclgdb	%v0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: wclgdb	%v0, %v0, 16, 0

	wclgdb	%v0, %v0, 0, -1
	wclgdb	%v0, %v0, 0, 16
	wclgdb	%v0, %v0, -1, 0
	wclgdb	%v0, %v0, 16, 0

#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfasb	%v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfaxb	%v0, %v0, %v0

	wfasb	%v0, %v0, %v0
	wfaxb	%v0, %v0, %v0

#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfcsb	%v0, %v0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfcxb	%v0, %v0

	wfcsb	%v0, %v0
	wfcxb	%v0, %v0

#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfcesb	%v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfcesbs	%v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfcexb	%v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfcexbs	%v0, %v0, %v0

	wfcesb	%v0, %v0, %v0
	wfcesbs	%v0, %v0, %v0
	wfcexb	%v0, %v0, %v0
	wfcexbs	%v0, %v0, %v0

#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfchsb	%v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfchsbs	%v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfchxb	%v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfchxbs	%v0, %v0, %v0

	wfchsb	%v0, %v0, %v0
	wfchsbs	%v0, %v0, %v0
	wfchxb	%v0, %v0, %v0
	wfchxbs	%v0, %v0, %v0

#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfchesb	%v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfchesbs %v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfchexb	%v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfchexbs %v0, %v0, %v0

	wfchesb	%v0, %v0, %v0
	wfchesbs %v0, %v0, %v0
	wfchexb	%v0, %v0, %v0
	wfchexbs %v0, %v0, %v0

#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfdsb	%v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfdxb	%v0, %v0, %v0

	wfdsb	%v0, %v0, %v0
	wfdxb	%v0, %v0, %v0

#CHECK: error: invalid operand
#CHECK: wfidb	%v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: wfidb	%v0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: wfidb	%v0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: wfidb	%v0, %v0, 16, 0

	wfidb	%v0, %v0, 0, -1
	wfidb	%v0, %v0, 0, 16
	wfidb	%v0, %v0, -1, 0
	wfidb	%v0, %v0, 16, 0

#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfisb	%v0, %v0, 0, 0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfixb	%v0, %v0, 0, 0

	wfisb	%v0, %v0, 0, 0
	wfixb	%v0, %v0, 0, 0

#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfksb	%v0, %v0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfkxb	%v0, %v0

	wfksb	%v0, %v0
	wfkxb	%v0, %v0

#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfkedb	%v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfkedbs	%v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfkesb	%v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfkesbs	%v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfkexb	%v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfkexbs	%v0, %v0, %v0

	wfkedb	%v0, %v0, %v0
	wfkedbs	%v0, %v0, %v0
	wfkesb	%v0, %v0, %v0
	wfkesbs	%v0, %v0, %v0
	wfkexb	%v0, %v0, %v0
	wfkexbs	%v0, %v0, %v0

#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfkhdb	%v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfkhdbs	%v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfkhsb	%v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfkhsbs	%v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfkhxb	%v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfkhxbs	%v0, %v0, %v0

	wfkhdb	%v0, %v0, %v0
	wfkhdbs	%v0, %v0, %v0
	wfkhsb	%v0, %v0, %v0
	wfkhsbs	%v0, %v0, %v0
	wfkhxb	%v0, %v0, %v0
	wfkhxbs	%v0, %v0, %v0

#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfkhedb	%v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfkhedbs %v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfkhesb	%v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfkhesbs %v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfkhexb	%v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfkhexbs %v0, %v0, %v0

	wfkhedb	%v0, %v0, %v0
	wfkhedbs %v0, %v0, %v0
	wfkhesb	%v0, %v0, %v0
	wfkhesbs %v0, %v0, %v0
	wfkhexb	%v0, %v0, %v0
	wfkhexbs %v0, %v0, %v0

#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfpsosb	%v0, %v0, 0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfpsoxb	%v0, %v0, 0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wflcsb	%v0, %v0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wflcxb	%v0, %v0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wflnsb	%v0, %v0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wflnxb	%v0, %v0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wflpsb	%v0, %v0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wflpxb	%v0, %v0

	wfpsosb	%v0, %v0, 0
	wfpsoxb	%v0, %v0, 0
	wflcsb	%v0, %v0
	wflcxb	%v0, %v0
	wflnsb	%v0, %v0
	wflnxb	%v0, %v0
	wflpsb	%v0, %v0
	wflpxb	%v0, %v0

#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wflls	%v0, %v0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wflld	%v0, %v0

	wflls	%v0, %v0
	wflld	%v0, %v0

#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wflrd	%v0, %v0, 0, 0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wflrx	%v0, %v0, 0, 0

	wflrd	%v0, %v0, 0, 0
	wflrx	%v0, %v0, 0, 0

#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfmaxdb	%v0, %v0, %v0, 0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfmaxsb	%v0, %v0, %v0, 0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfmaxxb	%v0, %v0, %v0, 0

	wfmaxdb	%v0, %v0, %v0, 0
	wfmaxsb	%v0, %v0, %v0, 0
	wfmaxxb	%v0, %v0, %v0, 0

#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfmindb	%v0, %v0, %v0, 0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfminsb	%v0, %v0, %v0, 0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfminxb	%v0, %v0, %v0, 0

	wfmindb	%v0, %v0, %v0, 0
	wfminsb	%v0, %v0, %v0, 0
	wfminxb	%v0, %v0, %v0, 0

#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfmasb	%v0, %v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfmaxb	%v0, %v0, %v0, %v0

	wfmasb	%v0, %v0, %v0, %v0
	wfmaxb	%v0, %v0, %v0, %v0

#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfmsb	%v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfmxb	%v0, %v0, %v0

	wfmsb	%v0, %v0, %v0
	wfmxb	%v0, %v0, %v0

#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfmssb	%v0, %v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfmsxb	%v0, %v0, %v0, %v0

	wfmssb	%v0, %v0, %v0, %v0
	wfmsxb	%v0, %v0, %v0, %v0

#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfnmadb	%v0, %v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfnmasb	%v0, %v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfnmaxb	%v0, %v0, %v0, %v0

	wfnmadb	%v0, %v0, %v0, %v0
	wfnmasb	%v0, %v0, %v0, %v0
	wfnmaxb	%v0, %v0, %v0, %v0

#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfnmsdb	%v0, %v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfnmssb	%v0, %v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfnmsxb	%v0, %v0, %v0, %v0

	wfnmsdb	%v0, %v0, %v0, %v0
	wfnmssb	%v0, %v0, %v0, %v0
	wfnmsxb	%v0, %v0, %v0, %v0

#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfssb	%v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfsxb	%v0, %v0, %v0

	wfssb	%v0, %v0, %v0
	wfsxb	%v0, %v0, %v0

#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfsqsb	%v0, %v0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wfsqxb	%v0, %v0

	wfsqsb	%v0, %v0
	wfsqxb	%v0, %v0

#CHECK: error: invalid operand
#CHECK: wftcidb	%v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: wftcidb	%v0, %v0, 4096

	wftcidb	%v0, %v0, -1
	wftcidb	%v0, %v0, 4096

#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wftcisb	%v0, %v0, 0
#CHECK: error: instruction requires: vector-enhancements-1
#CHECK: wftcixb	%v0, %v0, 0

	wftcisb	%v0, %v0, 0
	wftcixb	%v0, %v0, 0

#CHECK: error: invalid operand
#CHECK: wledb	%v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: wledb	%v0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: wledb	%v0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: wledb	%v0, %v0, 16, 0

	wledb	%v0, %v0, 0, -1
	wledb	%v0, %v0, 0, 16
	wledb	%v0, %v0, -1, 0
	wledb	%v0, %v0, 16, 0

