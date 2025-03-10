# For arch15 only.
# RUN: not llvm-mc -triple s390x-linux-gnu -mcpu=arch15 < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid use of indexed addressing
#CHECK: cal	%r2, 160(%r1,%r15), 160(%r15)
#CHECK: error: invalid operand
#CHECK: cal	%r2, -1(%r1), 160(%r15)
#CHECK: error: invalid operand
#CHECK: cal	%r2, 4096(%r1), 160(%r15)
#CHECK: error: invalid operand
#CHECK: cal	%r2, 0(%r1), -1(%r15)
#CHECK: error: invalid operand
#CHECK: cal	%r2, 0(%r1), 4096(%r15)

	cal	%r2, 160(%r1,%r15), 160(%r15)
	cal	%r2, -1(%r1), 160(%r15)
	cal	%r2, 4096(%r1), 160(%r15)
	cal	%r2, 0(%r1), -1(%r15)
	cal	%r2, 0(%r1), 4096(%r15)

#CHECK: error: invalid use of indexed addressing
#CHECK: calg	%r2, 160(%r1,%r15), 160(%r15)
#CHECK: error: invalid operand
#CHECK: calg	%r2, -1(%r1), 160(%r15)
#CHECK: error: invalid operand
#CHECK: calg	%r2, 4096(%r1), 160(%r15)
#CHECK: error: invalid operand
#CHECK: calg	%r2, 0(%r1), -1(%r15)
#CHECK: error: invalid operand
#CHECK: calg	%r2, 0(%r1), 4096(%r15)

	calg	%r2, 160(%r1,%r15), 160(%r15)
	calg	%r2, -1(%r1), 160(%r15)
	calg	%r2, 4096(%r1), 160(%r15)
	calg	%r2, 0(%r1), -1(%r15)
	calg	%r2, 0(%r1), 4096(%r15)

#CHECK: error: invalid use of indexed addressing
#CHECK: calgf	%r2, 160(%r1,%r15), 160(%r15)
#CHECK: error: invalid operand
#CHECK: calgf	%r2, -1(%r1), 160(%r15)
#CHECK: error: invalid operand
#CHECK: calgf	%r2, 4096(%r1), 160(%r15)
#CHECK: error: invalid operand
#CHECK: calgf	%r2, 0(%r1), -1(%r15)
#CHECK: error: invalid operand
#CHECK: calgf	%r2, 0(%r1), 4096(%r15)

	calgf	%r2, 160(%r1,%r15), 160(%r15)
	calgf	%r2, -1(%r1), 160(%r15)
	calgf	%r2, 4096(%r1), 160(%r15)
	calgf	%r2, 0(%r1), -1(%r15)
	calgf	%r2, 0(%r1), 4096(%r15)

#CHECK: error: invalid operand
#CHECK: kimd	%r0, %r0, -1
#CHECK: error: invalid operand
#CHECK: kimd	%r0, %r0, 16

	kimd	%r0, %r0, -1
	kimd	%r0, %r0, 16

#CHECK: error: invalid operand
#CHECK: klmd	%r0, %r0, -1
#CHECK: error: invalid operand
#CHECK: klmd	%r0, %r0, 16

	klmd	%r0, %r0, -1
	klmd	%r0, %r0, 16

#CHECK: error: invalid operand
#CHECK: lxab	%r0, -524289
#CHECK: error: invalid operand
#CHECK: lxab	%r0, 524288

	lxab	%r0, -524289
	lxab	%r0, 524288

#CHECK: error: invalid operand
#CHECK: lxah	%r0, -524289
#CHECK: error: invalid operand
#CHECK: lxah	%r0, 524288

	lxah	%r0, -524289
	lxah	%r0, 524288

#CHECK: error: invalid operand
#CHECK: lxaf	%r0, -524289
#CHECK: error: invalid operand
#CHECK: lxaf	%r0, 524288

	lxaf	%r0, -524289
	lxaf	%r0, 524288

#CHECK: error: invalid operand
#CHECK: lxag	%r0, -524289
#CHECK: error: invalid operand
#CHECK: lxag	%r0, 524288

	lxag	%r0, -524289
	lxag	%r0, 524288

#CHECK: error: invalid operand
#CHECK: lxaq	%r0, -524289
#CHECK: error: invalid operand
#CHECK: lxaq	%r0, 524288

	lxaq	%r0, -524289
	lxaq	%r0, 524288

#CHECK: error: invalid operand
#CHECK: llxab	%r0, -524289
#CHECK: error: invalid operand
#CHECK: llxab	%r0, 524288

	llxab	%r0, -524289
	llxab	%r0, 524288

#CHECK: error: invalid operand
#CHECK: llxah	%r0, -524289
#CHECK: error: invalid operand
#CHECK: llxah	%r0, 524288

	llxah	%r0, -524289
	llxah	%r0, 524288

#CHECK: error: invalid operand
#CHECK: llxaf	%r0, -524289
#CHECK: error: invalid operand
#CHECK: llxaf	%r0, 524288

	llxaf	%r0, -524289
	llxaf	%r0, 524288

#CHECK: error: invalid operand
#CHECK: llxag	%r0, -524289
#CHECK: error: invalid operand
#CHECK: llxag	%r0, 524288

	llxag	%r0, -524289
	llxag	%r0, 524288

#CHECK: error: invalid operand
#CHECK: llxaq	%r0, -524289
#CHECK: error: invalid operand
#CHECK: llxaq	%r0, 524288

	llxaq	%r0, -524289
	llxaq	%r0, 524288

#CHECK: error: invalid operand
#CHECK: pfcr	%r0, %r0, -524289
#CHECK: error: invalid operand
#CHECK: pfcr	%r0, %r0, 524288
#CHECK: error: invalid use of indexed addressing
#CHECK: pfcr	%r0, %r0, 0(%r1,%r2)

	pfcr	%r0, %r0, -524289
	pfcr	%r0, %r0, 524288
	pfcr	%r0, %r0, 0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: vcvbq	%v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vcvbq	%v0, %v0, 16

	vcvbq	%v0, %v0, -1
	vcvbq	%v0, %v0, 16

#CHECK: error: invalid operand
#CHECK: vcvdq	%v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vcvdq	%v0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: vcvdq	%v0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vcvdq	%v0, %v0, 256, 0

	vcvdq	%v0, %v0, 0, -1
	vcvdq	%v0, %v0, 0, 16
	vcvdq	%v0, %v0, -1, 0
	vcvdq	%v0, %v0, 256, 0

#CHECK: error: invalid operand
#CHECK: vd	%v0, %v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vd	%v0, %v0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: vd	%v0, %v0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vd	%v0, %v0, %v0, 16, 0

	vd	%v0, %v0, %v0, 0, -1
	vd	%v0, %v0, %v0, 0, 16
	vd	%v0, %v0, %v0, -1, 0
	vd	%v0, %v0, %v0, 16, 0

#CHECK: error: invalid operand
#CHECK: vdf	%v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vdf	%v0, %v0, %v0, 16

	vdf	%v0, %v0, %v0, -1
	vdf	%v0, %v0, %v0, 16

#CHECK: error: invalid operand
#CHECK: vdg	%v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vdg	%v0, %v0, %v0, 16

	vdg	%v0, %v0, %v0, -1
	vdg	%v0, %v0, %v0, 16

#CHECK: error: invalid operand
#CHECK: vdq	%v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vdq	%v0, %v0, %v0, 16

	vdq	%v0, %v0, %v0, -1
	vdq	%v0, %v0, %v0, 16

#CHECK: error: invalid operand
#CHECK: vdl	%v0, %v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vdl	%v0, %v0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: vdl	%v0, %v0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vdl	%v0, %v0, %v0, 16, 0

	vdl	%v0, %v0, %v0, 0, -1
	vdl	%v0, %v0, %v0, 0, 16
	vdl	%v0, %v0, %v0, -1, 0
	vdl	%v0, %v0, %v0, 16, 0

#CHECK: error: invalid operand
#CHECK: vdlf	%v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vdlf	%v0, %v0, %v0, 16

	vdlf	%v0, %v0, %v0, -1
	vdlf	%v0, %v0, %v0, 16

#CHECK: error: invalid operand
#CHECK: vdlg	%v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vdlg	%v0, %v0, %v0, 16

	vdlg	%v0, %v0, %v0, -1
	vdlg	%v0, %v0, %v0, 16

#CHECK: error: invalid operand
#CHECK: vdlq	%v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vdlq	%v0, %v0, %v0, 16

	vdlq	%v0, %v0, %v0, -1
	vdlq	%v0, %v0, %v0, 16

#CHECK: error: invalid operand
#CHECK: veval	%v0, %v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: veval	%v0, %v0, %v0, %v0, 256

	veval	%v0, %v0, %v0, %v0, -1
	veval	%v0, %v0, %v0, %v0, 256

#CHECK: error: invalid operand
#CHECK: vr	%v0, %v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vr	%v0, %v0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: vr	%v0, %v0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vr	%v0, %v0, %v0, 16, 0

	vr	%v0, %v0, %v0, 0, -1
	vr	%v0, %v0, %v0, 0, 16
	vr	%v0, %v0, %v0, -1, 0
	vr	%v0, %v0, %v0, 16, 0

#CHECK: error: invalid operand
#CHECK: vrf	%v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vrf	%v0, %v0, %v0, 16

	vrf	%v0, %v0, %v0, -1
	vrf	%v0, %v0, %v0, 16

#CHECK: error: invalid operand
#CHECK: vrg	%v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vrg	%v0, %v0, %v0, 16

	vrg	%v0, %v0, %v0, -1
	vrg	%v0, %v0, %v0, 16

#CHECK: error: invalid operand
#CHECK: vrq	%v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vrq	%v0, %v0, %v0, 16

	vrq	%v0, %v0, %v0, -1
	vrq	%v0, %v0, %v0, 16

#CHECK: error: invalid operand
#CHECK: vrl	%v0, %v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vrl	%v0, %v0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: vrl	%v0, %v0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vrl	%v0, %v0, %v0, 16, 0

	vrl	%v0, %v0, %v0, 0, -1
	vrl	%v0, %v0, %v0, 0, 16
	vrl	%v0, %v0, %v0, -1, 0
	vrl	%v0, %v0, %v0, 16, 0

#CHECK: error: invalid operand
#CHECK: vrlf	%v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vrlf	%v0, %v0, %v0, 16

	vrlf	%v0, %v0, %v0, -1
	vrlf	%v0, %v0, %v0, 16

#CHECK: error: invalid operand
#CHECK: vrlg	%v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vrlg	%v0, %v0, %v0, 16

	vrlg	%v0, %v0, %v0, -1
	vrlg	%v0, %v0, %v0, 16

#CHECK: error: invalid operand
#CHECK: vrlq	%v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vrlq	%v0, %v0, %v0, 16

	vrlq	%v0, %v0, %v0, -1
	vrlq	%v0, %v0, %v0, 16

#CHECK: error: invalid operand
#CHECK: vtp	%v0, -1
#CHECK: error: invalid operand
#CHECK: vtp	%v0, 65536

	vtp	%v0, -1
	vtp	%v0, 65536

#CHECK: error: invalid operand
#CHECK: vtz	%v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vtz	%v0, %v0, 65536

	vtz	%v0, %v0, -1
	vtz	%v0, %v0, 65536

