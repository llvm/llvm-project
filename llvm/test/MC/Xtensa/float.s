# RUN: llvm-mc %s -triple=xtensa -mattr=+fp -mattr=+bool -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s

.align	4
LBL0:

# CHECK-INST: abs.s	f2, f3
# CHECK: encoding: [0x10,0x23,0xfa]
	abs.s	f2, f3
# CHECK-INST: add.s	f2, f3, f4
# CHECK: encoding: [0x40,0x23,0x0a]
	add.s	f2, f3, f4
# CHECK-INST: addexp.s	f2, f3
# CHECK: encoding: [0xe0,0x23,0xfa]
	addexp.s	f2, f3
# CHECK-INST: addexpm.s	f2, f3
# CHECK: encoding: [0xf0,0x23,0xfa]
	addexpm.s	f2, f3

# CHECK-INST: ceil.s a2, f3, 5
# CHECK: encoding: [0x50,0x23,0xba]
	ceil.s	a2, f3, 5
# CHECK-INST: const.s	f3, 5
# CHECK: encoding: [0x30,0x35,0xfa]
	const.s	f3, 5

# CHECK-INST: div0.s f2, f3
# CHECK: encoding: [0x70,0x23,0xfa]
	div0.s f2, f3
# CHECK-INST: divn.s f2, f3, f4
# CHECK: encoding: [0x40,0x23,0x7a]
	divn.s f2, f3, f4

# CHECK-INST: float.s	f2, a3, 5
# CHECK: encoding: [0x50,0x23,0xca]
	float.s	f2, a3, 5
# CHECK-INST: floor.s a2, f3, 5
# CHECK: encoding: [0x50,0x23,0xaa]
	floor.s	a2, f3, 5

# CHECK-INST: lsi f2, a3, 8
# CHECK: encoding: [0x23,0x03,0x02]
	lsi f2, a3, 8
# CHECK-INST: lsip f2, a3, 8
# CHECK: encoding: [0x23,0x83,0x02]
	lsip f2, a3, 8
# CHECK-INST: lsx f2, a3, a4
# CHECK: encoding: [0x40,0x23,0x08]
	lsx f2, a3, a4
# CHECK-INST: lsxp f2, a3, a4
# CHECK: encoding: [0x40,0x23,0x18]
	lsxp f2, a3, a4

# CHECK-INST: madd.s f2, f3, f4
# CHECK: encoding: [0x40,0x23,0x4a]
	madd.s f2, f3, f4
# CHECK-INST: maddn.s f2, f3, f4
# CHECK: encoding: [0x40,0x23,0x6a]
	maddn.s f2, f3, f4
# CHECK-INST: mkdadj.s f2, f3
# CHECK: encoding: [0xd0,0x23,0xfa]
	mkdadj.s f2, f3
# CHECK-INST: mksadj.s f2, f3
# CHECK: encoding: [0xc0,0x23,0xfa]
	mksadj.s f2, f3

# CHECK-INST: mov.s f2, f3
# CHECK: encoding: [0x00,0x23,0xfa]
	mov.s f2, f3

# CHECK-INST: moveqz.s f2, f3, a4
# CHECK: encoding: [0x40,0x23,0x8b]
	moveqz.s f2, f3, a4
# CHECK-INST: movf.s f2, f3, b0
# CHECK: encoding: [0x00,0x23,0xcb]
	movf.s f2, f3, b0
# CHECK-INST: movgez.s f2, f3, a4
# CHECK: encoding: [0x40,0x23,0xbb]
	movgez.s f2, f3, a4
# CHECK-INST: movltz.s f2, f3, a4
# CHECK: encoding: [0x40,0x23,0xab]
	movltz.s f2, f3, a4
# CHECK-INST: movnez.s f2, f3, a4
# CHECK: encoding: [0x40,0x23,0x9b]
	movnez.s f2, f3, a4
# CHECK-INST: movt.s f2, f3, b0
# CHECK: encoding: [0x00,0x23,0xdb]
	movt.s f2, f3, b0

# CHECK-INST: msub.s f2, f3, f4
# CHECK: encoding: [0x40,0x23,0x5a]
	msub.s f2, f3, f4
# CHECK-INST: mul.s	f2, f3, f4
# CHECK: encoding: [0x40,0x23,0x2a]
	mul.s	f2, f3, f4
# CHECK-INST: neg.s f2, f3
# CHECK: encoding: [0x60,0x23,0xfa]
	neg.s f2, f3

# CHECK-INST: nexp01.s f2, f3
# CHECK: encoding: [0xb0,0x23,0xfa]
	nexp01.s f2, f3

# CHECK-INST: oeq.s b0, f2, f3
# CHECK: encoding: [0x30,0x02,0x2b]
	oeq.s b0, f2, f3
# CHECK-INST: ole.s b0, f2, f3
# CHECK: encoding: [0x30,0x02,0x6b]
	ole.s b0, f2, f3
# CHECK-INST: olt.s b0, f2, f3
# CHECK: encoding: [0x30,0x02,0x4b]
	olt.s b0, f2, f3

# CHECK-INST: recip0.s f2, f3
# CHECK: encoding: [0x80,0x23,0xfa]
	recip0.s f2, f3

# CHECK-INST: rfr a2, f3
# CHECK: encoding: [0x40,0x23,0xfa]
	rfr a2, f3

# CHECK-INST: round.s a2, f3, 5
# CHECK: encoding: [0x50,0x23,0x8a]
	round.s	a2, f3, 5
# CHECK-INST: rsqrt0.s f2, f3
# CHECK: encoding: [0xa0,0x23,0xfa]
	rsqrt0.s f2, f3
# CHECK-INST: sqrt0.s f2, f3
# CHECK: encoding: [0x90,0x23,0xfa]
	sqrt0.s f2, f3

# CHECK-INST: ssi f2, a3, 8
# CHECK: encoding: [0x23,0x43,0x02]
	ssi f2, a3, 8
# CHECK-INST: ssip f2, a3, 8
# CHECK: encoding: [0x23,0xc3,0x02]
	ssip f2, a3, 8
# CHECK-INST: ssx f2, a3, a4
# CHECK: encoding: [0x40,0x23,0x48]
	ssx f2, a3, a4
# CHECK-INST: ssxp f2, a3, a4
# CHECK: encoding: [0x40,0x23,0x58]
	ssxp f2, a3, a4

# CHECK-INST: sub.s	f2, f3, f4
# CHECK: encoding: [0x40,0x23,0x1a]
	sub.s	f2, f3, f4

# CHECK-INST: trunc.s a2, f3, 5
# CHECK: encoding: [0x50,0x23,0x9a]
	trunc.s	a2, f3, 5

# CHECK-INST: ueq.s b0, f2, f3
# CHECK: encoding: [0x30,0x02,0x3b]
	ueq.s b0, f2, f3

# CHECK-INST: ufloat.s	f2, a3, 5
# CHECK: encoding: [0x50,0x23,0xda]
	ufloat.s	f2, a3, 5

# CHECK-INST: ule.s b0, f2, f3
# CHECK: encoding: [0x30,0x02,0x7b]
	ule.s b0, f2, f3
# CHECK-INST: ult.s b0, f2, f3
# CHECK: encoding: [0x30,0x02,0x5b]
	ult.s b0, f2, f3
# CHECK-INST: un.s b0, f2, f3
# CHECK: encoding: [0x30,0x02,0x1b]
	un.s b0, f2, f3

# CHECK-INST: utrunc.s a2, f3, 5
# CHECK: encoding: [0x50,0x23,0xea]
	utrunc.s	a2, f3, 5

# CHECK-INST: wfr f2, a3
# CHECK: encoding: [0x50,0x23,0xfa]
	wfr f2, a3

# CHECK-INST: rur a3, fcr
# CHECK: encoding: [0x80,0x3e,0xe3]
	rur a3, fcr

# CHECK-INST: rur a3, fcr
# CHECK: encoding: [0x80,0x3e,0xe3]
	rur a3, 232

# CHECK-INST: rur a3, fcr
# CHECK: encoding: [0x80,0x3e,0xe3]
	rur.fcr a3

# CHECK-INST: wur a3, fcr
# CHECK: encoding: [0x30,0xe8,0xf3]
	wur a3, fcr

# CHECK-INST: rur a3, fsr
# CHECK: encoding: [0x90,0x3e,0xe3]
	rur a3, fsr

# CHECK-INST: rur a3, fsr
# CHECK: encoding: [0x90,0x3e,0xe3]
	rur a3, 233

# CHECK-INST: rur a3, fsr
# CHECK: encoding: [0x90,0x3e,0xe3]
	rur.fsr a3

# CHECK-INST: wur a3, fsr
# CHECK: encoding: [0x30,0xe9,0xf3]
	wur a3, fsr
