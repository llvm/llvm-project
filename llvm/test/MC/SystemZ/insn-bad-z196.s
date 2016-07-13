# For z196 only.
# RUN: not llvm-mc -triple s390x-linux-gnu -mcpu=z196 < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: aghik	%r0, %r1, -32769
#CHECK: error: invalid operand
#CHECK: aghik	%r0, %r1, 32768
#CHECK: error: invalid operand
#CHECK: aghik	%r0, %r1, foo

	aghik	%r0, %r1, -32769
	aghik	%r0, %r1, 32768
	aghik	%r0, %r1, foo

#CHECK: error: invalid operand
#CHECK: ahik	%r0, %r1, -32769
#CHECK: error: invalid operand
#CHECK: ahik	%r0, %r1, 32768
#CHECK: error: invalid operand
#CHECK: ahik	%r0, %r1, foo

	ahik	%r0, %r1, -32769
	ahik	%r0, %r1, 32768
	ahik	%r0, %r1, foo

#CHECK: error: invalid operand
#CHECK: aih	%r0, (-1 << 31) - 1
#CHECK: error: invalid operand
#CHECK: aih	%r0, (1 << 31)

	aih	%r0, (-1 << 31) - 1
	aih	%r0, (1 << 31)

#CHECK: error: invalid operand
#CHECK: cdlfbr	%f0, 0, %r0, -1
#CHECK: error: invalid operand
#CHECK: cdlfbr	%f0, 0, %r0, 16
#CHECK: error: invalid operand
#CHECK: cdlfbr	%f0, -1, %r0, 0
#CHECK: error: invalid operand
#CHECK: cdlfbr	%f0, 16, %r0, 0

	cdlfbr	%f0, 0, %r0, -1
	cdlfbr	%f0, 0, %r0, 16
	cdlfbr	%f0, -1, %r0, 0
	cdlfbr	%f0, 16, %r0, 0

#CHECK: error: invalid operand
#CHECK: cdlgbr	%f0, 0, %r0, -1
#CHECK: error: invalid operand
#CHECK: cdlgbr	%f0, 0, %r0, 16
#CHECK: error: invalid operand
#CHECK: cdlgbr	%f0, -1, %r0, 0
#CHECK: error: invalid operand
#CHECK: cdlgbr	%f0, 16, %r0, 0

	cdlgbr	%f0, 0, %r0, -1
	cdlgbr	%f0, 0, %r0, 16
	cdlgbr	%f0, -1, %r0, 0
	cdlgbr	%f0, 16, %r0, 0

#CHECK: error: invalid operand
#CHECK: celfbr	%f0, 0, %r0, -1
#CHECK: error: invalid operand
#CHECK: celfbr	%f0, 0, %r0, 16
#CHECK: error: invalid operand
#CHECK: celfbr	%f0, -1, %r0, 0
#CHECK: error: invalid operand
#CHECK: celfbr	%f0, 16, %r0, 0

	celfbr	%f0, 0, %r0, -1
	celfbr	%f0, 0, %r0, 16
	celfbr	%f0, -1, %r0, 0
	celfbr	%f0, 16, %r0, 0

#CHECK: error: invalid operand
#CHECK: celgbr	%f0, 0, %r0, -1
#CHECK: error: invalid operand
#CHECK: celgbr	%f0, 0, %r0, 16
#CHECK: error: invalid operand
#CHECK: celgbr	%f0, -1, %r0, 0
#CHECK: error: invalid operand
#CHECK: celgbr	%f0, 16, %r0, 0

	celgbr	%f0, 0, %r0, -1
	celgbr	%f0, 0, %r0, 16
	celgbr	%f0, -1, %r0, 0
	celgbr	%f0, 16, %r0, 0

#CHECK: error: invalid operand
#CHECK: chf	%r0, -524289
#CHECK: error: invalid operand
#CHECK: chf	%r0, 524288

	chf	%r0, -524289
	chf	%r0, 524288

#CHECK: error: invalid operand
#CHECK: cih	%r0, (-1 << 31) - 1
#CHECK: error: invalid operand
#CHECK: cih	%r0, (1 << 31)

	cih	%r0, (-1 << 31) - 1
	cih	%r0, (1 << 31)

#CHECK: error: invalid operand
#CHECK: clfdbr	%r0, 0, %f0, -1
#CHECK: error: invalid operand
#CHECK: clfdbr	%r0, 0, %f0, 16
#CHECK: error: invalid operand
#CHECK: clfdbr	%r0, -1, %f0, 0
#CHECK: error: invalid operand
#CHECK: clfdbr	%r0, 16, %f0, 0

	clfdbr	%r0, 0, %f0, -1
	clfdbr	%r0, 0, %f0, 16
	clfdbr	%r0, -1, %f0, 0
	clfdbr	%r0, 16, %f0, 0

#CHECK: error: invalid operand
#CHECK: clfebr	%r0, 0, %f0, -1
#CHECK: error: invalid operand
#CHECK: clfebr	%r0, 0, %f0, 16
#CHECK: error: invalid operand
#CHECK: clfebr	%r0, -1, %f0, 0
#CHECK: error: invalid operand
#CHECK: clfebr	%r0, 16, %f0, 0

	clfebr	%r0, 0, %f0, -1
	clfebr	%r0, 0, %f0, 16
	clfebr	%r0, -1, %f0, 0
	clfebr	%r0, 16, %f0, 0

#CHECK: error: invalid operand
#CHECK: clfxbr	%r0, 0, %f0, -1
#CHECK: error: invalid operand
#CHECK: clfxbr	%r0, 0, %f0, 16
#CHECK: error: invalid operand
#CHECK: clfxbr	%r0, -1, %f0, 0
#CHECK: error: invalid operand
#CHECK: clfxbr	%r0, 16, %f0, 0
#CHECK: error: invalid register pair
#CHECK: clfxbr	%r0, 0, %f14, 0

	clfxbr	%r0, 0, %f0, -1
	clfxbr	%r0, 0, %f0, 16
	clfxbr	%r0, -1, %f0, 0
	clfxbr	%r0, 16, %f0, 0
	clfxbr	%r0, 0, %f14, 0

#CHECK: error: invalid operand
#CHECK: clgdbr	%r0, 0, %f0, -1
#CHECK: error: invalid operand
#CHECK: clgdbr	%r0, 0, %f0, 16
#CHECK: error: invalid operand
#CHECK: clgdbr	%r0, -1, %f0, 0
#CHECK: error: invalid operand
#CHECK: clgdbr	%r0, 16, %f0, 0

	clgdbr	%r0, 0, %f0, -1
	clgdbr	%r0, 0, %f0, 16
	clgdbr	%r0, -1, %f0, 0
	clgdbr	%r0, 16, %f0, 0

#CHECK: error: invalid operand
#CHECK: clgebr	%r0, 0, %f0, -1
#CHECK: error: invalid operand
#CHECK: clgebr	%r0, 0, %f0, 16
#CHECK: error: invalid operand
#CHECK: clgebr	%r0, -1, %f0, 0
#CHECK: error: invalid operand
#CHECK: clgebr	%r0, 16, %f0, 0

	clgebr	%r0, 0, %f0, -1
	clgebr	%r0, 0, %f0, 16
	clgebr	%r0, -1, %f0, 0
	clgebr	%r0, 16, %f0, 0

#CHECK: error: invalid operand
#CHECK: clgxbr	%r0, 0, %f0, -1
#CHECK: error: invalid operand
#CHECK: clgxbr	%r0, 0, %f0, 16
#CHECK: error: invalid operand
#CHECK: clgxbr	%r0, -1, %f0, 0
#CHECK: error: invalid operand
#CHECK: clgxbr	%r0, 16, %f0, 0
#CHECK: error: invalid register pair
#CHECK: clgxbr	%r0, 0, %f14, 0

	clgxbr	%r0, 0, %f0, -1
	clgxbr	%r0, 0, %f0, 16
	clgxbr	%r0, -1, %f0, 0
	clgxbr	%r0, 16, %f0, 0
	clgxbr	%r0, 0, %f14, 0

#CHECK: error: invalid operand
#CHECK: clhf	%r0, -524289
#CHECK: error: invalid operand
#CHECK: clhf	%r0, 524288

	clhf	%r0, -524289
	clhf	%r0, 524288

#CHECK: error: invalid operand
#CHECK: clih	%r0, -1
#CHECK: error: invalid operand
#CHECK: clih	%r0, (1 << 32)

	clih	%r0, -1
	clih	%r0, (1 << 32)

#CHECK: error: invalid operand
#CHECK: cxlfbr	%f0, 0, %r0, -1
#CHECK: error: invalid operand
#CHECK: cxlfbr	%f0, 0, %r0, 16
#CHECK: error: invalid operand
#CHECK: cxlfbr	%f0, -1, %r0, 0
#CHECK: error: invalid operand
#CHECK: cxlfbr	%f0, 16, %r0, 0
#CHECK: error: invalid register pair
#CHECK: cxlfbr	%f2, 0, %r0, 0

	cxlfbr	%f0, 0, %r0, -1
	cxlfbr	%f0, 0, %r0, 16
	cxlfbr	%f0, -1, %r0, 0
	cxlfbr	%f0, 16, %r0, 0
	cxlfbr	%f2, 0, %r0, 0

#CHECK: error: invalid operand
#CHECK: cxlgbr	%f0, 0, %r0, -1
#CHECK: error: invalid operand
#CHECK: cxlgbr	%f0, 0, %r0, 16
#CHECK: error: invalid operand
#CHECK: cxlgbr	%f0, -1, %r0, 0
#CHECK: error: invalid operand
#CHECK: cxlgbr	%f0, 16, %r0, 0
#CHECK: error: invalid register pair
#CHECK: cxlgbr	%f2, 0, %r0, 0

	cxlgbr	%f0, 0, %r0, -1
	cxlgbr	%f0, 0, %r0, 16
	cxlgbr	%f0, -1, %r0, 0
	cxlgbr	%f0, 16, %r0, 0
	cxlgbr	%f2, 0, %r0, 0

#CHECK: error: {{(instruction requires: transactional-execution)?}}
#CHECK: etnd	%r7

	etnd	%r7

#CHECK: error: invalid operand
#CHECK: fidbra	%f0, 0, %f0, -1
#CHECK: error: invalid operand
#CHECK: fidbra	%f0, 0, %f0, 16
#CHECK: error: invalid operand
#CHECK: fidbra	%f0, -1, %f0, 0
#CHECK: error: invalid operand
#CHECK: fidbra	%f0, 16, %f0, 0

	fidbra	%f0, 0, %f0, -1
	fidbra	%f0, 0, %f0, 16
	fidbra	%f0, -1, %f0, 0
	fidbra	%f0, 16, %f0, 0

#CHECK: error: invalid operand
#CHECK: fiebra	%f0, 0, %f0, -1
#CHECK: error: invalid operand
#CHECK: fiebra	%f0, 0, %f0, 16
#CHECK: error: invalid operand
#CHECK: fiebra	%f0, -1, %f0, 0
#CHECK: error: invalid operand
#CHECK: fiebra	%f0, 16, %f0, 0

	fiebra	%f0, 0, %f0, -1
	fiebra	%f0, 0, %f0, 16
	fiebra	%f0, -1, %f0, 0
	fiebra	%f0, 16, %f0, 0

#CHECK: error: invalid operand
#CHECK: fixbra	%f0, 0, %f0, -1
#CHECK: error: invalid operand
#CHECK: fixbra	%f0, 0, %f0, 16
#CHECK: error: invalid operand
#CHECK: fixbra	%f0, -1, %f0, 0
#CHECK: error: invalid operand
#CHECK: fixbra	%f0, 16, %f0, 0
#CHECK: error: invalid register pair
#CHECK: fixbra	%f0, 0, %f2, 0
#CHECK: error: invalid register pair
#CHECK: fixbra	%f2, 0, %f0, 0

	fixbra	%f0, 0, %f0, -1
	fixbra	%f0, 0, %f0, 16
	fixbra	%f0, -1, %f0, 0
	fixbra	%f0, 16, %f0, 0
	fixbra	%f0, 0, %f2, 0
	fixbra	%f2, 0, %f0, 0

#CHECK: error: invalid operand
#CHECK: laa	%r0, %r0, -524289
#CHECK: error: invalid operand
#CHECK: laa	%r0, %r0, 524288
#CHECK: error: invalid use of indexed addressing
#CHECK: laa	%r0, %r0, 0(%r1,%r2)

	laa	%r0, %r0, -524289
	laa	%r0, %r0, 524288
	laa	%r0, %r0, 0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: laag	%r0, %r0, -524289
#CHECK: error: invalid operand
#CHECK: laag	%r0, %r0, 524288
#CHECK: error: invalid use of indexed addressing
#CHECK: laag	%r0, %r0, 0(%r1,%r2)

	laag	%r0, %r0, -524289
	laag	%r0, %r0, 524288
	laag	%r0, %r0, 0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: laal	%r0, %r0, -524289
#CHECK: error: invalid operand
#CHECK: laal	%r0, %r0, 524288
#CHECK: error: invalid use of indexed addressing
#CHECK: laal	%r0, %r0, 0(%r1,%r2)

	laal	%r0, %r0, -524289
	laal	%r0, %r0, 524288
	laal	%r0, %r0, 0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: laalg	%r0, %r0, -524289
#CHECK: error: invalid operand
#CHECK: laalg	%r0, %r0, 524288
#CHECK: error: invalid use of indexed addressing
#CHECK: laalg	%r0, %r0, 0(%r1,%r2)

	laalg	%r0, %r0, -524289
	laalg	%r0, %r0, 524288
	laalg	%r0, %r0, 0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: lan	%r0, %r0, -524289
#CHECK: error: invalid operand
#CHECK: lan	%r0, %r0, 524288
#CHECK: error: invalid use of indexed addressing
#CHECK: lan	%r0, %r0, 0(%r1,%r2)

	lan	%r0, %r0, -524289
	lan	%r0, %r0, 524288
	lan	%r0, %r0, 0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: lang	%r0, %r0, -524289
#CHECK: error: invalid operand
#CHECK: lang	%r0, %r0, 524288
#CHECK: error: invalid use of indexed addressing
#CHECK: lang	%r0, %r0, 0(%r1,%r2)

	lang	%r0, %r0, -524289
	lang	%r0, %r0, 524288
	lang	%r0, %r0, 0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: lao	%r0, %r0, -524289
#CHECK: error: invalid operand
#CHECK: lao	%r0, %r0, 524288
#CHECK: error: invalid use of indexed addressing
#CHECK: lao	%r0, %r0, 0(%r1,%r2)

	lao	%r0, %r0, -524289
	lao	%r0, %r0, 524288
	lao	%r0, %r0, 0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: laog	%r0, %r0, -524289
#CHECK: error: invalid operand
#CHECK: laog	%r0, %r0, 524288
#CHECK: error: invalid use of indexed addressing
#CHECK: laog	%r0, %r0, 0(%r1,%r2)

	laog	%r0, %r0, -524289
	laog	%r0, %r0, 524288
	laog	%r0, %r0, 0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: lax	%r0, %r0, -524289
#CHECK: error: invalid operand
#CHECK: lax	%r0, %r0, 524288
#CHECK: error: invalid use of indexed addressing
#CHECK: lax	%r0, %r0, 0(%r1,%r2)

	lax	%r0, %r0, -524289
	lax	%r0, %r0, 524288
	lax	%r0, %r0, 0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: laxg	%r0, %r0, -524289
#CHECK: error: invalid operand
#CHECK: laxg	%r0, %r0, 524288
#CHECK: error: invalid use of indexed addressing
#CHECK: laxg	%r0, %r0, 0(%r1,%r2)

	laxg	%r0, %r0, -524289
	laxg	%r0, %r0, 524288
	laxg	%r0, %r0, 0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: lbh	%r0, -524289
#CHECK: error: invalid operand
#CHECK: lbh	%r0, 524288

	lbh	%r0, -524289
	lbh	%r0, 524288

#CHECK: error: invalid operand
#CHECK: ldxbra	%f0, 0, %f0, -1
#CHECK: error: invalid operand
#CHECK: ldxbra	%f0, 0, %f0, 16
#CHECK: error: invalid operand
#CHECK: ldxbra	%f0, -1, %f0, 0
#CHECK: error: invalid operand
#CHECK: ldxbra	%f0, 16, %f0, 0
#CHECK: error: invalid register pair
#CHECK: ldxbra	%f0, 0, %f2, 0
#CHECK: error: invalid register pair
#CHECK: ldxbra	%f2, 0, %f0, 0

	ldxbra	%f0, 0, %f0, -1
	ldxbra	%f0, 0, %f0, 16
	ldxbra	%f0, -1, %f0, 0
	ldxbra	%f0, 16, %f0, 0
	ldxbra	%f0, 0, %f2, 0
	ldxbra	%f2, 0, %f0, 0

#CHECK: error: invalid operand
#CHECK: ledbra	%f0, 0, %f0, -1
#CHECK: error: invalid operand
#CHECK: ledbra	%f0, 0, %f0, 16
#CHECK: error: invalid operand
#CHECK: ledbra	%f0, -1, %f0, 0
#CHECK: error: invalid operand
#CHECK: ledbra	%f0, 16, %f0, 0

	ledbra	%f0, 0, %f0, -1
	ledbra	%f0, 0, %f0, 16
	ledbra	%f0, -1, %f0, 0
	ledbra	%f0, 16, %f0, 0

#CHECK: error: invalid operand
#CHECK: lexbra	%f0, 0, %f0, -1
#CHECK: error: invalid operand
#CHECK: lexbra	%f0, 0, %f0, 16
#CHECK: error: invalid operand
#CHECK: lexbra	%f0, -1, %f0, 0
#CHECK: error: invalid operand
#CHECK: lexbra	%f0, 16, %f0, 0
#CHECK: error: invalid register pair
#CHECK: lexbra	%f0, 0, %f2, 0
#CHECK: error: invalid register pair
#CHECK: lexbra	%f2, 0, %f0, 0

	lexbra	%f0, 0, %f0, -1
	lexbra	%f0, 0, %f0, 16
	lexbra	%f0, -1, %f0, 0
	lexbra	%f0, 16, %f0, 0
	lexbra	%f0, 0, %f2, 0
	lexbra	%f2, 0, %f0, 0

#CHECK: error: invalid operand
#CHECK: lfh	%r0, -524289
#CHECK: error: invalid operand
#CHECK: lfh	%r0, 524288

	lfh	%r0, -524289
	lfh	%r0, 524288

#CHECK: error: invalid operand
#CHECK: lhh	%r0, -524289
#CHECK: error: invalid operand
#CHECK: lhh	%r0, 524288

	lhh	%r0, -524289
	lhh	%r0, 524288

#CHECK: error: invalid operand
#CHECK: llch	%r0, -524289
#CHECK: error: invalid operand
#CHECK: llch	%r0, 524288

	llch	%r0, -524289
	llch	%r0, 524288

#CHECK: error: invalid operand
#CHECK: llhh	%r0, -524289
#CHECK: error: invalid operand
#CHECK: llhh	%r0, 524288

	llhh	%r0, -524289
	llhh	%r0, 524288

#CHECK: error: invalid operand
#CHECK: loc	%r0,0,-1
#CHECK: error: invalid operand
#CHECK: loc	%r0,0,16
#CHECK: error: invalid operand
#CHECK: loc	%r0,-524289,1
#CHECK: error: invalid operand
#CHECK: loc	%r0,524288,1
#CHECK: error: invalid use of indexed addressing
#CHECK: loc	%r0,0(%r1,%r2),1

	loc	%r0,0,-1
	loc	%r0,0,16
	loc	%r0,-524289,1
	loc	%r0,524288,1
	loc	%r0,0(%r1,%r2),1

#CHECK: error: invalid operand
#CHECK: locg	%r0,0,-1
#CHECK: error: invalid operand
#CHECK: locg	%r0,0,16
#CHECK: error: invalid operand
#CHECK: locg	%r0,-524289,1
#CHECK: error: invalid operand
#CHECK: locg	%r0,524288,1
#CHECK: error: invalid use of indexed addressing
#CHECK: locg	%r0,0(%r1,%r2),1

	locg	%r0,0,-1
	locg	%r0,0,16
	locg	%r0,-524289,1
	locg	%r0,524288,1
	locg	%r0,0(%r1,%r2),1

#CHECK: error: invalid operand
#CHECK: locgr	%r0,%r0,-1
#CHECK: error: invalid operand
#CHECK: locgr	%r0,%r0,16

	locgr	%r0,%r0,-1
	locgr	%r0,%r0,16

#CHECK: error: invalid operand
#CHECK: locr	%r0,%r0,-1
#CHECK: error: invalid operand
#CHECK: locr	%r0,%r0,16

	locr	%r0,%r0,-1
	locr	%r0,%r0,16

#CHECK: error: {{(instruction requires: transactional-execution)?}}
#CHECK: ntstg	%r0, 524287(%r1,%r15)

	ntstg	%r0, 524287(%r1,%r15)

#CHECK: error: {{(instruction requires: processor-assist)?}}
#CHECK: ppa	%r4, %r6, 7

	ppa	%r4, %r6, 7

#CHECK: error: {{(instruction requires: miscellaneous-extensions)?}}
#CHECK: risbgn	%r1, %r2, 0, 0, 0

	risbgn	%r1, %r2, 0, 0, 0

#CHECK: error: invalid operand
#CHECK: risbhg	%r0,%r0,0,0,-1
#CHECK: error: invalid operand
#CHECK: risbhg	%r0,%r0,0,0,64
#CHECK: error: invalid operand
#CHECK: risbhg	%r0,%r0,0,-1,0
#CHECK: error: invalid operand
#CHECK: risbhg	%r0,%r0,0,256,0
#CHECK: error: invalid operand
#CHECK: risbhg	%r0,%r0,-1,0,0
#CHECK: error: invalid operand
#CHECK: risbhg	%r0,%r0,256,0,0

	risbhg	%r0,%r0,0,0,-1
	risbhg	%r0,%r0,0,0,64
	risbhg	%r0,%r0,0,-1,0
	risbhg	%r0,%r0,0,256,0
	risbhg	%r0,%r0,-1,0,0
	risbhg	%r0,%r0,256,0,0

#CHECK: error: invalid operand
#CHECK: risblg	%r0,%r0,0,0,-1
#CHECK: error: invalid operand
#CHECK: risblg	%r0,%r0,0,0,64
#CHECK: error: invalid operand
#CHECK: risblg	%r0,%r0,0,-1,0
#CHECK: error: invalid operand
#CHECK: risblg	%r0,%r0,0,256,0
#CHECK: error: invalid operand
#CHECK: risblg	%r0,%r0,-1,0,0
#CHECK: error: invalid operand
#CHECK: risblg	%r0,%r0,256,0,0

	risblg	%r0,%r0,0,0,-1
	risblg	%r0,%r0,0,0,64
	risblg	%r0,%r0,0,-1,0
	risblg	%r0,%r0,0,256,0
	risblg	%r0,%r0,-1,0,0
	risblg	%r0,%r0,256,0,0

#CHECK: error: invalid operand
#CHECK: slak	%r0,%r0,-524289
#CHECK: error: invalid operand
#CHECK: slak	%r0,%r0,524288
#CHECK: error: %r0 used in an address
#CHECK: slak	%r0,%r0,0(%r0)
#CHECK: error: invalid use of indexed addressing
#CHECK: slak	%r0,%r0,0(%r1,%r2)

	slak	%r0,%r0,-524289
	slak	%r0,%r0,524288
	slak	%r0,%r0,0(%r0)
	slak	%r0,%r0,0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: sllk	%r0,%r0,-524289
#CHECK: error: invalid operand
#CHECK: sllk	%r0,%r0,524288
#CHECK: error: %r0 used in an address
#CHECK: sllk	%r0,%r0,0(%r0)
#CHECK: error: invalid use of indexed addressing
#CHECK: sllk	%r0,%r0,0(%r1,%r2)

	sllk	%r0,%r0,-524289
	sllk	%r0,%r0,524288
	sllk	%r0,%r0,0(%r0)
	sllk	%r0,%r0,0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: srak	%r0,%r0,-524289
#CHECK: error: invalid operand
#CHECK: srak	%r0,%r0,524288
#CHECK: error: %r0 used in an address
#CHECK: srak	%r0,%r0,0(%r0)
#CHECK: error: invalid use of indexed addressing
#CHECK: srak	%r0,%r0,0(%r1,%r2)

	srak	%r0,%r0,-524289
	srak	%r0,%r0,524288
	srak	%r0,%r0,0(%r0)
	srak	%r0,%r0,0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: srlk	%r0,%r0,-524289
#CHECK: error: invalid operand
#CHECK: srlk	%r0,%r0,524288
#CHECK: error: %r0 used in an address
#CHECK: srlk	%r0,%r0,0(%r0)
#CHECK: error: invalid use of indexed addressing
#CHECK: srlk	%r0,%r0,0(%r1,%r2)

	srlk	%r0,%r0,-524289
	srlk	%r0,%r0,524288
	srlk	%r0,%r0,0(%r0)
	srlk	%r0,%r0,0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: stch	%r0, -524289
#CHECK: error: invalid operand
#CHECK: stch	%r0, 524288

	stch	%r0, -524289
	stch	%r0, 524288

#CHECK: error: invalid operand
#CHECK: sthh	%r0, -524289
#CHECK: error: invalid operand
#CHECK: sthh	%r0, 524288

	sthh	%r0, -524289
	sthh	%r0, 524288

#CHECK: error: invalid operand
#CHECK: stfh	%r0, -524289
#CHECK: error: invalid operand
#CHECK: stfh	%r0, 524288

	stfh	%r0, -524289
	stfh	%r0, 524288

#CHECK: error: invalid operand
#CHECK: stoc	%r0,0,-1
#CHECK: error: invalid operand
#CHECK: stoc	%r0,0,16
#CHECK: error: invalid operand
#CHECK: stoc	%r0,-524289,1
#CHECK: error: invalid operand
#CHECK: stoc	%r0,524288,1
#CHECK: error: invalid use of indexed addressing
#CHECK: stoc	%r0,0(%r1,%r2),1

	stoc	%r0,0,-1
	stoc	%r0,0,16
	stoc	%r0,-524289,1
	stoc	%r0,524288,1
	stoc	%r0,0(%r1,%r2),1

#CHECK: error: invalid operand
#CHECK: stocg	%r0,0,-1
#CHECK: error: invalid operand
#CHECK: stocg	%r0,0,16
#CHECK: error: invalid operand
#CHECK: stocg	%r0,-524289,1
#CHECK: error: invalid operand
#CHECK: stocg	%r0,524288,1
#CHECK: error: invalid use of indexed addressing
#CHECK: stocg	%r0,0(%r1,%r2),1

	stocg	%r0,0,-1
	stocg	%r0,0,16
	stocg	%r0,-524289,1
	stocg	%r0,524288,1
	stocg	%r0,0(%r1,%r2),1

#CHECK: error: {{(instruction requires: transactional-execution)?}}
#CHECK: tabort	4095(%r1)

	tabort	4095(%r1)

#CHECK: error: {{(instruction requires: transactional-execution)?}}
#CHECK: tbegin	4095(%r1), 42

	tbegin	4095(%r1), 42

#CHECK: error: {{(instruction requires: transactional-execution)?}}
#CHECK: tbeginc	4095(%r1), 42

	tbeginc	4095(%r1), 42

#CHECK: error: {{(instruction requires: transactional-execution)?}}
#CHECK: tend

	tend

