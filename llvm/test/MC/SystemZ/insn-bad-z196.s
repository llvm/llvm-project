# For z196 only.
# RUN: not llvm-mc -triple s390x-linux-gnu -mcpu=z196 < %s 2> %t
# RUN: FileCheck < %t %s
# RUN: not llvm-mc -triple s390x-linux-gnu -mcpu=arch9 < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: adtra	%f0, %f0, %f0, -1
#CHECK: error: invalid operand
#CHECK: adtra	%f0, %f0, %f0, 16

	adtra	%f0, %f0, %f0, -1
	adtra	%f0, %f0, %f0, 16

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
#CHECK: axtra	%f0, %f0, %f0, -1
#CHECK: error: invalid operand
#CHECK: axtra	%f0, %f0, %f0, 16
#CHECK: error: invalid register pair
#CHECK: axtra	%f0, %f0, %f2, 0
#CHECK: error: invalid register pair
#CHECK: axtra	%f0, %f2, %f0, 0
#CHECK: error: invalid register pair
#CHECK: axtra	%f2, %f0, %f0, 0

	axtra	%f0, %f0, %f0, -1
	axtra	%f0, %f0, %f0, 16
	axtra	%f0, %f0, %f2, 0
	axtra	%f0, %f2, %f0, 0
	axtra	%f2, %f0, %f0, 0

#CHECK: error: instruction requires: execution-hint
#CHECK: bpp	0, 0, 0

	bpp	0, 0, 0

#CHECK: error: instruction requires: execution-hint
#CHECK: bprp	0, 0, 0

	bprp	0, 0, 0

#CHECK: error: offset out of range
#CHECK: brcth   %r0, -0x1000000002
#CHECK: error: offset out of range
#CHECK: brcth   %r0, -1
#CHECK: error: offset out of range
#CHECK: brcth   %r0, 1
#CHECK: error: offset out of range
#CHECK: brcth   %r0, 0x100000000

        brcth   %r0, -0x1000000002
        brcth   %r0, -1
        brcth   %r0, 1
        brcth   %r0, 0x100000000

#CHECK: error: invalid operand
#CHECK: cdfbra	%f0, 0, %r0, -1
#CHECK: error: invalid operand
#CHECK: cdfbra	%f0, 0, %r0, 16
#CHECK: error: invalid operand
#CHECK: cdfbra	%f0, -1, %r0, 0
#CHECK: error: invalid operand
#CHECK: cdfbra	%f0, 16, %r0, 0

	cdfbra	%f0, 0, %r0, -1
	cdfbra	%f0, 0, %r0, 16
	cdfbra	%f0, -1, %r0, 0
	cdfbra	%f0, 16, %r0, 0

#CHECK: error: invalid operand
#CHECK: cdftr	%f0, 0, %r0, -1
#CHECK: error: invalid operand
#CHECK: cdftr	%f0, 0, %r0, 16
#CHECK: error: invalid operand
#CHECK: cdftr	%f0, -1, %r0, 0
#CHECK: error: invalid operand
#CHECK: cdftr	%f0, 16, %r0, 0

	cdftr	%f0, 0, %r0, -1
	cdftr	%f0, 0, %r0, 16
	cdftr	%f0, -1, %r0, 0
	cdftr	%f0, 16, %r0, 0

#CHECK: error: invalid operand
#CHECK: cdgbra	%f0, 0, %r0, -1
#CHECK: error: invalid operand
#CHECK: cdgbra	%f0, 0, %r0, 16
#CHECK: error: invalid operand
#CHECK: cdgbra	%f0, -1, %r0, 0
#CHECK: error: invalid operand
#CHECK: cdgbra	%f0, 16, %r0, 0

	cdgbra	%f0, 0, %r0, -1
	cdgbra	%f0, 0, %r0, 16
	cdgbra	%f0, -1, %r0, 0
	cdgbra	%f0, 16, %r0, 0

#CHECK: error: invalid operand
#CHECK: cdgtra	%f0, 0, %r0, -1
#CHECK: error: invalid operand
#CHECK: cdgtra	%f0, 0, %r0, 16
#CHECK: error: invalid operand
#CHECK: cdgtra	%f0, -1, %r0, 0
#CHECK: error: invalid operand
#CHECK: cdgtra	%f0, 16, %r0, 0

	cdgtra	%f0, 0, %r0, -1
	cdgtra	%f0, 0, %r0, 16
	cdgtra	%f0, -1, %r0, 0
	cdgtra	%f0, 16, %r0, 0

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
#CHECK: cdlftr	%f0, 0, %r0, -1
#CHECK: error: invalid operand
#CHECK: cdlftr	%f0, 0, %r0, 16
#CHECK: error: invalid operand
#CHECK: cdlftr	%f0, -1, %r0, 0
#CHECK: error: invalid operand
#CHECK: cdlftr	%f0, 16, %r0, 0

	cdlftr	%f0, 0, %r0, -1
	cdlftr	%f0, 0, %r0, 16
	cdlftr	%f0, -1, %r0, 0
	cdlftr	%f0, 16, %r0, 0

#CHECK: error: instruction requires: dfp-zoned-conversion
#CHECK: cdzt	%f0, 0(1), 0

	cdzt	%f0, 0(1), 0

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
#CHECK: cdlgtr	%f0, 0, %r0, -1
#CHECK: error: invalid operand
#CHECK: cdlgtr	%f0, 0, %r0, 16
#CHECK: error: invalid operand
#CHECK: cdlgtr	%f0, -1, %r0, 0
#CHECK: error: invalid operand
#CHECK: cdlgtr	%f0, 16, %r0, 0

	cdlgtr	%f0, 0, %r0, -1
	cdlgtr	%f0, 0, %r0, 16
	cdlgtr	%f0, -1, %r0, 0
	cdlgtr	%f0, 16, %r0, 0

#CHECK: error: invalid operand
#CHECK: cefbra	%f0, 0, %r0, -1
#CHECK: error: invalid operand
#CHECK: cefbra	%f0, 0, %r0, 16
#CHECK: error: invalid operand
#CHECK: cefbra	%f0, -1, %r0, 0
#CHECK: error: invalid operand
#CHECK: cefbra	%f0, 16, %r0, 0

	cefbra	%f0, 0, %r0, -1
	cefbra	%f0, 0, %r0, 16
	cefbra	%f0, -1, %r0, 0
	cefbra	%f0, 16, %r0, 0

#CHECK: error: invalid operand
#CHECK: cegbra	%f0, 0, %r0, -1
#CHECK: error: invalid operand
#CHECK: cegbra	%f0, 0, %r0, 16
#CHECK: error: invalid operand
#CHECK: cegbra	%f0, -1, %r0, 0
#CHECK: error: invalid operand
#CHECK: cegbra	%f0, 16, %r0, 0

	cegbra	%f0, 0, %r0, -1
	cegbra	%f0, 0, %r0, 16
	cegbra	%f0, -1, %r0, 0
	cegbra	%f0, 16, %r0, 0

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
#CHECK: cfdbra	%r0, 0, %f0, -1
#CHECK: error: invalid operand
#CHECK: cfdbra	%r0, 0, %f0, 16
#CHECK: error: invalid operand
#CHECK: cfdbra	%r0, -1, %f0, 0
#CHECK: error: invalid operand
#CHECK: cfdbra	%r0, 16, %f0, 0

	cfdbra	%r0, 0, %f0, -1
	cfdbra	%r0, 0, %f0, 16
	cfdbra	%r0, -1, %f0, 0
	cfdbra	%r0, 16, %f0, 0

#CHECK: error: invalid operand
#CHECK: cfdtr	%r0, 0, %f0, -1
#CHECK: error: invalid operand
#CHECK: cfdtr	%r0, 0, %f0, 16
#CHECK: error: invalid operand
#CHECK: cfdtr	%r0, -1, %f0, 0
#CHECK: error: invalid operand
#CHECK: cfdtr	%r0, 16, %f0, 0

	cfdtr	%r0, 0, %f0, -1
	cfdtr	%r0, 0, %f0, 16
	cfdtr	%r0, -1, %f0, 0
	cfdtr	%r0, 16, %f0, 0

#CHECK: error: invalid operand
#CHECK: cfebra	%r0, 0, %f0, -1
#CHECK: error: invalid operand
#CHECK: cfebra	%r0, 0, %f0, 16
#CHECK: error: invalid operand
#CHECK: cfebra	%r0, -1, %f0, 0
#CHECK: error: invalid operand
#CHECK: cfebra	%r0, 16, %f0, 0

	cfebra	%r0, 0, %f0, -1
	cfebra	%r0, 0, %f0, 16
	cfebra	%r0, -1, %f0, 0
	cfebra	%r0, 16, %f0, 0

#CHECK: error: invalid operand
#CHECK: cfxbra	%r0, 0, %f0, -1
#CHECK: error: invalid operand
#CHECK: cfxbra	%r0, 0, %f0, 16
#CHECK: error: invalid operand
#CHECK: cfxbra	%r0, -1, %f0, 0
#CHECK: error: invalid operand
#CHECK: cfxbra	%r0, 16, %f0, 0
#CHECK: error: invalid register pair
#CHECK: cfxbra	%r0, 0, %f14, 0

	cfxbra	%r0, 0, %f0, -1
	cfxbra	%r0, 0, %f0, 16
	cfxbra	%r0, -1, %f0, 0
	cfxbra	%r0, 16, %f0, 0
	cfxbra	%r0, 0, %f14, 0

#CHECK: error: invalid operand
#CHECK: cfxtr	%r0, 0, %f0, -1
#CHECK: error: invalid operand
#CHECK: cfxtr	%r0, 0, %f0, 16
#CHECK: error: invalid operand
#CHECK: cfxtr	%r0, -1, %f0, 0
#CHECK: error: invalid operand
#CHECK: cfxtr	%r0, 16, %f0, 0
#CHECK: error: invalid register pair
#CHECK: cfxtr	%r0, 0, %f14, 0

	cfxtr	%r0, 0, %f0, -1
	cfxtr	%r0, 0, %f0, 16
	cfxtr	%r0, -1, %f0, 0
	cfxtr	%r0, 16, %f0, 0
	cfxtr	%r0, 0, %f14, 0

#CHECK: error: invalid operand
#CHECK: cgdbra	%r0, 0, %f0, -1
#CHECK: error: invalid operand
#CHECK: cgdbra	%r0, 0, %f0, 16
#CHECK: error: invalid operand
#CHECK: cgdbra	%r0, -1, %f0, 0
#CHECK: error: invalid operand
#CHECK: cgdbra	%r0, 16, %f0, 0

	cgdbra	%r0, 0, %f0, -1
	cgdbra	%r0, 0, %f0, 16
	cgdbra	%r0, -1, %f0, 0
	cgdbra	%r0, 16, %f0, 0

#CHECK: error: invalid operand
#CHECK: cgdtra	%r0, 0, %f0, -1
#CHECK: error: invalid operand
#CHECK: cgdtra	%r0, 0, %f0, 16
#CHECK: error: invalid operand
#CHECK: cgdtra	%r0, -1, %f0, 0
#CHECK: error: invalid operand
#CHECK: cgdtra	%r0, 16, %f0, 0

	cgdtra	%r0, 0, %f0, -1
	cgdtra	%r0, 0, %f0, 16
	cgdtra	%r0, -1, %f0, 0
	cgdtra	%r0, 16, %f0, 0

#CHECK: error: invalid operand
#CHECK: cgebra	%r0, 0, %f0, -1
#CHECK: error: invalid operand
#CHECK: cgebra	%r0, 0, %f0, 16
#CHECK: error: invalid operand
#CHECK: cgebra	%r0, -1, %f0, 0
#CHECK: error: invalid operand
#CHECK: cgebra	%r0, 16, %f0, 0

	cgebra	%r0, 0, %f0, -1
	cgebra	%r0, 0, %f0, 16
	cgebra	%r0, -1, %f0, 0
	cgebra	%r0, 16, %f0, 0

#CHECK: error: invalid operand
#CHECK: cgxbra	%r0, 0, %f0, -1
#CHECK: error: invalid operand
#CHECK: cgxbra	%r0, 0, %f0, 16
#CHECK: error: invalid operand
#CHECK: cgxbra	%r0, -1, %f0, 0
#CHECK: error: invalid operand
#CHECK: cgxbra	%r0, 16, %f0, 0
#CHECK: error: invalid register pair
#CHECK: cgxbra	%r0, 0, %f14, 0

	cgxbra	%r0, 0, %f0, -1
	cgxbra	%r0, 0, %f0, 16
	cgxbra	%r0, -1, %f0, 0
	cgxbra	%r0, 16, %f0, 0
	cgxbra	%r0, 0, %f14, 0

#CHECK: error: invalid operand
#CHECK: cgxtra	%r0, 0, %f0, -1
#CHECK: error: invalid operand
#CHECK: cgxtra	%r0, 0, %f0, 16
#CHECK: error: invalid operand
#CHECK: cgxtra	%r0, -1, %f0, 0
#CHECK: error: invalid operand
#CHECK: cgxtra	%r0, 16, %f0, 0
#CHECK: error: invalid register pair
#CHECK: cgxtra	%r0, 0, %f14, 0

	cgxtra	%r0, 0, %f0, -1
	cgxtra	%r0, 0, %f0, 16
	cgxtra	%r0, -1, %f0, 0
	cgxtra	%r0, 16, %f0, 0
	cgxtra	%r0, 0, %f14, 0

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
#CHECK: clfdtr	%r0, 0, %f0, -1
#CHECK: error: invalid operand
#CHECK: clfdtr	%r0, 0, %f0, 16
#CHECK: error: invalid operand
#CHECK: clfdtr	%r0, -1, %f0, 0
#CHECK: error: invalid operand
#CHECK: clfdtr	%r0, 16, %f0, 0

	clfdtr	%r0, 0, %f0, -1
	clfdtr	%r0, 0, %f0, 16
	clfdtr	%r0, -1, %f0, 0
	clfdtr	%r0, 16, %f0, 0

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
#CHECK: clfxtr	%r0, 0, %f0, -1
#CHECK: error: invalid operand
#CHECK: clfxtr	%r0, 0, %f0, 16
#CHECK: error: invalid operand
#CHECK: clfxtr	%r0, -1, %f0, 0
#CHECK: error: invalid operand
#CHECK: clfxtr	%r0, 16, %f0, 0
#CHECK: error: invalid register pair
#CHECK: clfxtr	%r0, 0, %f14, 0

	clfxtr	%r0, 0, %f0, -1
	clfxtr	%r0, 0, %f0, 16
	clfxtr	%r0, -1, %f0, 0
	clfxtr	%r0, 16, %f0, 0
	clfxtr	%r0, 0, %f14, 0

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
#CHECK: clgdtr	%r0, 0, %f0, -1
#CHECK: error: invalid operand
#CHECK: clgdtr	%r0, 0, %f0, 16
#CHECK: error: invalid operand
#CHECK: clgdtr	%r0, -1, %f0, 0
#CHECK: error: invalid operand
#CHECK: clgdtr	%r0, 16, %f0, 0

	clgdtr	%r0, 0, %f0, -1
	clgdtr	%r0, 0, %f0, 16
	clgdtr	%r0, -1, %f0, 0
	clgdtr	%r0, 16, %f0, 0

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
#CHECK: clgxtr	%r0, 0, %f0, -1
#CHECK: error: invalid operand
#CHECK: clgxtr	%r0, 0, %f0, 16
#CHECK: error: invalid operand
#CHECK: clgxtr	%r0, -1, %f0, 0
#CHECK: error: invalid operand
#CHECK: clgxtr	%r0, 16, %f0, 0
#CHECK: error: invalid register pair
#CHECK: clgxtr	%r0, 0, %f14, 0

	clgxtr	%r0, 0, %f0, -1
	clgxtr	%r0, 0, %f0, 16
	clgxtr	%r0, -1, %f0, 0
	clgxtr	%r0, 16, %f0, 0
	clgxtr	%r0, 0, %f14, 0

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
#CHECK: cxfbra	%f0, 0, %r0, -1
#CHECK: error: invalid operand
#CHECK: cxfbra	%f0, 0, %r0, 16
#CHECK: error: invalid operand
#CHECK: cxfbra	%f0, -1, %r0, 0
#CHECK: error: invalid operand
#CHECK: cxfbra	%f0, 16, %r0, 0
#CHECK: error: invalid register pair
#CHECK: cxfbra	%f2, 0, %r0, 0

	cxfbra	%f0, 0, %r0, -1
	cxfbra	%f0, 0, %r0, 16
	cxfbra	%f0, -1, %r0, 0
	cxfbra	%f0, 16, %r0, 0
	cxfbra	%f2, 0, %r0, 0

#CHECK: error: invalid operand
#CHECK: cxftr	%f0, 0, %r0, -1
#CHECK: error: invalid operand
#CHECK: cxftr	%f0, 0, %r0, 16
#CHECK: error: invalid operand
#CHECK: cxftr	%f0, -1, %r0, 0
#CHECK: error: invalid operand
#CHECK: cxftr	%f0, 16, %r0, 0
#CHECK: error: invalid register pair
#CHECK: cxftr	%f2, 0, %r0, 0

	cxftr	%f0, 0, %r0, -1
	cxftr	%f0, 0, %r0, 16
	cxftr	%f0, -1, %r0, 0
	cxftr	%f0, 16, %r0, 0
	cxftr	%f2, 0, %r0, 0

#CHECK: error: invalid operand
#CHECK: cxgbra	%f0, 0, %r0, -1
#CHECK: error: invalid operand
#CHECK: cxgbra	%f0, 0, %r0, 16
#CHECK: error: invalid operand
#CHECK: cxgbra	%f0, -1, %r0, 0
#CHECK: error: invalid operand
#CHECK: cxgbra	%f0, 16, %r0, 0
#CHECK: error: invalid register pair
#CHECK: cxgbra	%f2, 0, %r0, 0

	cxgbra	%f0, 0, %r0, -1
	cxgbra	%f0, 0, %r0, 16
	cxgbra	%f0, -1, %r0, 0
	cxgbra	%f0, 16, %r0, 0
	cxgbra	%f2, 0, %r0, 0

#CHECK: error: invalid operand
#CHECK: cxgtra	%f0, 0, %r0, -1
#CHECK: error: invalid operand
#CHECK: cxgtra	%f0, 0, %r0, 16
#CHECK: error: invalid operand
#CHECK: cxgtra	%f0, -1, %r0, 0
#CHECK: error: invalid operand
#CHECK: cxgtra	%f0, 16, %r0, 0
#CHECK: error: invalid register pair
#CHECK: cxgtra	%f2, 0, %r0, 0

	cxgtra	%f0, 0, %r0, -1
	cxgtra	%f0, 0, %r0, 16
	cxgtra	%f0, -1, %r0, 0
	cxgtra	%f0, 16, %r0, 0
	cxgtra	%f2, 0, %r0, 0

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
#CHECK: cxlftr	%f0, 0, %r0, -1
#CHECK: error: invalid operand
#CHECK: cxlftr	%f0, 0, %r0, 16
#CHECK: error: invalid operand
#CHECK: cxlftr	%f0, -1, %r0, 0
#CHECK: error: invalid operand
#CHECK: cxlftr	%f0, 16, %r0, 0
#CHECK: error: invalid register pair
#CHECK: cxlftr	%f2, 0, %r0, 0

	cxlftr	%f0, 0, %r0, -1
	cxlftr	%f0, 0, %r0, 16
	cxlftr	%f0, -1, %r0, 0
	cxlftr	%f0, 16, %r0, 0
	cxlftr	%f2, 0, %r0, 0

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

#CHECK: error: invalid operand
#CHECK: cxlgtr	%f0, 0, %r0, -1
#CHECK: error: invalid operand
#CHECK: cxlgtr	%f0, 0, %r0, 16
#CHECK: error: invalid operand
#CHECK: cxlgtr	%f0, -1, %r0, 0
#CHECK: error: invalid operand
#CHECK: cxlgtr	%f0, 16, %r0, 0
#CHECK: error: invalid register pair
#CHECK: cxlgtr	%f2, 0, %r0, 0

	cxlgtr	%f0, 0, %r0, -1
	cxlgtr	%f0, 0, %r0, 16
	cxlgtr	%f0, -1, %r0, 0
	cxlgtr	%f0, 16, %r0, 0
	cxlgtr	%f2, 0, %r0, 0

#CHECK: error: instruction requires: dfp-zoned-conversion
#CHECK: cxzt	%f0, 0(1), 0

	cxzt	%f0, 0(1), 0

#CHECK: error: instruction requires: dfp-zoned-conversion
#CHECK: czdt	%f0, 0(1), 0

	czdt	%f0, 0(1), 0

#CHECK: error: instruction requires: dfp-zoned-conversion
#CHECK: czxt	%f0, 0(1), 0

	czxt	%f0, 0(1), 0

#CHECK: error: invalid operand
#CHECK: ddtra	%f0, %f0, %f0, -1
#CHECK: error: invalid operand
#CHECK: ddtra	%f0, %f0, %f0, 16

	ddtra	%f0, %f0, %f0, -1
	ddtra	%f0, %f0, %f0, 16

#CHECK: error: invalid operand
#CHECK: dxtra	%f0, %f0, %f0, -1
#CHECK: error: invalid operand
#CHECK: dxtra	%f0, %f0, %f0, 16
#CHECK: error: invalid register pair
#CHECK: dxtra	%f0, %f0, %f2, 0
#CHECK: error: invalid register pair
#CHECK: dxtra	%f0, %f2, %f0, 0
#CHECK: error: invalid register pair
#CHECK: dxtra	%f2, %f0, %f0, 0

	dxtra	%f0, %f0, %f0, -1
	dxtra	%f0, %f0, %f0, 16
	dxtra	%f0, %f0, %f2, 0
	dxtra	%f0, %f2, %f0, 0
	dxtra	%f2, %f0, %f0, 0

#CHECK: error: instruction requires: transactional-execution
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

#CHECK: error: invalid register pair
#CHECK: kmctr	%r1, %r2, %r4
#CHECK: error: invalid register pair
#CHECK: kmctr	%r2, %r1, %r4
#CHECK: error: invalid register pair
#CHECK: kmctr	%r2, %r4, %r1

	kmctr	%r1, %r2, %r4
	kmctr	%r2, %r1, %r4
	kmctr	%r2, %r4, %r1

#CHECK: error: invalid register pair
#CHECK: kmf	%r1, %r2
#CHECK: error: invalid register pair
#CHECK: kmf	%r2, %r1

	kmf	%r1, %r2
	kmf	%r2, %r1

#CHECK: error: invalid register pair
#CHECK: kmo	%r1, %r2
#CHECK: error: invalid register pair
#CHECK: kmo	%r2, %r1

	kmo	%r1, %r2
	kmo	%r2, %r1

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

#CHECK: error: invalid register pair
#CHECK: lpd	%r1, 0, 0
#CHECK: error: invalid use of indexed addressing
#CHECK: lpd	%r2, 160(%r1,%r15), 160(%r15)
#CHECK: error: invalid operand
#CHECK: lpd	%r2, -1(%r1), 160(%r15)
#CHECK: error: invalid operand
#CHECK: lpd	%r2, 4096(%r1), 160(%r15)
#CHECK: error: invalid operand
#CHECK: lpd	%r2, 0(%r1), -1(%r15)
#CHECK: error: invalid operand
#CHECK: lpd	%r2, 0(%r1), 4096(%r15)

	lpd	%r1, 0, 0
	lpd	%r2, 160(%r1,%r15), 160(%r15)
	lpd	%r2, -1(%r1), 160(%r15)
	lpd	%r2, 4096(%r1), 160(%r15)
	lpd	%r2, 0(%r1), -1(%r15)
	lpd	%r2, 0(%r1), 4096(%r15)

#CHECK: error: invalid register pair
#CHECK: lpdg	%r1, 0, 0
#CHECK: error: invalid use of indexed addressing
#CHECK: lpdg	%r2, 160(%r1,%r15), 160(%r15)
#CHECK: error: invalid operand
#CHECK: lpdg	%r2, -1(%r1), 160(%r15)
#CHECK: error: invalid operand
#CHECK: lpdg	%r2, 4096(%r1), 160(%r15)
#CHECK: error: invalid operand
#CHECK: lpdg	%r2, 0(%r1), -1(%r15)
#CHECK: error: invalid operand
#CHECK: lpdg	%r2, 0(%r1), 4096(%r15)

	lpdg	%r1, 0, 0
	lpdg	%r2, 160(%r1,%r15), 160(%r15)
	lpdg	%r2, -1(%r1), 160(%r15)
	lpdg	%r2, 4096(%r1), 160(%r15)
	lpdg	%r2, 0(%r1), -1(%r15)
	lpdg	%r2, 0(%r1), 4096(%r15)

#CHECK: error: invalid operand
#CHECK: mdtra	%f0, %f0, %f0, -1
#CHECK: error: invalid operand
#CHECK: mdtra	%f0, %f0, %f0, 16

	mdtra	%f0, %f0, %f0, -1
	mdtra	%f0, %f0, %f0, 16

#CHECK: error: invalid operand
#CHECK: mxtra	%f0, %f0, %f0, -1
#CHECK: error: invalid operand
#CHECK: mxtra	%f0, %f0, %f0, 16
#CHECK: error: invalid register pair
#CHECK: mxtra	%f0, %f0, %f2, 0
#CHECK: error: invalid register pair
#CHECK: mxtra	%f0, %f2, %f0, 0
#CHECK: error: invalid register pair
#CHECK: mxtra	%f2, %f0, %f0, 0

	mxtra	%f0, %f0, %f0, -1
	mxtra	%f0, %f0, %f0, 16
	mxtra	%f0, %f0, %f2, 0
	mxtra	%f0, %f2, %f0, 0
	mxtra	%f2, %f0, %f0, 0

#CHECK: error: instruction requires: execution-hint
#CHECK: niai	0, 0

	niai	0, 0

#CHECK: error: instruction requires: transactional-execution
#CHECK: ntstg	%r0, 524287(%r1,%r15)

	ntstg	%r0, 524287(%r1,%r15)

#CHECK: error: instruction requires: processor-assist
#CHECK: ppa	%r4, %r6, 7

	ppa	%r4, %r6, 7

#CHECK: error: instruction requires: miscellaneous-extensions
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
#CHECK: sdtra	%f0, %f0, %f0, -1
#CHECK: error: invalid operand
#CHECK: sdtra	%f0, %f0, %f0, 16

	sdtra	%f0, %f0, %f0, -1
	sdtra	%f0, %f0, %f0, 16

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
#CHECK: srnmb	-1
#CHECK: error: invalid operand
#CHECK: srnmb	4096
#CHECK: error: invalid use of indexed addressing
#CHECK: srnmb	0(%r1,%r2)

	srnmb	-1
	srnmb	4096
	srnmb	0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: stch	%r0, -524289
#CHECK: error: invalid operand
#CHECK: stch	%r0, 524288

	stch	%r0, -524289
	stch	%r0, 524288

#CHECK: error: invalid operand
#CHECK: stfh	%r0, -524289
#CHECK: error: invalid operand
#CHECK: stfh	%r0, 524288

	stfh	%r0, -524289
	stfh	%r0, 524288

#CHECK: error: invalid operand
#CHECK: sthh	%r0, -524289
#CHECK: error: invalid operand
#CHECK: sthh	%r0, 524288

	sthh	%r0, -524289
	sthh	%r0, 524288

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

#CHECK: error: invalid operand
#CHECK: sxtra	%f0, %f0, %f0, -1
#CHECK: error: invalid operand
#CHECK: sxtra	%f0, %f0, %f0, 16
#CHECK: error: invalid register pair
#CHECK: sxtra	%f0, %f0, %f2, 0
#CHECK: error: invalid register pair
#CHECK: sxtra	%f0, %f2, %f0, 0
#CHECK: error: invalid register pair
#CHECK: sxtra	%f2, %f0, %f0, 0

	sxtra	%f0, %f0, %f0, -1
	sxtra	%f0, %f0, %f0, 16
	sxtra	%f0, %f0, %f2, 0
	sxtra	%f0, %f2, %f0, 0
	sxtra	%f2, %f0, %f0, 0

#CHECK: error: instruction requires: transactional-execution
#CHECK: tabort	4095(%r1)

	tabort	4095(%r1)

#CHECK: error: instruction requires: transactional-execution
#CHECK: tbegin	4095(%r1), 42

	tbegin	4095(%r1), 42

#CHECK: error: instruction requires: transactional-execution
#CHECK: tbeginc	4095(%r1), 42

	tbeginc	4095(%r1), 42

#CHECK: error: instruction requires: transactional-execution
#CHECK: tend

	tend

