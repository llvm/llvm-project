# For z10 only.
# RUN: not llvm-mc -triple s390x-linux-gnu -mcpu=z10 < %s 2> %t
# RUN: FileCheck < %t %s
# RUN: not llvm-mc -triple s390x-linux-gnu -mcpu=arch8 < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: a	%r0, -1
#CHECK: error: invalid operand
#CHECK: a	%r0, 4096

	a	%r0, -1
	a	%r0, 4096

#CHECK: error: invalid operand
#CHECK: ad	%f0, -1
#CHECK: error: invalid operand
#CHECK: ad	%f0, 4096

	ad	%f0, -1
	ad	%f0, 4096

#CHECK: error: invalid operand
#CHECK: adb	%f0, -1
#CHECK: error: invalid operand
#CHECK: adb	%f0, 4096

	adb	%f0, -1
	adb	%f0, 4096

#CHECK: error: instruction requires: fp-extension
#CHECK: adtra	%f0, %f0, %f0, 0

	adtra	%f0, %f0, %f0, 0

#CHECK: error: invalid operand
#CHECK: ae	%f0, -1
#CHECK: error: invalid operand
#CHECK: ae	%f0, 4096

	ae	%f0, -1
	ae	%f0, 4096

#CHECK: error: invalid operand
#CHECK: aeb	%f0, -1
#CHECK: error: invalid operand
#CHECK: aeb	%f0, 4096

	aeb	%f0, -1
	aeb	%f0, 4096

#CHECK: error: invalid operand
#CHECK: afi	%r0, (-1 << 31) - 1
#CHECK: error: invalid operand
#CHECK: afi	%r0, (1 << 31)

	afi	%r0, (-1 << 31) - 1
	afi	%r0, (1 << 31)

#CHECK: error: invalid operand
#CHECK: ag	%r0, -524289
#CHECK: error: invalid operand
#CHECK: ag	%r0, 524288

	ag	%r0, -524289
	ag	%r0, 524288

#CHECK: error: invalid operand
#CHECK: agf	%r0, -524289
#CHECK: error: invalid operand
#CHECK: agf	%r0, 524288

	agf	%r0, -524289
	agf	%r0, 524288

#CHECK: error: invalid operand
#CHECK: agfi	%r0, (-1 << 31) - 1
#CHECK: error: invalid operand
#CHECK: agfi	%r0, (1 << 31)

	agfi	%r0, (-1 << 31) - 1
	agfi	%r0, (1 << 31)

#CHECK: error: invalid operand
#CHECK: aghi	%r0, -32769
#CHECK: error: invalid operand
#CHECK: aghi	%r0, 32768
#CHECK: error: invalid operand
#CHECK: aghi	%r0, foo

	aghi	%r0, -32769
	aghi	%r0, 32768
	aghi	%r0, foo

#CHECK: error: instruction requires: distinct-ops
#CHECK: aghik	%r1, %r2, 3

	aghik	%r1, %r2, 3

#CHECK: error: instruction requires: distinct-ops
#CHECK: agrk	%r2,%r3,%r4

	agrk	%r2,%r3,%r4

#CHECK: error: invalid operand
#CHECK: agsi	-524289, 0
#CHECK: error: invalid operand
#CHECK: agsi	524288, 0
#CHECK: error: invalid use of indexed addressing
#CHECK: agsi	0(%r1,%r2), 0
#CHECK: error: invalid operand
#CHECK: agsi	0, -129
#CHECK: error: invalid operand
#CHECK: agsi	0, 128

	agsi	-524289, 0
	agsi	524288, 0
	agsi	0(%r1,%r2), 0
	agsi	0, -129
	agsi	0, 128

#CHECK: error: invalid operand
#CHECK: ah	%r0, -1
#CHECK: error: invalid operand
#CHECK: ah	%r0, 4096

	ah	%r0, -1
	ah	%r0, 4096

#CHECK: error: instruction requires: high-word
#CHECK: ahhhr	%r0, %r0, %r0

	ahhhr	%r0, %r0, %r0

#CHECK: error: instruction requires: high-word
#CHECK: ahhlr	%r0, %r0, %r0

	ahhlr	%r0, %r0, %r0

#CHECK: error: invalid operand
#CHECK: ahi	%r0, -32769
#CHECK: error: invalid operand
#CHECK: ahi	%r0, 32768
#CHECK: error: invalid operand
#CHECK: ahi	%r0, foo

	ahi	%r0, -32769
	ahi	%r0, 32768
	ahi	%r0, foo

#CHECK: error: instruction requires: distinct-ops
#CHECK: ahik	%r1, %r2, 3

	ahik	%r1, %r2, 3

#CHECK: error: invalid operand
#CHECK: ahy	%r0, -524289
#CHECK: error: invalid operand
#CHECK: ahy	%r0, 524288

	ahy	%r0, -524289
	ahy	%r0, 524288

#CHECK: error: instruction requires: high-word
#CHECK: aih	%r0, 0

	aih	%r0, 0

#CHECK: error: invalid operand
#CHECK: al	%r0, -1
#CHECK: error: invalid operand
#CHECK: al	%r0, 4096

	al	%r0, -1
	al	%r0, 4096

#CHECK: error: invalid operand
#CHECK: alc	%r0, -524289
#CHECK: error: invalid operand
#CHECK: alc	%r0, 524288

	alc	%r0, -524289
	alc	%r0, 524288

#CHECK: error: invalid operand
#CHECK: alcg	%r0, -524289
#CHECK: error: invalid operand
#CHECK: alcg	%r0, 524288

	alcg	%r0, -524289
	alcg	%r0, 524288

#CHECK: error: invalid operand
#CHECK: alfi	%r0, -1
#CHECK: error: invalid operand
#CHECK: alfi	%r0, (1 << 32)

	alfi	%r0, -1
	alfi	%r0, (1 << 32)

#CHECK: error: invalid operand
#CHECK: alg	%r0, -524289
#CHECK: error: invalid operand
#CHECK: alg	%r0, 524288

	alg	%r0, -524289
	alg	%r0, 524288

#CHECK: error: invalid operand
#CHECK: algf	%r0, -524289
#CHECK: error: invalid operand
#CHECK: algf	%r0, 524288

	algf	%r0, -524289
	algf	%r0, 524288

#CHECK: error: invalid operand
#CHECK: algfi	%r0, -1
#CHECK: error: invalid operand
#CHECK: algfi	%r0, (1 << 32)

	algfi	%r0, -1
	algfi	%r0, (1 << 32)

#CHECK: error: instruction requires: distinct-ops
#CHECK: alghsik	%r1, %r2, 3

	alghsik	%r1, %r2, 3

#CHECK: error: instruction requires: distinct-ops
#CHECK: algrk	%r2,%r3,%r4

	algrk	%r2,%r3,%r4

#CHECK: error: instruction requires: high-word
#CHECK: alhhhr	%r0, %r0, %r0

	alhhhr	%r0, %r0, %r0

#CHECK: error: instruction requires: high-word
#CHECK: alhhlr	%r0, %r0, %r0

	alhhlr	%r0, %r0, %r0

#CHECK: error: instruction requires: distinct-ops
#CHECK: alhsik	%r1, %r2, 3

	alhsik	%r1, %r2, 3

#CHECK: error: instruction requires: distinct-ops
#CHECK: alrk	%r2,%r3,%r4

	alrk	%r2,%r3,%r4

#CHECK: error: invalid operand
#CHECK: algsi	-524289, 0
#CHECK: error: invalid operand
#CHECK: algsi	524288, 0
#CHECK: error: invalid use of indexed addressing
#CHECK: algsi	0(%r1,%r2), 0
#CHECK: error: invalid operand
#CHECK: algsi	0, -129
#CHECK: error: invalid operand
#CHECK: algsi	0, 128

	algsi	-524289, 0
	algsi	524288, 0
	algsi	0(%r1,%r2), 0
	algsi	0, -129
	algsi	0, 128

#CHECK: error: invalid operand
#CHECK: alsi	-524289, 0
#CHECK: error: invalid operand
#CHECK: alsi	524288, 0
#CHECK: error: invalid use of indexed addressing
#CHECK: alsi	0(%r1,%r2), 0
#CHECK: error: invalid operand
#CHECK: alsi	0, -129
#CHECK: error: invalid operand
#CHECK: alsi	0, 128

	alsi	-524289, 0
	alsi	524288, 0
	alsi	0(%r1,%r2), 0
	alsi	0, -129
	alsi	0, 128

#CHECK: error: instruction requires: high-word
#CHECK: alsih	%r0, 0

	alsih	%r0, 0

#CHECK: error: instruction requires: high-word
#CHECK: alsihn	%r0, 0

	alsihn	%r0, 0

#CHECK: error: invalid operand
#CHECK: aly	%r0, -524289
#CHECK: error: invalid operand
#CHECK: aly	%r0, 524288

	aly	%r0, -524289
	aly	%r0, 524288

#CHECK: error: missing length in address
#CHECK: ap	0, 0(1)
#CHECK: error: missing length in address
#CHECK: ap	0(1), 0
#CHECK: error: missing length in address
#CHECK: ap	0(%r1), 0(1,%r1)
#CHECK: error: missing length in address
#CHECK: ap	0(1,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: ap	0(0,%r1), 0(1,%r1)
#CHECK: error: invalid operand
#CHECK: ap	0(1,%r1), 0(0,%r1)
#CHECK: error: invalid operand
#CHECK: ap	0(17,%r1), 0(1,%r1)
#CHECK: error: invalid operand
#CHECK: ap	0(1,%r1), 0(17,%r1)
#CHECK: error: invalid operand
#CHECK: ap	-1(1,%r1), 0(1,%r1)
#CHECK: error: invalid operand
#CHECK: ap	4096(1,%r1), 0(1,%r1)
#CHECK: error: invalid operand
#CHECK: ap	0(1,%r1), -1(1,%r1)
#CHECK: error: invalid operand
#CHECK: ap	0(1,%r1), 4096(1,%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: ap	0(%r1,%r2), 0(1,%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: ap	0(1,%r2), 0(%r1,%r2)
#CHECK: error: unknown token in expression
#CHECK: ap	0(-), 0(1)

	ap	0, 0(1)
	ap	0(1), 0
	ap	0(%r1), 0(1,%r1)
	ap	0(1,%r1), 0(%r1)
	ap	0(0,%r1), 0(1,%r1)
	ap	0(1,%r1), 0(0,%r1)
	ap	0(17,%r1), 0(1,%r1)
	ap	0(1,%r1), 0(17,%r1)
	ap	-1(1,%r1), 0(1,%r1)
	ap	4096(1,%r1), 0(1,%r1)
	ap	0(1,%r1), -1(1,%r1)
	ap	0(1,%r1), 4096(1,%r1)
	ap	0(%r1,%r2), 0(1,%r1)
	ap	0(1,%r2), 0(%r1,%r2)
	ap	0(-), 0(1)

#CHECK: error: instruction requires: distinct-ops
#CHECK: ark	%r2,%r3,%r4

	ark	%r2,%r3,%r4

#CHECK: error: invalid operand
#CHECK: asi	-524289, 0
#CHECK: error: invalid operand
#CHECK: asi	524288, 0
#CHECK: error: invalid use of indexed addressing
#CHECK: asi	0(%r1,%r2), 0
#CHECK: error: invalid operand
#CHECK: asi	0, -129
#CHECK: error: invalid operand
#CHECK: asi	0, 128

	asi	-524289, 0
	asi	524288, 0
	asi	0(%r1,%r2), 0
	asi	0, -129
	asi	0, 128

#CHECK: error: invalid operand
#CHECK: au	%f0, -1
#CHECK: error: invalid operand
#CHECK: au	%f0, 4096

	au	%f0, -1
	au	%f0, 4096

#CHECK: error: invalid operand
#CHECK: aw	%f0, -1
#CHECK: error: invalid operand
#CHECK: aw	%f0, 4096

	aw	%f0, -1
	aw	%f0, 4096

#CHECK: error: invalid register pair
#CHECK: axbr	%f0, %f2
#CHECK: error: invalid register pair
#CHECK: axbr	%f2, %f0

	axbr	%f0, %f2
	axbr	%f2, %f0

#CHECK: error: invalid register pair
#CHECK: axr	%f0, %f2
#CHECK: error: invalid register pair
#CHECK: axr	%f2, %f0

	axr	%f0, %f2
	axr	%f2, %f0

#CHECK: error: invalid register pair
#CHECK: axtr	%f0, %f0, %f2
#CHECK: error: invalid register pair
#CHECK: axtr	%f0, %f2, %f0
#CHECK: error: invalid register pair
#CHECK: axtr	%f2, %f0, %f0

	axtr	%f0, %f0, %f2
	axtr	%f0, %f2, %f0
	axtr	%f2, %f0, %f0

#CHECK: error: instruction requires: fp-extension
#CHECK: axtra	%f0, %f0, %f0, 0

	axtra	%f0, %f0, %f0, 0

#CHECK: error: invalid operand
#CHECK: ay	%r0, -524289
#CHECK: error: invalid operand
#CHECK: ay	%r0, 524288

	ay	%r0, -524289
	ay	%r0, 524288

#CHECK: error: invalid operand
#CHECK: bal	%r0, -1
#CHECK: error: invalid operand
#CHECK: bal	%r0, 4096

	bal	%r0, -1
	bal	%r0, 4096

#CHECK: error: invalid operand
#CHECK: bas	%r0, -1
#CHECK: error: invalid operand
#CHECK: bas	%r0, 4096

	bas	%r0, -1
	bas	%r0, 4096

#CHECK: error: invalid operand
#CHECK: bc	-1, 0(%r1)
#CHECK: error: invalid operand
#CHECK: bc	16, 0(%r1)
#CHECK: error: invalid operand
#CHECK: bc	0, -1
#CHECK: error: invalid operand
#CHECK: bc	0, 4096

	bc	-1, 0(%r1)
	bc	16, 0(%r1)
	bc	0, -1
	bc	0, 4096

#CHECK: error: invalid operand
#CHECK: bcr	-1, %r1
#CHECK: error: invalid operand
#CHECK: bcr	16, %r1

	bcr	-1, %r1
	bcr	16, %r1

#CHECK: error: invalid operand
#CHECK: bct	%r0, -1
#CHECK: error: invalid operand
#CHECK: bct	%r0, 4096

	bct	%r0, -1
	bct	%r0, 4096

#CHECK: error: invalid operand
#CHECK: bctg	%r0, -524289
#CHECK: error: invalid operand
#CHECK: bctg	%r0, 524288

	bctg	%r0, -524289
	bctg	%r0, 524288

#CHECK: error: offset out of range
#CHECK: bras	%r0, -0x100002
#CHECK: error: offset out of range
#CHECK: bras	%r0, -1
#CHECK: error: offset out of range
#CHECK: bras	%r0, 1
#CHECK: error: offset out of range
#CHECK: bras	%r0, 0x10000
#CHECK: error: offset out of range
#CHECK: jas	%r0, -0x100002
#CHECK: error: offset out of range
#CHECK: jas	%r0, -1
#CHECK: error: offset out of range
#CHECK: jas	%r0, 1
#CHECK: error: offset out of range
#CHECK: jas	%r0, 0x10000

	bras	%r0, -0x100002
	bras	%r0, -1
	bras	%r0, 1
	bras	%r0, 0x10000
	jas	%r0, -0x100002
	jas	%r0, -1
	jas	%r0, 1
	jas	%r0, 0x10000

#CHECK: error: offset out of range
#CHECK: brasl	%r0, -0x1000000002
#CHECK: error: offset out of range
#CHECK: brasl	%r0, -1
#CHECK: error: offset out of range
#CHECK: brasl	%r0, 1
#CHECK: error: offset out of range
#CHECK: brasl	%r0, 0x100000000
#CHECK: error: offset out of range
#CHECK: jasl	%r0, -0x1000000002
#CHECK: error: offset out of range
#CHECK: jasl	%r0, -1
#CHECK: error: offset out of range
#CHECK: jasl	%r0, 1
#CHECK: error: offset out of range
#CHECK: jasl	%r0, 0x100000000

	brasl	%r0, -0x1000000002
	brasl	%r0, -1
	brasl	%r0, 1
	brasl	%r0, 0x100000000
	jasl	%r0, -0x1000000002
	jasl	%r0, -1
	jasl	%r0, 1
	jasl	%r0, 0x100000000

#CHECK: error: offset out of range
#CHECK: brc	0, -0x100002
#CHECK: error: offset out of range
#CHECK: brc	0, -1
#CHECK: error: offset out of range
#CHECK: brc	0, 1
#CHECK: error: offset out of range
#CHECK: brc	0, 0x10000
#CHECK: error: offset out of range
#CHECK: jnop -0x100002
#CHECK: error: offset out of range
#CHECK: jnop    -1
#CHECK: error: offset out of range
#CHECK: jnop    1
#CHECK: error: offset out of range
#CHECK: jnop    0x10000

	brc	0, -0x100002
	brc	0, -1
	brc	0, 1
	brc	0, 0x10000
	jnop	-0x100002
	jnop	-1
	jnop	1
	jnop	0x10000

#CHECK: error: invalid operand
#CHECK: brc	foo, bar
#CHECK: error: invalid operand
#CHECK: brc	-1, bar
#CHECK: error: invalid operand
#CHECK: brc	16, bar

	brc	foo, bar
	brc	-1, bar
	brc	16, bar

#CHECK: error: offset out of range
#CHECK: brcl	0, -0x1000000002
#CHECK: error: offset out of range
#CHECK: brcl	0, -1
#CHECK: error: offset out of range
#CHECK: brcl	0, 1
#CHECK: error: offset out of range
#CHECK: brcl	0, 0x100000000
#CHECK: error: offset out of range
#CHECK: jgnop	-0x1000000002
#CHECK: error: offset out of range
#CHECK: jgnop	-1
#CHECK: error: offset out of range
#CHECK: jgnop	1
#CHECK: error: offset out of range
#CHECK: jgnop	0x100000000

	brcl	0, -0x1000000002
	brcl	0, -1
	brcl	0, 1
	brcl	0, 0x100000000
	jgnop	-0x1000000002
	jgnop	-1
	jgnop	1
	jgnop	0x100000000

#CHECK: error: invalid operand
#CHECK: brcl	foo, bar
#CHECK: error: invalid operand
#CHECK: brcl	-1, bar
#CHECK: error: invalid operand
#CHECK: brcl	16, bar

	brcl	foo, bar
	brcl	-1, bar
	brcl	16, bar

#CHECK: error: offset out of range
#CHECK: brct	%r0, -0x100002
#CHECK: error: offset out of range
#CHECK: brct	%r0, -1
#CHECK: error: offset out of range
#CHECK: brct	%r0, 1
#CHECK: error: offset out of range
#CHECK: brct	%r0, 0x10000

	brct	%r0, -0x100002
	brct	%r0, -1
	brct	%r0, 1
	brct	%r0, 0x10000

#CHECK: error: offset out of range
#CHECK: brctg	%r0, -0x100002
#CHECK: error: offset out of range
#CHECK: brctg	%r0, -1
#CHECK: error: offset out of range
#CHECK: brctg	%r0, 1
#CHECK: error: offset out of range
#CHECK: brctg	%r0, 0x10000

	brctg	%r0, -0x100002
	brctg	%r0, -1
	brctg	%r0, 1
	brctg	%r0, 0x10000

#CHECK: error: instruction requires: high-word
#CHECK: brcth	%r0, 0

	brcth	%r0, 0

#CHECK: error: offset out of range
#CHECK: brxh	%r0, %r2, -0x100002
#CHECK: error: offset out of range
#CHECK: brxh	%r0, %r2, -1
#CHECK: error: offset out of range
#CHECK: brxh	%r0, %r2, 1
#CHECK: error: offset out of range
#CHECK: brxh	%r0, %r2, 0x10000
#CHECK: error: offset out of range
#CHECK: jxh	%r0, %r2, -0x100002
#CHECK: error: offset out of range
#CHECK: jxh	%r0, %r2, -1
#CHECK: error: offset out of range
#CHECK: jxh	%r0, %r2, 1
#CHECK: error: offset out of range
#CHECK: jxh	%r0, %r2, 0x10000

	brxh	%r0, %r2, -0x100002
	brxh	%r0, %r2, -1
	brxh	%r0, %r2, 1
	brxh	%r0, %r2, 0x10000
	jxh	%r0, %r2, -0x100002
	jxh	%r0, %r2, -1
	jxh	%r0, %r2, 1
	jxh	%r0, %r2, 0x10000

#CHECK: error: offset out of range
#CHECK: brxhg	%r0, %r2, -0x100002
#CHECK: error: offset out of range
#CHECK: brxhg	%r0, %r2, -1
#CHECK: error: offset out of range
#CHECK: brxhg	%r0, %r2, 1
#CHECK: error: offset out of range
#CHECK: brxhg	%r0, %r2, 0x10000
#CHECK: error: offset out of range
#CHECK: jxhg	%r0, %r2, -0x100002
#CHECK: error: offset out of range
#CHECK: jxhg	%r0, %r2, -1
#CHECK: error: offset out of range
#CHECK: jxhg	%r0, %r2, 1
#CHECK: error: offset out of range
#CHECK: jxhg	%r0, %r2, 0x10000

	brxhg	%r0, %r2, -0x100002
	brxhg	%r0, %r2, -1
	brxhg	%r0, %r2, 1
	brxhg	%r0, %r2, 0x10000
	jxhg	%r0, %r2, -0x100002
	jxhg	%r0, %r2, -1
	jxhg	%r0, %r2, 1
	jxhg	%r0, %r2, 0x10000

#CHECK: error: offset out of range
#CHECK: brxle	%r0, %r2, -0x100002
#CHECK: error: offset out of range
#CHECK: brxle	%r0, %r2, -1
#CHECK: error: offset out of range
#CHECK: brxle	%r0, %r2, 1
#CHECK: error: offset out of range
#CHECK: brxle	%r0, %r2, 0x10000
#CHECK: error: offset out of range
#CHECK: jxle	%r0, %r2, -0x100002
#CHECK: error: offset out of range
#CHECK: jxle	%r0, %r2, -1
#CHECK: error: offset out of range
#CHECK: jxle	%r0, %r2, 1
#CHECK: error: offset out of range
#CHECK: jxle	%r0, %r2, 0x10000

	brxle	%r0, %r2, -0x100002
	brxle	%r0, %r2, -1
	brxle	%r0, %r2, 1
	brxle	%r0, %r2, 0x10000
	jxle	%r0, %r2, -0x100002
	jxle	%r0, %r2, -1
	jxle	%r0, %r2, 1
	jxle	%r0, %r2, 0x10000

#CHECK: error: offset out of range
#CHECK: brxlg	%r0, %r2, -0x100002
#CHECK: error: offset out of range
#CHECK: brxlg	%r0, %r2, -1
#CHECK: error: offset out of range
#CHECK: brxlg	%r0, %r2, 1
#CHECK: error: offset out of range
#CHECK: brxlg	%r0, %r2, 0x10000
#CHECK: error: offset out of range
#CHECK: jxleg	%r0, %r2, -0x100002
#CHECK: error: offset out of range
#CHECK: jxleg	%r0, %r2, -1
#CHECK: error: offset out of range
#CHECK: jxleg	%r0, %r2, 1
#CHECK: error: offset out of range
#CHECK: jxleg	%r0, %r2, 0x10000

	brxlg	%r0, %r2, -0x100002
	brxlg	%r0, %r2, -1
	brxlg	%r0, %r2, 1
	brxlg	%r0, %r2, 0x10000
	jxleg	%r0, %r2, -0x100002
	jxleg	%r0, %r2, -1
	jxleg	%r0, %r2, 1
	jxleg	%r0, %r2, 0x10000

#CHECK: error: invalid operand
#CHECK: bxh	%r0, %r0, 4096
#CHECK: error: invalid use of indexed addressing
#CHECK: bxh	%r0, %r0, 0(%r1,%r2)

	bxh	%r0, %r0, 4096
	bxh	%r0, %r0, 0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: bxhg	%r0, %r0, -524289
#CHECK: error: invalid operand
#CHECK: bxhg	%r0, %r0, 524288
#CHECK: error: invalid use of indexed addressing
#CHECK: bxhg	%r0, %r0, 0(%r1,%r2)

	bxhg	%r0, %r0, -524289
	bxhg	%r0, %r0, 524288
	bxhg	%r0, %r0, 0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: bxle	%r0, %r0, 4096
#CHECK: error: invalid use of indexed addressing
#CHECK: bxle	%r0, %r0, 0(%r1,%r2)

	bxle	%r0, %r0, 4096
	bxle	%r0, %r0, 0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: bxleg	%r0, %r0, -524289
#CHECK: error: invalid operand
#CHECK: bxleg	%r0, %r0, 524288
#CHECK: error: invalid use of indexed addressing
#CHECK: bxleg	%r0, %r0, 0(%r1,%r2)

	bxleg	%r0, %r0, -524289
	bxleg	%r0, %r0, 524288
	bxleg	%r0, %r0, 0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: c	%r0, -1
#CHECK: error: invalid operand
#CHECK: c	%r0, 4096

	c	%r0, -1
	c	%r0, 4096

#CHECK: error: invalid operand
#CHECK: cd	%f0, -1
#CHECK: error: invalid operand
#CHECK: cd	%f0, 4096

	cd	%f0, -1
	cd	%f0, 4096

#CHECK: error: invalid operand
#CHECK: cdb	%f0, -1
#CHECK: error: invalid operand
#CHECK: cdb	%f0, 4096

	cdb	%f0, -1
	cdb	%f0, 4096

#CHECK: error: instruction requires: fp-extension
#CHECK: cdfbra	%f0, 0, %r0, 0

	cdfbra	%f0, 0, %r0, 0

#CHECK: error: instruction requires: fp-extension
#CHECK: cdftr	%f0, 0, %r0, 0

	cdftr	%f0, 0, %r0, 0

#CHECK: error: instruction requires: fp-extension
#CHECK: cdgbra	%f0, 0, %r0, 0

	cdgbra	%f0, 0, %r0, 0

#CHECK: error: instruction requires: fp-extension
#CHECK: cdgtra	%f0, 0, %r0, 0

	cdgtra	%f0, 0, %r0, 0

#CHECK: error: instruction requires: fp-extension
#CHECK: cdlfbr	%f0, 0, %r0, 0

	cdlfbr	%f0, 0, %r0, 0

#CHECK: error: instruction requires: fp-extension
#CHECK: cdlftr	%f0, 0, %r0, 0

	cdlftr	%f0, 0, %r0, 0

#CHECK: error: instruction requires: fp-extension
#CHECK: cdlgbr	%f0, 0, %r0, 0

	cdlgbr	%f0, 0, %r0, 0

#CHECK: error: instruction requires: fp-extension
#CHECK: cdlgtr	%f0, 0, %r0, 0

	cdlgtr	%f0, 0, %r0, 0

#CHECK: error: invalid register pair
#CHECK: cds	%r1, %r0, 0
#CHECK: error: invalid register pair
#CHECK: cds	%r0, %r1, 0
#CHECK: error: invalid operand
#CHECK: cds	%r0, %r0, -1
#CHECK: error: invalid operand
#CHECK: cds	%r0, %r0, 4096
#CHECK: error: invalid use of indexed addressing
#CHECK: cds	%r0, %r0, 0(%r1,%r2)

	cds	%r1, %r0, 0
	cds	%r0, %r1, 0
	cds	%r0, %r0, -1
	cds	%r0, %r0, 4096
	cds	%r0, %r0, 0(%r1,%r2)

#CHECK: error: invalid register pair
#CHECK: cdsg	%r1, %r0, 0
#CHECK: error: invalid register pair
#CHECK: cdsg	%r0, %r1, 0
#CHECK: error: invalid operand
#CHECK: cdsg	%r0, %r0, -524289
#CHECK: error: invalid operand
#CHECK: cdsg	%r0, %r0, 524288
#CHECK: error: invalid use of indexed addressing
#CHECK: cdsg	%r0, %r0, 0(%r1,%r2)

	cdsg	%r1, %r0, 0
	cdsg	%r0, %r1, 0
	cdsg	%r0, %r0, -524289
	cdsg	%r0, %r0, 524288
	cdsg	%r0, %r0, 0(%r1,%r2)

#CHECK: error: invalid register pair
#CHECK: cdsy	%r1, %r0, 0
#CHECK: error: invalid register pair
#CHECK: cdsy	%r0, %r1, 0
#CHECK: error: invalid operand
#CHECK: cdsy	%r0, %r0, -524289
#CHECK: error: invalid operand
#CHECK: cdsy	%r0, %r0, 524288
#CHECK: error: invalid use of indexed addressing
#CHECK: cdsy	%r0, %r0, 0(%r1,%r2)

	cdsy	%r1, %r0, 0
	cdsy	%r0, %r1, 0
	cdsy	%r0, %r0, -524289
	cdsy	%r0, %r0, 524288
	cdsy	%r0, %r0, 0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: ce	%f0, -1
#CHECK: error: invalid operand
#CHECK: ce	%f0, 4096

	ce	%f0, -1
	ce	%f0, 4096

#CHECK: error: invalid operand
#CHECK: ceb	%f0, -1
#CHECK: error: invalid operand
#CHECK: ceb	%f0, 4096

	ceb	%f0, -1
	ceb	%f0, 4096

#CHECK: error: instruction requires: fp-extension
#CHECK: cefbra	%f0, 0, %r0, 0

	cefbra	%f0, 0, %r0, 0

#CHECK: error: instruction requires: fp-extension
#CHECK: cegbra	%f0, 0, %r0, 0

	cegbra	%f0, 0, %r0, 0

#CHECK: error: instruction requires: fp-extension
#CHECK: celfbr	%f0, 0, %r0, 0

	celfbr	%f0, 0, %r0, 0

#CHECK: error: instruction requires: fp-extension
#CHECK: celgbr	%f0, 0, %r0, 0

	celgbr	%f0, 0, %r0, 0

#CHECK: error: invalid register pair
#CHECK: cextr	%f0, %f2
#CHECK: error: invalid register pair
#CHECK: cextr	%f2, %f0

	cextr	%f0, %f2
	cextr	%f2, %f0

#CHECK: error: invalid operand
#CHECK: cfc	-1
#CHECK: error: invalid operand
#CHECK: cfc	4096
#CHECK: error: invalid use of indexed addressing
#CHECK: cfc	0(%r1,%r2)

	cfc	-1
	cfc	4096
	cfc	0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: cfdbr	%r0, -1, %f0
#CHECK: error: invalid operand
#CHECK: cfdbr	%r0, 16, %f0

	cfdbr	%r0, -1, %f0
	cfdbr	%r0, 16, %f0

#CHECK: error: instruction requires: fp-extension
#CHECK: cfdbra	%r0, 0, %f0, 0

	cfdbra	%r0, 0, %f0, 0

#CHECK: error: instruction requires: fp-extension
#CHECK: cfdtr	%r0, 0, %f0, 0

	cfdtr	%r0, 0, %f0, 0

#CHECK: error: invalid operand
#CHECK: cfebr	%r0, -1, %f0
#CHECK: error: invalid operand
#CHECK: cfebr	%r0, 16, %f0

	cfebr	%r0, -1, %f0
	cfebr	%r0, 16, %f0

#CHECK: error: instruction requires: fp-extension
#CHECK: cfebra	%r0, 0, %f0, 0

	cfebra	%r0, 0, %f0, 0

#CHECK: error: invalid operand
#CHECK: cfi	%r0, (-1 << 31) - 1
#CHECK: error: invalid operand
#CHECK: cfi	%r0, (1 << 31)

	cfi	%r0, (-1 << 31) - 1
	cfi	%r0, (1 << 31)

#CHECK: error: invalid operand
#CHECK: cfxbr	%r0, -1, %f0
#CHECK: error: invalid operand
#CHECK: cfxbr	%r0, 16, %f0
#CHECK: error: invalid register pair
#CHECK: cfxbr	%r0, 0, %f2

	cfxbr	%r0, -1, %f0
	cfxbr	%r0, 16, %f0
	cfxbr	%r0, 0, %f2

#CHECK: error: instruction requires: fp-extension
#CHECK: cfxbra	%r0, 0, %f0, 0

	cfxbra	%r0, 0, %f0, 0

#CHECK: error: instruction requires: fp-extension
#CHECK: cfxtr	%r0, 0, %f0, 0

	cfxtr	%r0, 0, %f0, 0

#CHECK: error: invalid operand
#CHECK: cfxr	%r0, -1, %f0
#CHECK: error: invalid operand
#CHECK: cfxr	%r0, 16, %f0
#CHECK: error: invalid register pair
#CHECK: cfxr	%r0, 0, %f2

	cfxr	%r0, -1, %f0
	cfxr	%r0, 16, %f0
	cfxr	%r0, 0, %f2

#CHECK: error: invalid operand
#CHECK: cg	%r0, -524289
#CHECK: error: invalid operand
#CHECK: cg	%r0, 524288

	cg	%r0, -524289
	cg	%r0, 524288

#CHECK: error: invalid operand
#CHECK: cgdbr	%r0, -1, %f0
#CHECK: error: invalid operand
#CHECK: cgdbr	%r0, 16, %f0

	cgdbr	%r0, -1, %f0
	cgdbr	%r0, 16, %f0

#CHECK: error: instruction requires: fp-extension
#CHECK: cgdbra	%r0, 0, %f0, 0

	cgdbra	%r0, 0, %f0, 0

#CHECK: error: invalid operand
#CHECK: cgdtr	%r0, -1, %f0
#CHECK: error: invalid operand
#CHECK: cgdtr	%r0, 16, %f0

	cgdtr	%r0, -1, %f0
	cgdtr	%r0, 16, %f0

#CHECK: error: instruction requires: fp-extension
#CHECK: cgdtra	%r0, 0, %f0, 0

	cgdtra	%r0, 0, %f0, 0

#CHECK: error: invalid operand
#CHECK: cgebr	%r0, -1, %f0
#CHECK: error: invalid operand
#CHECK: cgebr	%r0, 16, %f0

	cgebr	%r0, -1, %f0
	cgebr	%r0, 16, %f0

#CHECK: error: instruction requires: fp-extension
#CHECK: cgebra	%r0, 0, %f0, 0

	cgebra	%r0, 0, %f0, 0

#CHECK: error: invalid operand
#CHECK: cgf	%r0, -524289
#CHECK: error: invalid operand
#CHECK: cgf	%r0, 524288

	cgf	%r0, -524289
	cgf	%r0, 524288

#CHECK: error: invalid operand
#CHECK: cgfi	%r0, (-1 << 31) - 1
#CHECK: error: invalid operand
#CHECK: cgfi	%r0, (1 << 31)

	cgfi	%r0, (-1 << 31) - 1
	cgfi	%r0, (1 << 31)

#CHECK: error: offset out of range
#CHECK: cgfrl	%r0, -0x1000000002
#CHECK: error: offset out of range
#CHECK: cgfrl	%r0, -1
#CHECK: error: offset out of range
#CHECK: cgfrl	%r0, 1
#CHECK: error: offset out of range
#CHECK: cgfrl	%r0, 0x100000000

	cgfrl	%r0, -0x1000000002
	cgfrl	%r0, -1
	cgfrl	%r0, 1
	cgfrl	%r0, 0x100000000

#CHECK: error: invalid operand
#CHECK: cgh	%r0, -524289
#CHECK: error: invalid operand
#CHECK: cgh	%r0, 524288

	cgh	%r0, -524289
	cgh	%r0, 524288

#CHECK: error: invalid operand
#CHECK: cghi	%r0, -32769
#CHECK: error: invalid operand
#CHECK: cghi	%r0, 32768
#CHECK: error: invalid operand
#CHECK: cghi	%r0, foo

	cghi	%r0, -32769
	cghi	%r0, 32768
	cghi	%r0, foo

#CHECK: error: offset out of range
#CHECK: cghrl	%r0, -0x1000000002
#CHECK: error: offset out of range
#CHECK: cghrl	%r0, -1
#CHECK: error: offset out of range
#CHECK: cghrl	%r0, 1
#CHECK: error: offset out of range
#CHECK: cghrl	%r0, 0x100000000

	cghrl	%r0, -0x1000000002
	cghrl	%r0, -1
	cghrl	%r0, 1
	cghrl	%r0, 0x100000000

#CHECK: error: invalid operand
#CHECK: cghsi	-1, 0
#CHECK: error: invalid operand
#CHECK: cghsi	4096, 0
#CHECK: error: invalid use of indexed addressing
#CHECK: cghsi	0(%r1,%r2), 0
#CHECK: error: invalid operand
#CHECK: cghsi	0, -32769
#CHECK: error: invalid operand
#CHECK: cghsi	0, 32768

	cghsi	-1, 0
	cghsi	4096, 0
	cghsi	0(%r1,%r2), 0
	cghsi	0, -32769
	cghsi	0, 32768

#CHECK: error: invalid operand
#CHECK: cgij	%r0, -129, 0, 0
#CHECK: error: invalid operand
#CHECK: cgij	%r0, 128, 0, 0

	cgij	%r0, -129, 0, 0
	cgij	%r0, 128, 0, 0

#CHECK: error: offset out of range
#CHECK: cgij	%r0, 0, 0, -0x100002
#CHECK: error: offset out of range
#CHECK: cgij	%r0, 0, 0, -1
#CHECK: error: offset out of range
#CHECK: cgij	%r0, 0, 0, 1
#CHECK: error: offset out of range
#CHECK: cgij	%r0, 0, 0, 0x10000

	cgij	%r0, 0, 0, -0x100002
	cgij	%r0, 0, 0, -1
	cgij	%r0, 0, 0, 1
	cgij	%r0, 0, 0, 0x10000

#CHECK: error: invalid instruction
#CHECK:	cgijno	%r0, 0, 0, 0
#CHECK: error: invalid instruction
#CHECK:	cgijo	%r0, 0, 0, 0

	cgijno	%r0, 0, 0, 0
	cgijo	%r0, 0, 0, 0

#CHECK: error: invalid operand
#CHECK: cgit     %r0, -32769
#CHECK: error: invalid operand
#CHECK: cgit     %r0, 32768
#CHECK: error: invalid instruction
#CHECK: cgitno   %r0, 0
#CHECK: error: invalid instruction
#CHECK: cgito    %r0, 0

        cgit     %r0, -32769
        cgit     %r0, 32768
        cgitno   %r0, 0
        cgito    %r0, 0

#CHECK: error: offset out of range
#CHECK: cgrj	%r0, %r0, 0, -0x100002
#CHECK: error: offset out of range
#CHECK: cgrj	%r0, %r0, 0, -1
#CHECK: error: offset out of range
#CHECK: cgrj	%r0, %r0, 0, 1
#CHECK: error: offset out of range
#CHECK: cgrj	%r0, %r0, 0, 0x10000

	cgrj	%r0, %r0, 0, -0x100002
	cgrj	%r0, %r0, 0, -1
	cgrj	%r0, %r0, 0, 1
	cgrj	%r0, %r0, 0, 0x10000

#CHECK: error: invalid instruction
#CHECK:	cgrjno	%r0, %r0, 0, 0
#CHECK: error: invalid instruction
#CHECK:	cgrjo	%r0, %r0, 0, 0

	cgrjno	%r0, %r0, 0, 0
	cgrjo	%r0, %r0, 0, 0

#CHECK: error: offset out of range
#CHECK: cgrl	%r0, -0x1000000002
#CHECK: error: offset out of range
#CHECK: cgrl	%r0, -1
#CHECK: error: offset out of range
#CHECK: cgrl	%r0, 1
#CHECK: error: offset out of range
#CHECK: cgrl	%r0, 0x100000000

	cgrl	%r0, -0x1000000002
	cgrl	%r0, -1
	cgrl	%r0, 1
	cgrl	%r0, 0x100000000

#CHECK: error: invalid instruction
#CHECK: cgrtno   %r0, %r0
#CHECK: error: invalid instruction
#CHECK: cgrto    %r0, %r0

        cgrtno   %r0, %r0
        cgrto    %r0, %r0

#CHECK: error: invalid operand
#CHECK: cgxbr	%r0, -1, %f0
#CHECK: error: invalid operand
#CHECK: cgxbr	%r0, 16, %f0
#CHECK: error: invalid register pair
#CHECK: cgxbr	%r0, 0, %f2

	cgxbr	%r0, -1, %f0
	cgxbr	%r0, 16, %f0
	cgxbr	%r0, 0, %f2

#CHECK: error: instruction requires: fp-extension
#CHECK: cgxbra	%r0, 0, %f0, 0

	cgxbra	%r0, 0, %f0, 0

#CHECK: error: invalid operand
#CHECK: cgxtr	%r0, -1, %f0
#CHECK: error: invalid operand
#CHECK: cgxtr	%r0, 16, %f0
#CHECK: error: invalid register pair
#CHECK: cgxtr	%r0, 0, %f2

	cgxtr	%r0, -1, %f0
	cgxtr	%r0, 16, %f0
	cgxtr	%r0, 0, %f2

#CHECK: error: instruction requires: fp-extension
#CHECK: cgxtra	%r0, 0, %f0, 0

	cgxtra	%r0, 0, %f0, 0

#CHECK: error: invalid operand
#CHECK: cgxr	%r0, -1, %f0
#CHECK: error: invalid operand
#CHECK: cgxr	%r0, 16, %f0
#CHECK: error: invalid register pair
#CHECK: cgxr	%r0, 0, %f2

	cgxr	%r0, -1, %f0
	cgxr	%r0, 16, %f0
	cgxr	%r0, 0, %f2

#CHECK: error: invalid operand
#CHECK: ch	%r0, -1
#CHECK: error: invalid operand
#CHECK: ch	%r0, 4096

	ch	%r0, -1
	ch	%r0, 4096

#CHECK: error: instruction requires: high-word
#CHECK: chf	%r0, 0

	chf	%r0, 0

#CHECK: error: instruction requires: high-word
#CHECK: chhr	%r0, %r0

	chhr	%r0, %r0

#CHECK: error: invalid operand
#CHECK: chhsi	-1, 0
#CHECK: error: invalid operand
#CHECK: chhsi	4096, 0
#CHECK: error: invalid use of indexed addressing
#CHECK: chhsi	0(%r1,%r2), 0
#CHECK: error: invalid operand
#CHECK: chhsi	0, -32769
#CHECK: error: invalid operand
#CHECK: chhsi	0, 32768

	chhsi	-1, 0
	chhsi	4096, 0
	chhsi	0(%r1,%r2), 0
	chhsi	0, -32769
	chhsi	0, 32768

#CHECK: error: invalid operand
#CHECK: chi	%r0, -32769
#CHECK: error: invalid operand
#CHECK: chi	%r0, 32768
#CHECK: error: invalid operand
#CHECK: chi	%r0, foo

	chi	%r0, -32769
	chi	%r0, 32768
	chi	%r0, foo

#CHECK: error: instruction requires: high-word
#CHECK: chlr	%r0, %r0

	chlr	%r0, %r0

#CHECK: error: offset out of range
#CHECK: chrl	%r0, -0x1000000002
#CHECK: error: offset out of range
#CHECK: chrl	%r0, -1
#CHECK: error: offset out of range
#CHECK: chrl	%r0, 1
#CHECK: error: offset out of range
#CHECK: chrl	%r0, 0x100000000

	chrl	%r0, -0x1000000002
	chrl	%r0, -1
	chrl	%r0, 1
	chrl	%r0, 0x100000000

#CHECK: error: invalid operand
#CHECK: chsi	-1, 0
#CHECK: error: invalid operand
#CHECK: chsi	4096, 0
#CHECK: error: invalid use of indexed addressing
#CHECK: chsi	0(%r1,%r2), 0
#CHECK: error: invalid operand
#CHECK: chsi	0, -32769
#CHECK: error: invalid operand
#CHECK: chsi	0, 32768

	chsi	-1, 0
	chsi	4096, 0
	chsi	0(%r1,%r2), 0
	chsi	0, -32769
	chsi	0, 32768

#CHECK: error: invalid operand
#CHECK: chy	%r0, -524289
#CHECK: error: invalid operand
#CHECK: chy	%r0, 524288

	chy	%r0, -524289
	chy	%r0, 524288

#CHECK: error: instruction requires: high-word
#CHECK: cih	%r0, 0

	cih	%r0, 0

#CHECK: error: invalid operand
#CHECK: cij	%r0, -129, 0, 0
#CHECK: error: invalid operand
#CHECK: cij	%r0, 128, 0, 0

	cij	%r0, -129, 0, 0
	cij	%r0, 128, 0, 0

#CHECK: error: offset out of range
#CHECK: cij	%r0, 0, 0, -0x100002
#CHECK: error: offset out of range
#CHECK: cij	%r0, 0, 0, -1
#CHECK: error: offset out of range
#CHECK: cij	%r0, 0, 0, 1
#CHECK: error: offset out of range
#CHECK: cij	%r0, 0, 0, 0x10000

	cij	%r0, 0, 0, -0x100002
	cij	%r0, 0, 0, -1
	cij	%r0, 0, 0, 1
	cij	%r0, 0, 0, 0x10000

#CHECK: error: invalid instruction
#CHECK:	cijno	%r0, 0, 0, 0
#CHECK: error: invalid instruction
#CHECK:	cijo	%r0, 0, 0, 0

	cijno	%r0, 0, 0, 0
	cijo	%r0, 0, 0, 0

#CHECK: error: invalid operand
#CHECK: cit     %r0, -32769
#CHECK: error: invalid operand
#CHECK: cit     %r0, 32768
#CHECK: error: invalid instruction
#CHECK: citno   %r0, 0
#CHECK: error: invalid instruction
#CHECK: cito    %r0, 0

        cit     %r0, -32769
        cit     %r0, 32768
        citno   %r0, 0
        cito    %r0, 0

#CHECK: error: invalid register pair
#CHECK: cksm	%r0, %r1

	cksm	%r0, %r1

#CHECK: error: invalid operand
#CHECK: cl	%r0, -1
#CHECK: error: invalid operand
#CHECK: cl	%r0, 4096

	cl	%r0, -1
	cl	%r0, 4096

#CHECK: error: missing length in address
#CHECK: clc	0, 0
#CHECK: error: missing length in address
#CHECK: clc	0(%r1), 0(%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: clc	0(1,%r1), 0(2,%r1)
#CHECK: error: invalid operand
#CHECK: clc	0(0,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: clc	0(257,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: clc	-1(1,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: clc	4096(1,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: clc	0(1,%r1), -1(%r1)
#CHECK: error: invalid operand
#CHECK: clc	0(1,%r1), 4096(%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: clc	0(%r1,%r2), 0(%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: clc	0(1,%r2), 0(%r1,%r2)
#CHECK: error: unknown token in expression
#CHECK: clc	0(-), 0

	clc	0, 0
	clc	0(%r1), 0(%r1)
	clc	0(1,%r1), 0(2,%r1)
	clc	0(0,%r1), 0(%r1)
	clc	0(257,%r1), 0(%r1)
	clc	-1(1,%r1), 0(%r1)
	clc	4096(1,%r1), 0(%r1)
	clc	0(1,%r1), -1(%r1)
	clc	0(1,%r1), 4096(%r1)
	clc	0(%r1,%r2), 0(%r1)
	clc	0(1,%r2), 0(%r1,%r2)
	clc	0(-), 0

#CHECK: error: invalid register pair
#CHECK: clcl	%r1, %r0
#CHECK: error: invalid register pair
#CHECK: clcl	%r0, %r1

	clcl	%r1, %r0
	clcl	%r0, %r1

#CHECK: error: invalid register pair
#CHECK: clcle	%r1, %r0
#CHECK: error: invalid register pair
#CHECK: clcle	%r0, %r1
#CHECK: error: invalid operand
#CHECK: clcle	%r0, %r0, -1
#CHECK: error: invalid operand
#CHECK: clcle	%r0, %r0, 4096

	clcle	%r1, %r0, 0
	clcle	%r0, %r1, 0
	clcle	%r0, %r0, -1
	clcle	%r0, %r0, 4096

#CHECK: error: invalid register pair
#CHECK: clclu	%r1, %r0
#CHECK: error: invalid register pair
#CHECK: clclu	%r0, %r1
#CHECK: error: invalid operand
#CHECK: clclu	%r0, %r0, -524289
#CHECK: error: invalid operand
#CHECK: clclu	%r0, %r0, 524288

	clclu	%r1, %r0, 0
	clclu	%r0, %r1, 0
	clclu	%r0, %r0, -524289
	clclu	%r0, %r0, 524288

#CHECK: error: instruction requires: fp-extension
#CHECK: clfdbr	%r0, 0, %f0, 0

	clfdbr	%r0, 0, %f0, 0

#CHECK: error: instruction requires: fp-extension
#CHECK: clfdtr	%r0, 0, %f0, 0

	clfdtr	%r0, 0, %f0, 0

#CHECK: error: instruction requires: fp-extension
#CHECK: clfebr	%r0, 0, %f0, 0

	clfebr	%r0, 0, %f0, 0

#CHECK: error: invalid operand
#CHECK: clfhsi	-1, 0
#CHECK: error: invalid operand
#CHECK: clfhsi	4096, 0
#CHECK: error: invalid use of indexed addressing
#CHECK: clfhsi	0(%r1,%r2), 0
#CHECK: error: invalid operand
#CHECK: clfhsi	0, -1
#CHECK: error: invalid operand
#CHECK: clfhsi	0, 65536

	clfhsi	-1, 0
	clfhsi	4096, 0
	clfhsi	0(%r1,%r2), 0
	clfhsi	0, -1
	clfhsi	0, 65536

#CHECK: error: invalid operand
#CHECK: clfi	%r0, -1
#CHECK: error: invalid operand
#CHECK: clfi	%r0, (1 << 32)

	clfi	%r0, -1
	clfi	%r0, (1 << 32)

#CHECK: error: invalid operand
#CHECK: clfit   %r0, -1
#CHECK: error: invalid operand
#CHECK: clfit   %r0, 65536
#CHECK: error: invalid instruction
#CHECK: clfitno %r0, 0
#CHECK: error: invalid instruction
#CHECK: clfito  %r0, 0

        clfit   %r0, -1
        clfit   %r0, 65536
        clfitno %r0, 0
        clfito  %r0, 0

#CHECK: error: instruction requires: fp-extension
#CHECK: clfxbr	%r0, 0, %f0, 0

	clfxbr	%r0, 0, %f0, 0

#CHECK: error: instruction requires: fp-extension
#CHECK: clfxtr	%r0, 0, %f0, 0

	clfxtr	%r0, 0, %f0, 0

#CHECK: error: invalid operand
#CHECK: clg	%r0, -524289
#CHECK: error: invalid operand
#CHECK: clg	%r0, 524288

	clg	%r0, -524289
	clg	%r0, 524288

#CHECK: error: instruction requires: fp-extension
#CHECK: clgdbr	%r0, 0, %f0, 0

	clgdbr	%r0, 0, %f0, 0

#CHECK: error: instruction requires: fp-extension
#CHECK: clgdtr	%r0, 0, %f0, 0

	clgdtr	%r0, 0, %f0, 0

#CHECK: error: instruction requires: fp-extension
#CHECK: clgebr	%r0, 0, %f0, 0

	clgebr	%r0, 0, %f0, 0

#CHECK: error: invalid operand
#CHECK: clgf	%r0, -524289
#CHECK: error: invalid operand
#CHECK: clgf	%r0, 524288

	clgf	%r0, -524289
	clgf	%r0, 524288

#CHECK: error: invalid operand
#CHECK: clgfi	%r0, -1
#CHECK: error: invalid operand
#CHECK: clgfi	%r0, (1 << 32)

	clgfi	%r0, -1
	clgfi	%r0, (1 << 32)

#CHECK: error: offset out of range
#CHECK: clgfrl	%r0, -0x1000000002
#CHECK: error: offset out of range
#CHECK: clgfrl	%r0, -1
#CHECK: error: offset out of range
#CHECK: clgfrl	%r0, 1
#CHECK: error: offset out of range
#CHECK: clgfrl	%r0, 0x100000000

	clgfrl	%r0, -0x1000000002
	clgfrl	%r0, -1
	clgfrl	%r0, 1
	clgfrl	%r0, 0x100000000

#CHECK: error: offset out of range
#CHECK: clghrl	%r0, -0x1000000002
#CHECK: error: offset out of range
#CHECK: clghrl	%r0, -1
#CHECK: error: offset out of range
#CHECK: clghrl	%r0, 1
#CHECK: error: offset out of range
#CHECK: clghrl	%r0, 0x100000000

	clghrl	%r0, -0x1000000002
	clghrl	%r0, -1
	clghrl	%r0, 1
	clghrl	%r0, 0x100000000

#CHECK: error: invalid operand
#CHECK: clghsi	-1, 0
#CHECK: error: invalid operand
#CHECK: clghsi	4096, 0
#CHECK: error: invalid use of indexed addressing
#CHECK: clghsi	0(%r1,%r2), 0
#CHECK: error: invalid operand
#CHECK: clghsi	0, -1
#CHECK: error: invalid operand
#CHECK: clghsi	0, 65536

	clghsi	-1, 0
	clghsi	4096, 0
	clghsi	0(%r1,%r2), 0
	clghsi	0, -1
	clghsi	0, 65536

#CHECK: error: invalid operand
#CHECK: clgij	%r0, -1, 0, 0
#CHECK: error: invalid operand
#CHECK: clgij	%r0, 256, 0, 0

	clgij	%r0, -1, 0, 0
	clgij	%r0, 256, 0, 0

#CHECK: error: offset out of range
#CHECK: clgij	%r0, 0, 0, -0x100002
#CHECK: error: offset out of range
#CHECK: clgij	%r0, 0, 0, -1
#CHECK: error: offset out of range
#CHECK: clgij	%r0, 0, 0, 1
#CHECK: error: offset out of range
#CHECK: clgij	%r0, 0, 0, 0x10000

	clgij	%r0, 0, 0, -0x100002
	clgij	%r0, 0, 0, -1
	clgij	%r0, 0, 0, 1
	clgij	%r0, 0, 0, 0x10000

#CHECK: error: invalid instruction
#CHECK:	clgijno	%r0, 0, 0, 0
#CHECK: error: invalid instruction
#CHECK:	clgijo	%r0, 0, 0, 0

	clgijno	%r0, 0, 0, 0
	clgijo	%r0, 0, 0, 0

#CHECK: error: invalid operand
#CHECK: clgit   %r0, -1
#CHECK: error: invalid operand
#CHECK: clgit   %r0, 65536
#CHECK: error: invalid instruction
#CHECK: clgitno %r0, 0
#CHECK: error: invalid instruction
#CHECK: clgito  %r0, 0

        clgit   %r0, -1
        clgit   %r0, 65536
        clgitno %r0, 0
        clgito  %r0, 0

#CHECK: error: offset out of range
#CHECK: clgrj	%r0, %r0, 0, -0x100002
#CHECK: error: offset out of range
#CHECK: clgrj	%r0, %r0, 0, -1
#CHECK: error: offset out of range
#CHECK: clgrj	%r0, %r0, 0, 1
#CHECK: error: offset out of range
#CHECK: clgrj	%r0, %r0, 0, 0x10000

	clgrj	%r0, %r0, 0, -0x100002
	clgrj	%r0, %r0, 0, -1
	clgrj	%r0, %r0, 0, 1
	clgrj	%r0, %r0, 0, 0x10000

#CHECK: error: offset out of range
#CHECK: clgrl	%r0, -0x1000000002
#CHECK: error: offset out of range
#CHECK: clgrl	%r0, -1
#CHECK: error: offset out of range
#CHECK: clgrl	%r0, 1
#CHECK: error: offset out of range
#CHECK: clgrl	%r0, 0x100000000

	clgrl	%r0, -0x1000000002
	clgrl	%r0, -1
	clgrl	%r0, 1
	clgrl	%r0, 0x100000000

#CHECK: error: invalid instruction
#CHECK: clgrtno   %r0, %r0
#CHECK: error: invalid instruction
#CHECK: clgrto    %r0, %r0

        clgrtno   %r0, %r0
        clgrto    %r0, %r0

#CHECK: error: instruction requires: fp-extension
#CHECK: clgxbr	%r0, 0, %f0, 0

	clgxbr	%r0, 0, %f0, 0

#CHECK: error: instruction requires: fp-extension
#CHECK: clgxtr	%r0, 0, %f0, 0

	clgxtr	%r0, 0, %f0, 0

#CHECK: error: instruction requires: high-word
#CHECK: clhf	%r0, 0

	clhf	%r0, 0

#CHECK: error: instruction requires: high-word
#CHECK: clhhr	%r0, %r0

	clhhr	%r0, %r0

#CHECK: error: invalid operand
#CHECK: clhhsi	-1, 0
#CHECK: error: invalid operand
#CHECK: clhhsi	4096, 0
#CHECK: error: invalid use of indexed addressing
#CHECK: clhhsi	0(%r1,%r2), 0
#CHECK: error: invalid operand
#CHECK: clhhsi	0, -1
#CHECK: error: invalid operand
#CHECK: clhhsi	0, 65536

	clhhsi	-1, 0
	clhhsi	4096, 0
	clhhsi	0(%r1,%r2), 0
	clhhsi	0, -1
	clhhsi	0, 65536

#CHECK: error: instruction requires: high-word
#CHECK: clhlr	%r0, %r0

	clhlr	%r0, %r0

#CHECK: error: offset out of range
#CHECK: clhrl	%r0, -0x1000000002
#CHECK: error: offset out of range
#CHECK: clhrl	%r0, -1
#CHECK: error: offset out of range
#CHECK: clhrl	%r0, 1
#CHECK: error: offset out of range
#CHECK: clhrl	%r0, 0x100000000

	clhrl	%r0, -0x1000000002
	clhrl	%r0, -1
	clhrl	%r0, 1
	clhrl	%r0, 0x100000000

#CHECK: error: invalid operand
#CHECK: cli	-1, 0
#CHECK: error: invalid operand
#CHECK: cli	4096, 0
#CHECK: error: invalid use of indexed addressing
#CHECK: cli	0(%r1,%r2), 0
#CHECK: error: invalid operand
#CHECK: cli	0, -1
#CHECK: error: invalid operand
#CHECK: cli	0, 256

	cli	-1, 0
	cli	4096, 0
	cli	0(%r1,%r2), 0
	cli	0, -1
	cli	0, 256

#CHECK: error: instruction requires: high-word
#CHECK: clih	%r0, 0

	clih	%r0, 0

#CHECK: error: invalid operand
#CHECK: clij	%r0, -1, 0, 0
#CHECK: error: invalid operand
#CHECK: clij	%r0, 256, 0, 0

	clij	%r0, -1, 0, 0
	clij	%r0, 256, 0, 0

#CHECK: error: offset out of range
#CHECK: clij	%r0, 0, 0, -0x100002
#CHECK: error: offset out of range
#CHECK: clij	%r0, 0, 0, -1
#CHECK: error: offset out of range
#CHECK: clij	%r0, 0, 0, 1
#CHECK: error: offset out of range
#CHECK: clij	%r0, 0, 0, 0x10000

	clij	%r0, 0, 0, -0x100002
	clij	%r0, 0, 0, -1
	clij	%r0, 0, 0, 1
	clij	%r0, 0, 0, 0x10000

#CHECK: error: invalid instruction
#CHECK:	clijno	%r0, 0, 0, 0
#CHECK: error: invalid instruction
#CHECK:	clijo	%r0, 0, 0, 0

	clijno	%r0, 0, 0, 0
	clijo	%r0, 0, 0, 0

#CHECK: error: invalid operand
#CHECK: cliy	-524289, 0
#CHECK: error: invalid operand
#CHECK: cliy	524288, 0
#CHECK: error: invalid use of indexed addressing
#CHECK: cliy	0(%r1,%r2), 0
#CHECK: error: invalid operand
#CHECK: cliy	0, -1
#CHECK: error: invalid operand
#CHECK: cliy	0, 256

	cliy	-524289, 0
	cliy	524288, 0
	cliy	0(%r1,%r2), 0
	cliy	0, -1
	cliy	0, 256

#CHECK: error: invalid operand
#CHECK: clm	%r0, 0, -1
#CHECK: error: invalid operand
#CHECK: clm	%r0, 0, 4096
#CHECK: error: invalid operand
#CHECK: clm	%r0, -1, 0
#CHECK: error: invalid operand
#CHECK: clm	%r0, 16, 0

	clm	%r0, 0, -1
	clm	%r0, 0, 4096
	clm	%r0, -1, 0
	clm	%r0, 16, 0

#CHECK: error: invalid operand
#CHECK: clmh	%r0, 0, -524289
#CHECK: error: invalid operand
#CHECK: clmh	%r0, 0, 524288
#CHECK: error: invalid operand
#CHECK: clmh	%r0, -1, 0
#CHECK: error: invalid operand
#CHECK: clmh	%r0, 16, 0

	clmh	%r0, 0, -524289
	clmh	%r0, 0, 524288
	clmh	%r0, -1, 0
	clmh	%r0, 16, 0

#CHECK: error: invalid operand
#CHECK: clmy	%r0, 0, -524289
#CHECK: error: invalid operand
#CHECK: clmy	%r0, 0, 524288
#CHECK: error: invalid operand
#CHECK: clmy	%r0, -1, 0
#CHECK: error: invalid operand
#CHECK: clmy	%r0, 16, 0

	clmy	%r0, 0, -524289
	clmy	%r0, 0, 524288
	clmy	%r0, -1, 0
	clmy	%r0, 16, 0

#CHECK: error: offset out of range
#CHECK: clrj	%r0, %r0, 0, -0x100002
#CHECK: error: offset out of range
#CHECK: clrj	%r0, %r0, 0, -1
#CHECK: error: offset out of range
#CHECK: clrj	%r0, %r0, 0, 1
#CHECK: error: offset out of range
#CHECK: clrj	%r0, %r0, 0, 0x10000

	clrj	%r0, %r0, 0, -0x100002
	clrj	%r0, %r0, 0, -1
	clrj	%r0, %r0, 0, 1
	clrj	%r0, %r0, 0, 0x10000

#CHECK: error: invalid instruction
#CHECK:	clrjno	%r0, %r0, 0, 0
#CHECK: error: invalid instruction
#CHECK:	clrjo	%r0, %r0, 0, 0

	clrjno	%r0, %r0, 0, 0
	clrjo	%r0, %r0, 0, 0

#CHECK: error: offset out of range
#CHECK: clrl	%r0, -0x1000000002
#CHECK: error: offset out of range
#CHECK: clrl	%r0, -1
#CHECK: error: offset out of range
#CHECK: clrl	%r0, 1
#CHECK: error: offset out of range
#CHECK: clrl	%r0, 0x100000000

	clrl	%r0, -0x1000000002
	clrl	%r0, -1
	clrl	%r0, 1
	clrl	%r0, 0x100000000

#CHECK: error: invalid instruction
#CHECK: clrtno   %r0, %r0
#CHECK: error: invalid instruction
#CHECK: clrto    %r0, %r0

        clrtno   %r0, %r0
        clrto    %r0, %r0

#CHECK: error: invalid operand
#CHECK: cly	%r0, -524289
#CHECK: error: invalid operand
#CHECK: cly	%r0, 524288

	cly	%r0, -524289
	cly	%r0, 524288

#CHECK: error: invalid register pair
#CHECK: cmpsc	%r1, %r0
#CHECK: error: invalid register pair
#CHECK: cmpsc	%r0, %r1

	cmpsc	%r1, %r0
	cmpsc	%r0, %r1

#CHECK: error: missing length in address
#CHECK: cp	0, 0(1)
#CHECK: error: missing length in address
#CHECK: cp	0(1), 0
#CHECK: error: missing length in address
#CHECK: cp	0(%r1), 0(1,%r1)
#CHECK: error: missing length in address
#CHECK: cp	0(1,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: cp	0(0,%r1), 0(1,%r1)
#CHECK: error: invalid operand
#CHECK: cp	0(1,%r1), 0(0,%r1)
#CHECK: error: invalid operand
#CHECK: cp	0(17,%r1), 0(1,%r1)
#CHECK: error: invalid operand
#CHECK: cp	0(1,%r1), 0(17,%r1)
#CHECK: error: invalid operand
#CHECK: cp	-1(1,%r1), 0(1,%r1)
#CHECK: error: invalid operand
#CHECK: cp	4096(1,%r1), 0(1,%r1)
#CHECK: error: invalid operand
#CHECK: cp	0(1,%r1), -1(1,%r1)
#CHECK: error: invalid operand
#CHECK: cp	0(1,%r1), 4096(1,%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: cp	0(%r1,%r2), 0(1,%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: cp	0(1,%r2), 0(%r1,%r2)
#CHECK: error: unknown token in expression
#CHECK: cp	0(-), 0(1)

	cp	0, 0(1)
	cp	0(1), 0
	cp	0(%r1), 0(1,%r1)
	cp	0(1,%r1), 0(%r1)
	cp	0(0,%r1), 0(1,%r1)
	cp	0(1,%r1), 0(0,%r1)
	cp	0(17,%r1), 0(1,%r1)
	cp	0(1,%r1), 0(17,%r1)
	cp	-1(1,%r1), 0(1,%r1)
	cp	4096(1,%r1), 0(1,%r1)
	cp	0(1,%r1), -1(1,%r1)
	cp	0(1,%r1), 4096(1,%r1)
	cp	0(%r1,%r2), 0(1,%r1)
	cp	0(1,%r2), 0(%r1,%r2)
	cp	0(-), 0(1)

#CHECK: error: offset out of range
#CHECK: crj	%r0, %r0, 0, -0x100002
#CHECK: error: offset out of range
#CHECK: crj	%r0, %r0, 0, -1
#CHECK: error: offset out of range
#CHECK: crj	%r0, %r0, 0, 1
#CHECK: error: offset out of range
#CHECK: crj	%r0, %r0, 0, 0x10000

	crj	%r0, %r0, 0, -0x100002
	crj	%r0, %r0, 0, -1
	crj	%r0, %r0, 0, 1
	crj	%r0, %r0, 0, 0x10000

#CHECK: error: invalid instruction
#CHECK:	crjno	%r0, %r0, 0, 0
#CHECK: error: invalid instruction
#CHECK:	crjo	%r0, %r0, 0, 0

	crjno	%r0, %r0, 0, 0
	crjo	%r0, %r0, 0, 0

#CHECK: error: offset out of range
#CHECK: crl	%r0, -0x1000000002
#CHECK: error: offset out of range
#CHECK: crl	%r0, -1
#CHECK: error: offset out of range
#CHECK: crl	%r0, 1
#CHECK: error: offset out of range
#CHECK: crl	%r0, 0x100000000

	crl	%r0, -0x1000000002
	crl	%r0, -1
	crl	%r0, 1
	crl	%r0, 0x100000000

#CHECK: error: invalid instruction
#CHECK: crtno   %r0, %r0
#CHECK: error: invalid instruction
#CHECK: crto    %r0, %r0

        crtno   %r0, %r0
        crto    %r0, %r0

#CHECK: error: invalid operand
#CHECK: cs	%r0, %r0, -1
#CHECK: error: invalid operand
#CHECK: cs	%r0, %r0, 4096
#CHECK: error: invalid use of indexed addressing
#CHECK: cs	%r0, %r0, 0(%r1,%r2)

	cs	%r0, %r0, -1
	cs	%r0, %r0, 4096
	cs	%r0, %r0, 0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: csdtr	%r0, %f0, -1
#CHECK: error: invalid operand
#CHECK: csdtr	%r0, %f0, 16

	csdtr	%r0, %f0, -1
	csdtr	%r0, %f0, 16

#CHECK: error: invalid operand
#CHECK: csg	%r0, %r0, -524289
#CHECK: error: invalid operand
#CHECK: csg	%r0, %r0, 524288
#CHECK: error: invalid use of indexed addressing
#CHECK: csg	%r0, %r0, 0(%r1,%r2)

	csg	%r0, %r0, -524289
	csg	%r0, %r0, 524288
	csg	%r0, %r0, 0(%r1,%r2)

#CHECK: error: invalid register pair
#CHECK: csp	%r1, %r0

	csp	%r1, %r0

#CHECK: error: invalid register pair
#CHECK: cspg	%r1, %r0

	cspg	%r1, %r0

#CHECK: error: invalid use of indexed addressing
#CHECK: csst	160(%r1,%r15), 160(%r15), %r2
#CHECK: error: invalid operand
#CHECK: csst	-1(%r1), 160(%r15), %r2
#CHECK: error: invalid operand
#CHECK: csst	4096(%r1), 160(%r15), %r2
#CHECK: error: invalid operand
#CHECK: csst	0(%r1), -1(%r15), %r2
#CHECK: error: invalid operand
#CHECK: csst	0(%r1), 4096(%r15), %r2

        csst	160(%r1,%r15), 160(%r15), %r2
        csst	-1(%r1), 160(%r15), %r2
        csst	4096(%r1), 160(%r15), %r2
        csst	0(%r1), -1(%r15), %r2
        csst	0(%r1), 4096(%r15), %r2

#CHECK: error: invalid operand
#CHECK: csxtr	%r0, %f0, -1
#CHECK: error: invalid operand
#CHECK: csxtr	%r0, %f0, 16
#CHECK: error: invalid register pair
#CHECK: csxtr	%r0, %f2, 0
#CHECK: error: invalid register pair
#CHECK: csxtr	%r1, %f0, 0

	csxtr	%r0, %f0, -1
	csxtr	%r0, %f0, 16
	csxtr	%r0, %f2, 0
	csxtr	%r1, %f0, 0

#CHECK: error: invalid operand
#CHECK: csy	%r0, %r0, -524289
#CHECK: error: invalid operand
#CHECK: csy	%r0, %r0, 524288
#CHECK: error: invalid use of indexed addressing
#CHECK: csy	%r0, %r0, 0(%r1,%r2)

	csy	%r0, %r0, -524289
	csy	%r0, %r0, 524288
	csy	%r0, %r0, 0(%r1,%r2)

#CHECK: error: invalid register pair
#CHECK: cu12	%r1, %r0
#CHECK: error: invalid register pair
#CHECK: cu12	%r0, %r1
#CHECK: error: invalid operand
#CHECK: cu12	%r2, %r4, -1
#CHECK: error: invalid operand
#CHECK: cu12	%r2, %r4, 16

	cu12	%r1, %r0
	cu12	%r0, %r1
	cu12	%r2, %r4, -1
	cu12	%r2, %r4, 16

#CHECK: error: invalid register pair
#CHECK: cu14	%r1, %r0
#CHECK: error: invalid register pair
#CHECK: cu14	%r0, %r1
#CHECK: error: invalid operand
#CHECK: cu14	%r2, %r4, -1
#CHECK: error: invalid operand
#CHECK: cu14	%r2, %r4, 16

	cu14	%r1, %r0
	cu14	%r0, %r1
	cu14	%r2, %r4, -1
	cu14	%r2, %r4, 16

#CHECK: error: invalid register pair
#CHECK: cu21	%r1, %r0
#CHECK: error: invalid register pair
#CHECK: cu21	%r0, %r1
#CHECK: error: invalid operand
#CHECK: cu21	%r2, %r4, -1
#CHECK: error: invalid operand
#CHECK: cu21	%r2, %r4, 16

	cu21	%r1, %r0
	cu21	%r0, %r1
	cu21	%r2, %r4, -1
	cu21	%r2, %r4, 16

#CHECK: error: invalid register pair
#CHECK: cu24	%r1, %r0
#CHECK: error: invalid register pair
#CHECK: cu24	%r0, %r1
#CHECK: error: invalid operand
#CHECK: cu24	%r2, %r4, -1
#CHECK: error: invalid operand
#CHECK: cu24	%r2, %r4, 16

	cu24	%r1, %r0
	cu24	%r0, %r1
	cu24	%r2, %r4, -1
	cu24	%r2, %r4, 16

#CHECK: error: invalid register pair
#CHECK: cu41	%r1, %r0
#CHECK: error: invalid register pair
#CHECK: cu41	%r0, %r1

	cu41	%r1, %r0
	cu41	%r0, %r1

#CHECK: error: invalid register pair
#CHECK: cu42	%r1, %r0
#CHECK: error: invalid register pair
#CHECK: cu42	%r0, %r1

	cu42	%r1, %r0
	cu42	%r0, %r1

#CHECK: error: invalid register pair
#CHECK: cuse	%r1, %r0
#CHECK: error: invalid register pair
#CHECK: cuse	%r0, %r1

	cuse	%r1, %r0
	cuse	%r0, %r1

#CHECK: error: invalid register pair
#CHECK: cutfu	%r1, %r0
#CHECK: error: invalid register pair
#CHECK: cutfu	%r0, %r1
#CHECK: error: invalid operand
#CHECK: cutfu	%r2, %r4, -1
#CHECK: error: invalid operand
#CHECK: cutfu	%r2, %r4, 16

	cutfu	%r1, %r0
	cutfu	%r0, %r1
	cutfu	%r2, %r4, -1
	cutfu	%r2, %r4, 16

#CHECK: error: invalid register pair
#CHECK: cuutf	%r1, %r0
#CHECK: error: invalid register pair
#CHECK: cuutf	%r0, %r1
#CHECK: error: invalid operand
#CHECK: cuutf	%r2, %r4, -1
#CHECK: error: invalid operand
#CHECK: cuutf	%r2, %r4, 16

	cuutf	%r1, %r0
	cuutf	%r0, %r1
	cuutf	%r2, %r4, -1
	cuutf	%r2, %r4, 16

#CHECK: error: invalid register pair
#CHECK: cuxtr	%r0, %f2
#CHECK: error: invalid register pair
#CHECK: cuxtr	%r1, %f0

	cuxtr	%r0, %f2
	cuxtr	%r1, %f0

#CHECK: error: invalid operand
#CHECK: cvb	%r0, -1
#CHECK: error: invalid operand
#CHECK: cvb	%r0, 4096

	cvb	%r0, -1
	cvb	%r0, 4096

#CHECK: error: invalid operand
#CHECK: cvbg	%r0, -524289
#CHECK: error: invalid operand
#CHECK: cvbg	%r0, 524288

	cvbg	%r0, -524289
	cvbg	%r0, 524288

#CHECK: error: invalid operand
#CHECK: cvby	%r0, -524289
#CHECK: error: invalid operand
#CHECK: cvby	%r0, 524288

	cvby	%r0, -524289
	cvby	%r0, 524288

#CHECK: error: invalid operand
#CHECK: cvd	%r0, -1
#CHECK: error: invalid operand
#CHECK: cvd	%r0, 4096

	cvd	%r0, -1
	cvd	%r0, 4096

#CHECK: error: invalid operand
#CHECK: cvdg	%r0, -524289
#CHECK: error: invalid operand
#CHECK: cvdg	%r0, 524288

	cvdg	%r0, -524289
	cvdg	%r0, 524288

#CHECK: error: invalid operand
#CHECK: cvdy	%r0, -524289
#CHECK: error: invalid operand
#CHECK: cvdy	%r0, 524288

	cvdy	%r0, -524289
	cvdy	%r0, 524288

#CHECK: error: invalid register pair
#CHECK: cxbr	%f0, %f2
#CHECK: error: invalid register pair
#CHECK: cxbr	%f2, %f0

	cxbr	%f0, %f2
	cxbr	%f2, %f0

#CHECK: error: invalid register pair
#CHECK: cxfbr	%f2, %r0

	cxfbr	%f2, %r0

#CHECK: error: instruction requires: fp-extension
#CHECK: cxfbra	%f0, 0, %r0, 0

	cxfbra	%f0, 0, %r0, 0

#CHECK: error: instruction requires: fp-extension
#CHECK: cxftr	%f0, 0, %r0, 0

	cxftr	%f0, 0, %r0, 0

#CHECK: error: invalid register pair
#CHECK: cxfr	%f2, %r0

	cxfr	%f2, %r0

#CHECK: error: invalid register pair
#CHECK: cxgbr	%f2, %r0

	cxgbr	%f2, %r0

#CHECK: error: instruction requires: fp-extension
#CHECK: cxgbra	%f0, 0, %r0, 0

	cxgbra	%f0, 0, %r0, 0

#CHECK: error: invalid register pair
#CHECK: cxgr	%f2, %r0

	cxgr	%f2, %r0

#CHECK: error: invalid register pair
#CHECK: cxgtr	%f2, %r0

	cxgtr	%f2, %r0

#CHECK: error: instruction requires: fp-extension
#CHECK: cxgtra	%f0, 0, %r0, 0

	cxgtra	%f0, 0, %r0, 0

#CHECK: error: instruction requires: fp-extension
#CHECK: cxlfbr	%f0, 0, %r0, 0

	cxlfbr	%f0, 0, %r0, 0

#CHECK: error: instruction requires: fp-extension
#CHECK: cxlftr	%f0, 0, %r0, 0

	cxlftr	%f0, 0, %r0, 0

#CHECK: error: instruction requires: fp-extension
#CHECK: cxlgbr	%f0, 0, %r0, 0

	cxlgbr	%f0, 0, %r0, 0

#CHECK: error: instruction requires: fp-extension
#CHECK: cxlgtr	%f0, 0, %r0, 0

	cxlgtr	%f0, 0, %r0, 0

#CHECK: error: invalid register pair
#CHECK: cxr	%f0, %f2
#CHECK: error: invalid register pair
#CHECK: cxr	%f2, %f0

	cxr	%f0, %f2
	cxr	%f2, %f0

#CHECK: error: invalid register pair
#CHECK: cxstr	%f0, %r1
#CHECK: error: invalid register pair
#CHECK: cxstr	%f2, %r0

	cxstr	%f0, %r1
	cxstr	%f2, %r0

#CHECK: error: invalid register pair
#CHECK: cxtr	%f0, %f2
#CHECK: error: invalid register pair
#CHECK: cxtr	%f2, %f0

	cxtr	%f0, %f2
	cxtr	%f2, %f0

#CHECK: error: invalid register pair
#CHECK: cxutr	%f0, %r1
#CHECK: error: invalid register pair
#CHECK: cxutr	%f2, %r0

	cxutr	%f0, %r1
	cxutr	%f2, %r0

#CHECK: error: invalid operand
#CHECK: cy	%r0, -524289
#CHECK: error: invalid operand
#CHECK: cy	%r0, 524288

	cy	%r0, -524289
	cy	%r0, 524288

#CHECK: error: invalid operand
#CHECK: d	%r0, -1
#CHECK: error: invalid operand
#CHECK: d	%r0, 4096
#CHECK: error: invalid register pair
#CHECK: d	%r1, 0

	d	%r0, -1
	d	%r0, 4096
	d	%r1, 0

#CHECK: error: invalid operand
#CHECK: dd	%f0, -1
#CHECK: error: invalid operand
#CHECK: dd	%f0, 4096

	dd	%f0, -1
	dd	%f0, 4096

#CHECK: error: invalid operand
#CHECK: ddb	%f0, -1
#CHECK: error: invalid operand
#CHECK: ddb	%f0, 4096

	ddb	%f0, -1
	ddb	%f0, 4096

#CHECK: error: instruction requires: fp-extension
#CHECK: ddtra	%f0, %f0, %f0, 0

	ddtra	%f0, %f0, %f0, 0

#CHECK: error: invalid operand
#CHECK: de	%f0, -1
#CHECK: error: invalid operand
#CHECK: de	%f0, 4096

	de	%f0, -1
	de	%f0, 4096

#CHECK: error: invalid operand
#CHECK: deb	%f0, -1
#CHECK: error: invalid operand
#CHECK: deb	%f0, 4096

	deb	%f0, -1
	deb	%f0, 4096

#CHECK: error: invalid operand
#CHECK: diag	%r0, %r0, -1
#CHECK: error: invalid operand
#CHECK: diag	%r0, %r0, 4096
#CHECK: error: invalid use of indexed addressing
#CHECK: diag	%r0, %r0, 0(%r1,%r2)

	diag	%r0, %r0, -1
	diag	%r0, %r0, 4096
	diag	%r0, %r0, 0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: didbr	%f0, %f0, %f0, -1
#CHECK: error: invalid operand
#CHECK: didbr	%f0, %f0, %f0, 16

	didbr	%f0, %f0, %f0, -1
	didbr	%f0, %f0, %f0, 16

#CHECK: error: invalid operand
#CHECK: diebr	%f0, %f0, %f0, -1
#CHECK: error: invalid operand
#CHECK: diebr	%f0, %f0, %f0, 16

	diebr	%f0, %f0, %f0, -1
	diebr	%f0, %f0, %f0, 16

#CHECK: error: invalid operand
#CHECK: dl	%r0, -524289
#CHECK: error: invalid operand
#CHECK: dl	%r0, 524288
#CHECK: error: invalid register pair
#CHECK: dl	%r1, 0

	dl	%r0, -524289
	dl	%r0, 524288
	dl	%r1, 0

#CHECK: error: invalid register pair
#CHECK: dr	%r1, %r0

	dr	%r1, %r0

#CHECK: error: invalid operand
#CHECK: dlg	%r0, -524289
#CHECK: error: invalid operand
#CHECK: dlg	%r0, 524288
#CHECK: error: invalid register pair
#CHECK: dlg	%r1, 0

	dlg	%r0, -524289
	dlg	%r0, 524288
	dlg	%r1, 0

#CHECK: error: invalid register pair
#CHECK: dlgr	%r1, %r0

	dlgr	%r1, %r0

#CHECK: error: invalid register pair
#CHECK: dlr	%r1, %r0

	dlr	%r1, %r0

#CHECK: error: missing length in address
#CHECK: dp	0, 0(1)
#CHECK: error: missing length in address
#CHECK: dp	0(1), 0
#CHECK: error: missing length in address
#CHECK: dp	0(%r1), 0(1,%r1)
#CHECK: error: missing length in address
#CHECK: dp	0(1,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: dp	0(0,%r1), 0(1,%r1)
#CHECK: error: invalid operand
#CHECK: dp	0(1,%r1), 0(0,%r1)
#CHECK: error: invalid operand
#CHECK: dp	0(17,%r1), 0(1,%r1)
#CHECK: error: invalid operand
#CHECK: dp	0(1,%r1), 0(17,%r1)
#CHECK: error: invalid operand
#CHECK: dp	-1(1,%r1), 0(1,%r1)
#CHECK: error: invalid operand
#CHECK: dp	4096(1,%r1), 0(1,%r1)
#CHECK: error: invalid operand
#CHECK: dp	0(1,%r1), -1(1,%r1)
#CHECK: error: invalid operand
#CHECK: dp	0(1,%r1), 4096(1,%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: dp	0(%r1,%r2), 0(1,%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: dp	0(1,%r2), 0(%r1,%r2)
#CHECK: error: unknown token in expression
#CHECK: dp	0(-), 0(1)

	dp	0, 0(1)
	dp	0(1), 0
	dp	0(%r1), 0(1,%r1)
	dp	0(1,%r1), 0(%r1)
	dp	0(0,%r1), 0(1,%r1)
	dp	0(1,%r1), 0(0,%r1)
	dp	0(17,%r1), 0(1,%r1)
	dp	0(1,%r1), 0(17,%r1)
	dp	-1(1,%r1), 0(1,%r1)
	dp	4096(1,%r1), 0(1,%r1)
	dp	0(1,%r1), -1(1,%r1)
	dp	0(1,%r1), 4096(1,%r1)
	dp	0(%r1,%r2), 0(1,%r1)
	dp	0(1,%r2), 0(%r1,%r2)
	dp	0(-), 0(1)

#CHECK: error: invalid operand
#CHECK: dsg	%r0, -524289
#CHECK: error: invalid operand
#CHECK: dsg	%r0, 524288
#CHECK: error: invalid register pair
#CHECK: dsg	%r1, 0

	dsg	%r0, -524289
	dsg	%r0, 524288
	dsg	%r1, 0

#CHECK: error: invalid operand
#CHECK: dsgf	%r0, -524289
#CHECK: error: invalid operand
#CHECK: dsgf	%r0, 524288
#CHECK: error: invalid register pair
#CHECK: dsgf	%r1, 0

	dsgf	%r0, -524289
	dsgf	%r0, 524288
	dsgf	%r1, 0

#CHECK: error: invalid register pair
#CHECK: dsgfr	%r1, %r0

	dsgfr	%r1, %r0

#CHECK: error: invalid register pair
#CHECK: dsgr	%r1, %r0

	dsgr	%r1, %r0

#CHECK: error: invalid register pair
#CHECK: dxbr	%f0, %f2
#CHECK: error: invalid register pair
#CHECK: dxbr	%f2, %f0

	dxbr	%f0, %f2
	dxbr	%f2, %f0

#CHECK: error: invalid register pair
#CHECK: dxr	%f0, %f2
#CHECK: error: invalid register pair
#CHECK: dxr	%f2, %f0

	dxr	%f0, %f2
	dxr	%f2, %f0

#CHECK: error: invalid register pair
#CHECK: dxtr	%f0, %f0, %f2
#CHECK: error: invalid register pair
#CHECK: dxtr	%f0, %f2, %f0
#CHECK: error: invalid register pair
#CHECK: dxtr	%f2, %f0, %f0

	dxtr	%f0, %f0, %f2
	dxtr	%f0, %f2, %f0
	dxtr	%f2, %f0, %f0

#CHECK: error: instruction requires: fp-extension
#CHECK: dxtra	%f0, %f0, %f0, 0

	dxtra	%f0, %f0, %f0, 0

#CHECK: error: invalid operand
#CHECK: ecag	%r0, %r0, -524289
#CHECK: error: invalid operand
#CHECK: ecag	%r0, %r0, 524288
#CHECK: error: invalid use of indexed addressing
#CHECK: ecag	%r0, %r0, 0(%r1,%r2)

	ecag	%r0, %r0, -524289
	ecag	%r0, %r0, 524288
	ecag	%r0, %r0, 0(%r1,%r2)

#CHECK: error: invalid use of indexed addressing
#CHECK: ectg    160(%r1,%r15),160(%r15), %r2
#CHECK: error: invalid operand
#CHECK: ectg    -1(%r1),160(%r15), %r2
#CHECK: error: invalid operand
#CHECK: ectg    4096(%r1),160(%r15), %r2
#CHECK: error: invalid operand
#CHECK: ectg    0(%r1),-1(%r15), %r2
#CHECK: error: invalid operand
#CHECK: ectg    0(%r1),4096(%r15), %r2

        ectg    160(%r1,%r15),160(%r15), %r2
        ectg    -1(%r1),160(%r15), %r2
        ectg    4096(%r1),160(%r15), %r2
        ectg    0(%r1),-1(%r15), %r2
        ectg    0(%r1),4096(%r15), %r2

#CHECK: error: missing length in address
#CHECK: ed	0, 0
#CHECK: error: missing length in address
#CHECK: ed	0(%r1), 0(%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: ed	0(1,%r1), 0(2,%r1)
#CHECK: error: invalid operand
#CHECK: ed	0(0,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: ed	0(257,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: ed	-1(1,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: ed	4096(1,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: ed	0(1,%r1), -1(%r1)
#CHECK: error: invalid operand
#CHECK: ed	0(1,%r1), 4096(%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: ed	0(%r1,%r2), 0(%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: ed	0(1,%r2), 0(%r1,%r2)
#CHECK: error: unknown token in expression
#CHECK: ed	0(-), 0

	ed	0, 0
	ed	0(%r1), 0(%r1)
	ed	0(1,%r1), 0(2,%r1)
	ed	0(0,%r1), 0(%r1)
	ed	0(257,%r1), 0(%r1)
	ed	-1(1,%r1), 0(%r1)
	ed	4096(1,%r1), 0(%r1)
	ed	0(1,%r1), -1(%r1)
	ed	0(1,%r1), 4096(%r1)
	ed	0(%r1,%r2), 0(%r1)
	ed	0(1,%r2), 0(%r1,%r2)
	ed	0(-), 0

#CHECK: error: missing length in address
#CHECK: edmk	0, 0
#CHECK: error: missing length in address
#CHECK: edmk	0(%r1), 0(%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: edmk	0(1,%r1), 0(2,%r1)
#CHECK: error: invalid operand
#CHECK: edmk	0(0,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: edmk	0(257,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: edmk	-1(1,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: edmk	4096(1,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: edmk	0(1,%r1), -1(%r1)
#CHECK: error: invalid operand
#CHECK: edmk	0(1,%r1), 4096(%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: edmk	0(%r1,%r2), 0(%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: edmk	0(1,%r2), 0(%r1,%r2)
#CHECK: error: unknown token in expression
#CHECK: edmk	0(-), 0

	edmk	0, 0
	edmk	0(%r1), 0(%r1)
	edmk	0(1,%r1), 0(2,%r1)
	edmk	0(0,%r1), 0(%r1)
	edmk	0(257,%r1), 0(%r1)
	edmk	-1(1,%r1), 0(%r1)
	edmk	4096(1,%r1), 0(%r1)
	edmk	0(1,%r1), -1(%r1)
	edmk	0(1,%r1), 4096(%r1)
	edmk	0(%r1,%r2), 0(%r1)
	edmk	0(1,%r2), 0(%r1,%r2)
	edmk	0(-), 0

#CHECK: error: invalid register pair
#CHECK: eextr	%f0, %f2
#CHECK: error: invalid register pair
#CHECK: eextr	%f2, %f0

	eextr	%f0, %f2
	eextr	%f2, %f0

#CHECK: error: invalid register pair
#CHECK: esta	%r1, %r0

	esta	%r1, %r0

#CHECK: error: invalid register pair
#CHECK: esxtr	%f0, %f2
#CHECK: error: invalid register pair
#CHECK: esxtr	%f2, %f0

	esxtr	%f0, %f2
	esxtr	%f2, %f0

#CHECK: error: invalid operand
#CHECK: ex      %r0, -1
#CHECK: error: invalid operand
#CHECK: ex      %r0, 4096

        ex      %r0, -1
        ex      %r0, 4096

#CHECK: error: invalid operand
#CHECK: fidbr	%f0, -1, %f0
#CHECK: error: invalid operand
#CHECK: fidbr	%f0, 16, %f0

	fidbr	%f0, -1, %f0
	fidbr	%f0, 16, %f0

#CHECK: error: instruction requires: fp-extension
#CHECK: fidbra	%f0, 0, %f0, 0

	fidbra	%f0, 0, %f0, 0

#CHECK: error: invalid operand
#CHECK: fidtr	%f0, 0, %f0, -1
#CHECK: error: invalid operand
#CHECK: fidtr	%f0, 0, %f0, 16
#CHECK: error: invalid operand
#CHECK: fidtr	%f0, -1, %f0, 0
#CHECK: error: invalid operand
#CHECK: fidtr	%f0, 16, %f0, 0

	fidtr	%f0, 0, %f0, -1
	fidtr	%f0, 0, %f0, 16
	fidtr	%f0, -1, %f0, 0
	fidtr	%f0, 16, %f0, 0

#CHECK: error: invalid operand
#CHECK: fiebr	%f0, -1, %f0
#CHECK: error: invalid operand
#CHECK: fiebr	%f0, 16, %f0

	fiebr	%f0, -1, %f0
	fiebr	%f0, 16, %f0

#CHECK: error: instruction requires: fp-extension
#CHECK: fiebra	%f0, 0, %f0, 0

	fiebra	%f0, 0, %f0, 0

#CHECK: error: invalid operand
#CHECK: fixbr	%f0, -1, %f0
#CHECK: error: invalid operand
#CHECK: fixbr	%f0, 16, %f0
#CHECK: error: invalid register pair
#CHECK: fixbr	%f0, 0, %f2
#CHECK: error: invalid register pair
#CHECK: fixbr	%f2, 0, %f0

	fixbr	%f0, -1, %f0
	fixbr	%f0, 16, %f0
	fixbr	%f0, 0, %f2
	fixbr	%f2, 0, %f0

#CHECK: error: instruction requires: fp-extension
#CHECK: fixbra	%f0, 0, %f0, 0

	fixbra	%f0, 0, %f0, 0

#CHECK: error: invalid register pair
#CHECK: fixr	%f0, %f2
#CHECK: error: invalid register pair
#CHECK: fixr	%f2, %f0

	fixr	%f0, %f2
	fixr	%f2, %f0

#CHECK: error: invalid operand
#CHECK: fixtr	%f0, 0, %f0, -1
#CHECK: error: invalid operand
#CHECK: fixtr	%f0, 0, %f0, 16
#CHECK: error: invalid operand
#CHECK: fixtr	%f0, -1, %f0, 0
#CHECK: error: invalid operand
#CHECK: fixtr	%f0, 16, %f0, 0
#CHECK: error: invalid register pair
#CHECK: fixtr	%f0, 0, %f2, 0
#CHECK: error: invalid register pair
#CHECK: fixtr	%f2, 0, %f0, 0

	fixtr	%f0, 0, %f0, -1
	fixtr	%f0, 0, %f0, 16
	fixtr	%f0, -1, %f0, 0
	fixtr	%f0, 16, %f0, 0
	fixtr	%f0, 0, %f2, 0
	fixtr	%f2, 0, %f0, 0

#CHECK: error: invalid register pair
#CHECK: flogr	%r1, %r0

	flogr	%r1, %r0

#CHECK: error: invalid operand
#CHECK: ic	%r0, -1
#CHECK: error: invalid operand
#CHECK: ic	%r0, 4096

	ic	%r0, -1
	ic	%r0, 4096

#CHECK: error: invalid operand
#CHECK: icm	%r0, 0, -1
#CHECK: error: invalid operand
#CHECK: icm	%r0, 0, 4096
#CHECK: error: invalid operand
#CHECK: icm	%r0, -1, 0
#CHECK: error: invalid operand
#CHECK: icm	%r0, 16, 0

	icm	%r0, 0, -1
	icm	%r0, 0, 4096
	icm	%r0, -1, 0
	icm	%r0, 16, 0

#CHECK: error: invalid operand
#CHECK: icmh	%r0, 0, -524289
#CHECK: error: invalid operand
#CHECK: icmh	%r0, 0, 524288
#CHECK: error: invalid operand
#CHECK: icmh	%r0, -1, 0
#CHECK: error: invalid operand
#CHECK: icmh	%r0, 16, 0

	icmh	%r0, 0, -524289
	icmh	%r0, 0, 524288
	icmh	%r0, -1, 0
	icmh	%r0, 16, 0

#CHECK: error: invalid operand
#CHECK: icmy	%r0, 0, -524289
#CHECK: error: invalid operand
#CHECK: icmy	%r0, 0, 524288
#CHECK: error: invalid operand
#CHECK: icmy	%r0, -1, 0
#CHECK: error: invalid operand
#CHECK: icmy	%r0, 16, 0

	icmy	%r0, 0, -524289
	icmy	%r0, 0, 524288
	icmy	%r0, -1, 0
	icmy	%r0, 16, 0

#CHECK: error: invalid operand
#CHECK: icy	%r0, -524289
#CHECK: error: invalid operand
#CHECK: icy	%r0, 524288

	icy	%r0, -524289
	icy	%r0, 524288

#CHECK: error: invalid operand
#CHECK: idte	%r0, %r0, %r0, -1
#CHECK: error: invalid operand
#CHECK: idte	%r0, %r0, %r0, 16

	idte	%r0, %r0, %r0, -1
	idte	%r0, %r0, %r0, 16

#CHECK: error: invalid register pair
#CHECK: iextr	%f0, %f0, %f2
#CHECK: error: invalid register pair
#CHECK: iextr	%f0, %f2, %f0
#CHECK: error: invalid register pair
#CHECK: iextr	%f2, %f0, %f0

	iextr	%f0, %f0, %f2
	iextr	%f0, %f2, %f0
	iextr	%f2, %f0, %f0

#CHECK: error: invalid operand
#CHECK: iihf	%r0, -1
#CHECK: error: invalid operand
#CHECK: iihf	%r0, 1 << 32

	iihf	%r0, -1
	iihf	%r0, 1 << 32

#CHECK: error: invalid operand
#CHECK: iihh	%r0, -1
#CHECK: error: invalid operand
#CHECK: iihh	%r0, 0x10000

	iihh	%r0, -1
	iihh	%r0, 0x10000

#CHECK: error: invalid operand
#CHECK: iihl	%r0, -1
#CHECK: error: invalid operand
#CHECK: iihl	%r0, 0x10000

	iihl	%r0, -1
	iihl	%r0, 0x10000

#CHECK: error: invalid operand
#CHECK: iilf	%r0, -1
#CHECK: error: invalid operand
#CHECK: iilf	%r0, 1 << 32

	iilf	%r0, -1
	iilf	%r0, 1 << 32

#CHECK: error: invalid operand
#CHECK: iilh	%r0, -1
#CHECK: error: invalid operand
#CHECK: iilh	%r0, 0x10000

	iilh	%r0, -1
	iilh	%r0, 0x10000

#CHECK: error: invalid operand
#CHECK: iill	%r0, -1
#CHECK: error: invalid operand
#CHECK: iill	%r0, 0x10000

	iill	%r0, -1
	iill	%r0, 0x10000

#CHECK: error: invalid operand
#CHECK: ipte	%r0, %r0, %r0, -1
#CHECK: error: invalid operand
#CHECK: ipte	%r0, %r0, %r0, 16

	ipte	%r0, %r0, %r0, -1
	ipte	%r0, %r0, %r0, 16

#CHECK: error: invalid operand
#CHECK: kdb	%f0, -1
#CHECK: error: invalid operand
#CHECK: kdb	%f0, 4096

	kdb	%f0, -1
	kdb	%f0, 4096

#CHECK: error: invalid operand
#CHECK: keb	%f0, -1
#CHECK: error: invalid operand
#CHECK: keb	%f0, 4096

	keb	%f0, -1
	keb	%f0, 4096

#CHECK: error: invalid register pair
#CHECK: kimd	%r0, %r1

	kimd	%r0, %r1

#CHECK: error: invalid register pair
#CHECK: klmd	%r0, %r1

	klmd	%r0, %r1

#CHECK: error: invalid register pair
#CHECK: km	%r1, %r2
#CHECK: error: invalid register pair
#CHECK: km	%r2, %r1

	km	%r1, %r2
	km	%r2, %r1

#CHECK: error: invalid register pair
#CHECK: kmac	%r0, %r1

	kmac	%r0, %r1

#CHECK: error: invalid register pair
#CHECK: kmc	%r1, %r2
#CHECK: error: invalid register pair
#CHECK: kmc	%r2, %r1

	kmc	%r1, %r2
	kmc	%r2, %r1

#CHECK: error: instruction requires: message-security-assist-extension4
#CHECK: kmctr	%r2, %r4, %r6

	kmctr	%r2, %r4, %r6

#CHECK: error: instruction requires: message-security-assist-extension4
#CHECK: kmf	%r2, %r4

	kmf	%r2, %r4

#CHECK: error: instruction requires: message-security-assist-extension4
#CHECK: kmo	%r2, %r4

	kmo	%r2, %r4

#CHECK: error: invalid register pair
#CHECK: kxbr	%f0, %f2
#CHECK: error: invalid register pair
#CHECK: kxbr	%f2, %f0

	kxbr	%f0, %f2
	kxbr	%f2, %f0

#CHECK: error: invalid register pair
#CHECK: kxtr	%f0, %f2
#CHECK: error: invalid register pair
#CHECK: kxtr	%f2, %f0

	kxtr	%f0, %f2
	kxtr	%f2, %f0

#CHECK: error: invalid operand
#CHECK: l	%r0, -1
#CHECK: error: invalid operand
#CHECK: l	%r0, 4096

	l	%r0, -1
	l	%r0, 4096

#CHECK: error: invalid operand
#CHECK: la	%r0, -1
#CHECK: error: invalid operand
#CHECK: la	%r0, 4096

	la	%r0, -1
	la	%r0, 4096

#CHECK: error: instruction requires: interlocked-access1
#CHECK: laa	%r1, %r2, 100(%r3)
	laa	%r1, %r2, 100(%r3)

#CHECK: error: instruction requires: interlocked-access1
#CHECK: laag	%r1, %r2, 100(%r3)
	laag	%r1, %r2, 100(%r3)

#CHECK: error: instruction requires: interlocked-access1
#CHECK: laal	%r1, %r2, 100(%r3)
	laal	%r1, %r2, 100(%r3)

#CHECK: error: instruction requires: interlocked-access1
#CHECK: laalg	%r1, %r2, 100(%r3)
	laalg	%r1, %r2, 100(%r3)

#CHECK: error: invalid operand
#CHECK: lae	%r0, -1
#CHECK: error: invalid operand
#CHECK: lae	%r0, 4096

	lae	%r0, -1
	lae	%r0, 4096

#CHECK: error: invalid operand
#CHECK: laey	%r0, -524289
#CHECK: error: invalid operand
#CHECK: laey	%r0, 524288

	laey	%r0, -524289
	laey	%r0, 524288

#CHECK: error: invalid operand
#CHECK: lam	%a0, %a0, 4096
#CHECK: error: invalid use of indexed addressing
#CHECK: lam	%a0, %a0, 0(%r1,%r2)

	lam	%a0, %a0, 4096
	lam	%a0, %a0, 0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: lamy	%a0, %a0, -524289
#CHECK: error: invalid operand
#CHECK: lamy	%a0, %a0, 524288
#CHECK: error: invalid use of indexed addressing
#CHECK: lamy	%a0, %a0, 0(%r1,%r2)

	lamy	%a0, %a0, -524289
	lamy	%a0, %a0, 524288
	lamy	%a0, %a0, 0(%r1,%r2)

#CHECK: error: instruction requires: interlocked-access1
#CHECK: lan	%r1, %r2, 100(%r3)
	lan	%r1, %r2, 100(%r3)

#CHECK: error: instruction requires: interlocked-access1
#CHECK: lang	%r1, %r2, 100(%r3)
	lang	%r1, %r2, 100(%r3)

#CHECK: error: instruction requires: interlocked-access1
#CHECK: lao	%r1, %r2, 100(%r3)
	lao	%r1, %r2, 100(%r3)

#CHECK: error: instruction requires: interlocked-access1
#CHECK: laog	%r1, %r2, 100(%r3)
	laog	%r1, %r2, 100(%r3)

#CHECK: error: offset out of range
#CHECK: larl	%r0, -0x1000000002
#CHECK: error: offset out of range
#CHECK: larl	%r0, -1
#CHECK: error: offset out of range
#CHECK: larl	%r0, 1
#CHECK: error: offset out of range
#CHECK: larl	%r0, 0x100000000
#CHECK: error: offset out of range
#CHECK: larl	%r1, __unnamed_1+3564822854692

	larl	%r0, -0x1000000002
	larl	%r0, -1
	larl	%r0, 1
	larl	%r0, 0x100000000
	larl	%r1, __unnamed_1+3564822854692

#CHECK: error: invalid use of indexed addressing
#CHECK: lasp	160(%r1,%r15),160(%r15)
#CHECK: error: invalid operand
#CHECK: lasp	-1(%r1),160(%r15)
#CHECK: error: invalid operand
#CHECK: lasp	4096(%r1),160(%r15)
#CHECK: error: invalid operand
#CHECK: lasp	0(%r1),-1(%r15)
#CHECK: error: invalid operand
#CHECK: lasp	0(%r1),4096(%r15)

        lasp	160(%r1,%r15),160(%r15)
        lasp	-1(%r1),160(%r15)
        lasp	4096(%r1),160(%r15)
        lasp	0(%r1),-1(%r15)
        lasp	0(%r1),4096(%r15)

#CHECK: error: instruction requires: interlocked-access1
#CHECK: lax	%r1, %r2, 100(%r3)
	lax	%r1, %r2, 100(%r3)

#CHECK: error: instruction requires: interlocked-access1
#CHECK: laxg	%r1, %r2, 100(%r3)
	laxg	%r1, %r2, 100(%r3)

#CHECK: error: invalid operand
#CHECK: lay	%r0, -524289
#CHECK: error: invalid operand
#CHECK: lay	%r0, 524288

	lay	%r0, -524289
	lay	%r0, 524288

#CHECK: error: invalid operand
#CHECK: lb	%r0, -524289
#CHECK: error: invalid operand
#CHECK: lb	%r0, 524288

	lb	%r0, -524289
	lb	%r0, 524288

#CHECK: error: instruction requires: high-word
#CHECK: lbh	%r0, 0

	lbh	%r0, 0

#CHECK: error: invalid operand
#CHECK: lcctl	-1
#CHECK: error: invalid operand
#CHECK: lcctl	4096
#CHECK: error: invalid use of indexed addressing
#CHECK: lcctl	0(%r1,%r2)

	lcctl	-1
	lcctl	4096
	lcctl	0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: lctl	%c0, %c0, -1
#CHECK: error: invalid operand
#CHECK: lctl	%c0, %c0, 4096
#CHECK: error: invalid use of indexed addressing
#CHECK: lctl	%c0, %c0, 0(%r1,%r2)

	lctl	%c0, %c0, -1
	lctl	%c0, %c0, 4096
	lctl	%c0, %c0, 0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: lctlg	%c0, %c0, -524289
#CHECK: error: invalid operand
#CHECK: lctlg	%c0, %c0, 524288
#CHECK: error: invalid use of indexed addressing
#CHECK: lctlg	%c0, %c0, 0(%r1,%r2)

	lctlg	%c0, %c0, -524289
	lctlg	%c0, %c0, 524288
	lctlg	%c0, %c0, 0(%r1,%r2)

#CHECK: error: invalid register pair
#CHECK: lcxbr	%f0, %f2
#CHECK: error: invalid register pair
#CHECK: lcxbr	%f2, %f0

	lcxbr	%f0, %f2
	lcxbr	%f2, %f0

#CHECK: error: invalid register pair
#CHECK: lcxr	%f0, %f2
#CHECK: error: invalid register pair
#CHECK: lcxr	%f2, %f0

	lcxr	%f0, %f2
	lcxr	%f2, %f0

#CHECK: error: invalid operand
#CHECK: ld	%f0, -1
#CHECK: error: invalid operand
#CHECK: ld	%f0, 4096

	ld	%f0, -1
	ld	%f0, 4096

#CHECK: error: invalid operand
#CHECK: ldeb	%f0, -1
#CHECK: error: invalid operand
#CHECK: ldeb	%f0, 4096

	ldeb	%f0, -1
	ldeb	%f0, 4096

#CHECK: error: invalid operand
#CHECK: ldetr	%f0, %f0, -1
#CHECK: error: invalid operand
#CHECK: ldetr	%f0, %f0, 16

	ldetr	%f0, %f0, -1
	ldetr	%f0, %f0, 16

#CHECK: error: invalid register pair
#CHECK: ldxbr	%f0, %f2
#CHECK: error: invalid register pair
#CHECK: ldxbr	%f2, %f0

	ldxbr	%f0, %f2
	ldxbr	%f2, %f0

#CHECK: error: instruction requires: fp-extension
#CHECK: ldxbra	%f0, 0, %f0, 0

	ldxbra	%f0, 0, %f0, 0

#CHECK: error: invalid register pair
#CHECK: ldxr	%f0, %f2

	ldxr	%f0, %f2

#CHECK: error: invalid operand
#CHECK: ldxtr	%f0, 0, %f0, -1
#CHECK: error: invalid operand
#CHECK: ldxtr	%f0, 0, %f0, 16
#CHECK: error: invalid operand
#CHECK: ldxtr	%f0, -1, %f0, 0
#CHECK: error: invalid operand
#CHECK: ldxtr	%f0, 16, %f0, 0
#CHECK: error: invalid register pair
#CHECK: ldxtr	%f0, 0, %f2, 0
#CHECK: error: invalid register pair
#CHECK: ldxtr	%f2, 0, %f0, 0

	ldxtr	%f0, 0, %f0, -1
	ldxtr	%f0, 0, %f0, 16
	ldxtr	%f0, -1, %f0, 0
	ldxtr	%f0, 16, %f0, 0
	ldxtr	%f0, 0, %f2, 0
	ldxtr	%f2, 0, %f0, 0

#CHECK: error: invalid operand
#CHECK: ldy	%f0, -524289
#CHECK: error: invalid operand
#CHECK: ldy	%f0, 524288

	ldy	%f0, -524289
	ldy	%f0, 524288

#CHECK: error: invalid operand
#CHECK: le	%f0, -1
#CHECK: error: invalid operand
#CHECK: le	%f0, 4096

	le	%f0, -1
	le	%f0, 4096

#CHECK: error: instruction requires: fp-extension
#CHECK: ledbra	%f0, 0, %f0, 0

	ledbra	%f0, 0, %f0, 0

#CHECK: error: invalid operand
#CHECK: ledtr	%f0, 0, %f0, -1
#CHECK: error: invalid operand
#CHECK: ledtr	%f0, 0, %f0, 16
#CHECK: error: invalid operand
#CHECK: ledtr	%f0, -1, %f0, 0
#CHECK: error: invalid operand
#CHECK: ledtr	%f0, 16, %f0, 0

	ledtr	%f0, 0, %f0, -1
	ledtr	%f0, 0, %f0, 16
	ledtr	%f0, -1, %f0, 0
	ledtr	%f0, 16, %f0, 0

#CHECK: error: invalid register pair
#CHECK: lexbr	%f0, %f2
#CHECK: error: invalid register pair
#CHECK: lexbr	%f2, %f0

	lexbr	%f0, %f2
	lexbr	%f2, %f0

#CHECK: error: instruction requires: fp-extension
#CHECK: lexbra	%f0, 0, %f0, 0

	lexbra	%f0, 0, %f0, 0

#CHECK: error: invalid register pair
#CHECK: lexr	%f0, %f2

	lexr	%f0, %f2

#CHECK: error: invalid operand
#CHECK: ley	%f0, -524289
#CHECK: error: invalid operand
#CHECK: ley	%f0, 524288

	ley	%f0, -524289
	ley	%f0, 524288

#CHECK: error: invalid operand
#CHECK: lfas	-1
#CHECK: error: invalid operand
#CHECK: lfas	4096
#CHECK: error: invalid use of indexed addressing
#CHECK: lfas	0(%r1,%r2)

	lfas	-1
	lfas	4096
	lfas	0(%r1,%r2)

#CHECK: error: instruction requires: high-word
#CHECK: lfh	%r0, 0

	lfh	%r0, 0

#CHECK: error: invalid operand
#CHECK: lfpc	-1
#CHECK: error: invalid operand
#CHECK: lfpc	4096
#CHECK: error: invalid use of indexed addressing
#CHECK: lfpc	0(%r1,%r2)

	lfpc	-1
	lfpc	4096
	lfpc	0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: lg	%r0, -524289
#CHECK: error: invalid operand
#CHECK: lg	%r0, 524288

	lg	%r0, -524289
	lg	%r0, 524288

#CHECK: error: invalid operand
#CHECK: lgb	%r0, -524289
#CHECK: error: invalid operand
#CHECK: lgb	%r0, 524288

	lgb	%r0, -524289
	lgb	%r0, 524288

#CHECK: error: invalid operand
#CHECK: lgf	%r0, -524289
#CHECK: error: invalid operand
#CHECK: lgf	%r0, 524288

	lgf	%r0, -524289
	lgf	%r0, 524288

#CHECK: error: invalid operand
#CHECK: lgfi	%r0, (-1 << 31) - 1
#CHECK: error: invalid operand
#CHECK: lgfi	%r0, (1 << 31)

	lgfi	%r0, (-1 << 31) - 1
	lgfi	%r0, (1 << 31)

#CHECK: error: offset out of range
#CHECK: lgfrl	%r0, -0x1000000002
#CHECK: error: offset out of range
#CHECK: lgfrl	%r0, -1
#CHECK: error: offset out of range
#CHECK: lgfrl	%r0, 1
#CHECK: error: offset out of range
#CHECK: lgfrl	%r0, 0x100000000

	lgfrl	%r0, -0x1000000002
	lgfrl	%r0, -1
	lgfrl	%r0, 1
	lgfrl	%r0, 0x100000000

#CHECK: error: invalid operand
#CHECK: lgh	%r0, -524289
#CHECK: error: invalid operand
#CHECK: lgh	%r0, 524288

	lgh	%r0, -524289
	lgh	%r0, 524288

#CHECK: error: invalid operand
#CHECK: lghi	%r0, -32769
#CHECK: error: invalid operand
#CHECK: lghi	%r0, 32768
#CHECK: error: invalid operand
#CHECK: lghi	%r0, foo

	lghi	%r0, -32769
	lghi	%r0, 32768
	lghi	%r0, foo

#CHECK: error: offset out of range
#CHECK: lghrl	%r0, -0x1000000002
#CHECK: error: offset out of range
#CHECK: lghrl	%r0, -1
#CHECK: error: offset out of range
#CHECK: lghrl	%r0, 1
#CHECK: error: offset out of range
#CHECK: lghrl	%r0, 0x100000000

	lghrl	%r0, -0x1000000002
	lghrl	%r0, -1
	lghrl	%r0, 1
	lghrl	%r0, 0x100000000

#CHECK: error: offset out of range
#CHECK: lgrl	%r0, -0x1000000002
#CHECK: error: offset out of range
#CHECK: lgrl	%r0, -1
#CHECK: error: offset out of range
#CHECK: lgrl	%r0, 1
#CHECK: error: offset out of range
#CHECK: lgrl	%r0, 0x100000000

	lgrl	%r0, -0x1000000002
	lgrl	%r0, -1
	lgrl	%r0, 1
	lgrl	%r0, 0x100000000

#CHECK: error: invalid operand
#CHECK: lh	%r0, -1
#CHECK: error: invalid operand
#CHECK: lh	%r0, 4096

	lh	%r0, -1
	lh	%r0, 4096

#CHECK: error: instruction requires: high-word
#CHECK: lhh	%r0, 0

	lhh	%r0, 0

#CHECK: error: invalid operand
#CHECK: lhi	%r0, -32769
#CHECK: error: invalid operand
#CHECK: lhi	%r0, 32768
#CHECK: error: invalid operand
#CHECK: lhi	%r0, foo

	lhi	%r0, -32769
	lhi	%r0, 32768
	lhi	%r0, foo

#CHECK: error: offset out of range
#CHECK: lhrl	%r0, -0x1000000002
#CHECK: error: offset out of range
#CHECK: lhrl	%r0, -1
#CHECK: error: offset out of range
#CHECK: lhrl	%r0, 1
#CHECK: error: offset out of range
#CHECK: lhrl	%r0, 0x100000000

	lhrl	%r0, -0x1000000002
	lhrl	%r0, -1
	lhrl	%r0, 1
	lhrl	%r0, 0x100000000

#CHECK: error: invalid operand
#CHECK: lhy	%r0, -524289
#CHECK: error: invalid operand
#CHECK: lhy	%r0, 524288

	lhy	%r0, -524289
	lhy	%r0, 524288

#CHECK: error: invalid operand
#CHECK: llc	%r0, -524289
#CHECK: error: invalid operand
#CHECK: llc	%r0, 524288

	llc	%r0, -524289
	llc	%r0, 524288

#CHECK: error: instruction requires: high-word
#CHECK: llch	%r0, 0

	llch	%r0, 0

#CHECK: error: invalid operand
#CHECK: llgc	%r0, -524289
#CHECK: error: invalid operand
#CHECK: llgc	%r0, 524288

	llgc	%r0, -524289
	llgc	%r0, 524288

#CHECK: error: invalid operand
#CHECK: llgf	%r0, -524289
#CHECK: error: invalid operand
#CHECK: llgf	%r0, 524288

	llgf	%r0, -524289
	llgf	%r0, 524288

#CHECK: error: offset out of range
#CHECK: llgfrl	%r0, -0x1000000002
#CHECK: error: offset out of range
#CHECK: llgfrl	%r0, -1
#CHECK: error: offset out of range
#CHECK: llgfrl	%r0, 1
#CHECK: error: offset out of range
#CHECK: llgfrl	%r0, 0x100000000

	llgfrl	%r0, -0x1000000002
	llgfrl	%r0, -1
	llgfrl	%r0, 1
	llgfrl	%r0, 0x100000000

#CHECK: error: invalid operand
#CHECK: llgh	%r0, -524289
#CHECK: error: invalid operand
#CHECK: llgh	%r0, 524288

	llgh	%r0, -524289
	llgh	%r0, 524288

#CHECK: error: offset out of range
#CHECK: llghrl	%r0, -0x1000000002
#CHECK: error: offset out of range
#CHECK: llghrl	%r0, -1
#CHECK: error: offset out of range
#CHECK: llghrl	%r0, 1
#CHECK: error: offset out of range
#CHECK: llghrl	%r0, 0x100000000

	llghrl	%r0, -0x1000000002
	llghrl	%r0, -1
	llghrl	%r0, 1
	llghrl	%r0, 0x100000000

#CHECK: error: invalid operand
#CHECK: llgt	%r0, -524289
#CHECK: error: invalid operand
#CHECK: llgt	%r0, 524288

	llgt	%r0, -524289
	llgt	%r0, 524288

#CHECK: error: invalid operand
#CHECK: llh	%r0, -524289
#CHECK: error: invalid operand
#CHECK: llh	%r0, 524288

	llh	%r0, -524289
	llh	%r0, 524288

#CHECK: error: instruction requires: high-word
#CHECK: llhh	%r0, 0

	llhh	%r0, 0

#CHECK: error: offset out of range
#CHECK: llhrl	%r0, -0x1000000002
#CHECK: error: offset out of range
#CHECK: llhrl	%r0, -1
#CHECK: error: offset out of range
#CHECK: llhrl	%r0, 1
#CHECK: error: offset out of range
#CHECK: llhrl	%r0, 0x100000000

	llhrl	%r0, -0x1000000002
	llhrl	%r0, -1
	llhrl	%r0, 1
	llhrl	%r0, 0x100000000

#CHECK: error: invalid operand
#CHECK: llihf	%r0, -1
#CHECK: error: invalid operand
#CHECK: llihf	%r0, 1 << 32

	llihf	%r0, -1
	llihf	%r0, 1 << 32

#CHECK: error: invalid operand
#CHECK: llihh	%r0, -1
#CHECK: error: invalid operand
#CHECK: llihh	%r0, 0x10000

	llihh	%r0, -1
	llihh	%r0, 0x10000

#CHECK: error: invalid operand
#CHECK: llihl	%r0, -1
#CHECK: error: invalid operand
#CHECK: llihl	%r0, 0x10000

	llihl	%r0, -1
	llihl	%r0, 0x10000

#CHECK: error: invalid operand
#CHECK: llilf	%r0, -1
#CHECK: error: invalid operand
#CHECK: llilf	%r0, 1 << 32

	llilf	%r0, -1
	llilf	%r0, 1 << 32

#CHECK: error: invalid operand
#CHECK: llilh	%r0, -1
#CHECK: error: invalid operand
#CHECK: llilh	%r0, 0x10000

	llilh	%r0, -1
	llilh	%r0, 0x10000

#CHECK: error: invalid operand
#CHECK: llill	%r0, -1
#CHECK: error: invalid operand
#CHECK: llill	%r0, 0x10000

	llill	%r0, -1
	llill	%r0, 0x10000

#CHECK: error: invalid operand
#CHECK: lm	%r0, %r0, 4096
#CHECK: error: invalid use of indexed addressing
#CHECK: lm	%r0, %r0, 0(%r1,%r2)

	lm	%r0, %r0, 4096
	lm	%r0, %r0, 0(%r1,%r2)

#CHECK: error: invalid use of indexed addressing
#CHECK: lmd	%r2, %r4, 160(%r1,%r15), 160(%r15)
#CHECK: error: invalid operand
#CHECK: lmd	%r2, %r4, -1(%r1), 160(%r15)
#CHECK: error: invalid operand
#CHECK: lmd	%r2, %r4, 4096(%r1), 160(%r15)
#CHECK: error: invalid operand
#CHECK: lmd	%r2, %r4, 0(%r1), -1(%r15)
#CHECK: error: invalid operand
#CHECK: lmd	%r2, %r4, 0(%r1), 4096(%r15)

        lmd	%r2, %r4, 160(%r1,%r15), 160(%r15)
        lmd	%r2, %r4, -1(%r1), 160(%r15)
        lmd	%r2, %r4, 4096(%r1), 160(%r15)
        lmd	%r2, %r4, 0(%r1), -1(%r15)
        lmd	%r2, %r4, 0(%r1), 4096(%r15)

#CHECK: error: invalid operand
#CHECK: lmg	%r0, %r0, -524289
#CHECK: error: invalid operand
#CHECK: lmg	%r0, %r0, 524288
#CHECK: error: invalid use of indexed addressing
#CHECK: lmg	%r0, %r0, 0(%r1,%r2)

	lmg	%r0, %r0, -524289
	lmg	%r0, %r0, 524288
	lmg	%r0, %r0, 0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: lmh	%r0, %r0, -524289
#CHECK: error: invalid operand
#CHECK: lmh	%r0, %r0, 524288
#CHECK: error: invalid use of indexed addressing
#CHECK: lmh	%r0, %r0, 0(%r1,%r2)

	lmh	%r0, %r0, -524289
	lmh	%r0, %r0, 524288
	lmh	%r0, %r0, 0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: lmy	%r0, %r0, -524289
#CHECK: error: invalid operand
#CHECK: lmy	%r0, %r0, 524288
#CHECK: error: invalid use of indexed addressing
#CHECK: lmy	%r0, %r0, 0(%r1,%r2)

	lmy	%r0, %r0, -524289
	lmy	%r0, %r0, 524288
	lmy	%r0, %r0, 0(%r1,%r2)

#CHECK: error: invalid register pair
#CHECK: lnxbr	%f0, %f2
#CHECK: error: invalid register pair
#CHECK: lnxbr	%f2, %f0

	lnxbr	%f0, %f2
	lnxbr	%f2, %f0

#CHECK: error: invalid register pair
#CHECK: lnxr	%f0, %f2
#CHECK: error: invalid register pair
#CHECK: lnxr	%f2, %f0

	lnxr	%f0, %f2
	lnxr	%f2, %f0

#CHECK: error: invalid operand
#CHECK: lpctl	-1
#CHECK: error: invalid operand
#CHECK: lpctl	4096
#CHECK: error: invalid use of indexed addressing
#CHECK: lpctl	0(%r1,%r2)

	lpctl	-1
	lpctl	4096
	lpctl	0(%r1,%r2)

#CHECK: error: instruction requires: interlocked-access1
#CHECK: lpd	%r0, 0, 0
	lpd	%r0, 0, 0

#CHECK: error: instruction requires: interlocked-access1
#CHECK: lpdg	%r0, 0, 0
	lpdg	%r0, 0, 0

#CHECK: error: invalid operand
#CHECK: lpp	-1
#CHECK: error: invalid operand
#CHECK: lpp	4096
#CHECK: error: invalid use of indexed addressing
#CHECK: lpp	0(%r1,%r2)

	lpp	-1
	lpp	4096
	lpp	0(%r1,%r2)

#CHECK: error: invalid register pair
#CHECK: lpq	%r1, 0
#CHECK: error: invalid operand
#CHECK: lpq	%r0, -524289
#CHECK: error: invalid operand
#CHECK: lpq	%r0, 524288

	lpq	%r1, 0
	lpq	%r0, -524289
	lpq	%r0, 524288

#CHECK: error: invalid operand
#CHECK: lptea	%r0, %r0, %r0, -1
#CHECK: error: invalid operand
#CHECK: lptea	%r0, %r0, %r0, 16

	lptea	%r0, %r0, %r0, -1
	lptea	%r0, %r0, %r0, 16

#CHECK: error: invalid operand
#CHECK: lpsw	-1
#CHECK: error: invalid operand
#CHECK: lpsw	4096
#CHECK: error: invalid use of indexed addressing
#CHECK: lpsw	0(%r1,%r2)

	lpsw	-1
	lpsw	4096
	lpsw	0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: lpswe	-1
#CHECK: error: invalid operand
#CHECK: lpswe	4096
#CHECK: error: invalid use of indexed addressing
#CHECK: lpswe	0(%r1,%r2)

	lpswe	-1
	lpswe	4096
	lpswe	0(%r1,%r2)

#CHECK: error: invalid register pair
#CHECK: lpxbr	%f0, %f2
#CHECK: error: invalid register pair
#CHECK: lpxbr	%f2, %f0

	lpxbr	%f0, %f2
	lpxbr	%f2, %f0

#CHECK: error: invalid register pair
#CHECK: lpxr	%f0, %f2
#CHECK: error: invalid register pair
#CHECK: lpxr	%f2, %f0

	lpxr	%f0, %f2
	lpxr	%f2, %f0

#CHECK: error: invalid operand
#CHECK: lra	%r0, -1
#CHECK: error: invalid operand
#CHECK: lra	%r0, 4096

	lra	%r0, -1
	lra	%r0, 4096

#CHECK: error: invalid operand
#CHECK: lrag	%r0, -524289
#CHECK: error: invalid operand
#CHECK: lrag	%r0, 524288

	lrag	%r0, -524289
	lrag	%r0, 524288

#CHECK: error: invalid operand
#CHECK: lray	%r0, -524289
#CHECK: error: invalid operand
#CHECK: lray	%r0, 524288

	lray	%r0, -524289
	lray	%r0, 524288

#CHECK: error: invalid register pair
#CHECK: lrdr	%f0, %f2

	lrdr	%f0, %f2

#CHECK: error: offset out of range
#CHECK: lrl	%r0, -0x1000000002
#CHECK: error: offset out of range
#CHECK: lrl	%r0, -1
#CHECK: error: offset out of range
#CHECK: lrl	%r0, 1
#CHECK: error: offset out of range
#CHECK: lrl	%r0, 0x100000000
#CHECK: error: offset out of range
#CHECK: lrl	%r1, __unnamed_1+3564822854692

	lrl	%r0, -0x1000000002
	lrl	%r0, -1
	lrl	%r0, 1
	lrl	%r0, 0x100000000
	lrl	%r1, __unnamed_1+3564822854692

#CHECK: error: invalid operand
#CHECK: lrv	%r0, -524289
#CHECK: error: invalid operand
#CHECK: lrv	%r0, 524288

	lrv	%r0, -524289
	lrv	%r0, 524288

#CHECK: error: invalid operand
#CHECK: lrvg	%r0, -524289
#CHECK: error: invalid operand
#CHECK: lrvg	%r0, 524288

	lrvg	%r0, -524289
	lrvg	%r0, 524288

#CHECK: error: invalid operand
#CHECK: lsctl	-1
#CHECK: error: invalid operand
#CHECK: lsctl	4096
#CHECK: error: invalid use of indexed addressing
#CHECK: lsctl	0(%r1,%r2)

	lsctl	-1
	lsctl	4096
	lsctl	0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: lt	%r0, -524289
#CHECK: error: invalid operand
#CHECK: lt	%r0, 524288

	lt	%r0, -524289
	lt	%r0, 524288

#CHECK: error: invalid operand
#CHECK: ltg	%r0, -524289
#CHECK: error: invalid operand
#CHECK: ltg	%r0, 524288

	ltg	%r0, -524289
	ltg	%r0, 524288

#CHECK: error: invalid operand
#CHECK: ltgf	%r0, -524289
#CHECK: error: invalid operand
#CHECK: ltgf	%r0, 524288

	ltgf	%r0, -524289
	ltgf	%r0, 524288

#CHECK: error: invalid register pair
#CHECK: ltxbr	%f0, %f14
#CHECK: error: invalid register pair
#CHECK: ltxbr	%f14, %f0

	ltxbr	%f0, %f14
	ltxbr	%f14, %f0

#CHECK: error: invalid register pair
#CHECK: ltxr	%f0, %f14
#CHECK: error: invalid register pair
#CHECK: ltxr	%f14, %f0

	ltxr	%f0, %f14
	ltxr	%f14, %f0

#CHECK: error: invalid register pair
#CHECK: ltxtr	%f0, %f14
#CHECK: error: invalid register pair
#CHECK: ltxtr	%f14, %f0

	ltxtr	%f0, %f14
	ltxtr	%f14, %f0

#CHECK: error: invalid operand
#CHECK: lxd	%f0, -1
#CHECK: error: invalid operand
#CHECK: lxd	%f0, 4096
#CHECK: error: invalid register pair
#CHECK: lxd	%f2, 0

	lxd	%f0, -1
	lxd	%f0, 4096
	lxd	%f2, 0

#CHECK: error: invalid operand
#CHECK: lxdb	%f0, -1
#CHECK: error: invalid operand
#CHECK: lxdb	%f0, 4096
#CHECK: error: invalid register pair
#CHECK: lxdb	%f2, 0

	lxdb	%f0, -1
	lxdb	%f0, 4096
	lxdb	%f2, 0

#CHECK: error: invalid register pair
#CHECK: lxdbr	%f2, %f0

	lxdbr	%f2, %f0

#CHECK: error: invalid register pair
#CHECK: lxdr	%f2, %f0

	lxdr	%f2, %f0

#CHECK: error: invalid operand
#CHECK: lxdtr	%f0, %f0, -1
#CHECK: error: invalid operand
#CHECK: lxdtr	%f0, %f0, 16
#CHECK: error: invalid register pair
#CHECK: lxdtr	%f2, %f0, 0

	lxdtr	%f0, %f0, -1
	lxdtr	%f0, %f0, 16
	lxdtr	%f2, %f0, 0

#CHECK: error: invalid operand
#CHECK: lxe	%f0, -1
#CHECK: error: invalid operand
#CHECK: lxe	%f0, 4096
#CHECK: error: invalid register pair
#CHECK: lxe	%f2, 0

	lxe	%f0, -1
	lxe	%f0, 4096
	lxe	%f2, 0

#CHECK: error: invalid operand
#CHECK: lxeb	%f0, -1
#CHECK: error: invalid operand
#CHECK: lxeb	%f0, 4096
#CHECK: error: invalid register pair
#CHECK: lxeb	%f2, 0

	lxeb	%f0, -1
	lxeb	%f0, 4096
	lxeb	%f2, 0

#CHECK: error: invalid register pair
#CHECK: lxebr	%f2, %f0

	lxebr	%f2, %f0

#CHECK: error: invalid register pair
#CHECK: lxer	%f2, %f0

	lxer	%f2, %f0

#CHECK: error: invalid register pair
#CHECK: lxr	%f0, %f2
#CHECK: error: invalid register pair
#CHECK: lxr	%f2, %f0

	lxr	%f0, %f2
	lxr	%f2, %f0

#CHECK: error: invalid operand
#CHECK: ly	%r0, -524289
#CHECK: error: invalid operand
#CHECK: ly	%r0, 524288

	ly	%r0, -524289
	ly	%r0, 524288

#CHECK: error: invalid register pair
#CHECK: lzxr	%f2

	lzxr	%f2

#CHECK: error: invalid operand
#CHECK: m	%r0, -1
#CHECK: error: invalid operand
#CHECK: m	%r0, 4096
#CHECK: error: invalid register pair
#CHECK: m	%r1, 0

	m	%r0, -1
	m	%r0, 4096
	m	%r1, 0

#CHECK: error: invalid operand
#CHECK: mad	%f0, %f0, -1
#CHECK: error: invalid operand
#CHECK: mad	%f0, %f0, 4096

	mad	%f0, %f0, -1
	mad	%f0, %f0, 4096

#CHECK: error: invalid operand
#CHECK: madb	%f0, %f0, -1
#CHECK: error: invalid operand
#CHECK: madb	%f0, %f0, 4096

	madb	%f0, %f0, -1
	madb	%f0, %f0, 4096

#CHECK: error: invalid operand
#CHECK: mae	%f0, %f0, -1
#CHECK: error: invalid operand
#CHECK: mae	%f0, %f0, 4096

	mae	%f0, %f0, -1
	mae	%f0, %f0, 4096

#CHECK: error: invalid operand
#CHECK: maeb	%f0, %f0, -1
#CHECK: error: invalid operand
#CHECK: maeb	%f0, %f0, 4096

	maeb	%f0, %f0, -1
	maeb	%f0, %f0, 4096

#CHECK: error: invalid operand
#CHECK: may	%f0, %f0, -1
#CHECK: error: invalid operand
#CHECK: may	%f0, %f0, 4096
#CHECK: error: invalid register pair
#CHECK: may	%f2, %f0, 0

	may	%f0, %f0, -1
	may	%f0, %f0, 4096
	may	%f2, %f0, 0

#CHECK: error: invalid operand
#CHECK: mayh	%f0, %f0, -1
#CHECK: error: invalid operand
#CHECK: mayh	%f0, %f0, 4096

	mayh	%f0, %f0, -1
	mayh	%f0, %f0, 4096

#CHECK: error: invalid operand
#CHECK: mayl	%f0, %f0, -1
#CHECK: error: invalid operand
#CHECK: mayl	%f0, %f0, 4096

	mayl	%f0, %f0, -1
	mayl	%f0, %f0, 4096

#CHECK: error: invalid register pair
#CHECK: mayr	%f2, %f0, %f0

	mayr	%f2, %f0, %f0

#CHECK: error: invalid operand
#CHECK: mc	-1, 0
#CHECK: error: invalid operand
#CHECK: mc	4096, 0
#CHECK: error: invalid use of indexed addressing
#CHECK: mc	0(%r1,%r2), 0
#CHECK: error: invalid operand
#CHECK: mc	0, -1
#CHECK: error: invalid operand
#CHECK: mc	0, 256

	mc	-1, 0
	mc	4096, 0
	mc	0(%r1,%r2), 0
	mc	0, -1
	mc	0, 256

#CHECK: error: invalid operand
#CHECK: md	%f0, -1
#CHECK: error: invalid operand
#CHECK: md	%f0, 4096

	md	%f0, -1
	md	%f0, 4096

#CHECK: error: invalid operand
#CHECK: mdb	%f0, -1
#CHECK: error: invalid operand
#CHECK: mdb	%f0, 4096

	mdb	%f0, -1
	mdb	%f0, 4096

#CHECK: error: invalid operand
#CHECK: mde	%f0, -1
#CHECK: error: invalid operand
#CHECK: mde	%f0, 4096

	mde	%f0, -1
	mde	%f0, 4096

#CHECK: error: invalid operand
#CHECK: mdeb	%f0, -1
#CHECK: error: invalid operand
#CHECK: mdeb	%f0, 4096

	mdeb	%f0, -1
	mdeb	%f0, 4096

#CHECK: error: instruction requires: fp-extension
#CHECK: mdtra	%f0, %f0, %f0, 0

	mdtra	%f0, %f0, %f0, 0

#CHECK: error: invalid operand
#CHECK: me	%f0, -1
#CHECK: error: invalid operand
#CHECK: me	%f0, 4096

	me	%f0, -1
	me	%f0, 4096

#CHECK: error: invalid operand
#CHECK: mee	%f0, -1
#CHECK: error: invalid operand
#CHECK: mee	%f0, 4096

	mee	%f0, -1
	mee	%f0, 4096

#CHECK: error: invalid operand
#CHECK: meeb	%f0, -1
#CHECK: error: invalid operand
#CHECK: meeb	%f0, 4096

	meeb	%f0, -1
	meeb	%f0, 4096

#CHECK: error: invalid operand
#CHECK: mfy	%r0, -524289
#CHECK: error: invalid operand
#CHECK: mfy	%r0, 524288
#CHECK: error: invalid register pair
#CHECK: mfy	%r1, 0

	mfy	%r0, -524289
	mfy	%r0, 524288
	mfy	%r1, 0

#CHECK: error: invalid operand
#CHECK: mghi	%r0, -32769
#CHECK: error: invalid operand
#CHECK: mghi	%r0, 32768
#CHECK: error: invalid operand
#CHECK: mghi	%r0, foo

	mghi	%r0, -32769
	mghi	%r0, 32768
	mghi	%r0, foo

#CHECK: error: invalid operand
#CHECK: mh	%r0, -1
#CHECK: error: invalid operand
#CHECK: mh	%r0, 4096

	mh	%r0, -1
	mh	%r0, 4096

#CHECK: error: invalid operand
#CHECK: mhi	%r0, -32769
#CHECK: error: invalid operand
#CHECK: mhi	%r0, 32768
#CHECK: error: invalid operand
#CHECK: mhi	%r0, foo

	mhi	%r0, -32769
	mhi	%r0, 32768
	mhi	%r0, foo

#CHECK: error: invalid operand
#CHECK: mhy	%r0, -524289
#CHECK: error: invalid operand
#CHECK: mhy	%r0, 524288

	mhy	%r0, -524289
	mhy	%r0, 524288

#CHECK: error: invalid operand
#CHECK: ml	%r0, -524289
#CHECK: error: invalid operand
#CHECK: ml	%r0, 524288
#CHECK: error: invalid register pair
#CHECK: ml	%r1, 0

	ml	%r0, -524289
	ml	%r0, 524288
	ml	%r1, 0

#CHECK: error: invalid operand
#CHECK: mlg	%r0, -524289
#CHECK: error: invalid operand
#CHECK: mlg	%r0, 524288
#CHECK: error: invalid register pair
#CHECK: mlg	%r1, 0

	mlg	%r0, -524289
	mlg	%r0, 524288
	mlg	%r1, 0

#CHECK: error: invalid register pair
#CHECK: mlgr	%r1, %r0

	mlgr	%r1, %r0

#CHECK: error: invalid register pair
#CHECK: mlr	%r1, %r0

	mlr	%r1, %r0

#CHECK: error: missing length in address
#CHECK: mp	0, 0(1)
#CHECK: error: missing length in address
#CHECK: mp	0(1), 0
#CHECK: error: missing length in address
#CHECK: mp	0(%r1), 0(1,%r1)
#CHECK: error: missing length in address
#CHECK: mp	0(1,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: mp	0(0,%r1), 0(1,%r1)
#CHECK: error: invalid operand
#CHECK: mp	0(1,%r1), 0(0,%r1)
#CHECK: error: invalid operand
#CHECK: mp	0(17,%r1), 0(1,%r1)
#CHECK: error: invalid operand
#CHECK: mp	0(1,%r1), 0(17,%r1)
#CHECK: error: invalid operand
#CHECK: mp	-1(1,%r1), 0(1,%r1)
#CHECK: error: invalid operand
#CHECK: mp	4096(1,%r1), 0(1,%r1)
#CHECK: error: invalid operand
#CHECK: mp	0(1,%r1), -1(1,%r1)
#CHECK: error: invalid operand
#CHECK: mp	0(1,%r1), 4096(1,%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: mp	0(%r1,%r2), 0(1,%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: mp	0(1,%r2), 0(%r1,%r2)
#CHECK: error: unknown token in expression
#CHECK: mp	0(-), 0(1)

	mp	0, 0(1)
	mp	0(1), 0
	mp	0(%r1), 0(1,%r1)
	mp	0(1,%r1), 0(%r1)
	mp	0(0,%r1), 0(1,%r1)
	mp	0(1,%r1), 0(0,%r1)
	mp	0(17,%r1), 0(1,%r1)
	mp	0(1,%r1), 0(17,%r1)
	mp	-1(1,%r1), 0(1,%r1)
	mp	4096(1,%r1), 0(1,%r1)
	mp	0(1,%r1), -1(1,%r1)
	mp	0(1,%r1), 4096(1,%r1)
	mp	0(%r1,%r2), 0(1,%r1)
	mp	0(1,%r2), 0(%r1,%r2)
	mp	0(-), 0(1)

#CHECK: error: invalid register pair
#CHECK: mr	%r1, %r0

	mr	%r1, %r0

#CHECK: error: invalid operand
#CHECK: ms	%r0, -1
#CHECK: error: invalid operand
#CHECK: ms	%r0, 4096

	ms	%r0, -1
	ms	%r0, 4096

#CHECK: error: invalid operand
#CHECK: msch	-1
#CHECK: error: invalid operand
#CHECK: msch	4096
#CHECK: error: invalid use of indexed addressing
#CHECK: msch	0(%r1,%r2)

	msch	-1
	msch	4096
	msch	0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: msd	%f0, %f0, -1
#CHECK: error: invalid operand
#CHECK: msd	%f0, %f0, 4096

	msd	%f0, %f0, -1
	msd	%f0, %f0, 4096

#CHECK: error: invalid operand
#CHECK: msdb	%f0, %f0, -1
#CHECK: error: invalid operand
#CHECK: msdb	%f0, %f0, 4096

	msdb	%f0, %f0, -1
	msdb	%f0, %f0, 4096

#CHECK: error: invalid operand
#CHECK: mse	%f0, %f0, -1
#CHECK: error: invalid operand
#CHECK: mse	%f0, %f0, 4096

	mse	%f0, %f0, -1
	mse	%f0, %f0, 4096

#CHECK: error: invalid operand
#CHECK: mseb	%f0, %f0, -1
#CHECK: error: invalid operand
#CHECK: mseb	%f0, %f0, 4096

	mseb	%f0, %f0, -1
	mseb	%f0, %f0, 4096

#CHECK: error: invalid operand
#CHECK: msfi	%r0, (-1 << 31) - 1
#CHECK: error: invalid operand
#CHECK: msfi	%r0, (1 << 31)

	msfi	%r0, (-1 << 31) - 1
	msfi	%r0, (1 << 31)

#CHECK: error: invalid operand
#CHECK: msg	%r0, -524289
#CHECK: error: invalid operand
#CHECK: msg	%r0, 524288

	msg	%r0, -524289
	msg	%r0, 524288

#CHECK: error: invalid operand
#CHECK: msgf	%r0, -524289
#CHECK: error: invalid operand
#CHECK: msgf	%r0, 524288

	msgf	%r0, -524289
	msgf	%r0, 524288

#CHECK: error: invalid operand
#CHECK: msgfi	%r0, (-1 << 31) - 1
#CHECK: error: invalid operand
#CHECK: msgfi	%r0, (1 << 31)

	msgfi	%r0, (-1 << 31) - 1
	msgfi	%r0, (1 << 31)

#CHECK: error: invalid register pair
#CHECK: msta	%r1

	msta	%r1

#CHECK: error: invalid operand
#CHECK: msy	%r0, -524289
#CHECK: error: invalid operand
#CHECK: msy	%r0, 524288

	msy	%r0, -524289
	msy	%r0, 524288

#CHECK: error: missing length in address
#CHECK: mvc	0, 0
#CHECK: error: missing length in address
#CHECK: mvc	0(%r1), 0(%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: mvc	0(1,%r1), 0(2,%r1)
#CHECK: error: invalid operand
#CHECK: mvc	0(0,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: mvc	0(257,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: mvc	-1(1,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: mvc	4096(1,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: mvc	0(1,%r1), -1(%r1)
#CHECK: error: invalid operand
#CHECK: mvc	0(1,%r1), 4096(%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: mvc	0(%r1,%r2), 0(%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: mvc	0(1,%r2), 0(%r1,%r2)
#CHECK: error: unknown token in expression
#CHECK: mvc	0(-), 0

	mvc	0, 0
	mvc	0(%r1), 0(%r1)
	mvc	0(1,%r1), 0(2,%r1)
	mvc	0(0,%r1), 0(%r1)
	mvc	0(257,%r1), 0(%r1)
	mvc	-1(1,%r1), 0(%r1)
	mvc	4096(1,%r1), 0(%r1)
	mvc	0(1,%r1), -1(%r1)
	mvc	0(1,%r1), 4096(%r1)
	mvc	0(%r1,%r2), 0(%r1)
	mvc	0(1,%r2), 0(%r1,%r2)
	mvc	0(-), 0

#CHECK: error: invalid use of indexed addressing
#CHECK: mvcdk	160(%r1,%r15),160(%r15)
#CHECK: error: invalid operand
#CHECK: mvcdk	-1(%r1),160(%r15)
#CHECK: error: invalid operand
#CHECK: mvcdk	4096(%r1),160(%r15)
#CHECK: error: invalid operand
#CHECK: mvcdk	0(%r1),-1(%r15)
#CHECK: error: invalid operand
#CHECK: mvcdk	0(%r1),4096(%r15)

        mvcdk	160(%r1,%r15),160(%r15)
        mvcdk	-1(%r1),160(%r15)
        mvcdk	4096(%r1),160(%r15)
        mvcdk	0(%r1),-1(%r15)
        mvcdk	0(%r1),4096(%r15)

#CHECK: error: missing length in address
#CHECK: mvcin	0, 0
#CHECK: error: missing length in address
#CHECK: mvcin	0(%r1), 0(%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: mvcin	0(1,%r1), 0(2,%r1)
#CHECK: error: invalid operand
#CHECK: mvcin	0(0,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: mvcin	0(257,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: mvcin	-1(1,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: mvcin	4096(1,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: mvcin	0(1,%r1), -1(%r1)
#CHECK: error: invalid operand
#CHECK: mvcin	0(1,%r1), 4096(%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: mvcin	0(%r1,%r2), 0(%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: mvcin	0(1,%r2), 0(%r1,%r2)
#CHECK: error: unknown token in expression
#CHECK: mvcin	0(-), 0

	mvcin	0, 0
	mvcin	0(%r1), 0(%r1)
	mvcin	0(1,%r1), 0(2,%r1)
	mvcin	0(0,%r1), 0(%r1)
	mvcin	0(257,%r1), 0(%r1)
	mvcin	-1(1,%r1), 0(%r1)
	mvcin	4096(1,%r1), 0(%r1)
	mvcin	0(1,%r1), -1(%r1)
	mvcin	0(1,%r1), 4096(%r1)
	mvcin	0(%r1,%r2), 0(%r1)
	mvcin	0(1,%r2), 0(%r1,%r2)
	mvcin	0(-), 0

#CHECK: error: invalid use of indexed addressing
#CHECK: mvck	0(%r1,%r1), 0(2,%r1), %r3
#CHECK: error: invalid operand
#CHECK: mvck	-1(%r1,%r1), 0(%r1), %r3
#CHECK: error: invalid operand
#CHECK: mvck	4096(%r1,%r1), 0(%r1), %r3
#CHECK: error: invalid operand
#CHECK: mvck	0(%r1,%r1), -1(%r1), %r3
#CHECK: error: invalid operand
#CHECK: mvck	0(%r1,%r1), 4096(%r1), %r3
#CHECK: error: invalid use of indexed addressing
#CHECK: mvck	0(%r1,%r2), 0(%r1,%r2), %r3
#CHECK: error: unexpected token in address
#CHECK: mvck	0(-), 0, %r3

	mvck	0(%r1,%r1), 0(2,%r1), %r3
	mvck	-1(%r1,%r1), 0(%r1), %r3
	mvck	4096(%r1,%r1), 0(%r1), %r3
	mvck	0(%r1,%r1), -1(%r1), %r3
	mvck	0(%r1,%r1), 4096(%r1), %r3
	mvck	0(%r1,%r2), 0(%r1,%r2), %r3
	mvck	0(-), 0, %r3

#CHECK: error: invalid register pair
#CHECK: mvcl	%r1, %r0
#CHECK: error: invalid register pair
#CHECK: mvcl	%r0, %r1

	mvcl	%r1, %r0
	mvcl	%r0, %r1

#CHECK: error: invalid register pair
#CHECK: mvcle	%r1, %r0
#CHECK: error: invalid register pair
#CHECK: mvcle	%r0, %r1
#CHECK: error: invalid operand
#CHECK: mvcle	%r0, %r0, -1
#CHECK: error: invalid operand
#CHECK: mvcle	%r0, %r0, 4096

	mvcle	%r1, %r0, 0
	mvcle	%r0, %r1, 0
	mvcle	%r0, %r0, -1
	mvcle	%r0, %r0, 4096

#CHECK: error: invalid register pair
#CHECK: mvclu	%r1, %r0
#CHECK: error: invalid register pair
#CHECK: mvclu	%r0, %r1
#CHECK: error: invalid operand
#CHECK: mvclu	%r0, %r0, -524289
#CHECK: error: invalid operand
#CHECK: mvclu	%r0, %r0, 524288

	mvclu	%r1, %r0, 0
	mvclu	%r0, %r1, 0
	mvclu	%r0, %r0, -524289
	mvclu	%r0, %r0, 524288

#CHECK: error: invalid use of indexed addressing
#CHECK: mvcos	160(%r1,%r15), 160(%r15), %r2
#CHECK: error: invalid operand
#CHECK: mvcos	-1(%r1), 160(%r15), %r2
#CHECK: error: invalid operand
#CHECK: mvcos	4096(%r1), 160(%r15), %r2
#CHECK: error: invalid operand
#CHECK: mvcos	0(%r1), -1(%r15), %r2
#CHECK: error: invalid operand
#CHECK: mvcos	0(%r1), 4096(%r15), %r2

        mvcos	160(%r1,%r15), 160(%r15), %r2
        mvcos	-1(%r1), 160(%r15), %r2
        mvcos	4096(%r1), 160(%r15), %r2
        mvcos	0(%r1), -1(%r15), %r2
        mvcos	0(%r1), 4096(%r15), %r2

#CHECK: error: invalid use of indexed addressing
#CHECK: mvcp	0(%r1,%r1), 0(2,%r1), %r3
#CHECK: error: invalid operand
#CHECK: mvcp	-1(%r1,%r1), 0(%r1), %r3
#CHECK: error: invalid operand
#CHECK: mvcp	4096(%r1,%r1), 0(%r1), %r3
#CHECK: error: invalid operand
#CHECK: mvcp	0(%r1,%r1), -1(%r1), %r3
#CHECK: error: invalid operand
#CHECK: mvcp	0(%r1,%r1), 4096(%r1), %r3
#CHECK: error: invalid use of indexed addressing
#CHECK: mvcp	0(%r1,%r2), 0(%r1,%r2), %r3
#CHECK: error: unexpected token in address
#CHECK: mvcp	0(-), 0, %r3

	mvcp	0(%r1,%r1), 0(2,%r1), %r3
	mvcp	-1(%r1,%r1), 0(%r1), %r3
	mvcp	4096(%r1,%r1), 0(%r1), %r3
	mvcp	0(%r1,%r1), -1(%r1), %r3
	mvcp	0(%r1,%r1), 4096(%r1), %r3
	mvcp	0(%r1,%r2), 0(%r1,%r2), %r3
	mvcp	0(-), 0, %r3

#CHECK: error: invalid use of indexed addressing
#CHECK: mvcs	0(%r1,%r1), 0(2,%r1), %r3
#CHECK: error: invalid operand
#CHECK: mvcs	-1(%r1,%r1), 0(%r1), %r3
#CHECK: error: invalid operand
#CHECK: mvcs	4096(%r1,%r1), 0(%r1), %r3
#CHECK: error: invalid operand
#CHECK: mvcs	0(%r1,%r1), -1(%r1), %r3
#CHECK: error: invalid operand
#CHECK: mvcs	0(%r1,%r1), 4096(%r1), %r3
#CHECK: error: invalid use of indexed addressing
#CHECK: mvcs	0(%r1,%r2), 0(%r1,%r2), %r3
#CHECK: error: unexpected token in address
#CHECK: mvcs	0(-), 0, %r3

	mvcs	0(%r1,%r1), 0(2,%r1), %r3
	mvcs	-1(%r1,%r1), 0(%r1), %r3
	mvcs	4096(%r1,%r1), 0(%r1), %r3
	mvcs	0(%r1,%r1), -1(%r1), %r3
	mvcs	0(%r1,%r1), 4096(%r1), %r3
	mvcs	0(%r1,%r2), 0(%r1,%r2), %r3
	mvcs	0(-), 0, %r3

#CHECK: error: invalid use of indexed addressing
#CHECK: mvcsk	160(%r1,%r15),160(%r15)
#CHECK: error: invalid operand
#CHECK: mvcsk	-1(%r1),160(%r15)
#CHECK: error: invalid operand
#CHECK: mvcsk	4096(%r1),160(%r15)
#CHECK: error: invalid operand
#CHECK: mvcsk	0(%r1),-1(%r15)
#CHECK: error: invalid operand
#CHECK: mvcsk	0(%r1),4096(%r15)

        mvcsk	160(%r1,%r15),160(%r15)
        mvcsk	-1(%r1),160(%r15)
        mvcsk	4096(%r1),160(%r15)
        mvcsk	0(%r1),-1(%r15)
        mvcsk	0(%r1),4096(%r15)

#CHECK: error: invalid operand
#CHECK: mvghi	-1, 0
#CHECK: error: invalid operand
#CHECK: mvghi	4096, 0
#CHECK: error: invalid use of indexed addressing
#CHECK: mvghi	0(%r1,%r2), 0
#CHECK: error: invalid operand
#CHECK: mvghi	0, -32769
#CHECK: error: invalid operand
#CHECK: mvghi	0, 32768

	mvghi	-1, 0
	mvghi	4096, 0
	mvghi	0(%r1,%r2), 0
	mvghi	0, -32769
	mvghi	0, 32768

#CHECK: error: invalid operand
#CHECK: mvhhi	-1, 0
#CHECK: error: invalid operand
#CHECK: mvhhi	4096, 0
#CHECK: error: invalid use of indexed addressing
#CHECK: mvhhi	0(%r1,%r2), 0
#CHECK: error: invalid operand
#CHECK: mvhhi	0, -32769
#CHECK: error: invalid operand
#CHECK: mvhhi	0, 32768

	mvhhi	-1, 0
	mvhhi	4096, 0
	mvhhi	0(%r1,%r2), 0
	mvhhi	0, -32769
	mvhhi	0, 32768

#CHECK: error: invalid operand
#CHECK: mvhi	-1, 0
#CHECK: error: invalid operand
#CHECK: mvhi	4096, 0
#CHECK: error: invalid use of indexed addressing
#CHECK: mvhi	0(%r1,%r2), 0
#CHECK: error: invalid operand
#CHECK: mvhi	0, -32769
#CHECK: error: invalid operand
#CHECK: mvhi	0, 32768

	mvhi	-1, 0
	mvhi	4096, 0
	mvhi	0(%r1,%r2), 0
	mvhi	0, -32769
	mvhi	0, 32768

#CHECK: error: invalid operand
#CHECK: mvi	-1, 0
#CHECK: error: invalid operand
#CHECK: mvi	4096, 0
#CHECK: error: invalid use of indexed addressing
#CHECK: mvi	0(%r1,%r2), 0
#CHECK: error: invalid operand
#CHECK: mvi	0, -1
#CHECK: error: invalid operand
#CHECK: mvi	0, 256

	mvi	-1, 0
	mvi	4096, 0
	mvi	0(%r1,%r2), 0
	mvi	0, -1
	mvi	0, 256

#CHECK: error: invalid operand
#CHECK: mviy	-524289, 0
#CHECK: error: invalid operand
#CHECK: mviy	524288, 0
#CHECK: error: invalid use of indexed addressing
#CHECK: mviy	0(%r1,%r2), 0
#CHECK: error: invalid operand
#CHECK: mviy	0, -1
#CHECK: error: invalid operand
#CHECK: mviy	0, 256

	mviy	-524289, 0
	mviy	524288, 0
	mviy	0(%r1,%r2), 0
	mviy	0, -1
	mviy	0, 256

#CHECK: error: missing length in address
#CHECK: mvn	0, 0
#CHECK: error: missing length in address
#CHECK: mvn	0(%r1), 0(%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: mvn	0(1,%r1), 0(2,%r1)
#CHECK: error: invalid operand
#CHECK: mvn	0(0,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: mvn	0(257,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: mvn	-1(1,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: mvn	4096(1,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: mvn	0(1,%r1), -1(%r1)
#CHECK: error: invalid operand
#CHECK: mvn	0(1,%r1), 4096(%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: mvn	0(%r1,%r2), 0(%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: mvn	0(1,%r2), 0(%r1,%r2)
#CHECK: error: unknown token in expression
#CHECK: mvn	0(-), 0

	mvn	0, 0
	mvn	0(%r1), 0(%r1)
	mvn	0(1,%r1), 0(2,%r1)
	mvn	0(0,%r1), 0(%r1)
	mvn	0(257,%r1), 0(%r1)
	mvn	-1(1,%r1), 0(%r1)
	mvn	4096(1,%r1), 0(%r1)
	mvn	0(1,%r1), -1(%r1)
	mvn	0(1,%r1), 4096(%r1)
	mvn	0(%r1,%r2), 0(%r1)
	mvn	0(1,%r2), 0(%r1,%r2)
	mvn	0(-), 0

#CHECK: error: missing length in address
#CHECK: mvo	0, 0(1)
#CHECK: error: missing length in address
#CHECK: mvo	0(1), 0
#CHECK: error: missing length in address
#CHECK: mvo	0(%r1), 0(1,%r1)
#CHECK: error: missing length in address
#CHECK: mvo	0(1,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: mvo	0(0,%r1), 0(1,%r1)
#CHECK: error: invalid operand
#CHECK: mvo	0(1,%r1), 0(0,%r1)
#CHECK: error: invalid operand
#CHECK: mvo	0(17,%r1), 0(1,%r1)
#CHECK: error: invalid operand
#CHECK: mvo	0(1,%r1), 0(17,%r1)
#CHECK: error: invalid operand
#CHECK: mvo	-1(1,%r1), 0(1,%r1)
#CHECK: error: invalid operand
#CHECK: mvo	4096(1,%r1), 0(1,%r1)
#CHECK: error: invalid operand
#CHECK: mvo	0(1,%r1), -1(1,%r1)
#CHECK: error: invalid operand
#CHECK: mvo	0(1,%r1), 4096(1,%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: mvo	0(%r1,%r2), 0(1,%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: mvo	0(1,%r2), 0(%r1,%r2)
#CHECK: error: unknown token in expression
#CHECK: mvo	0(-), 0(1)

	mvo	0, 0(1)
	mvo	0(1), 0
	mvo	0(%r1), 0(1,%r1)
	mvo	0(1,%r1), 0(%r1)
	mvo	0(0,%r1), 0(1,%r1)
	mvo	0(1,%r1), 0(0,%r1)
	mvo	0(17,%r1), 0(1,%r1)
	mvo	0(1,%r1), 0(17,%r1)
	mvo	-1(1,%r1), 0(1,%r1)
	mvo	4096(1,%r1), 0(1,%r1)
	mvo	0(1,%r1), -1(1,%r1)
	mvo	0(1,%r1), 4096(1,%r1)
	mvo	0(%r1,%r2), 0(1,%r1)
	mvo	0(1,%r2), 0(%r1,%r2)
	mvo	0(-), 0(1)

#CHECK: error: missing length in address
#CHECK: mvz	0, 0
#CHECK: error: missing length in address
#CHECK: mvz	0(%r1), 0(%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: mvz	0(1,%r1), 0(2,%r1)
#CHECK: error: invalid operand
#CHECK: mvz	0(0,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: mvz	0(257,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: mvz	-1(1,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: mvz	4096(1,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: mvz	0(1,%r1), -1(%r1)
#CHECK: error: invalid operand
#CHECK: mvz	0(1,%r1), 4096(%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: mvz	0(%r1,%r2), 0(%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: mvz	0(1,%r2), 0(%r1,%r2)
#CHECK: error: unknown token in expression
#CHECK: mvz	0(-), 0

	mvz	0, 0
	mvz	0(%r1), 0(%r1)
	mvz	0(1,%r1), 0(2,%r1)
	mvz	0(0,%r1), 0(%r1)
	mvz	0(257,%r1), 0(%r1)
	mvz	-1(1,%r1), 0(%r1)
	mvz	4096(1,%r1), 0(%r1)
	mvz	0(1,%r1), -1(%r1)
	mvz	0(1,%r1), 4096(%r1)
	mvz	0(%r1,%r2), 0(%r1)
	mvz	0(1,%r2), 0(%r1,%r2)
	mvz	0(-), 0

#CHECK: error: invalid register pair
#CHECK: mxbr	%f0, %f2
#CHECK: error: invalid register pair
#CHECK: mxbr	%f2, %f0

	mxbr	%f0, %f2
	mxbr	%f2, %f0

#CHECK: error: invalid register pair
#CHECK: mxd	%f2, 0
#CHECK: error: invalid operand
#CHECK: mxd	%f0, -1
#CHECK: error: invalid operand
#CHECK: mxd	%f0, 4096

	mxd	%f2, 0
	mxd	%f0, -1
	mxd	%f0, 4096

#CHECK: error: invalid register pair
#CHECK: mxdb	%f2, 0
#CHECK: error: invalid operand
#CHECK: mxdb	%f0, -1
#CHECK: error: invalid operand
#CHECK: mxdb	%f0, 4096

	mxdb	%f2, 0
	mxdb	%f0, -1
	mxdb	%f0, 4096

#CHECK: error: invalid register pair
#CHECK: mxdbr	%f2, %f0

	mxdbr	%f2, %f0

#CHECK: error: invalid register pair
#CHECK: mxdr	%f2, %f0

	mxdr	%f2, %f0

#CHECK: error: invalid register pair
#CHECK: mxr	%f0, %f2
#CHECK: error: invalid register pair
#CHECK: mxr	%f2, %f0

	mxr	%f0, %f2
	mxr	%f2, %f0

#CHECK: error: invalid register pair
#CHECK: mxtr	%f0, %f0, %f2
#CHECK: error: invalid register pair
#CHECK: mxtr	%f0, %f2, %f0
#CHECK: error: invalid register pair
#CHECK: mxtr	%f2, %f0, %f0

	mxtr	%f0, %f0, %f2
	mxtr	%f0, %f2, %f0
	mxtr	%f2, %f0, %f0

#CHECK: error: instruction requires: fp-extension
#CHECK: mxtra	%f0, %f0, %f0, 0

	mxtra	%f0, %f0, %f0, 0

#CHECK: error: invalid operand
#CHECK: my	%f0, %f0, -1
#CHECK: error: invalid operand
#CHECK: my	%f0, %f0, 4096
#CHECK: error: invalid register pair
#CHECK: my	%f2, %f0, 0

	my	%f0, %f0, -1
	my	%f0, %f0, 4096
	my	%f2, %f0, 0

#CHECK: error: invalid operand
#CHECK: myh	%f0, %f0, -1
#CHECK: error: invalid operand
#CHECK: myh	%f0, %f0, 4096

	myh	%f0, %f0, -1
	myh	%f0, %f0, 4096

#CHECK: error: invalid operand
#CHECK: myl	%f0, %f0, -1
#CHECK: error: invalid operand
#CHECK: myl	%f0, %f0, 4096

	myl	%f0, %f0, -1
	myl	%f0, %f0, 4096

#CHECK: error: invalid register pair
#CHECK: myr	%f2, %f0, %f0

	myr	%f2, %f0, %f0

#CHECK: error: invalid operand
#CHECK: n	%r0, -1
#CHECK: error: invalid operand
#CHECK: n	%r0, 4096

	n	%r0, -1
	n	%r0, 4096

#CHECK: error: missing length in address
#CHECK: nc	0, 0
#CHECK: error: missing length in address
#CHECK: nc	0(%r1), 0(%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: nc	0(1,%r1), 0(2,%r1)
#CHECK: error: invalid operand
#CHECK: nc	0(0,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: nc	0(257,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: nc	-1(1,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: nc	4096(1,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: nc	0(1,%r1), -1(%r1)
#CHECK: error: invalid operand
#CHECK: nc	0(1,%r1), 4096(%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: nc	0(%r1,%r2), 0(%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: nc	0(1,%r2), 0(%r1,%r2)
#CHECK: error: unknown token in expression
#CHECK: nc	0(-), 0

	nc	0, 0
	nc	0(%r1), 0(%r1)
	nc	0(1,%r1), 0(2,%r1)
	nc	0(0,%r1), 0(%r1)
	nc	0(257,%r1), 0(%r1)
	nc	-1(1,%r1), 0(%r1)
	nc	4096(1,%r1), 0(%r1)
	nc	0(1,%r1), -1(%r1)
	nc	0(1,%r1), 4096(%r1)
	nc	0(%r1,%r2), 0(%r1)
	nc	0(1,%r2), 0(%r1,%r2)
	nc	0(-), 0

#CHECK: error: invalid operand
#CHECK: ng	%r0, -524289
#CHECK: error: invalid operand
#CHECK: ng	%r0, 524288

	ng	%r0, -524289
	ng	%r0, 524288

#CHECK: error: instruction requires: distinct-ops
#CHECK: ngrk	%r2,%r3,%r4

	ngrk	%r2,%r3,%r4

#CHECK: error: invalid operand
#CHECK: ni	-1, 0
#CHECK: error: invalid operand
#CHECK: ni	4096, 0
#CHECK: error: invalid use of indexed addressing
#CHECK: ni	0(%r1,%r2), 0
#CHECK: error: invalid operand
#CHECK: ni	0, -1
#CHECK: error: invalid operand
#CHECK: ni	0, 256

	ni	-1, 0
	ni	4096, 0
	ni	0(%r1,%r2), 0
	ni	0, -1
	ni	0, 256

#CHECK: error: invalid operand
#CHECK: nihf	%r0, -1
#CHECK: error: invalid operand
#CHECK: nihf	%r0, 1 << 32

	nihf	%r0, -1
	nihf	%r0, 1 << 32

#CHECK: error: invalid operand
#CHECK: nihh	%r0, -1
#CHECK: error: invalid operand
#CHECK: nihh	%r0, 0x10000

	nihh	%r0, -1
	nihh	%r0, 0x10000

#CHECK: error: invalid operand
#CHECK: nihl	%r0, -1
#CHECK: error: invalid operand
#CHECK: nihl	%r0, 0x10000

	nihl	%r0, -1
	nihl	%r0, 0x10000

#CHECK: error: invalid operand
#CHECK: nilf	%r0, -1
#CHECK: error: invalid operand
#CHECK: nilf	%r0, 1 << 32

	nilf	%r0, -1
	nilf	%r0, 1 << 32

#CHECK: error: invalid operand
#CHECK: nilh	%r0, -1
#CHECK: error: invalid operand
#CHECK: nilh	%r0, 0x10000

	nilh	%r0, -1
	nilh	%r0, 0x10000

#CHECK: error: invalid operand
#CHECK: nill	%r0, -1
#CHECK: error: invalid operand
#CHECK: nill	%r0, 0x10000

	nill	%r0, -1
	nill	%r0, 0x10000

#CHECK: error: invalid operand
#CHECK: niy	-524289, 0
#CHECK: error: invalid operand
#CHECK: niy	524288, 0
#CHECK: error: invalid use of indexed addressing
#CHECK: niy	0(%r1,%r2), 0
#CHECK: error: invalid operand
#CHECK: niy	0, -1
#CHECK: error: invalid operand
#CHECK: niy	0, 256

	niy	-524289, 0
	niy	524288, 0
	niy	0(%r1,%r2), 0
	niy	0, -1
	niy	0, 256

#CHECK: error: instruction requires: distinct-ops
#CHECK: nrk	%r2,%r3,%r4

	nrk	%r2,%r3,%r4

#CHECK: error: invalid operand
#CHECK: ny	%r0, -524289
#CHECK: error: invalid operand
#CHECK: ny	%r0, 524288

	ny	%r0, -524289
	ny	%r0, 524288

#CHECK: error: invalid operand
#CHECK: o	%r0, -1
#CHECK: error: invalid operand
#CHECK: o	%r0, 4096

	o	%r0, -1
	o	%r0, 4096

#CHECK: error: missing length in address
#CHECK: oc	0, 0
#CHECK: error: missing length in address
#CHECK: oc	0(%r1), 0(%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: oc	0(1,%r1), 0(2,%r1)
#CHECK: error: invalid operand
#CHECK: oc	0(0,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: oc	0(257,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: oc	-1(1,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: oc	4096(1,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: oc	0(1,%r1), -1(%r1)
#CHECK: error: invalid operand
#CHECK: oc	0(1,%r1), 4096(%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: oc	0(%r1,%r2), 0(%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: oc	0(1,%r2), 0(%r1,%r2)
#CHECK: error: unknown token in expression
#CHECK: oc	0(-), 0

	oc	0, 0
	oc	0(%r1), 0(%r1)
	oc	0(1,%r1), 0(2,%r1)
	oc	0(0,%r1), 0(%r1)
	oc	0(257,%r1), 0(%r1)
	oc	-1(1,%r1), 0(%r1)
	oc	4096(1,%r1), 0(%r1)
	oc	0(1,%r1), -1(%r1)
	oc	0(1,%r1), 4096(%r1)
	oc	0(%r1,%r2), 0(%r1)
	oc	0(1,%r2), 0(%r1,%r2)
	oc	0(-), 0

#CHECK: error: invalid operand
#CHECK: og	%r0, -524289
#CHECK: error: invalid operand
#CHECK: og	%r0, 524288

	og	%r0, -524289
	og	%r0, 524288

#CHECK: error: instruction requires: distinct-ops
#CHECK: ogrk	%r2,%r3,%r4

	ogrk	%r2,%r3,%r4

#CHECK: error: invalid operand
#CHECK: oi	-1, 0
#CHECK: error: invalid operand
#CHECK: oi	4096, 0
#CHECK: error: invalid use of indexed addressing
#CHECK: oi	0(%r1,%r2), 0
#CHECK: error: invalid operand
#CHECK: oi	0, -1
#CHECK: error: invalid operand
#CHECK: oi	0, 256

	oi	-1, 0
	oi	4096, 0
	oi	0(%r1,%r2), 0
	oi	0, -1
	oi	0, 256

#CHECK: error: invalid operand
#CHECK: oihf	%r0, -1
#CHECK: error: invalid operand
#CHECK: oihf	%r0, 1 << 32

	oihf	%r0, -1
	oihf	%r0, 1 << 32

#CHECK: error: invalid operand
#CHECK: oihh	%r0, -1
#CHECK: error: invalid operand
#CHECK: oihh	%r0, 0x10000

	oihh	%r0, -1
	oihh	%r0, 0x10000

#CHECK: error: invalid operand
#CHECK: oihl	%r0, -1
#CHECK: error: invalid operand
#CHECK: oihl	%r0, 0x10000

	oihl	%r0, -1
	oihl	%r0, 0x10000

#CHECK: error: invalid operand
#CHECK: oilf	%r0, -1
#CHECK: error: invalid operand
#CHECK: oilf	%r0, 1 << 32

	oilf	%r0, -1
	oilf	%r0, 1 << 32

#CHECK: error: invalid operand
#CHECK: oilh	%r0, -1
#CHECK: error: invalid operand
#CHECK: oilh	%r0, 0x10000

	oilh	%r0, -1
	oilh	%r0, 0x10000

#CHECK: error: invalid operand
#CHECK: oill	%r0, -1
#CHECK: error: invalid operand
#CHECK: oill	%r0, 0x10000

	oill	%r0, -1
	oill	%r0, 0x10000

#CHECK: error: invalid operand
#CHECK: oiy	-524289, 0
#CHECK: error: invalid operand
#CHECK: oiy	524288, 0
#CHECK: error: invalid use of indexed addressing
#CHECK: oiy	0(%r1,%r2), 0
#CHECK: error: invalid operand
#CHECK: oiy	0, -1
#CHECK: error: invalid operand
#CHECK: oiy	0, 256

	oiy	-524289, 0
	oiy	524288, 0
	oiy	0(%r1,%r2), 0
	oiy	0, -1
	oiy	0, 256

#CHECK: error: instruction requires: distinct-ops
#CHECK: ork	%r2,%r3,%r4

	ork	%r2,%r3,%r4

#CHECK: error: invalid operand
#CHECK: oy	%r0, -524289
#CHECK: error: invalid operand
#CHECK: oy	%r0, 524288

	oy	%r0, -524289
	oy	%r0, 524288

#CHECK: error: missing length in address
#CHECK: pack	0, 0(1)
#CHECK: error: missing length in address
#CHECK: pack	0(1), 0
#CHECK: error: missing length in address
#CHECK: pack	0(%r1), 0(1,%r1)
#CHECK: error: missing length in address
#CHECK: pack	0(1,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: pack	0(0,%r1), 0(1,%r1)
#CHECK: error: invalid operand
#CHECK: pack	0(1,%r1), 0(0,%r1)
#CHECK: error: invalid operand
#CHECK: pack	0(17,%r1), 0(1,%r1)
#CHECK: error: invalid operand
#CHECK: pack	0(1,%r1), 0(17,%r1)
#CHECK: error: invalid operand
#CHECK: pack	-1(1,%r1), 0(1,%r1)
#CHECK: error: invalid operand
#CHECK: pack	4096(1,%r1), 0(1,%r1)
#CHECK: error: invalid operand
#CHECK: pack	0(1,%r1), -1(1,%r1)
#CHECK: error: invalid operand
#CHECK: pack	0(1,%r1), 4096(1,%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: pack	0(%r1,%r2), 0(1,%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: pack	0(1,%r2), 0(%r1,%r2)
#CHECK: error: unknown token in expression
#CHECK: pack	0(-), 0(1)

	pack	0, 0(1)
	pack	0(1), 0
	pack	0(%r1), 0(1,%r1)
	pack	0(1,%r1), 0(%r1)
	pack	0(0,%r1), 0(1,%r1)
	pack	0(1,%r1), 0(0,%r1)
	pack	0(17,%r1), 0(1,%r1)
	pack	0(1,%r1), 0(17,%r1)
	pack	-1(1,%r1), 0(1,%r1)
	pack	4096(1,%r1), 0(1,%r1)
	pack	0(1,%r1), -1(1,%r1)
	pack	0(1,%r1), 4096(1,%r1)
	pack	0(%r1,%r2), 0(1,%r1)
	pack	0(1,%r2), 0(%r1,%r2)
	pack	0(-), 0(1)

#CHECK: error: invalid operand
#CHECK: pc	-1
#CHECK: error: invalid operand
#CHECK: pc	4096
#CHECK: error: invalid use of indexed addressing
#CHECK: pc	0(%r1,%r2)

	pc	-1
	pc	4096
	pc	0(%r1,%r2)

#CHECK: error: instruction requires: message-security-assist-extension4
#CHECK: pcc

	pcc

#CHECK: error: instruction requires: message-security-assist-extension3
#CHECK: pckmo

	pckmo

#CHECK: error: invalid operand
#CHECK: pfd	-1, 0
#CHECK: error: invalid operand
#CHECK: pfd	16, 0
#CHECK: error: invalid operand
#CHECK: pfd	1, -524289
#CHECK: error: invalid operand
#CHECK: pfd	1, 524288

	pfd	-1, 0
	pfd	16, 0
	pfd	1, -524289
	pfd	1, 524288

#CHECK: error: invalid operand
#CHECK: pfdrl	-1, 0
#CHECK: error: invalid operand
#CHECK: pfdrl	16, 0
#CHECK: error: offset out of range
#CHECK: pfdrl	1, -0x1000000002
#CHECK: error: offset out of range
#CHECK: pfdrl	1, -1
#CHECK: error: offset out of range
#CHECK: pfdrl	1, 1
#CHECK: error: offset out of range
#CHECK: pfdrl	1, 0x100000000

	pfdrl	-1, 0
	pfdrl	16, 0
	pfdrl	1, -0x1000000002
	pfdrl	1, -1
	pfdrl	1, 1
	pfdrl	1, 0x100000000

#CHECK: error: missing length in address
#CHECK: pka	0, 0
#CHECK: error: missing length in address
#CHECK: pka	0(%r1), 0(%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: pka	0(1,%r1), 0(2,%r1)
#CHECK: error: invalid operand
#CHECK: pka	0(%r1), 0(0,%r1)
#CHECK: error: invalid operand
#CHECK: pka	0(%r1), 0(257,%r1)
#CHECK: error: invalid operand
#CHECK: pka	-1(%r1), 0(1,%r1)
#CHECK: error: invalid operand
#CHECK: pka	4096(%r1), 0(1,%r1)
#CHECK: error: invalid operand
#CHECK: pka	0(%r1), -1(1,%r1)
#CHECK: error: invalid operand
#CHECK: pka	0(%r1), 4096(1,%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: pka	0(%r1,%r2), 0(1,%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: pka	0(%r2), 0(%r1,%r2)
#CHECK: error: unknown token in expression
#CHECK: pka	0, 0(-)

	pka	0, 0
	pka	0(%r1), 0(%r1)
	pka	0(1,%r1), 0(2,%r1)
	pka	0(%r1), 0(0,%r1)
	pka	0(%r1), 0(257,%r1)
	pka	-1(%r1), 0(1,%r1)
	pka	4096(%r1), 0(1,%r1)
	pka	0(%r1), -1(1,%r1)
	pka	0(%r1), 4096(1,%r1)
	pka	0(%r1,%r2), 0(1,%r1)
	pka	0(%r2), 0(%r1,%r2)
	pka	0, 0(-)

#CHECK: error: missing length in address
#CHECK: pku	0, 0
#CHECK: error: missing length in address
#CHECK: pku	0(%r1), 0(%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: pku	0(1,%r1), 0(2,%r1)
#CHECK: error: invalid operand
#CHECK: pku	0(%r1), 0(0,%r1)
#CHECK: error: invalid operand
#CHECK: pku	0(%r1), 0(257,%r1)
#CHECK: error: invalid operand
#CHECK: pku	-1(%r1), 0(1,%r1)
#CHECK: error: invalid operand
#CHECK: pku	4096(%r1), 0(1,%r1)
#CHECK: error: invalid operand
#CHECK: pku	0(%r1), -1(1,%r1)
#CHECK: error: invalid operand
#CHECK: pku	0(%r1), 4096(1,%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: pku	0(%r1,%r2), 0(1,%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: pku	0(%r2), 0(%r1,%r2)
#CHECK: error: unknown token in expression
#CHECK: pku	0, 0(-)

	pku	0, 0
	pku	0(%r1), 0(%r1)
	pku	0(1,%r1), 0(2,%r1)
	pku	0(%r1), 0(0,%r1)
	pku	0(%r1), 0(257,%r1)
	pku	-1(%r1), 0(1,%r1)
	pku	4096(%r1), 0(1,%r1)
	pku	0(%r1), -1(1,%r1)
	pku	0(%r1), 4096(1,%r1)
	pku	0(%r0), 0(1,%r1)
	pku	0(%r1), 0(1,%r0)
	pku	0(%r1,%r2), 0(1,%r1)
	pku	0(%r2), 0(%r1,%r2)
	pku	0, 0(-)

#CHECK: error: invalid use of indexed addressing
#CHECK: plo	%r2, 160(%r1,%r15), %r4, 160(%r15)
#CHECK: error: invalid operand
#CHECK: plo	%r2, -1(%r1), %r4, 160(%r15)
#CHECK: error: invalid operand
#CHECK: plo	%r2, 4096(%r1), %r4, 160(%r15)
#CHECK: error: invalid operand
#CHECK: plo	%r2, 0(%r1), %r4, -1(%r15)
#CHECK: error: invalid operand
#CHECK: plo	%r2, 0(%r1), %r4, 4096(%r15)

        plo	%r2, 160(%r1,%r15), %r4, 160(%r15)
        plo	%r2, -1(%r1), %r4, 160(%r15)
        plo	%r2, 4096(%r1), %r4, 160(%r15)
        plo	%r2, 0(%r1), %r4, -1(%r15)
        plo	%r2, 0(%r1), %r4, 4096(%r15)

#CHECK: error: instruction requires: population-count
#CHECK: popcnt	%r0, %r0

	popcnt	%r0, %r0

#CHECK: error: invalid operand
#CHECK: pr    %r0
        pr    %r0

#CHECK: error: invalid operand
#CHECK: qadtr	%f0, %f0, %f0, -1
#CHECK: error: invalid operand
#CHECK: qadtr	%f0, %f0, %f0, 16

	qadtr	%f0, %f0, %f0, -1
	qadtr	%f0, %f0, %f0, 16

#CHECK: error: invalid operand
#CHECK: qaxtr	%f0, %f0, %f0, -1
#CHECK: error: invalid operand
#CHECK: qaxtr	%f0, %f0, %f0, 16
#CHECK: error: invalid register pair
#CHECK: qaxtr	%f0, %f0, %f2, 0
#CHECK: error: invalid register pair
#CHECK: qaxtr	%f0, %f2, %f0, 0
#CHECK: error: invalid register pair
#CHECK: qaxtr	%f2, %f0, %f0, 0

	qaxtr	%f0, %f0, %f0, -1
	qaxtr	%f0, %f0, %f0, 16
	qaxtr	%f0, %f0, %f2, 0
	qaxtr	%f0, %f2, %f0, 0
	qaxtr	%f2, %f0, %f0, 0

#CHECK: error: invalid operand
#CHECK: qctri	-1
#CHECK: error: invalid operand
#CHECK: qctri	4096
#CHECK: error: invalid use of indexed addressing
#CHECK: qctri	0(%r1,%r2)

	qctri	-1
	qctri	4096
	qctri	0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: qsi	-1
#CHECK: error: invalid operand
#CHECK: qsi	4096
#CHECK: error: invalid use of indexed addressing
#CHECK: qsi	0(%r1,%r2)

	qsi	-1
	qsi	4096
	qsi	0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: risbg	%r0,%r0,0,0,-1
#CHECK: error: invalid operand
#CHECK: risbg	%r0,%r0,0,0,64
#CHECK: error: invalid operand
#CHECK: risbg	%r0,%r0,0,-1,0
#CHECK: error: invalid operand
#CHECK: risbg	%r0,%r0,0,256,0
#CHECK: error: invalid operand
#CHECK: risbg	%r0,%r0,-1,0,0
#CHECK: error: invalid operand
#CHECK: risbg	%r0,%r0,256,0,0

	risbg	%r0,%r0,0,0,-1
	risbg	%r0,%r0,0,0,64
	risbg	%r0,%r0,0,-1,0
	risbg	%r0,%r0,0,256,0
	risbg	%r0,%r0,-1,0,0
	risbg	%r0,%r0,256,0,0

#CHECK: error: instruction requires: high-word
#CHECK: risbhg	%r1, %r2, 0, 0, 0

	risbhg	%r1, %r2, 0, 0, 0

#CHECK: error: instruction requires: high-word
#CHECK: risblg	%r1, %r2, 0, 0, 0

	risblg	%r1, %r2, 0, 0, 0

#CHECK: error: invalid operand
#CHECK: rll	%r0,%r0,-524289
#CHECK: error: invalid operand
#CHECK: rll	%r0,%r0,524288
#CHECK: error: invalid use of indexed addressing
#CHECK: rll	%r0,%r0,0(%r1,%r2)

	rll	%r0,%r0,-524289
	rll	%r0,%r0,524288
	rll	%r0,%r0,0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: rllg	%r0,%r0,-524289
#CHECK: error: invalid operand
#CHECK: rllg	%r0,%r0,524288
#CHECK: error: invalid use of indexed addressing
#CHECK: rllg	%r0,%r0,0(%r1,%r2)

	rllg	%r0,%r0,-524289
	rllg	%r0,%r0,524288
	rllg	%r0,%r0,0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: rnsbg	%r0,%r0,0,0,-1
#CHECK: error: invalid operand
#CHECK: rnsbg	%r0,%r0,0,0,64
#CHECK: error: invalid operand
#CHECK: rnsbg	%r0,%r0,0,-1,0
#CHECK: error: invalid operand
#CHECK: rnsbg	%r0,%r0,0,256,0
#CHECK: error: invalid operand
#CHECK: rnsbg	%r0,%r0,-1,0,0
#CHECK: error: invalid operand
#CHECK: rnsbg	%r0,%r0,256,0,0

	rnsbg	%r0,%r0,0,0,-1
	rnsbg	%r0,%r0,0,0,64
	rnsbg	%r0,%r0,0,-1,0
	rnsbg	%r0,%r0,0,256,0
	rnsbg	%r0,%r0,-1,0,0
	rnsbg	%r0,%r0,256,0,0

#CHECK: error: invalid operand
#CHECK: rosbg	%r0,%r0,0,0,-1
#CHECK: error: invalid operand
#CHECK: rosbg	%r0,%r0,0,0,64
#CHECK: error: invalid operand
#CHECK: rosbg	%r0,%r0,0,-1,0
#CHECK: error: invalid operand
#CHECK: rosbg	%r0,%r0,0,256,0
#CHECK: error: invalid operand
#CHECK: rosbg	%r0,%r0,-1,0,0
#CHECK: error: invalid operand
#CHECK: rosbg	%r0,%r0,256,0,0

	rosbg	%r0,%r0,0,0,-1
	rosbg	%r0,%r0,0,0,64
	rosbg	%r0,%r0,0,-1,0
	rosbg	%r0,%r0,0,256,0
	rosbg	%r0,%r0,-1,0,0
	rosbg	%r0,%r0,256,0,0

#CHECK: error: invalid operand
#CHECK: rp	-1
#CHECK: error: invalid operand
#CHECK: rp	4096
#CHECK: error: invalid use of indexed addressing
#CHECK: rp	0(%r1,%r2)

	rp	-1
	rp	4096
	rp	0(%r1,%r2)

#CHECK: error: instruction requires: reset-reference-bits-multiple
#CHECK: rrbm	%r0, %r0

	rrbm	%r0, %r0

#CHECK: error: invalid operand
#CHECK: rrdtr	%f0, %f0, %f0, -1
#CHECK: error: invalid operand
#CHECK: rrdtr	%f0, %f0, %f0, 16

	rrdtr	%f0, %f0, %f0, -1
	rrdtr	%f0, %f0, %f0, 16

#CHECK: error: invalid operand
#CHECK: rrxtr	%f0, %f0, %f0, -1
#CHECK: error: invalid operand
#CHECK: rrxtr	%f0, %f0, %f0, 16
#CHECK: error: invalid register pair
#CHECK: rrxtr	%f0, %f0, %f2, 0
#CHECK: error: invalid register pair
#CHECK: rrxtr	%f0, %f2, %f0, 0
#CHECK: error: invalid register pair
#CHECK: rrxtr	%f2, %f0, %f0, 0

	rrxtr	%f0, %f0, %f0, -1
	rrxtr	%f0, %f0, %f0, 16
	rrxtr	%f0, %f0, %f2, 0
	rrxtr	%f0, %f2, %f0, 0
	rrxtr	%f2, %f0, %f0, 0

#CHECK: error: invalid operand
#CHECK: rxsbg	%r0,%r0,0,0,-1
#CHECK: error: invalid operand
#CHECK: rxsbg	%r0,%r0,0,0,64
#CHECK: error: invalid operand
#CHECK: rxsbg	%r0,%r0,0,-1,0
#CHECK: error: invalid operand
#CHECK: rxsbg	%r0,%r0,0,256,0
#CHECK: error: invalid operand
#CHECK: rxsbg	%r0,%r0,-1,0,0
#CHECK: error: invalid operand
#CHECK: rxsbg	%r0,%r0,256,0,0

	rxsbg	%r0,%r0,0,0,-1
	rxsbg	%r0,%r0,0,0,64
	rxsbg	%r0,%r0,0,-1,0
	rxsbg	%r0,%r0,0,256,0
	rxsbg	%r0,%r0,-1,0,0
	rxsbg	%r0,%r0,256,0,0

#CHECK: error: invalid operand
#CHECK: s	%r0, -1
#CHECK: error: invalid operand
#CHECK: s	%r0, 4096

	s	%r0, -1
	s	%r0, 4096

#CHECK: error: invalid operand
#CHECK: sac	-1
#CHECK: error: invalid operand
#CHECK: sac	4096
#CHECK: error: invalid use of indexed addressing
#CHECK: sac	0(%r1,%r2)

	sac	-1
	sac	4096
	sac	0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: sacf	-1
#CHECK: error: invalid operand
#CHECK: sacf	4096
#CHECK: error: invalid use of indexed addressing
#CHECK: sacf	0(%r1,%r2)

	sacf	-1
	sacf	4096
	sacf	0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: sck	-1
#CHECK: error: invalid operand
#CHECK: sck	4096
#CHECK: error: invalid use of indexed addressing
#CHECK: sck	0(%r1,%r2)

	sck	-1
	sck	4096
	sck	0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: sckc	-1
#CHECK: error: invalid operand
#CHECK: sckc	4096
#CHECK: error: invalid use of indexed addressing
#CHECK: sckc	0(%r1,%r2)

	sckc	-1
	sckc	4096
	sckc	0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: sd	%f0, -1
#CHECK: error: invalid operand
#CHECK: sd	%f0, 4096

	sd	%f0, -1
	sd	%f0, 4096

#CHECK: error: invalid operand
#CHECK: sdb	%f0, -1
#CHECK: error: invalid operand
#CHECK: sdb	%f0, 4096

	sdb	%f0, -1
	sdb	%f0, 4096

#CHECK: error: instruction requires: fp-extension
#CHECK: sdtra	%f0, %f0, %f0, 0

	sdtra	%f0, %f0, %f0, 0

#CHECK: error: invalid operand
#CHECK: se	%f0, -1
#CHECK: error: invalid operand
#CHECK: se	%f0, 4096

	se	%f0, -1
	se	%f0, 4096

#CHECK: error: invalid operand
#CHECK: seb	%f0, -1
#CHECK: error: invalid operand
#CHECK: seb	%f0, 4096

	seb	%f0, -1
	seb	%f0, 4096

#CHECK: error: invalid operand
#CHECK: sg	%r0, -524289
#CHECK: error: invalid operand
#CHECK: sg	%r0, 524288

	sg	%r0, -524289
	sg	%r0, 524288

#CHECK: error: invalid operand
#CHECK: sgf	%r0, -524289
#CHECK: error: invalid operand
#CHECK: sgf	%r0, 524288

	sgf	%r0, -524289
	sgf	%r0, 524288

#CHECK: error: instruction requires: distinct-ops
#CHECK: sgrk	%r2,%r3,%r4

	sgrk	%r2,%r3,%r4

#CHECK: error: invalid operand
#CHECK: sh	%r0, -1
#CHECK: error: invalid operand
#CHECK: sh	%r0, 4096

	sh	%r0, -1
	sh	%r0, 4096

#CHECK: error: instruction requires: high-word
#CHECK: shhhr	%r0, %r0, %r0

	shhhr	%r0, %r0, %r0

#CHECK: error: instruction requires: high-word
#CHECK: shhlr	%r0, %r0, %r0

	shhlr	%r0, %r0, %r0

#CHECK: error: invalid operand
#CHECK: shy	%r0, -524289
#CHECK: error: invalid operand
#CHECK: shy	%r0, 524288

	shy	%r0, -524289
	shy	%r0, 524288

#CHECK: error: invalid operand
#CHECK: sie	-1
#CHECK: error: invalid operand
#CHECK: sie	4096
#CHECK: error: invalid use of indexed addressing
#CHECK: sie	0(%r1,%r2)

	sie	-1
	sie	4096
	sie	0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: siga	-1
#CHECK: error: invalid operand
#CHECK: siga	4096
#CHECK: error: invalid use of indexed addressing
#CHECK: siga	0(%r1,%r2)

	siga	-1
	siga	4096
	siga	0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: sigp	%r0, %r0, -1
#CHECK: error: invalid operand
#CHECK: sigp	%r0, %r0, 4096
#CHECK: error: invalid use of indexed addressing
#CHECK: sigp	%r0, %r0, 0(%r1,%r2)

	sigp	%r0, %r0, -1
	sigp	%r0, %r0, 4096
	sigp	%r0, %r0, 0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: sl	%r0, -1
#CHECK: error: invalid operand
#CHECK: sl	%r0, 4096

	sl	%r0, -1
	sl	%r0, 4096

#CHECK: error: invalid operand
#CHECK: sla	%r0,-1
#CHECK: error: invalid operand
#CHECK: sla	%r0,4096
#CHECK: error: invalid use of indexed addressing
#CHECK: sla	%r0,0(%r1,%r2)

	sla	%r0,-1
	sla	%r0,4096
	sla	%r0,0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: slag	%r0,%r0,-524289
#CHECK: error: invalid operand
#CHECK: slag	%r0,%r0,524288
#CHECK: error: invalid use of indexed addressing
#CHECK: slag	%r0,%r0,0(%r1,%r2)

	slag	%r0,%r0,-524289
	slag	%r0,%r0,524288
	slag	%r0,%r0,0(%r1,%r2)

#CHECK: error: instruction requires: distinct-ops
#CHECK: slak	%r2,%r3,4(%r5)

	slak	%r2,%r3,4(%r5)

#CHECK: error: invalid operand
#CHECK: slb	%r0, -524289
#CHECK: error: invalid operand
#CHECK: slb	%r0, 524288

	slb	%r0, -524289
	slb	%r0, 524288

#CHECK: error: invalid operand
#CHECK: slbg	%r0, -524289
#CHECK: error: invalid operand
#CHECK: slbg	%r0, 524288

	slbg	%r0, -524289
	slbg	%r0, 524288

#CHECK: error: invalid register pair
#CHECK: slda	%r1,0
#CHECK: error: invalid operand
#CHECK: slda	%r0,-1
#CHECK: error: invalid operand
#CHECK: slda	%r0,4096
#CHECK: error: invalid use of indexed addressing
#CHECK: slda	%r0,0(%r1,%r2)

	slda	%r1,0
	slda	%r0,-1
	slda	%r0,4096
	slda	%r0,0(%r1,%r2)

#CHECK: error: invalid register pair
#CHECK: sldl	%r1,0
#CHECK: error: invalid operand
#CHECK: sldl	%r0,-1
#CHECK: error: invalid operand
#CHECK: sldl	%r0,4096
#CHECK: error: invalid use of indexed addressing
#CHECK: sldl	%r0,0(%r1,%r2)

	sldl	%r1,0
	sldl	%r0,-1
	sldl	%r0,4096
	sldl	%r0,0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: sldt	%f0, %f0, -1
#CHECK: error: invalid operand
#CHECK: sldt	%f0, %f0, 4096

	sldt	%f0, %f0, -1
	sldt	%f0, %f0, 4096

#CHECK: error: invalid operand
#CHECK: slfi	%r0, -1
#CHECK: error: invalid operand
#CHECK: slfi	%r0, (1 << 32)

	slfi	%r0, -1
	slfi	%r0, (1 << 32)

#CHECK: error: invalid operand
#CHECK: slg	%r0, -524289
#CHECK: error: invalid operand
#CHECK: slg	%r0, 524288

	slg	%r0, -524289
	slg	%r0, 524288

#CHECK: error: invalid operand
#CHECK: slgf	%r0, -524289
#CHECK: error: invalid operand
#CHECK: slgf	%r0, 524288

	slgf	%r0, -524289
	slgf	%r0, 524288

#CHECK: error: invalid operand
#CHECK: slgfi	%r0, -1
#CHECK: error: invalid operand
#CHECK: slgfi	%r0, (1 << 32)

	slgfi	%r0, -1
	slgfi	%r0, (1 << 32)

#CHECK: error: instruction requires: distinct-ops
#CHECK: slgrk	%r2,%r3,%r4

	slgrk	%r2,%r3,%r4

#CHECK: error: instruction requires: high-word
#CHECK: slhhhr	%r0, %r0, %r0

	slhhhr	%r0, %r0, %r0

#CHECK: error: instruction requires: high-word
#CHECK: slhhlr	%r0, %r0, %r0

	slhhlr	%r0, %r0, %r0

#CHECK: error: invalid operand
#CHECK: sll	%r0,-1
#CHECK: error: invalid operand
#CHECK: sll	%r0,4096
#CHECK: error: invalid use of indexed addressing
#CHECK: sll	%r0,0(%r1,%r2)

	sll	%r0,-1
	sll	%r0,4096
	sll	%r0,0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: sllg	%r0,%r0,-524289
#CHECK: error: invalid operand
#CHECK: sllg	%r0,%r0,524288
#CHECK: error: invalid use of indexed addressing
#CHECK: sllg	%r0,%r0,0(%r1,%r2)

	sllg	%r0,%r0,-524289
	sllg	%r0,%r0,524288
	sllg	%r0,%r0,0(%r1,%r2)

#CHECK: error: instruction requires: distinct-ops
#CHECK: sllk	%r2,%r3,4(%r5)

	sllk	%r2,%r3,4(%r5)

#CHECK: error: instruction requires: distinct-ops
#CHECK: slrk	%r2,%r3,%r4

	slrk	%r2,%r3,%r4

#CHECK: error: invalid operand
#CHECK: slxt	%f0, %f0, -1
#CHECK: error: invalid operand
#CHECK: slxt	%f0, %f0, 4096
#CHECK: error: invalid register pair
#CHECK: slxt	%f0, %f2, 0
#CHECK: error: invalid register pair
#CHECK: slxt	%f2, %f0, 0

	slxt	%f0, %f0, -1
	slxt	%f0, %f0, 4096
	slxt	%f0, %f2, 0
	slxt	%f2, %f0, 0

#CHECK: error: invalid operand
#CHECK: sly	%r0, -524289
#CHECK: error: invalid operand
#CHECK: sly	%r0, 524288

	sly	%r0, -524289
	sly	%r0, 524288

#CHECK: error: missing length in address
#CHECK: sp	0, 0(1)
#CHECK: error: missing length in address
#CHECK: sp	0(1), 0
#CHECK: error: missing length in address
#CHECK: sp	0(%r1), 0(1,%r1)
#CHECK: error: missing length in address
#CHECK: sp	0(1,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: sp	0(0,%r1), 0(1,%r1)
#CHECK: error: invalid operand
#CHECK: sp	0(1,%r1), 0(0,%r1)
#CHECK: error: invalid operand
#CHECK: sp	0(17,%r1), 0(1,%r1)
#CHECK: error: invalid operand
#CHECK: sp	0(1,%r1), 0(17,%r1)
#CHECK: error: invalid operand
#CHECK: sp	-1(1,%r1), 0(1,%r1)
#CHECK: error: invalid operand
#CHECK: sp	4096(1,%r1), 0(1,%r1)
#CHECK: error: invalid operand
#CHECK: sp	0(1,%r1), -1(1,%r1)
#CHECK: error: invalid operand
#CHECK: sp	0(1,%r1), 4096(1,%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: sp	0(%r1,%r2), 0(1,%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: sp	0(1,%r2), 0(%r1,%r2)
#CHECK: error: unknown token in expression
#CHECK: sp	0(-), 0(1)

	sp	0, 0(1)
	sp	0(1), 0
	sp	0(%r1), 0(1,%r1)
	sp	0(1,%r1), 0(%r1)
	sp	0(0,%r1), 0(1,%r1)
	sp	0(1,%r1), 0(0,%r1)
	sp	0(17,%r1), 0(1,%r1)
	sp	0(1,%r1), 0(17,%r1)
	sp	-1(1,%r1), 0(1,%r1)
	sp	4096(1,%r1), 0(1,%r1)
	sp	0(1,%r1), -1(1,%r1)
	sp	0(1,%r1), 4096(1,%r1)
	sp	0(%r1,%r2), 0(1,%r1)
	sp	0(1,%r2), 0(%r1,%r2)
	sp	0(-), 0(1)

#CHECK: error: invalid operand
#CHECK: spka	-1
#CHECK: error: invalid operand
#CHECK: spka	4096
#CHECK: error: invalid use of indexed addressing
#CHECK: spka	0(%r1,%r2)

	spka	-1
	spka	4096
	spka	0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: spt	-1
#CHECK: error: invalid operand
#CHECK: spt	4096
#CHECK: error: invalid use of indexed addressing
#CHECK: spt	0(%r1,%r2)

	spt	-1
	spt	4096
	spt	0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: spx	-1
#CHECK: error: invalid operand
#CHECK: spx	4096
#CHECK: error: invalid use of indexed addressing
#CHECK: spx	0(%r1,%r2)

	spx	-1
	spx	4096
	spx	0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: sqd	%f0, -1
#CHECK: error: invalid operand
#CHECK: sqd	%f0, 4096

	sqd	%f0, -1
	sqd	%f0, 4096

#CHECK: error: invalid operand
#CHECK: sqdb	%f0, -1
#CHECK: error: invalid operand
#CHECK: sqdb	%f0, 4096

	sqdb	%f0, -1
	sqdb	%f0, 4096

#CHECK: error: invalid operand
#CHECK: sqe	%f0, -1
#CHECK: error: invalid operand
#CHECK: sqe	%f0, 4096

	sqe	%f0, -1
	sqe	%f0, 4096

#CHECK: error: invalid operand
#CHECK: sqeb	%f0, -1
#CHECK: error: invalid operand
#CHECK: sqeb	%f0, 4096

	sqeb	%f0, -1
	sqeb	%f0, 4096

#CHECK: error: invalid register pair
#CHECK: sqxbr	%f0, %f2
#CHECK: error: invalid register pair
#CHECK: sqxbr	%f2, %f0

	sqxbr	%f0, %f2
	sqxbr	%f2, %f0

#CHECK: error: invalid register pair
#CHECK: sqxr	%f0, %f2
#CHECK: error: invalid register pair
#CHECK: sqxr	%f2, %f0

	sqxr	%f0, %f2
	sqxr	%f2, %f0

#CHECK: error: invalid operand
#CHECK: sra	%r0,-1
#CHECK: error: invalid operand
#CHECK: sra	%r0,4096
#CHECK: error: invalid use of indexed addressing
#CHECK: sra	%r0,0(%r1,%r2)

	sra	%r0,-1
	sra	%r0,4096
	sra	%r0,0(%r0)
	sra	%r0,0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: srag	%r0,%r0,-524289
#CHECK: error: invalid operand
#CHECK: srag	%r0,%r0,524288
#CHECK: error: invalid use of indexed addressing
#CHECK: srag	%r0,%r0,0(%r1,%r2)

	srag	%r0,%r0,-524289
	srag	%r0,%r0,524288
	srag	%r0,%r0,0(%r0)
	srag	%r0,%r0,0(%r1,%r2)

#CHECK: error: instruction requires: distinct-ops
#CHECK: srak	%r2,%r3,4(%r5)

	srak	%r2,%r3,4(%r5)

#CHECK: error: invalid register pair
#CHECK: srda	%r1,0
#CHECK: error: invalid operand
#CHECK: srda	%r0,-1
#CHECK: error: invalid operand
#CHECK: srda	%r0,4096
#CHECK: error: invalid use of indexed addressing
#CHECK: srda	%r0,0(%r1,%r2)

	srda	%r1,0
	srda	%r0,-1
	srda	%r0,4096
	srda	%r0,0(%r1,%r2)

#CHECK: error: invalid register pair
#CHECK: srdl	%r1,0
#CHECK: error: invalid operand
#CHECK: srdl	%r0,-1
#CHECK: error: invalid operand
#CHECK: srdl	%r0,4096
#CHECK: error: invalid use of indexed addressing
#CHECK: srdl	%r0,0(%r1,%r2)

	srdl	%r1,0
	srdl	%r0,-1
	srdl	%r0,4096
	srdl	%r0,0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: srdt	%f0, %f0, -1
#CHECK: error: invalid operand
#CHECK: srdt	%f0, %f0, 4096

	srdt	%f0, %f0, -1
	srdt	%f0, %f0, 4096

#CHECK: error: instruction requires: distinct-ops
#CHECK: srk	%r2,%r3,%r4

	srk	%r2,%r3,%r4

#CHECK: error: invalid operand
#CHECK: srl	%r0,-1
#CHECK: error: invalid operand
#CHECK: srl	%r0,4096
#CHECK: error: invalid use of indexed addressing
#CHECK: srl	%r0,0(%r1,%r2)

	srl	%r0,-1
	srl	%r0,4096
	srl	%r0,0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: srlg	%r0,%r0,-524289
#CHECK: error: invalid operand
#CHECK: srlg	%r0,%r0,524288
#CHECK: error: invalid use of indexed addressing
#CHECK: srlg	%r0,%r0,0(%r1,%r2)

	srlg	%r0,%r0,-524289
	srlg	%r0,%r0,524288
	srlg	%r0,%r0,0(%r1,%r2)

#CHECK: error: instruction requires: distinct-ops
#CHECK: srlk	%r2,%r3,4(%r5)

	srlk	%r2,%r3,4(%r5)

#CHECK: error: invalid operand
#CHECK: srnm	-1
#CHECK: error: invalid operand
#CHECK: srnm	4096
#CHECK: error: invalid use of indexed addressing
#CHECK: srnm	0(%r1,%r2)

	srnm	-1
	srnm	4096
	srnm	0(%r1,%r2)

#CHECK: error: instruction requires: fp-extension
#CHECK: srnmb	0(%r1)

	srnmb	0(%r1)

#CHECK: error: invalid operand
#CHECK: srnmt	-1
#CHECK: error: invalid operand
#CHECK: srnmt	4096
#CHECK: error: invalid use of indexed addressing
#CHECK: srnmt	0(%r1,%r2)

	srnmt	-1
	srnmt	4096
	srnmt	0(%r1,%r2)

#CHECK: error: missing length in address
#CHECK: srp	0, 0, 0
#CHECK: error: missing length in address
#CHECK: srp	0(%r1), 0(%r1), 0
#CHECK: error: invalid use of indexed addressing
#CHECK: srp	0(1,%r1), 0(2,%r1), 0
#CHECK: error: invalid operand
#CHECK: srp	0(0,%r1), 0(%r1), 0
#CHECK: error: invalid operand
#CHECK: srp	0(17,%r1), 0(%r1), 0
#CHECK: error: invalid operand
#CHECK: srp	-1(1,%r1), 0(%r1), 0
#CHECK: error: invalid operand
#CHECK: srp	4096(1,%r1), 0(%r1), 0
#CHECK: error: invalid operand
#CHECK: srp	0(1,%r1), -1(%r1), 0
#CHECK: error: invalid operand
#CHECK: srp	0(1,%r1), 4096(%r1), 0
#CHECK: error: invalid use of indexed addressing
#CHECK: srp	0(%r1,%r2), 0(%r1), 0
#CHECK: error: invalid use of indexed addressing
#CHECK: srp	0(1,%r2), 0(%r1,%r2), 0
#CHECK: error: invalid operand
#CHECK: srp	0(1), 0, -1
#CHECK: error: invalid operand
#CHECK: srp	0(1), 0, 16
#CHECK: error: unknown token in expression
#CHECK: srp	0(-), 0, 0

	srp	0, 0, 0
	srp	0(%r1), 0(%r1), 0
	srp	0(1,%r1), 0(2,%r1), 0
	srp	0(0,%r1), 0(%r1), 0
	srp	0(17,%r1), 0(%r1), 0
	srp	-1(1,%r1), 0(%r1), 0
	srp	4096(1,%r1), 0(%r1), 0
	srp	0(1,%r1), -1(%r1), 0
	srp	0(1,%r1), 4096(%r1), 0
	srp	0(%r1,%r2), 0(%r1), 0
	srp	0(1,%r2), 0(%r1,%r2), 0
	srp	0(1), 0, -1
	srp	0(1), 0, 16
	srp	0(-), 0, 0

#CHECK: error: invalid operand
#CHECK: srxt	%f0, %f0, -1
#CHECK: error: invalid operand
#CHECK: srxt	%f0, %f0, 4096
#CHECK: error: invalid register pair
#CHECK: srxt	%f0, %f2, 0
#CHECK: error: invalid register pair
#CHECK: srxt	%f2, %f0, 0

	srxt	%f0, %f0, -1
	srxt	%f0, %f0, 4096
	srxt	%f0, %f2, 0
	srxt	%f2, %f0, 0

#CHECK: error: invalid operand
#CHECK: ssch	-1
#CHECK: error: invalid operand
#CHECK: ssch	4096
#CHECK: error: invalid use of indexed addressing
#CHECK: ssch	0(%r1,%r2)

	ssch	-1
	ssch	4096
	ssch	0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: sske	%r0, %r0, -1
#CHECK: error: invalid operand
#CHECK: sske	%r0, %r0, 16

	sske	%r0, %r0, -1
	sske	%r0, %r0, 16

#CHECK: error: invalid operand
#CHECK: ssm	-1
#CHECK: error: invalid operand
#CHECK: ssm	4096
#CHECK: error: invalid use of indexed addressing
#CHECK: ssm	0(%r1,%r2)

	ssm	-1
	ssm	4096
	ssm	0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: st	%r0, -1
#CHECK: error: invalid operand
#CHECK: st	%r0, 4096

	st	%r0, -1
	st	%r0, 4096

#CHECK: error: invalid operand
#CHECK: stam	%a0, %a0, 4096
#CHECK: error: invalid use of indexed addressing
#CHECK: stam	%a0, %a0, 0(%r1,%r2)

	stam	%a0, %a0, 4096
	stam	%a0, %a0, 0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: stamy	%a0, %a0, -524289
#CHECK: error: invalid operand
#CHECK: stamy	%a0, %a0, 524288
#CHECK: error: invalid use of indexed addressing
#CHECK: stamy	%a0, %a0, 0(%r1,%r2)

	stamy	%a0, %a0, -524289
	stamy	%a0, %a0, 524288
	stamy	%a0, %a0, 0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: stap	-1
#CHECK: error: invalid operand
#CHECK: stap	4096
#CHECK: error: invalid use of indexed addressing
#CHECK: stap	0(%r1,%r2)

	stap	-1
	stap	4096
	stap	0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: stc	%r0, -1
#CHECK: error: invalid operand
#CHECK: stc	%r0, 4096

	stc	%r0, -1
	stc	%r0, 4096

#CHECK: error: instruction requires: high-word
#CHECK: stch	%r0, 0

	stch	%r0, 0

#CHECK: error: invalid operand
#CHECK: stck	-1
#CHECK: error: invalid operand
#CHECK: stck	4096
#CHECK: error: invalid use of indexed addressing
#CHECK: stck	0(%r1,%r2)

	stck	-1
	stck	4096
	stck	0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: stckc	-1
#CHECK: error: invalid operand
#CHECK: stckc	4096
#CHECK: error: invalid use of indexed addressing
#CHECK: stckc	0(%r1,%r2)

	stckc	-1
	stckc	4096
	stckc	0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: stcke	-1
#CHECK: error: invalid operand
#CHECK: stcke	4096
#CHECK: error: invalid use of indexed addressing
#CHECK: stcke	0(%r1,%r2)

	stcke	-1
	stcke	4096
	stcke	0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: stckf	-1
#CHECK: error: invalid operand
#CHECK: stckf	4096
#CHECK: error: invalid use of indexed addressing
#CHECK: stckf	0(%r1,%r2)

	stckf	-1
	stckf	4096
	stckf	0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: stcm	%r0, 0, -1
#CHECK: error: invalid operand
#CHECK: stcm	%r0, 0, 4096
#CHECK: error: invalid operand
#CHECK: stcm	%r0, -1, 0
#CHECK: error: invalid operand
#CHECK: stcm	%r0, 16, 0

	stcm	%r0, 0, -1
	stcm	%r0, 0, 4096
	stcm	%r0, -1, 0
	stcm	%r0, 16, 0

#CHECK: error: invalid operand
#CHECK: stcmy	%r0, 0, -524289
#CHECK: error: invalid operand
#CHECK: stcmy	%r0, 0, 524288
#CHECK: error: invalid operand
#CHECK: stcmy	%r0, -1, 0
#CHECK: error: invalid operand
#CHECK: stcmy	%r0, 16, 0

	stcmy	%r0, 0, -524289
	stcmy	%r0, 0, 524288
	stcmy	%r0, -1, 0
	stcmy	%r0, 16, 0

#CHECK: error: invalid operand
#CHECK: stcmy	%r0, 0, -524289
#CHECK: error: invalid operand
#CHECK: stcmy	%r0, 0, 524288
#CHECK: error: invalid operand
#CHECK: stcmy	%r0, -1, 0
#CHECK: error: invalid operand
#CHECK: stcmy	%r0, 16, 0

	stcmy	%r0, 0, -524289
	stcmy	%r0, 0, 524288
	stcmy	%r0, -1, 0
	stcmy	%r0, 16, 0

#CHECK: error: invalid operand
#CHECK: stcps	-1
#CHECK: error: invalid operand
#CHECK: stcps	4096
#CHECK: error: invalid use of indexed addressing
#CHECK: stcps	0(%r1,%r2)

	stcps	-1
	stcps	4096
	stcps	0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: stcrw	-1
#CHECK: error: invalid operand
#CHECK: stcrw	4096
#CHECK: error: invalid use of indexed addressing
#CHECK: stcrw	0(%r1,%r2)

	stcrw	-1
	stcrw	4096
	stcrw	0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: stctg	%c0, %c0, -524289
#CHECK: error: invalid operand
#CHECK: stctg	%c0, %c0, 524288
#CHECK: error: invalid use of indexed addressing
#CHECK: stctg	%c0, %c0, 0(%r1,%r2)

	stctg	%c0, %c0, -524289
	stctg	%c0, %c0, 524288
	stctg	%c0, %c0, 0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: stctl	%c0, %c0, -1
#CHECK: error: invalid operand
#CHECK: stctl	%c0, %c0, 4096
#CHECK: error: invalid use of indexed addressing
#CHECK: stctl	%c0, %c0, 0(%r1,%r2)

	stctl	%c0, %c0, -1
	stctl	%c0, %c0, 4096
	stctl	%c0, %c0, 0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: stcy	%r0, -524289
#CHECK: error: invalid operand
#CHECK: stcy	%r0, 524288

	stcy	%r0, -524289
	stcy	%r0, 524288

#CHECK: error: invalid operand
#CHECK: std	%f0, -1
#CHECK: error: invalid operand
#CHECK: std	%f0, 4096

	std	%f0, -1
	std	%f0, 4096

#CHECK: error: invalid operand
#CHECK: stdy	%f0, -524289
#CHECK: error: invalid operand
#CHECK: stdy	%f0, 524288

	stdy	%f0, -524289
	stdy	%f0, 524288

#CHECK: error: invalid operand
#CHECK: ste	%f0, -1
#CHECK: error: invalid operand
#CHECK: ste	%f0, 4096

	ste	%f0, -1
	ste	%f0, 4096

#CHECK: error: invalid operand
#CHECK: stey	%f0, -524289
#CHECK: error: invalid operand
#CHECK: stey	%f0, 524288

	stey	%f0, -524289
	stey	%f0, 524288

#CHECK: error: instruction requires: high-word
#CHECK: stfh	%r0, 0

	stfh	%r0, 0

#CHECK: error: invalid operand
#CHECK: stfl	-1
#CHECK: error: invalid operand
#CHECK: stfl	4096
#CHECK: error: invalid use of indexed addressing
#CHECK: stfl	0(%r1,%r2)

	stfl	-1
	stfl	4096
	stfl	0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: stfle	-1
#CHECK: error: invalid operand
#CHECK: stfle	4096
#CHECK: error: invalid use of indexed addressing
#CHECK: stfle	0(%r1,%r2)

	stfle	-1
	stfle	4096
	stfle	0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: stfpc	-1
#CHECK: error: invalid operand
#CHECK: stfpc	4096
#CHECK: error: invalid use of indexed addressing
#CHECK: stfpc	0(%r1,%r2)

	stfpc	-1
	stfpc	4096
	stfpc	0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: stg	%r0, -524289
#CHECK: error: invalid operand
#CHECK: stg	%r0, 524288

	stg	%r0, -524289
	stg	%r0, 524288

#CHECK: error: offset out of range
#CHECK: stgrl	%r0, -0x1000000002
#CHECK: error: offset out of range
#CHECK: stgrl	%r0, -1
#CHECK: error: offset out of range
#CHECK: stgrl	%r0, 1
#CHECK: error: offset out of range
#CHECK: stgrl	%r0, 0x100000000

	stgrl	%r0, -0x1000000002
	stgrl	%r0, -1
	stgrl	%r0, 1
	stgrl	%r0, 0x100000000

#CHECK: error: invalid operand
#CHECK: sth	%r0, -1
#CHECK: error: invalid operand
#CHECK: sth	%r0, 4096

	sth	%r0, -1
	sth	%r0, 4096

#CHECK: error: instruction requires: high-word
#CHECK: sthh	%r0, 0

	sthh	%r0, 0

#CHECK: error: offset out of range
#CHECK: sthrl	%r0, -0x1000000002
#CHECK: error: offset out of range
#CHECK: sthrl	%r0, -1
#CHECK: error: offset out of range
#CHECK: sthrl	%r0, 1
#CHECK: error: offset out of range
#CHECK: sthrl	%r0, 0x100000000

	sthrl	%r0, -0x1000000002
	sthrl	%r0, -1
	sthrl	%r0, 1
	sthrl	%r0, 0x100000000

#CHECK: error: invalid operand
#CHECK: sthy	%r0, -524289
#CHECK: error: invalid operand
#CHECK: sthy	%r0, 524288

	sthy	%r0, -524289
	sthy	%r0, 524288

#CHECK: error: invalid operand
#CHECK: stidp	-1
#CHECK: error: invalid operand
#CHECK: stidp	4096
#CHECK: error: invalid use of indexed addressing
#CHECK: stidp	0(%r1,%r2)

	stidp	-1
	stidp	4096
	stidp	0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: stm	%r0, %r0, 4096
#CHECK: error: invalid use of indexed addressing
#CHECK: stm	%r0, %r0, 0(%r1,%r2)

	stm	%r0, %r0, 4096
	stm	%r0, %r0, 0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: stmg	%r0, %r0, -524289
#CHECK: error: invalid operand
#CHECK: stmg	%r0, %r0, 524288
#CHECK: error: invalid use of indexed addressing
#CHECK: stmg	%r0, %r0, 0(%r1,%r2)

	stmg	%r0, %r0, -524289
	stmg	%r0, %r0, 524288
	stmg	%r0, %r0, 0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: stmh	%r0, %r0, -524289
#CHECK: error: invalid operand
#CHECK: stmh	%r0, %r0, 524288
#CHECK: error: invalid use of indexed addressing
#CHECK: stmh	%r0, %r0, 0(%r1,%r2)

	stmh	%r0, %r0, -524289
	stmh	%r0, %r0, 524288
	stmh	%r0, %r0, 0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: stmy	%r0, %r0, -524289
#CHECK: error: invalid operand
#CHECK: stmy	%r0, %r0, 524288
#CHECK: error: invalid use of indexed addressing
#CHECK: stmy	%r0, %r0, 0(%r1,%r2)

	stmy	%r0, %r0, -524289
	stmy	%r0, %r0, 524288
	stmy	%r0, %r0, 0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: stnsm	-1, 0
#CHECK: error: invalid operand
#CHECK: stnsm	4096, 0
#CHECK: error: invalid use of indexed addressing
#CHECK: stnsm	0(%r1,%r2), 0
#CHECK: error: invalid operand
#CHECK: stnsm	0, -1
#CHECK: error: invalid operand
#CHECK: stnsm	0, 256

	stnsm	-1, 0
	stnsm	4096, 0
	stnsm	0(%r1,%r2), 0
	stnsm	0, -1
	stnsm	0, 256

#CHECK: error: invalid operand
#CHECK: stosm	-1, 0
#CHECK: error: invalid operand
#CHECK: stosm	4096, 0
#CHECK: error: invalid use of indexed addressing
#CHECK: stosm	0(%r1,%r2), 0
#CHECK: error: invalid operand
#CHECK: stosm	0, -1
#CHECK: error: invalid operand
#CHECK: stosm	0, 256

	stosm	-1, 0
	stosm	4096, 0
	stosm	0(%r1,%r2), 0
	stosm	0, -1
	stosm	0, 256

#CHECK: error: invalid operand
#CHECK: stpt	-1
#CHECK: error: invalid operand
#CHECK: stpt	4096
#CHECK: error: invalid use of indexed addressing
#CHECK: stpt	0(%r1,%r2)

	stpt	-1
	stpt	4096
	stpt	0(%r1,%r2)

#CHECK: error: invalid register pair
#CHECK: stpq	%r1, 0
#CHECK: error: invalid operand
#CHECK: stpq	%r0, -524289
#CHECK: error: invalid operand
#CHECK: stpq	%r0, 524288

	stpq	%r1, 0
	stpq	%r0, -524289
	stpq	%r0, 524288

#CHECK: error: invalid operand
#CHECK: stpx	-1
#CHECK: error: invalid operand
#CHECK: stpx	4096
#CHECK: error: invalid use of indexed addressing
#CHECK: stpx	0(%r1,%r2)

	stpx	-1
	stpx	4096
	stpx	0(%r1,%r2)

#CHECK: error: invalid use of indexed addressing
#CHECK: strag   160(%r1,%r15),160(%r15)
#CHECK: error: invalid operand
#CHECK: strag   -1(%r1),160(%r15)
#CHECK: error: invalid operand
#CHECK: strag   4096(%r1),160(%r15)
#CHECK: error: invalid operand
#CHECK: strag   0(%r1),-1(%r15)
#CHECK: error: invalid operand
#CHECK: strag   0(%r1),4096(%r15)

        strag   160(%r1,%r15),160(%r15)
        strag   -1(%r1),160(%r15)
        strag   4096(%r1),160(%r15)
        strag   0(%r1),-1(%r15)
        strag   0(%r1),4096(%r15)

#CHECK: error: offset out of range
#CHECK: strl	%r0, -0x1000000002
#CHECK: error: offset out of range
#CHECK: strl	%r0, -1
#CHECK: error: offset out of range
#CHECK: strl	%r0, 1
#CHECK: error: offset out of range
#CHECK: strl	%r0, 0x100000000

	strl	%r0, -0x1000000002
	strl	%r0, -1
	strl	%r0, 1
	strl	%r0, 0x100000000

#CHECK: error: invalid operand
#CHECK: strv	%r0, -524289
#CHECK: error: invalid operand
#CHECK: strv	%r0, 524288

	strv	%r0, -524289
	strv	%r0, 524288

#CHECK: error: invalid operand
#CHECK: strvg	%r0, -524289
#CHECK: error: invalid operand
#CHECK: strvg	%r0, 524288

	strvg	%r0, -524289
	strvg	%r0, 524288

#CHECK: error: invalid operand
#CHECK: stsch	-1
#CHECK: error: invalid operand
#CHECK: stsch	4096
#CHECK: error: invalid use of indexed addressing
#CHECK: stsch	0(%r1,%r2)

	stsch	-1
	stsch	4096
	stsch	0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: stsi	-1
#CHECK: error: invalid operand
#CHECK: stsi	4096
#CHECK: error: invalid use of indexed addressing
#CHECK: stsi	0(%r1,%r2)

	stsi	-1
	stsi	4096
	stsi	0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: sty	%r0, -524289
#CHECK: error: invalid operand
#CHECK: sty	%r0, 524288

	sty	%r0, -524289
	sty	%r0, 524288

#CHECK: error: invalid operand
#CHECK: su	%f0, -1
#CHECK: error: invalid operand
#CHECK: su	%f0, 4096

	su	%f0, -1
	su	%f0, 4096

#CHECK: error: invalid operand
#CHECK: sw	%f0, -1
#CHECK: error: invalid operand
#CHECK: sw	%f0, 4096

	sw	%f0, -1
	sw	%f0, 4096

#CHECK: error: invalid register pair
#CHECK: sxbr	%f0, %f2
#CHECK: error: invalid register pair
#CHECK: sxbr	%f2, %f0

	sxbr	%f0, %f2
	sxbr	%f2, %f0

#CHECK: error: invalid register pair
#CHECK: sxr	%f0, %f2
#CHECK: error: invalid register pair
#CHECK: sxr	%f2, %f0

	sxr	%f0, %f2
	sxr	%f2, %f0

#CHECK: error: invalid register pair
#CHECK: sxtr	%f0, %f0, %f2
#CHECK: error: invalid register pair
#CHECK: sxtr	%f0, %f2, %f0
#CHECK: error: invalid register pair
#CHECK: sxtr	%f2, %f0, %f0

	sxtr	%f0, %f0, %f2
	sxtr	%f0, %f2, %f0
	sxtr	%f2, %f0, %f0

#CHECK: error: instruction requires: fp-extension
#CHECK: sxtra	%f0, %f0, %f0, 0

	sxtra	%f0, %f0, %f0, 0

#CHECK: error: invalid operand
#CHECK: sy	%r0, -524289
#CHECK: error: invalid operand
#CHECK: sy	%r0, 524288

	sy	%r0, -524289
	sy	%r0, 524288

#CHECK: error: invalid operand
#CHECK: tbdr	%f0, -1, %f0
#CHECK: error: invalid operand
#CHECK: tbdr	%f0, 16, %f0

	tbdr	%f0, -1, %f0
	tbdr	%f0, 16, %f0

#CHECK: error: invalid operand
#CHECK: tbedr	%f0, -1, %f0
#CHECK: error: invalid operand
#CHECK: tbedr	%f0, 16, %f0

	tbedr	%f0, -1, %f0
	tbedr	%f0, 16, %f0

#CHECK: error: invalid operand
#CHECK: tcdb	%f0, -1
#CHECK: error: invalid operand
#CHECK: tcdb	%f0, 4096

	tcdb	%f0, -1
	tcdb	%f0, 4096

#CHECK: error: invalid operand
#CHECK: tceb	%f0, -1
#CHECK: error: invalid operand
#CHECK: tceb	%f0, 4096

	tceb	%f0, -1
	tceb	%f0, 4096

#CHECK: error: invalid operand
#CHECK: tcxb	%f0, -1
#CHECK: error: invalid operand
#CHECK: tcxb	%f0, 4096

	tcxb	%f0, -1
	tcxb	%f0, 4096

#CHECK: error: invalid operand
#CHECK: tdcdt	%f0, -1
#CHECK: error: invalid operand
#CHECK: tdcdt	%f0, 4096

	tdcdt	%f0, -1
	tdcdt	%f0, 4096

#CHECK: error: invalid operand
#CHECK: tdcet	%f0, -1
#CHECK: error: invalid operand
#CHECK: tdcet	%f0, 4096

	tdcet	%f0, -1
	tdcet	%f0, 4096

#CHECK: error: invalid operand
#CHECK: tdcxt	%f0, -1
#CHECK: error: invalid operand
#CHECK: tdcxt	%f0, 4096
#CHECK: error: invalid register pair
#CHECK: tdcxt	%f2, 0

	tdcxt	%f0, -1
	tdcxt	%f0, 4096
	tdcxt	%f2, 0

#CHECK: error: invalid operand
#CHECK: tdgdt	%f0, -1
#CHECK: error: invalid operand
#CHECK: tdgdt	%f0, 4096

	tdgdt	%f0, -1
	tdgdt	%f0, 4096

#CHECK: error: invalid operand
#CHECK: tdget	%f0, -1
#CHECK: error: invalid operand
#CHECK: tdget	%f0, 4096

	tdget	%f0, -1
	tdget	%f0, 4096

#CHECK: error: invalid operand
#CHECK: tdgxt	%f0, -1
#CHECK: error: invalid operand
#CHECK: tdgxt	%f0, 4096
#CHECK: error: invalid register pair
#CHECK: tdgxt	%f2, 0

	tdgxt	%f0, -1
	tdgxt	%f0, 4096
	tdgxt	%f2, 0

#CHECK: error: invalid operand
#CHECK: tm	-1, 0
#CHECK: error: invalid operand
#CHECK: tm	4096, 0
#CHECK: error: invalid use of indexed addressing
#CHECK: tm	0(%r1,%r2), 0
#CHECK: error: invalid operand
#CHECK: tm	0, -1
#CHECK: error: invalid operand
#CHECK: tm	0, 256

	tm	-1, 0
	tm	4096, 0
	tm	0(%r1,%r2), 0
	tm	0, -1
	tm	0, 256

#CHECK: error: invalid operand
#CHECK: tmh	%r0, -1
#CHECK: error: invalid operand
#CHECK: tmh	%r0, 0x10000

	tmh	%r0, -1
	tmh	%r0, 0x10000

#CHECK: error: invalid operand
#CHECK: tmhh	%r0, -1
#CHECK: error: invalid operand
#CHECK: tmhh	%r0, 0x10000

	tmhh	%r0, -1
	tmhh	%r0, 0x10000

#CHECK: error: invalid operand
#CHECK: tmhl	%r0, -1
#CHECK: error: invalid operand
#CHECK: tmhl	%r0, 0x10000

	tmhl	%r0, -1
	tmhl	%r0, 0x10000

#CHECK: error: invalid operand
#CHECK: tml	%r0, -1
#CHECK: error: invalid operand
#CHECK: tml	%r0, 0x10000

	tml	%r0, -1
	tml	%r0, 0x10000

#CHECK: error: invalid operand
#CHECK: tmlh	%r0, -1
#CHECK: error: invalid operand
#CHECK: tmlh	%r0, 0x10000

	tmlh	%r0, -1
	tmlh	%r0, 0x10000

#CHECK: error: invalid operand
#CHECK: tmll	%r0, -1
#CHECK: error: invalid operand
#CHECK: tmll	%r0, 0x10000

	tmll	%r0, -1
	tmll	%r0, 0x10000

#CHECK: error: invalid operand
#CHECK: tmy	-524289, 0
#CHECK: error: invalid operand
#CHECK: tmy	524288, 0
#CHECK: error: invalid use of indexed addressing
#CHECK: tmy	0(%r1,%r2), 0
#CHECK: error: invalid operand
#CHECK: tmy	0, -1
#CHECK: error: invalid operand
#CHECK: tmy	0, 256

	tmy	-524289, 0
	tmy	524288, 0
	tmy	0(%r1,%r2), 0
	tmy	0, -1
	tmy	0, 256

#CHECK: error: missing length in address
#CHECK: tp	0
#CHECK: error: missing length in address
#CHECK: tp	0(%r1)
#CHECK: error: invalid operand
#CHECK: tp	0(0,%r1)
#CHECK: error: invalid operand
#CHECK: tp	0(17,%r1)
#CHECK: error: invalid operand
#CHECK: tp	-1(1,%r1)
#CHECK: error: invalid operand
#CHECK: tp	4096(1,%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: tp	0(%r1,%r2)
#CHECK: error: unknown token in expression
#CHECK: tp	0(-)

	tp	0
	tp	0(%r1)
	tp	0(0,%r1)
	tp	0(17,%r1)
	tp	-1(1,%r1)
	tp	4096(1,%r1)
	tp	0(%r1,%r2)
	tp	0(-)

#CHECK: error: invalid operand
#CHECK: tpi	-1
#CHECK: error: invalid operand
#CHECK: tpi	4096
#CHECK: error: invalid use of indexed addressing
#CHECK: tpi	0(%r1,%r2)

	tpi	-1
	tpi	4096
	tpi	0(%r1,%r2)

#CHECK: error: invalid use of indexed addressing
#CHECK: tprot   160(%r1,%r15),160(%r15)
#CHECK: error: invalid operand
#CHECK: tprot   -1(%r1),160(%r15)
#CHECK: error: invalid operand
#CHECK: tprot   4096(%r1),160(%r15)
#CHECK: error: invalid operand
#CHECK: tprot   0(%r1),-1(%r15)
#CHECK: error: invalid operand
#CHECK: tprot   0(%r1),4096(%r15)

        tprot   160(%r1,%r15),160(%r15)
        tprot   -1(%r1),160(%r15)
        tprot   4096(%r1),160(%r15)
        tprot   0(%r1),-1(%r15)
        tprot   0(%r1),4096(%r15)

#CHECK: error: missing length in address
#CHECK: tr	0, 0
#CHECK: error: missing length in address
#CHECK: tr	0(%r1), 0(%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: tr	0(1,%r1), 0(2,%r1)
#CHECK: error: invalid operand
#CHECK: tr	0(0,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: tr	0(257,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: tr	-1(1,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: tr	4096(1,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: tr	0(1,%r1), -1(%r1)
#CHECK: error: invalid operand
#CHECK: tr	0(1,%r1), 4096(%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: tr	0(%r1,%r2), 0(%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: tr	0(1,%r2), 0(%r1,%r2)
#CHECK: error: unknown token in expression
#CHECK: tr	0(-), 0

	tr	0, 0
	tr	0(%r1), 0(%r1)
	tr	0(1,%r1), 0(2,%r1)
	tr	0(0,%r1), 0(%r1)
	tr	0(257,%r1), 0(%r1)
	tr	-1(1,%r1), 0(%r1)
	tr	4096(1,%r1), 0(%r1)
	tr	0(1,%r1), -1(%r1)
	tr	0(1,%r1), 4096(%r1)
	tr	0(%r1,%r2), 0(%r1)
	tr	0(1,%r2), 0(%r1,%r2)
	tr	0(-), 0

#CHECK: error: invalid operand
#CHECK: trace	%r0, %r0, -1
#CHECK: error: invalid operand
#CHECK: trace	%r0, %r0, 4096
#CHECK: error: invalid use of indexed addressing
#CHECK: trace	%r0, %r0, 0(%r1,%r2)

	trace	%r0, %r0, -1
	trace	%r0, %r0, 4096
	trace	%r0, %r0, 0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: tracg	%r0, %r0, -524289
#CHECK: error: invalid operand
#CHECK: tracg	%r0, %r0, 524288
#CHECK: error: invalid use of indexed addressing
#CHECK: tracg	%r0, %r0, 0(%r1,%r2)

	tracg	%r0, %r0, -524289
	tracg	%r0, %r0, 524288
	tracg	%r0, %r0, 0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: trap4	-1
#CHECK: error: invalid operand
#CHECK: trap4	4096
#CHECK: error: invalid use of indexed addressing
#CHECK: trap4	0(%r1,%r2)

	trap4	-1
	trap4	4096
	trap4	0(%r1,%r2)

#CHECK: error: invalid register pair
#CHECK: tre	%r1, %r0

	tre	%r1, %r0

#CHECK: error: invalid register pair
#CHECK: troo	%r1, %r0
#CHECK: error: invalid operand
#CHECK: troo	%r2, %r4, -1
#CHECK: error: invalid operand
#CHECK: troo	%r2, %r4, 16

	troo	%r1, %r0
	troo	%r2, %r4, -1
	troo	%r2, %r4, 16

#CHECK: error: invalid register pair
#CHECK: trot	%r1, %r0
#CHECK: error: invalid operand
#CHECK: trot	%r2, %r4, -1
#CHECK: error: invalid operand
#CHECK: trot	%r2, %r4, 16

	trot	%r1, %r0
	trot	%r2, %r4, -1
	trot	%r2, %r4, 16

#CHECK: error: missing length in address
#CHECK: trt	0, 0
#CHECK: error: missing length in address
#CHECK: trt	0(%r1), 0(%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: trt	0(1,%r1), 0(2,%r1)
#CHECK: error: invalid operand
#CHECK: trt	0(0,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: trt	0(257,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: trt	-1(1,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: trt	4096(1,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: trt	0(1,%r1), -1(%r1)
#CHECK: error: invalid operand
#CHECK: trt	0(1,%r1), 4096(%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: trt	0(%r1,%r2), 0(%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: trt	0(1,%r2), 0(%r1,%r2)
#CHECK: error: unknown token in expression
#CHECK: trt	0(-), 0

	trt	0, 0
	trt	0(%r1), 0(%r1)
	trt	0(1,%r1), 0(2,%r1)
	trt	0(0,%r1), 0(%r1)
	trt	0(257,%r1), 0(%r1)
	trt	-1(1,%r1), 0(%r1)
	trt	4096(1,%r1), 0(%r1)
	trt	0(1,%r1), -1(%r1)
	trt	0(1,%r1), 4096(%r1)
	trt	0(%r1,%r2), 0(%r1)
	trt	0(1,%r2), 0(%r1,%r2)
	trt	0(-), 0

#CHECK: error: invalid register pair
#CHECK: trte	%r1, %r0
#CHECK: error: invalid operand
#CHECK: trte	%r2, %r4, -1
#CHECK: error: invalid operand
#CHECK: trte	%r2, %r4, 16

	trte	%r1, %r0
	trte	%r2, %r4, -1
	trte	%r2, %r4, 16

#CHECK: error: invalid register pair
#CHECK: trto	%r1, %r0
#CHECK: error: invalid operand
#CHECK: trto	%r2, %r4, -1
#CHECK: error: invalid operand
#CHECK: trto	%r2, %r4, 16

	trto	%r1, %r0
	trto	%r2, %r4, -1
	trto	%r2, %r4, 16

#CHECK: error: missing length in address
#CHECK: trtr	0, 0
#CHECK: error: missing length in address
#CHECK: trtr	0(%r1), 0(%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: trtr	0(1,%r1), 0(2,%r1)
#CHECK: error: invalid operand
#CHECK: trtr	0(0,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: trtr	0(257,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: trtr	-1(1,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: trtr	4096(1,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: trtr	0(1,%r1), -1(%r1)
#CHECK: error: invalid operand
#CHECK: trtr	0(1,%r1), 4096(%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: trtr	0(%r1,%r2), 0(%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: trtr	0(1,%r2), 0(%r1,%r2)
#CHECK: error: unknown token in expression
#CHECK: trtr	0(-), 0

	trtr	0, 0
	trtr	0(%r1), 0(%r1)
	trtr	0(1,%r1), 0(2,%r1)
	trtr	0(0,%r1), 0(%r1)
	trtr	0(257,%r1), 0(%r1)
	trtr	-1(1,%r1), 0(%r1)
	trtr	4096(1,%r1), 0(%r1)
	trtr	0(1,%r1), -1(%r1)
	trtr	0(1,%r1), 4096(%r1)
	trtr	0(%r1,%r2), 0(%r1)
	trtr	0(1,%r2), 0(%r1,%r2)
	trtr	0(-), 0

#CHECK: error: invalid register pair
#CHECK: trtre	%r1, %r0
#CHECK: error: invalid operand
#CHECK: trtre	%r2, %r4, -1
#CHECK: error: invalid operand
#CHECK: trtre	%r2, %r4, 16

	trtre	%r1, %r0
	trtre	%r2, %r4, -1
	trtre	%r2, %r4, 16

#CHECK: error: invalid register pair
#CHECK: trtt	%r1, %r0
#CHECK: error: invalid operand
#CHECK: trtt	%r2, %r4, -1
#CHECK: error: invalid operand
#CHECK: trtt	%r2, %r4, 16

	trtt	%r1, %r0
	trtt	%r2, %r4, -1
	trtt	%r2, %r4, 16

#CHECK: error: invalid operand
#CHECK: ts	-1
#CHECK: error: invalid operand
#CHECK: ts	4096
#CHECK: error: invalid use of indexed addressing
#CHECK: ts	0(%r1,%r2)

	ts	-1
	ts	4096
	ts	0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: tsch	-1
#CHECK: error: invalid operand
#CHECK: tsch	4096
#CHECK: error: invalid use of indexed addressing
#CHECK: tsch	0(%r1,%r2)

	tsch	-1
	tsch	4096
	tsch	0(%r1,%r2)

#CHECK: error: missing length in address
#CHECK: unpk	0, 0(1)
#CHECK: error: missing length in address
#CHECK: unpk	0(1), 0
#CHECK: error: missing length in address
#CHECK: unpk	0(%r1), 0(1,%r1)
#CHECK: error: missing length in address
#CHECK: unpk	0(1,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: unpk	0(0,%r1), 0(1,%r1)
#CHECK: error: invalid operand
#CHECK: unpk	0(1,%r1), 0(0,%r1)
#CHECK: error: invalid operand
#CHECK: unpk	0(17,%r1), 0(1,%r1)
#CHECK: error: invalid operand
#CHECK: unpk	0(1,%r1), 0(17,%r1)
#CHECK: error: invalid operand
#CHECK: unpk	-1(1,%r1), 0(1,%r1)
#CHECK: error: invalid operand
#CHECK: unpk	4096(1,%r1), 0(1,%r1)
#CHECK: error: invalid operand
#CHECK: unpk	0(1,%r1), -1(1,%r1)
#CHECK: error: invalid operand
#CHECK: unpk	0(1,%r1), 4096(1,%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: unpk	0(%r1,%r2), 0(1,%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: unpk	0(1,%r2), 0(%r1,%r2)
#CHECK: error: unknown token in expression
#CHECK: unpk	0(-), 0(1)

	unpk	0, 0(1)
	unpk	0(1), 0
	unpk	0(%r1), 0(1,%r1)
	unpk	0(1,%r1), 0(%r1)
	unpk	0(0,%r1), 0(1,%r1)
	unpk	0(1,%r1), 0(0,%r1)
	unpk	0(17,%r1), 0(1,%r1)
	unpk	0(1,%r1), 0(17,%r1)
	unpk	-1(1,%r1), 0(1,%r1)
	unpk	4096(1,%r1), 0(1,%r1)
	unpk	0(1,%r1), -1(1,%r1)
	unpk	0(1,%r1), 4096(1,%r1)
	unpk	0(%r1,%r2), 0(1,%r1)
	unpk	0(1,%r2), 0(%r1,%r2)
	unpk	0(-), 0(1)

#CHECK: error: missing length in address
#CHECK: unpka	0, 0
#CHECK: error: missing length in address
#CHECK: unpka	0(%r1), 0(%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: unpka	0(1,%r1), 0(2,%r1)
#CHECK: error: invalid operand
#CHECK: unpka	0(0,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: unpka	0(257,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: unpka	-1(1,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: unpka	4096(1,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: unpka	0(1,%r1), -1(%r1)
#CHECK: error: invalid operand
#CHECK: unpka	0(1,%r1), 4096(%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: unpka	0(%r1,%r2), 0(%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: unpka	0(1,%r2), 0(%r1,%r2)
#CHECK: error: unknown token in expression
#CHECK: unpka	0(-), 0

	unpka	0, 0
	unpka	0(%r1), 0(%r1)
	unpka	0(1,%r1), 0(2,%r1)
	unpka	0(0,%r1), 0(%r1)
	unpka	0(257,%r1), 0(%r1)
	unpka	-1(1,%r1), 0(%r1)
	unpka	4096(1,%r1), 0(%r1)
	unpka	0(1,%r1), -1(%r1)
	unpka	0(1,%r1), 4096(%r1)
	unpka	0(%r1,%r2), 0(%r1)
	unpka	0(1,%r2), 0(%r1,%r2)
	unpka	0(-), 0

#CHECK: error: missing length in address
#CHECK: unpku	0, 0
#CHECK: error: missing length in address
#CHECK: unpku	0(%r1), 0(%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: unpku	0(1,%r1), 0(2,%r1)
#CHECK: error: invalid operand
#CHECK: unpku	0(0,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: unpku	0(257,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: unpku	-1(1,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: unpku	4096(1,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: unpku	0(1,%r1), -1(%r1)
#CHECK: error: invalid operand
#CHECK: unpku	0(1,%r1), 4096(%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: unpku	0(%r1,%r2), 0(%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: unpku	0(1,%r2), 0(%r1,%r2)
#CHECK: error: unknown token in expression
#CHECK: unpku	0(-), 0

	unpku	0, 0
	unpku	0(%r1), 0(%r1)
	unpku	0(1,%r1), 0(2,%r1)
	unpku	0(0,%r1), 0(%r1)
	unpku	0(257,%r1), 0(%r1)
	unpku	-1(1,%r1), 0(%r1)
	unpku	4096(1,%r1), 0(%r1)
	unpku	0(1,%r1), -1(%r1)
	unpku	0(1,%r1), 4096(%r1)
	unpku	0(%r1,%r2), 0(%r1)
	unpku	0(1,%r2), 0(%r1,%r2)
	unpku	0(-), 0

#CHECK: error: invalid operand
#CHECK: x	%r0, -1
#CHECK: error: invalid operand
#CHECK: x	%r0, 4096

	x	%r0, -1
	x	%r0, 4096

#CHECK: error: missing length in address
#CHECK: xc	0, 0
#CHECK: error: missing length in address
#CHECK: xc	0(%r1), 0(%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: xc	0(1,%r1), 0(2,%r1)
#CHECK: error: invalid operand
#CHECK: xc	0(0,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: xc	0(257,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: xc	-1(1,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: xc	4096(1,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: xc	0(1,%r1), -1(%r1)
#CHECK: error: invalid operand
#CHECK: xc	0(1,%r1), 4096(%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: xc	0(%r1,%r2), 0(%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: xc	0(1,%r2), 0(%r1,%r2)
#CHECK: error: unknown token in expression
#CHECK: xc	0(-), 0

	xc	0, 0
	xc	0(%r1), 0(%r1)
	xc	0(1,%r1), 0(2,%r1)
	xc	0(0,%r1), 0(%r1)
	xc	0(257,%r1), 0(%r1)
	xc	-1(1,%r1), 0(%r1)
	xc	4096(1,%r1), 0(%r1)
	xc	0(1,%r1), -1(%r1)
	xc	0(1,%r1), 4096(%r1)
	xc	0(%r1,%r2), 0(%r1)
	xc	0(1,%r2), 0(%r1,%r2)
	xc	0(-), 0

#CHECK: error: invalid operand
#CHECK: xg	%r0, -524289
#CHECK: error: invalid operand
#CHECK: xg	%r0, 524288

	xg	%r0, -524289
	xg	%r0, 524288

#CHECK: error: instruction requires: distinct-ops
#CHECK: xgrk	%r2,%r3,%r4

	xgrk	%r2,%r3,%r4

#CHECK: error: invalid operand
#CHECK: xi	-1, 0
#CHECK: error: invalid operand
#CHECK: xi	4096, 0
#CHECK: error: invalid use of indexed addressing
#CHECK: xi	0(%r1,%r2), 0
#CHECK: error: invalid operand
#CHECK: xi	0, -1
#CHECK: error: invalid operand
#CHECK: xi	0, 256

	xi	-1, 0
	xi	4096, 0
	xi	0(%r1,%r2), 0
	xi	0, -1
	xi	0, 256

#CHECK: error: invalid operand
#CHECK: xihf	%r0, -1
#CHECK: error: invalid operand
#CHECK: xihf	%r0, 1 << 32

	xihf	%r0, -1
	xihf	%r0, 1 << 32

#CHECK: error: invalid operand
#CHECK: xilf	%r0, -1
#CHECK: error: invalid operand
#CHECK: xilf	%r0, 1 << 32

	xilf	%r0, -1
	xilf	%r0, 1 << 32

#CHECK: error: invalid operand
#CHECK: xiy	-524289, 0
#CHECK: error: invalid operand
#CHECK: xiy	524288, 0
#CHECK: error: invalid use of indexed addressing
#CHECK: xiy	0(%r1,%r2), 0
#CHECK: error: invalid operand
#CHECK: xiy	0, -1
#CHECK: error: invalid operand
#CHECK: xiy	0, 256

	xiy	-524289, 0
	xiy	524288, 0
	xiy	0(%r1,%r2), 0
	xiy	0, -1
	xiy	0, 256

#CHECK: error: instruction requires: distinct-ops
#CHECK: xrk	%r2,%r3,%r4

	xrk	%r2,%r3,%r4

#CHECK: error: invalid operand
#CHECK: xy	%r0, -524289
#CHECK: error: invalid operand
#CHECK: xy	%r0, 524288

	xy	%r0, -524289
	xy	%r0, 524288

#CHECK: error: missing length in address
#CHECK: zap	0, 0(1)
#CHECK: error: missing length in address
#CHECK: zap	0(1), 0
#CHECK: error: missing length in address
#CHECK: zap	0(%r1), 0(1,%r1)
#CHECK: error: missing length in address
#CHECK: zap	0(1,%r1), 0(%r1)
#CHECK: error: invalid operand
#CHECK: zap	0(0,%r1), 0(1,%r1)
#CHECK: error: invalid operand
#CHECK: zap	0(1,%r1), 0(0,%r1)
#CHECK: error: invalid operand
#CHECK: zap	0(17,%r1), 0(1,%r1)
#CHECK: error: invalid operand
#CHECK: zap	0(1,%r1), 0(17,%r1)
#CHECK: error: invalid operand
#CHECK: zap	-1(1,%r1), 0(1,%r1)
#CHECK: error: invalid operand
#CHECK: zap	4096(1,%r1), 0(1,%r1)
#CHECK: error: invalid operand
#CHECK: zap	0(1,%r1), -1(1,%r1)
#CHECK: error: invalid operand
#CHECK: zap	0(1,%r1), 4096(1,%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: zap	0(%r1,%r2), 0(1,%r1)
#CHECK: error: invalid use of indexed addressing
#CHECK: zap	0(1,%r2), 0(%r1,%r2)
#CHECK: error: unknown token in expression
#CHECK: zap	0(-), 0(1)

	zap	0, 0(1)
	zap	0(1), 0
	zap	0(%r1), 0(1,%r1)
	zap	0(1,%r1), 0(%r1)
	zap	0(0,%r1), 0(1,%r1)
	zap	0(1,%r1), 0(0,%r1)
	zap	0(17,%r1), 0(1,%r1)
	zap	0(1,%r1), 0(17,%r1)
	zap	-1(1,%r1), 0(1,%r1)
	zap	4096(1,%r1), 0(1,%r1)
	zap	0(1,%r1), -1(1,%r1)
	zap	0(1,%r1), 4096(1,%r1)
	zap	0(%r1,%r2), 0(1,%r1)
	zap	0(1,%r2), 0(%r1,%r2)
	zap	0(-), 0(1)
