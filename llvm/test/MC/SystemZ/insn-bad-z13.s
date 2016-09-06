# For z13 only.
# RUN: not llvm-mc -triple s390x-linux-gnu -mcpu=z13 < %s 2> %t
# RUN: FileCheck < %t %s

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

#CHECK: error: invalid operand
#CHECK: vftcidb	%v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vftcidb	%v0, %v0, 4096

	vftcidb	%v0, %v0, -1
	vftcidb	%v0, %v0, 4096

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
#CHECK: vgmb	%v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vgmb	%v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vgmb	%v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vgmb	%v0, 256, 0

	vgmb	%v0, 0, -1
	vgmb	%v0, 0, -1
	vgmb	%v0, -1, 0
	vgmb	%v0, 256, 0

#CHECK: error: invalid operand
#CHECK: vgmf	%v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vgmf	%v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vgmf	%v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vgmf	%v0, 256, 0

	vgmf	%v0, 0, -1
	vgmf	%v0, 0, -1
	vgmf	%v0, -1, 0
	vgmf	%v0, 256, 0

#CHECK: error: invalid operand
#CHECK: vgmg	%v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vgmg	%v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vgmg	%v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vgmg	%v0, 256, 0

	vgmg	%v0, 0, -1
	vgmg	%v0, 0, -1
	vgmg	%v0, -1, 0
	vgmg	%v0, 256, 0

#CHECK: error: invalid operand
#CHECK: vgmh	%v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vgmh	%v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vgmh	%v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vgmh	%v0, 256, 0

	vgmh	%v0, 0, -1
	vgmh	%v0, 0, -1
	vgmh	%v0, -1, 0
	vgmh	%v0, 256, 0

#CHECK: error: invalid operand
#CHECK: vl	%v0, -1
#CHECK: error: invalid operand
#CHECK: vl	%v0, 4096
#CHECK: error: invalid use of vector addressing
#CHECK: vl	%v0, 0(%v1,%r2)

	vl	%v0, -1
	vl	%v0, 4096
	vl	%v0, 0(%v1,%r2)

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
#CHECK: vlgvb	%r0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vlgvb	%r0, %v0, 4096
#CHECK: error: %r0 used in an address
#CHECK: vlgvb	%r0, %v0, 0(%r0)

	vlgvb	%r0, %v0, -1
	vlgvb	%r0, %v0, 4096
	vlgvb	%r0, %v0, 0(%r0)

#CHECK: error: invalid operand
#CHECK: vlgvf	%r0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vlgvf	%r0, %v0, 4096
#CHECK: error: %r0 used in an address
#CHECK: vlgvf	%r0, %v0, 0(%r0)

	vlgvf	%r0, %v0, -1
	vlgvf	%r0, %v0, 4096
	vlgvf	%r0, %v0, 0(%r0)

#CHECK: error: invalid operand
#CHECK: vlgvg	%r0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vlgvg	%r0, %v0, 4096
#CHECK: error: %r0 used in an address
#CHECK: vlgvg	%r0, %v0, 0(%r0)

	vlgvg	%r0, %v0, -1
	vlgvg	%r0, %v0, 4096
	vlgvg	%r0, %v0, 0(%r0)

#CHECK: error: invalid operand
#CHECK: vlgvh	%r0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vlgvh	%r0, %v0, 4096
#CHECK: error: %r0 used in an address
#CHECK: vlgvh	%r0, %v0, 0(%r0)

	vlgvh	%r0, %v0, -1
	vlgvh	%r0, %v0, 4096
	vlgvh	%r0, %v0, 0(%r0)

#CHECK: error: invalid operand
#CHECK: vll	%v0, %r0, -1
#CHECK: error: invalid operand
#CHECK: vll	%v0, %r0, 4096
#CHECK: error: %r0 used in an address
#CHECK: vll	%v0, %r0, 0(%r0)

	vll	%v0, %r0, -1
	vll	%v0, %r0, 4096
	vll	%v0, %r0, 0(%r0)

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

#CHECK: error: invalid operand
#CHECK: vlm	%v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vlm	%v0, %v0, 4096

	vlm	%v0, %v0, -1
	vlm	%v0, %v0, 4096

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

#CHECK: error: invalid operand
#CHECK: vlvgb	%v0, %r0, -1
#CHECK: error: invalid operand
#CHECK: vlvgb	%v0, %r0, 4096
#CHECK: error: %r0 used in an address
#CHECK: vlvgb	%v0, %r0, 0(%r0)

	vlvgb	%v0, %r0, -1
	vlvgb	%v0, %r0, 4096
	vlvgb	%v0, %r0, 0(%r0)

#CHECK: error: invalid operand
#CHECK: vlvgf	%v0, %r0, -1
#CHECK: error: invalid operand
#CHECK: vlvgf	%v0, %r0, 4096
#CHECK: error: %r0 used in an address
#CHECK: vlvgf	%v0, %r0, 0(%r0)

	vlvgf	%v0, %r0, -1
	vlvgf	%v0, %r0, 4096
	vlvgf	%v0, %r0, 0(%r0)

#CHECK: error: invalid operand
#CHECK: vlvgg	%v0, %r0, -1
#CHECK: error: invalid operand
#CHECK: vlvgg	%v0, %r0, 4096
#CHECK: error: %r0 used in an address
#CHECK: vlvgg	%v0, %r0, 0(%r0)

	vlvgg	%v0, %r0, -1
	vlvgg	%v0, %r0, 4096
	vlvgg	%v0, %r0, 0(%r0)

#CHECK: error: invalid operand
#CHECK: vlvgh	%v0, %r0, -1
#CHECK: error: invalid operand
#CHECK: vlvgh	%v0, %r0, 4096
#CHECK: error: %r0 used in an address
#CHECK: vlvgh	%v0, %r0, 0(%r0)

	vlvgh	%v0, %r0, -1
	vlvgh	%v0, %r0, 4096
	vlvgh	%v0, %r0, 0(%r0)

#CHECK: error: invalid operand
#CHECK: vpdi	%v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vpdi	%v0, %v0, %v0, 16

	vpdi	%v0, %v0, %v0, -1
	vpdi	%v0, %v0, %v0, 16

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

#CHECK: error: invalid operand
#CHECK: vsldb	%v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vsldb	%v0, %v0, %v0, 256

	vsldb	%v0, %v0, %v0, -1
	vsldb	%v0, %v0, %v0, 256

#CHECK: error: invalid operand
#CHECK: vst	%v0, -1
#CHECK: error: invalid operand
#CHECK: vst	%v0, 4096
#CHECK: error: invalid use of vector addressing
#CHECK: vst	%v0, 0(%v1,%r2)

	vst	%v0, -1
	vst	%v0, 4096
	vst	%v0, 0(%v1,%r2)

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
#CHECK: error: %r0 used in an address
#CHECK: vstl	%v0, %r0, 0(%r0)

	vstl	%v0, %r0, -1
	vstl	%v0, %r0, 4096
	vstl	%v0, %r0, 0(%r0)

#CHECK: error: invalid operand
#CHECK: vstm	%v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vstm	%v0, %v0, 4096

	vstm	%v0, %v0, -1
	vstm	%v0, %v0, 4096

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

#CHECK: error: invalid operand
#CHECK: wftcidb	%v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: wftcidb	%v0, %v0, 4096

	wftcidb	%v0, %v0, -1
	wftcidb	%v0, %v0, 4096

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
#CHECK: locghie	%r0, 66000
#CHECK: error: invalid operand
#CHECK: locghie	%f0, 0
#CHECK: error: invalid operand
#CHECK: locghie	0, %r0
        
        locghie	%r0, 66000
        locghie	%f0, 0
        locghie	0, %r0        
        
