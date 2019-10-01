# For z15 only.
# RUN: not llvm-mc -triple s390x-linux-gnu -mcpu=z15 < %s 2> %t
# RUN: FileCheck < %t %s
# RUN: not llvm-mc -triple s390x-linux-gnu -mcpu=arch13 < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid register pair
#CHECK: dfltcc	%r1, %r2, %r4
#CHECK: error: invalid register pair
#CHECK: dfltcc	%r2, %r1, %r4

	dfltcc	%r1, %r2, %r4
	dfltcc	%r2, %r1, %r4

#CHECK: error: invalid register pair
#CHECK: kdsa	%r0, %r1

	kdsa	%r0, %r1

#CHECK: error: invalid operand
#CHECK: ldrv	%f0, -1
#CHECK: error: invalid operand
#CHECK: ldrv	%f0, 4096
#CHECK: error: invalid use of vector addressing
#CHECK: ldrv	%f0, 0(%v1,%r2)

	ldrv	%f0, -1
	ldrv	%f0, 4096
	ldrv	%f0, 0(%v1,%r2)

#CHECK: error: invalid operand
#CHECK: lerv	%f0, -1
#CHECK: error: invalid operand
#CHECK: lerv	%f0, 4096
#CHECK: error: invalid use of vector addressing
#CHECK: lerv	%f0, 0(%v1,%r2)

	lerv	%f0, -1
	lerv	%f0, 4096
	lerv	%f0, 0(%v1,%r2)

#CHECK: error: invalid use of indexed addressing
#CHECK: mvcrl	160(%r1,%r15),160(%r15)
#CHECK: error: invalid operand
#CHECK: mvcrl	-1(%r1),160(%r15)
#CHECK: error: invalid operand
#CHECK: mvcrl	4096(%r1),160(%r15)
#CHECK: error: invalid operand
#CHECK: mvcrl	0(%r1),-1(%r15)
#CHECK: error: invalid operand
#CHECK: mvcrl	0(%r1),4096(%r15)

        mvcrl	160(%r1,%r15),160(%r15)
        mvcrl	-1(%r1),160(%r15)
        mvcrl	4096(%r1),160(%r15)
        mvcrl	0(%r1),-1(%r15)
        mvcrl	0(%r1),4096(%r15)

#CHECK: error: invalid operand
#CHECK: popcnt	%r2, %r4, -1
#CHECK: error: invalid operand
#CHECK: popcnt	%r2, %r4, 16

	popcnt	%r2, %r4, -1
	popcnt	%r2, %r4, 16

#CHECK: error: invalid operand
#CHECK: selgr	%r0, %r0, %r0, -1
#CHECK: error: invalid operand
#CHECK: selgr	%r0, %r0, %r0, 16

	selgr	%r0, %r0, %r0, -1
	selgr	%r0, %r0, %r0, 16

#CHECK: error: invalid operand
#CHECK: selfhr	%r0, %r0, %r0, -1
#CHECK: error: invalid operand
#CHECK: selfhr	%r0, %r0, %r0, 16

	selfhr	%r0, %r0, %r0, -1
	selfhr	%r0, %r0, %r0, 16

#CHECK: error: invalid operand
#CHECK: selr	%r0, %r0, %r0, -1
#CHECK: error: invalid operand
#CHECK: selr	%r0, %r0, %r0, 16

	selr	%r0, %r0, %r0, -1
	selr	%r0, %r0, %r0, 16

#CHECK: error: invalid register pair
#CHECK: sortl	%r1, %r2
#CHECK: error: invalid register pair
#CHECK: sortl	%r2, %r1

	sortl	%r1, %r2
	sortl	%r2, %r1

#CHECK: error: invalid operand
#CHECK: stdrv	%f0, -1
#CHECK: error: invalid operand
#CHECK: stdrv	%f0, 4096
#CHECK: error: invalid use of vector addressing
#CHECK: stdrv	%f0, 0(%v1,%r2)

	stdrv	%f0, -1
	stdrv	%f0, 4096
	stdrv	%f0, 0(%v1,%r2)

#CHECK: error: invalid operand
#CHECK: sterv	%f0, -1
#CHECK: error: invalid operand
#CHECK: sterv	%f0, 4096
#CHECK: error: invalid use of vector addressing
#CHECK: sterv	%f0, 0(%v1,%r2)

	sterv	%f0, -1
	sterv	%f0, 4096
	sterv	%f0, 0(%v1,%r2)

#CHECK: error: invalid operand
#CHECK: vcefb	%v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vcefb	%v0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: vcefb	%v0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vcefb	%v0, %v0, 16, 0

	vcefb	%v0, %v0, 0, -1
	vcefb	%v0, %v0, 0, 16
	vcefb	%v0, %v0, -1, 0
	vcefb	%v0, %v0, 16, 0

#CHECK: error: invalid operand
#CHECK: vcelfb	%v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vcelfb	%v0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: vcelfb	%v0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vcelfb	%v0, %v0, 16, 0

	vcelfb	%v0, %v0, 0, -1
	vcelfb	%v0, %v0, 0, 16
	vcelfb	%v0, %v0, -1, 0
	vcelfb	%v0, %v0, 16, 0

#CHECK: error: invalid operand
#CHECK: vcfeb	%v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vcfeb	%v0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: vcfeb	%v0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vcfeb	%v0, %v0, 16, 0

	vcfeb	%v0, %v0, 0, -1
	vcfeb	%v0, %v0, 0, 16
	vcfeb	%v0, %v0, -1, 0
	vcfeb	%v0, %v0, 16, 0

#CHECK: error: invalid operand
#CHECK: vcfpl	%v0, %v0, 0, 0, -1
#CHECK: error: invalid operand
#CHECK: vcfpl	%v0, %v0, 0, 0, 16
#CHECK: error: invalid operand
#CHECK: vcfpl	%v0, %v0, 0, -1, 0
#CHECK: error: invalid operand
#CHECK: vcfpl	%v0, %v0, 0, 16, 0
#CHECK: error: invalid operand
#CHECK: vcfpl	%v0, %v0, -1, 0, 0
#CHECK: error: invalid operand
#CHECK: vcfpl	%v0, %v0, 16, 0, 0

	vcfpl	%v0, %v0, 0, 0, -1
	vcfpl	%v0, %v0, 0, 0, 16
	vcfpl	%v0, %v0, 0, -1, 0
	vcfpl	%v0, %v0, 0, 16, 0
	vcfpl	%v0, %v0, -1, 0, 0
	vcfpl	%v0, %v0, 16, 0, 0

#CHECK: error: invalid operand
#CHECK: vcfps	%v0, %v0, 0, 0, -1
#CHECK: error: invalid operand
#CHECK: vcfps	%v0, %v0, 0, 0, 16
#CHECK: error: invalid operand
#CHECK: vcfps	%v0, %v0, 0, -1, 0
#CHECK: error: invalid operand
#CHECK: vcfps	%v0, %v0, 0, 16, 0
#CHECK: error: invalid operand
#CHECK: vcfps	%v0, %v0, -1, 0, 0
#CHECK: error: invalid operand
#CHECK: vcfps	%v0, %v0, 16, 0, 0

	vcfps	%v0, %v0, 0, 0, -1
	vcfps	%v0, %v0, 0, 0, 16
	vcfps	%v0, %v0, 0, -1, 0
	vcfps	%v0, %v0, 0, 16, 0
	vcfps	%v0, %v0, -1, 0, 0
	vcfps	%v0, %v0, 16, 0, 0

#CHECK: error: invalid operand
#CHECK: vclfeb	%v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vclfeb	%v0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: vclfeb	%v0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vclfeb	%v0, %v0, 16, 0

	vclfeb	%v0, %v0, 0, -1
	vclfeb	%v0, %v0, 0, 16
	vclfeb	%v0, %v0, -1, 0
	vclfeb	%v0, %v0, 16, 0

#CHECK: error: invalid operand
#CHECK: vclfp	%v0, %v0, 0, 0, -1
#CHECK: error: invalid operand
#CHECK: vclfp	%v0, %v0, 0, 0, 16
#CHECK: error: invalid operand
#CHECK: vclfp	%v0, %v0, 0, -1, 0
#CHECK: error: invalid operand
#CHECK: vclfp	%v0, %v0, 0, 16, 0
#CHECK: error: invalid operand
#CHECK: vclfp	%v0, %v0, -1, 0, 0
#CHECK: error: invalid operand
#CHECK: vclfp	%v0, %v0, 16, 0, 0

	vclfp	%v0, %v0, 0, 0, -1
	vclfp	%v0, %v0, 0, 0, 16
	vclfp	%v0, %v0, 0, -1, 0
	vclfp	%v0, %v0, 0, 16, 0
	vclfp	%v0, %v0, -1, 0, 0
	vclfp	%v0, %v0, 16, 0, 0

#CHECK: error: invalid operand
#CHECK: vcsfp	%v0, %v0, 0, 0, -1
#CHECK: error: invalid operand
#CHECK: vcsfp	%v0, %v0, 0, 0, 16
#CHECK: error: invalid operand
#CHECK: vcsfp	%v0, %v0, 0, -1, 0
#CHECK: error: invalid operand
#CHECK: vcsfp	%v0, %v0, 0, 16, 0
#CHECK: error: invalid operand
#CHECK: vcsfp	%v0, %v0, -1, 0, 0
#CHECK: error: invalid operand
#CHECK: vcsfp	%v0, %v0, 16, 0, 0

	vcsfp	%v0, %v0, 0, 0, -1
	vcsfp	%v0, %v0, 0, 0, 16
	vcsfp	%v0, %v0, 0, -1, 0
	vcsfp	%v0, %v0, 0, 16, 0
	vcsfp	%v0, %v0, -1, 0, 0
	vcsfp	%v0, %v0, 16, 0, 0

#CHECK: error: invalid operand
#CHECK: vcvb	%r0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vcvb	%r0, %v0, 0, 16

	vcvb	%r0, %v0, 0, -1
	vcvb	%r0, %v0, 0, 16

#CHECK: error: invalid operand
#CHECK: vcvbg	%r0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vcvbg	%r0, %v0, 0, 16

	vcvbg	%r0, %v0, 0, -1
	vcvbg	%r0, %v0, 0, 16

#CHECK: error: invalid operand
#CHECK: vlbr	%v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vlbr	%v0, 0, 16
#CHECK: error: invalid operand
#CHECK: vlbr	%v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vlbr	%v0, 4096, 0
#CHECK: error: invalid use of vector addressing
#CHECK: vlbr	%v0, 0(%v1,%r2), 0

	vlbr	%v0, 0, -1
	vlbr	%v0, 0, 16
	vlbr	%v0, -1, 0
	vlbr	%v0, 4096, 0
	vlbr	%v0, 0(%v1,%r2), 0

#CHECK: error: invalid operand
#CHECK: vlbrf	%v0, -1
#CHECK: error: invalid operand
#CHECK: vlbrf	%v0, 4096
#CHECK: error: invalid use of vector addressing
#CHECK: vlbrf	%v0, 0(%v1,%r2)

	vlbrf	%v0, -1
	vlbrf	%v0, 4096
	vlbrf	%v0, 0(%v1,%r2)

#CHECK: error: invalid operand
#CHECK: vlbrg	%v0, -1
#CHECK: error: invalid operand
#CHECK: vlbrg	%v0, 4096
#CHECK: error: invalid use of vector addressing
#CHECK: vlbrg	%v0, 0(%v1,%r2)

	vlbrg	%v0, -1
	vlbrg	%v0, 4096
	vlbrg	%v0, 0(%v1,%r2)

#CHECK: error: invalid operand
#CHECK: vlbrh	%v0, -1
#CHECK: error: invalid operand
#CHECK: vlbrh	%v0, 4096
#CHECK: error: invalid use of vector addressing
#CHECK: vlbrh	%v0, 0(%v1,%r2)

	vlbrh	%v0, -1
	vlbrh	%v0, 4096
	vlbrh	%v0, 0(%v1,%r2)

#CHECK: error: invalid operand
#CHECK: vlbrq	%v0, -1
#CHECK: error: invalid operand
#CHECK: vlbrq	%v0, 4096
#CHECK: error: invalid use of vector addressing
#CHECK: vlbrq	%v0, 0(%v1,%r2)

	vlbrq	%v0, -1
	vlbrq	%v0, 4096
	vlbrq	%v0, 0(%v1,%r2)

#CHECK: error: invalid operand
#CHECK: vlbrrep	%v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vlbrrep	%v0, 0, 16
#CHECK: error: invalid operand
#CHECK: vlbrrep	%v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vlbrrep	%v0, 4096, 0
#CHECK: error: invalid use of vector addressing
#CHECK: vlbrrep	%v0, 0(%v1,%r2), 0

	vlbrrep	%v0, 0, -1
	vlbrrep	%v0, 0, 16
	vlbrrep	%v0, -1, 0
	vlbrrep	%v0, 4096, 0
	vlbrrep	%v0, 0(%v1,%r2), 0

#CHECK: error: invalid operand
#CHECK: vlbrrepf %v0, -1
#CHECK: error: invalid operand
#CHECK: vlbrrepf %v0, 4096
#CHECK: error: invalid use of vector addressing
#CHECK: vlbrrepf %v0, 0(%v1,%r2)

	vlbrrepf %v0, -1
	vlbrrepf %v0, 4096
	vlbrrepf %v0, 0(%v1,%r2)

#CHECK: error: invalid operand
#CHECK: vlbrrepg %v0, -1
#CHECK: error: invalid operand
#CHECK: vlbrrepg %v0, 4096
#CHECK: error: invalid use of vector addressing
#CHECK: vlbrrepg %v0, 0(%v1,%r2)

	vlbrrepg %v0, -1
	vlbrrepg %v0, 4096
	vlbrrepg %v0, 0(%v1,%r2)

#CHECK: error: invalid operand
#CHECK: vlbrreph %v0, -1
#CHECK: error: invalid operand
#CHECK: vlbrreph %v0, 4096
#CHECK: error: invalid use of vector addressing
#CHECK: vlbrreph %v0, 0(%v1,%r2)

	vlbrreph %v0, -1
	vlbrreph %v0, 4096
	vlbrreph %v0, 0(%v1,%r2)

#CHECK: error: invalid operand
#CHECK: vlebrf	%v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vlebrf	%v0, 0, 4
#CHECK: error: invalid operand
#CHECK: vlebrf	%v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vlebrf	%v0, 4096, 0
#CHECK: error: invalid use of vector addressing
#CHECK: vlebrf	%v0, 0(%v1,%r2), 0

	vlebrf	%v0, 0, -1
	vlebrf	%v0, 0, 4
	vlebrf	%v0, -1, 0
	vlebrf	%v0, 4096, 0
	vlebrf	%v0, 0(%v1,%r2), 0

#CHECK: error: invalid operand
#CHECK: vlebrg	%v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vlebrg	%v0, 0, 2
#CHECK: error: invalid operand
#CHECK: vlebrg	%v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vlebrg	%v0, 4096, 0
#CHECK: error: invalid use of vector addressing
#CHECK: vlebrg	%v0, 0(%v1,%r2), 0

	vlebrg	%v0, 0, -1
	vlebrg	%v0, 0, 2
	vlebrg	%v0, -1, 0
	vlebrg	%v0, 4096, 0
	vlebrg	%v0, 0(%v1,%r2), 0

#CHECK: error: invalid operand
#CHECK: vlebrh	%v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vlebrh	%v0, 0, 8
#CHECK: error: invalid operand
#CHECK: vlebrh	%v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vlebrh	%v0, 4096, 0
#CHECK: error: invalid use of vector addressing
#CHECK: vlebrh	%v0, 0(%v1,%r2), 0

	vlebrh	%v0, 0, -1
	vlebrh	%v0, 0, 8
	vlebrh	%v0, -1, 0
	vlebrh	%v0, 4096, 0
	vlebrh	%v0, 0(%v1,%r2), 0

#CHECK: error: invalid operand
#CHECK: vler	%v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vler	%v0, 0, 16
#CHECK: error: invalid operand
#CHECK: vler	%v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vler	%v0, 4096, 0
#CHECK: error: invalid use of vector addressing
#CHECK: vler	%v0, 0(%v1,%r2), 0

	vler	%v0, 0, -1
	vler	%v0, 0, 16
	vler	%v0, -1, 0
	vler	%v0, 4096, 0
	vler	%v0, 0(%v1,%r2), 0

#CHECK: error: invalid operand
#CHECK: vlerf	%v0, -1
#CHECK: error: invalid operand
#CHECK: vlerf	%v0, 4096
#CHECK: error: invalid use of vector addressing
#CHECK: vlerf	%v0, 0(%v1,%r2)

	vlerf	%v0, -1
	vlerf	%v0, 4096
	vlerf	%v0, 0(%v1,%r2)

#CHECK: error: invalid operand
#CHECK: vlerg	%v0, -1
#CHECK: error: invalid operand
#CHECK: vlerg	%v0, 4096
#CHECK: error: invalid use of vector addressing
#CHECK: vlerg	%v0, 0(%v1,%r2)

	vlerg	%v0, -1
	vlerg	%v0, 4096
	vlerg	%v0, 0(%v1,%r2)

#CHECK: error: invalid operand
#CHECK: vlerh	%v0, -1
#CHECK: error: invalid operand
#CHECK: vlerh	%v0, 4096
#CHECK: error: invalid use of vector addressing
#CHECK: vlerh	%v0, 0(%v1,%r2)

	vlerh	%v0, -1
	vlerh	%v0, 4096
	vlerh	%v0, 0(%v1,%r2)

#CHECK: error: invalid operand
#CHECK: vllebrz	%v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vllebrz	%v0, 0, 16
#CHECK: error: invalid operand
#CHECK: vllebrz	%v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vllebrz	%v0, 4096, 0
#CHECK: error: invalid use of vector addressing
#CHECK: vllebrz	%v0, 0(%v1,%r2), 0

	vllebrz	%v0, 0, -1
	vllebrz	%v0, 0, 16
	vllebrz	%v0, -1, 0
	vllebrz	%v0, 4096, 0
	vllebrz	%v0, 0(%v1,%r2), 0

#CHECK: error: invalid operand
#CHECK: vllebrze	%v0, -1
#CHECK: error: invalid operand
#CHECK: vllebrze	%v0, 4096
#CHECK: error: invalid use of vector addressing
#CHECK: vllebrze	%v0, 0(%v1,%r2)

	vllebrze	%v0, -1
	vllebrze	%v0, 4096
	vllebrze	%v0, 0(%v1,%r2)

#CHECK: error: invalid operand
#CHECK: vllebrzf	%v0, -1
#CHECK: error: invalid operand
#CHECK: vllebrzf	%v0, 4096
#CHECK: error: invalid use of vector addressing
#CHECK: vllebrzf	%v0, 0(%v1,%r2)

	vllebrzf	%v0, -1
	vllebrzf	%v0, 4096
	vllebrzf	%v0, 0(%v1,%r2)

#CHECK: error: invalid operand
#CHECK: vllebrzg	%v0, -1
#CHECK: error: invalid operand
#CHECK: vllebrzg	%v0, 4096
#CHECK: error: invalid use of vector addressing
#CHECK: vllebrzg	%v0, 0(%v1,%r2)

	vllebrzg	%v0, -1
	vllebrzg	%v0, 4096
	vllebrzg	%v0, 0(%v1,%r2)

#CHECK: error: invalid operand
#CHECK: vllebrzh	%v0, -1
#CHECK: error: invalid operand
#CHECK: vllebrzh	%v0, 4096
#CHECK: error: invalid use of vector addressing
#CHECK: vllebrzh	%v0, 0(%v1,%r2)

	vllebrzh	%v0, -1
	vllebrzh	%v0, 4096
	vllebrzh	%v0, 0(%v1,%r2)

#CHECK: error: invalid operand
#CHECK: vsld	%v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vsld	%v0, %v0, %v0, 256

	vsld	%v0, %v0, %v0, -1
	vsld	%v0, %v0, %v0, 256

#CHECK: error: invalid operand
#CHECK: vsrd	%v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vsrd	%v0, %v0, %v0, 256

	vsrd	%v0, %v0, %v0, -1
	vsrd	%v0, %v0, %v0, 256

#CHECK: error: invalid operand
#CHECK: vstbr	%v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vstbr	%v0, 0, 16
#CHECK: error: invalid operand
#CHECK: vstbr	%v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vstbr	%v0, 4096, 0
#CHECK: error: invalid use of vector addressing
#CHECK: vstbr	%v0, 0(%v1,%r2), 0

	vstbr	%v0, 0, -1
	vstbr	%v0, 0, 16
	vstbr	%v0, -1, 0
	vstbr	%v0, 4096, 0
	vstbr	%v0, 0(%v1,%r2), 0

#CHECK: error: invalid operand
#CHECK: vstbrf	%v0, -1
#CHECK: error: invalid operand
#CHECK: vstbrf	%v0, 4096
#CHECK: error: invalid use of vector addressing
#CHECK: vstbrf	%v0, 0(%v1,%r2)

	vstbrf	%v0, -1
	vstbrf	%v0, 4096
	vstbrf	%v0, 0(%v1,%r2)

#CHECK: error: invalid operand
#CHECK: vstbrg	%v0, -1
#CHECK: error: invalid operand
#CHECK: vstbrg	%v0, 4096
#CHECK: error: invalid use of vector addressing
#CHECK: vstbrg	%v0, 0(%v1,%r2)

	vstbrg	%v0, -1
	vstbrg	%v0, 4096
	vstbrg	%v0, 0(%v1,%r2)

#CHECK: error: invalid operand
#CHECK: vstbrh	%v0, -1
#CHECK: error: invalid operand
#CHECK: vstbrh	%v0, 4096
#CHECK: error: invalid use of vector addressing
#CHECK: vstbrh	%v0, 0(%v1,%r2)

	vstbrh	%v0, -1
	vstbrh	%v0, 4096
	vstbrh	%v0, 0(%v1,%r2)

#CHECK: error: invalid operand
#CHECK: vstbrq	%v0, -1
#CHECK: error: invalid operand
#CHECK: vstbrq	%v0, 4096
#CHECK: error: invalid use of vector addressing
#CHECK: vstbrq	%v0, 0(%v1,%r2)

	vstbrq	%v0, -1
	vstbrq	%v0, 4096
	vstbrq	%v0, 0(%v1,%r2)

#CHECK: error: invalid operand
#CHECK: vstebrf	%v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vstebrf	%v0, 0, 4
#CHECK: error: invalid operand
#CHECK: vstebrf	%v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vstebrf	%v0, 4096, 0
#CHECK: error: invalid use of vector addressing
#CHECK: vstebrf	%v0, 0(%v1,%r2), 0

	vstebrf	%v0, 0, -1
	vstebrf	%v0, 0, 4
	vstebrf	%v0, -1, 0
	vstebrf	%v0, 4096, 0
	vstebrf	%v0, 0(%v1,%r2), 0

#CHECK: error: invalid operand
#CHECK: vstebrg	%v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vstebrg	%v0, 0, 2
#CHECK: error: invalid operand
#CHECK: vstebrg	%v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vstebrg	%v0, 4096, 0
#CHECK: error: invalid use of vector addressing
#CHECK: vstebrg	%v0, 0(%v1,%r2), 0

	vstebrg	%v0, 0, -1
	vstebrg	%v0, 0, 2
	vstebrg	%v0, -1, 0
	vstebrg	%v0, 4096, 0
	vstebrg	%v0, 0(%v1,%r2), 0

#CHECK: error: invalid operand
#CHECK: vstebrh	%v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vstebrh	%v0, 0, 8
#CHECK: error: invalid operand
#CHECK: vstebrh	%v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vstebrh	%v0, 4096, 0
#CHECK: error: invalid use of vector addressing
#CHECK: vstebrh	%v0, 0(%v1,%r2), 0

	vstebrh	%v0, 0, -1
	vstebrh	%v0, 0, 8
	vstebrh	%v0, -1, 0
	vstebrh	%v0, 4096, 0
	vstebrh	%v0, 0(%v1,%r2), 0

#CHECK: error: invalid operand
#CHECK: vster	%v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vster	%v0, 0, 16
#CHECK: error: invalid operand
#CHECK: vster	%v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vster	%v0, 4096, 0
#CHECK: error: invalid use of vector addressing
#CHECK: vster	%v0, 0(%v1,%r2), 0

	vster	%v0, 0, -1
	vster	%v0, 0, 16
	vster	%v0, -1, 0
	vster	%v0, 4096, 0
	vster	%v0, 0(%v1,%r2), 0

#CHECK: error: invalid operand
#CHECK: vsterf	%v0, -1
#CHECK: error: invalid operand
#CHECK: vsterf	%v0, 4096
#CHECK: error: invalid use of vector addressing
#CHECK: vsterf	%v0, 0(%v1,%r2)

	vsterf	%v0, -1
	vsterf	%v0, 4096
	vsterf	%v0, 0(%v1,%r2)

#CHECK: error: invalid operand
#CHECK: vsterg	%v0, -1
#CHECK: error: invalid operand
#CHECK: vsterg	%v0, 4096
#CHECK: error: invalid use of vector addressing
#CHECK: vsterg	%v0, 0(%v1,%r2)

	vsterg	%v0, -1
	vsterg	%v0, 4096
	vsterg	%v0, 0(%v1,%r2)

#CHECK: error: invalid operand
#CHECK: vsterh	%v0, -1
#CHECK: error: invalid operand
#CHECK: vsterh	%v0, 4096
#CHECK: error: invalid use of vector addressing
#CHECK: vsterh	%v0, 0(%v1,%r2)

	vsterh	%v0, -1
	vsterh	%v0, 4096
	vsterh	%v0, 0(%v1,%r2)

#CHECK: error: invalid operand
#CHECK: vstrs    %v0, %v0, %v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vstrs    %v0, %v0, %v0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: vstrs    %v0, %v0, %v0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vstrs    %v0, %v0, %v0, %v0, 16, 0
#CHECK: error: too few operands
#CHECK: vstrs    %v0, %v0, %v0, %v0
#CHECK: error: invalid operand
#CHECK: vstrs    %v0, %v0, %v0, %v0, 0, 0, 0

	vstrs    %v0, %v0, %v0, %v0, 0, -1
	vstrs    %v0, %v0, %v0, %v0, 0, 16
	vstrs    %v0, %v0, %v0, %v0, -1, 0
	vstrs    %v0, %v0, %v0, %v0, 16, 0
	vstrs    %v0, %v0, %v0, %v0
	vstrs    %v0, %v0, %v0, %v0, 0, 0, 0

#CHECK: error: invalid operand
#CHECK: vstrsb   %v0, %v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vstrsb   %v0, %v0, %v0, %v0, 16
#CHECK: error: too few operands
#CHECK: vstrsb   %v0, %v0, %v0
#CHECK: error: invalid operand
#CHECK: vstrsb   %v0, %v0, %v0, %v0, 0, 0

	vstrsb   %v0, %v0, %v0, %v0, -1
	vstrsb   %v0, %v0, %v0, %v0, 16
	vstrsb   %v0, %v0, %v0
	vstrsb   %v0, %v0, %v0, %v0, 0, 0

#CHECK: error: invalid operand
#CHECK: vstrsf   %v0, %v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vstrsf   %v0, %v0, %v0, %v0, 16
#CHECK: error: too few operands
#CHECK: vstrsf   %v0, %v0, %v0
#CHECK: error: invalid operand
#CHECK: vstrsf   %v0, %v0, %v0, %v0, 0, 0

	vstrsf   %v0, %v0, %v0, %v0, -1
	vstrsf   %v0, %v0, %v0, %v0, 16
	vstrsf   %v0, %v0, %v0
	vstrsf   %v0, %v0, %v0, %v0, 0, 0

#CHECK: error: invalid operand
#CHECK: vstrsh   %v0, %v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vstrsh   %v0, %v0, %v0, %v0, 16
#CHECK: error: too few operands
#CHECK: vstrsh   %v0, %v0, %v0
#CHECK: error: invalid operand
#CHECK: vstrsh   %v0, %v0, %v0, %v0, 0, 0

	vstrsh   %v0, %v0, %v0, %v0, -1
	vstrsh   %v0, %v0, %v0, %v0, 16
	vstrsh   %v0, %v0, %v0
	vstrsh   %v0, %v0, %v0, %v0, 0, 0

#CHECK: error: invalid operand
#CHECK: vstrszb  %v0, %v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vstrszb  %v0, %v0, %v0, %v0, 16
#CHECK: error: too few operands
#CHECK: vstrszb  %v0, %v0, %v0
#CHECK: error: invalid operand
#CHECK: vstrszb  %v0, %v0, %v0, %v0, 0, 0

	vstrszb  %v0, %v0, %v0, %v0, -1
	vstrszb  %v0, %v0, %v0, %v0, 16
	vstrszb  %v0, %v0, %v0
	vstrszb  %v0, %v0, %v0, %v0, 0, 0

#CHECK: error: invalid operand
#CHECK: vstrszf  %v0, %v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vstrszf  %v0, %v0, %v0, %v0, 16
#CHECK: error: too few operands
#CHECK: vstrszf  %v0, %v0, %v0
#CHECK: error: invalid operand
#CHECK: vstrszf  %v0, %v0, %v0, %v0, 0, 0

	vstrszf  %v0, %v0, %v0, %v0, -1
	vstrszf  %v0, %v0, %v0, %v0, 16
	vstrszf  %v0, %v0, %v0
	vstrszf  %v0, %v0, %v0, %v0, 0, 0

#CHECK: error: invalid operand
#CHECK: vstrszh  %v0, %v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vstrszh  %v0, %v0, %v0, %v0, 16
#CHECK: error: too few operands
#CHECK: vstrszh  %v0, %v0, %v0
#CHECK: error: invalid operand
#CHECK: vstrszh  %v0, %v0, %v0, %v0, 0, 0

	vstrszh  %v0, %v0, %v0, %v0, -1
	vstrszh  %v0, %v0, %v0, %v0, 16
	vstrszh  %v0, %v0, %v0
	vstrszh  %v0, %v0, %v0, %v0, 0, 0

#CHECK: error: invalid operand
#CHECK: wcefb	%v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: wcefb	%v0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: wcefb	%v0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: wcefb	%v0, %v0, 16, 0

	wcefb	%v0, %v0, 0, -1
	wcefb	%v0, %v0, 0, 16
	wcefb	%v0, %v0, -1, 0
	wcefb	%v0, %v0, 16, 0

#CHECK: error: invalid operand
#CHECK: wcelfb	%v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: wcelfb	%v0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: wcelfb	%v0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: wcelfb	%v0, %v0, 16, 0

	wcelfb	%v0, %v0, 0, -1
	wcelfb	%v0, %v0, 0, 16
	wcelfb	%v0, %v0, -1, 0
	wcelfb	%v0, %v0, 16, 0

#CHECK: error: invalid operand
#CHECK: wcfeb	%v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: wcfeb	%v0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: wcfeb	%v0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: wcfeb	%v0, %v0, 16, 0

	wcfeb	%v0, %v0, 0, -1
	wcfeb	%v0, %v0, 0, 16
	wcfeb	%v0, %v0, -1, 0
	wcfeb	%v0, %v0, 16, 0

#CHECK: error: invalid operand
#CHECK: wclfeb	%v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: wclfeb	%v0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: wclfeb	%v0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: wclfeb	%v0, %v0, 16, 0

	wclfeb	%v0, %v0, 0, -1
	wclfeb	%v0, %v0, 0, 16
	wclfeb	%v0, %v0, -1, 0
	wclfeb	%v0, %v0, 16, 0

