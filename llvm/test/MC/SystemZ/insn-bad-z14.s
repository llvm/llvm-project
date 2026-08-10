# For z14 only.
# RUN: not llvm-mc -triple s390x-linux-gnu -mcpu=z14 < %s 2> %t
# RUN: FileCheck < %t %s
# RUN: not llvm-mc -triple s390x-linux-gnu -mcpu=arch12 < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: bi	-524289
#CHECK: error: invalid operand
#CHECK: bi	524288

	bi	-524289
	bi	524288

#CHECK: error: invalid operand
#CHECK: bic	-1, 0(%r1)
#CHECK: error: invalid operand
#CHECK: bic	16, 0(%r1)
#CHECK: error: invalid operand
#CHECK: bic	0, -524289
#CHECK: error: invalid operand
#CHECK: bic	0, 524288

	bic	-1, 0(%r1)
	bic	16, 0(%r1)
	bic	0, -524289
	bic	0, 524288

#CHECK: error: invalid operand
#CHECK: agh	%r0, -524289
#CHECK: error: invalid operand
#CHECK: agh	%r0, 524288

	agh	%r0, -524289
	agh	%r0, 524288

#CHECK: error: instruction requires: deflate-conversion
#CHECK: dfltcc	%r2, %r4, %r6

	dfltcc	%r2, %r4, %r6

#CHECK: error: instruction requires: message-security-assist-extension9
#CHECK: kdsa	%r0, %r2

	kdsa	%r0, %r2

#CHECK: error: invalid register pair
#CHECK: kma	%r1, %r2, %r4
#CHECK: error: invalid register pair
#CHECK: kma	%r2, %r1, %r4
#CHECK: error: invalid register pair
#CHECK: kma	%r2, %r4, %r1

	kma	%r1, %r2, %r4
	kma	%r2, %r1, %r4
	kma	%r2, %r4, %r1

#CHECK: error: invalid operand
#CHECK: lgg	%r0, -524289
#CHECK: error: invalid operand
#CHECK: lgg	%r0, 524288

	lgg	%r0, -524289
	lgg	%r0, 524288

#CHECK: error: invalid operand
#CHECK: lgsc	%r0, -524289
#CHECK: error: invalid operand
#CHECK: lgsc	%r0, 524288

	lgsc	%r0, -524289
	lgsc	%r0, 524288

#CHECK: error: invalid operand
#CHECK: llgfsg	%r0, -524289
#CHECK: error: invalid operand
#CHECK: llgfsg	%r0, 524288

	llgfsg	%r0, -524289
	llgfsg	%r0, 524288

#CHECK: error: invalid operand
#CHECK: mg	%r0, -524289
#CHECK: error: invalid operand
#CHECK: mg	%r0, 524288
#CHECK: error: invalid register pair
#CHECK: mg	%r1, 0

	mg	%r0, -524289
	mg	%r0, 524288
	mg	%r1, 0

#CHECK: error: invalid operand
#CHECK: mgh	%r0, -524289
#CHECK: error: invalid operand
#CHECK: mgh	%r0, 524288

	mgh	%r0, -524289
	mgh	%r0, 524288

#CHECK: error: invalid register pair
#CHECK: mgrk	%r1, %r0, %r0

	mgrk	%r1, %r0, %r0

#CHECK: error: invalid operand
#CHECK: msc	%r0, -524289
#CHECK: error: invalid operand
#CHECK: msc	%r0, 524288

	msc	%r0, -524289
	msc	%r0, 524288

#CHECK: error: invalid operand
#CHECK: msgc	%r0, -524289
#CHECK: error: invalid operand
#CHECK: msgc	%r0, 524288

	msgc	%r0, -524289
	msgc	%r0, 524288

#CHECK: error: instruction requires: miscellaneous-extensions-3
#CHECK: mvcrl	0, 0

	mvcrl	0, 0

#CHECK: error: instruction requires: miscellaneous-extensions-3
#CHECK: ncgrk	%r0, %r0, %r0

	ncgrk	%r0, %r0, %r0

#CHECK: error: instruction requires: miscellaneous-extensions-3
#CHECK: ncrk	%r0, %r0, %r0

	ncrk	%r0, %r0, %r0

#CHECK: error: instruction requires: miscellaneous-extensions-3
#CHECK: nngrk	%r0, %r0, %r0

	nngrk	%r0, %r0, %r0

#CHECK: error: instruction requires: miscellaneous-extensions-3
#CHECK: nnrk	%r0, %r0, %r0

	nnrk	%r0, %r0, %r0

#CHECK: error: instruction requires: miscellaneous-extensions-3
#CHECK: nogrk	%r0, %r0, %r0

	nogrk	%r0, %r0, %r0

#CHECK: error: instruction requires: miscellaneous-extensions-3
#CHECK: nork	%r0, %r0, %r0

	nork	%r0, %r0, %r0

#CHECK: error: instruction requires: miscellaneous-extensions-3
#CHECK: nxgrk	%r0, %r0, %r0

	nxgrk	%r0, %r0, %r0

#CHECK: error: instruction requires: miscellaneous-extensions-3
#CHECK: nxrk	%r0, %r0, %r0

	nxrk	%r0, %r0, %r0

#CHECK: error: instruction requires: miscellaneous-extensions-3
#CHECK: ocgrk	%r0, %r0, %r0

	ocgrk	%r0, %r0, %r0

#CHECK: error: instruction requires: miscellaneous-extensions-3
#CHECK: ocrk	%r0, %r0, %r0

	ocrk	%r0, %r0, %r0

#CHECK: error: instruction requires: miscellaneous-extensions-3
#CHECK: popcnt	%r2, %r4, 1

	popcnt	%r2, %r4, 1

#CHECK: error: invalid register pair
#CHECK: prno	%r1, %r2
#CHECK: error: invalid register pair
#CHECK: prno	%r2, %r1

	prno	%r1, %r2
	prno	%r2, %r1

#CHECK: error: instruction requires: miscellaneous-extensions-3
#CHECK: selgr	%r0, %r0, %r0, 0
#CHECK: error: instruction requires: miscellaneous-extensions-3
#CHECK: selgre	%r0, %r0, %r0

	selgr	%r0, %r0, %r0, 0
	selgre	%r0, %r0, %r0

#CHECK: error: instruction requires: miscellaneous-extensions-3
#CHECK: selfhr  %r0, %r0, %r0, 0
#CHECK: error: instruction requires: miscellaneous-extensions-3
#CHECK: selfhre  %r0, %r0, %r0

	selfhr  %r0, %r0, %r0, 0
	selfhre %r0, %r0, %r0

#CHECK: error: instruction requires: miscellaneous-extensions-3
#CHECK: selr	%r0, %r0, %r0, 0
#CHECK: error: instruction requires: miscellaneous-extensions-3
#CHECK: selre	%r0, %r0, %r0

	selr	%r0, %r0, %r0, 0
	selre	%r0, %r0, %r0

#CHECK: error: invalid operand
#CHECK: sgh	%r0, -524289
#CHECK: error: invalid operand
#CHECK: sgh	%r0, 524288

	sgh	%r0, -524289
	sgh	%r0, 524288

#CHECK: error: instruction requires: enhanced-sort
#CHECK: sortl	%r2, %r4

	sortl	%r2, %r4

#CHECK: error: invalid operand
#CHECK: stgsc	%r0, -524289
#CHECK: error: invalid operand
#CHECK: stgsc	%r0, 524288

	stgsc	%r0, -524289
	stgsc	%r0, 524288

#CHECK: error: invalid operand
#CHECK: vap	%v0, %v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vap	%v0, %v0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: vap	%v0, %v0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vap	%v0, %v0, %v0, 256, 0

	vap	%v0, %v0, %v0, 0, -1
	vap	%v0, %v0, %v0, 0, 16
	vap	%v0, %v0, %v0, -1, 0
	vap	%v0, %v0, %v0, 256, 0

#CHECK: error: instruction requires: vector-enhancements-2
#CHECK: vcefb	%v0, %v0, 0, 0

	vcefb	%v0, %v0, 0, 0

#CHECK: error: instruction requires: vector-enhancements-2
#CHECK: vcelfb	%v0, %v0, 0, 0

	vcelfb	%v0, %v0, 0, 0

#CHECK: error: instruction requires: vector-enhancements-2
#CHECK: vcfeb	%v0, %v0, 0, 0

	vcfeb	%v0, %v0, 0, 0

#CHECK: error: instruction requires: vector-enhancements-2
#CHECK: vcfpl	%v0, %v0, 0, 0, 0

	vcfpl	%v0, %v0, 0, 0, 0

#CHECK: error: instruction requires: vector-enhancements-2
#CHECK: vcfps	%v0, %v0, 0, 0, 0

	vcfps	%v0, %v0, 0, 0, 0

#CHECK: error: instruction requires: vector-enhancements-2
#CHECK: vclfeb	%v0, %v0, 0, 0

	vclfeb	%v0, %v0, 0, 0

#CHECK: error: instruction requires: vector-enhancements-2
#CHECK: vclfp	%v0, %v0, 0, 0, 0

	vclfp	%v0, %v0, 0, 0, 0

#CHECK: error: invalid operand
#CHECK: vcp	%v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vcp	%v0, %v0, 16

	vcp	%v0, %v0, -1
	vcp	%v0, %v0, 16

#CHECK: error: instruction requires: vector-enhancements-2
#CHECK: vcsfp	%v0, %v0, 0, 0, 0

	vcsfp	%v0, %v0, 0, 0, 0

#CHECK: error: invalid operand
#CHECK: vcvb	%r0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vcvb	%r0, %v0, 16
#CHECK: error: instruction requires: vector-packed-decimal-enhancement
#CHECK: vcvb	%r0, %v0, 0, 1

	vcvb	%r0, %v0, -1
	vcvb	%r0, %v0, 16
	vcvb	%r0, %v0, 0, 1

#CHECK: error: invalid operand
#CHECK: vcvbg	%r0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vcvbg	%r0, %v0, 16
#CHECK: error: instruction requires: vector-packed-decimal-enhancement
#CHECK: vcvbg	%r0, %v0, 0, 1

	vcvbg	%r0, %v0, -1
	vcvbg	%r0, %v0, 16
	vcvbg	%r0, %v0, 0, 1

#CHECK: error: invalid operand
#CHECK: vcvd	%r0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vcvd	%r0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: vcvd	%r0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vcvd	%r0, %v0, 256, 0

	vcvd	%r0, %v0, 0, -1
	vcvd	%r0, %v0, 0, 16
	vcvd	%r0, %v0, -1, 0
	vcvd	%r0, %v0, 256, 0

#CHECK: error: invalid operand
#CHECK: vcvdg	%r0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vcvdg	%r0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: vcvdg	%r0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vcvdg	%r0, %v0, 256, 0

	vcvdg	%r0, %v0, 0, -1
	vcvdg	%r0, %v0, 0, 16
	vcvdg	%r0, %v0, -1, 0
	vcvdg	%r0, %v0, 256, 0

#CHECK: error: invalid operand
#CHECK: vdp	%v0, %v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vdp	%v0, %v0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: vdp	%v0, %v0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vdp	%v0, %v0, %v0, 256, 0

	vdp	%v0, %v0, %v0, 0, -1
	vdp	%v0, %v0, %v0, 0, 16
	vdp	%v0, %v0, %v0, -1, 0
	vdp	%v0, %v0, %v0, 256, 0

#CHECK: error: invalid operand
#CHECK: vfisb	%v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vfisb	%v0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: vfisb	%v0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vfisb	%v0, %v0, 16, 0

	vfisb	%v0, %v0, 0, -1
	vfisb	%v0, %v0, 0, 16
	vfisb	%v0, %v0, -1, 0
	vfisb	%v0, %v0, 16, 0

#CHECK: error: invalid operand
#CHECK: vfll	%v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vfll	%v0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: vfll	%v0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vfll	%v0, %v0, 16, 0

	vfll	%v0, %v0, 0, -1
	vfll	%v0, %v0, 0, 16
	vfll	%v0, %v0, -1, 0
	vfll	%v0, %v0, 16, 0

#CHECK: error: invalid operand
#CHECK: vflr	%v0, %v0, 0, 0, -1
#CHECK: error: invalid operand
#CHECK: vflr	%v0, %v0, 0, 0, 16
#CHECK: error: invalid operand
#CHECK: vflr	%v0, %v0, 0, -1, 0
#CHECK: error: invalid operand
#CHECK: vflr	%v0, %v0, 0, 16, 0
#CHECK: error: invalid operand
#CHECK: vflr	%v0, %v0, -1, 0, 0
#CHECK: error: invalid operand
#CHECK: vflr	%v0, %v0, 16, 0, 0

	vflr	%v0, %v0, 0, 0, -1
	vflr	%v0, %v0, 0, 0, 16
	vflr	%v0, %v0, 0, -1, 0
	vflr	%v0, %v0, 0, 16, 0
	vflr	%v0, %v0, -1, 0, 0
	vflr	%v0, %v0, 16, 0, 0

#CHECK: error: invalid operand
#CHECK: vflrd	%v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vflrd	%v0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: vflrd	%v0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vflrd	%v0, %v0, 16, 0

	vflrd	%v0, %v0, 0, -1
	vflrd	%v0, %v0, 0, 16
	vflrd	%v0, %v0, -1, 0
	vflrd	%v0, %v0, 16, 0

#CHECK: error: invalid operand
#CHECK: vfmax	%v0, %v0, %v0, 0, 0, -1
#CHECK: error: invalid operand
#CHECK: vfmax	%v0, %v0, %v0, 0, 0, 16
#CHECK: error: invalid operand
#CHECK: vfmax	%v0, %v0, %v0, 0, -1, 0
#CHECK: error: invalid operand
#CHECK: vfmax	%v0, %v0, %v0, 0, 16, 0
#CHECK: error: invalid operand
#CHECK: vfmax	%v0, %v0, %v0, -1, 0, 0
#CHECK: error: invalid operand
#CHECK: vfmax	%v0, %v0, %v0, 16, 0, 0

	vfmax	%v0, %v0, %v0, 0, 0, -1
	vfmax	%v0, %v0, %v0, 0, 0, 16
	vfmax	%v0, %v0, %v0, 0, -1, 0
	vfmax	%v0, %v0, %v0, 0, 16, 0
	vfmax	%v0, %v0, %v0, -1, 0, 0
	vfmax	%v0, %v0, %v0, 16, 0, 0

#CHECK: error: invalid operand
#CHECK: vfmaxdb	%v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vfmaxdb	%v0, %v0, %v0, 16

	vfmaxdb	%v0, %v0, %v0, -1
	vfmaxdb	%v0, %v0, %v0, 16

#CHECK: error: invalid operand
#CHECK: vfmaxsb	%v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vfmaxsb	%v0, %v0, %v0, 16

	vfmaxsb	%v0, %v0, %v0, -1
	vfmaxsb	%v0, %v0, %v0, 16

#CHECK: error: invalid operand
#CHECK: vfmin	%v0, %v0, %v0, 0, 0, -1
#CHECK: error: invalid operand
#CHECK: vfmin	%v0, %v0, %v0, 0, 0, 16
#CHECK: error: invalid operand
#CHECK: vfmin	%v0, %v0, %v0, 0, -1, 0
#CHECK: error: invalid operand
#CHECK: vfmin	%v0, %v0, %v0, 0, 16, 0
#CHECK: error: invalid operand
#CHECK: vfmin	%v0, %v0, %v0, -1, 0, 0
#CHECK: error: invalid operand
#CHECK: vfmin	%v0, %v0, %v0, 16, 0, 0

	vfmin	%v0, %v0, %v0, 0, 0, -1
	vfmin	%v0, %v0, %v0, 0, 0, 16
	vfmin	%v0, %v0, %v0, 0, -1, 0
	vfmin	%v0, %v0, %v0, 0, 16, 0
	vfmin	%v0, %v0, %v0, -1, 0, 0
	vfmin	%v0, %v0, %v0, 16, 0, 0

#CHECK: error: invalid operand
#CHECK: vfmindb	%v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vfmindb	%v0, %v0, %v0, 16

	vfmindb	%v0, %v0, %v0, -1
	vfmindb	%v0, %v0, %v0, 16

#CHECK: error: invalid operand
#CHECK: vfminsb	%v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vfminsb	%v0, %v0, %v0, 16

	vfminsb	%v0, %v0, %v0, -1
	vfminsb	%v0, %v0, %v0, 16

#CHECK: error: invalid operand
#CHECK: vfnma	%v0, %v0, %v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vfnma	%v0, %v0, %v0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: vfnma	%v0, %v0, %v0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vfnma	%v0, %v0, %v0, %v0, 16, 0

	vfnma	%v0, %v0, %v0, %v0, 0, -1
	vfnma	%v0, %v0, %v0, %v0, 0, 16
	vfnma	%v0, %v0, %v0, %v0, -1, 0
	vfnma	%v0, %v0, %v0, %v0, 16, 0

#CHECK: error: invalid operand
#CHECK: vfnms	%v0, %v0, %v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vfnms	%v0, %v0, %v0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: vfnms	%v0, %v0, %v0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vfnms	%v0, %v0, %v0, %v0, 16, 0

	vfnms	%v0, %v0, %v0, %v0, 0, -1
	vfnms	%v0, %v0, %v0, %v0, 0, 16
	vfnms	%v0, %v0, %v0, %v0, -1, 0
	vfnms	%v0, %v0, %v0, %v0, 16, 0

#CHECK: error: invalid operand
#CHECK: vftcisb	%v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vftcisb	%v0, %v0, 4096

	vftcisb	%v0, %v0, -1
	vftcisb	%v0, %v0, 4096

#CHECK: error: invalid operand
#CHECK: vlip	%v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vlip	%v0, 0, 16
#CHECK: error: invalid operand
#CHECK: vlip	%v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vlip	%v0, 65536, 0

	vlip	%v0, 0, -1
	vlip	%v0, 0, 16
	vlip	%v0, -1, 0
	vlip	%v0, 65536, 0

#CHECK: error: invalid operand
#CHECK: vllezlf	%v0, -1
#CHECK: error: invalid operand
#CHECK: vllezlf	%v0, 4096
#CHECK: error: invalid use of vector addressing
#CHECK: vllezlf	%v0, 0(%v1,%r2)

	vllezlf	%v0, -1
	vllezlf	%v0, 4096
	vllezlf	%v0, 0(%v1,%r2)

#CHECK: error: instruction requires: vector-enhancements-2
#CHECK: vlbr	%v0, 0, 0
#CHECK: error: instruction requires: vector-enhancements-2
#CHECK: vlbrf	%v0, 0
#CHECK: error: instruction requires: vector-enhancements-2
#CHECK: vlbrg	%v0, 0
#CHECK: error: instruction requires: vector-enhancements-2
#CHECK: vlbrh	%v0, 0
#CHECK: error: instruction requires: vector-enhancements-2
#CHECK: vlbrq	%v0, 0

	vlbr	%v0, 0, 0
	vlbrf	%v0, 0
	vlbrg	%v0, 0
	vlbrh	%v0, 0
	vlbrq	%v0, 0

#CHECK: error: instruction requires: vector-enhancements-2
#CHECK: vlbrrep  %v0, 0, 0
#CHECK: error: instruction requires: vector-enhancements-2
#CHECK: vlbrrepf %v0, 0
#CHECK: error: instruction requires: vector-enhancements-2
#CHECK: vlbrrepg %v0, 0
#CHECK: error: instruction requires: vector-enhancements-2
#CHECK: vlbrreph %v0, 0

	vlbrrep  %v0, 0, 0
	vlbrrepf %v0, 0
	vlbrrepg %v0, 0
	vlbrreph %v0, 0

#CHECK: error: instruction requires: vector-enhancements-2
#CHECK: vlebrf	%v0, 0, 0
#CHECK: error: instruction requires: vector-enhancements-2
#CHECK: vlebrg	%v0, 0, 0
#CHECK: error: instruction requires: vector-enhancements-2
#CHECK: vlebrh	%v0, 0, 0

	vlebrf	%v0, 0, 0
	vlebrg	%v0, 0, 0
	vlebrh	%v0, 0, 0

#CHECK: error: instruction requires: vector-enhancements-2
#CHECK: vler	%v0, 0, 0
#CHECK: error: instruction requires: vector-enhancements-2
#CHECK: vlerf	%v0, 0
#CHECK: error: instruction requires: vector-enhancements-2
#CHECK: vlerg	%v0, 0
#CHECK: error: instruction requires: vector-enhancements-2
#CHECK: vlerh	%v0, 0

	vler	%v0, 0, 0
	vlerf	%v0, 0
	vlerg	%v0, 0
	vlerh	%v0, 0

#CHECK: error: instruction requires: vector-enhancements-2
#CHECK: vllebrz  %v0, 0, 0
#CHECK: error: instruction requires: vector-enhancements-2
#CHECK: vllebrze %v0, 0
#CHECK: error: instruction requires: vector-enhancements-2
#CHECK: vllebrzf %v0, 0
#CHECK: error: instruction requires: vector-enhancements-2
#CHECK: vllebrzg %v0, 0
#CHECK: error: instruction requires: vector-enhancements-2
#CHECK: vllebrzh %v0, 0

	vllebrz  %v0, 0, 0
	vllebrze %v0, 0
	vllebrzf %v0, 0
	vllebrzg %v0, 0
	vllebrzh %v0, 0

#CHECK: error: invalid operand
#CHECK: vlrl	%v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vlrl	%v0, 0, 256
#CHECK: error: invalid operand
#CHECK: vlrl	%v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vlrl	%v0, 4096, 0

	vlrl	%v0, 0, -1
	vlrl	%v0, 0, 256
	vlrl	%v0, -1, 0
	vlrl	%v0, 4096, 0

#CHECK: error: invalid operand
#CHECK: vlrlr	%v0, %r0, -1
#CHECK: error: invalid operand
#CHECK: vlrlr	%v0, %r0, 4096

	vlrlr	%v0, %r0, -1
	vlrlr	%v0, %r0, 4096

#CHECK: error: invalid operand
#CHECK: vmp	%v0, %v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vmp	%v0, %v0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: vmp	%v0, %v0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vmp	%v0, %v0, %v0, 256, 0

	vmp	%v0, %v0, %v0, 0, -1
	vmp	%v0, %v0, %v0, 0, 16
	vmp	%v0, %v0, %v0, -1, 0
	vmp	%v0, %v0, %v0, 256, 0

#CHECK: error: invalid operand
#CHECK: vmsp	%v0, %v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vmsp	%v0, %v0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: vmsp	%v0, %v0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vmsp	%v0, %v0, %v0, 256, 0

	vmsp	%v0, %v0, %v0, 0, -1
	vmsp	%v0, %v0, %v0, 0, 16
	vmsp	%v0, %v0, %v0, -1, 0
	vmsp	%v0, %v0, %v0, 256, 0

#CHECK: error: invalid operand
#CHECK: vmsl	%v0, %v0, %v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vmsl	%v0, %v0, %v0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: vmsl	%v0, %v0, %v0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vmsl	%v0, %v0, %v0, %v0, 16, 0

	vmsl	%v0, %v0, %v0, %v0, 0, -1
	vmsl	%v0, %v0, %v0, %v0, 0, 16
	vmsl	%v0, %v0, %v0, %v0, -1, 0
	vmsl	%v0, %v0, %v0, %v0, 16, 0

#CHECK: error: invalid operand
#CHECK: vmslg	%v0, %v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: vmslg	%v0, %v0, %v0, %v0, 16

	vmslg	%v0, %v0, %v0, %v0, -1
	vmslg	%v0, %v0, %v0, %v0, 16

#CHECK: error: invalid operand
#CHECK: vpkz	%v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vpkz	%v0, 0, 256
#CHECK: error: invalid operand
#CHECK: vpkz	%v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vpkz	%v0, 4096, 0

	vpkz	%v0, 0, -1
	vpkz	%v0, 0, 256
	vpkz	%v0, -1, 0
	vpkz	%v0, 4096, 0

#CHECK: error: invalid operand
#CHECK: vpsop	%v0, %v0, 0, 0, -1
#CHECK: error: invalid operand
#CHECK: vpsop	%v0, %v0, 0, 0, 16
#CHECK: error: invalid operand
#CHECK: vpsop	%v0, %v0, 0, -1, 0
#CHECK: error: invalid operand
#CHECK: vpsop	%v0, %v0, 0, 256, 0
#CHECK: error: invalid operand
#CHECK: vpsop	%v0, %v0, -1, 0, 0
#CHECK: error: invalid operand
#CHECK: vpsop	%v0, %v0, 256, 0, 0

	vpsop	%v0, %v0, 0, 0, -1
	vpsop	%v0, %v0, 0, 0, 16
	vpsop	%v0, %v0, 0, -1, 0
	vpsop	%v0, %v0, 0, 256, 0
	vpsop	%v0, %v0, -1, 0, 0
	vpsop	%v0, %v0, 256, 0, 0

#CHECK: error: invalid operand
#CHECK: vrp	%v0, %v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vrp	%v0, %v0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: vrp	%v0, %v0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vrp	%v0, %v0, %v0, 256, 0

	vrp	%v0, %v0, %v0, 0, -1
	vrp	%v0, %v0, %v0, 0, 16
	vrp	%v0, %v0, %v0, -1, 0
	vrp	%v0, %v0, %v0, 256, 0

#CHECK: error: invalid operand
#CHECK: vsdp	%v0, %v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vsdp	%v0, %v0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: vsdp	%v0, %v0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vsdp	%v0, %v0, %v0, 256, 0

	vsdp	%v0, %v0, %v0, 0, -1
	vsdp	%v0, %v0, %v0, 0, 16
	vsdp	%v0, %v0, %v0, -1, 0
	vsdp	%v0, %v0, %v0, 256, 0

#CHECK: error: instruction requires: vector-enhancements-2
#CHECK: vsld	%v0, %v0, %v0, 0

	vsld	%v0, %v0, %v0, 0

#CHECK: error: invalid operand
#CHECK: vsp	%v0, %v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vsp	%v0, %v0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: vsp	%v0, %v0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vsp	%v0, %v0, %v0, 256, 0

	vsp	%v0, %v0, %v0, 0, -1
	vsp	%v0, %v0, %v0, 0, 16
	vsp	%v0, %v0, %v0, -1, 0
	vsp	%v0, %v0, %v0, 256, 0

#CHECK: error: instruction requires: vector-enhancements-2
#CHECK: vsrd	%v0, %v0, %v0, 0

	vsrd	%v0, %v0, %v0, 0

#CHECK: error: invalid operand
#CHECK: vsrp	%v0, %v0, 0, 0, -1
#CHECK: error: invalid operand
#CHECK: vsrp	%v0, %v0, 0, 0, 16
#CHECK: error: invalid operand
#CHECK: vsrp	%v0, %v0, 0, -1, 0
#CHECK: error: invalid operand
#CHECK: vsrp	%v0, %v0, 0, 256, 0
#CHECK: error: invalid operand
#CHECK: vsrp	%v0, %v0, -1, 0, 0
#CHECK: error: invalid operand
#CHECK: vsrp	%v0, %v0, 256, 0, 0

	vsrp	%v0, %v0, 0, 0, -1
	vsrp	%v0, %v0, 0, 0, 16
	vsrp	%v0, %v0, 0, -1, 0
	vsrp	%v0, %v0, 0, 256, 0
	vsrp	%v0, %v0, -1, 0, 0
	vsrp	%v0, %v0, 256, 0, 0

#CHECK: error: instruction requires: vector-enhancements-2
#CHECK: vstbr	%v0, 0, 0
#CHECK: error: instruction requires: vector-enhancements-2
#CHECK: vstbrf	%v0, 0
#CHECK: error: instruction requires: vector-enhancements-2
#CHECK: vstbrg	%v0, 0
#CHECK: error: instruction requires: vector-enhancements-2
#CHECK: vstbrh	%v0, 0
#CHECK: error: instruction requires: vector-enhancements-2
#CHECK: vstbrq	%v0, 0

	vstbr	%v0, 0, 0
	vstbrf	%v0, 0
	vstbrg	%v0, 0
	vstbrh	%v0, 0
	vstbrq	%v0, 0

#CHECK: error: instruction requires: vector-enhancements-2
#CHECK: vstebrf	%v0, 0, 0
#CHECK: error: instruction requires: vector-enhancements-2
#CHECK: vstebrg	%v0, 0, 0
#CHECK: error: instruction requires: vector-enhancements-2
#CHECK: vstebrh	%v0, 0, 0

	vstebrf	%v0, 0, 0
	vstebrg	%v0, 0, 0
	vstebrh	%v0, 0, 0

#CHECK: error: instruction requires: vector-enhancements-2
#CHECK: vster	%v0, 0, 0
#CHECK: error: instruction requires: vector-enhancements-2
#CHECK: vsterf	%v0, 0
#CHECK: error: instruction requires: vector-enhancements-2
#CHECK: vsterg	%v0, 0
#CHECK: error: instruction requires: vector-enhancements-2
#CHECK: vsterh	%v0, 0

	vster	%v0, 0, 0
	vsterf	%v0, 0
	vsterg	%v0, 0
	vsterh	%v0, 0

#CHECK: error: invalid operand
#CHECK: vstrl	%v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vstrl	%v0, 0, 256
#CHECK: error: invalid operand
#CHECK: vstrl	%v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vstrl	%v0, 4096, 0

	vstrl	%v0, 0, -1
	vstrl	%v0, 0, 256
	vstrl	%v0, -1, 0
	vstrl	%v0, 4096, 0

#CHECK: error: invalid operand
#CHECK: vstrlr	%v0, %r0, -1
#CHECK: error: invalid operand
#CHECK: vstrlr	%v0, %r0, 4096

	vstrlr	%v0, %r0, -1
	vstrlr	%v0, %r0, 4096

#CHECK: error: instruction requires: vector-enhancements-2
#CHECK: vstrs	%v0, %v0, %v0, %v0, 0
#CHECK: error: instruction requires: vector-enhancements-2
#CHECK: vstrsb  %v0, %v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-2
#CHECK: vstrsf  %v0, %v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-2
#CHECK: vstrsh  %v0, %v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-2
#CHECK: vstrszb %v0, %v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-2
#CHECK: vstrszf %v0, %v0, %v0, %v0
#CHECK: error: instruction requires: vector-enhancements-2
#CHECK: vstrszh %v0, %v0, %v0, %v0

        vstrs	%v0, %v0, %v0, %v0, 0
        vstrsb  %v0, %v0, %v0, %v0
        vstrsf  %v0, %v0, %v0, %v0
        vstrsh  %v0, %v0, %v0, %v0
        vstrszb %v0, %v0, %v0, %v0
        vstrszf %v0, %v0, %v0, %v0
        vstrszh %v0, %v0, %v0, %v0

#CHECK: error: invalid operand
#CHECK: vupkz	%v0, 0, -1
#CHECK: error: invalid operand
#CHECK: vupkz	%v0, 0, 256
#CHECK: error: invalid operand
#CHECK: vupkz	%v0, -1, 0
#CHECK: error: invalid operand
#CHECK: vupkz	%v0, 4096, 0

	vupkz	%v0, 0, -1
	vupkz	%v0, 0, 256
	vupkz	%v0, -1, 0
	vupkz	%v0, 4096, 0

#CHECK: error: instruction requires: vector-enhancements-2
#CHECK: wcefb	%v0, %v0, 0, 0

	wcefb	%v0, %v0, 0, 0

#CHECK: error: instruction requires: vector-enhancements-2
#CHECK: wcelfb	%v0, %v0, 0, 0

	wcelfb	%v0, %v0, 0, 0

#CHECK: error: instruction requires: vector-enhancements-2
#CHECK: wcfeb	%v0, %v0, 0, 0

	wcfeb	%v0, %v0, 0, 0

#CHECK: error: instruction requires: vector-enhancements-2
#CHECK: wclfeb	%v0, %v0, 0, 0

	wclfeb	%v0, %v0, 0, 0

#CHECK: error: invalid operand
#CHECK: wfisb	%v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: wfisb	%v0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: wfisb	%v0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: wfisb	%v0, %v0, 16, 0

	wfisb	%v0, %v0, 0, -1
	wfisb	%v0, %v0, 0, 16
	wfisb	%v0, %v0, -1, 0
	wfisb	%v0, %v0, 16, 0

#CHECK: error: invalid operand
#CHECK: wfixb	%v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: wfixb	%v0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: wfixb	%v0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: wfixb	%v0, %v0, 16, 0

	wfixb	%v0, %v0, 0, -1
	wfixb	%v0, %v0, 0, 16
	wfixb	%v0, %v0, -1, 0
	wfixb	%v0, %v0, 16, 0

#CHECK: error: invalid operand
#CHECK: wflrd	%v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: wflrd	%v0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: wflrd	%v0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: wflrd	%v0, %v0, 16, 0

	wflrd	%v0, %v0, 0, -1
	wflrd	%v0, %v0, 0, 16
	wflrd	%v0, %v0, -1, 0
	wflrd	%v0, %v0, 16, 0

#CHECK: error: invalid operand
#CHECK: wflrx	%v0, %v0, 0, -1
#CHECK: error: invalid operand
#CHECK: wflrx	%v0, %v0, 0, 16
#CHECK: error: invalid operand
#CHECK: wflrx	%v0, %v0, -1, 0
#CHECK: error: invalid operand
#CHECK: wflrx	%v0, %v0, 16, 0

	wflrx	%v0, %v0, 0, -1
	wflrx	%v0, %v0, 0, 16
	wflrx	%v0, %v0, -1, 0
	wflrx	%v0, %v0, 16, 0

#CHECK: error: invalid operand
#CHECK: wfmaxdb	%v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: wfmaxdb	%v0, %v0, %v0, 16

	wfmaxdb	%v0, %v0, %v0, -1
	wfmaxdb	%v0, %v0, %v0, 16

#CHECK: error: invalid operand
#CHECK: wfmaxsb	%v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: wfmaxsb	%v0, %v0, %v0, 16

	wfmaxsb	%v0, %v0, %v0, -1
	wfmaxsb	%v0, %v0, %v0, 16

#CHECK: error: invalid operand
#CHECK: wfmaxxb	%v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: wfmaxxb	%v0, %v0, %v0, 16

	wfmaxxb	%v0, %v0, %v0, -1
	wfmaxxb	%v0, %v0, %v0, 16

#CHECK: error: invalid operand
#CHECK: wfmindb	%v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: wfmindb	%v0, %v0, %v0, 16

	wfmindb	%v0, %v0, %v0, -1
	wfmindb	%v0, %v0, %v0, 16

#CHECK: error: invalid operand
#CHECK: wfminsb	%v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: wfminsb	%v0, %v0, %v0, 16

	wfminsb	%v0, %v0, %v0, -1
	wfminsb	%v0, %v0, %v0, 16

#CHECK: error: invalid operand
#CHECK: wfminxb	%v0, %v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: wfminxb	%v0, %v0, %v0, 16

	wfminxb	%v0, %v0, %v0, -1
	wfminxb	%v0, %v0, %v0, 16

#CHECK: error: invalid operand
#CHECK: wftcisb	%v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: wftcisb	%v0, %v0, 4096

	wftcisb	%v0, %v0, -1
	wftcisb	%v0, %v0, 4096

#CHECK: error: invalid operand
#CHECK: wftcixb	%v0, %v0, -1
#CHECK: error: invalid operand
#CHECK: wftcixb	%v0, %v0, 4096

	wftcixb	%v0, %v0, -1
	wftcixb	%v0, %v0, 4096

