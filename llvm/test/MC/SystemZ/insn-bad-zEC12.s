# For zEC12 only.
# RUN: not llvm-mc -triple s390x-linux-gnu -mcpu=zEC12 < %s 2> %t
# RUN: FileCheck < %t %s
# RUN: not llvm-mc -triple s390x-linux-gnu -mcpu=arch10 < %s 2> %t
# RUN: FileCheck < %t %s

#CHECK: error: invalid operand
#CHECK: bpp	-1, 0, 0
#CHECK: error: invalid operand
#CHECK: bpp	16, 0, 0
#CHECK: error: offset out of range
#CHECK: bpp	0, -0x10002, 0
#CHECK: error: offset out of range
#CHECK: bpp	0, -1, 0
#CHECK: error: offset out of range
#CHECK: bpp	0, 1, 0
#CHECK: error: offset out of range
#CHECK: bpp	0, 0x10000, 0
#CHECK: error: invalid operand
#CHECK: bpp	0, 0, -1
#CHECK: error: invalid operand
#CHECK: bpp	0, 0, 4096

	bpp	-1, 0, 0
	bpp	16, 0, 0
	bpp	0, -0x10002, 0
	bpp	0, -1, 0
	bpp	0, 1, 0
	bpp	0, 0x10000, 0
	bpp	0, 0, -1
	bpp	0, 0, 4096

#CHECK: error: invalid operand
#CHECK:	bprp	-1, 0, 0
#CHECK: error: invalid operand
#CHECK:	bprp	16, 0, 0
#CHECK: error: offset out of range
#CHECK:	bprp	0, -0x1002, 0
#CHECK: error: offset out of range
#CHECK:	bprp	0, -1, 0
#CHECK: error: offset out of range
#CHECK:	bprp	0, 1, 0
#CHECK: error: offset out of range
#CHECK:	bprp	0, 0x1000, 0
#CHECK: error: offset out of range
#CHECK:	bprp	0, 0, -0x1000002
#CHECK: error: offset out of range
#CHECK:	bprp	0, 0, -1
#CHECK: error: offset out of range
#CHECK:	bprp	0, 0, 1
#CHECK: error: offset out of range
#CHECK:	bprp	0, 0, 0x1000000

	bprp	-1, 0, 0
	bprp	16, 0, 0
	bprp	0, -0x1002, 0
	bprp	0, -1, 0
	bprp	0, 1, 0
	bprp	0, 0x1000, 0
	bprp	0, 0, -0x1000002
	bprp	0, 0, -1
	bprp	0, 0, 1
	bprp	0, 0, 0x1000000

#CHECK: error: invalid operand
#CHECK: clgt	%r0, -1, 0
#CHECK: error: invalid operand
#CHECK: clgt	%r0, 16, 0
#CHECK: error: invalid operand
#CHECK: clgt	%r0, 12, -524289
#CHECK: error: invalid operand
#CHECK: clgt	%r0, 12, 524288
#CHECK: error: invalid use of indexed addressing
#CHECK: clgt	%r0, 12, 0(%r1,%r2)

	clgt	%r0, -1, 0
	clgt	%r0, 16, 0
	clgt	%r0, 12, -524289
	clgt	%r0, 12, 524288
	clgt	%r0, 12, 0(%r1,%r2)

#CHECK: error: invalid instruction
#CHECK: clgtno   %r0, 0
#CHECK: error: invalid instruction
#CHECK: clgto    %r0, 0

        clgtno   %r0, 0
        clgto    %r0, 0

#CHECK: error: invalid operand
#CHECK: clt	%r0, -1, 0
#CHECK: error: invalid operand
#CHECK: clt	%r0, 16, 0
#CHECK: error: invalid operand
#CHECK: clt	%r0, 12, -524289
#CHECK: error: invalid operand
#CHECK: clt	%r0, 12, 524288
#CHECK: error: invalid use of indexed addressing
#CHECK: clt	%r0, 12, 0(%r1,%r2)

	clt	%r0, -1, 0
	clt	%r0, 16, 0
	clt	%r0, 12, -524289
	clt	%r0, 12, 524288
	clt	%r0, 12, 0(%r1,%r2)

#CHECK: error: invalid instruction
#CHECK: cltno   %r0, 0
#CHECK: error: invalid instruction
#CHECK: clto    %r0, 0

        cltno   %r0, 0
        clto    %r0, 0

#CHECK: error: invalid operand
#CHECK: lat	%r0, -524289
#CHECK: error: invalid operand
#CHECK: lat	%r0, 524288

	lat	%r0, -524289
	lat	%r0, 524288

#CHECK: error: instruction requires: vector
#CHECK: lcbb	%r0, 0, 0

	lcbb	%r0, 0, 0

#CHECK: error: invalid operand
#CHECK: lfhat	%r0, -524289
#CHECK: error: invalid operand
#CHECK: lfhat	%r0, 524288

	lfhat	%r0, -524289
	lfhat	%r0, 524288

#CHECK: error: invalid operand
#CHECK: lgat	%r0, -524289
#CHECK: error: invalid operand
#CHECK: lgat	%r0, 524288

	lgat	%r0, -524289
	lgat	%r0, 524288

#CHECK: error: invalid operand
#CHECK: llgfat	%r0, -524289
#CHECK: error: invalid operand
#CHECK: llgfat	%r0, 524288

	llgfat	%r0, -524289
	llgfat	%r0, 524288

#CHECK: error: invalid operand
#CHECK: llgtat	%r0, -524289
#CHECK: error: invalid operand
#CHECK: llgtat	%r0, 524288

	llgtat	%r0, -524289
	llgtat	%r0, 524288

#CHECK: error: instruction requires: load-store-on-cond-2
#CHECK: locghio %r11, 42

        locghio %r11, 42

#CHECK: error: instruction requires: load-store-on-cond-2
#CHECK: lochio %r11, 42

        lochio %r11, 42

#CHECK: error: invalid operand
#CHECK:	niai	-1, 0
#CHECK: error: invalid operand
#CHECK:	niai	16, 0
#CHECK: error: invalid operand
#CHECK:	niai	0, -1
#CHECK: error: invalid operand
#CHECK:	niai	0, 16

	niai	-1, 0
	niai	16, 0
	niai	0, -1
	niai	0, 16

#CHECK: error: invalid operand
#CHECK: ntstg	%r0, -524289
#CHECK: error: invalid operand
#CHECK: ntstg	%r0, 524288

	ntstg	%r0, -524289
	ntstg	%r0, 524288

#CHECK: error: invalid operand
#CHECK: ppa	%r0, %r0, -1
#CHECK: error: invalid operand
#CHECK: ppa	%r0, %r0, 16

	ppa	%r0, %r0, -1
	ppa	%r0, %r0, 16

#CHECK: error: instruction requires: message-security-assist-extension5
#CHECK: ppno	%r2, %r4

	ppno	%r2, %r4

#CHECK: error: invalid operand
#CHECK: risbgn	%r0,%r0,0,0,-1
#CHECK: error: invalid operand
#CHECK: risbgn	%r0,%r0,0,0,64
#CHECK: error: invalid operand
#CHECK: risbgn	%r0,%r0,0,-1,0
#CHECK: error: invalid operand
#CHECK: risbgn	%r0,%r0,0,256,0
#CHECK: error: invalid operand
#CHECK: risbgn	%r0,%r0,-1,0,0
#CHECK: error: invalid operand
#CHECK: risbgn	%r0,%r0,256,0,0

	risbgn	%r0,%r0,0,0,-1
	risbgn	%r0,%r0,0,0,64
	risbgn	%r0,%r0,0,-1,0
	risbgn	%r0,%r0,0,256,0
	risbgn	%r0,%r0,-1,0,0
	risbgn	%r0,%r0,256,0,0

#CHECK: error: invalid operand
#CHECK: tabort	-1
#CHECK: error: invalid operand
#CHECK: tabort	4096
#CHECK: error: invalid use of indexed addressing
#CHECK: tabort	0(%r1,%r2)

	tabort	-1
	tabort	4096
	tabort	0(%r1,%r2)

#CHECK: error: invalid operand
#CHECK: tbegin	-1, 0
#CHECK: error: invalid operand
#CHECK: tbegin	4096, 0
#CHECK: error: invalid use of indexed addressing
#CHECK: tbegin	0(%r1,%r2), 0
#CHECK: error: invalid operand
#CHECK: tbegin	0, -1
#CHECK: error: invalid operand
#CHECK: tbegin	0, 65536

	tbegin	-1, 0
	tbegin	4096, 0
	tbegin	0(%r1,%r2), 0
	tbegin	0, -1
	tbegin	0, 65536

#CHECK: error: invalid operand
#CHECK: tbeginc	-1, 0
#CHECK: error: invalid operand
#CHECK: tbeginc	4096, 0
#CHECK: error: invalid use of indexed addressing
#CHECK: tbeginc	0(%r1,%r2), 0
#CHECK: error: invalid operand
#CHECK: tbeginc	0, -1
#CHECK: error: invalid operand
#CHECK: tbeginc	0, 65536

	tbeginc	-1, 0
	tbeginc	4096, 0
	tbeginc	0(%r1,%r2), 0
	tbeginc	0, -1
	tbeginc	0, 65536

#CHECK: error: instruction requires: vector
#CHECK: vab	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vaf	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vag	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vah	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vaq	%v0, %v0, %v0

	vab	%v0, %v0, %v0
	vaf	%v0, %v0, %v0
	vag	%v0, %v0, %v0
	vah	%v0, %v0, %v0
	vaq	%v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vaccb	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vaccf	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vaccg	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vacch	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vaccq	%v0, %v0, %v0

	vaccb	%v0, %v0, %v0
	vaccf	%v0, %v0, %v0
	vaccg	%v0, %v0, %v0
	vacch	%v0, %v0, %v0
	vaccq	%v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vacccq	%v0, %v0, %v0, %v0

	vacccq	%v0, %v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vacq	%v0, %v0, %v0, %v0

	vacq	%v0, %v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vavgb	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vavgf	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vavgg	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vavgh	%v0, %v0, %v0

	vavgb	%v0, %v0, %v0
	vavgf	%v0, %v0, %v0
	vavgg	%v0, %v0, %v0
	vavgh	%v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vavglb	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vavglf	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vavglg	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vavglh	%v0, %v0, %v0

	vavglb	%v0, %v0, %v0
	vavglf	%v0, %v0, %v0
	vavglg	%v0, %v0, %v0
	vavglh	%v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vcdgb	%v0, %v0, 0, 0

	vcdgb	%v0, %v0, 0, 0

#CHECK: error: instruction requires: vector
#CHECK: vcdlgb	%v0, %v0, 0, 0

	vcdlgb	%v0, %v0, 0, 0

#CHECK: error: instruction requires: vector
#CHECK: vceqb	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vceqbs	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vceqf	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vceqfs	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vceqg	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vceqgs	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vceqh	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vceqhs	%v0, %v0, %v0

	vceqb	%v0, %v0, %v0
	vceqbs	%v0, %v0, %v0
	vceqf	%v0, %v0, %v0
	vceqfs	%v0, %v0, %v0
	vceqg	%v0, %v0, %v0
	vceqgs	%v0, %v0, %v0
	vceqh	%v0, %v0, %v0
	vceqhs	%v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vcgdb	%v0, %v0, 0, 0

	vcgdb	%v0, %v0, 0, 0

#CHECK: error: instruction requires: vector
#CHECK: vchb	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vchbs	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vchf	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vchfs	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vchg	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vchgs	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vchh	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vchhs	%v0, %v0, %v0

	vchb	%v0, %v0, %v0
	vchbs	%v0, %v0, %v0
	vchf	%v0, %v0, %v0
	vchfs	%v0, %v0, %v0
	vchg	%v0, %v0, %v0
	vchgs	%v0, %v0, %v0
	vchh	%v0, %v0, %v0
	vchhs	%v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vchlb	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vchlbs	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vchlf	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vchlfs	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vchlg	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vchlgs	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vchlh	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vchlhs	%v0, %v0, %v0

	vchlb	%v0, %v0, %v0
	vchlbs	%v0, %v0, %v0
	vchlf	%v0, %v0, %v0
	vchlfs	%v0, %v0, %v0
	vchlg	%v0, %v0, %v0
	vchlgs	%v0, %v0, %v0
	vchlh	%v0, %v0, %v0
	vchlhs	%v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vcksm	%v0, %v0, %v0

	vcksm	%v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vclgdb	%v0, %v0, 0, 0

	vclgdb	%v0, %v0, 0, 0

#CHECK: error: instruction requires: vector
#CHECK: vclzb	%v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vclzf	%v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vclzg	%v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vclzh	%v0, %v0

	vclzb	%v0, %v0
	vclzf	%v0, %v0
	vclzg	%v0, %v0
	vclzh	%v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vctzb	%v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vctzf	%v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vctzg	%v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vctzh	%v0, %v0

	vctzb	%v0, %v0
	vctzf	%v0, %v0
	vctzg	%v0, %v0
	vctzh	%v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vecb	%v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vecf	%v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vecg	%v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vech	%v0, %v0

	vecb	%v0, %v0
	vecf	%v0, %v0
	vecg	%v0, %v0
	vech	%v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: veclb	%v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: veclf	%v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: veclg	%v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: veclh	%v0, %v0

	veclb	%v0, %v0
	veclf	%v0, %v0
	veclg	%v0, %v0
	veclh	%v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: verimb	%v0, %v0, %v0, 0
#CHECK: error: instruction requires: vector
#CHECK: verimf	%v0, %v0, %v0, 0
#CHECK: error: instruction requires: vector
#CHECK: verimg	%v0, %v0, %v0, 0
#CHECK: error: instruction requires: vector
#CHECK: verimh	%v0, %v0, %v0, 0

	verimb	%v0, %v0, %v0, 0
	verimf	%v0, %v0, %v0, 0
	verimg	%v0, %v0, %v0, 0
	verimh	%v0, %v0, %v0, 0

#CHECK: error: instruction requires: vector
#CHECK: verllb	%v0, %v0, 0
#CHECK: error: instruction requires: vector
#CHECK: verllf	%v0, %v0, 0
#CHECK: error: instruction requires: vector
#CHECK: verllg	%v0, %v0, 0
#CHECK: error: instruction requires: vector
#CHECK: verllh	%v0, %v0, 0

	verllb	%v0, %v0, 0
	verllf	%v0, %v0, 0
	verllg	%v0, %v0, 0
	verllh	%v0, %v0, 0

#CHECK: error: instruction requires: vector
#CHECK: verllvb	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: verllvf	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: verllvg	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: verllvh	%v0, %v0, %v0

	verllvb	%v0, %v0, %v0
	verllvf	%v0, %v0, %v0
	verllvg	%v0, %v0, %v0
	verllvh	%v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: veslb	%v0, %v0, 0
#CHECK: error: instruction requires: vector
#CHECK: veslf	%v0, %v0, 0
#CHECK: error: instruction requires: vector
#CHECK: veslg	%v0, %v0, 0
#CHECK: error: instruction requires: vector
#CHECK: veslh	%v0, %v0, 0

	veslb	%v0, %v0, 0
	veslf	%v0, %v0, 0
	veslg	%v0, %v0, 0
	veslh	%v0, %v0, 0

#CHECK: error: instruction requires: vector
#CHECK: veslvb	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: veslvf	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: veslvg	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: veslvh	%v0, %v0, %v0

	veslvb	%v0, %v0, %v0
	veslvf	%v0, %v0, %v0
	veslvg	%v0, %v0, %v0
	veslvh	%v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vesrab	%v0, %v0, 0
#CHECK: error: instruction requires: vector
#CHECK: vesraf	%v0, %v0, 0
#CHECK: error: instruction requires: vector
#CHECK: vesrag	%v0, %v0, 0
#CHECK: error: instruction requires: vector
#CHECK: vesrah	%v0, %v0, 0

	vesrab	%v0, %v0, 0
	vesraf	%v0, %v0, 0
	vesrag	%v0, %v0, 0
	vesrah	%v0, %v0, 0

#CHECK: error: instruction requires: vector
#CHECK: vesravb	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vesravf	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vesravg	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vesravh	%v0, %v0, %v0

	vesravb	%v0, %v0, %v0
	vesravf	%v0, %v0, %v0
	vesravg	%v0, %v0, %v0
	vesravh	%v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vesrlb	%v0, %v0, 0
#CHECK: error: instruction requires: vector
#CHECK: vesrlf	%v0, %v0, 0
#CHECK: error: instruction requires: vector
#CHECK: vesrlg	%v0, %v0, 0
#CHECK: error: instruction requires: vector
#CHECK: vesrlh	%v0, %v0, 0

	vesrlb	%v0, %v0, 0
	vesrlf	%v0, %v0, 0
	vesrlg	%v0, %v0, 0
	vesrlh	%v0, %v0, 0

#CHECK: error: instruction requires: vector
#CHECK: vesrlvb	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vesrlvf	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vesrlvg	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vesrlvh	%v0, %v0, %v0

	vesrlvb	%v0, %v0, %v0
	vesrlvf	%v0, %v0, %v0
	vesrlvg	%v0, %v0, %v0
	vesrlvh	%v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vfadb	%v0, %v0, %v0

	vfadb	%v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vfaeb	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vfaebs	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vfaef	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vfaefs	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vfaeh	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vfaehs	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vfaezb	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vfaezbs	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vfaezf	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vfaezfs	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vfaezh	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vfaezhs	%v0, %v0, %v0

	vfaeb	%v0, %v0, %v0
	vfaebs	%v0, %v0, %v0
	vfaef	%v0, %v0, %v0
	vfaefs	%v0, %v0, %v0
	vfaeh	%v0, %v0, %v0
	vfaehs	%v0, %v0, %v0
	vfaezb	%v0, %v0, %v0
	vfaezbs	%v0, %v0, %v0
	vfaezf	%v0, %v0, %v0
	vfaezfs	%v0, %v0, %v0
	vfaezh	%v0, %v0, %v0
	vfaezhs	%v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vfcedb	%v0, %v0, %v0
#CHECK: vfcedbs	%v0, %v0, %v0

	vfcedb	%v0, %v0, %v0
	vfcedbs	%v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vfchdb	%v0, %v0, %v0
#CHECK: vfchdbs	%v0, %v0, %v0

	vfchdb	%v0, %v0, %v0
	vfchdbs	%v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vfddb	%v0, %v0, %v0

	vfddb	%v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vfeeb	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vfeebs	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vfeef	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vfeefs	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vfeeh	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vfeehs	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vfeezb	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vfeezbs	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vfeezf	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vfeezfs	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vfeezh	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vfeezhs	%v0, %v0, %v0

	vfeeb	%v0, %v0, %v0
	vfeebs	%v0, %v0, %v0
	vfeef	%v0, %v0, %v0
	vfeefs	%v0, %v0, %v0
	vfeeh	%v0, %v0, %v0
	vfeehs	%v0, %v0, %v0
	vfeezb	%v0, %v0, %v0
	vfeezbs	%v0, %v0, %v0
	vfeezf	%v0, %v0, %v0
	vfeezfs	%v0, %v0, %v0
	vfeezh	%v0, %v0, %v0
	vfeezhs	%v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vfeneb   %v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vfenebs  %v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vfenef   %v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vfenefs  %v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vfeneh   %v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vfenehs  %v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vfenezb  %v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vfenezbs %v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vfenezf  %v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vfenezfs %v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vfenezh  %v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vfenezhs %v0, %v0, %v0

	vfeneb   %v0, %v0, %v0
	vfenebs  %v0, %v0, %v0
	vfenef   %v0, %v0, %v0
	vfenefs  %v0, %v0, %v0
	vfeneh   %v0, %v0, %v0
	vfenehs  %v0, %v0, %v0
	vfenezb  %v0, %v0, %v0
	vfenezbs %v0, %v0, %v0
	vfenezf  %v0, %v0, %v0
	vfenezfs %v0, %v0, %v0
	vfenezh  %v0, %v0, %v0
	vfenezhs %v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vfidb	%v0, %v0, 0, 0

	vfidb	%v0, %v0, 0, 0

#CHECK: error: instruction requires: vector
#CHECK: vflcdb	%v0, %v0

	vflcdb	%v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vflndb	%v0, %v0

	vflndb	%v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vflpdb	%v0, %v0

	vflpdb	%v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vfmadb	%v0, %v0, %v0, %v0

	vfmadb	%v0, %v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vfmdb	%v0, %v0, %v0

	vfmdb	%v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vfmsdb	%v0, %v0, %v0, %v0

	vfmsdb	%v0, %v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vfsdb	%v0, %v0, %v0

	vfsdb	%v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vfsqdb	%v0, %v0

	vfsqdb	%v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vftcidb	%v0, %v0, 0

	vftcidb	%v0, %v0, 0

#CHECK: error: instruction requires: vector
#CHECK: vgbm	%v0, 0

	vgbm	%v0, 0

#CHECK: error: instruction requires: vector
#CHECK: vgef	%v0, 0(%v0, %r1), 0
#CHECK: error: instruction requires: vector
#CHECK: vgeg	%v0, 0(%v0, %r1), 0

	vgef	%v0, 0(%v0, %r1), 0
	vgeg	%v0, 0(%v0, %r1), 0

#CHECK: error: instruction requires: vector
#CHECK: vgfmab	%v0, %v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vgfmaf	%v0, %v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vgfmag	%v0, %v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vgfmah	%v0, %v0, %v0, %v0

	vgfmab	%v0, %v0, %v0, %v0
	vgfmaf	%v0, %v0, %v0, %v0
	vgfmag	%v0, %v0, %v0, %v0
	vgfmah	%v0, %v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vgfmb	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vgfmf	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vgfmg	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vgfmh	%v0, %v0, %v0

	vgfmb	%v0, %v0, %v0
	vgfmf	%v0, %v0, %v0
	vgfmg	%v0, %v0, %v0
	vgfmh	%v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vgmb	%v0, 0, 0
#CHECK: error: instruction requires: vector
#CHECK: vgmf	%v0, 0, 0
#CHECK: error: instruction requires: vector
#CHECK: vgmg	%v0, 0, 0
#CHECK: error: instruction requires: vector
#CHECK: vgmh	%v0, 0, 0

	vgmb	%v0, 0, 0
	vgmf	%v0, 0, 0
	vgmg	%v0, 0, 0
	vgmh	%v0, 0, 0

#CHECK: error: instruction requires: vector
#CHECK: vistrb	%v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vistrbs	%v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vistrf	%v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vistrfs	%v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vistrh	%v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vistrhs	%v0, %v0

	vistrb	%v0, %v0
	vistrbs	%v0, %v0
	vistrf	%v0, %v0
	vistrfs	%v0, %v0
	vistrh	%v0, %v0
	vistrhs	%v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vl	%v0, 0

	vl	%v0, 0

#CHECK: error: instruction requires: vector
#CHECK: vlbb	%v0, 0, 0

	vlbb	%v0, 0, 0

#CHECK: error: instruction requires: vector
#CHECK: vlcb	%v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vlcf	%v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vlcg	%v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vlch	%v0, %v0

	vlcb	%v0, %v0
	vlcf	%v0, %v0
	vlcg	%v0, %v0
	vlch	%v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vldeb	%v0, %v0

	vldeb	%v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vleb	%v0, 0, 0
#CHECK: error: instruction requires: vector
#CHECK: vlef	%v0, 0, 0
#CHECK: error: instruction requires: vector
#CHECK: vleg	%v0, 0, 0
#CHECK: error: instruction requires: vector
#CHECK: vleh	%v0, 0, 0

	vleb	%v0, 0, 0
	vlef	%v0, 0, 0
	vleg	%v0, 0, 0
	vleh	%v0, 0, 0

#CHECK: error: instruction requires: vector
#CHECK: vledb	%v0, %v0, 0, 0

	vledb	%v0, %v0, 0, 0

#CHECK: error: instruction requires: vector
#CHECK: vleib	%v0, 0, 0
#CHECK: error: instruction requires: vector
#CHECK: vleif	%v0, 0, 0
#CHECK: error: instruction requires: vector
#CHECK: vleig	%v0, 0, 0
#CHECK: error: instruction requires: vector
#CHECK: vleih	%v0, 0, 0

	vleib	%v0, 0, 0
	vleif	%v0, 0, 0
	vleig	%v0, 0, 0
	vleih	%v0, 0, 0

#CHECK: error: instruction requires: vector
#CHECK: vlgvb	%r0, %v0, 0
#CHECK: error: instruction requires: vector
#CHECK: vlgvf	%r0, %v0, 0
#CHECK: error: instruction requires: vector
#CHECK: vlgvg	%r0, %v0, 0
#CHECK: error: instruction requires: vector
#CHECK: vlgvh	%r0, %v0, 0

	vlgvb	%r0, %v0, 0
	vlgvf	%r0, %v0, 0
	vlgvg	%r0, %v0, 0
	vlgvh	%r0, %v0, 0

#CHECK: error: instruction requires: vector
#CHECK: vll	%v0, %r0, 0

	vll	%v0, %r0, 0

#CHECK: error: instruction requires: vector
#CHECK: vllezb	%v0, 0
#CHECK: error: instruction requires: vector
#CHECK: vllezf	%v0, 0
#CHECK: error: instruction requires: vector
#CHECK: vllezg	%v0, 0
#CHECK: error: instruction requires: vector
#CHECK: vllezh	%v0, 0

	vllezb	%v0, 0
	vllezf	%v0, 0
	vllezg	%v0, 0
	vllezh	%v0, 0

#CHECK: error: instruction requires: vector
#CHECK: vlm	%v0, %v0, 0

	vlm	%v0, %v0, 0

#CHECK: error: instruction requires: vector
#CHECK: vlpb	%v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vlpf	%v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vlpg	%v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vlph	%v0, %v0

	vlpb	%v0, %v0
	vlpf	%v0, %v0
	vlpg	%v0, %v0
	vlph	%v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vlr	%v0, %v0

	vlr	%v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vlrepb	%v0, 0
#CHECK: error: instruction requires: vector
#CHECK: vlrepf	%v0, 0
#CHECK: error: instruction requires: vector
#CHECK: vlrepg	%v0, 0
#CHECK: error: instruction requires: vector
#CHECK: vlreph	%v0, 0

	vlrepb	%v0, 0
	vlrepf	%v0, 0
	vlrepg	%v0, 0
	vlreph	%v0, 0

#CHECK: error: instruction requires: vector
#CHECK: vlvgb	%v0, %r0, 0
#CHECK: error: instruction requires: vector
#CHECK: vlvgf	%v0, %r0, 0
#CHECK: error: instruction requires: vector
#CHECK: vlvgg	%v0, %r0, 0
#CHECK: error: instruction requires: vector
#CHECK: vlvgh	%v0, %r0, 0

	vlvgb	%v0, %r0, 0
	vlvgf	%v0, %r0, 0
	vlvgg	%v0, %r0, 0
	vlvgh	%v0, %r0, 0

#CHECK: error: instruction requires: vector
#CHECK: vlvgp	%v0, %r0, %r0

	vlvgp	%v0, %r0, %r0

#CHECK: error: instruction requires: vector
#CHECK: vmaeb	%v0, %v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vmaef	%v0, %v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vmaeh	%v0, %v0, %v0, %v0

	vmaeb	%v0, %v0, %v0, %v0
	vmaef	%v0, %v0, %v0, %v0
	vmaeh	%v0, %v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vmahb	%v0, %v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vmahf	%v0, %v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vmahh	%v0, %v0, %v0, %v0

	vmahb	%v0, %v0, %v0, %v0
	vmahf	%v0, %v0, %v0, %v0
	vmahh	%v0, %v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vmalb	%v0, %v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vmalf	%v0, %v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vmalhw	%v0, %v0, %v0, %v0

	vmalb	%v0, %v0, %v0, %v0
	vmalf	%v0, %v0, %v0, %v0
	vmalhw	%v0, %v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vmaleb	%v0, %v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vmalef	%v0, %v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vmaleh	%v0, %v0, %v0, %v0

	vmaleb	%v0, %v0, %v0, %v0
	vmalef	%v0, %v0, %v0, %v0
	vmaleh	%v0, %v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vmalhb	%v0, %v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vmalhf	%v0, %v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vmalhh	%v0, %v0, %v0, %v0

	vmalhb	%v0, %v0, %v0, %v0
	vmalhf	%v0, %v0, %v0, %v0
	vmalhh	%v0, %v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vmalob	%v0, %v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vmalof	%v0, %v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vmaloh	%v0, %v0, %v0, %v0

	vmalob	%v0, %v0, %v0, %v0
	vmalof	%v0, %v0, %v0, %v0
	vmaloh	%v0, %v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vmaob	%v0, %v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vmaof	%v0, %v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vmaoh	%v0, %v0, %v0, %v0

	vmaob	%v0, %v0, %v0, %v0
	vmaof	%v0, %v0, %v0, %v0
	vmaoh	%v0, %v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vmeb	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vmef	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vmeh	%v0, %v0, %v0

	vmeb	%v0, %v0, %v0
	vmef	%v0, %v0, %v0
	vmeh	%v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vmhb	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vmhf	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vmhh	%v0, %v0, %v0

	vmhb	%v0, %v0, %v0
	vmhf	%v0, %v0, %v0
	vmhh	%v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vmlb	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vmlf	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vmlhw	%v0, %v0, %v0

	vmlb	%v0, %v0, %v0
	vmlf	%v0, %v0, %v0
	vmlhw	%v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vmleb	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vmlef	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vmleh	%v0, %v0, %v0

	vmleb	%v0, %v0, %v0
	vmlef	%v0, %v0, %v0
	vmleh	%v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vmlhb	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vmlhf	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vmlhh	%v0, %v0, %v0

	vmlhb	%v0, %v0, %v0
	vmlhf	%v0, %v0, %v0
	vmlhh	%v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vmlob	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vmlof	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vmloh	%v0, %v0, %v0

	vmlob	%v0, %v0, %v0
	vmlof	%v0, %v0, %v0
	vmloh	%v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vmnb	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vmnf	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vmng	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vmnh	%v0, %v0, %v0

	vmnb	%v0, %v0, %v0
	vmnf	%v0, %v0, %v0
	vmng	%v0, %v0, %v0
	vmnh	%v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vmnlb	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vmnlf	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vmnlg	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vmnlh	%v0, %v0, %v0

	vmnlb	%v0, %v0, %v0
	vmnlf	%v0, %v0, %v0
	vmnlg	%v0, %v0, %v0
	vmnlh	%v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vmob	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vmof	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vmoh	%v0, %v0, %v0

	vmob	%v0, %v0, %v0
	vmof	%v0, %v0, %v0
	vmoh	%v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vmrhb	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vmrhf	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vmrhg	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vmrhh	%v0, %v0, %v0

	vmrhb	%v0, %v0, %v0
	vmrhf	%v0, %v0, %v0
	vmrhg	%v0, %v0, %v0
	vmrhh	%v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vmrlb	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vmrlf	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vmrlg	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vmrlh	%v0, %v0, %v0

	vmrlb	%v0, %v0, %v0
	vmrlf	%v0, %v0, %v0
	vmrlg	%v0, %v0, %v0
	vmrlh	%v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vmxb	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vmxf	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vmxg	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vmxh	%v0, %v0, %v0

	vmxb	%v0, %v0, %v0
	vmxf	%v0, %v0, %v0
	vmxg	%v0, %v0, %v0
	vmxh	%v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vmxlb	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vmxlf	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vmxlg	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vmxlh	%v0, %v0, %v0

	vmxlb	%v0, %v0, %v0
	vmxlf	%v0, %v0, %v0
	vmxlg	%v0, %v0, %v0
	vmxlh	%v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vn	%v0, %v0, %v0

	vn	%v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vnc	%v0, %v0, %v0

	vnc	%v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vno	%v0, %v0, %v0

	vno	%v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vo	%v0, %v0, %v0

	vo	%v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vone	%v0

	vone	%v0

#CHECK: error: instruction requires: vector
#CHECK: vpdi	%v0, %v0, %v0, 0

	vpdi	%v0, %v0, %v0, 0

#CHECK: error: instruction requires: vector
#CHECK: vperm	%v0, %v0, %v0, %v0

	vperm	%v0, %v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vpkf	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vpkg	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vpkh	%v0, %v0, %v0

	vpkf	%v0, %v0, %v0
	vpkg	%v0, %v0, %v0
	vpkh	%v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vpklsf	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vpklsfs	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vpklsg	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vpklsgs	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vpklsh	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vpklshs	%v0, %v0, %v0

	vpklsf	%v0, %v0, %v0
	vpklsfs	%v0, %v0, %v0
	vpklsg	%v0, %v0, %v0
	vpklsgs	%v0, %v0, %v0
	vpklsh	%v0, %v0, %v0
	vpklshs	%v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vpksf	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vpksfs	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vpksg	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vpksgs	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vpksh	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vpkshs	%v0, %v0, %v0

	vpksf	%v0, %v0, %v0
	vpksfs	%v0, %v0, %v0
	vpksg	%v0, %v0, %v0
	vpksgs	%v0, %v0, %v0
	vpksh	%v0, %v0, %v0
	vpkshs	%v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vpopct	%v0, %v0, 0

	vpopct	%v0, %v0, 0

#CHECK: error: instruction requires: vector
#CHECK: vrepb	%v0, %v0, 0
#CHECK: error: instruction requires: vector
#CHECK: vrepf	%v0, %v0, 0
#CHECK: error: instruction requires: vector
#CHECK: vrepg	%v0, %v0, 0
#CHECK: error: instruction requires: vector
#CHECK: vreph	%v0, %v0, 0

	vrepb	%v0, %v0, 0
	vrepf	%v0, %v0, 0
	vrepg	%v0, %v0, 0
	vreph	%v0, %v0, 0

#CHECK: error: instruction requires: vector
#CHECK: vrepib	%v0, 0
#CHECK: error: instruction requires: vector
#CHECK: vrepif	%v0, 0
#CHECK: error: instruction requires: vector
#CHECK: vrepig	%v0, 0
#CHECK: error: instruction requires: vector
#CHECK: vrepih	%v0, 0

	vrepib	%v0, 0
	vrepif	%v0, 0
	vrepig	%v0, 0
	vrepih	%v0, 0

#CHECK: error: instruction requires: vector
#CHECK: vsb	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vsf	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vsg	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vsh	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vsq	%v0, %v0, %v0

	vsb	%v0, %v0, %v0
	vsf	%v0, %v0, %v0
	vsg	%v0, %v0, %v0
	vsh	%v0, %v0, %v0
	vsq	%v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vsbcbiq	%v0, %v0, %v0, %v0

	vsbcbiq	%v0, %v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vsbiq	%v0, %v0, %v0, %v0

	vsbiq	%v0, %v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vscbib	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vscbif	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vscbig	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vscbih	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vscbiq	%v0, %v0, %v0

	vscbib	%v0, %v0, %v0
	vscbif	%v0, %v0, %v0
	vscbig	%v0, %v0, %v0
	vscbih	%v0, %v0, %v0
	vscbiq	%v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vscef	%v0, 0(%v0, %r1), 0
#CHECK: error: instruction requires: vector
#CHECK: vsceg	%v0, 0(%v0, %r1), 0

	vscef	%v0, 0(%v0, %r1), 0
	vsceg	%v0, 0(%v0, %r1), 0

#CHECK: error: instruction requires: vector
#CHECK: vsegb	%v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vsegf	%v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vsegh	%v0, %v0

	vsegb	%v0, %v0
	vsegf	%v0, %v0
	vsegh	%v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vsel	%v0, %v0, %v0, %v0

	vsel	%v0, %v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vsl	%v0, %v0, %v0

	vsl	%v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vslb	%v0, %v0, %v0

	vslb	%v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vsldb	%v0, %v0, %v0, 0

	vsldb	%v0, %v0, %v0, 0

#CHECK: error: instruction requires: vector
#CHECK: vsra	%v0, %v0, %v0

	vsra	%v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vsrab	%v0, %v0, %v0

	vsrab	%v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vsrl	%v0, %v0, %v0

	vsrl	%v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vsrlb	%v0, %v0, %v0

	vsrlb	%v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vst	%v0, 0

	vst	%v0, 0

#CHECK: error: instruction requires: vector
#CHECK: vstl	%v0, %r0, 0

	vstl	%v0, %r0, 0

#CHECK: error: instruction requires: vector
#CHECK: vstm	%v0, %v0, 0

	vstm	%v0, %v0, 0

#CHECK: error: instruction requires: vector
#CHECK: vstrcb   %v0, %v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vstrcbs  %v0, %v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vstrcf   %v0, %v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vstrcfs  %v0, %v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vstrch   %v0, %v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vstrchs  %v0, %v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vstrczb  %v0, %v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vstrczbs %v0, %v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vstrczf  %v0, %v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vstrczfs %v0, %v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vstrczh  %v0, %v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vstrczhs %v0, %v0, %v0, %v0

        vstrcb   %v0, %v0, %v0, %v0
        vstrcbs  %v0, %v0, %v0, %v0
        vstrcf   %v0, %v0, %v0, %v0
        vstrcfs  %v0, %v0, %v0, %v0
        vstrch   %v0, %v0, %v0, %v0
        vstrchs  %v0, %v0, %v0, %v0
        vstrczb  %v0, %v0, %v0, %v0
        vstrczbs %v0, %v0, %v0, %v0
        vstrczf  %v0, %v0, %v0, %v0
        vstrczfs %v0, %v0, %v0, %v0
        vstrczh  %v0, %v0, %v0, %v0
        vstrczhs %v0, %v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vsumb	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vsumh	%v0, %v0, %v0

	vsumb	%v0, %v0, %v0
	vsumh	%v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vsumgh	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vsumgf	%v0, %v0, %v0

	vsumgh	%v0, %v0, %v0
	vsumgf	%v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vsumqf	%v0, %v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vsumqg	%v0, %v0, %v0

	vsumqf	%v0, %v0, %v0
	vsumqg	%v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vtm	%v0, %v0

	vtm	%v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vuphb	%v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vuphf	%v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vuphh	%v0, %v0

	vuphb	%v0, %v0
	vuphf	%v0, %v0
	vuphh	%v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vuplb	%v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vuplf	%v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vuplhw	%v0, %v0

	vuplb	%v0, %v0
	vuplf	%v0, %v0
	vuplhw	%v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vuplhb	%v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vuplhf	%v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vuplhh	%v0, %v0

	vuplhb	%v0, %v0
	vuplhf	%v0, %v0
	vuplhh	%v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vupllb	%v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vupllf	%v0, %v0
#CHECK: error: instruction requires: vector
#CHECK: vupllh	%v0, %v0

	vupllb	%v0, %v0
	vupllf	%v0, %v0
	vupllh	%v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vx	%v0, %v0, %v0

	vx	%v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: vzero	%v0

	vzero	%v0

#CHECK: error: instruction requires: vector
#CHECK: wcdgb	%v0, %v0, 0, 0

	wcdgb	%v0, %v0, 0, 0

#CHECK: error: instruction requires: vector
#CHECK: wcdlgb	%v0, %v0, 0, 0

	wcdlgb	%v0, %v0, 0, 0

#CHECK: error: instruction requires: vector
#CHECK: wcgdb	%v0, %v0, 0, 0

	wcgdb	%v0, %v0, 0, 0

#CHECK: error: instruction requires: vector
#CHECK: wclgdb	%v0, %v0, 0, 0

	wclgdb	%v0, %v0, 0, 0

#CHECK: error: instruction requires: vector
#CHECK: wfadb	%v0, %v0, %v0

	wfadb	%v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: wfcdb	%v0, %v0

	wfcdb	%v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: wfcedb	%v0, %v0, %v0
#CHECK: wfcedbs	%v0, %v0, %v0

	wfcedb	%v0, %v0, %v0
	wfcedbs	%v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: wfchdb	%v0, %v0, %v0
#CHECK: wfchdbs	%v0, %v0, %v0

	wfchdb	%v0, %v0, %v0
	wfchdbs	%v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: wfchedb	%v0, %v0, %v0
#CHECK: wfchedbs %v0, %v0, %v0

	wfchedb	%v0, %v0, %v0
	wfchedbs %v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: wfddb	%v0, %v0, %v0

	wfddb	%v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: wfidb	%v0, %v0, 0, 0

	wfidb	%v0, %v0, 0, 0

#CHECK: error: instruction requires: vector
#CHECK: wfkdb	%v0, %v0

	wfkdb	%v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: wflcdb	%v0, %v0

	wflcdb	%v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: wflndb	%v0, %v0

	wflndb	%v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: wflpdb	%v0, %v0

	wflpdb	%v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: wfmadb	%v0, %v0, %v0, %v0

	wfmadb	%v0, %v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: wfmdb	%v0, %v0, %v0

	wfmdb	%v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: wfmsdb	%v0, %v0, %v0, %v0

	wfmsdb	%v0, %v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: wfsdb	%v0, %v0, %v0

	wfsdb	%v0, %v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: wfsqdb	%v0, %v0

	wfsqdb	%v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: wftcidb	%v0, %v0, 0

	wftcidb	%v0, %v0, 0

#CHECK: error: instruction requires: vector
#CHECK: wldeb	%v0, %v0

	wldeb	%v0, %v0

#CHECK: error: instruction requires: vector
#CHECK: wledb	%v0, %v0, 0, 0

	wledb	%v0, %v0, 0, 0

