# For zEC12 and above.
# RUN: llvm-mc -triple s390x-linux-gnu -mcpu=zEC12 -show-encoding %s | FileCheck %s
# RUN: llvm-mc -triple s390x-linux-gnu -mcpu=arch10 -show-encoding %s | FileCheck %s

#CHECK: bpp	0, .[[LAB:L.*]]-65536, 0   # encoding: [0xc7,0x00,0x00,0x00,A,A]
#CHECK: fixup A - offset: 4, value: (.[[LAB]]-65536)+4, kind: FK_390_PC16DBL
        bpp	0, -0x10000, 0
#CHECK: bpp     0, .[[LAB:L.*]]-2, 0       # encoding: [0xc7,0x00,0x00,0x00,A,A]
#CHECK: fixup A - offset: 4, value: (.[[LAB]]-2)+4, kind: FK_390_PC16DBL
        bpp	0, -2, 0
#CHECK: bpp	0, .[[LAB:L.*]], 0         # encoding: [0xc7,0x00,0x00,0x00,A,A]
#CHECK: fixup A - offset: 4, value: .[[LAB]]+4, kind: FK_390_PC16DBL
        bpp    0, 0, 0
#CHECK: bpp  	0, .[[LAB:L.*]]+65534, 0   # encoding: [0xc7,0x00,0x00,0x00,A,A]
#CHECK: fixup A - offset: 4, value: (.[[LAB]]+65534)+4, kind: FK_390_PC16DBL
        bpp    	0, 0xfffe, 0

#CHECK: bpp	0, foo, 4095(%r3)          # encoding: [0xc7,0x00,0x3f,0xff,A,A]
#CHECK: fixup A - offset: 4, value: foo+4, kind: FK_390_PC16DBL
#CHECK: bpp	15, foo, 1(%r11)           # encoding: [0xc7,0xf0,0xb0,0x01,A,A]
#CHECK: fixup A - offset: 4, value: foo+4, kind: FK_390_PC16DBL

	bpp	0, foo, 4095(%r3)
	bpp	15, foo, 1(%r11)

#CHECK: bpp	3, bar+100, 4095           # encoding: [0xc7,0x30,0x0f,0xff,A,A]
#CHECK: fixup A - offset: 4, value: (bar+100)+4, kind: FK_390_PC16DBL
#CHECK: bpp	4, bar+100, 1              # encoding: [0xc7,0x40,0x00,0x01,A,A]
#CHECK: fixup A - offset: 4, value: (bar+100)+4, kind: FK_390_PC16DBL

	bpp	3, bar+100, 4095
	bpp	4, bar+100, 1

#CHECK: bpp	7, frob@PLT, 0              # encoding: [0xc7,0x70,0x00,0x00,A,A]
#CHECK: fixup A - offset: 4, value: frob@PLT+4, kind: FK_390_PC16DBL
#CHECK: bpp	8, frob@PLT, 0              # encoding: [0xc7,0x80,0x00,0x00,A,A]
#CHECK: fixup A - offset: 4, value: frob@PLT+4, kind: FK_390_PC16DBL

	bpp	7, frob@PLT, 0
	bpp	8, frob@PLT, 0

#CHECK: bprp   	0, .[[LABA:L.*]]-4096, .[[LABB:L.*]]      # encoding: [0xc5,0b0000AAAA,A,B,B,B]
#CHECK: fixup A - offset: 1, value: (.[[LABA]]-4096)+1, kind: FK_390_PC12DBL
#CHECK: fixup B - offset: 3, value: .[[LABB]]+3, kind: FK_390_PC24DBL
        bprp   	0, -0x1000, 0
#CHECK: bprp   	0, .[[LABA:L.*]]-2, .[[LABB:L.*]]         # encoding: [0xc5,0b0000AAAA,A,B,B,B]
#CHECK: fixup A - offset: 1, value: (.[[LABA]]-2)+1, kind: FK_390_PC12DBL
#CHECK: fixup B - offset: 3, value: .[[LABB]]+3, kind: FK_390_PC24DBL
        bprp   	0, -2, 0
#CHECK: bprp   	0, .[[LABA:L.*]], .[[LABB:L.*]]           # encoding: [0xc5,0b0000AAAA,A,B,B,B]
#CHECK: fixup A - offset: 1, value: .[[LABA]]+1, kind: FK_390_PC12DBL
#CHECK: fixup B - offset: 3, value: .[[LABB]]+3, kind: FK_390_PC24DBL
        bprp   	0, 0, 0
#CHECK: bprp   	0, .[[LABA:L.*]]+4094, .[[LABB:L.*]]      # encoding: [0xc5,0b0000AAAA,A,B,B,B]
#CHECK: fixup A - offset: 1, value: (.[[LABA]]+4094)+1, kind: FK_390_PC12DBL
#CHECK: fixup B - offset: 3, value: .[[LABB]]+3, kind: FK_390_PC24DBL
        bprp   	0, 0xffe, 0
#CHECK: bprp   	15, .[[LABA:L.*]], .[[LABB:L.*]]-16777216 # encoding: [0xc5,0b1111AAAA,A,B,B,B]
#CHECK: fixup A - offset: 1, value: .[[LABA]]+1, kind: FK_390_PC12DBL
#CHECK: fixup B - offset: 3, value: (.[[LABB]]-16777216)+3, kind: FK_390_PC24DBL
        bprp   	15, 0, -0x1000000
#CHECK: bprp   	15, .[[LABA:L.*]], .[[LABB:L.*]]-2        # encoding: [0xc5,0b1111AAAA,A,B,B,B]
#CHECK: fixup A - offset: 1, value: .[[LABA]]+1, kind: FK_390_PC12DBL
#CHECK: fixup B - offset: 3, value: (.[[LABB]]-2)+3, kind: FK_390_PC24DBL
        bprp   	15, 0, -2
#CHECK: bprp   	15, .[[LABA:L.*]], .[[LABB:L.*]]          # encoding: [0xc5,0b1111AAAA,A,B,B,B]
#CHECK: fixup A - offset: 1, value: .[[LABA]]+1, kind: FK_390_PC12DBL
#CHECK: fixup B - offset: 3, value: .[[LABB]]+3, kind: FK_390_PC24DBL
        bprp   	15, 0, 0
#CHECK: bprp   	15, .[[LABA:L.*]], .[[LABB:L.*]]+16777214 # encoding: [0xc5,0b1111AAAA,A,B,B,B]
#CHECK: fixup A - offset: 1, value: .[[LABA]]+1, kind: FK_390_PC12DBL
#CHECK: fixup B - offset: 3, value: (.[[LABB]]+16777214)+3, kind: FK_390_PC24DBL
        bprp   	15, 0, 0xfffffe

#CHECK: bprp	1, branch, target           # encoding: [0xc5,0b0001AAAA,A,B,B,B]
#CHECK: fixup A - offset: 1, value: branch+1, kind: FK_390_PC12DBL
#CHECK: fixup B - offset: 3, value: target+3, kind: FK_390_PC24DBL
#CHECK: bprp	2, branch, target           # encoding: [0xc5,0b0010AAAA,A,B,B,B]
#CHECK: fixup A - offset: 1, value: branch+1, kind: FK_390_PC12DBL
#CHECK: fixup B - offset: 3, value: target+3, kind: FK_390_PC24DBL
#CHECK: bprp	3, branch, target           # encoding: [0xc5,0b0011AAAA,A,B,B,B]
#CHECK: fixup A - offset: 1, value: branch+1, kind: FK_390_PC12DBL
#CHECK: fixup B - offset: 3, value: target+3, kind: FK_390_PC24DBL

	bprp	1, branch, target
	bprp	2, branch, target
	bprp	3, branch, target

#CHECK: bprp	4, branch+100, target       # encoding: [0xc5,0b0100AAAA,A,B,B,B]
#CHECK: fixup A - offset: 1, value: (branch+100)+1, kind: FK_390_PC12DBL
#CHECK: fixup B - offset: 3, value: target+3, kind: FK_390_PC24DBL
#CHECK: bprp	5, branch, target+100       # encoding: [0xc5,0b0101AAAA,A,B,B,B]
#CHECK: fixup A - offset: 1, value: branch+1, kind: FK_390_PC12DBL
#CHECK: fixup B - offset: 3, value: (target+100)+3, kind: FK_390_PC24DBL
#CHECK: bprp	6, branch+100, target+100   # encoding: [0xc5,0b0110AAAA,A,B,B,B]
#CHECK: fixup A - offset: 1, value: (branch+100)+1, kind: FK_390_PC12DBL
#CHECK: fixup B - offset: 3, value: (target+100)+3, kind: FK_390_PC24DBL

	bprp	4, branch+100, target
	bprp	5, branch, target+100
	bprp	6, branch+100, target+100

#CHECK: bprp	7, branch@PLT, target       # encoding: [0xc5,0b0111AAAA,A,B,B,B]
#CHECK: fixup A - offset: 1, value: branch@PLT+1, kind: FK_390_PC12DBL
#CHECK: fixup B - offset: 3, value: target+3, kind: FK_390_PC24DBL
#CHECK: bprp	8, branch, target@PLT       # encoding: [0xc5,0b1000AAAA,A,B,B,B]
#CHECK: fixup A - offset: 1, value: branch+1, kind: FK_390_PC12DBL
#CHECK: fixup B - offset: 3, value: target@PLT+3, kind: FK_390_PC24DBL
#CHECK: bprp	9, branch@PLT, target@PLT   # encoding: [0xc5,0b1001AAAA,A,B,B,B]
#CHECK: fixup A - offset: 1, value: branch@PLT+1, kind: FK_390_PC12DBL
#CHECK: fixup B - offset: 3, value: target@PLT+3, kind: FK_390_PC24DBL

	bprp	7, branch@plt, target
	bprp	8, branch, target@plt
	bprp	9, branch@plt, target@plt

#CHECK: cdzt	%f0, 0(1), 0                # encoding: [0xed,0x00,0x00,0x00,0x00,0xaa]
#CHECK: cdzt	%f15, 0(1), 0               # encoding: [0xed,0x00,0x00,0x00,0xf0,0xaa]
#CHECK: cdzt	%f0, 0(1), 15               # encoding: [0xed,0x00,0x00,0x00,0x0f,0xaa]
#CHECK: cdzt	%f0, 0(1,%r1), 0            # encoding: [0xed,0x00,0x10,0x00,0x00,0xaa]
#CHECK: cdzt	%f0, 0(1,%r15), 0           # encoding: [0xed,0x00,0xf0,0x00,0x00,0xaa]
#CHECK: cdzt	%f0, 4095(1,%r1), 0         # encoding: [0xed,0x00,0x1f,0xff,0x00,0xaa]
#CHECK: cdzt	%f0, 4095(1,%r15), 0        # encoding: [0xed,0x00,0xff,0xff,0x00,0xaa]
#CHECK: cdzt	%f0, 0(256,%r1), 0          # encoding: [0xed,0xff,0x10,0x00,0x00,0xaa]
#CHECK: cdzt	%f0, 0(256,%r15), 0         # encoding: [0xed,0xff,0xf0,0x00,0x00,0xaa]

	cdzt	%f0, 0(1), 0
	cdzt	%f15, 0(1), 0
	cdzt	%f0, 0(1), 15
	cdzt	%f0, 0(1,%r1), 0
	cdzt	%f0, 0(1,%r15), 0
	cdzt	%f0, 4095(1,%r1), 0
	cdzt	%f0, 4095(1,%r15), 0
	cdzt	%f0, 0(256,%r1), 0
	cdzt	%f0, 0(256,%r15), 0

#CHECK: clt	%r0, 12, -524288            # encoding: [0xeb,0x0c,0x00,0x00,0x80,0x23]
#CHECK: clt	%r0, 12, -1                 # encoding: [0xeb,0x0c,0x0f,0xff,0xff,0x23]
#CHECK: clt	%r0, 12, 0                  # encoding: [0xeb,0x0c,0x00,0x00,0x00,0x23]
#CHECK: clt	%r0, 12, 1                  # encoding: [0xeb,0x0c,0x00,0x01,0x00,0x23]
#CHECK: clt	%r0, 12, 524287             # encoding: [0xeb,0x0c,0x0f,0xff,0x7f,0x23]
#CHECK: clt	%r0, 12, 0(%r1)             # encoding: [0xeb,0x0c,0x10,0x00,0x00,0x23]
#CHECK: clt	%r0, 12, 0(%r15)            # encoding: [0xeb,0x0c,0xf0,0x00,0x00,0x23]
#CHECK: clt	%r0, 12, 12345(%r6)         # encoding: [0xeb,0x0c,0x60,0x39,0x03,0x23]
#CHECK: clt	%r15, 12, 0                 # encoding: [0xeb,0xfc,0x00,0x00,0x00,0x23]
#CHECK: clth	%r0, 0(%r15)                # encoding: [0xeb,0x02,0xf0,0x00,0x00,0x23]
#CHECK: cltl	%r0, 0(%r15)                # encoding: [0xeb,0x04,0xf0,0x00,0x00,0x23]
#CHECK: clte	%r0, 0(%r15)                # encoding: [0xeb,0x08,0xf0,0x00,0x00,0x23]
#CHECK: cltne	%r0, 0(%r15)                # encoding: [0xeb,0x06,0xf0,0x00,0x00,0x23]
#CHECK: cltnl	%r0, 0(%r15)                # encoding: [0xeb,0x0a,0xf0,0x00,0x00,0x23]
#CHECK: cltnh	%r0, 0(%r15)                # encoding: [0xeb,0x0c,0xf0,0x00,0x00,0x23]

	clt	%r0, 12, -524288
	clt	%r0, 12, -1
	clt	%r0, 12, 0
	clt	%r0, 12, 1
	clt	%r0, 12, 524287
	clt	%r0, 12, 0(%r1)
	clt	%r0, 12, 0(%r15)
	clt	%r0, 12, 12345(%r6)
	clt	%r15, 12, 0
	clth	%r0, 0(%r15)
	cltl	%r0, 0(%r15)
	clte	%r0, 0(%r15)
	cltne	%r0, 0(%r15)
	cltnl	%r0, 0(%r15)
	cltnh	%r0, 0(%r15)

#CHECK: clgt	%r0, 12, -524288            # encoding: [0xeb,0x0c,0x00,0x00,0x80,0x2b]
#CHECK: clgt	%r0, 12, -1                 # encoding: [0xeb,0x0c,0x0f,0xff,0xff,0x2b]
#CHECK: clgt	%r0, 12, 0                  # encoding: [0xeb,0x0c,0x00,0x00,0x00,0x2b]
#CHECK: clgt	%r0, 12, 1                  # encoding: [0xeb,0x0c,0x00,0x01,0x00,0x2b]
#CHECK: clgt	%r0, 12, 524287             # encoding: [0xeb,0x0c,0x0f,0xff,0x7f,0x2b]
#CHECK: clgt	%r0, 12, 0(%r1)             # encoding: [0xeb,0x0c,0x10,0x00,0x00,0x2b]
#CHECK: clgt	%r0, 12, 0(%r15)            # encoding: [0xeb,0x0c,0xf0,0x00,0x00,0x2b]
#CHECK: clgt	%r0, 12, 12345(%r6)         # encoding: [0xeb,0x0c,0x60,0x39,0x03,0x2b]
#CHECK: clgt	%r15, 12, 0                 # encoding: [0xeb,0xfc,0x00,0x00,0x00,0x2b]
#CHECK: clgth	%r0, 0(%r15)                # encoding: [0xeb,0x02,0xf0,0x00,0x00,0x2b]
#CHECK: clgtl	%r0, 0(%r15)                # encoding: [0xeb,0x04,0xf0,0x00,0x00,0x2b]
#CHECK: clgte	%r0, 0(%r15)                # encoding: [0xeb,0x08,0xf0,0x00,0x00,0x2b]
#CHECK: clgtne	%r0, 0(%r15)                # encoding: [0xeb,0x06,0xf0,0x00,0x00,0x2b]
#CHECK: clgtnl	%r0, 0(%r15)                # encoding: [0xeb,0x0a,0xf0,0x00,0x00,0x2b]
#CHECK: clgtnh	%r0, 0(%r15)                # encoding: [0xeb,0x0c,0xf0,0x00,0x00,0x2b]

	clgt	%r0, 12, -524288
	clgt	%r0, 12, -1
	clgt	%r0, 12, 0
	clgt	%r0, 12, 1
	clgt	%r0, 12, 524287
	clgt	%r0, 12, 0(%r1)
	clgt	%r0, 12, 0(%r15)
	clgt	%r0, 12, 12345(%r6)
	clgt	%r15, 12, 0
	clgth	%r0, 0(%r15)
	clgtl	%r0, 0(%r15)
	clgte	%r0, 0(%r15)
	clgtne	%r0, 0(%r15)
	clgtnl	%r0, 0(%r15)
	clgtnh	%r0, 0(%r15)

#CHECK: cxzt	%f0, 0(1), 0                # encoding: [0xed,0x00,0x00,0x00,0x00,0xab]
#CHECK: cxzt	%f13, 0(1), 0               # encoding: [0xed,0x00,0x00,0x00,0xd0,0xab]
#CHECK: cxzt	%f0, 0(1), 15               # encoding: [0xed,0x00,0x00,0x00,0x0f,0xab]
#CHECK: cxzt	%f0, 0(1,%r1), 0            # encoding: [0xed,0x00,0x10,0x00,0x00,0xab]
#CHECK: cxzt	%f0, 0(1,%r15), 0           # encoding: [0xed,0x00,0xf0,0x00,0x00,0xab]
#CHECK: cxzt	%f0, 4095(1,%r1), 0         # encoding: [0xed,0x00,0x1f,0xff,0x00,0xab]
#CHECK: cxzt	%f0, 4095(1,%r15), 0        # encoding: [0xed,0x00,0xff,0xff,0x00,0xab]
#CHECK: cxzt	%f0, 0(256,%r1), 0          # encoding: [0xed,0xff,0x10,0x00,0x00,0xab]
#CHECK: cxzt	%f0, 0(256,%r15), 0         # encoding: [0xed,0xff,0xf0,0x00,0x00,0xab]

	cxzt	%f0, 0(1), 0
	cxzt	%f13, 0(1), 0
	cxzt	%f0, 0(1), 15
	cxzt	%f0, 0(1,%r1), 0
	cxzt	%f0, 0(1,%r15), 0
	cxzt	%f0, 4095(1,%r1), 0
	cxzt	%f0, 4095(1,%r15), 0
	cxzt	%f0, 0(256,%r1), 0
	cxzt	%f0, 0(256,%r15), 0

#CHECK: czdt	%f0, 0(1), 0                # encoding: [0xed,0x00,0x00,0x00,0x00,0xa8]
#CHECK: czdt	%f15, 0(1), 0               # encoding: [0xed,0x00,0x00,0x00,0xf0,0xa8]
#CHECK: czdt	%f0, 0(1), 15               # encoding: [0xed,0x00,0x00,0x00,0x0f,0xa8]
#CHECK: czdt	%f0, 0(1,%r1), 0            # encoding: [0xed,0x00,0x10,0x00,0x00,0xa8]
#CHECK: czdt	%f0, 0(1,%r15), 0           # encoding: [0xed,0x00,0xf0,0x00,0x00,0xa8]
#CHECK: czdt	%f0, 4095(1,%r1), 0         # encoding: [0xed,0x00,0x1f,0xff,0x00,0xa8]
#CHECK: czdt	%f0, 4095(1,%r15), 0        # encoding: [0xed,0x00,0xff,0xff,0x00,0xa8]
#CHECK: czdt	%f0, 0(256,%r1), 0          # encoding: [0xed,0xff,0x10,0x00,0x00,0xa8]
#CHECK: czdt	%f0, 0(256,%r15), 0         # encoding: [0xed,0xff,0xf0,0x00,0x00,0xa8]

	czdt	%f0, 0(1), 0
	czdt	%f15, 0(1), 0
	czdt	%f0, 0(1), 15
	czdt	%f0, 0(1,%r1), 0
	czdt	%f0, 0(1,%r15), 0
	czdt	%f0, 4095(1,%r1), 0
	czdt	%f0, 4095(1,%r15), 0
	czdt	%f0, 0(256,%r1), 0
	czdt	%f0, 0(256,%r15), 0

#CHECK: czxt	%f0, 0(1), 0                # encoding: [0xed,0x00,0x00,0x00,0x00,0xa9]
#CHECK: czxt	%f13, 0(1), 0               # encoding: [0xed,0x00,0x00,0x00,0xd0,0xa9]
#CHECK: czxt	%f0, 0(1), 15               # encoding: [0xed,0x00,0x00,0x00,0x0f,0xa9]
#CHECK: czxt	%f0, 0(1,%r1), 0            # encoding: [0xed,0x00,0x10,0x00,0x00,0xa9]
#CHECK: czxt	%f0, 0(1,%r15), 0           # encoding: [0xed,0x00,0xf0,0x00,0x00,0xa9]
#CHECK: czxt	%f0, 4095(1,%r1), 0         # encoding: [0xed,0x00,0x1f,0xff,0x00,0xa9]
#CHECK: czxt	%f0, 4095(1,%r15), 0        # encoding: [0xed,0x00,0xff,0xff,0x00,0xa9]
#CHECK: czxt	%f0, 0(256,%r1), 0          # encoding: [0xed,0xff,0x10,0x00,0x00,0xa9]
#CHECK: czxt	%f0, 0(256,%r15), 0         # encoding: [0xed,0xff,0xf0,0x00,0x00,0xa9]

	czxt	%f0, 0(1), 0
	czxt	%f13, 0(1), 0
	czxt	%f0, 0(1), 15
	czxt	%f0, 0(1,%r1), 0
	czxt	%f0, 0(1,%r15), 0
	czxt	%f0, 4095(1,%r1), 0
	czxt	%f0, 4095(1,%r15), 0
	czxt	%f0, 0(256,%r1), 0
	czxt	%f0, 0(256,%r15), 0

#CHECK: etnd	%r0                     # encoding: [0xb2,0xec,0x00,0x00]
#CHECK: etnd	%r15                    # encoding: [0xb2,0xec,0x00,0xf0]
#CHECK: etnd	%r7                     # encoding: [0xb2,0xec,0x00,0x70]

	etnd	%r0
	etnd	%r15
	etnd	%r7

#CHECK: lat	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x9f]
#CHECK: lat	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x9f]
#CHECK: lat	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x9f]
#CHECK: lat	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x9f]
#CHECK: lat	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x9f]
#CHECK: lat	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x9f]
#CHECK: lat	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x9f]
#CHECK: lat	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x9f]
#CHECK: lat	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x9f]
#CHECK: lat	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x9f]

	lat	%r0, -524288
	lat	%r0, -1
	lat	%r0, 0
	lat	%r0, 1
	lat	%r0, 524287
	lat	%r0, 0(%r1)
	lat	%r0, 0(%r15)
	lat	%r0, 524287(%r1,%r15)
	lat	%r0, 524287(%r15,%r1)
	lat	%r15, 0

#CHECK: lfhat	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0xc8]
#CHECK: lfhat	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0xc8]
#CHECK: lfhat	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0xc8]
#CHECK: lfhat	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0xc8]
#CHECK: lfhat	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0xc8]
#CHECK: lfhat	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0xc8]
#CHECK: lfhat	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0xc8]
#CHECK: lfhat	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0xc8]
#CHECK: lfhat	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0xc8]
#CHECK: lfhat	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0xc8]

	lfhat	%r0, -524288
	lfhat	%r0, -1
	lfhat	%r0, 0
	lfhat	%r0, 1
	lfhat	%r0, 524287
	lfhat	%r0, 0(%r1)
	lfhat	%r0, 0(%r15)
	lfhat	%r0, 524287(%r1,%r15)
	lfhat	%r0, 524287(%r15,%r1)
	lfhat	%r15, 0

#CHECK: lgat	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x85]
#CHECK: lgat	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x85]
#CHECK: lgat	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x85]
#CHECK: lgat	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x85]
#CHECK: lgat	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x85]
#CHECK: lgat	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x85]
#CHECK: lgat	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x85]
#CHECK: lgat	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x85]
#CHECK: lgat	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x85]
#CHECK: lgat	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x85]

	lgat	%r0, -524288
	lgat	%r0, -1
	lgat	%r0, 0
	lgat	%r0, 1
	lgat	%r0, 524287
	lgat	%r0, 0(%r1)
	lgat	%r0, 0(%r15)
	lgat	%r0, 524287(%r1,%r15)
	lgat	%r0, 524287(%r15,%r1)
	lgat	%r15, 0

#CHECK: llgfat	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x9d]
#CHECK: llgfat	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x9d]
#CHECK: llgfat	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x9d]
#CHECK: llgfat	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x9d]
#CHECK: llgfat	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x9d]
#CHECK: llgfat	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x9d]
#CHECK: llgfat	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x9d]
#CHECK: llgfat	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x9d]
#CHECK: llgfat	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x9d]
#CHECK: llgfat	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x9d]

	llgfat	%r0, -524288
	llgfat	%r0, -1
	llgfat	%r0, 0
	llgfat	%r0, 1
	llgfat	%r0, 524287
	llgfat	%r0, 0(%r1)
	llgfat	%r0, 0(%r15)
	llgfat	%r0, 524287(%r1,%r15)
	llgfat	%r0, 524287(%r15,%r1)
	llgfat	%r15, 0

#CHECK: llgtat	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x9c]
#CHECK: llgtat	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x9c]
#CHECK: llgtat	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x9c]
#CHECK: llgtat	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x9c]
#CHECK: llgtat	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x9c]
#CHECK: llgtat	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x9c]
#CHECK: llgtat	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x9c]
#CHECK: llgtat	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x9c]
#CHECK: llgtat	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x9c]
#CHECK: llgtat	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x9c]

	llgtat	%r0, -524288
	llgtat	%r0, -1
	llgtat	%r0, 0
	llgtat	%r0, 1
	llgtat	%r0, 524287
	llgtat	%r0, 0(%r1)
	llgtat	%r0, 0(%r15)
	llgtat	%r0, 524287(%r1,%r15)
	llgtat	%r0, 524287(%r15,%r1)
	llgtat	%r15, 0

#CHECK: niai	0, 0                    # encoding: [0xb2,0xfa,0x00,0x00]
#CHECK: niai	15, 0                   # encoding: [0xb2,0xfa,0x00,0xf0]
#CHECK: niai	0, 15                   # encoding: [0xb2,0xfa,0x00,0x0f]
#CHECK: niai	15, 15                  # encoding: [0xb2,0xfa,0x00,0xff]

	niai	0, 0
	niai	15, 0
	niai	0, 15
	niai	15, 15

#CHECK: ntstg	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x25]
#CHECK: ntstg	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x25]
#CHECK: ntstg	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x25]
#CHECK: ntstg	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x25]
#CHECK: ntstg	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x25]
#CHECK: ntstg	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x25]
#CHECK: ntstg	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x25]
#CHECK: ntstg	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x25]
#CHECK: ntstg	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x25]
#CHECK: ntstg	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x25]

	ntstg	%r0, -524288
	ntstg	%r0, -1
	ntstg	%r0, 0
	ntstg	%r0, 1
	ntstg	%r0, 524287
	ntstg	%r0, 0(%r1)
	ntstg	%r0, 0(%r15)
	ntstg	%r0, 524287(%r1,%r15)
	ntstg	%r0, 524287(%r15,%r1)
	ntstg	%r15, 0

#CHECK: ppa	%r0, %r0, 0             # encoding: [0xb2,0xe8,0x00,0x00]
#CHECK: ppa	%r0, %r0, 15            # encoding: [0xb2,0xe8,0xf0,0x00]
#CHECK: ppa	%r0, %r15, 0            # encoding: [0xb2,0xe8,0x00,0x0f]
#CHECK: ppa	%r4, %r6, 7             # encoding: [0xb2,0xe8,0x70,0x46]
#CHECK: ppa	%r15, %r0, 0            # encoding: [0xb2,0xe8,0x00,0xf0]

	ppa	%r0, %r0, 0
	ppa	%r0, %r0, 15
	ppa	%r0, %r15, 0
	ppa	%r4, %r6, 7
	ppa	%r15, %r0, 0

#CHECK: risbgn	%r0, %r0, 0, 0, 0       # encoding: [0xec,0x00,0x00,0x00,0x00,0x59]
#CHECK: risbgn	%r0, %r0, 0, 0, 63      # encoding: [0xec,0x00,0x00,0x00,0x3f,0x59]
#CHECK: risbgn	%r0, %r0, 0, 255, 0     # encoding: [0xec,0x00,0x00,0xff,0x00,0x59]
#CHECK: risbgn	%r0, %r0, 255, 0, 0     # encoding: [0xec,0x00,0xff,0x00,0x00,0x59]
#CHECK: risbgn	%r0, %r15, 0, 0, 0      # encoding: [0xec,0x0f,0x00,0x00,0x00,0x59]
#CHECK: risbgn	%r15, %r0, 0, 0, 0      # encoding: [0xec,0xf0,0x00,0x00,0x00,0x59]
#CHECK: risbgn	%r4, %r5, 6, 7, 8       # encoding: [0xec,0x45,0x06,0x07,0x08,0x59]

	risbgn	%r0,%r0,0,0,0
	risbgn	%r0,%r0,0,0,63
	risbgn	%r0,%r0,0,255,0
	risbgn	%r0,%r0,255,0,0
	risbgn	%r0,%r15,0,0,0
	risbgn	%r15,%r0,0,0,0
	risbgn	%r4,%r5,6,7,8

#CHECK: tabort	0                       # encoding: [0xb2,0xfc,0x00,0x00]
#CHECK: tabort	0(%r1)                  # encoding: [0xb2,0xfc,0x10,0x00]
#CHECK: tabort	0(%r15)                 # encoding: [0xb2,0xfc,0xf0,0x00]
#CHECK: tabort	4095                    # encoding: [0xb2,0xfc,0x0f,0xff]
#CHECK: tabort	4095(%r1)               # encoding: [0xb2,0xfc,0x1f,0xff]
#CHECK: tabort	4095(%r15)              # encoding: [0xb2,0xfc,0xff,0xff]

	tabort	0
	tabort	0(%r1)
	tabort	0(%r15)
	tabort	4095
	tabort	4095(%r1)
	tabort	4095(%r15)

#CHECK: tbegin	0, 0                    # encoding: [0xe5,0x60,0x00,0x00,0x00,0x00]
#CHECK: tbegin	4095, 0                 # encoding: [0xe5,0x60,0x0f,0xff,0x00,0x00]
#CHECK: tbegin	0, 0                    # encoding: [0xe5,0x60,0x00,0x00,0x00,0x00]
#CHECK: tbegin	0, 1                    # encoding: [0xe5,0x60,0x00,0x00,0x00,0x01]
#CHECK: tbegin	0, 32767                # encoding: [0xe5,0x60,0x00,0x00,0x7f,0xff]
#CHECK: tbegin	0, 32768                # encoding: [0xe5,0x60,0x00,0x00,0x80,0x00]
#CHECK: tbegin	0, 65535                # encoding: [0xe5,0x60,0x00,0x00,0xff,0xff]
#CHECK: tbegin	0(%r1), 42              # encoding: [0xe5,0x60,0x10,0x00,0x00,0x2a]
#CHECK: tbegin	0(%r15), 42             # encoding: [0xe5,0x60,0xf0,0x00,0x00,0x2a]
#CHECK: tbegin	4095(%r1), 42           # encoding: [0xe5,0x60,0x1f,0xff,0x00,0x2a]
#CHECK: tbegin	4095(%r15), 42          # encoding: [0xe5,0x60,0xff,0xff,0x00,0x2a]

	tbegin	0, 0
	tbegin	4095, 0
	tbegin	0, 0
	tbegin	0, 1
	tbegin	0, 32767
	tbegin	0, 32768
	tbegin	0, 65535
	tbegin	0(%r1), 42
	tbegin	0(%r15), 42
	tbegin	4095(%r1), 42
	tbegin	4095(%r15), 42

#CHECK: tbeginc	0, 0                    # encoding: [0xe5,0x61,0x00,0x00,0x00,0x00]
#CHECK: tbeginc	4095, 0                 # encoding: [0xe5,0x61,0x0f,0xff,0x00,0x00]
#CHECK: tbeginc	0, 0                    # encoding: [0xe5,0x61,0x00,0x00,0x00,0x00]
#CHECK: tbeginc	0, 1                    # encoding: [0xe5,0x61,0x00,0x00,0x00,0x01]
#CHECK: tbeginc	0, 32767                # encoding: [0xe5,0x61,0x00,0x00,0x7f,0xff]
#CHECK: tbeginc	0, 32768                # encoding: [0xe5,0x61,0x00,0x00,0x80,0x00]
#CHECK: tbeginc	0, 65535                # encoding: [0xe5,0x61,0x00,0x00,0xff,0xff]
#CHECK: tbeginc	0(%r1), 42              # encoding: [0xe5,0x61,0x10,0x00,0x00,0x2a]
#CHECK: tbeginc	0(%r15), 42             # encoding: [0xe5,0x61,0xf0,0x00,0x00,0x2a]
#CHECK: tbeginc	4095(%r1), 42           # encoding: [0xe5,0x61,0x1f,0xff,0x00,0x2a]
#CHECK: tbeginc	4095(%r15), 42          # encoding: [0xe5,0x61,0xff,0xff,0x00,0x2a]

	tbeginc	0, 0
	tbeginc	4095, 0
	tbeginc	0, 0
	tbeginc	0, 1
	tbeginc	0, 32767
	tbeginc	0, 32768
	tbeginc	0, 65535
	tbeginc	0(%r1), 42
	tbeginc	0(%r15), 42
	tbeginc	4095(%r1), 42
	tbeginc	4095(%r15), 42

#CHECK: tend                            # encoding: [0xb2,0xf8,0x00,0x00]

	tend
