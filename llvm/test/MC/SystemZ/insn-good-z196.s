# For z196 and above.
# RUN: llvm-mc -triple s390x-linux-gnu -mcpu=z196 -show-encoding %s | FileCheck %s
# RUN: llvm-mc -triple s390x-linux-gnu -mcpu=arch9 -show-encoding %s | FileCheck %s

#CHECK: adtra	%f0, %f0, %f0, 0        # encoding: [0xb3,0xd2,0x00,0x00]
#CHECK: adtra	%f0, %f0, %f0, 15       # encoding: [0xb3,0xd2,0x0f,0x00]
#CHECK: adtra	%f0, %f0, %f15, 0       # encoding: [0xb3,0xd2,0xf0,0x00]
#CHECK: adtra	%f0, %f15, %f0, 0       # encoding: [0xb3,0xd2,0x00,0x0f]
#CHECK: adtra	%f15, %f0, %f0, 0       # encoding: [0xb3,0xd2,0x00,0xf0]
#CHECK: adtra	%f7, %f8, %f9, 10       # encoding: [0xb3,0xd2,0x9a,0x78]

	adtra	%f0, %f0, %f0, 0
	adtra	%f0, %f0, %f0, 15
	adtra	%f0, %f0, %f15, 0
	adtra	%f0, %f15, %f0, 0
	adtra	%f15, %f0, %f0, 0
	adtra	%f7, %f8, %f9, 10

#CHECK: aghik	%r0, %r0, -32768        # encoding: [0xec,0x00,0x80,0x00,0x00,0xd9]
#CHECK: aghik	%r0, %r0, -1            # encoding: [0xec,0x00,0xff,0xff,0x00,0xd9]
#CHECK: aghik	%r0, %r0, 0             # encoding: [0xec,0x00,0x00,0x00,0x00,0xd9]
#CHECK: aghik	%r0, %r0, 1             # encoding: [0xec,0x00,0x00,0x01,0x00,0xd9]
#CHECK: aghik	%r0, %r0, 32767         # encoding: [0xec,0x00,0x7f,0xff,0x00,0xd9]
#CHECK: aghik	%r0, %r15, 0            # encoding: [0xec,0x0f,0x00,0x00,0x00,0xd9]
#CHECK: aghik	%r15, %r0, 0            # encoding: [0xec,0xf0,0x00,0x00,0x00,0xd9]
#CHECK: aghik	%r7, %r8, -16           # encoding: [0xec,0x78,0xff,0xf0,0x00,0xd9]
#CHECK: aghik	%r0, %r1, foo           # encoding: [0xec,0x01,A,A,0x00,0xd9]

	aghik	%r0, %r0, -32768
	aghik	%r0, %r0, -1
	aghik	%r0, %r0, 0
	aghik	%r0, %r0, 1
	aghik	%r0, %r0, 32767
	aghik	%r0, %r15, 0
	aghik	%r15, %r0, 0
	aghik	%r7, %r8, -16
	aghik	%r0, %r1, foo

#CHECK: agrk	%r0, %r0, %r0           # encoding: [0xb9,0xe8,0x00,0x00]
#CHECK: agrk	%r0, %r0, %r15          # encoding: [0xb9,0xe8,0xf0,0x00]
#CHECK: agrk	%r0, %r15, %r0          # encoding: [0xb9,0xe8,0x00,0x0f]
#CHECK: agrk	%r15, %r0, %r0          # encoding: [0xb9,0xe8,0x00,0xf0]
#CHECK: agrk	%r7, %r8, %r9           # encoding: [0xb9,0xe8,0x90,0x78]

	agrk	%r0,%r0,%r0
	agrk	%r0,%r0,%r15
	agrk	%r0,%r15,%r0
	agrk	%r15,%r0,%r0
	agrk	%r7,%r8,%r9

#CHECK: ahhhr	%r0, %r0, %r0           # encoding: [0xb9,0xc8,0x00,0x00]
#CHECK: ahhhr	%r0, %r0, %r15          # encoding: [0xb9,0xc8,0xf0,0x00]
#CHECK: ahhhr	%r0, %r15, %r0          # encoding: [0xb9,0xc8,0x00,0x0f]
#CHECK: ahhhr	%r15, %r0, %r0          # encoding: [0xb9,0xc8,0x00,0xf0]
#CHECK: ahhhr	%r7, %r8, %r9           # encoding: [0xb9,0xc8,0x90,0x78]

	ahhhr	%r0, %r0, %r0
	ahhhr	%r0, %r0, %r15
	ahhhr	%r0, %r15, %r0
	ahhhr	%r15, %r0, %r0
	ahhhr	%r7, %r8, %r9

#CHECK: ahhlr	%r0, %r0, %r0           # encoding: [0xb9,0xd8,0x00,0x00]
#CHECK: ahhlr	%r0, %r0, %r15          # encoding: [0xb9,0xd8,0xf0,0x00]
#CHECK: ahhlr	%r0, %r15, %r0          # encoding: [0xb9,0xd8,0x00,0x0f]
#CHECK: ahhlr	%r15, %r0, %r0          # encoding: [0xb9,0xd8,0x00,0xf0]
#CHECK: ahhlr	%r7, %r8, %r9           # encoding: [0xb9,0xd8,0x90,0x78]

	ahhlr	%r0, %r0, %r0
	ahhlr	%r0, %r0, %r15
	ahhlr	%r0, %r15, %r0
	ahhlr	%r15, %r0, %r0
	ahhlr	%r7, %r8, %r9

#CHECK: ahik	%r0, %r0, -32768        # encoding: [0xec,0x00,0x80,0x00,0x00,0xd8]
#CHECK: ahik	%r0, %r0, -1            # encoding: [0xec,0x00,0xff,0xff,0x00,0xd8]
#CHECK: ahik	%r0, %r0, 0             # encoding: [0xec,0x00,0x00,0x00,0x00,0xd8]
#CHECK: ahik	%r0, %r0, 1             # encoding: [0xec,0x00,0x00,0x01,0x00,0xd8]
#CHECK: ahik	%r0, %r0, 32767         # encoding: [0xec,0x00,0x7f,0xff,0x00,0xd8]
#CHECK: ahik	%r0, %r15, 0            # encoding: [0xec,0x0f,0x00,0x00,0x00,0xd8]
#CHECK: ahik	%r15, %r0, 0            # encoding: [0xec,0xf0,0x00,0x00,0x00,0xd8]
#CHECK: ahik	%r7, %r8, -16           # encoding: [0xec,0x78,0xff,0xf0,0x00,0xd8]
#CHECK: ahik	%r0, %r1, foo           # encoding: [0xec,0x01,A,A,0x00,0xd8]

	ahik	%r0, %r0, -32768
	ahik	%r0, %r0, -1
	ahik	%r0, %r0, 0
	ahik	%r0, %r0, 1
	ahik	%r0, %r0, 32767
	ahik	%r0, %r15, 0
	ahik	%r15, %r0, 0
	ahik	%r7, %r8, -16
	ahik	%r0, %r1, foo

#CHECK: aih	%r0, -2147483648        # encoding: [0xcc,0x08,0x80,0x00,0x00,0x00]
#CHECK: aih	%r0, -1                 # encoding: [0xcc,0x08,0xff,0xff,0xff,0xff]
#CHECK: aih	%r0, 0                  # encoding: [0xcc,0x08,0x00,0x00,0x00,0x00]
#CHECK: aih	%r0, 1                  # encoding: [0xcc,0x08,0x00,0x00,0x00,0x01]
#CHECK: aih	%r0, 2147483647         # encoding: [0xcc,0x08,0x7f,0xff,0xff,0xff]
#CHECK: aih	%r15, 0                 # encoding: [0xcc,0xf8,0x00,0x00,0x00,0x00]

	aih	%r0, -1 << 31
	aih	%r0, -1
	aih	%r0, 0
	aih	%r0, 1
	aih	%r0, (1 << 31) - 1
	aih	%r15, 0

#CHECK: alghsik	%r0, %r0, -32768        # encoding: [0xec,0x00,0x80,0x00,0x00,0xdb]
#CHECK: alghsik	%r0, %r0, -1            # encoding: [0xec,0x00,0xff,0xff,0x00,0xdb]
#CHECK: alghsik	%r0, %r0, 0             # encoding: [0xec,0x00,0x00,0x00,0x00,0xdb]
#CHECK: alghsik	%r0, %r0, 1             # encoding: [0xec,0x00,0x00,0x01,0x00,0xdb]
#CHECK: alghsik	%r0, %r0, 32767         # encoding: [0xec,0x00,0x7f,0xff,0x00,0xdb]
#CHECK: alghsik	%r0, %r15, 0            # encoding: [0xec,0x0f,0x00,0x00,0x00,0xdb]
#CHECK: alghsik	%r15, %r0, 0            # encoding: [0xec,0xf0,0x00,0x00,0x00,0xdb]
#CHECK: alghsik	%r7, %r8, -16           # encoding: [0xec,0x78,0xff,0xf0,0x00,0xdb]

	alghsik	%r0, %r0, -32768
	alghsik	%r0, %r0, -1
	alghsik	%r0, %r0, 0
	alghsik	%r0, %r0, 1
	alghsik	%r0, %r0, 32767
	alghsik	%r0, %r15, 0
	alghsik	%r15, %r0, 0
	alghsik	%r7, %r8, -16

#CHECK: algrk	%r0, %r0, %r0           # encoding: [0xb9,0xea,0x00,0x00]
#CHECK: algrk	%r0, %r0, %r15          # encoding: [0xb9,0xea,0xf0,0x00]
#CHECK: algrk	%r0, %r15, %r0          # encoding: [0xb9,0xea,0x00,0x0f]
#CHECK: algrk	%r15, %r0, %r0          # encoding: [0xb9,0xea,0x00,0xf0]
#CHECK: algrk	%r7, %r8, %r9           # encoding: [0xb9,0xea,0x90,0x78]

	algrk	%r0,%r0,%r0
	algrk	%r0,%r0,%r15
	algrk	%r0,%r15,%r0
	algrk	%r15,%r0,%r0
	algrk	%r7,%r8,%r9

#CHECK: alhhhr	%r0, %r0, %r0           # encoding: [0xb9,0xca,0x00,0x00]
#CHECK: alhhhr	%r0, %r0, %r15          # encoding: [0xb9,0xca,0xf0,0x00]
#CHECK: alhhhr	%r0, %r15, %r0          # encoding: [0xb9,0xca,0x00,0x0f]
#CHECK: alhhhr	%r15, %r0, %r0          # encoding: [0xb9,0xca,0x00,0xf0]
#CHECK: alhhhr	%r7, %r8, %r9           # encoding: [0xb9,0xca,0x90,0x78]

	alhhhr	%r0, %r0, %r0
	alhhhr	%r0, %r0, %r15
	alhhhr	%r0, %r15, %r0
	alhhhr	%r15, %r0, %r0
	alhhhr	%r7, %r8, %r9

#CHECK: alhhlr	%r0, %r0, %r0           # encoding: [0xb9,0xda,0x00,0x00]
#CHECK: alhhlr	%r0, %r0, %r15          # encoding: [0xb9,0xda,0xf0,0x00]
#CHECK: alhhlr	%r0, %r15, %r0          # encoding: [0xb9,0xda,0x00,0x0f]
#CHECK: alhhlr	%r15, %r0, %r0          # encoding: [0xb9,0xda,0x00,0xf0]
#CHECK: alhhlr	%r7, %r8, %r9           # encoding: [0xb9,0xda,0x90,0x78]

	alhhlr	%r0, %r0, %r0
	alhhlr	%r0, %r0, %r15
	alhhlr	%r0, %r15, %r0
	alhhlr	%r15, %r0, %r0
	alhhlr	%r7, %r8, %r9

#CHECK: alhsik	%r0, %r0, -32768        # encoding: [0xec,0x00,0x80,0x00,0x00,0xda]
#CHECK: alhsik	%r0, %r0, -1            # encoding: [0xec,0x00,0xff,0xff,0x00,0xda]
#CHECK: alhsik	%r0, %r0, 0             # encoding: [0xec,0x00,0x00,0x00,0x00,0xda]
#CHECK: alhsik	%r0, %r0, 1             # encoding: [0xec,0x00,0x00,0x01,0x00,0xda]
#CHECK: alhsik	%r0, %r0, 32767         # encoding: [0xec,0x00,0x7f,0xff,0x00,0xda]
#CHECK: alhsik	%r0, %r15, 0            # encoding: [0xec,0x0f,0x00,0x00,0x00,0xda]
#CHECK: alhsik	%r15, %r0, 0            # encoding: [0xec,0xf0,0x00,0x00,0x00,0xda]
#CHECK: alhsik	%r7, %r8, -16           # encoding: [0xec,0x78,0xff,0xf0,0x00,0xda]

	alhsik	%r0, %r0, -32768
	alhsik	%r0, %r0, -1
	alhsik	%r0, %r0, 0
	alhsik	%r0, %r0, 1
	alhsik	%r0, %r0, 32767
	alhsik	%r0, %r15, 0
	alhsik	%r15, %r0, 0
	alhsik	%r7, %r8, -16

#CHECK: alrk	%r0, %r0, %r0           # encoding: [0xb9,0xfa,0x00,0x00]
#CHECK: alrk	%r0, %r0, %r15          # encoding: [0xb9,0xfa,0xf0,0x00]
#CHECK: alrk	%r0, %r15, %r0          # encoding: [0xb9,0xfa,0x00,0x0f]
#CHECK: alrk	%r15, %r0, %r0          # encoding: [0xb9,0xfa,0x00,0xf0]
#CHECK: alrk	%r7, %r8, %r9           # encoding: [0xb9,0xfa,0x90,0x78]

	alrk	%r0,%r0,%r0
	alrk	%r0,%r0,%r15
	alrk	%r0,%r15,%r0
	alrk	%r15,%r0,%r0
	alrk	%r7,%r8,%r9

#CHECK: alsih	%r0, -2147483648        # encoding: [0xcc,0x0a,0x80,0x00,0x00,0x00]
#CHECK: alsih	%r0, -1                 # encoding: [0xcc,0x0a,0xff,0xff,0xff,0xff]
#CHECK: alsih	%r0, 0                  # encoding: [0xcc,0x0a,0x00,0x00,0x00,0x00]
#CHECK: alsih	%r0, 1                  # encoding: [0xcc,0x0a,0x00,0x00,0x00,0x01]
#CHECK: alsih	%r0, 2147483647         # encoding: [0xcc,0x0a,0x7f,0xff,0xff,0xff]
#CHECK: alsih	%r15, 0                 # encoding: [0xcc,0xfa,0x00,0x00,0x00,0x00]

	alsih	%r0, -1 << 31
	alsih	%r0, -1
	alsih	%r0, 0
	alsih	%r0, 1
	alsih	%r0, (1 << 31) - 1
	alsih	%r15, 0

#CHECK: alsihn	%r0, -2147483648        # encoding: [0xcc,0x0b,0x80,0x00,0x00,0x00]
#CHECK: alsihn	%r0, -1                 # encoding: [0xcc,0x0b,0xff,0xff,0xff,0xff]
#CHECK: alsihn	%r0, 0                  # encoding: [0xcc,0x0b,0x00,0x00,0x00,0x00]
#CHECK: alsihn	%r0, 1                  # encoding: [0xcc,0x0b,0x00,0x00,0x00,0x01]
#CHECK: alsihn	%r0, 2147483647         # encoding: [0xcc,0x0b,0x7f,0xff,0xff,0xff]
#CHECK: alsihn	%r15, 0                 # encoding: [0xcc,0xfb,0x00,0x00,0x00,0x00]

	alsihn	%r0, -1 << 31
	alsihn	%r0, -1
	alsihn	%r0, 0
	alsihn	%r0, 1
	alsihn	%r0, (1 << 31) - 1
	alsihn	%r15, 0

#CHECK: ark	%r0, %r0, %r0           # encoding: [0xb9,0xf8,0x00,0x00]
#CHECK: ark	%r0, %r0, %r15          # encoding: [0xb9,0xf8,0xf0,0x00]
#CHECK: ark	%r0, %r15, %r0          # encoding: [0xb9,0xf8,0x00,0x0f]
#CHECK: ark	%r15, %r0, %r0          # encoding: [0xb9,0xf8,0x00,0xf0]
#CHECK: ark	%r7, %r8, %r9           # encoding: [0xb9,0xf8,0x90,0x78]

	ark	%r0,%r0,%r0
	ark	%r0,%r0,%r15
	ark	%r0,%r15,%r0
	ark	%r15,%r0,%r0
	ark	%r7,%r8,%r9

#CHECK: axtra	%f0, %f0, %f0, 0        # encoding: [0xb3,0xda,0x00,0x00]
#CHECK: axtra	%f0, %f0, %f0, 15       # encoding: [0xb3,0xda,0x0f,0x00]
#CHECK: axtra	%f0, %f0, %f13, 0       # encoding: [0xb3,0xda,0xd0,0x00]
#CHECK: axtra	%f0, %f13, %f0, 0       # encoding: [0xb3,0xda,0x00,0x0d]
#CHECK: axtra	%f13, %f0, %f0, 0       # encoding: [0xb3,0xda,0x00,0xd0]
#CHECK: axtra	%f8, %f8, %f8, 8        # encoding: [0xb3,0xda,0x88,0x88]

	axtra	%f0, %f0, %f0, 0
	axtra	%f0, %f0, %f0, 15
	axtra	%f0, %f0, %f13, 0
	axtra	%f0, %f13, %f0, 0
	axtra	%f13, %f0, %f0, 0
	axtra	%f8, %f8, %f8, 8

#CHECK: brcth	%r0, .[[LAB:L.*]]-4294967296 # encoding: [0xcc,0x06,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-4294967296)+2, kind: FK_390_PC32DBL
	brcth	%r0, -0x100000000
#CHECK: brcth	%r0, .[[LAB:L.*]]-2	# encoding: [0xcc,0x06,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC32DBL
	brcth	%r0, -2
#CHECK: brcth	%r0, .[[LAB:L.*]]	# encoding: [0xcc,0x06,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC32DBL
	brcth	%r0, 0
#CHECK: brcth	%r0, .[[LAB:L.*]]+4294967294 # encoding: [0xcc,0x06,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+4294967294)+2, kind: FK_390_PC32DBL
	brcth	%r0, 0xfffffffe

#CHECK: brcth	%r0, foo                # encoding: [0xcc,0x06,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: brcth	%r15, foo               # encoding: [0xcc,0xf6,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL

	brcth	%r0,foo
	brcth	%r15,foo

#CHECK: brcth	%r3, bar+100            # encoding: [0xcc,0x36,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: brcth	%r4, bar+100            # encoding: [0xcc,0x46,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL

	brcth	%r3,bar+100
	brcth	%r4,bar+100

#CHECK: brcth	%r7, frob@PLT           # encoding: [0xcc,0x76,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL
#CHECK: brcth	%r8, frob@PLT           # encoding: [0xcc,0x86,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL

	brcth	%r7,frob@PLT
	brcth	%r8,frob@PLT

#CHECK: cdfbra	%f0, 0, %r0, 0          # encoding: [0xb3,0x95,0x00,0x00]
#CHECK: cdfbra	%f0, 0, %r0, 15         # encoding: [0xb3,0x95,0x0f,0x00]
#CHECK: cdfbra	%f0, 0, %r15, 0         # encoding: [0xb3,0x95,0x00,0x0f]
#CHECK: cdfbra	%f0, 15, %r0, 0         # encoding: [0xb3,0x95,0xf0,0x00]
#CHECK: cdfbra	%f4, 5, %r6, 7          # encoding: [0xb3,0x95,0x57,0x46]
#CHECK: cdfbra	%f15, 0, %r0, 0         # encoding: [0xb3,0x95,0x00,0xf0]

	cdfbra	%f0, 0, %r0, 0
	cdfbra	%f0, 0, %r0, 15
	cdfbra	%f0, 0, %r15, 0
	cdfbra	%f0, 15, %r0, 0
	cdfbra	%f4, 5, %r6, 7
	cdfbra	%f15, 0, %r0, 0

#CHECK: cdftr	%f0, 0, %r0, 0          # encoding: [0xb9,0x51,0x00,0x00]
#CHECK: cdftr	%f0, 0, %r0, 15         # encoding: [0xb9,0x51,0x0f,0x00]
#CHECK: cdftr	%f0, 0, %r15, 0         # encoding: [0xb9,0x51,0x00,0x0f]
#CHECK: cdftr	%f0, 15, %r0, 0         # encoding: [0xb9,0x51,0xf0,0x00]
#CHECK: cdftr	%f4, 5, %r6, 7          # encoding: [0xb9,0x51,0x57,0x46]
#CHECK: cdftr	%f15, 0, %r0, 0         # encoding: [0xb9,0x51,0x00,0xf0]

	cdftr	%f0, 0, %r0, 0
	cdftr	%f0, 0, %r0, 15
	cdftr	%f0, 0, %r15, 0
	cdftr	%f0, 15, %r0, 0
	cdftr	%f4, 5, %r6, 7
	cdftr	%f15, 0, %r0, 0

#CHECK: cdgbra	%f0, 0, %r0, 0          # encoding: [0xb3,0xa5,0x00,0x00]
#CHECK: cdgbra	%f0, 0, %r0, 15         # encoding: [0xb3,0xa5,0x0f,0x00]
#CHECK: cdgbra	%f0, 0, %r15, 0         # encoding: [0xb3,0xa5,0x00,0x0f]
#CHECK: cdgbra	%f0, 15, %r0, 0         # encoding: [0xb3,0xa5,0xf0,0x00]
#CHECK: cdgbra	%f4, 5, %r6, 7          # encoding: [0xb3,0xa5,0x57,0x46]
#CHECK: cdgbra	%f15, 0, %r0, 0         # encoding: [0xb3,0xa5,0x00,0xf0]

	cdgbra	%f0, 0, %r0, 0
	cdgbra	%f0, 0, %r0, 15
	cdgbra	%f0, 0, %r15, 0
	cdgbra	%f0, 15, %r0, 0
	cdgbra	%f4, 5, %r6, 7
	cdgbra	%f15, 0, %r0, 0

#CHECK: cdgtra	%f0, 0, %r0, 0          # encoding: [0xb3,0xf1,0x00,0x00]
#CHECK: cdgtra	%f0, 0, %r0, 15         # encoding: [0xb3,0xf1,0x0f,0x00]
#CHECK: cdgtra	%f0, 0, %r15, 0         # encoding: [0xb3,0xf1,0x00,0x0f]
#CHECK: cdgtra	%f0, 15, %r0, 0         # encoding: [0xb3,0xf1,0xf0,0x00]
#CHECK: cdgtra	%f4, 5, %r6, 7          # encoding: [0xb3,0xf1,0x57,0x46]
#CHECK: cdgtra	%f15, 0, %r0, 0         # encoding: [0xb3,0xf1,0x00,0xf0]

	cdgtra	%f0, 0, %r0, 0
	cdgtra	%f0, 0, %r0, 15
	cdgtra	%f0, 0, %r15, 0
	cdgtra	%f0, 15, %r0, 0
	cdgtra	%f4, 5, %r6, 7
	cdgtra	%f15, 0, %r0, 0

#CHECK: cdlfbr	%f0, 0, %r0, 0          # encoding: [0xb3,0x91,0x00,0x00]
#CHECK: cdlfbr	%f0, 0, %r0, 15         # encoding: [0xb3,0x91,0x0f,0x00]
#CHECK: cdlfbr	%f0, 0, %r15, 0         # encoding: [0xb3,0x91,0x00,0x0f]
#CHECK: cdlfbr	%f0, 15, %r0, 0         # encoding: [0xb3,0x91,0xf0,0x00]
#CHECK: cdlfbr	%f4, 5, %r6, 7          # encoding: [0xb3,0x91,0x57,0x46]
#CHECK: cdlfbr	%f15, 0, %r0, 0         # encoding: [0xb3,0x91,0x00,0xf0]

	cdlfbr	%f0, 0, %r0, 0
	cdlfbr	%f0, 0, %r0, 15
	cdlfbr	%f0, 0, %r15, 0
	cdlfbr	%f0, 15, %r0, 0
	cdlfbr	%f4, 5, %r6, 7
	cdlfbr	%f15, 0, %r0, 0

#CHECK: cdlftr	%f0, 0, %r0, 0          # encoding: [0xb9,0x53,0x00,0x00]
#CHECK: cdlftr	%f0, 0, %r0, 15         # encoding: [0xb9,0x53,0x0f,0x00]
#CHECK: cdlftr	%f0, 0, %r15, 0         # encoding: [0xb9,0x53,0x00,0x0f]
#CHECK: cdlftr	%f0, 15, %r0, 0         # encoding: [0xb9,0x53,0xf0,0x00]
#CHECK: cdlftr	%f4, 5, %r6, 7          # encoding: [0xb9,0x53,0x57,0x46]
#CHECK: cdlftr	%f15, 0, %r0, 0         # encoding: [0xb9,0x53,0x00,0xf0]

	cdlftr	%f0, 0, %r0, 0
	cdlftr	%f0, 0, %r0, 15
	cdlftr	%f0, 0, %r15, 0
	cdlftr	%f0, 15, %r0, 0
	cdlftr	%f4, 5, %r6, 7
	cdlftr	%f15, 0, %r0, 0

#CHECK: cdlgbr	%f0, 0, %r0, 0          # encoding: [0xb3,0xa1,0x00,0x00]
#CHECK: cdlgbr	%f0, 0, %r0, 15         # encoding: [0xb3,0xa1,0x0f,0x00]
#CHECK: cdlgbr	%f0, 0, %r15, 0         # encoding: [0xb3,0xa1,0x00,0x0f]
#CHECK: cdlgbr	%f0, 15, %r0, 0         # encoding: [0xb3,0xa1,0xf0,0x00]
#CHECK: cdlgbr	%f4, 5, %r6, 7          # encoding: [0xb3,0xa1,0x57,0x46]
#CHECK: cdlgbr	%f15, 0, %r0, 0         # encoding: [0xb3,0xa1,0x00,0xf0]

	cdlgbr	%f0, 0, %r0, 0
	cdlgbr	%f0, 0, %r0, 15
	cdlgbr	%f0, 0, %r15, 0
	cdlgbr	%f0, 15, %r0, 0
	cdlgbr	%f4, 5, %r6, 7
	cdlgbr	%f15, 0, %r0, 0

#CHECK: cdlgtr	%f0, 0, %r0, 0          # encoding: [0xb9,0x52,0x00,0x00]
#CHECK: cdlgtr	%f0, 0, %r0, 15         # encoding: [0xb9,0x52,0x0f,0x00]
#CHECK: cdlgtr	%f0, 0, %r15, 0         # encoding: [0xb9,0x52,0x00,0x0f]
#CHECK: cdlgtr	%f0, 15, %r0, 0         # encoding: [0xb9,0x52,0xf0,0x00]
#CHECK: cdlgtr	%f4, 5, %r6, 7          # encoding: [0xb9,0x52,0x57,0x46]
#CHECK: cdlgtr	%f15, 0, %r0, 0         # encoding: [0xb9,0x52,0x00,0xf0]

	cdlgtr	%f0, 0, %r0, 0
	cdlgtr	%f0, 0, %r0, 15
	cdlgtr	%f0, 0, %r15, 0
	cdlgtr	%f0, 15, %r0, 0
	cdlgtr	%f4, 5, %r6, 7
	cdlgtr	%f15, 0, %r0, 0

#CHECK: cefbra	%f0, 0, %r0, 0          # encoding: [0xb3,0x94,0x00,0x00]
#CHECK: cefbra	%f0, 0, %r0, 15         # encoding: [0xb3,0x94,0x0f,0x00]
#CHECK: cefbra	%f0, 0, %r15, 0         # encoding: [0xb3,0x94,0x00,0x0f]
#CHECK: cefbra	%f0, 15, %r0, 0         # encoding: [0xb3,0x94,0xf0,0x00]
#CHECK: cefbra	%f4, 5, %r6, 7          # encoding: [0xb3,0x94,0x57,0x46]
#CHECK: cefbra	%f15, 0, %r0, 0         # encoding: [0xb3,0x94,0x00,0xf0]

	cefbra	%f0, 0, %r0, 0
	cefbra	%f0, 0, %r0, 15
	cefbra	%f0, 0, %r15, 0
	cefbra	%f0, 15, %r0, 0
	cefbra	%f4, 5, %r6, 7
	cefbra	%f15, 0, %r0, 0

#CHECK: cegbra	%f0, 0, %r0, 0          # encoding: [0xb3,0xa4,0x00,0x00]
#CHECK: cegbra	%f0, 0, %r0, 15         # encoding: [0xb3,0xa4,0x0f,0x00]
#CHECK: cegbra	%f0, 0, %r15, 0         # encoding: [0xb3,0xa4,0x00,0x0f]
#CHECK: cegbra	%f0, 15, %r0, 0         # encoding: [0xb3,0xa4,0xf0,0x00]
#CHECK: cegbra	%f4, 5, %r6, 7          # encoding: [0xb3,0xa4,0x57,0x46]
#CHECK: cegbra	%f15, 0, %r0, 0         # encoding: [0xb3,0xa4,0x00,0xf0]

	cegbra	%f0, 0, %r0, 0
	cegbra	%f0, 0, %r0, 15
	cegbra	%f0, 0, %r15, 0
	cegbra	%f0, 15, %r0, 0
	cegbra	%f4, 5, %r6, 7
	cegbra	%f15, 0, %r0, 0

#CHECK: celfbr	%f0, 0, %r0, 0          # encoding: [0xb3,0x90,0x00,0x00]
#CHECK: celfbr	%f0, 0, %r0, 15         # encoding: [0xb3,0x90,0x0f,0x00]
#CHECK: celfbr	%f0, 0, %r15, 0         # encoding: [0xb3,0x90,0x00,0x0f]
#CHECK: celfbr	%f0, 15, %r0, 0         # encoding: [0xb3,0x90,0xf0,0x00]
#CHECK: celfbr	%f4, 5, %r6, 7          # encoding: [0xb3,0x90,0x57,0x46]
#CHECK: celfbr	%f15, 0, %r0, 0         # encoding: [0xb3,0x90,0x00,0xf0]

	celfbr	%f0, 0, %r0, 0
	celfbr	%f0, 0, %r0, 15
	celfbr	%f0, 0, %r15, 0
	celfbr	%f0, 15, %r0, 0
	celfbr	%f4, 5, %r6, 7
	celfbr	%f15, 0, %r0, 0

#CHECK: celgbr	%f0, 0, %r0, 0          # encoding: [0xb3,0xa0,0x00,0x00]
#CHECK: celgbr	%f0, 0, %r0, 15         # encoding: [0xb3,0xa0,0x0f,0x00]
#CHECK: celgbr	%f0, 0, %r15, 0         # encoding: [0xb3,0xa0,0x00,0x0f]
#CHECK: celgbr	%f0, 15, %r0, 0         # encoding: [0xb3,0xa0,0xf0,0x00]
#CHECK: celgbr	%f4, 5, %r6, 7          # encoding: [0xb3,0xa0,0x57,0x46]
#CHECK: celgbr	%f15, 0, %r0, 0         # encoding: [0xb3,0xa0,0x00,0xf0]

	celgbr	%f0, 0, %r0, 0
	celgbr	%f0, 0, %r0, 15
	celgbr	%f0, 0, %r15, 0
	celgbr	%f0, 15, %r0, 0
	celgbr	%f4, 5, %r6, 7
	celgbr	%f15, 0, %r0, 0

#CHECK: cfdbra	%r0, 0, %f0, 0          # encoding: [0xb3,0x99,0x00,0x00]
#CHECK: cfdbra	%r0, 0, %f0, 15         # encoding: [0xb3,0x99,0x0f,0x00]
#CHECK: cfdbra	%r0, 0, %f15, 0         # encoding: [0xb3,0x99,0x00,0x0f]
#CHECK: cfdbra	%r0, 15, %f0, 0         # encoding: [0xb3,0x99,0xf0,0x00]
#CHECK: cfdbra	%r4, 5, %f6, 7          # encoding: [0xb3,0x99,0x57,0x46]
#CHECK: cfdbra	%r15, 0, %f0, 0         # encoding: [0xb3,0x99,0x00,0xf0]

	cfdbra	%r0, 0, %f0, 0
	cfdbra	%r0, 0, %f0, 15
	cfdbra	%r0, 0, %f15, 0
	cfdbra	%r0, 15, %f0, 0
	cfdbra	%r4, 5, %f6, 7
	cfdbra	%r15, 0, %f0, 0

#CHECK: cfdtr	%r0, 0, %f0, 0          # encoding: [0xb9,0x41,0x00,0x00]
#CHECK: cfdtr	%r0, 0, %f0, 15         # encoding: [0xb9,0x41,0x0f,0x00]
#CHECK: cfdtr	%r0, 0, %f15, 0         # encoding: [0xb9,0x41,0x00,0x0f]
#CHECK: cfdtr	%r0, 15, %f0, 0         # encoding: [0xb9,0x41,0xf0,0x00]
#CHECK: cfdtr	%r4, 5, %f6, 7          # encoding: [0xb9,0x41,0x57,0x46]
#CHECK: cfdtr	%r15, 0, %f0, 0         # encoding: [0xb9,0x41,0x00,0xf0]

	cfdtr	%r0, 0, %f0, 0
	cfdtr	%r0, 0, %f0, 15
	cfdtr	%r0, 0, %f15, 0
	cfdtr	%r0, 15, %f0, 0
	cfdtr	%r4, 5, %f6, 7
	cfdtr	%r15, 0, %f0, 0

#CHECK: cfebra	%r0, 0, %f0, 0          # encoding: [0xb3,0x98,0x00,0x00]
#CHECK: cfebra	%r0, 0, %f0, 15         # encoding: [0xb3,0x98,0x0f,0x00]
#CHECK: cfebra	%r0, 0, %f15, 0         # encoding: [0xb3,0x98,0x00,0x0f]
#CHECK: cfebra	%r0, 15, %f0, 0         # encoding: [0xb3,0x98,0xf0,0x00]
#CHECK: cfebra	%r4, 5, %f6, 7          # encoding: [0xb3,0x98,0x57,0x46]
#CHECK: cfebra	%r15, 0, %f0, 0         # encoding: [0xb3,0x98,0x00,0xf0]

	cfebra	%r0, 0, %f0, 0
	cfebra	%r0, 0, %f0, 15
	cfebra	%r0, 0, %f15, 0
	cfebra	%r0, 15, %f0, 0
	cfebra	%r4, 5, %f6, 7
	cfebra	%r15, 0, %f0, 0

#CHECK: cfxbra	%r0, 0, %f0, 0          # encoding: [0xb3,0x9a,0x00,0x00]
#CHECK: cfxbra	%r0, 0, %f0, 15         # encoding: [0xb3,0x9a,0x0f,0x00]
#CHECK: cfxbra	%r0, 0, %f13, 0         # encoding: [0xb3,0x9a,0x00,0x0d]
#CHECK: cfxbra	%r0, 15, %f0, 0         # encoding: [0xb3,0x9a,0xf0,0x00]
#CHECK: cfxbra	%r7, 5, %f8, 9          # encoding: [0xb3,0x9a,0x59,0x78]
#CHECK: cfxbra	%r15, 0, %f0, 0         # encoding: [0xb3,0x9a,0x00,0xf0]

	cfxbra	%r0, 0, %f0, 0
	cfxbra	%r0, 0, %f0, 15
	cfxbra	%r0, 0, %f13, 0
	cfxbra	%r0, 15, %f0, 0
	cfxbra	%r7, 5, %f8, 9
	cfxbra	%r15, 0, %f0, 0

#CHECK: cfxtr	%r0, 0, %f0, 0          # encoding: [0xb9,0x49,0x00,0x00]
#CHECK: cfxtr	%r0, 0, %f0, 15         # encoding: [0xb9,0x49,0x0f,0x00]
#CHECK: cfxtr	%r0, 0, %f13, 0         # encoding: [0xb9,0x49,0x00,0x0d]
#CHECK: cfxtr	%r0, 15, %f0, 0         # encoding: [0xb9,0x49,0xf0,0x00]
#CHECK: cfxtr	%r7, 5, %f8, 9          # encoding: [0xb9,0x49,0x59,0x78]
#CHECK: cfxtr	%r15, 0, %f0, 0         # encoding: [0xb9,0x49,0x00,0xf0]

	cfxtr	%r0, 0, %f0, 0
	cfxtr	%r0, 0, %f0, 15
	cfxtr	%r0, 0, %f13, 0
	cfxtr	%r0, 15, %f0, 0
	cfxtr	%r7, 5, %f8, 9
	cfxtr	%r15, 0, %f0, 0

#CHECK: cgdbra	%r0, 0, %f0, 0          # encoding: [0xb3,0xa9,0x00,0x00]
#CHECK: cgdbra	%r0, 0, %f0, 15         # encoding: [0xb3,0xa9,0x0f,0x00]
#CHECK: cgdbra	%r0, 0, %f15, 0         # encoding: [0xb3,0xa9,0x00,0x0f]
#CHECK: cgdbra	%r0, 15, %f0, 0         # encoding: [0xb3,0xa9,0xf0,0x00]
#CHECK: cgdbra	%r4, 5, %f6, 7          # encoding: [0xb3,0xa9,0x57,0x46]
#CHECK: cgdbra	%r15, 0, %f0, 0         # encoding: [0xb3,0xa9,0x00,0xf0]

	cgdbra	%r0, 0, %f0, 0
	cgdbra	%r0, 0, %f0, 15
	cgdbra	%r0, 0, %f15, 0
	cgdbra	%r0, 15, %f0, 0
	cgdbra	%r4, 5, %f6, 7
	cgdbra	%r15, 0, %f0, 0

#CHECK: cgdtra	%r0, 0, %f0, 0          # encoding: [0xb3,0xe1,0x00,0x00]
#CHECK: cgdtra	%r0, 0, %f0, 15         # encoding: [0xb3,0xe1,0x0f,0x00]
#CHECK: cgdtra	%r0, 0, %f15, 0         # encoding: [0xb3,0xe1,0x00,0x0f]
#CHECK: cgdtra	%r0, 15, %f0, 0         # encoding: [0xb3,0xe1,0xf0,0x00]
#CHECK: cgdtra	%r4, 5, %f6, 7          # encoding: [0xb3,0xe1,0x57,0x46]
#CHECK: cgdtra	%r15, 0, %f0, 0         # encoding: [0xb3,0xe1,0x00,0xf0]

	cgdtra	%r0, 0, %f0, 0
	cgdtra	%r0, 0, %f0, 15
	cgdtra	%r0, 0, %f15, 0
	cgdtra	%r0, 15, %f0, 0
	cgdtra	%r4, 5, %f6, 7
	cgdtra	%r15, 0, %f0, 0

#CHECK: cgebra	%r0, 0, %f0, 0          # encoding: [0xb3,0xa8,0x00,0x00]
#CHECK: cgebra	%r0, 0, %f0, 15         # encoding: [0xb3,0xa8,0x0f,0x00]
#CHECK: cgebra	%r0, 0, %f15, 0         # encoding: [0xb3,0xa8,0x00,0x0f]
#CHECK: cgebra	%r0, 15, %f0, 0         # encoding: [0xb3,0xa8,0xf0,0x00]
#CHECK: cgebra	%r4, 5, %f6, 7          # encoding: [0xb3,0xa8,0x57,0x46]
#CHECK: cgebra	%r15, 0, %f0, 0         # encoding: [0xb3,0xa8,0x00,0xf0]

	cgebra	%r0, 0, %f0, 0
	cgebra	%r0, 0, %f0, 15
	cgebra	%r0, 0, %f15, 0
	cgebra	%r0, 15, %f0, 0
	cgebra	%r4, 5, %f6, 7
	cgebra	%r15, 0, %f0, 0

#CHECK: cgxbra	%r0, 0, %f0, 0          # encoding: [0xb3,0xaa,0x00,0x00]
#CHECK: cgxbra	%r0, 0, %f0, 15         # encoding: [0xb3,0xaa,0x0f,0x00]
#CHECK: cgxbra	%r0, 0, %f13, 0         # encoding: [0xb3,0xaa,0x00,0x0d]
#CHECK: cgxbra	%r0, 15, %f0, 0         # encoding: [0xb3,0xaa,0xf0,0x00]
#CHECK: cgxbra	%r7, 5, %f8, 9          # encoding: [0xb3,0xaa,0x59,0x78]
#CHECK: cgxbra	%r15, 0, %f0, 0         # encoding: [0xb3,0xaa,0x00,0xf0]

	cgxbra	%r0, 0, %f0, 0
	cgxbra	%r0, 0, %f0, 15
	cgxbra	%r0, 0, %f13, 0
	cgxbra	%r0, 15, %f0, 0
	cgxbra	%r7, 5, %f8, 9
	cgxbra	%r15, 0, %f0, 0

#CHECK: cgxtra	%r0, 0, %f0, 0          # encoding: [0xb3,0xe9,0x00,0x00]
#CHECK: cgxtra	%r0, 0, %f0, 15         # encoding: [0xb3,0xe9,0x0f,0x00]
#CHECK: cgxtra	%r0, 0, %f13, 0         # encoding: [0xb3,0xe9,0x00,0x0d]
#CHECK: cgxtra	%r0, 15, %f0, 0         # encoding: [0xb3,0xe9,0xf0,0x00]
#CHECK: cgxtra	%r7, 5, %f8, 9          # encoding: [0xb3,0xe9,0x59,0x78]
#CHECK: cgxtra	%r15, 0, %f0, 0         # encoding: [0xb3,0xe9,0x00,0xf0]

	cgxtra	%r0, 0, %f0, 0
	cgxtra	%r0, 0, %f0, 15
	cgxtra	%r0, 0, %f13, 0
	cgxtra	%r0, 15, %f0, 0
	cgxtra	%r7, 5, %f8, 9
	cgxtra	%r15, 0, %f0, 0

#CHECK: chf	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0xcd]
#CHECK: chf	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0xcd]
#CHECK: chf	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0xcd]
#CHECK: chf	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0xcd]
#CHECK: chf	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0xcd]
#CHECK: chf	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0xcd]
#CHECK: chf	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0xcd]
#CHECK: chf	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0xcd]
#CHECK: chf	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0xcd]
#CHECK: chf	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0xcd]

	chf	%r0, -524288
	chf	%r0, -1
	chf	%r0, 0
	chf	%r0, 1
	chf	%r0, 524287
	chf	%r0, 0(%r1)
	chf	%r0, 0(%r15)
	chf	%r0, 524287(%r1,%r15)
	chf	%r0, 524287(%r15,%r1)
	chf	%r15, 0

#CHECK: chhr	%r0, %r0                # encoding: [0xb9,0xcd,0x00,0x00]
#CHECK: chhr	%r0, %r15               # encoding: [0xb9,0xcd,0x00,0x0f]
#CHECK: chhr	%r15, %r0               # encoding: [0xb9,0xcd,0x00,0xf0]
#CHECK: chhr	%r7, %r8                # encoding: [0xb9,0xcd,0x00,0x78]

	chhr	%r0,%r0
	chhr	%r0,%r15
	chhr	%r15,%r0
	chhr	%r7,%r8

#CHECK: chlr	%r0, %r0                # encoding: [0xb9,0xdd,0x00,0x00]
#CHECK: chlr	%r0, %r15               # encoding: [0xb9,0xdd,0x00,0x0f]
#CHECK: chlr	%r15, %r0               # encoding: [0xb9,0xdd,0x00,0xf0]
#CHECK: chlr	%r7, %r8                # encoding: [0xb9,0xdd,0x00,0x78]

	chlr	%r0,%r0
	chlr	%r0,%r15
	chlr	%r15,%r0
	chlr	%r7,%r8

#CHECK: cih	%r0, -2147483648        # encoding: [0xcc,0x0d,0x80,0x00,0x00,0x00]
#CHECK: cih	%r0, -1                 # encoding: [0xcc,0x0d,0xff,0xff,0xff,0xff]
#CHECK: cih	%r0, 0                  # encoding: [0xcc,0x0d,0x00,0x00,0x00,0x00]
#CHECK: cih	%r0, 1                  # encoding: [0xcc,0x0d,0x00,0x00,0x00,0x01]
#CHECK: cih	%r0, 2147483647         # encoding: [0xcc,0x0d,0x7f,0xff,0xff,0xff]
#CHECK: cih	%r15, 0                 # encoding: [0xcc,0xfd,0x00,0x00,0x00,0x00]

	cih	%r0, -1 << 31
	cih	%r0, -1
	cih	%r0, 0
	cih	%r0, 1
	cih	%r0, (1 << 31) - 1
	cih	%r15, 0

#CHECK: clfdbr	%r0, 0, %f0, 0          # encoding: [0xb3,0x9d,0x00,0x00]
#CHECK: clfdbr	%r0, 0, %f0, 15         # encoding: [0xb3,0x9d,0x0f,0x00]
#CHECK: clfdbr	%r0, 0, %f15, 0         # encoding: [0xb3,0x9d,0x00,0x0f]
#CHECK: clfdbr	%r0, 15, %f0, 0         # encoding: [0xb3,0x9d,0xf0,0x00]
#CHECK: clfdbr	%r4, 5, %f6, 7          # encoding: [0xb3,0x9d,0x57,0x46]
#CHECK: clfdbr	%r15, 0, %f0, 0         # encoding: [0xb3,0x9d,0x00,0xf0]

	clfdbr	%r0, 0, %f0, 0
	clfdbr	%r0, 0, %f0, 15
	clfdbr	%r0, 0, %f15, 0
	clfdbr	%r0, 15, %f0, 0
	clfdbr	%r4, 5, %f6, 7
	clfdbr	%r15, 0, %f0, 0

#CHECK: clfdtr	%r0, 0, %f0, 0          # encoding: [0xb9,0x43,0x00,0x00]
#CHECK: clfdtr	%r0, 0, %f0, 15         # encoding: [0xb9,0x43,0x0f,0x00]
#CHECK: clfdtr	%r0, 0, %f15, 0         # encoding: [0xb9,0x43,0x00,0x0f]
#CHECK: clfdtr	%r0, 15, %f0, 0         # encoding: [0xb9,0x43,0xf0,0x00]
#CHECK: clfdtr	%r4, 5, %f6, 7          # encoding: [0xb9,0x43,0x57,0x46]
#CHECK: clfdtr	%r15, 0, %f0, 0         # encoding: [0xb9,0x43,0x00,0xf0]

	clfdtr	%r0, 0, %f0, 0
	clfdtr	%r0, 0, %f0, 15
	clfdtr	%r0, 0, %f15, 0
	clfdtr	%r0, 15, %f0, 0
	clfdtr	%r4, 5, %f6, 7
	clfdtr	%r15, 0, %f0, 0

#CHECK: clfebr	%r0, 0, %f0, 0          # encoding: [0xb3,0x9c,0x00,0x00]
#CHECK: clfebr	%r0, 0, %f0, 15         # encoding: [0xb3,0x9c,0x0f,0x00]
#CHECK: clfebr	%r0, 0, %f15, 0         # encoding: [0xb3,0x9c,0x00,0x0f]
#CHECK: clfebr	%r0, 15, %f0, 0         # encoding: [0xb3,0x9c,0xf0,0x00]
#CHECK: clfebr	%r4, 5, %f6, 7          # encoding: [0xb3,0x9c,0x57,0x46]
#CHECK: clfebr	%r15, 0, %f0, 0         # encoding: [0xb3,0x9c,0x00,0xf0]

	clfebr	%r0, 0, %f0, 0
	clfebr	%r0, 0, %f0, 15
	clfebr	%r0, 0, %f15, 0
	clfebr	%r0, 15, %f0, 0
	clfebr	%r4, 5, %f6, 7
	clfebr	%r15, 0, %f0, 0

#CHECK: clfxbr	%r0, 0, %f0, 0          # encoding: [0xb3,0x9e,0x00,0x00]
#CHECK: clfxbr	%r0, 0, %f0, 15         # encoding: [0xb3,0x9e,0x0f,0x00]
#CHECK: clfxbr	%r0, 0, %f13, 0         # encoding: [0xb3,0x9e,0x00,0x0d]
#CHECK: clfxbr	%r0, 15, %f0, 0         # encoding: [0xb3,0x9e,0xf0,0x00]
#CHECK: clfxbr	%r7, 5, %f8, 9          # encoding: [0xb3,0x9e,0x59,0x78]
#CHECK: clfxbr	%r15, 0, %f0, 0         # encoding: [0xb3,0x9e,0x00,0xf0]

	clfxbr	%r0, 0, %f0, 0
	clfxbr	%r0, 0, %f0, 15
	clfxbr	%r0, 0, %f13, 0
	clfxbr	%r0, 15, %f0, 0
	clfxbr	%r7, 5, %f8, 9
	clfxbr	%r15, 0, %f0, 0

#CHECK: clfxtr	%r0, 0, %f0, 0          # encoding: [0xb9,0x4b,0x00,0x00]
#CHECK: clfxtr	%r0, 0, %f0, 15         # encoding: [0xb9,0x4b,0x0f,0x00]
#CHECK: clfxtr	%r0, 0, %f13, 0         # encoding: [0xb9,0x4b,0x00,0x0d]
#CHECK: clfxtr	%r0, 15, %f0, 0         # encoding: [0xb9,0x4b,0xf0,0x00]
#CHECK: clfxtr	%r7, 5, %f8, 9          # encoding: [0xb9,0x4b,0x59,0x78]
#CHECK: clfxtr	%r15, 0, %f0, 0         # encoding: [0xb9,0x4b,0x00,0xf0]

	clfxtr	%r0, 0, %f0, 0
	clfxtr	%r0, 0, %f0, 15
	clfxtr	%r0, 0, %f13, 0
	clfxtr	%r0, 15, %f0, 0
	clfxtr	%r7, 5, %f8, 9
	clfxtr	%r15, 0, %f0, 0

#CHECK: clgdbr	%r0, 0, %f0, 0          # encoding: [0xb3,0xad,0x00,0x00]
#CHECK: clgdbr	%r0, 0, %f0, 15         # encoding: [0xb3,0xad,0x0f,0x00]
#CHECK: clgdbr	%r0, 0, %f15, 0         # encoding: [0xb3,0xad,0x00,0x0f]
#CHECK: clgdbr	%r0, 15, %f0, 0         # encoding: [0xb3,0xad,0xf0,0x00]
#CHECK: clgdbr	%r4, 5, %f6, 7          # encoding: [0xb3,0xad,0x57,0x46]
#CHECK: clgdbr	%r15, 0, %f0, 0         # encoding: [0xb3,0xad,0x00,0xf0]

	clgdbr	%r0, 0, %f0, 0
	clgdbr	%r0, 0, %f0, 15
	clgdbr	%r0, 0, %f15, 0
	clgdbr	%r0, 15, %f0, 0
	clgdbr	%r4, 5, %f6, 7
	clgdbr	%r15, 0, %f0, 0

#CHECK: clgdtr	%r0, 0, %f0, 0          # encoding: [0xb9,0x42,0x00,0x00]
#CHECK: clgdtr	%r0, 0, %f0, 15         # encoding: [0xb9,0x42,0x0f,0x00]
#CHECK: clgdtr	%r0, 0, %f15, 0         # encoding: [0xb9,0x42,0x00,0x0f]
#CHECK: clgdtr	%r0, 15, %f0, 0         # encoding: [0xb9,0x42,0xf0,0x00]
#CHECK: clgdtr	%r4, 5, %f6, 7          # encoding: [0xb9,0x42,0x57,0x46]
#CHECK: clgdtr	%r15, 0, %f0, 0         # encoding: [0xb9,0x42,0x00,0xf0]

	clgdtr	%r0, 0, %f0, 0
	clgdtr	%r0, 0, %f0, 15
	clgdtr	%r0, 0, %f15, 0
	clgdtr	%r0, 15, %f0, 0
	clgdtr	%r4, 5, %f6, 7
	clgdtr	%r15, 0, %f0, 0

#CHECK: clgebr	%r0, 0, %f0, 0          # encoding: [0xb3,0xac,0x00,0x00]
#CHECK: clgebr	%r0, 0, %f0, 15         # encoding: [0xb3,0xac,0x0f,0x00]
#CHECK: clgebr	%r0, 0, %f15, 0         # encoding: [0xb3,0xac,0x00,0x0f]
#CHECK: clgebr	%r0, 15, %f0, 0         # encoding: [0xb3,0xac,0xf0,0x00]
#CHECK: clgebr	%r4, 5, %f6, 7          # encoding: [0xb3,0xac,0x57,0x46]
#CHECK: clgebr	%r15, 0, %f0, 0         # encoding: [0xb3,0xac,0x00,0xf0]

	clgebr	%r0, 0, %f0, 0
	clgebr	%r0, 0, %f0, 15
	clgebr	%r0, 0, %f15, 0
	clgebr	%r0, 15, %f0, 0
	clgebr	%r4, 5, %f6, 7
	clgebr	%r15, 0, %f0, 0

#CHECK: clgxbr	%r0, 0, %f0, 0          # encoding: [0xb3,0xae,0x00,0x00]
#CHECK: clgxbr	%r0, 0, %f0, 15         # encoding: [0xb3,0xae,0x0f,0x00]
#CHECK: clgxbr	%r0, 0, %f13, 0         # encoding: [0xb3,0xae,0x00,0x0d]
#CHECK: clgxbr	%r0, 15, %f0, 0         # encoding: [0xb3,0xae,0xf0,0x00]
#CHECK: clgxbr	%r7, 5, %f8, 9          # encoding: [0xb3,0xae,0x59,0x78]
#CHECK: clgxbr	%r15, 0, %f0, 0         # encoding: [0xb3,0xae,0x00,0xf0]

	clgxbr	%r0, 0, %f0, 0
	clgxbr	%r0, 0, %f0, 15
	clgxbr	%r0, 0, %f13, 0
	clgxbr	%r0, 15, %f0, 0
	clgxbr	%r7, 5, %f8, 9
	clgxbr	%r15, 0, %f0, 0

#CHECK: clgxtr	%r0, 0, %f0, 0          # encoding: [0xb9,0x4a,0x00,0x00]
#CHECK: clgxtr	%r0, 0, %f0, 15         # encoding: [0xb9,0x4a,0x0f,0x00]
#CHECK: clgxtr	%r0, 0, %f13, 0         # encoding: [0xb9,0x4a,0x00,0x0d]
#CHECK: clgxtr	%r0, 15, %f0, 0         # encoding: [0xb9,0x4a,0xf0,0x00]
#CHECK: clgxtr	%r7, 5, %f8, 9          # encoding: [0xb9,0x4a,0x59,0x78]
#CHECK: clgxtr	%r15, 0, %f0, 0         # encoding: [0xb9,0x4a,0x00,0xf0]

	clgxtr	%r0, 0, %f0, 0
	clgxtr	%r0, 0, %f0, 15
	clgxtr	%r0, 0, %f13, 0
	clgxtr	%r0, 15, %f0, 0
	clgxtr	%r7, 5, %f8, 9
	clgxtr	%r15, 0, %f0, 0

#CHECK: clhf	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0xcf]
#CHECK: clhf	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0xcf]
#CHECK: clhf	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0xcf]
#CHECK: clhf	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0xcf]
#CHECK: clhf	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0xcf]
#CHECK: clhf	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0xcf]
#CHECK: clhf	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0xcf]
#CHECK: clhf	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0xcf]
#CHECK: clhf	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0xcf]
#CHECK: clhf	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0xcf]

	clhf	%r0, -524288
	clhf	%r0, -1
	clhf	%r0, 0
	clhf	%r0, 1
	clhf	%r0, 524287
	clhf	%r0, 0(%r1)
	clhf	%r0, 0(%r15)
	clhf	%r0, 524287(%r1,%r15)
	clhf	%r0, 524287(%r15,%r1)
	clhf	%r15, 0

#CHECK: clhhr	%r0, %r0                # encoding: [0xb9,0xcf,0x00,0x00]
#CHECK: clhhr	%r0, %r15               # encoding: [0xb9,0xcf,0x00,0x0f]
#CHECK: clhhr	%r15, %r0               # encoding: [0xb9,0xcf,0x00,0xf0]
#CHECK: clhhr	%r7, %r8                # encoding: [0xb9,0xcf,0x00,0x78]

	clhhr	%r0,%r0
	clhhr	%r0,%r15
	clhhr	%r15,%r0
	clhhr	%r7,%r8

#CHECK: clhlr	%r0, %r0                # encoding: [0xb9,0xdf,0x00,0x00]
#CHECK: clhlr	%r0, %r15               # encoding: [0xb9,0xdf,0x00,0x0f]
#CHECK: clhlr	%r15, %r0               # encoding: [0xb9,0xdf,0x00,0xf0]
#CHECK: clhlr	%r7, %r8                # encoding: [0xb9,0xdf,0x00,0x78]

	clhlr	%r0,%r0
	clhlr	%r0,%r15
	clhlr	%r15,%r0
	clhlr	%r7,%r8

#CHECK: clih	%r0, 0                  # encoding: [0xcc,0x0f,0x00,0x00,0x00,0x00]
#CHECK: clih	%r0, 1                  # encoding: [0xcc,0x0f,0x00,0x00,0x00,0x01]
#CHECK: clih	%r0, 4294967295         # encoding: [0xcc,0x0f,0xff,0xff,0xff,0xff]
#CHECK: clih	%r15, 0                 # encoding: [0xcc,0xff,0x00,0x00,0x00,0x00]

	clih	%r0, 0
	clih	%r0, 1
	clih	%r0, (1 << 32) - 1
	clih	%r15, 0

#CHECK: cxfbra	%f0, 0, %r0, 0          # encoding: [0xb3,0x96,0x00,0x00]
#CHECK: cxfbra	%f0, 0, %r0, 15         # encoding: [0xb3,0x96,0x0f,0x00]
#CHECK: cxfbra	%f0, 0, %r15, 0         # encoding: [0xb3,0x96,0x00,0x0f]
#CHECK: cxfbra	%f0, 15, %r0, 0         # encoding: [0xb3,0x96,0xf0,0x00]
#CHECK: cxfbra	%f4, 5, %r9, 10         # encoding: [0xb3,0x96,0x5a,0x49]
#CHECK: cxfbra	%f13, 0, %r0, 0         # encoding: [0xb3,0x96,0x00,0xd0]

	cxfbra	%f0, 0, %r0, 0
	cxfbra	%f0, 0, %r0, 15
	cxfbra	%f0, 0, %r15, 0
	cxfbra	%f0, 15, %r0, 0
	cxfbra	%f4, 5, %r9, 10
	cxfbra	%f13, 0, %r0, 0

#CHECK: cxftr	%f0, 0, %r0, 0          # encoding: [0xb9,0x59,0x00,0x00]
#CHECK: cxftr	%f0, 0, %r0, 15         # encoding: [0xb9,0x59,0x0f,0x00]
#CHECK: cxftr	%f0, 0, %r15, 0         # encoding: [0xb9,0x59,0x00,0x0f]
#CHECK: cxftr	%f0, 15, %r0, 0         # encoding: [0xb9,0x59,0xf0,0x00]
#CHECK: cxftr	%f4, 5, %r9, 10         # encoding: [0xb9,0x59,0x5a,0x49]
#CHECK: cxftr	%f13, 0, %r0, 0         # encoding: [0xb9,0x59,0x00,0xd0]

	cxftr	%f0, 0, %r0, 0
	cxftr	%f0, 0, %r0, 15
	cxftr	%f0, 0, %r15, 0
	cxftr	%f0, 15, %r0, 0
	cxftr	%f4, 5, %r9, 10
	cxftr	%f13, 0, %r0, 0

#CHECK: cxgbra	%f0, 0, %r0, 0          # encoding: [0xb3,0xa6,0x00,0x00]
#CHECK: cxgbra	%f0, 0, %r0, 15         # encoding: [0xb3,0xa6,0x0f,0x00]
#CHECK: cxgbra	%f0, 0, %r15, 0         # encoding: [0xb3,0xa6,0x00,0x0f]
#CHECK: cxgbra	%f0, 15, %r0, 0         # encoding: [0xb3,0xa6,0xf0,0x00]
#CHECK: cxgbra	%f4, 5, %r9, 10         # encoding: [0xb3,0xa6,0x5a,0x49]
#CHECK: cxgbra	%f13, 0, %r0, 0         # encoding: [0xb3,0xa6,0x00,0xd0]

	cxgbra	%f0, 0, %r0, 0
	cxgbra	%f0, 0, %r0, 15
	cxgbra	%f0, 0, %r15, 0
	cxgbra	%f0, 15, %r0, 0
	cxgbra	%f4, 5, %r9, 10
	cxgbra	%f13, 0, %r0, 0

#CHECK: cxgtra	%f0, 0, %r0, 0          # encoding: [0xb3,0xf9,0x00,0x00]
#CHECK: cxgtra	%f0, 0, %r0, 15         # encoding: [0xb3,0xf9,0x0f,0x00]
#CHECK: cxgtra	%f0, 0, %r15, 0         # encoding: [0xb3,0xf9,0x00,0x0f]
#CHECK: cxgtra	%f0, 15, %r0, 0         # encoding: [0xb3,0xf9,0xf0,0x00]
#CHECK: cxgtra	%f4, 5, %r9, 10         # encoding: [0xb3,0xf9,0x5a,0x49]
#CHECK: cxgtra	%f13, 0, %r0, 0         # encoding: [0xb3,0xf9,0x00,0xd0]

	cxgtra	%f0, 0, %r0, 0
	cxgtra	%f0, 0, %r0, 15
	cxgtra	%f0, 0, %r15, 0
	cxgtra	%f0, 15, %r0, 0
	cxgtra	%f4, 5, %r9, 10
	cxgtra	%f13, 0, %r0, 0

#CHECK: cxlfbr	%f0, 0, %r0, 0          # encoding: [0xb3,0x92,0x00,0x00]
#CHECK: cxlfbr	%f0, 0, %r0, 15         # encoding: [0xb3,0x92,0x0f,0x00]
#CHECK: cxlfbr	%f0, 0, %r15, 0         # encoding: [0xb3,0x92,0x00,0x0f]
#CHECK: cxlfbr	%f0, 15, %r0, 0         # encoding: [0xb3,0x92,0xf0,0x00]
#CHECK: cxlfbr	%f4, 5, %r9, 10         # encoding: [0xb3,0x92,0x5a,0x49]
#CHECK: cxlfbr	%f13, 0, %r0, 0         # encoding: [0xb3,0x92,0x00,0xd0]

	cxlfbr	%f0, 0, %r0, 0
	cxlfbr	%f0, 0, %r0, 15
	cxlfbr	%f0, 0, %r15, 0
	cxlfbr	%f0, 15, %r0, 0
	cxlfbr	%f4, 5, %r9, 10
	cxlfbr	%f13, 0, %r0, 0

#CHECK: cxlftr	%f0, 0, %r0, 0          # encoding: [0xb9,0x5b,0x00,0x00]
#CHECK: cxlftr	%f0, 0, %r0, 15         # encoding: [0xb9,0x5b,0x0f,0x00]
#CHECK: cxlftr	%f0, 0, %r15, 0         # encoding: [0xb9,0x5b,0x00,0x0f]
#CHECK: cxlftr	%f0, 15, %r0, 0         # encoding: [0xb9,0x5b,0xf0,0x00]
#CHECK: cxlftr	%f4, 5, %r9, 10         # encoding: [0xb9,0x5b,0x5a,0x49]
#CHECK: cxlftr	%f13, 0, %r0, 0         # encoding: [0xb9,0x5b,0x00,0xd0]

	cxlftr	%f0, 0, %r0, 0
	cxlftr	%f0, 0, %r0, 15
	cxlftr	%f0, 0, %r15, 0
	cxlftr	%f0, 15, %r0, 0
	cxlftr	%f4, 5, %r9, 10
	cxlftr	%f13, 0, %r0, 0

#CHECK: cxlgbr	%f0, 0, %r0, 0          # encoding: [0xb3,0xa2,0x00,0x00]
#CHECK: cxlgbr	%f0, 0, %r0, 15         # encoding: [0xb3,0xa2,0x0f,0x00]
#CHECK: cxlgbr	%f0, 0, %r15, 0         # encoding: [0xb3,0xa2,0x00,0x0f]
#CHECK: cxlgbr	%f0, 15, %r0, 0         # encoding: [0xb3,0xa2,0xf0,0x00]
#CHECK: cxlgbr	%f4, 5, %r9, 10         # encoding: [0xb3,0xa2,0x5a,0x49]
#CHECK: cxlgbr	%f13, 0, %r0, 0         # encoding: [0xb3,0xa2,0x00,0xd0]

	cxlgbr	%f0, 0, %r0, 0
	cxlgbr	%f0, 0, %r0, 15
	cxlgbr	%f0, 0, %r15, 0
	cxlgbr	%f0, 15, %r0, 0
	cxlgbr	%f4, 5, %r9, 10
	cxlgbr	%f13, 0, %r0, 0

#CHECK: cxlgtr	%f0, 0, %r0, 0          # encoding: [0xb9,0x5a,0x00,0x00]
#CHECK: cxlgtr	%f0, 0, %r0, 15         # encoding: [0xb9,0x5a,0x0f,0x00]
#CHECK: cxlgtr	%f0, 0, %r15, 0         # encoding: [0xb9,0x5a,0x00,0x0f]
#CHECK: cxlgtr	%f0, 15, %r0, 0         # encoding: [0xb9,0x5a,0xf0,0x00]
#CHECK: cxlgtr	%f4, 5, %r9, 10         # encoding: [0xb9,0x5a,0x5a,0x49]
#CHECK: cxlgtr	%f13, 0, %r0, 0         # encoding: [0xb9,0x5a,0x00,0xd0]

	cxlgtr	%f0, 0, %r0, 0
	cxlgtr	%f0, 0, %r0, 15
	cxlgtr	%f0, 0, %r15, 0
	cxlgtr	%f0, 15, %r0, 0
	cxlgtr	%f4, 5, %r9, 10
	cxlgtr	%f13, 0, %r0, 0

#CHECK: ddtra	%f0, %f0, %f0, 0        # encoding: [0xb3,0xd1,0x00,0x00]
#CHECK: ddtra	%f0, %f0, %f0, 15       # encoding: [0xb3,0xd1,0x0f,0x00]
#CHECK: ddtra	%f0, %f0, %f15, 0       # encoding: [0xb3,0xd1,0xf0,0x00]
#CHECK: ddtra	%f0, %f15, %f0, 0       # encoding: [0xb3,0xd1,0x00,0x0f]
#CHECK: ddtra	%f15, %f0, %f0, 0       # encoding: [0xb3,0xd1,0x00,0xf0]
#CHECK: ddtra	%f7, %f8, %f9, 10       # encoding: [0xb3,0xd1,0x9a,0x78]

	ddtra	%f0, %f0, %f0, 0
	ddtra	%f0, %f0, %f0, 15
	ddtra	%f0, %f0, %f15, 0
	ddtra	%f0, %f15, %f0, 0
	ddtra	%f15, %f0, %f0, 0
	ddtra	%f7, %f8, %f9, 10

#CHECK: dxtra	%f0, %f0, %f0, 0        # encoding: [0xb3,0xd9,0x00,0x00]
#CHECK: dxtra	%f0, %f0, %f0, 15       # encoding: [0xb3,0xd9,0x0f,0x00]
#CHECK: dxtra	%f0, %f0, %f13, 0       # encoding: [0xb3,0xd9,0xd0,0x00]
#CHECK: dxtra	%f0, %f13, %f0, 0       # encoding: [0xb3,0xd9,0x00,0x0d]
#CHECK: dxtra	%f13, %f0, %f0, 0       # encoding: [0xb3,0xd9,0x00,0xd0]
#CHECK: dxtra	%f8, %f8, %f8, 8        # encoding: [0xb3,0xd9,0x88,0x88]

	dxtra	%f0, %f0, %f0, 0
	dxtra	%f0, %f0, %f0, 15
	dxtra	%f0, %f0, %f13, 0
	dxtra	%f0, %f13, %f0, 0
	dxtra	%f13, %f0, %f0, 0
	dxtra	%f8, %f8, %f8, 8

#CHECK: fidbra	%f0, 0, %f0, 0          # encoding: [0xb3,0x5f,0x00,0x00]
#CHECK: fidbra	%f0, 0, %f0, 15         # encoding: [0xb3,0x5f,0x0f,0x00]
#CHECK: fidbra	%f0, 0, %f15, 0         # encoding: [0xb3,0x5f,0x00,0x0f]
#CHECK: fidbra	%f0, 15, %f0, 0         # encoding: [0xb3,0x5f,0xf0,0x00]
#CHECK: fidbra	%f4, 5, %f6, 7          # encoding: [0xb3,0x5f,0x57,0x46]
#CHECK: fidbra	%f15, 0, %f0, 0         # encoding: [0xb3,0x5f,0x00,0xf0]

	fidbra	%f0, 0, %f0, 0
	fidbra	%f0, 0, %f0, 15
	fidbra	%f0, 0, %f15, 0
	fidbra	%f0, 15, %f0, 0
	fidbra	%f4, 5, %f6, 7
	fidbra	%f15, 0, %f0, 0

#CHECK: fiebra	%f0, 0, %f0, 0          # encoding: [0xb3,0x57,0x00,0x00]
#CHECK: fiebra	%f0, 0, %f0, 15         # encoding: [0xb3,0x57,0x0f,0x00]
#CHECK: fiebra	%f0, 0, %f15, 0         # encoding: [0xb3,0x57,0x00,0x0f]
#CHECK: fiebra	%f0, 15, %f0, 0         # encoding: [0xb3,0x57,0xf0,0x00]
#CHECK: fiebra	%f4, 5, %f6, 7          # encoding: [0xb3,0x57,0x57,0x46]
#CHECK: fiebra	%f15, 0, %f0, 0         # encoding: [0xb3,0x57,0x00,0xf0]

	fiebra	%f0, 0, %f0, 0
	fiebra	%f0, 0, %f0, 15
	fiebra	%f0, 0, %f15, 0
	fiebra	%f0, 15, %f0, 0
	fiebra	%f4, 5, %f6, 7
	fiebra	%f15, 0, %f0, 0

#CHECK: fixbra	%f0, 0, %f0, 0          # encoding: [0xb3,0x47,0x00,0x00]
#CHECK: fixbra	%f0, 0, %f0, 15         # encoding: [0xb3,0x47,0x0f,0x00]
#CHECK: fixbra	%f0, 0, %f13, 0         # encoding: [0xb3,0x47,0x00,0x0d]
#CHECK: fixbra	%f0, 15, %f0, 0         # encoding: [0xb3,0x47,0xf0,0x00]
#CHECK: fixbra	%f4, 5, %f8, 9          # encoding: [0xb3,0x47,0x59,0x48]
#CHECK: fixbra	%f13, 0, %f0, 0         # encoding: [0xb3,0x47,0x00,0xd0]

	fixbra	%f0, 0, %f0, 0
	fixbra	%f0, 0, %f0, 15
	fixbra	%f0, 0, %f13, 0
	fixbra	%f0, 15, %f0, 0
	fixbra	%f4, 5, %f8, 9
	fixbra	%f13, 0, %f0, 0

#CHECK: kmctr	%r2, %r2, %r2           # encoding: [0xb9,0x2d,0x20,0x22]
#CHECK: kmctr	%r2, %r8, %r14          # encoding: [0xb9,0x2d,0x80,0x2e]
#CHECK: kmctr	%r14, %r8, %r2          # encoding: [0xb9,0x2d,0x80,0xe2]
#CHECK: kmctr	%r6, %r8, %r10          # encoding: [0xb9,0x2d,0x80,0x6a]

	kmctr	%r2, %r2, %r2
	kmctr	%r2, %r8, %r14
	kmctr	%r14, %r8, %r2
	kmctr	%r6, %r8, %r10

#CHECK: kmf	%r2, %r2                # encoding: [0xb9,0x2a,0x00,0x22]
#CHECK: kmf	%r2, %r14               # encoding: [0xb9,0x2a,0x00,0x2e]
#CHECK: kmf	%r14, %r2               # encoding: [0xb9,0x2a,0x00,0xe2]
#CHECK: kmf	%r6, %r10               # encoding: [0xb9,0x2a,0x00,0x6a]

	kmf	%r2, %r2
	kmf	%r2, %r14
	kmf	%r14, %r2
	kmf	%r6, %r10

#CHECK: kmo	%r2, %r2                # encoding: [0xb9,0x2b,0x00,0x22]
#CHECK: kmo	%r2, %r14               # encoding: [0xb9,0x2b,0x00,0x2e]
#CHECK: kmo	%r14, %r2               # encoding: [0xb9,0x2b,0x00,0xe2]
#CHECK: kmo	%r6, %r10               # encoding: [0xb9,0x2b,0x00,0x6a]

	kmo	%r2, %r2
	kmo	%r2, %r14
	kmo	%r14, %r2
	kmo	%r6, %r10

#CHECK: laa	%r0, %r0, -524288       # encoding: [0xeb,0x00,0x00,0x00,0x80,0xf8]
#CHECK: laa	%r0, %r0, -1            # encoding: [0xeb,0x00,0x0f,0xff,0xff,0xf8]
#CHECK: laa	%r0, %r0, 0             # encoding: [0xeb,0x00,0x00,0x00,0x00,0xf8]
#CHECK: laa	%r0, %r0, 1             # encoding: [0xeb,0x00,0x00,0x01,0x00,0xf8]
#CHECK: laa	%r0, %r0, 524287        # encoding: [0xeb,0x00,0x0f,0xff,0x7f,0xf8]
#CHECK: laa	%r0, %r0, 0(%r1)        # encoding: [0xeb,0x00,0x10,0x00,0x00,0xf8]
#CHECK: laa	%r0, %r0, 0(%r15)       # encoding: [0xeb,0x00,0xf0,0x00,0x00,0xf8]
#CHECK: laa	%r0, %r0, 524287(%r1)   # encoding: [0xeb,0x00,0x1f,0xff,0x7f,0xf8]
#CHECK: laa	%r0, %r0, 524287(%r15)  # encoding: [0xeb,0x00,0xff,0xff,0x7f,0xf8]
#CHECK: laa	%r0, %r15, 0            # encoding: [0xeb,0x0f,0x00,0x00,0x00,0xf8]
#CHECK: laa	%r15, %r0, 0            # encoding: [0xeb,0xf0,0x00,0x00,0x00,0xf8]

	laa	%r0, %r0, -524288
	laa	%r0, %r0, -1
	laa	%r0, %r0, 0
	laa	%r0, %r0, 1
	laa	%r0, %r0, 524287
	laa	%r0, %r0, 0(%r1)
	laa	%r0, %r0, 0(%r15)
	laa	%r0, %r0, 524287(%r1)
	laa	%r0, %r0, 524287(%r15)
	laa	%r0, %r15, 0
	laa	%r15, %r0, 0

#CHECK: laag	%r0, %r0, -524288       # encoding: [0xeb,0x00,0x00,0x00,0x80,0xe8]
#CHECK: laag	%r0, %r0, -1            # encoding: [0xeb,0x00,0x0f,0xff,0xff,0xe8]
#CHECK: laag	%r0, %r0, 0             # encoding: [0xeb,0x00,0x00,0x00,0x00,0xe8]
#CHECK: laag	%r0, %r0, 1             # encoding: [0xeb,0x00,0x00,0x01,0x00,0xe8]
#CHECK: laag	%r0, %r0, 524287        # encoding: [0xeb,0x00,0x0f,0xff,0x7f,0xe8]
#CHECK: laag	%r0, %r0, 0(%r1)        # encoding: [0xeb,0x00,0x10,0x00,0x00,0xe8]
#CHECK: laag	%r0, %r0, 0(%r15)       # encoding: [0xeb,0x00,0xf0,0x00,0x00,0xe8]
#CHECK: laag	%r0, %r0, 524287(%r1)   # encoding: [0xeb,0x00,0x1f,0xff,0x7f,0xe8]
#CHECK: laag	%r0, %r0, 524287(%r15)  # encoding: [0xeb,0x00,0xff,0xff,0x7f,0xe8]
#CHECK: laag	%r0, %r15, 0            # encoding: [0xeb,0x0f,0x00,0x00,0x00,0xe8]
#CHECK: laag	%r15, %r0, 0            # encoding: [0xeb,0xf0,0x00,0x00,0x00,0xe8]

	laag	%r0, %r0, -524288
	laag	%r0, %r0, -1
	laag	%r0, %r0, 0
	laag	%r0, %r0, 1
	laag	%r0, %r0, 524287
	laag	%r0, %r0, 0(%r1)
	laag	%r0, %r0, 0(%r15)
	laag	%r0, %r0, 524287(%r1)
	laag	%r0, %r0, 524287(%r15)
	laag	%r0, %r15, 0
	laag	%r15, %r0, 0

#CHECK: laal	%r0, %r0, -524288       # encoding: [0xeb,0x00,0x00,0x00,0x80,0xfa]
#CHECK: laal	%r0, %r0, -1            # encoding: [0xeb,0x00,0x0f,0xff,0xff,0xfa]
#CHECK: laal	%r0, %r0, 0             # encoding: [0xeb,0x00,0x00,0x00,0x00,0xfa]
#CHECK: laal	%r0, %r0, 1             # encoding: [0xeb,0x00,0x00,0x01,0x00,0xfa]
#CHECK: laal	%r0, %r0, 524287        # encoding: [0xeb,0x00,0x0f,0xff,0x7f,0xfa]
#CHECK: laal	%r0, %r0, 0(%r1)        # encoding: [0xeb,0x00,0x10,0x00,0x00,0xfa]
#CHECK: laal	%r0, %r0, 0(%r15)       # encoding: [0xeb,0x00,0xf0,0x00,0x00,0xfa]
#CHECK: laal	%r0, %r0, 524287(%r1)   # encoding: [0xeb,0x00,0x1f,0xff,0x7f,0xfa]
#CHECK: laal	%r0, %r0, 524287(%r15)  # encoding: [0xeb,0x00,0xff,0xff,0x7f,0xfa]
#CHECK: laal	%r0, %r15, 0            # encoding: [0xeb,0x0f,0x00,0x00,0x00,0xfa]
#CHECK: laal	%r15, %r0, 0            # encoding: [0xeb,0xf0,0x00,0x00,0x00,0xfa]

	laal	%r0, %r0, -524288
	laal	%r0, %r0, -1
	laal	%r0, %r0, 0
	laal	%r0, %r0, 1
	laal	%r0, %r0, 524287
	laal	%r0, %r0, 0(%r1)
	laal	%r0, %r0, 0(%r15)
	laal	%r0, %r0, 524287(%r1)
	laal	%r0, %r0, 524287(%r15)
	laal	%r0, %r15, 0
	laal	%r15, %r0, 0

#CHECK: laalg	%r0, %r0, -524288       # encoding: [0xeb,0x00,0x00,0x00,0x80,0xea]
#CHECK: laalg	%r0, %r0, -1            # encoding: [0xeb,0x00,0x0f,0xff,0xff,0xea]
#CHECK: laalg	%r0, %r0, 0             # encoding: [0xeb,0x00,0x00,0x00,0x00,0xea]
#CHECK: laalg	%r0, %r0, 1             # encoding: [0xeb,0x00,0x00,0x01,0x00,0xea]
#CHECK: laalg	%r0, %r0, 524287        # encoding: [0xeb,0x00,0x0f,0xff,0x7f,0xea]
#CHECK: laalg	%r0, %r0, 0(%r1)        # encoding: [0xeb,0x00,0x10,0x00,0x00,0xea]
#CHECK: laalg	%r0, %r0, 0(%r15)       # encoding: [0xeb,0x00,0xf0,0x00,0x00,0xea]
#CHECK: laalg	%r0, %r0, 524287(%r1)   # encoding: [0xeb,0x00,0x1f,0xff,0x7f,0xea]
#CHECK: laalg	%r0, %r0, 524287(%r15)  # encoding: [0xeb,0x00,0xff,0xff,0x7f,0xea]
#CHECK: laalg	%r0, %r15, 0            # encoding: [0xeb,0x0f,0x00,0x00,0x00,0xea]
#CHECK: laalg	%r15, %r0, 0            # encoding: [0xeb,0xf0,0x00,0x00,0x00,0xea]

	laalg	%r0, %r0, -524288
	laalg	%r0, %r0, -1
	laalg	%r0, %r0, 0
	laalg	%r0, %r0, 1
	laalg	%r0, %r0, 524287
	laalg	%r0, %r0, 0(%r1)
	laalg	%r0, %r0, 0(%r15)
	laalg	%r0, %r0, 524287(%r1)
	laalg	%r0, %r0, 524287(%r15)
	laalg	%r0, %r15, 0
	laalg	%r15, %r0, 0

#CHECK: lan	%r0, %r0, -524288       # encoding: [0xeb,0x00,0x00,0x00,0x80,0xf4]
#CHECK: lan	%r0, %r0, -1            # encoding: [0xeb,0x00,0x0f,0xff,0xff,0xf4]
#CHECK: lan	%r0, %r0, 0             # encoding: [0xeb,0x00,0x00,0x00,0x00,0xf4]
#CHECK: lan	%r0, %r0, 1             # encoding: [0xeb,0x00,0x00,0x01,0x00,0xf4]
#CHECK: lan	%r0, %r0, 524287        # encoding: [0xeb,0x00,0x0f,0xff,0x7f,0xf4]
#CHECK: lan	%r0, %r0, 0(%r1)        # encoding: [0xeb,0x00,0x10,0x00,0x00,0xf4]
#CHECK: lan	%r0, %r0, 0(%r15)       # encoding: [0xeb,0x00,0xf0,0x00,0x00,0xf4]
#CHECK: lan	%r0, %r0, 524287(%r1)   # encoding: [0xeb,0x00,0x1f,0xff,0x7f,0xf4]
#CHECK: lan	%r0, %r0, 524287(%r15)  # encoding: [0xeb,0x00,0xff,0xff,0x7f,0xf4]
#CHECK: lan	%r0, %r15, 0            # encoding: [0xeb,0x0f,0x00,0x00,0x00,0xf4]
#CHECK: lan	%r15, %r0, 0            # encoding: [0xeb,0xf0,0x00,0x00,0x00,0xf4]

	lan	%r0, %r0, -524288
	lan	%r0, %r0, -1
	lan	%r0, %r0, 0
	lan	%r0, %r0, 1
	lan	%r0, %r0, 524287
	lan	%r0, %r0, 0(%r1)
	lan	%r0, %r0, 0(%r15)
	lan	%r0, %r0, 524287(%r1)
	lan	%r0, %r0, 524287(%r15)
	lan	%r0, %r15, 0
	lan	%r15, %r0, 0

#CHECK: lang	%r0, %r0, -524288       # encoding: [0xeb,0x00,0x00,0x00,0x80,0xe4]
#CHECK: lang	%r0, %r0, -1            # encoding: [0xeb,0x00,0x0f,0xff,0xff,0xe4]
#CHECK: lang	%r0, %r0, 0             # encoding: [0xeb,0x00,0x00,0x00,0x00,0xe4]
#CHECK: lang	%r0, %r0, 1             # encoding: [0xeb,0x00,0x00,0x01,0x00,0xe4]
#CHECK: lang	%r0, %r0, 524287        # encoding: [0xeb,0x00,0x0f,0xff,0x7f,0xe4]
#CHECK: lang	%r0, %r0, 0(%r1)        # encoding: [0xeb,0x00,0x10,0x00,0x00,0xe4]
#CHECK: lang	%r0, %r0, 0(%r15)       # encoding: [0xeb,0x00,0xf0,0x00,0x00,0xe4]
#CHECK: lang	%r0, %r0, 524287(%r1)   # encoding: [0xeb,0x00,0x1f,0xff,0x7f,0xe4]
#CHECK: lang	%r0, %r0, 524287(%r15)  # encoding: [0xeb,0x00,0xff,0xff,0x7f,0xe4]
#CHECK: lang	%r0, %r15, 0            # encoding: [0xeb,0x0f,0x00,0x00,0x00,0xe4]
#CHECK: lang	%r15, %r0, 0            # encoding: [0xeb,0xf0,0x00,0x00,0x00,0xe4]

	lang	%r0, %r0, -524288
	lang	%r0, %r0, -1
	lang	%r0, %r0, 0
	lang	%r0, %r0, 1
	lang	%r0, %r0, 524287
	lang	%r0, %r0, 0(%r1)
	lang	%r0, %r0, 0(%r15)
	lang	%r0, %r0, 524287(%r1)
	lang	%r0, %r0, 524287(%r15)
	lang	%r0, %r15, 0
	lang	%r15, %r0, 0

#CHECK: lao	%r0, %r0, -524288       # encoding: [0xeb,0x00,0x00,0x00,0x80,0xf6]
#CHECK: lao	%r0, %r0, -1            # encoding: [0xeb,0x00,0x0f,0xff,0xff,0xf6]
#CHECK: lao	%r0, %r0, 0             # encoding: [0xeb,0x00,0x00,0x00,0x00,0xf6]
#CHECK: lao	%r0, %r0, 1             # encoding: [0xeb,0x00,0x00,0x01,0x00,0xf6]
#CHECK: lao	%r0, %r0, 524287        # encoding: [0xeb,0x00,0x0f,0xff,0x7f,0xf6]
#CHECK: lao	%r0, %r0, 0(%r1)        # encoding: [0xeb,0x00,0x10,0x00,0x00,0xf6]
#CHECK: lao	%r0, %r0, 0(%r15)       # encoding: [0xeb,0x00,0xf0,0x00,0x00,0xf6]
#CHECK: lao	%r0, %r0, 524287(%r1)   # encoding: [0xeb,0x00,0x1f,0xff,0x7f,0xf6]
#CHECK: lao	%r0, %r0, 524287(%r15)  # encoding: [0xeb,0x00,0xff,0xff,0x7f,0xf6]
#CHECK: lao	%r0, %r15, 0            # encoding: [0xeb,0x0f,0x00,0x00,0x00,0xf6]
#CHECK: lao	%r15, %r0, 0            # encoding: [0xeb,0xf0,0x00,0x00,0x00,0xf6]

	lao	%r0, %r0, -524288
	lao	%r0, %r0, -1
	lao	%r0, %r0, 0
	lao	%r0, %r0, 1
	lao	%r0, %r0, 524287
	lao	%r0, %r0, 0(%r1)
	lao	%r0, %r0, 0(%r15)
	lao	%r0, %r0, 524287(%r1)
	lao	%r0, %r0, 524287(%r15)
	lao	%r0, %r15, 0
	lao	%r15, %r0, 0

#CHECK: laog	%r0, %r0, -524288       # encoding: [0xeb,0x00,0x00,0x00,0x80,0xe6]
#CHECK: laog	%r0, %r0, -1            # encoding: [0xeb,0x00,0x0f,0xff,0xff,0xe6]
#CHECK: laog	%r0, %r0, 0             # encoding: [0xeb,0x00,0x00,0x00,0x00,0xe6]
#CHECK: laog	%r0, %r0, 1             # encoding: [0xeb,0x00,0x00,0x01,0x00,0xe6]
#CHECK: laog	%r0, %r0, 524287        # encoding: [0xeb,0x00,0x0f,0xff,0x7f,0xe6]
#CHECK: laog	%r0, %r0, 0(%r1)        # encoding: [0xeb,0x00,0x10,0x00,0x00,0xe6]
#CHECK: laog	%r0, %r0, 0(%r15)       # encoding: [0xeb,0x00,0xf0,0x00,0x00,0xe6]
#CHECK: laog	%r0, %r0, 524287(%r1)   # encoding: [0xeb,0x00,0x1f,0xff,0x7f,0xe6]
#CHECK: laog	%r0, %r0, 524287(%r15)  # encoding: [0xeb,0x00,0xff,0xff,0x7f,0xe6]
#CHECK: laog	%r0, %r15, 0            # encoding: [0xeb,0x0f,0x00,0x00,0x00,0xe6]
#CHECK: laog	%r15, %r0, 0            # encoding: [0xeb,0xf0,0x00,0x00,0x00,0xe6]

	laog	%r0, %r0, -524288
	laog	%r0, %r0, -1
	laog	%r0, %r0, 0
	laog	%r0, %r0, 1
	laog	%r0, %r0, 524287
	laog	%r0, %r0, 0(%r1)
	laog	%r0, %r0, 0(%r15)
	laog	%r0, %r0, 524287(%r1)
	laog	%r0, %r0, 524287(%r15)
	laog	%r0, %r15, 0
	laog	%r15, %r0, 0

#CHECK: lax	%r0, %r0, -524288       # encoding: [0xeb,0x00,0x00,0x00,0x80,0xf7]
#CHECK: lax	%r0, %r0, -1            # encoding: [0xeb,0x00,0x0f,0xff,0xff,0xf7]
#CHECK: lax	%r0, %r0, 0             # encoding: [0xeb,0x00,0x00,0x00,0x00,0xf7]
#CHECK: lax	%r0, %r0, 1             # encoding: [0xeb,0x00,0x00,0x01,0x00,0xf7]
#CHECK: lax	%r0, %r0, 524287        # encoding: [0xeb,0x00,0x0f,0xff,0x7f,0xf7]
#CHECK: lax	%r0, %r0, 0(%r1)        # encoding: [0xeb,0x00,0x10,0x00,0x00,0xf7]
#CHECK: lax	%r0, %r0, 0(%r15)       # encoding: [0xeb,0x00,0xf0,0x00,0x00,0xf7]
#CHECK: lax	%r0, %r0, 524287(%r1)   # encoding: [0xeb,0x00,0x1f,0xff,0x7f,0xf7]
#CHECK: lax	%r0, %r0, 524287(%r15)  # encoding: [0xeb,0x00,0xff,0xff,0x7f,0xf7]
#CHECK: lax	%r0, %r15, 0            # encoding: [0xeb,0x0f,0x00,0x00,0x00,0xf7]
#CHECK: lax	%r15, %r0, 0            # encoding: [0xeb,0xf0,0x00,0x00,0x00,0xf7]

	lax	%r0, %r0, -524288
	lax	%r0, %r0, -1
	lax	%r0, %r0, 0
	lax	%r0, %r0, 1
	lax	%r0, %r0, 524287
	lax	%r0, %r0, 0(%r1)
	lax	%r0, %r0, 0(%r15)
	lax	%r0, %r0, 524287(%r1)
	lax	%r0, %r0, 524287(%r15)
	lax	%r0, %r15, 0
	lax	%r15, %r0, 0

#CHECK: laxg	%r0, %r0, -524288       # encoding: [0xeb,0x00,0x00,0x00,0x80,0xe7]
#CHECK: laxg	%r0, %r0, -1            # encoding: [0xeb,0x00,0x0f,0xff,0xff,0xe7]
#CHECK: laxg	%r0, %r0, 0             # encoding: [0xeb,0x00,0x00,0x00,0x00,0xe7]
#CHECK: laxg	%r0, %r0, 1             # encoding: [0xeb,0x00,0x00,0x01,0x00,0xe7]
#CHECK: laxg	%r0, %r0, 524287        # encoding: [0xeb,0x00,0x0f,0xff,0x7f,0xe7]
#CHECK: laxg	%r0, %r0, 0(%r1)        # encoding: [0xeb,0x00,0x10,0x00,0x00,0xe7]
#CHECK: laxg	%r0, %r0, 0(%r15)       # encoding: [0xeb,0x00,0xf0,0x00,0x00,0xe7]
#CHECK: laxg	%r0, %r0, 524287(%r1)   # encoding: [0xeb,0x00,0x1f,0xff,0x7f,0xe7]
#CHECK: laxg	%r0, %r0, 524287(%r15)  # encoding: [0xeb,0x00,0xff,0xff,0x7f,0xe7]
#CHECK: laxg	%r0, %r15, 0            # encoding: [0xeb,0x0f,0x00,0x00,0x00,0xe7]
#CHECK: laxg	%r15, %r0, 0            # encoding: [0xeb,0xf0,0x00,0x00,0x00,0xe7]

	laxg	%r0, %r0, -524288
	laxg	%r0, %r0, -1
	laxg	%r0, %r0, 0
	laxg	%r0, %r0, 1
	laxg	%r0, %r0, 524287
	laxg	%r0, %r0, 0(%r1)
	laxg	%r0, %r0, 0(%r15)
	laxg	%r0, %r0, 524287(%r1)
	laxg	%r0, %r0, 524287(%r15)
	laxg	%r0, %r15, 0
	laxg	%r15, %r0, 0

#CHECK: lbh	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0xc0]
#CHECK: lbh	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0xc0]
#CHECK: lbh	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0xc0]
#CHECK: lbh	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0xc0]
#CHECK: lbh	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0xc0]
#CHECK: lbh	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0xc0]
#CHECK: lbh	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0xc0]
#CHECK: lbh	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0xc0]
#CHECK: lbh	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0xc0]
#CHECK: lbh	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0xc0]

	lbh	%r0, -524288
	lbh	%r0, -1
	lbh	%r0, 0
	lbh	%r0, 1
	lbh	%r0, 524287
	lbh	%r0, 0(%r1)
	lbh	%r0, 0(%r15)
	lbh	%r0, 524287(%r1,%r15)
	lbh	%r0, 524287(%r15,%r1)
	lbh	%r15, 0

#CHECK: ldxbra	%f0, 0, %f0, 0          # encoding: [0xb3,0x45,0x00,0x00]
#CHECK: ldxbra	%f0, 0, %f0, 15         # encoding: [0xb3,0x45,0x0f,0x00]
#CHECK: ldxbra	%f0, 0, %f13, 0         # encoding: [0xb3,0x45,0x00,0x0d]
#CHECK: ldxbra	%f0, 15, %f0, 0         # encoding: [0xb3,0x45,0xf0,0x00]
#CHECK: ldxbra	%f4, 5, %f8, 9          # encoding: [0xb3,0x45,0x59,0x48]
#CHECK: ldxbra	%f13, 0, %f0, 0         # encoding: [0xb3,0x45,0x00,0xd0]

	ldxbra	%f0, 0, %f0, 0
	ldxbra	%f0, 0, %f0, 15
	ldxbra	%f0, 0, %f13, 0
	ldxbra	%f0, 15, %f0, 0
	ldxbra	%f4, 5, %f8, 9
	ldxbra	%f13, 0, %f0, 0

#CHECK: ledbra	%f0, 0, %f0, 0          # encoding: [0xb3,0x44,0x00,0x00]
#CHECK: ledbra	%f0, 0, %f0, 15         # encoding: [0xb3,0x44,0x0f,0x00]
#CHECK: ledbra	%f0, 0, %f15, 0         # encoding: [0xb3,0x44,0x00,0x0f]
#CHECK: ledbra	%f0, 15, %f0, 0         # encoding: [0xb3,0x44,0xf0,0x00]
#CHECK: ledbra	%f4, 5, %f6, 7          # encoding: [0xb3,0x44,0x57,0x46]
#CHECK: ledbra	%f15, 0, %f0, 0         # encoding: [0xb3,0x44,0x00,0xf0]

	ledbra	%f0, 0, %f0, 0
	ledbra	%f0, 0, %f0, 15
	ledbra	%f0, 0, %f15, 0
	ledbra	%f0, 15, %f0, 0
	ledbra	%f4, 5, %f6, 7
	ledbra	%f15, 0, %f0, 0

#CHECK: lexbra	%f0, 0, %f0, 0          # encoding: [0xb3,0x46,0x00,0x00]
#CHECK: lexbra	%f0, 0, %f0, 15         # encoding: [0xb3,0x46,0x0f,0x00]
#CHECK: lexbra	%f0, 0, %f13, 0         # encoding: [0xb3,0x46,0x00,0x0d]
#CHECK: lexbra	%f0, 15, %f0, 0         # encoding: [0xb3,0x46,0xf0,0x00]
#CHECK: lexbra	%f4, 5, %f8, 9          # encoding: [0xb3,0x46,0x59,0x48]
#CHECK: lexbra	%f13, 0, %f0, 0         # encoding: [0xb3,0x46,0x00,0xd0]

	lexbra	%f0, 0, %f0, 0
	lexbra	%f0, 0, %f0, 15
	lexbra	%f0, 0, %f13, 0
	lexbra	%f0, 15, %f0, 0
	lexbra	%f4, 5, %f8, 9
	lexbra	%f13, 0, %f0, 0

#CHECK: lfh	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0xca]
#CHECK: lfh	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0xca]
#CHECK: lfh	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0xca]
#CHECK: lfh	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0xca]
#CHECK: lfh	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0xca]
#CHECK: lfh	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0xca]
#CHECK: lfh	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0xca]
#CHECK: lfh	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0xca]
#CHECK: lfh	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0xca]
#CHECK: lfh	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0xca]

	lfh	%r0, -524288
	lfh	%r0, -1
	lfh	%r0, 0
	lfh	%r0, 1
	lfh	%r0, 524287
	lfh	%r0, 0(%r1)
	lfh	%r0, 0(%r15)
	lfh	%r0, 524287(%r1,%r15)
	lfh	%r0, 524287(%r15,%r1)
	lfh	%r15, 0

#CHECK: lhh	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0xc4]
#CHECK: lhh	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0xc4]
#CHECK: lhh	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0xc4]
#CHECK: lhh	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0xc4]
#CHECK: lhh	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0xc4]
#CHECK: lhh	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0xc4]
#CHECK: lhh	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0xc4]
#CHECK: lhh	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0xc4]
#CHECK: lhh	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0xc4]
#CHECK: lhh	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0xc4]

	lhh	%r0, -524288
	lhh	%r0, -1
	lhh	%r0, 0
	lhh	%r0, 1
	lhh	%r0, 524287
	lhh	%r0, 0(%r1)
	lhh	%r0, 0(%r15)
	lhh	%r0, 524287(%r1,%r15)
	lhh	%r0, 524287(%r15,%r1)
	lhh	%r15, 0

#CHECK: llch	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0xc2]
#CHECK: llch	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0xc2]
#CHECK: llch	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0xc2]
#CHECK: llch	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0xc2]
#CHECK: llch	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0xc2]
#CHECK: llch	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0xc2]
#CHECK: llch	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0xc2]
#CHECK: llch	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0xc2]
#CHECK: llch	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0xc2]
#CHECK: llch	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0xc2]

	llch	%r0, -524288
	llch	%r0, -1
	llch	%r0, 0
	llch	%r0, 1
	llch	%r0, 524287
	llch	%r0, 0(%r1)
	llch	%r0, 0(%r15)
	llch	%r0, 524287(%r1,%r15)
	llch	%r0, 524287(%r15,%r1)
	llch	%r15, 0

#CHECK: llhh	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0xc6]
#CHECK: llhh	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0xc6]
#CHECK: llhh	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0xc6]
#CHECK: llhh	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0xc6]
#CHECK: llhh	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0xc6]
#CHECK: llhh	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0xc6]
#CHECK: llhh	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0xc6]
#CHECK: llhh	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0xc6]
#CHECK: llhh	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0xc6]
#CHECK: llhh	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0xc6]

	llhh	%r0, -524288
	llhh	%r0, -1
	llhh	%r0, 0
	llhh	%r0, 1
	llhh	%r0, 524287
	llhh	%r0, 0(%r1)
	llhh	%r0, 0(%r15)
	llhh	%r0, 524287(%r1,%r15)
	llhh	%r0, 524287(%r15,%r1)
	llhh	%r15, 0

#CHECK: loc	%r0, 0, 0               # encoding: [0xeb,0x00,0x00,0x00,0x00,0xf2]
#CHECK: loc	%r0, 0, 15              # encoding: [0xeb,0x0f,0x00,0x00,0x00,0xf2]
#CHECK: loc	%r0, -524288, 0         # encoding: [0xeb,0x00,0x00,0x00,0x80,0xf2]
#CHECK: loc	%r0, 524287, 0          # encoding: [0xeb,0x00,0x0f,0xff,0x7f,0xf2]
#CHECK: loc	%r0, 0(%r1), 0          # encoding: [0xeb,0x00,0x10,0x00,0x00,0xf2]
#CHECK: loc	%r0, 0(%r15), 0         # encoding: [0xeb,0x00,0xf0,0x00,0x00,0xf2]
#CHECK: loc	%r15, 0, 0              # encoding: [0xeb,0xf0,0x00,0x00,0x00,0xf2]
#CHECK: loc	%r1, 4095(%r2), 3       # encoding: [0xeb,0x13,0x2f,0xff,0x00,0xf2]

	loc	%r0,0,0
	loc	%r0,0,15
	loc	%r0,-524288,0
	loc	%r0,524287,0
	loc	%r0,0(%r1),0
	loc	%r0,0(%r15),0
	loc	%r15,0,0
	loc	%r1,4095(%r2),3

#CHECK: loco   %r1, 2(%r3)              # encoding: [0xeb,0x11,0x30,0x02,0x00,0xf2]
#CHECK: loch   %r1, 2(%r3)              # encoding: [0xeb,0x12,0x30,0x02,0x00,0xf2]
#CHECK: locp   %r1, 2(%r3)              # encoding: [0xeb,0x12,0x30,0x02,0x00,0xf2]
#CHECK: locnle %r1, 2(%r3)              # encoding: [0xeb,0x13,0x30,0x02,0x00,0xf2]
#CHECK: locl   %r1, 2(%r3)              # encoding: [0xeb,0x14,0x30,0x02,0x00,0xf2]
#CHECK: locm   %r1, 2(%r3)              # encoding: [0xeb,0x14,0x30,0x02,0x00,0xf2]
#CHECK: locnhe %r1, 2(%r3)              # encoding: [0xeb,0x15,0x30,0x02,0x00,0xf2]
#CHECK: loclh  %r1, 2(%r3)              # encoding: [0xeb,0x16,0x30,0x02,0x00,0xf2]
#CHECK: locne  %r1, 2(%r3)              # encoding: [0xeb,0x17,0x30,0x02,0x00,0xf2]
#CHECK: locnz  %r1, 2(%r3)              # encoding: [0xeb,0x17,0x30,0x02,0x00,0xf2]
#CHECK: loce   %r1, 2(%r3)              # encoding: [0xeb,0x18,0x30,0x02,0x00,0xf2]
#CHECK: locz   %r1, 2(%r3)              # encoding: [0xeb,0x18,0x30,0x02,0x00,0xf2]
#CHECK: locnlh %r1, 2(%r3)              # encoding: [0xeb,0x19,0x30,0x02,0x00,0xf2]
#CHECK: loche  %r1, 2(%r3)              # encoding: [0xeb,0x1a,0x30,0x02,0x00,0xf2]
#CHECK: locnl  %r1, 2(%r3)              # encoding: [0xeb,0x1b,0x30,0x02,0x00,0xf2]
#CHECK: locnm  %r1, 2(%r3)              # encoding: [0xeb,0x1b,0x30,0x02,0x00,0xf2]
#CHECK: locle  %r1, 2(%r3)              # encoding: [0xeb,0x1c,0x30,0x02,0x00,0xf2]
#CHECK: locnh  %r1, 2(%r3)              # encoding: [0xeb,0x1d,0x30,0x02,0x00,0xf2]
#CHECK: locnp  %r1, 2(%r3)              # encoding: [0xeb,0x1d,0x30,0x02,0x00,0xf2]
#CHECK: locno  %r1, 2(%r3)              # encoding: [0xeb,0x1e,0x30,0x02,0x00,0xf2]

	loco   %r1,2(%r3)
	loch   %r1,2(%r3)
	locp   %r1,2(%r3)
	locnle %r1,2(%r3)
	locl   %r1,2(%r3)
	locm   %r1,2(%r3)
	locnhe %r1,2(%r3)
	loclh  %r1,2(%r3)
	locne  %r1,2(%r3)
	locnz  %r1,2(%r3)
	loce   %r1,2(%r3)
	locz   %r1,2(%r3)
	locnlh %r1,2(%r3)
	loche  %r1,2(%r3)
	locnl  %r1,2(%r3)
	locnm  %r1,2(%r3)
	locle  %r1,2(%r3)
	locnh  %r1,2(%r3)
	locnp  %r1,2(%r3)
	locno  %r1,2(%r3)

#CHECK: locg	%r0, 0, 0               # encoding: [0xeb,0x00,0x00,0x00,0x00,0xe2]
#CHECK: locg	%r0, 0, 15              # encoding: [0xeb,0x0f,0x00,0x00,0x00,0xe2]
#CHECK: locg	%r0, -524288, 0         # encoding: [0xeb,0x00,0x00,0x00,0x80,0xe2]
#CHECK: locg	%r0, 524287, 0          # encoding: [0xeb,0x00,0x0f,0xff,0x7f,0xe2]
#CHECK: locg	%r0, 0(%r1), 0          # encoding: [0xeb,0x00,0x10,0x00,0x00,0xe2]
#CHECK: locg	%r0, 0(%r15), 0         # encoding: [0xeb,0x00,0xf0,0x00,0x00,0xe2]
#CHECK: locg	%r15, 0, 0              # encoding: [0xeb,0xf0,0x00,0x00,0x00,0xe2]
#CHECK: locg	%r1, 4095(%r2), 3       # encoding: [0xeb,0x13,0x2f,0xff,0x00,0xe2]

	locg	%r0,0,0
	locg	%r0,0,15
	locg	%r0,-524288,0
	locg	%r0,524287,0
	locg	%r0,0(%r1),0
	locg	%r0,0(%r15),0
	locg	%r15,0,0
	locg	%r1,4095(%r2),3

#CHECK: locgo   %r1, 2(%r3)             # encoding: [0xeb,0x11,0x30,0x02,0x00,0xe2]
#CHECK: locgh   %r1, 2(%r3)             # encoding: [0xeb,0x12,0x30,0x02,0x00,0xe2]
#CHECK: locgp   %r1, 2(%r3)             # encoding: [0xeb,0x12,0x30,0x02,0x00,0xe2]
#CHECK: locgnle %r1, 2(%r3)             # encoding: [0xeb,0x13,0x30,0x02,0x00,0xe2]
#CHECK: locgl   %r1, 2(%r3)             # encoding: [0xeb,0x14,0x30,0x02,0x00,0xe2]
#CHECK: locgm   %r1, 2(%r3)             # encoding: [0xeb,0x14,0x30,0x02,0x00,0xe2]
#CHECK: locgnhe %r1, 2(%r3)             # encoding: [0xeb,0x15,0x30,0x02,0x00,0xe2]
#CHECK: locglh  %r1, 2(%r3)             # encoding: [0xeb,0x16,0x30,0x02,0x00,0xe2]
#CHECK: locgne  %r1, 2(%r3)             # encoding: [0xeb,0x17,0x30,0x02,0x00,0xe2]
#CHECK: locgnz  %r1, 2(%r3)             # encoding: [0xeb,0x17,0x30,0x02,0x00,0xe2]
#CHECK: locge   %r1, 2(%r3)             # encoding: [0xeb,0x18,0x30,0x02,0x00,0xe2]
#CHECK: locgz   %r1, 2(%r3)             # encoding: [0xeb,0x18,0x30,0x02,0x00,0xe2]
#CHECK: locgnlh %r1, 2(%r3)             # encoding: [0xeb,0x19,0x30,0x02,0x00,0xe2]
#CHECK: locghe  %r1, 2(%r3)             # encoding: [0xeb,0x1a,0x30,0x02,0x00,0xe2]
#CHECK: locgnl  %r1, 2(%r3)             # encoding: [0xeb,0x1b,0x30,0x02,0x00,0xe2]
#CHECK: locgnm  %r1, 2(%r3)             # encoding: [0xeb,0x1b,0x30,0x02,0x00,0xe2]
#CHECK: locgle  %r1, 2(%r3)             # encoding: [0xeb,0x1c,0x30,0x02,0x00,0xe2]
#CHECK: locgnh  %r1, 2(%r3)             # encoding: [0xeb,0x1d,0x30,0x02,0x00,0xe2]
#CHECK: locgnp  %r1, 2(%r3)             # encoding: [0xeb,0x1d,0x30,0x02,0x00,0xe2]
#CHECK: locgno  %r1, 2(%r3)             # encoding: [0xeb,0x1e,0x30,0x02,0x00,0xe2]

	locgo   %r1,2(%r3)
	locgh   %r1,2(%r3)
	locgp   %r1,2(%r3)
	locgnle %r1,2(%r3)
	locgl   %r1,2(%r3)
	locgm   %r1,2(%r3)
	locgnhe %r1,2(%r3)
	locglh  %r1,2(%r3)
	locgne  %r1,2(%r3)
	locgnz  %r1,2(%r3)
	locge   %r1,2(%r3)
	locgz   %r1,2(%r3)
	locgnlh %r1,2(%r3)
	locghe  %r1,2(%r3)
	locgnl  %r1,2(%r3)
	locgnm  %r1,2(%r3)
	locgle  %r1,2(%r3)
	locgnh  %r1,2(%r3)
	locgnp  %r1,2(%r3)
	locgno  %r1,2(%r3)

#CHECK: locgr	%r1, %r2, 0             # encoding: [0xb9,0xe2,0x00,0x12]
#CHECK: locgr	%r1, %r2, 15            # encoding: [0xb9,0xe2,0xf0,0x12]

	locgr	%r1,%r2,0
	locgr	%r1,%r2,15

#CHECK: locgro   %r1, %r3               # encoding: [0xb9,0xe2,0x10,0x13]
#CHECK: locgrh   %r1, %r3               # encoding: [0xb9,0xe2,0x20,0x13]
#CHECK: locgrp   %r1, %r3               # encoding: [0xb9,0xe2,0x20,0x13]
#CHECK: locgrnle %r1, %r3               # encoding: [0xb9,0xe2,0x30,0x13]
#CHECK: locgrl   %r1, %r3               # encoding: [0xb9,0xe2,0x40,0x13]
#CHECK: locgrm   %r1, %r3               # encoding: [0xb9,0xe2,0x40,0x13]
#CHECK: locgrnhe %r1, %r3               # encoding: [0xb9,0xe2,0x50,0x13]
#CHECK: locgrlh  %r1, %r3               # encoding: [0xb9,0xe2,0x60,0x13]
#CHECK: locgrne  %r1, %r3               # encoding: [0xb9,0xe2,0x70,0x13]
#CHECK: locgrnz  %r1, %r3               # encoding: [0xb9,0xe2,0x70,0x13]
#CHECK: locgre   %r1, %r3               # encoding: [0xb9,0xe2,0x80,0x13]
#CHECK: locgrz   %r1, %r3               # encoding: [0xb9,0xe2,0x80,0x13]
#CHECK: locgrnlh %r1, %r3               # encoding: [0xb9,0xe2,0x90,0x13]
#CHECK: locgrhe  %r1, %r3               # encoding: [0xb9,0xe2,0xa0,0x13]
#CHECK: locgrnl  %r1, %r3               # encoding: [0xb9,0xe2,0xb0,0x13]
#CHECK: locgrnm  %r1, %r3               # encoding: [0xb9,0xe2,0xb0,0x13]
#CHECK: locgrle  %r1, %r3               # encoding: [0xb9,0xe2,0xc0,0x13]
#CHECK: locgrnh  %r1, %r3               # encoding: [0xb9,0xe2,0xd0,0x13]
#CHECK: locgrnp  %r1, %r3               # encoding: [0xb9,0xe2,0xd0,0x13]
#CHECK: locgrno  %r1, %r3               # encoding: [0xb9,0xe2,0xe0,0x13]

	locgro   %r1,%r3
	locgrh   %r1,%r3
	locgrp   %r1,%r3
	locgrnle %r1,%r3
	locgrl   %r1,%r3
	locgrm   %r1,%r3
	locgrnhe %r1,%r3
	locgrlh  %r1,%r3
	locgrne  %r1,%r3
	locgrnz  %r1,%r3
	locgre   %r1,%r3
	locgrz   %r1,%r3
	locgrnlh %r1,%r3
	locgrhe  %r1,%r3
	locgrnl  %r1,%r3
	locgrnm  %r1,%r3
	locgrle  %r1,%r3
	locgrnh  %r1,%r3
	locgrnp  %r1,%r3
	locgrno  %r1,%r3

#CHECK: locr	%r1, %r2, 0             # encoding: [0xb9,0xf2,0x00,0x12]
#CHECK: locr	%r1, %r2, 15            # encoding: [0xb9,0xf2,0xf0,0x12]

	locr	%r1,%r2,0
	locr	%r1,%r2,15

#CHECK: locro   %r1, %r3                # encoding: [0xb9,0xf2,0x10,0x13]
#CHECK: locrh   %r1, %r3                # encoding: [0xb9,0xf2,0x20,0x13]
#CHECK: locrp   %r1, %r3                # encoding: [0xb9,0xf2,0x20,0x13]
#CHECK: locrnle %r1, %r3                # encoding: [0xb9,0xf2,0x30,0x13]
#CHECK: locrl   %r1, %r3                # encoding: [0xb9,0xf2,0x40,0x13]
#CHECK: locrm   %r1, %r3                # encoding: [0xb9,0xf2,0x40,0x13]
#CHECK: locrnhe %r1, %r3                # encoding: [0xb9,0xf2,0x50,0x13]
#CHECK: locrlh  %r1, %r3                # encoding: [0xb9,0xf2,0x60,0x13]
#CHECK: locrne  %r1, %r3                # encoding: [0xb9,0xf2,0x70,0x13]
#CHECK: locrnz  %r1, %r3                # encoding: [0xb9,0xf2,0x70,0x13]
#CHECK: locre   %r1, %r3                # encoding: [0xb9,0xf2,0x80,0x13]
#CHECK: locrz   %r1, %r3                # encoding: [0xb9,0xf2,0x80,0x13]
#CHECK: locrnlh %r1, %r3                # encoding: [0xb9,0xf2,0x90,0x13]
#CHECK: locrhe  %r1, %r3                # encoding: [0xb9,0xf2,0xa0,0x13]
#CHECK: locrnl  %r1, %r3                # encoding: [0xb9,0xf2,0xb0,0x13]
#CHECK: locrnm  %r1, %r3                # encoding: [0xb9,0xf2,0xb0,0x13]
#CHECK: locrle  %r1, %r3                # encoding: [0xb9,0xf2,0xc0,0x13]
#CHECK: locrnh  %r1, %r3                # encoding: [0xb9,0xf2,0xd0,0x13]
#CHECK: locrnp  %r1, %r3                # encoding: [0xb9,0xf2,0xd0,0x13]
#CHECK: locrno  %r1, %r3                # encoding: [0xb9,0xf2,0xe0,0x13]

	locro   %r1,%r3
	locrh   %r1,%r3
	locrp   %r1,%r3
	locrnle %r1,%r3
	locrl   %r1,%r3
	locrm   %r1,%r3
	locrnhe %r1,%r3
	locrlh  %r1,%r3
	locrne  %r1,%r3
	locrnz  %r1,%r3
	locre   %r1,%r3
	locrz   %r1,%r3
	locrnlh %r1,%r3
	locrhe  %r1,%r3
	locrnl  %r1,%r3
	locrnm  %r1,%r3
	locrle  %r1,%r3
	locrnh  %r1,%r3
	locrnp  %r1,%r3
	locrno  %r1,%r3

#CHECK: lpd	%r0, 0, 0               # encoding: [0xc8,0x04,0x00,0x00,0x00,0x00]
#CHECK: lpd	%r2, 0(%r1), 0(%r15)    # encoding: [0xc8,0x24,0x10,0x00,0xf0,0x00]
#CHECK: lpd	%r2, 1(%r1), 0(%r15)    # encoding: [0xc8,0x24,0x10,0x01,0xf0,0x00]
#CHECK: lpd	%r2, 4095(%r1), 0(%r15) # encoding: [0xc8,0x24,0x1f,0xff,0xf0,0x00]
#CHECK: lpd	%r2, 0(%r1), 1(%r15)    # encoding: [0xc8,0x24,0x10,0x00,0xf0,0x01]
#CHECK: lpd	%r2, 0(%r1), 4095(%r15) # encoding: [0xc8,0x24,0x10,0x00,0xff,0xff]

	lpd	%r0, 0, 0
	lpd	%r2, 0(%r1), 0(%r15)
	lpd	%r2, 1(%r1), 0(%r15)
	lpd	%r2, 4095(%r1), 0(%r15)
	lpd	%r2, 0(%r1), 1(%r15)
	lpd	%r2, 0(%r1), 4095(%r15)

#CHECK: lpdg	%r0, 0, 0               # encoding: [0xc8,0x05,0x00,0x00,0x00,0x00]
#CHECK: lpdg	%r2, 0(%r1), 0(%r15)    # encoding: [0xc8,0x25,0x10,0x00,0xf0,0x00]
#CHECK: lpdg	%r2, 1(%r1), 0(%r15)    # encoding: [0xc8,0x25,0x10,0x01,0xf0,0x00]
#CHECK: lpdg	%r2, 4095(%r1), 0(%r15) # encoding: [0xc8,0x25,0x1f,0xff,0xf0,0x00]
#CHECK: lpdg	%r2, 0(%r1), 1(%r15)    # encoding: [0xc8,0x25,0x10,0x00,0xf0,0x01]
#CHECK: lpdg	%r2, 0(%r1), 4095(%r15) # encoding: [0xc8,0x25,0x10,0x00,0xff,0xff]

	lpdg	%r0, 0, 0
	lpdg	%r2, 0(%r1), 0(%r15)
	lpdg	%r2, 1(%r1), 0(%r15)
	lpdg	%r2, 4095(%r1), 0(%r15)
	lpdg	%r2, 0(%r1), 1(%r15)
	lpdg	%r2, 0(%r1), 4095(%r15)

#CHECK: mdtra	%f0, %f0, %f0, 0        # encoding: [0xb3,0xd0,0x00,0x00]
#CHECK: mdtra	%f0, %f0, %f0, 15       # encoding: [0xb3,0xd0,0x0f,0x00]
#CHECK: mdtra	%f0, %f0, %f15, 0       # encoding: [0xb3,0xd0,0xf0,0x00]
#CHECK: mdtra	%f0, %f15, %f0, 0       # encoding: [0xb3,0xd0,0x00,0x0f]
#CHECK: mdtra	%f15, %f0, %f0, 0       # encoding: [0xb3,0xd0,0x00,0xf0]
#CHECK: mdtra	%f7, %f8, %f9, 10       # encoding: [0xb3,0xd0,0x9a,0x78]

	mdtra	%f0, %f0, %f0, 0
	mdtra	%f0, %f0, %f0, 15
	mdtra	%f0, %f0, %f15, 0
	mdtra	%f0, %f15, %f0, 0
	mdtra	%f15, %f0, %f0, 0
	mdtra	%f7, %f8, %f9, 10

#CHECK: mxtra	%f0, %f0, %f0, 0        # encoding: [0xb3,0xd8,0x00,0x00]
#CHECK: mxtra	%f0, %f0, %f0, 15       # encoding: [0xb3,0xd8,0x0f,0x00]
#CHECK: mxtra	%f0, %f0, %f13, 0       # encoding: [0xb3,0xd8,0xd0,0x00]
#CHECK: mxtra	%f0, %f13, %f0, 0       # encoding: [0xb3,0xd8,0x00,0x0d]
#CHECK: mxtra	%f13, %f0, %f0, 0       # encoding: [0xb3,0xd8,0x00,0xd0]
#CHECK: mxtra	%f8, %f8, %f8, 8        # encoding: [0xb3,0xd8,0x88,0x88]

	mxtra	%f0, %f0, %f0, 0
	mxtra	%f0, %f0, %f0, 15
	mxtra	%f0, %f0, %f13, 0
	mxtra	%f0, %f13, %f0, 0
	mxtra	%f13, %f0, %f0, 0
	mxtra	%f8, %f8, %f8, 8

#CHECK: ngrk	%r0, %r0, %r0           # encoding: [0xb9,0xe4,0x00,0x00]
#CHECK: ngrk	%r0, %r0, %r15          # encoding: [0xb9,0xe4,0xf0,0x00]
#CHECK: ngrk	%r0, %r15, %r0          # encoding: [0xb9,0xe4,0x00,0x0f]
#CHECK: ngrk	%r15, %r0, %r0          # encoding: [0xb9,0xe4,0x00,0xf0]
#CHECK: ngrk	%r7, %r8, %r9           # encoding: [0xb9,0xe4,0x90,0x78]

	ngrk	%r0,%r0,%r0
	ngrk	%r0,%r0,%r15
	ngrk	%r0,%r15,%r0
	ngrk	%r15,%r0,%r0
	ngrk	%r7,%r8,%r9

#CHECK: nrk	%r0, %r0, %r0           # encoding: [0xb9,0xf4,0x00,0x00]
#CHECK: nrk	%r0, %r0, %r15          # encoding: [0xb9,0xf4,0xf0,0x00]
#CHECK: nrk	%r0, %r15, %r0          # encoding: [0xb9,0xf4,0x00,0x0f]
#CHECK: nrk	%r15, %r0, %r0          # encoding: [0xb9,0xf4,0x00,0xf0]
#CHECK: nrk	%r7, %r8, %r9           # encoding: [0xb9,0xf4,0x90,0x78]

	nrk	%r0,%r0,%r0
	nrk	%r0,%r0,%r15
	nrk	%r0,%r15,%r0
	nrk	%r15,%r0,%r0
	nrk	%r7,%r8,%r9

#CHECK: ogrk	%r0, %r0, %r0           # encoding: [0xb9,0xe6,0x00,0x00]
#CHECK: ogrk	%r0, %r0, %r15          # encoding: [0xb9,0xe6,0xf0,0x00]
#CHECK: ogrk	%r0, %r15, %r0          # encoding: [0xb9,0xe6,0x00,0x0f]
#CHECK: ogrk	%r15, %r0, %r0          # encoding: [0xb9,0xe6,0x00,0xf0]
#CHECK: ogrk	%r7, %r8, %r9           # encoding: [0xb9,0xe6,0x90,0x78]

	ogrk	%r0,%r0,%r0
	ogrk	%r0,%r0,%r15
	ogrk	%r0,%r15,%r0
	ogrk	%r15,%r0,%r0
	ogrk	%r7,%r8,%r9

#CHECK: ork	%r0, %r0, %r0           # encoding: [0xb9,0xf6,0x00,0x00]
#CHECK: ork	%r0, %r0, %r15          # encoding: [0xb9,0xf6,0xf0,0x00]
#CHECK: ork	%r0, %r15, %r0          # encoding: [0xb9,0xf6,0x00,0x0f]
#CHECK: ork	%r15, %r0, %r0          # encoding: [0xb9,0xf6,0x00,0xf0]
#CHECK: ork	%r7, %r8, %r9           # encoding: [0xb9,0xf6,0x90,0x78]

	ork	%r0,%r0,%r0
	ork	%r0,%r0,%r15
	ork	%r0,%r15,%r0
	ork	%r15,%r0,%r0
	ork	%r7,%r8,%r9

#CHECK: pcc                             # encoding: [0xb9,0x2c,0x00,0x00]

	pcc

#CHECK: pckmo                           # encoding: [0xb9,0x28,0x00,0x00]

	pckmo

#CHECK: popcnt	%r0, %r0                # encoding: [0xb9,0xe1,0x00,0x00]
#CHECK: popcnt	%r0, %r15               # encoding: [0xb9,0xe1,0x00,0x0f]
#CHECK: popcnt	%r15, %r0               # encoding: [0xb9,0xe1,0x00,0xf0]
#CHECK: popcnt	%r7, %r8                # encoding: [0xb9,0xe1,0x00,0x78]

	popcnt	%r0,%r0
	popcnt	%r0,%r15
	popcnt	%r15,%r0
	popcnt	%r7,%r8

#CHECK: risbhg	%r0, %r0, 0, 0, 0       # encoding: [0xec,0x00,0x00,0x00,0x00,0x5d]
#CHECK: risbhg	%r0, %r0, 0, 0, 63      # encoding: [0xec,0x00,0x00,0x00,0x3f,0x5d]
#CHECK: risbhg	%r0, %r0, 0, 0, 64      # encoding: [0xec,0x00,0x00,0x00,0x40,0x5d]
#CHECK: risbhg	%r0, %r0, 0, 0, 255     # encoding: [0xec,0x00,0x00,0x00,0xff,0x5d]
#CHECK: risbhg	%r0, %r0, 0, 255, 0     # encoding: [0xec,0x00,0x00,0xff,0x00,0x5d]
#CHECK: risbhg	%r0, %r0, 255, 0, 0     # encoding: [0xec,0x00,0xff,0x00,0x00,0x5d]
#CHECK: risbhg	%r0, %r15, 0, 0, 0      # encoding: [0xec,0x0f,0x00,0x00,0x00,0x5d]
#CHECK: risbhg	%r15, %r0, 0, 0, 0      # encoding: [0xec,0xf0,0x00,0x00,0x00,0x5d]
#CHECK: risbhg	%r4, %r5, 6, 7, 8       # encoding: [0xec,0x45,0x06,0x07,0x08,0x5d]

	risbhg	%r0,%r0,0,0,0
	risbhg	%r0,%r0,0,0,63
	risbhg	%r0,%r0,0,0,64
	risbhg	%r0,%r0,0,0,255
	risbhg	%r0,%r0,0,255,0
	risbhg	%r0,%r0,255,0,0
	risbhg	%r0,%r15,0,0,0
	risbhg	%r15,%r0,0,0,0
	risbhg	%r4,%r5,6,7,8

#CHECK: risblg	%r0, %r0, 0, 0, 0       # encoding: [0xec,0x00,0x00,0x00,0x00,0x51]
#CHECK: risblg	%r0, %r0, 0, 0, 63      # encoding: [0xec,0x00,0x00,0x00,0x3f,0x51]
#CHECK: risblg	%r0, %r0, 0, 0, 64      # encoding: [0xec,0x00,0x00,0x00,0x40,0x51]
#CHECK: risblg	%r0, %r0, 0, 0, 255     # encoding: [0xec,0x00,0x00,0x00,0xff,0x51]
#CHECK: risblg	%r0, %r0, 0, 255, 0     # encoding: [0xec,0x00,0x00,0xff,0x00,0x51]
#CHECK: risblg	%r0, %r0, 255, 0, 0     # encoding: [0xec,0x00,0xff,0x00,0x00,0x51]
#CHECK: risblg	%r0, %r15, 0, 0, 0      # encoding: [0xec,0x0f,0x00,0x00,0x00,0x51]
#CHECK: risblg	%r15, %r0, 0, 0, 0      # encoding: [0xec,0xf0,0x00,0x00,0x00,0x51]
#CHECK: risblg	%r4, %r5, 6, 7, 8       # encoding: [0xec,0x45,0x06,0x07,0x08,0x51]

	risblg	%r0,%r0,0,0,0
	risblg	%r0,%r0,0,0,63
	risblg	%r0,%r0,0,0,64
	risblg	%r0,%r0,0,0,255
	risblg	%r0,%r0,0,255,0
	risblg	%r0,%r0,255,0,0
	risblg	%r0,%r15,0,0,0
	risblg	%r15,%r0,0,0,0
	risblg	%r4,%r5,6,7,8

#CHECK: rrbm	%r0, %r0                # encoding: [0xb9,0xae,0x00,0x00]
#CHECK: rrbm	%r0, %r15               # encoding: [0xb9,0xae,0x00,0x0f]
#CHECK: rrbm	%r15, %r0               # encoding: [0xb9,0xae,0x00,0xf0]
#CHECK: rrbm	%r7, %r8                # encoding: [0xb9,0xae,0x00,0x78]
#CHECK: rrbm	%r15, %r15              # encoding: [0xb9,0xae,0x00,0xff]

	rrbm	%r0,%r0
	rrbm	%r0,%r15
	rrbm	%r15,%r0
	rrbm	%r7,%r8
	rrbm	%r15,%r15

#CHECK: sdtra	%f0, %f0, %f0, 0        # encoding: [0xb3,0xd3,0x00,0x00]
#CHECK: sdtra	%f0, %f0, %f0, 15       # encoding: [0xb3,0xd3,0x0f,0x00]
#CHECK: sdtra	%f0, %f0, %f15, 0       # encoding: [0xb3,0xd3,0xf0,0x00]
#CHECK: sdtra	%f0, %f15, %f0, 0       # encoding: [0xb3,0xd3,0x00,0x0f]
#CHECK: sdtra	%f15, %f0, %f0, 0       # encoding: [0xb3,0xd3,0x00,0xf0]
#CHECK: sdtra	%f7, %f8, %f9, 10       # encoding: [0xb3,0xd3,0x9a,0x78]

	sdtra	%f0, %f0, %f0, 0
	sdtra	%f0, %f0, %f0, 15
	sdtra	%f0, %f0, %f15, 0
	sdtra	%f0, %f15, %f0, 0
	sdtra	%f15, %f0, %f0, 0
	sdtra	%f7, %f8, %f9, 10

#CHECK: sgrk	%r0, %r0, %r0           # encoding: [0xb9,0xe9,0x00,0x00]
#CHECK: sgrk	%r0, %r0, %r15          # encoding: [0xb9,0xe9,0xf0,0x00]
#CHECK: sgrk	%r0, %r15, %r0          # encoding: [0xb9,0xe9,0x00,0x0f]
#CHECK: sgrk	%r15, %r0, %r0          # encoding: [0xb9,0xe9,0x00,0xf0]
#CHECK: sgrk	%r7, %r8, %r9           # encoding: [0xb9,0xe9,0x90,0x78]

	sgrk	%r0,%r0,%r0
	sgrk	%r0,%r0,%r15
	sgrk	%r0,%r15,%r0
	sgrk	%r15,%r0,%r0
	sgrk	%r7,%r8,%r9

#CHECK: shhhr	%r0, %r0, %r0           # encoding: [0xb9,0xc9,0x00,0x00]
#CHECK: shhhr	%r0, %r0, %r15          # encoding: [0xb9,0xc9,0xf0,0x00]
#CHECK: shhhr	%r0, %r15, %r0          # encoding: [0xb9,0xc9,0x00,0x0f]
#CHECK: shhhr	%r15, %r0, %r0          # encoding: [0xb9,0xc9,0x00,0xf0]
#CHECK: shhhr	%r7, %r8, %r9           # encoding: [0xb9,0xc9,0x90,0x78]

	shhhr	%r0, %r0, %r0
	shhhr	%r0, %r0, %r15
	shhhr	%r0, %r15, %r0
	shhhr	%r15, %r0, %r0
	shhhr	%r7, %r8, %r9

#CHECK: shhlr	%r0, %r0, %r0           # encoding: [0xb9,0xd9,0x00,0x00]
#CHECK: shhlr	%r0, %r0, %r15          # encoding: [0xb9,0xd9,0xf0,0x00]
#CHECK: shhlr	%r0, %r15, %r0          # encoding: [0xb9,0xd9,0x00,0x0f]
#CHECK: shhlr	%r15, %r0, %r0          # encoding: [0xb9,0xd9,0x00,0xf0]
#CHECK: shhlr	%r7, %r8, %r9           # encoding: [0xb9,0xd9,0x90,0x78]

	shhlr	%r0, %r0, %r0
	shhlr	%r0, %r0, %r15
	shhlr	%r0, %r15, %r0
	shhlr	%r15, %r0, %r0
	shhlr	%r7, %r8, %r9

#CHECK: slak	%r0, %r0, 0             # encoding: [0xeb,0x00,0x00,0x00,0x00,0xdd]
#CHECK: slak	%r15, %r1, 0            # encoding: [0xeb,0xf1,0x00,0x00,0x00,0xdd]
#CHECK: slak	%r1, %r15, 0            # encoding: [0xeb,0x1f,0x00,0x00,0x00,0xdd]
#CHECK: slak	%r15, %r15, 0           # encoding: [0xeb,0xff,0x00,0x00,0x00,0xdd]
#CHECK: slak	%r0, %r0, -524288       # encoding: [0xeb,0x00,0x00,0x00,0x80,0xdd]
#CHECK: slak	%r0, %r0, -1            # encoding: [0xeb,0x00,0x0f,0xff,0xff,0xdd]
#CHECK: slak	%r0, %r0, 1             # encoding: [0xeb,0x00,0x00,0x01,0x00,0xdd]
#CHECK: slak	%r0, %r0, 524287        # encoding: [0xeb,0x00,0x0f,0xff,0x7f,0xdd]
#CHECK: slak	%r0, %r0, 0(%r1)        # encoding: [0xeb,0x00,0x10,0x00,0x00,0xdd]
#CHECK: slak	%r0, %r0, 0(%r15)       # encoding: [0xeb,0x00,0xf0,0x00,0x00,0xdd]
#CHECK: slak	%r0, %r0, 524287(%r1)   # encoding: [0xeb,0x00,0x1f,0xff,0x7f,0xdd]
#CHECK: slak	%r0, %r0, 524287(%r15)  # encoding: [0xeb,0x00,0xff,0xff,0x7f,0xdd]

	slak	%r0,%r0,0
	slak	%r15,%r1,0
	slak	%r1,%r15,0
	slak	%r15,%r15,0
	slak	%r0,%r0,-524288
	slak	%r0,%r0,-1
	slak	%r0,%r0,1
	slak	%r0,%r0,524287
	slak	%r0,%r0,0(%r1)
	slak	%r0,%r0,0(%r15)
	slak	%r0,%r0,524287(%r1)
	slak	%r0,%r0,524287(%r15)

#CHECK: slgrk	%r0, %r0, %r0           # encoding: [0xb9,0xeb,0x00,0x00]
#CHECK: slgrk	%r0, %r0, %r15          # encoding: [0xb9,0xeb,0xf0,0x00]
#CHECK: slgrk	%r0, %r15, %r0          # encoding: [0xb9,0xeb,0x00,0x0f]
#CHECK: slgrk	%r15, %r0, %r0          # encoding: [0xb9,0xeb,0x00,0xf0]
#CHECK: slgrk	%r7, %r8, %r9           # encoding: [0xb9,0xeb,0x90,0x78]

	slgrk	%r0,%r0,%r0
	slgrk	%r0,%r0,%r15
	slgrk	%r0,%r15,%r0
	slgrk	%r15,%r0,%r0
	slgrk	%r7,%r8,%r9

#CHECK: slhhhr	%r0, %r0, %r0           # encoding: [0xb9,0xcb,0x00,0x00]
#CHECK: slhhhr	%r0, %r0, %r15          # encoding: [0xb9,0xcb,0xf0,0x00]
#CHECK: slhhhr	%r0, %r15, %r0          # encoding: [0xb9,0xcb,0x00,0x0f]
#CHECK: slhhhr	%r15, %r0, %r0          # encoding: [0xb9,0xcb,0x00,0xf0]
#CHECK: slhhhr	%r7, %r8, %r9           # encoding: [0xb9,0xcb,0x90,0x78]

	slhhhr	%r0, %r0, %r0
	slhhhr	%r0, %r0, %r15
	slhhhr	%r0, %r15, %r0
	slhhhr	%r15, %r0, %r0
	slhhhr	%r7, %r8, %r9

#CHECK: slhhlr	%r0, %r0, %r0           # encoding: [0xb9,0xdb,0x00,0x00]
#CHECK: slhhlr	%r0, %r0, %r15          # encoding: [0xb9,0xdb,0xf0,0x00]
#CHECK: slhhlr	%r0, %r15, %r0          # encoding: [0xb9,0xdb,0x00,0x0f]
#CHECK: slhhlr	%r15, %r0, %r0          # encoding: [0xb9,0xdb,0x00,0xf0]
#CHECK: slhhlr	%r7, %r8, %r9           # encoding: [0xb9,0xdb,0x90,0x78]

	slhhlr	%r0, %r0, %r0
	slhhlr	%r0, %r0, %r15
	slhhlr	%r0, %r15, %r0
	slhhlr	%r15, %r0, %r0
	slhhlr	%r7, %r8, %r9

#CHECK: sllk	%r0, %r0, 0             # encoding: [0xeb,0x00,0x00,0x00,0x00,0xdf]
#CHECK: sllk	%r15, %r1, 0            # encoding: [0xeb,0xf1,0x00,0x00,0x00,0xdf]
#CHECK: sllk	%r1, %r15, 0            # encoding: [0xeb,0x1f,0x00,0x00,0x00,0xdf]
#CHECK: sllk	%r15, %r15, 0           # encoding: [0xeb,0xff,0x00,0x00,0x00,0xdf]
#CHECK: sllk	%r0, %r0, -524288       # encoding: [0xeb,0x00,0x00,0x00,0x80,0xdf]
#CHECK: sllk	%r0, %r0, -1            # encoding: [0xeb,0x00,0x0f,0xff,0xff,0xdf]
#CHECK: sllk	%r0, %r0, 1             # encoding: [0xeb,0x00,0x00,0x01,0x00,0xdf]
#CHECK: sllk	%r0, %r0, 524287        # encoding: [0xeb,0x00,0x0f,0xff,0x7f,0xdf]
#CHECK: sllk	%r0, %r0, 0(%r1)        # encoding: [0xeb,0x00,0x10,0x00,0x00,0xdf]
#CHECK: sllk	%r0, %r0, 0(%r15)       # encoding: [0xeb,0x00,0xf0,0x00,0x00,0xdf]
#CHECK: sllk	%r0, %r0, 524287(%r1)   # encoding: [0xeb,0x00,0x1f,0xff,0x7f,0xdf]
#CHECK: sllk	%r0, %r0, 524287(%r15)  # encoding: [0xeb,0x00,0xff,0xff,0x7f,0xdf]

	sllk	%r0,%r0,0
	sllk	%r15,%r1,0
	sllk	%r1,%r15,0
	sllk	%r15,%r15,0
	sllk	%r0,%r0,-524288
	sllk	%r0,%r0,-1
	sllk	%r0,%r0,1
	sllk	%r0,%r0,524287
	sllk	%r0,%r0,0(%r1)
	sllk	%r0,%r0,0(%r15)
	sllk	%r0,%r0,524287(%r1)
	sllk	%r0,%r0,524287(%r15)

#CHECK: slrk	%r0, %r0, %r0           # encoding: [0xb9,0xfb,0x00,0x00]
#CHECK: slrk	%r0, %r0, %r15          # encoding: [0xb9,0xfb,0xf0,0x00]
#CHECK: slrk	%r0, %r15, %r0          # encoding: [0xb9,0xfb,0x00,0x0f]
#CHECK: slrk	%r15, %r0, %r0          # encoding: [0xb9,0xfb,0x00,0xf0]
#CHECK: slrk	%r7, %r8, %r9           # encoding: [0xb9,0xfb,0x90,0x78]

	slrk	%r0,%r0,%r0
	slrk	%r0,%r0,%r15
	slrk	%r0,%r15,%r0
	slrk	%r15,%r0,%r0
	slrk	%r7,%r8,%r9

#CHECK: srak	%r0, %r0, 0             # encoding: [0xeb,0x00,0x00,0x00,0x00,0xdc]
#CHECK: srak	%r15, %r1, 0            # encoding: [0xeb,0xf1,0x00,0x00,0x00,0xdc]
#CHECK: srak	%r1, %r15, 0            # encoding: [0xeb,0x1f,0x00,0x00,0x00,0xdc]
#CHECK: srak	%r15, %r15, 0           # encoding: [0xeb,0xff,0x00,0x00,0x00,0xdc]
#CHECK: srak	%r0, %r0, -524288       # encoding: [0xeb,0x00,0x00,0x00,0x80,0xdc]
#CHECK: srak	%r0, %r0, -1            # encoding: [0xeb,0x00,0x0f,0xff,0xff,0xdc]
#CHECK: srak	%r0, %r0, 1             # encoding: [0xeb,0x00,0x00,0x01,0x00,0xdc]
#CHECK: srak	%r0, %r0, 524287        # encoding: [0xeb,0x00,0x0f,0xff,0x7f,0xdc]
#CHECK: srak	%r0, %r0, 0(%r1)        # encoding: [0xeb,0x00,0x10,0x00,0x00,0xdc]
#CHECK: srak	%r0, %r0, 0(%r15)       # encoding: [0xeb,0x00,0xf0,0x00,0x00,0xdc]
#CHECK: srak	%r0, %r0, 524287(%r1)   # encoding: [0xeb,0x00,0x1f,0xff,0x7f,0xdc]
#CHECK: srak	%r0, %r0, 524287(%r15)  # encoding: [0xeb,0x00,0xff,0xff,0x7f,0xdc]

	srak	%r0,%r0,0
	srak	%r15,%r1,0
	srak	%r1,%r15,0
	srak	%r15,%r15,0
	srak	%r0,%r0,-524288
	srak	%r0,%r0,-1
	srak	%r0,%r0,1
	srak	%r0,%r0,524287
	srak	%r0,%r0,0(%r1)
	srak	%r0,%r0,0(%r15)
	srak	%r0,%r0,524287(%r1)
	srak	%r0,%r0,524287(%r15)

#CHECK: srk	%r0, %r0, %r0           # encoding: [0xb9,0xf9,0x00,0x00]
#CHECK: srk	%r0, %r0, %r15          # encoding: [0xb9,0xf9,0xf0,0x00]
#CHECK: srk	%r0, %r15, %r0          # encoding: [0xb9,0xf9,0x00,0x0f]
#CHECK: srk	%r15, %r0, %r0          # encoding: [0xb9,0xf9,0x00,0xf0]
#CHECK: srk	%r7, %r8, %r9           # encoding: [0xb9,0xf9,0x90,0x78]

	srk	%r0,%r0,%r0
	srk	%r0,%r0,%r15
	srk	%r0,%r15,%r0
	srk	%r15,%r0,%r0
	srk	%r7,%r8,%r9

#CHECK: srlk	%r0, %r0, 0             # encoding: [0xeb,0x00,0x00,0x00,0x00,0xde]
#CHECK: srlk	%r15, %r1, 0            # encoding: [0xeb,0xf1,0x00,0x00,0x00,0xde]
#CHECK: srlk	%r1, %r15, 0            # encoding: [0xeb,0x1f,0x00,0x00,0x00,0xde]
#CHECK: srlk	%r15, %r15, 0           # encoding: [0xeb,0xff,0x00,0x00,0x00,0xde]
#CHECK: srlk	%r0, %r0, -524288       # encoding: [0xeb,0x00,0x00,0x00,0x80,0xde]
#CHECK: srlk	%r0, %r0, -1            # encoding: [0xeb,0x00,0x0f,0xff,0xff,0xde]
#CHECK: srlk	%r0, %r0, 1             # encoding: [0xeb,0x00,0x00,0x01,0x00,0xde]
#CHECK: srlk	%r0, %r0, 524287        # encoding: [0xeb,0x00,0x0f,0xff,0x7f,0xde]
#CHECK: srlk	%r0, %r0, 0(%r1)        # encoding: [0xeb,0x00,0x10,0x00,0x00,0xde]
#CHECK: srlk	%r0, %r0, 0(%r15)       # encoding: [0xeb,0x00,0xf0,0x00,0x00,0xde]
#CHECK: srlk	%r0, %r0, 524287(%r1)   # encoding: [0xeb,0x00,0x1f,0xff,0x7f,0xde]
#CHECK: srlk	%r0, %r0, 524287(%r15)  # encoding: [0xeb,0x00,0xff,0xff,0x7f,0xde]

	srlk	%r0,%r0,0
	srlk	%r15,%r1,0
	srlk	%r1,%r15,0
	srlk	%r15,%r15,0
	srlk	%r0,%r0,-524288
	srlk	%r0,%r0,-1
	srlk	%r0,%r0,1
	srlk	%r0,%r0,524287
	srlk	%r0,%r0,0(%r1)
	srlk	%r0,%r0,0(%r15)
	srlk	%r0,%r0,524287(%r1)
	srlk	%r0,%r0,524287(%r15)

#CHECK: srnmb	0                       # encoding: [0xb2,0xb8,0x00,0x00]
#CHECK: srnmb	0(%r1)                  # encoding: [0xb2,0xb8,0x10,0x00]
#CHECK: srnmb	0(%r15)                 # encoding: [0xb2,0xb8,0xf0,0x00]
#CHECK: srnmb	4095                    # encoding: [0xb2,0xb8,0x0f,0xff]
#CHECK: srnmb	4095(%r1)               # encoding: [0xb2,0xb8,0x1f,0xff]
#CHECK: srnmb	4095(%r15)              # encoding: [0xb2,0xb8,0xff,0xff]

	srnmb	0
	srnmb	0(%r1)
	srnmb	0(%r15)
	srnmb	4095
	srnmb	4095(%r1)
	srnmb	4095(%r15)

#CHECK: stch	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0xc3]
#CHECK: stch	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0xc3]
#CHECK: stch	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0xc3]
#CHECK: stch	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0xc3]
#CHECK: stch	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0xc3]
#CHECK: stch	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0xc3]
#CHECK: stch	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0xc3]
#CHECK: stch	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0xc3]
#CHECK: stch	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0xc3]
#CHECK: stch	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0xc3]

	stch	%r0, -524288
	stch	%r0, -1
	stch	%r0, 0
	stch	%r0, 1
	stch	%r0, 524287
	stch	%r0, 0(%r1)
	stch	%r0, 0(%r15)
	stch	%r0, 524287(%r1,%r15)
	stch	%r0, 524287(%r15,%r1)
	stch	%r15, 0

#CHECK: stfh	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0xcb]
#CHECK: stfh	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0xcb]
#CHECK: stfh	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0xcb]
#CHECK: stfh	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0xcb]
#CHECK: stfh	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0xcb]
#CHECK: stfh	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0xcb]
#CHECK: stfh	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0xcb]
#CHECK: stfh	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0xcb]
#CHECK: stfh	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0xcb]
#CHECK: stfh	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0xcb]

	stfh	%r0, -524288
	stfh	%r0, -1
	stfh	%r0, 0
	stfh	%r0, 1
	stfh	%r0, 524287
	stfh	%r0, 0(%r1)
	stfh	%r0, 0(%r15)
	stfh	%r0, 524287(%r1,%r15)
	stfh	%r0, 524287(%r15,%r1)
	stfh	%r15, 0

#CHECK: sthh	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0xc7]
#CHECK: sthh	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0xc7]
#CHECK: sthh	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0xc7]
#CHECK: sthh	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0xc7]
#CHECK: sthh	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0xc7]
#CHECK: sthh	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0xc7]
#CHECK: sthh	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0xc7]
#CHECK: sthh	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0xc7]
#CHECK: sthh	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0xc7]
#CHECK: sthh	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0xc7]

	sthh	%r0, -524288
	sthh	%r0, -1
	sthh	%r0, 0
	sthh	%r0, 1
	sthh	%r0, 524287
	sthh	%r0, 0(%r1)
	sthh	%r0, 0(%r15)
	sthh	%r0, 524287(%r1,%r15)
	sthh	%r0, 524287(%r15,%r1)
	sthh	%r15, 0

#CHECK: stoc	%r0, 0, 0               # encoding: [0xeb,0x00,0x00,0x00,0x00,0xf3]
#CHECK: stoc	%r0, 0, 15              # encoding: [0xeb,0x0f,0x00,0x00,0x00,0xf3]
#CHECK: stoc	%r0, -524288, 0         # encoding: [0xeb,0x00,0x00,0x00,0x80,0xf3]
#CHECK: stoc	%r0, 524287, 0          # encoding: [0xeb,0x00,0x0f,0xff,0x7f,0xf3]
#CHECK: stoc	%r0, 0(%r1), 0          # encoding: [0xeb,0x00,0x10,0x00,0x00,0xf3]
#CHECK: stoc	%r0, 0(%r15), 0         # encoding: [0xeb,0x00,0xf0,0x00,0x00,0xf3]
#CHECK: stoc	%r15, 0, 0              # encoding: [0xeb,0xf0,0x00,0x00,0x00,0xf3]
#CHECK: stoc	%r1, 4095(%r2), 3       # encoding: [0xeb,0x13,0x2f,0xff,0x00,0xf3]

	stoc	%r0,0,0
	stoc	%r0,0,15
	stoc	%r0,-524288,0
	stoc	%r0,524287,0
	stoc	%r0,0(%r1),0
	stoc	%r0,0(%r15),0
	stoc	%r15,0,0
	stoc	%r1,4095(%r2),3

#CHECK: stoco   %r1, 2(%r3)             # encoding: [0xeb,0x11,0x30,0x02,0x00,0xf3]
#CHECK: stoch   %r1, 2(%r3)             # encoding: [0xeb,0x12,0x30,0x02,0x00,0xf3]
#CHECK: stocp   %r1, 2(%r3)             # encoding: [0xeb,0x12,0x30,0x02,0x00,0xf3]
#CHECK: stocnle %r1, 2(%r3)             # encoding: [0xeb,0x13,0x30,0x02,0x00,0xf3]
#CHECK: stocl   %r1, 2(%r3)             # encoding: [0xeb,0x14,0x30,0x02,0x00,0xf3]
#CHECK: stocm   %r1, 2(%r3)             # encoding: [0xeb,0x14,0x30,0x02,0x00,0xf3]
#CHECK: stocnhe %r1, 2(%r3)             # encoding: [0xeb,0x15,0x30,0x02,0x00,0xf3]
#CHECK: stoclh  %r1, 2(%r3)             # encoding: [0xeb,0x16,0x30,0x02,0x00,0xf3]
#CHECK: stocne  %r1, 2(%r3)             # encoding: [0xeb,0x17,0x30,0x02,0x00,0xf3]
#CHECK: stocnz  %r1, 2(%r3)             # encoding: [0xeb,0x17,0x30,0x02,0x00,0xf3]
#CHECK: stoce   %r1, 2(%r3)             # encoding: [0xeb,0x18,0x30,0x02,0x00,0xf3]
#CHECK: stocz   %r1, 2(%r3)             # encoding: [0xeb,0x18,0x30,0x02,0x00,0xf3]
#CHECK: stocnlh %r1, 2(%r3)             # encoding: [0xeb,0x19,0x30,0x02,0x00,0xf3]
#CHECK: stoche  %r1, 2(%r3)             # encoding: [0xeb,0x1a,0x30,0x02,0x00,0xf3]
#CHECK: stocnl  %r1, 2(%r3)             # encoding: [0xeb,0x1b,0x30,0x02,0x00,0xf3]
#CHECK: stocnm  %r1, 2(%r3)             # encoding: [0xeb,0x1b,0x30,0x02,0x00,0xf3]
#CHECK: stocle  %r1, 2(%r3)             # encoding: [0xeb,0x1c,0x30,0x02,0x00,0xf3]
#CHECK: stocnh  %r1, 2(%r3)             # encoding: [0xeb,0x1d,0x30,0x02,0x00,0xf3]
#CHECK: stocnp  %r1, 2(%r3)             # encoding: [0xeb,0x1d,0x30,0x02,0x00,0xf3]
#CHECK: stocno  %r1, 2(%r3)             # encoding: [0xeb,0x1e,0x30,0x02,0x00,0xf3]

	stoco   %r1,2(%r3)
	stoch   %r1,2(%r3)
	stocp   %r1,2(%r3)
	stocnle %r1,2(%r3)
	stocl   %r1,2(%r3)
	stocm   %r1,2(%r3)
	stocnhe %r1,2(%r3)
	stoclh  %r1,2(%r3)
	stocne  %r1,2(%r3)
	stocnz  %r1,2(%r3)
	stoce   %r1,2(%r3)
	stocz   %r1,2(%r3)
	stocnlh %r1,2(%r3)
	stoche  %r1,2(%r3)
	stocnl  %r1,2(%r3)
	stocnm  %r1,2(%r3)
	stocle  %r1,2(%r3)
	stocnh  %r1,2(%r3)
	stocnp  %r1,2(%r3)
	stocno  %r1,2(%r3)

#CHECK: stocg	%r0, 0, 0               # encoding: [0xeb,0x00,0x00,0x00,0x00,0xe3]
#CHECK: stocg	%r0, 0, 15              # encoding: [0xeb,0x0f,0x00,0x00,0x00,0xe3]
#CHECK: stocg	%r0, -524288, 0         # encoding: [0xeb,0x00,0x00,0x00,0x80,0xe3]
#CHECK: stocg	%r0, 524287, 0          # encoding: [0xeb,0x00,0x0f,0xff,0x7f,0xe3]
#CHECK: stocg	%r0, 0(%r1), 0          # encoding: [0xeb,0x00,0x10,0x00,0x00,0xe3]
#CHECK: stocg	%r0, 0(%r15), 0         # encoding: [0xeb,0x00,0xf0,0x00,0x00,0xe3]
#CHECK: stocg	%r15, 0, 0              # encoding: [0xeb,0xf0,0x00,0x00,0x00,0xe3]
#CHECK: stocg	%r1, 4095(%r2), 3       # encoding: [0xeb,0x13,0x2f,0xff,0x00,0xe3]

	stocg	%r0,0,0
	stocg	%r0,0,15
	stocg	%r0,-524288,0
	stocg	%r0,524287,0
	stocg	%r0,0(%r1),0
	stocg	%r0,0(%r15),0
	stocg	%r15,0,0
	stocg	%r1,4095(%r2),3

#CHECK: stocgo   %r1, 2(%r3)            # encoding: [0xeb,0x11,0x30,0x02,0x00,0xe3]
#CHECK: stocgh   %r1, 2(%r3)            # encoding: [0xeb,0x12,0x30,0x02,0x00,0xe3]
#CHECK: stocgp   %r1, 2(%r3)            # encoding: [0xeb,0x12,0x30,0x02,0x00,0xe3]
#CHECK: stocgnle %r1, 2(%r3)            # encoding: [0xeb,0x13,0x30,0x02,0x00,0xe3]
#CHECK: stocgl   %r1, 2(%r3)            # encoding: [0xeb,0x14,0x30,0x02,0x00,0xe3]
#CHECK: stocgm   %r1, 2(%r3)            # encoding: [0xeb,0x14,0x30,0x02,0x00,0xe3]
#CHECK: stocgnhe %r1, 2(%r3)            # encoding: [0xeb,0x15,0x30,0x02,0x00,0xe3]
#CHECK: stocglh  %r1, 2(%r3)            # encoding: [0xeb,0x16,0x30,0x02,0x00,0xe3]
#CHECK: stocgne  %r1, 2(%r3)            # encoding: [0xeb,0x17,0x30,0x02,0x00,0xe3]
#CHECK: stocgnz  %r1, 2(%r3)            # encoding: [0xeb,0x17,0x30,0x02,0x00,0xe3]
#CHECK: stocge   %r1, 2(%r3)            # encoding: [0xeb,0x18,0x30,0x02,0x00,0xe3]
#CHECK: stocgz   %r1, 2(%r3)            # encoding: [0xeb,0x18,0x30,0x02,0x00,0xe3]
#CHECK: stocgnlh %r1, 2(%r3)            # encoding: [0xeb,0x19,0x30,0x02,0x00,0xe3]
#CHECK: stocghe  %r1, 2(%r3)            # encoding: [0xeb,0x1a,0x30,0x02,0x00,0xe3]
#CHECK: stocgnl  %r1, 2(%r3)            # encoding: [0xeb,0x1b,0x30,0x02,0x00,0xe3]
#CHECK: stocgnm  %r1, 2(%r3)            # encoding: [0xeb,0x1b,0x30,0x02,0x00,0xe3]
#CHECK: stocgle  %r1, 2(%r3)            # encoding: [0xeb,0x1c,0x30,0x02,0x00,0xe3]
#CHECK: stocgnh  %r1, 2(%r3)            # encoding: [0xeb,0x1d,0x30,0x02,0x00,0xe3]
#CHECK: stocgnp  %r1, 2(%r3)            # encoding: [0xeb,0x1d,0x30,0x02,0x00,0xe3]
#CHECK: stocgno  %r1, 2(%r3)            # encoding: [0xeb,0x1e,0x30,0x02,0x00,0xe3]

	stocgo   %r1,2(%r3)
	stocgh   %r1,2(%r3)
	stocgp   %r1,2(%r3)
	stocgnle %r1,2(%r3)
	stocgl   %r1,2(%r3)
	stocgm   %r1,2(%r3)
	stocgnhe %r1,2(%r3)
	stocglh  %r1,2(%r3)
	stocgne  %r1,2(%r3)
	stocgnz  %r1,2(%r3)
	stocge   %r1,2(%r3)
	stocgz   %r1,2(%r3)
	stocgnlh %r1,2(%r3)
	stocghe  %r1,2(%r3)
	stocgnl  %r1,2(%r3)
	stocgnm  %r1,2(%r3)
	stocgle  %r1,2(%r3)
	stocgnh  %r1,2(%r3)
	stocgnp  %r1,2(%r3)
	stocgno  %r1,2(%r3)

#CHECK: sxtra	%f0, %f0, %f0, 0        # encoding: [0xb3,0xdb,0x00,0x00]
#CHECK: sxtra	%f0, %f0, %f0, 15       # encoding: [0xb3,0xdb,0x0f,0x00]
#CHECK: sxtra	%f0, %f0, %f13, 0       # encoding: [0xb3,0xdb,0xd0,0x00]
#CHECK: sxtra	%f0, %f13, %f0, 0       # encoding: [0xb3,0xdb,0x00,0x0d]
#CHECK: sxtra	%f13, %f0, %f0, 0       # encoding: [0xb3,0xdb,0x00,0xd0]
#CHECK: sxtra	%f8, %f8, %f8, 8        # encoding: [0xb3,0xdb,0x88,0x88]

	sxtra	%f0, %f0, %f0, 0
	sxtra	%f0, %f0, %f0, 15
	sxtra	%f0, %f0, %f13, 0
	sxtra	%f0, %f13, %f0, 0
	sxtra	%f13, %f0, %f0, 0
	sxtra	%f8, %f8, %f8, 8

#CHECK: xgrk	%r0, %r0, %r0           # encoding: [0xb9,0xe7,0x00,0x00]
#CHECK: xgrk	%r0, %r0, %r15          # encoding: [0xb9,0xe7,0xf0,0x00]
#CHECK: xgrk	%r0, %r15, %r0          # encoding: [0xb9,0xe7,0x00,0x0f]
#CHECK: xgrk	%r15, %r0, %r0          # encoding: [0xb9,0xe7,0x00,0xf0]
#CHECK: xgrk	%r7, %r8, %r9           # encoding: [0xb9,0xe7,0x90,0x78]

	xgrk	%r0,%r0,%r0
	xgrk	%r0,%r0,%r15
	xgrk	%r0,%r15,%r0
	xgrk	%r15,%r0,%r0
	xgrk	%r7,%r8,%r9

#CHECK: xrk	%r0, %r0, %r0           # encoding: [0xb9,0xf7,0x00,0x00]
#CHECK: xrk	%r0, %r0, %r15          # encoding: [0xb9,0xf7,0xf0,0x00]
#CHECK: xrk	%r0, %r15, %r0          # encoding: [0xb9,0xf7,0x00,0x0f]
#CHECK: xrk	%r15, %r0, %r0          # encoding: [0xb9,0xf7,0x00,0xf0]
#CHECK: xrk	%r7, %r8, %r9           # encoding: [0xb9,0xf7,0x90,0x78]

	xrk	%r0,%r0,%r0
	xrk	%r0,%r0,%r15
	xrk	%r0,%r15,%r0
	xrk	%r15,%r0,%r0
	xrk	%r7,%r8,%r9
