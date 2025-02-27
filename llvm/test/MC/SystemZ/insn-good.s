# For z10 and above.
# RUN: llvm-mc -triple s390x-linux-gnu -show-encoding %s | FileCheck %s

#CHECK: a	%r0, 0                  # encoding: [0x5a,0x00,0x00,0x00]
#CHECK: a	%r0, 4095               # encoding: [0x5a,0x00,0x0f,0xff]
#CHECK: a	%r0, 0(%r1)             # encoding: [0x5a,0x00,0x10,0x00]
#CHECK: a	%r0, 0(%r15)            # encoding: [0x5a,0x00,0xf0,0x00]
#CHECK: a	%r0, 4095(%r1,%r15)     # encoding: [0x5a,0x01,0xff,0xff]
#CHECK: a	%r0, 4095(%r15,%r1)     # encoding: [0x5a,0x0f,0x1f,0xff]
#CHECK: a	%r15, 0                 # encoding: [0x5a,0xf0,0x00,0x00]
#CHECK: a	%r0, 0                  # encoding: [0x5a,0x00,0x00,0x00]
#CHECK: a	%r0, 4095               # encoding: [0x5a,0x00,0x0f,0xff]
#CHECK: a	%r0, 0(%r1)             # encoding: [0x5a,0x00,0x10,0x00]
#CHECK: a	%r0, 0(%r15)            # encoding: [0x5a,0x00,0xf0,0x00]
#CHECK: a	%r0, 4095(%r1,%r15)     # encoding: [0x5a,0x01,0xff,0xff]
#CHECK: a	%r0, 4095(%r15,%r1)     # encoding: [0x5a,0x0f,0x1f,0xff]
#CHECK: a	%r15, 0                 # encoding: [0x5a,0xf0,0x00,0x00]

	a	%r0, 0
	a	%r0, 4095
	a	%r0, 0(%r1)
	a	%r0, 0(%r15)
	a	%r0, 4095(%r1,%r15)
	a	%r0, 4095(%r15,%r1)
	a	%r15, 0
	a	0, 0
	a	0, 4095
	a	0, 0(1)
	a	0, 0(15)
	a	0, 4095(1,15)
	a	0, 4095(15,1)
	a	15, 0


#CHECK: ad	%f0, 0                  # encoding: [0x6a,0x00,0x00,0x00]
#CHECK: ad	%f0, 4095               # encoding: [0x6a,0x00,0x0f,0xff]
#CHECK: ad	%f0, 0(%r1)             # encoding: [0x6a,0x00,0x10,0x00]
#CHECK: ad	%f0, 0(%r15)            # encoding: [0x6a,0x00,0xf0,0x00]
#CHECK: ad	%f0, 4095(%r1,%r15)     # encoding: [0x6a,0x01,0xff,0xff]
#CHECK: ad	%f0, 4095(%r15,%r1)     # encoding: [0x6a,0x0f,0x1f,0xff]
#CHECK: ad	%f15, 0                 # encoding: [0x6a,0xf0,0x00,0x00]

	ad	%f0, 0
	ad	%f0, 4095
	ad	%f0, 0(%r1)
	ad	%f0, 0(%r15)
	ad	%f0, 4095(%r1,%r15)
	ad	%f0, 4095(%r15,%r1)
	ad	%f15, 0

#CHECK: adb	%f0, 0                  # encoding: [0xed,0x00,0x00,0x00,0x00,0x1a]
#CHECK: adb	%f0, 4095               # encoding: [0xed,0x00,0x0f,0xff,0x00,0x1a]
#CHECK: adb	%f0, 0(%r1)             # encoding: [0xed,0x00,0x10,0x00,0x00,0x1a]
#CHECK: adb	%f0, 0(%r15)            # encoding: [0xed,0x00,0xf0,0x00,0x00,0x1a]
#CHECK: adb	%f0, 4095(%r1,%r15)     # encoding: [0xed,0x01,0xff,0xff,0x00,0x1a]
#CHECK: adb	%f0, 4095(%r15,%r1)     # encoding: [0xed,0x0f,0x1f,0xff,0x00,0x1a]
#CHECK: adb	%f15, 0                 # encoding: [0xed,0xf0,0x00,0x00,0x00,0x1a]

	adb	%f0, 0
	adb	%f0, 4095
	adb	%f0, 0(%r1)
	adb	%f0, 0(%r15)
	adb	%f0, 4095(%r1,%r15)
	adb	%f0, 4095(%r15,%r1)
	adb	%f15, 0

#CHECK: adbr	%f0, %f0                # encoding: [0xb3,0x1a,0x00,0x00]
#CHECK: adbr	%f0, %f15               # encoding: [0xb3,0x1a,0x00,0x0f]
#CHECK: adbr	%f7, %f8                # encoding: [0xb3,0x1a,0x00,0x78]
#CHECK: adbr	%f15, %f0               # encoding: [0xb3,0x1a,0x00,0xf0]

	adbr	%f0, %f0
	adbr	%f0, %f15
	adbr	%f7, %f8
	adbr	%f15, %f0

#CHECK: adr	%f0, %f0                # encoding: [0x2a,0x00]
#CHECK: adr	%f0, %f15               # encoding: [0x2a,0x0f]
#CHECK: adr	%f7, %f8                # encoding: [0x2a,0x78]
#CHECK: adr	%f15, %f0               # encoding: [0x2a,0xf0]

	adr	%f0, %f0
	adr	%f0, %f15
	adr	%f7, %f8
	adr	%f15, %f0

#CHECK: adtr	%f0, %f0, %f0           # encoding: [0xb3,0xd2,0x00,0x00]
#CHECK: adtr	%f0, %f0, %f15          # encoding: [0xb3,0xd2,0xf0,0x00]
#CHECK: adtr	%f0, %f15, %f0          # encoding: [0xb3,0xd2,0x00,0x0f]
#CHECK: adtr	%f15, %f0, %f0          # encoding: [0xb3,0xd2,0x00,0xf0]
#CHECK: adtr	%f7, %f8, %f9           # encoding: [0xb3,0xd2,0x90,0x78]

	adtr	%f0, %f0, %f0
	adtr	%f0, %f0, %f15
	adtr	%f0, %f15, %f0
	adtr	%f15, %f0, %f0
	adtr	%f7, %f8, %f9

#CHECK: ae	%f0, 0                  # encoding: [0x7a,0x00,0x00,0x00]
#CHECK: ae	%f0, 4095               # encoding: [0x7a,0x00,0x0f,0xff]
#CHECK: ae	%f0, 0(%r1)             # encoding: [0x7a,0x00,0x10,0x00]
#CHECK: ae	%f0, 0(%r15)            # encoding: [0x7a,0x00,0xf0,0x00]
#CHECK: ae	%f0, 4095(%r1,%r15)     # encoding: [0x7a,0x01,0xff,0xff]
#CHECK: ae	%f0, 4095(%r15,%r1)     # encoding: [0x7a,0x0f,0x1f,0xff]
#CHECK: ae	%f15, 0                 # encoding: [0x7a,0xf0,0x00,0x00]

	ae	%f0, 0
	ae	%f0, 4095
	ae	%f0, 0(%r1)
	ae	%f0, 0(%r15)
	ae	%f0, 4095(%r1,%r15)
	ae	%f0, 4095(%r15,%r1)
	ae	%f15, 0

#CHECK: aeb	%f0, 0                  # encoding: [0xed,0x00,0x00,0x00,0x00,0x0a]
#CHECK: aeb	%f0, 4095               # encoding: [0xed,0x00,0x0f,0xff,0x00,0x0a]
#CHECK: aeb	%f0, 0(%r1)             # encoding: [0xed,0x00,0x10,0x00,0x00,0x0a]
#CHECK: aeb	%f0, 0(%r15)            # encoding: [0xed,0x00,0xf0,0x00,0x00,0x0a]
#CHECK: aeb	%f0, 4095(%r1,%r15)     # encoding: [0xed,0x01,0xff,0xff,0x00,0x0a]
#CHECK: aeb	%f0, 4095(%r15,%r1)     # encoding: [0xed,0x0f,0x1f,0xff,0x00,0x0a]
#CHECK: aeb	%f15, 0                 # encoding: [0xed,0xf0,0x00,0x00,0x00,0x0a]

	aeb	%f0, 0
	aeb	%f0, 4095
	aeb	%f0, 0(%r1)
	aeb	%f0, 0(%r15)
	aeb	%f0, 4095(%r1,%r15)
	aeb	%f0, 4095(%r15,%r1)
	aeb	%f15, 0

#CHECK: aebr	%f0, %f0                # encoding: [0xb3,0x0a,0x00,0x00]
#CHECK: aebr	%f0, %f15               # encoding: [0xb3,0x0a,0x00,0x0f]
#CHECK: aebr	%f7, %f8                # encoding: [0xb3,0x0a,0x00,0x78]
#CHECK: aebr	%f15, %f0               # encoding: [0xb3,0x0a,0x00,0xf0]

	aebr	%f0, %f0
	aebr	%f0, %f15
	aebr	%f7, %f8
	aebr	%f15, %f0

#CHECK: aer	%f0, %f0                # encoding: [0x3a,0x00]
#CHECK: aer	%f0, %f15               # encoding: [0x3a,0x0f]
#CHECK: aer	%f7, %f8                # encoding: [0x3a,0x78]
#CHECK: aer	%f15, %f0               # encoding: [0x3a,0xf0]

	aer	%f0, %f0
	aer	%f0, %f15
	aer	%f7, %f8
	aer	%f15, %f0

#CHECK: afi	%r0, -2147483648        # encoding: [0xc2,0x09,0x80,0x00,0x00,0x00]
#CHECK: afi	%r0, -1                 # encoding: [0xc2,0x09,0xff,0xff,0xff,0xff]
#CHECK: afi	%r0, 0                  # encoding: [0xc2,0x09,0x00,0x00,0x00,0x00]
#CHECK: afi	%r0, 1                  # encoding: [0xc2,0x09,0x00,0x00,0x00,0x01]
#CHECK: afi	%r0, 2147483647         # encoding: [0xc2,0x09,0x7f,0xff,0xff,0xff]
#CHECK: afi	%r15, 0                 # encoding: [0xc2,0xf9,0x00,0x00,0x00,0x00]

	afi	%r0, -1 << 31
	afi	%r0, -1
	afi	%r0, 0
	afi	%r0, 1
	afi	%r0, (1 << 31) - 1
	afi	%r15, 0

#CHECK: ag	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x08]
#CHECK: ag	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x08]
#CHECK: ag	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x08]
#CHECK: ag	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x08]
#CHECK: ag	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x08]
#CHECK: ag	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x08]
#CHECK: ag	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x08]
#CHECK: ag	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x08]
#CHECK: ag	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x08]
#CHECK: ag	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x08]

	ag	%r0, -524288
	ag	%r0, -1
	ag	%r0, 0
	ag	%r0, 1
	ag	%r0, 524287
	ag	%r0, 0(%r1)
	ag	%r0, 0(%r15)
	ag	%r0, 524287(%r1,%r15)
	ag	%r0, 524287(%r15,%r1)
	ag	%r15, 0

#CHECK: agf	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x18]
#CHECK: agf	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x18]
#CHECK: agf	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x18]
#CHECK: agf	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x18]
#CHECK: agf	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x18]
#CHECK: agf	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x18]
#CHECK: agf	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x18]
#CHECK: agf	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x18]
#CHECK: agf	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x18]
#CHECK: agf	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x18]

	agf	%r0, -524288
	agf	%r0, -1
	agf	%r0, 0
	agf	%r0, 1
	agf	%r0, 524287
	agf	%r0, 0(%r1)
	agf	%r0, 0(%r15)
	agf	%r0, 524287(%r1,%r15)
	agf	%r0, 524287(%r15,%r1)
	agf	%r15, 0

#CHECK: agfi	%r0, -2147483648        # encoding: [0xc2,0x08,0x80,0x00,0x00,0x00]
#CHECK: agfi	%r0, -1                 # encoding: [0xc2,0x08,0xff,0xff,0xff,0xff]
#CHECK: agfi	%r0, 0                  # encoding: [0xc2,0x08,0x00,0x00,0x00,0x00]
#CHECK: agfi	%r0, 1                  # encoding: [0xc2,0x08,0x00,0x00,0x00,0x01]
#CHECK: agfi	%r0, 2147483647         # encoding: [0xc2,0x08,0x7f,0xff,0xff,0xff]
#CHECK: agfi	%r15, 0                 # encoding: [0xc2,0xf8,0x00,0x00,0x00,0x00]

	agfi	%r0, -1 << 31
	agfi	%r0, -1
	agfi	%r0, 0
	agfi	%r0, 1
	agfi	%r0, (1 << 31) - 1
	agfi	%r15, 0

#CHECK: agfr	%r0, %r0                # encoding: [0xb9,0x18,0x00,0x00]
#CHECK: agfr	%r0, %r15               # encoding: [0xb9,0x18,0x00,0x0f]
#CHECK: agfr	%r15, %r0               # encoding: [0xb9,0x18,0x00,0xf0]
#CHECK: agfr	%r7, %r8                # encoding: [0xb9,0x18,0x00,0x78]

	agfr	%r0,%r0
	agfr	%r0,%r15
	agfr	%r15,%r0
	agfr	%r7,%r8

#CHECK: aghi	%r0, -32768             # encoding: [0xa7,0x0b,0x80,0x00]
#CHECK: aghi	%r0, -1                 # encoding: [0xa7,0x0b,0xff,0xff]
#CHECK: aghi	%r0, 0                  # encoding: [0xa7,0x0b,0x00,0x00]
#CHECK: aghi	%r0, 1                  # encoding: [0xa7,0x0b,0x00,0x01]
#CHECK: aghi	%r0, 32767              # encoding: [0xa7,0x0b,0x7f,0xff]
#CHECK: aghi	%r15, 0                 # encoding: [0xa7,0xfb,0x00,0x00]
#CHECK: aghi	%r0, foo                # encoding: [0xa7,0x0b,A,A]

	aghi	%r0, -32768
	aghi	%r0, -1
	aghi	%r0, 0
	aghi	%r0, 1
	aghi	%r0, 32767
	aghi	%r15, 0
	aghi	%r0, foo

#CHECK: agr	%r0, %r0                # encoding: [0xb9,0x08,0x00,0x00]
#CHECK: agr	%r0, %r15               # encoding: [0xb9,0x08,0x00,0x0f]
#CHECK: agr	%r15, %r0               # encoding: [0xb9,0x08,0x00,0xf0]
#CHECK: agr	%r7, %r8                # encoding: [0xb9,0x08,0x00,0x78]

	agr	%r0,%r0
	agr	%r0,%r15
	agr	%r15,%r0
	agr	%r7,%r8

#CHECK: agsi	-524288, 0              # encoding: [0xeb,0x00,0x00,0x00,0x80,0x7a]
#CHECK: agsi	-1, 0                   # encoding: [0xeb,0x00,0x0f,0xff,0xff,0x7a]
#CHECK: agsi	0, 0                    # encoding: [0xeb,0x00,0x00,0x00,0x00,0x7a]
#CHECK: agsi	1, 0                    # encoding: [0xeb,0x00,0x00,0x01,0x00,0x7a]
#CHECK: agsi	524287, 0               # encoding: [0xeb,0x00,0x0f,0xff,0x7f,0x7a]
#CHECK: agsi	0, -128                 # encoding: [0xeb,0x80,0x00,0x00,0x00,0x7a]
#CHECK: agsi	0, -1                   # encoding: [0xeb,0xff,0x00,0x00,0x00,0x7a]
#CHECK: agsi	0, 1                    # encoding: [0xeb,0x01,0x00,0x00,0x00,0x7a]
#CHECK: agsi	0, 127                  # encoding: [0xeb,0x7f,0x00,0x00,0x00,0x7a]
#CHECK: agsi	0(%r1), 42              # encoding: [0xeb,0x2a,0x10,0x00,0x00,0x7a]
#CHECK: agsi	0(%r15), 42             # encoding: [0xeb,0x2a,0xf0,0x00,0x00,0x7a]
#CHECK: agsi	524287(%r1), 42         # encoding: [0xeb,0x2a,0x1f,0xff,0x7f,0x7a]
#CHECK: agsi	524287(%r15), 42        # encoding: [0xeb,0x2a,0xff,0xff,0x7f,0x7a]

	agsi	-524288, 0
	agsi	-1, 0
	agsi	0, 0
	agsi	1, 0
	agsi	524287, 0
	agsi	0, -128
	agsi	0, -1
	agsi	0, 1
	agsi	0, 127
	agsi	0(%r1), 42
	agsi	0(%r15), 42
	agsi	524287(%r1), 42
	agsi	524287(%r15), 42

#CHECK: ah	%r0, 0                  # encoding: [0x4a,0x00,0x00,0x00]
#CHECK: ah	%r0, 4095               # encoding: [0x4a,0x00,0x0f,0xff]
#CHECK: ah	%r0, 0(%r1)             # encoding: [0x4a,0x00,0x10,0x00]
#CHECK: ah	%r0, 0(%r15)            # encoding: [0x4a,0x00,0xf0,0x00]
#CHECK: ah	%r0, 4095(%r1,%r15)     # encoding: [0x4a,0x01,0xff,0xff]
#CHECK: ah	%r0, 4095(%r15,%r1)     # encoding: [0x4a,0x0f,0x1f,0xff]
#CHECK: ah	%r15, 0                 # encoding: [0x4a,0xf0,0x00,0x00]

	ah	%r0, 0
	ah	%r0, 4095
	ah	%r0, 0(%r1)
	ah	%r0, 0(%r15)
	ah	%r0, 4095(%r1,%r15)
	ah	%r0, 4095(%r15,%r1)
	ah	%r15, 0

#CHECK: ahi	%r0, -32768             # encoding: [0xa7,0x0a,0x80,0x00]
#CHECK: ahi	%r0, -1                 # encoding: [0xa7,0x0a,0xff,0xff]
#CHECK: ahi	%r0, 0                  # encoding: [0xa7,0x0a,0x00,0x00]
#CHECK: ahi	%r0, 1                  # encoding: [0xa7,0x0a,0x00,0x01]
#CHECK: ahi	%r0, 32767              # encoding: [0xa7,0x0a,0x7f,0xff]
#CHECK: ahi	%r15, 0                 # encoding: [0xa7,0xfa,0x00,0x00]
#CHECK: ahi	%r0, foo                # encoding: [0xa7,0x0a,A,A]

	ahi	%r0, -32768
	ahi	%r0, -1
	ahi	%r0, 0
	ahi	%r0, 1
	ahi	%r0, 32767
	ahi	%r15, 0
	ahi	%r0, foo

#CHECK: ahy	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x7a]
#CHECK: ahy	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x7a]
#CHECK: ahy	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x7a]
#CHECK: ahy	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x7a]
#CHECK: ahy	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x7a]
#CHECK: ahy	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x7a]
#CHECK: ahy	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x7a]
#CHECK: ahy	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x7a]
#CHECK: ahy	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x7a]
#CHECK: ahy	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x7a]

	ahy	%r0, -524288
	ahy	%r0, -1
	ahy	%r0, 0
	ahy	%r0, 1
	ahy	%r0, 524287
	ahy	%r0, 0(%r1)
	ahy	%r0, 0(%r15)
	ahy	%r0, 524287(%r1,%r15)
	ahy	%r0, 524287(%r15,%r1)
	ahy	%r15, 0

#CHECK: al	%r0, 0                  # encoding: [0x5e,0x00,0x00,0x00]
#CHECK: al	%r0, 4095               # encoding: [0x5e,0x00,0x0f,0xff]
#CHECK: al	%r0, 0(%r1)             # encoding: [0x5e,0x00,0x10,0x00]
#CHECK: al	%r0, 0(%r15)            # encoding: [0x5e,0x00,0xf0,0x00]
#CHECK: al	%r0, 4095(%r1,%r15)     # encoding: [0x5e,0x01,0xff,0xff]
#CHECK: al	%r0, 4095(%r15,%r1)     # encoding: [0x5e,0x0f,0x1f,0xff]
#CHECK: al	%r15, 0                 # encoding: [0x5e,0xf0,0x00,0x00]
#CHECK: al	%r0, 0                  # encoding: [0x5e,0x00,0x00,0x00]
#CHECK: al	%r0, 4095               # encoding: [0x5e,0x00,0x0f,0xff]
#CHECK: al	%r0, 0(%r1)             # encoding: [0x5e,0x00,0x10,0x00]
#CHECK: al	%r0, 0(%r15)            # encoding: [0x5e,0x00,0xf0,0x00]
#CHECK: al	%r0, 4095(%r1,%r15)     # encoding: [0x5e,0x01,0xff,0xff]
#CHECK: al	%r0, 4095(%r15,%r1)     # encoding: [0x5e,0x0f,0x1f,0xff]
#CHECK: al	%r15, 0                 # encoding: [0x5e,0xf0,0x00,0x00]

	al	%r0, 0
	al	%r0, 4095
	al	%r0, 0(%r1)
	al	%r0, 0(%r15)
	al	%r0, 4095(%r1,%r15)
	al	%r0, 4095(%r15,%r1)
	al	%r15, 0
	al	0, 0
	al	0, 4095
	al	0, 0(1)
	al	0, 0(15)
	al	0, 4095(1,15)
	al	0, 4095(15,1)
	al	15, 0

#CHECK: alc	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x98]
#CHECK: alc	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x98]
#CHECK: alc	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x98]
#CHECK: alc	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x98]
#CHECK: alc	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x98]
#CHECK: alc	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x98]
#CHECK: alc	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x98]
#CHECK: alc	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x98]
#CHECK: alc	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x98]
#CHECK: alc	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x98]

	alc	%r0, -524288
	alc	%r0, -1
	alc	%r0, 0
	alc	%r0, 1
	alc	%r0, 524287
	alc	%r0, 0(%r1)
	alc	%r0, 0(%r15)
	alc	%r0, 524287(%r1,%r15)
	alc	%r0, 524287(%r15,%r1)
	alc	%r15, 0

#CHECK: alcg	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x88]
#CHECK: alcg	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x88]
#CHECK: alcg	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x88]
#CHECK: alcg	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x88]
#CHECK: alcg	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x88]
#CHECK: alcg	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x88]
#CHECK: alcg	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x88]
#CHECK: alcg	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x88]
#CHECK: alcg	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x88]
#CHECK: alcg	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x88]

	alcg	%r0, -524288
	alcg	%r0, -1
	alcg	%r0, 0
	alcg	%r0, 1
	alcg	%r0, 524287
	alcg	%r0, 0(%r1)
	alcg	%r0, 0(%r15)
	alcg	%r0, 524287(%r1,%r15)
	alcg	%r0, 524287(%r15,%r1)
	alcg	%r15, 0

#CHECK: alcgr	%r0, %r0                # encoding: [0xb9,0x88,0x00,0x00]
#CHECK: alcgr	%r0, %r15               # encoding: [0xb9,0x88,0x00,0x0f]
#CHECK: alcgr	%r15, %r0               # encoding: [0xb9,0x88,0x00,0xf0]
#CHECK: alcgr	%r7, %r8                # encoding: [0xb9,0x88,0x00,0x78]

	alcgr	%r0,%r0
	alcgr	%r0,%r15
	alcgr	%r15,%r0
	alcgr	%r7,%r8

#CHECK: alcr	%r0, %r0                # encoding: [0xb9,0x98,0x00,0x00]
#CHECK: alcr	%r0, %r15               # encoding: [0xb9,0x98,0x00,0x0f]
#CHECK: alcr	%r15, %r0               # encoding: [0xb9,0x98,0x00,0xf0]
#CHECK: alcr	%r7, %r8                # encoding: [0xb9,0x98,0x00,0x78]

	alcr	%r0,%r0
	alcr	%r0,%r15
	alcr	%r15,%r0
	alcr	%r7,%r8

#CHECK: alfi	%r0, 0                  # encoding: [0xc2,0x0b,0x00,0x00,0x00,0x00]
#CHECK: alfi	%r0, 4294967295         # encoding: [0xc2,0x0b,0xff,0xff,0xff,0xff]
#CHECK: alfi	%r15, 0                 # encoding: [0xc2,0xfb,0x00,0x00,0x00,0x00]

	alfi	%r0, 0
	alfi	%r0, (1 << 32) - 1
	alfi	%r15, 0

#CHECK: alg	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x0a]
#CHECK: alg	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x0a]
#CHECK: alg	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x0a]
#CHECK: alg	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x0a]
#CHECK: alg	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x0a]
#CHECK: alg	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x0a]
#CHECK: alg	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x0a]
#CHECK: alg	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x0a]
#CHECK: alg	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x0a]
#CHECK: alg	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x0a]
#CHECK: alg	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x0a]
#CHECK: alg	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x0a]
#CHECK: alg	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x0a]
#CHECK: alg	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x0a]
#CHECK: alg	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x0a]
#CHECK: alg	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x0a]
#CHECK: alg	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x0a]
#CHECK: alg	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x0a]
#CHECK: alg	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x0a]
#CHECK: alg	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x0a]

	alg	%r0, -524288
	alg	%r0, -1
	alg	%r0, 0
	alg	%r0, 1
	alg	%r0, 524287
	alg	%r0, 0(%r1)
	alg	%r0, 0(%r15)
	alg	%r0, 524287(%r1,%r15)
	alg	%r0, 524287(%r15,%r1)
	alg	%r15, 0
	alg	0, -524288
	alg	0, -1
	alg	0, 0
	alg	0, 1
	alg	0, 524287
	alg	0, 0(1)
	alg	0, 0(15)
	alg	0, 524287(1,15)
	alg	0, 524287(15,1)
	alg	15, 0

#CHECK: algf	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x1a]
#CHECK: algf	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x1a]
#CHECK: algf	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x1a]
#CHECK: algf	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x1a]
#CHECK: algf	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x1a]
#CHECK: algf	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x1a]
#CHECK: algf	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x1a]
#CHECK: algf	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x1a]
#CHECK: algf	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x1a]
#CHECK: algf	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x1a]

	algf	%r0, -524288
	algf	%r0, -1
	algf	%r0, 0
	algf	%r0, 1
	algf	%r0, 524287
	algf	%r0, 0(%r1)
	algf	%r0, 0(%r15)
	algf	%r0, 524287(%r1,%r15)
	algf	%r0, 524287(%r15,%r1)
	algf	%r15, 0

#CHECK: algfi	%r0, 0                  # encoding: [0xc2,0x0a,0x00,0x00,0x00,0x00]
#CHECK: algfi	%r0, 4294967295         # encoding: [0xc2,0x0a,0xff,0xff,0xff,0xff]
#CHECK: algfi	%r15, 0                 # encoding: [0xc2,0xfa,0x00,0x00,0x00,0x00]

	algfi	%r0, 0
	algfi	%r0, (1 << 32) - 1
	algfi	%r15, 0

#CHECK: algfr	%r0, %r0                # encoding: [0xb9,0x1a,0x00,0x00]
#CHECK: algfr	%r0, %r15               # encoding: [0xb9,0x1a,0x00,0x0f]
#CHECK: algfr	%r15, %r0               # encoding: [0xb9,0x1a,0x00,0xf0]
#CHECK: algfr	%r7, %r8                # encoding: [0xb9,0x1a,0x00,0x78]

	algfr	%r0,%r0
	algfr	%r0,%r15
	algfr	%r15,%r0
	algfr	%r7,%r8

#CHECK: algr	%r0, %r0                # encoding: [0xb9,0x0a,0x00,0x00]
#CHECK: algr	%r0, %r15               # encoding: [0xb9,0x0a,0x00,0x0f]
#CHECK: algr	%r15, %r0               # encoding: [0xb9,0x0a,0x00,0xf0]
#CHECK: algr	%r7, %r8                # encoding: [0xb9,0x0a,0x00,0x78]

	algr	%r0,%r0
	algr	%r0,%r15
	algr	%r15,%r0
	algr	%r7,%r8

#CHECK: algsi	-524288, 0              # encoding: [0xeb,0x00,0x00,0x00,0x80,0x7e]
#CHECK: algsi	-1, 0                   # encoding: [0xeb,0x00,0x0f,0xff,0xff,0x7e]
#CHECK: algsi	0, 0                    # encoding: [0xeb,0x00,0x00,0x00,0x00,0x7e]
#CHECK: algsi	1, 0                    # encoding: [0xeb,0x00,0x00,0x01,0x00,0x7e]
#CHECK: algsi	524287, 0               # encoding: [0xeb,0x00,0x0f,0xff,0x7f,0x7e]
#CHECK: algsi	0, -128                 # encoding: [0xeb,0x80,0x00,0x00,0x00,0x7e]
#CHECK: algsi	0, -1                   # encoding: [0xeb,0xff,0x00,0x00,0x00,0x7e]
#CHECK: algsi	0, 1                    # encoding: [0xeb,0x01,0x00,0x00,0x00,0x7e]
#CHECK: algsi	0, 127                  # encoding: [0xeb,0x7f,0x00,0x00,0x00,0x7e]
#CHECK: algsi	0(%r1), 42              # encoding: [0xeb,0x2a,0x10,0x00,0x00,0x7e]
#CHECK: algsi	0(%r15), 42             # encoding: [0xeb,0x2a,0xf0,0x00,0x00,0x7e]
#CHECK: algsi	524287(%r1), 42         # encoding: [0xeb,0x2a,0x1f,0xff,0x7f,0x7e]
#CHECK: algsi	524287(%r15), 42        # encoding: [0xeb,0x2a,0xff,0xff,0x7f,0x7e]
#CHECK: algsi	-524288, 0              # encoding: [0xeb,0x00,0x00,0x00,0x80,0x7e]
#CHECK: algsi	-1, 0                   # encoding: [0xeb,0x00,0x0f,0xff,0xff,0x7e]
#CHECK: algsi	0, 0                    # encoding: [0xeb,0x00,0x00,0x00,0x00,0x7e]
#CHECK: algsi	1, 0                    # encoding: [0xeb,0x00,0x00,0x01,0x00,0x7e]
#CHECK: algsi	524287, 0               # encoding: [0xeb,0x00,0x0f,0xff,0x7f,0x7e]
#CHECK: algsi	0, -128                 # encoding: [0xeb,0x80,0x00,0x00,0x00,0x7e]
#CHECK: algsi	0, -1                   # encoding: [0xeb,0xff,0x00,0x00,0x00,0x7e]
#CHECK: algsi	0, 1                    # encoding: [0xeb,0x01,0x00,0x00,0x00,0x7e]
#CHECK: algsi	0, 127                  # encoding: [0xeb,0x7f,0x00,0x00,0x00,0x7e]
#CHECK: algsi	0(%r1), 42              # encoding: [0xeb,0x2a,0x10,0x00,0x00,0x7e]
#CHECK: algsi	0(%r15), 42             # encoding: [0xeb,0x2a,0xf0,0x00,0x00,0x7e]
#CHECK: algsi	524287(%r1), 42         # encoding: [0xeb,0x2a,0x1f,0xff,0x7f,0x7e]
#CHECK: algsi	524287(%r15), 42        # encoding: [0xeb,0x2a,0xff,0xff,0x7f,0x7e]

	algsi	-524288, 0
	algsi	-1, 0
	algsi	0, 0
	algsi	1, 0
	algsi	524287, 0
	algsi	0, -128
	algsi	0, -1
	algsi	0, 1
	algsi	0, 127
	algsi	0(%r1), 42
	algsi	0(%r15), 42
	algsi	524287(%r1), 42
	algsi	524287(%r15), 42
	algsi	-524288, 0
	algsi	-1, 0
	algsi	0, 0
	algsi	1, 0
	algsi	524287, 0
	algsi	0, -128
	algsi	0, -1
	algsi	0, 1
	algsi	0, 127
	algsi	0(1), 42
	algsi	0(15), 42
	algsi	524287(1), 42
	algsi	524287(15), 42

#CHECK: alr	%r0, %r0                # encoding: [0x1e,0x00]
#CHECK: alr	%r0, %r15               # encoding: [0x1e,0x0f]
#CHECK: alr	%r15, %r0               # encoding: [0x1e,0xf0]
#CHECK: alr	%r7, %r8                # encoding: [0x1e,0x78]

	alr	%r0,%r0
	alr	%r0,%r15
	alr	%r15,%r0
	alr	%r7,%r8

#CHECK: alsi	-524288, 0              # encoding: [0xeb,0x00,0x00,0x00,0x80,0x6e]
#CHECK: alsi	-1, 0                   # encoding: [0xeb,0x00,0x0f,0xff,0xff,0x6e]
#CHECK: alsi	0, 0                    # encoding: [0xeb,0x00,0x00,0x00,0x00,0x6e]
#CHECK: alsi	1, 0                    # encoding: [0xeb,0x00,0x00,0x01,0x00,0x6e]
#CHECK: alsi	524287, 0               # encoding: [0xeb,0x00,0x0f,0xff,0x7f,0x6e]
#CHECK: alsi	0, -128                 # encoding: [0xeb,0x80,0x00,0x00,0x00,0x6e]
#CHECK: alsi	0, -1                   # encoding: [0xeb,0xff,0x00,0x00,0x00,0x6e]
#CHECK: alsi	0, 1                    # encoding: [0xeb,0x01,0x00,0x00,0x00,0x6e]
#CHECK: alsi	0, 127                  # encoding: [0xeb,0x7f,0x00,0x00,0x00,0x6e]
#CHECK: alsi	0(%r1), 42              # encoding: [0xeb,0x2a,0x10,0x00,0x00,0x6e]
#CHECK: alsi	0(%r15), 42             # encoding: [0xeb,0x2a,0xf0,0x00,0x00,0x6e]
#CHECK: alsi	524287(%r1), 42         # encoding: [0xeb,0x2a,0x1f,0xff,0x7f,0x6e]
#CHECK: alsi	524287(%r15), 42        # encoding: [0xeb,0x2a,0xff,0xff,0x7f,0x6e]

	alsi	-524288, 0
	alsi	-1, 0
	alsi	0, 0
	alsi	1, 0
	alsi	524287, 0
	alsi	0, -128
	alsi	0, -1
	alsi	0, 1
	alsi	0, 127
	alsi	0(%r1), 42
	alsi	0(%r15), 42
	alsi	524287(%r1), 42
	alsi	524287(%r15), 42

#CHECK: aly	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x5e]
#CHECK: aly	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x5e]
#CHECK: aly	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x5e]
#CHECK: aly	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x5e]
#CHECK: aly	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x5e]
#CHECK: aly	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x5e]
#CHECK: aly	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x5e]
#CHECK: aly	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x5e]
#CHECK: aly	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x5e]
#CHECK: aly	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x5e]

	aly	%r0, -524288
	aly	%r0, -1
	aly	%r0, 0
	aly	%r0, 1
	aly	%r0, 524287
	aly	%r0, 0(%r1)
	aly	%r0, 0(%r15)
	aly	%r0, 524287(%r1,%r15)
	aly	%r0, 524287(%r15,%r1)
	aly	%r15, 0

#CHECK: ap	0(1), 0(1)              # encoding: [0xfa,0x00,0x00,0x00,0x00,0x00]
#CHECK: ap	0(1), 0(1,%r1)          # encoding: [0xfa,0x00,0x00,0x00,0x10,0x00]
#CHECK: ap	0(1), 0(1,%r15)         # encoding: [0xfa,0x00,0x00,0x00,0xf0,0x00]
#CHECK: ap	0(1), 4095(1)           # encoding: [0xfa,0x00,0x00,0x00,0x0f,0xff]
#CHECK: ap	0(1), 4095(1,%r1)       # encoding: [0xfa,0x00,0x00,0x00,0x1f,0xff]
#CHECK: ap	0(1), 4095(1,%r15)      # encoding: [0xfa,0x00,0x00,0x00,0xff,0xff]
#CHECK: ap	0(1,%r1), 0(1)          # encoding: [0xfa,0x00,0x10,0x00,0x00,0x00]
#CHECK: ap	0(1,%r15), 0(1)         # encoding: [0xfa,0x00,0xf0,0x00,0x00,0x00]
#CHECK: ap	4095(1,%r1), 0(1)       # encoding: [0xfa,0x00,0x1f,0xff,0x00,0x00]
#CHECK: ap	4095(1,%r15), 0(1)      # encoding: [0xfa,0x00,0xff,0xff,0x00,0x00]
#CHECK: ap	0(16,%r1), 0(1)         # encoding: [0xfa,0xf0,0x10,0x00,0x00,0x00]
#CHECK: ap	0(16,%r15), 0(1)        # encoding: [0xfa,0xf0,0xf0,0x00,0x00,0x00]
#CHECK: ap	0(1), 0(16,%r1)         # encoding: [0xfa,0x0f,0x00,0x00,0x10,0x00]
#CHECK: ap	0(1), 0(16,%r15)        # encoding: [0xfa,0x0f,0x00,0x00,0xf0,0x00]

	ap	0(1), 0(1)
	ap	0(1), 0(1,%r1)
	ap	0(1), 0(1,%r15)
	ap	0(1), 4095(1)
	ap	0(1), 4095(1,%r1)
	ap	0(1), 4095(1,%r15)
	ap	0(1,%r1), 0(1)
	ap	0(1,%r15), 0(1)
	ap	4095(1,%r1), 0(1)
	ap	4095(1,%r15), 0(1)
	ap	0(16,%r1), 0(1)
	ap	0(16,%r15), 0(1)
	ap	0(1), 0(16,%r1)
	ap	0(1), 0(16,%r15)
	ap	0(1), 0(1)
	ap	0(1), 0(1,1)
	ap	0(1), 0(1,15)
	ap	0(1), 4095(1)
	ap	0(1), 4095(1,1)
	ap	0(1), 4095(1,15)
	ap	0(1,1), 0(1)
	ap	0(1,15), 0(1)
	ap	4095(1,1), 0(1)
	ap	4095(1,15), 0(1)
	ap	0(16,1), 0(1)
	ap	0(16,15), 0(1)
	ap	0(1), 0(16,1)
	ap	0(1), 0(16,15)

#CHECK: ar	%r0, %r0                # encoding: [0x1a,0x00]
#CHECK: ar	%r0, %r15               # encoding: [0x1a,0x0f]
#CHECK: ar	%r15, %r0               # encoding: [0x1a,0xf0]
#CHECK: ar	%r7, %r8                # encoding: [0x1a,0x78]

	ar	%r0,%r0
	ar	%r0,%r15
	ar	%r15,%r0
	ar	%r7,%r8

#CHECK: asi	-524288, 0              # encoding: [0xeb,0x00,0x00,0x00,0x80,0x6a]
#CHECK: asi	-1, 0                   # encoding: [0xeb,0x00,0x0f,0xff,0xff,0x6a]
#CHECK: asi	0, 0                    # encoding: [0xeb,0x00,0x00,0x00,0x00,0x6a]
#CHECK: asi	1, 0                    # encoding: [0xeb,0x00,0x00,0x01,0x00,0x6a]
#CHECK: asi	524287, 0               # encoding: [0xeb,0x00,0x0f,0xff,0x7f,0x6a]
#CHECK: asi	0, -128                 # encoding: [0xeb,0x80,0x00,0x00,0x00,0x6a]
#CHECK: asi	0, -1                   # encoding: [0xeb,0xff,0x00,0x00,0x00,0x6a]
#CHECK: asi	0, 1                    # encoding: [0xeb,0x01,0x00,0x00,0x00,0x6a]
#CHECK: asi	0, 127                  # encoding: [0xeb,0x7f,0x00,0x00,0x00,0x6a]
#CHECK: asi	0(%r1), 42              # encoding: [0xeb,0x2a,0x10,0x00,0x00,0x6a]
#CHECK: asi	0(%r15), 42             # encoding: [0xeb,0x2a,0xf0,0x00,0x00,0x6a]
#CHECK: asi	524287(%r1), 42         # encoding: [0xeb,0x2a,0x1f,0xff,0x7f,0x6a]
#CHECK: asi	524287(%r15), 42        # encoding: [0xeb,0x2a,0xff,0xff,0x7f,0x6a]

	asi	-524288, 0
	asi	-1, 0
	asi	0, 0
	asi	1, 0
	asi	524287, 0
	asi	0, -128
	asi	0, -1
	asi	0, 1
	asi	0, 127
	asi	0(%r1), 42
	asi	0(%r15), 42
	asi	524287(%r1), 42
	asi	524287(%r15), 42

#CHECK: au	%f0, 0                  # encoding: [0x7e,0x00,0x00,0x00]
#CHECK: au	%f0, 4095               # encoding: [0x7e,0x00,0x0f,0xff]
#CHECK: au	%f0, 0(%r1)             # encoding: [0x7e,0x00,0x10,0x00]
#CHECK: au	%f0, 0(%r15)            # encoding: [0x7e,0x00,0xf0,0x00]
#CHECK: au	%f0, 4095(%r1,%r15)     # encoding: [0x7e,0x01,0xff,0xff]
#CHECK: au	%f0, 4095(%r15,%r1)     # encoding: [0x7e,0x0f,0x1f,0xff]
#CHECK: au	%f15, 0                 # encoding: [0x7e,0xf0,0x00,0x00]

	au	%f0, 0
	au	%f0, 4095
	au	%f0, 0(%r1)
	au	%f0, 0(%r15)
	au	%f0, 4095(%r1,%r15)
	au	%f0, 4095(%r15,%r1)
	au	%f15, 0

#CHECK: aur	%f0, %f0                # encoding: [0x3e,0x00]
#CHECK: aur	%f0, %f15               # encoding: [0x3e,0x0f]
#CHECK: aur	%f7, %f8                # encoding: [0x3e,0x78]
#CHECK: aur	%f15, %f0               # encoding: [0x3e,0xf0]

	aur	%f0, %f0
	aur	%f0, %f15
	aur	%f7, %f8
	aur	%f15, %f0

#CHECK: aw	%f0, 0                  # encoding: [0x6e,0x00,0x00,0x00]
#CHECK: aw	%f0, 4095               # encoding: [0x6e,0x00,0x0f,0xff]
#CHECK: aw	%f0, 0(%r1)             # encoding: [0x6e,0x00,0x10,0x00]
#CHECK: aw	%f0, 0(%r15)            # encoding: [0x6e,0x00,0xf0,0x00]
#CHECK: aw	%f0, 4095(%r1,%r15)     # encoding: [0x6e,0x01,0xff,0xff]
#CHECK: aw	%f0, 4095(%r15,%r1)     # encoding: [0x6e,0x0f,0x1f,0xff]
#CHECK: aw	%f15, 0                 # encoding: [0x6e,0xf0,0x00,0x00]

	aw	%f0, 0
	aw	%f0, 4095
	aw	%f0, 0(%r1)
	aw	%f0, 0(%r15)
	aw	%f0, 4095(%r1,%r15)
	aw	%f0, 4095(%r15,%r1)
	aw	%f15, 0

#CHECK: awr	%f0, %f0                # encoding: [0x2e,0x00]
#CHECK: awr	%f0, %f15               # encoding: [0x2e,0x0f]
#CHECK: awr	%f7, %f8                # encoding: [0x2e,0x78]
#CHECK: awr	%f15, %f0               # encoding: [0x2e,0xf0]

	awr	%f0, %f0
	awr	%f0, %f15
	awr	%f7, %f8
	awr	%f15, %f0

#CHECK: axbr	%f0, %f0                # encoding: [0xb3,0x4a,0x00,0x00]
#CHECK: axbr	%f0, %f13               # encoding: [0xb3,0x4a,0x00,0x0d]
#CHECK: axbr	%f8, %f8                # encoding: [0xb3,0x4a,0x00,0x88]
#CHECK: axbr	%f13, %f0               # encoding: [0xb3,0x4a,0x00,0xd0]

	axbr	%f0, %f0
	axbr	%f0, %f13
	axbr	%f8, %f8
	axbr	%f13, %f0

#CHECK: axr	%f0, %f0                # encoding: [0x36,0x00]
#CHECK: axr	%f0, %f13               # encoding: [0x36,0x0d]
#CHECK: axr	%f8, %f8                # encoding: [0x36,0x88]
#CHECK: axr	%f13, %f0               # encoding: [0x36,0xd0]

	axr	%f0, %f0
	axr	%f0, %f13
	axr	%f8, %f8
	axr	%f13, %f0

#CHECK: axtr	%f0, %f0, %f0           # encoding: [0xb3,0xda,0x00,0x00]
#CHECK: axtr	%f0, %f0, %f13          # encoding: [0xb3,0xda,0xd0,0x00]
#CHECK: axtr	%f0, %f13, %f0          # encoding: [0xb3,0xda,0x00,0x0d]
#CHECK: axtr	%f13, %f0, %f0          # encoding: [0xb3,0xda,0x00,0xd0]
#CHECK: axtr	%f8, %f8, %f8           # encoding: [0xb3,0xda,0x80,0x88]

	axtr	%f0, %f0, %f0
	axtr	%f0, %f0, %f13
	axtr	%f0, %f13, %f0
	axtr	%f13, %f0, %f0
	axtr	%f8, %f8, %f8

#CHECK: ay	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x5a]
#CHECK: ay	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x5a]
#CHECK: ay	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x5a]
#CHECK: ay	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x5a]
#CHECK: ay	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x5a]
#CHECK: ay	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x5a]
#CHECK: ay	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x5a]
#CHECK: ay	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x5a]
#CHECK: ay	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x5a]
#CHECK: ay	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x5a]

	ay	%r0, -524288
	ay	%r0, -1
	ay	%r0, 0
	ay	%r0, 1
	ay	%r0, 524287
	ay	%r0, 0(%r1)
	ay	%r0, 0(%r15)
	ay	%r0, 524287(%r1,%r15)
	ay	%r0, 524287(%r15,%r1)
	ay	%r15, 0

#CHECK: bakr	%r0, %r0                # encoding: [0xb2,0x40,0x00,0x00]
#CHECK: bakr	%r0, %r15               # encoding: [0xb2,0x40,0x00,0x0f]
#CHECK: bakr	%r15, %r0               # encoding: [0xb2,0x40,0x00,0xf0]
#CHECK: bakr	%r7, %r8                # encoding: [0xb2,0x40,0x00,0x78]

	bakr	%r0,%r0
	bakr	%r0,%r15
	bakr	%r15,%r0
	bakr	%r7,%r8

#CHECK: bal	%r0, 0                  # encoding: [0x45,0x00,0x00,0x00]
#CHECK: bal	%r1, 4095               # encoding: [0x45,0x10,0x0f,0xff]
#CHECK: bal	%r2, 0(%r1)             # encoding: [0x45,0x20,0x10,0x00]
#CHECK: bal	%r3, 0(%r15)            # encoding: [0x45,0x30,0xf0,0x00]
#CHECK: bal	%r14, 4095(%r1,%r15)    # encoding: [0x45,0xe1,0xff,0xff]
#CHECK: bal	%r15, 4095(%r15,%r1)    # encoding: [0x45,0xff,0x1f,0xff]

	bal	%r0, 0
	bal	%r1, 4095
	bal	%r2, 0(%r1)
	bal	%r3, 0(%r15)
	bal	%r14, 4095(%r1,%r15)
	bal	%r15, 4095(%r15,%r1)

#CHECK: balr	%r0, %r0                # encoding: [0x05,0x00]
#CHECK: balr	%r0, %r15               # encoding: [0x05,0x0f]
#CHECK: balr	%r14, %r9               # encoding: [0x05,0xe9]
#CHECK: balr	%r15, %r1               # encoding: [0x05,0xf1]

	balr	%r0,%r0
	balr	%r0,%r15
	balr	%r14,%r9
	balr	%r15,%r1

#CHECK: bas	%r0, 0                  # encoding: [0x4d,0x00,0x00,0x00]
#CHECK: bas	%r1, 4095               # encoding: [0x4d,0x10,0x0f,0xff]
#CHECK: bas	%r2, 0(%r1)             # encoding: [0x4d,0x20,0x10,0x00]
#CHECK: bas	%r3, 0(%r15)            # encoding: [0x4d,0x30,0xf0,0x00]
#CHECK: bas	%r14, 4095(%r1,%r15)    # encoding: [0x4d,0xe1,0xff,0xff]
#CHECK: bas	%r15, 4095(%r15,%r1)    # encoding: [0x4d,0xff,0x1f,0xff]

	bas	%r0, 0
	bas	%r1, 4095
	bas	%r2, 0(%r1)
	bas	%r3, 0(%r15)
	bas	%r14, 4095(%r1,%r15)
	bas	%r15, 4095(%r15,%r1)

#CHECK: basr	%r0, %r0                # encoding: [0x0d,0x00]
#CHECK: basr	%r0, %r15               # encoding: [0x0d,0x0f]
#CHECK: basr	%r14, %r9               # encoding: [0x0d,0xe9]
#CHECK: basr	%r15, %r1               # encoding: [0x0d,0xf1]

	basr	%r0,%r0
	basr	%r0,%r15
	basr	%r14,%r9
	basr	%r15,%r1

#CHECK: bassm	%r0, %r0                # encoding: [0x0c,0x00]
#CHECK: bassm	%r0, %r15               # encoding: [0x0c,0x0f]
#CHECK: bassm	%r14, %r9               # encoding: [0x0c,0xe9]
#CHECK: bassm	%r15, %r1               # encoding: [0x0c,0xf1]

	bassm	%r0,%r0
	bassm	%r0,%r15
	bassm	%r14,%r9
	bassm	%r15,%r1

#CHECK: bsa	%r0, %r0                # encoding: [0xb2,0x5a,0x00,0x00]
#CHECK: bsa	%r0, %r15               # encoding: [0xb2,0x5a,0x00,0x0f]
#CHECK: bsa	%r15, %r0               # encoding: [0xb2,0x5a,0x00,0xf0]
#CHECK: bsa	%r7, %r8                # encoding: [0xb2,0x5a,0x00,0x78]

	bsa	%r0,%r0
	bsa	%r0,%r15
	bsa	%r15,%r0
	bsa	%r7,%r8

#CHECK: bsg	%r0, %r0                # encoding: [0xb2,0x58,0x00,0x00]
#CHECK: bsg	%r0, %r15               # encoding: [0xb2,0x58,0x00,0x0f]
#CHECK: bsg	%r15, %r0               # encoding: [0xb2,0x58,0x00,0xf0]
#CHECK: bsg	%r7, %r8                # encoding: [0xb2,0x58,0x00,0x78]

	bsg	%r0,%r0
	bsg	%r0,%r15
	bsg	%r15,%r0
	bsg	%r7,%r8

#CHECK: bsm	%r0, %r0                # encoding: [0x0b,0x00]
#CHECK: bsm	%r0, %r15               # encoding: [0x0b,0x0f]
#CHECK: bsm	%r14, %r9               # encoding: [0x0b,0xe9]
#CHECK: bsm	%r15, %r1               # encoding: [0x0b,0xf1]

	bsm	%r0,%r0
	bsm	%r0,%r15
	bsm	%r14,%r9
	bsm	%r15,%r1

#CHECK: b	0                       # encoding: [0x47,0xf0,0x00,0x00]
#CHECK: b	4095                    # encoding: [0x47,0xf0,0x0f,0xff]
#CHECK: b	0(%r1)                  # encoding: [0x47,0xf0,0x10,0x00]
#CHECK: b	0(%r15)                 # encoding: [0x47,0xf0,0xf0,0x00]
#CHECK: b	4095(%r1,%r15)          # encoding: [0x47,0xf1,0xff,0xff]
#CHECK: b	4095(%r15,%r1)          # encoding: [0x47,0xff,0x1f,0xff]

	b	0
	b	4095
	b	0(%r1)
	b	0(%r15)
	b	4095(%r1,%r15)
	b	4095(%r15,%r1)

#CHECK: bc	0, 0                    # encoding: [0x47,0x00,0x00,0x00]
#CHECK: bc	0, 4095                 # encoding: [0x47,0x00,0x0f,0xff]
#CHECK: bc	0, 0(%r1)               # encoding: [0x47,0x00,0x10,0x00]
#CHECK: bc	0, 0(%r15)              # encoding: [0x47,0x00,0xf0,0x00]
#CHECK: bc	0, 4095(%r1,%r15)       # encoding: [0x47,0x01,0xff,0xff]
#CHECK: bc	0, 4095(%r15,%r1)       # encoding: [0x47,0x0f,0x1f,0xff]
#CHECK: bc	15, 0                   # encoding: [0x47,0xf0,0x00,0x00]

	bc	0, 0
	bc	0, 4095
	bc	0, 0(%r1)
	bc	0, 0(%r15)
	bc	0, 4095(%r1,%r15)
	bc	0, 4095(%r15,%r1)
	bc	15, 0

#CHECK:	bc	1, 0(%r7)		# encoding: [0x47,0x10,0x70,0x00]
#CHECK:	bo	0(%r15)			# encoding: [0x47,0x10,0xf0,0x00]

	bc	1, 0(%r7)
	bo	0(%r15)

#CHECK:	bc	2, 0(%r7)		# encoding: [0x47,0x20,0x70,0x00]
#CHECK:	bh	0(%r15)			# encoding: [0x47,0x20,0xf0,0x00]
#CHECK:	bp	0(%r10)			# encoding: [0x47,0x20,0xa0,0x00]

	bc	2, 0(%r7)
	bh	0(%r15)
	bp	0(%r10)

#CHECK:	bc	3, 0(%r7)		# encoding: [0x47,0x30,0x70,0x00]
#CHECK:	bnle	0(%r15)			# encoding: [0x47,0x30,0xf0,0x00]

	bc	3, 0(%r7)
	bnle	0(%r15)

#CHECK:	bc	4, 0(%r7)		# encoding: [0x47,0x40,0x70,0x00]
#CHECK:	bl	0(%r15)			# encoding: [0x47,0x40,0xf0,0x00]
#CHECK:	bm	0(%r14)			# encoding: [0x47,0x40,0xe0,0x00]

	bc	4, 0(%r7)
	bl	0(%r15)
	bm	0(%r14)

#CHECK:	bc	5, 0(%r7)		# encoding: [0x47,0x50,0x70,0x00]
#CHECK:	bnhe	0(%r15)			# encoding: [0x47,0x50,0xf0,0x00]

	bc	5, 0(%r7)
	bnhe	0(%r15)

#CHECK:	bc	6, 0(%r7)		# encoding: [0x47,0x60,0x70,0x00]
#CHECK:	blh	0(%r15)			# encoding: [0x47,0x60,0xf0,0x00]

	bc	6, 0(%r7)
	blh	0(%r15)

#CHECK:	bc	7, 0(%r7)		# encoding: [0x47,0x70,0x70,0x00]
#CHECK:	bne	0(%r15)			# encoding: [0x47,0x70,0xf0,0x00]
#CHECK:	bnz	0(%r11)			# encoding: [0x47,0x70,0xb0,0x00]

	bc	7, 0(%r7)
	bne	0(%r15)
	bnz	0(%r11)

#CHECK:	bc	8, 0(%r7)		# encoding: [0x47,0x80,0x70,0x00]
#CHECK:	be	0(%r15)			# encoding: [0x47,0x80,0xf0,0x00]

	bc	8, 0(%r7)
	be	0(%r15)

#CHECK:	bc	9, 0(%r7)		# encoding: [0x47,0x90,0x70,0x00]
#CHECK:	bnlh	0(%r15)			# encoding: [0x47,0x90,0xf0,0x00]

	bc	9, 0(%r7)
	bnlh	0(%r15)

#CHECK:	bc	10, 0(%r7)		# encoding: [0x47,0xa0,0x70,0x00]
#CHECK:	bhe	0(%r15)			# encoding: [0x47,0xa0,0xf0,0x00]

	bc	10, 0(%r7)
	bhe	0(%r15)

#CHECK:	bc	11, 0(%r7)		# encoding: [0x47,0xb0,0x70,0x00]
#CHECK:	bnl	0(%r15)			# encoding: [0x47,0xb0,0xf0,0x00]
#CHECK:	bnm	0(%r13)			# encoding: [0x47,0xb0,0xd0,0x00]

	bc	11, 0(%r7)
	bnl	0(%r15)
	bnm	0(%r13)

#CHECK:	bc	12, 0(%r7)		# encoding: [0x47,0xc0,0x70,0x00]
#CHECK:	ble	0(%r15)			# encoding: [0x47,0xc0,0xf0,0x00]

	bc	12, 0(%r7)
	ble	0(%r15)

#CHECK:	bc	13, 0(%r7)		# encoding: [0x47,0xd0,0x70,0x00]
#CHECK:	bnh	0(%r15)			# encoding: [0x47,0xd0,0xf0,0x00]
#CHECK:	bnp	0(%r12)			# encoding: [0x47,0xd0,0xc0,0x00]

	bc	13, 0(%r7)
	bnh	0(%r15)
	bnp	0(%r12)

#CHECK:	bc	14, 0(%r7)		# encoding: [0x47,0xe0,0x70,0x00]
#CHECK:	bno	0(%r15)			# encoding: [0x47,0xe0,0xf0,0x00]

	bc	14, 0(%r7)
	bno	0(%r15)

#CHECK:	bc	8, 0(%r13)			# encoding: [0x47,0x80,0xd0,0x00]
#CHECK:	bz	0(%r6)			# encoding: [0x47,0x80,0x60,0x00]

	bc	8, 0(%r13)
	bz	0(%r6)


#CHECK: bcr	0, %r0			# encoding: [0x07,0x00]
#CHECK:	bcr	0, %r15			# encoding: [0x07,0x0f]

	bcr	0, %r0
	bcr	0, %r15

#CHECK:	bcr	1, %r7			# encoding: [0x07,0x17]
#CHECK:	bor	%r0			# encoding: [0x07,0x10]
#CHECK:	bor	%r15			# encoding: [0x07,0x1f]

	bcr	1, %r7
	bor	%r0
	bor	%r15

#CHECK:	bcr	2, %r7			# encoding: [0x07,0x27]
#CHECK:	bhr	%r0			# encoding: [0x07,0x20]
#CHECK:	bhr	%r15			# encoding: [0x07,0x2f]

	bcr	2, %r7
	bhr	%r0
	bhr	%r15

#CHECK:	bcr	3, %r7			# encoding: [0x07,0x37]
#CHECK:	bnler	%r0			# encoding: [0x07,0x30]
#CHECK:	bnler	%r15			# encoding: [0x07,0x3f]

	bcr	3, %r7
	bnler	%r0
	bnler	%r15

#CHECK:	bcr	4, %r7			# encoding: [0x07,0x47]
#CHECK:	blr	%r0			# encoding: [0x07,0x40]
#CHECK:	blr	%r15			# encoding: [0x07,0x4f]

	bcr	4, %r7
	blr	%r0
	blr	%r15

#CHECK:	bcr	5, %r7			# encoding: [0x07,0x57]
#CHECK:	bnher	%r0			# encoding: [0x07,0x50]
#CHECK:	bnher	%r15			# encoding: [0x07,0x5f]

	bcr	5, %r7
	bnher	%r0
	bnher	%r15

#CHECK:	bcr	6, %r7			# encoding: [0x07,0x67]
#CHECK:	blhr	%r0			# encoding: [0x07,0x60]
#CHECK:	blhr	%r15			# encoding: [0x07,0x6f]

	bcr	6, %r7
	blhr	%r0
	blhr	%r15

#CHECK:	bcr	7, %r7			# encoding: [0x07,0x77]
#CHECK:	bner	%r0			# encoding: [0x07,0x70]
#CHECK:	bner	%r15			# encoding: [0x07,0x7f]

	bcr	7, %r7
	bner	%r0
	bner	%r15

#CHECK:	bcr	8, %r7			# encoding: [0x07,0x87]
#CHECK:	ber	%r0			# encoding: [0x07,0x80]
#CHECK:	ber	%r15			# encoding: [0x07,0x8f]

	bcr	8, %r7
	ber	%r0
	ber	%r15

#CHECK:	bcr	9, %r7			# encoding: [0x07,0x97]
#CHECK:	bnlhr	%r0			# encoding: [0x07,0x90]
#CHECK:	bnlhr	%r15			# encoding: [0x07,0x9f]

	bcr	9, %r7
	bnlhr	%r0
	bnlhr	%r15

#CHECK:	bcr	10, %r7			# encoding: [0x07,0xa7]
#CHECK:	bher	%r0			# encoding: [0x07,0xa0]
#CHECK:	bher	%r15			# encoding: [0x07,0xaf]

	bcr	10, %r7
	bher	%r0
	bher	%r15

#CHECK:	bcr	11, %r7			# encoding: [0x07,0xb7]
#CHECK:	bnlr	%r0			# encoding: [0x07,0xb0]
#CHECK:	bnlr	%r15			# encoding: [0x07,0xbf]

	bcr	11, %r7
	bnlr	%r0
	bnlr	%r15

#CHECK:	bcr	12, %r7			# encoding: [0x07,0xc7]
#CHECK:	bler	%r0			# encoding: [0x07,0xc0]
#CHECK:	bler	%r15			# encoding: [0x07,0xcf]

	bcr	12, %r7
	bler	%r0
	bler	%r15

#CHECK:	bcr	13, %r7			# encoding: [0x07,0xd7]
#CHECK:	bnhr	%r0			# encoding: [0x07,0xd0]
#CHECK:	bnhr	%r15			# encoding: [0x07,0xdf]

	bcr	13, %r7
	bnhr	%r0
	bnhr	%r15

#CHECK:	bcr	14, %r7			# encoding: [0x07,0xe7]
#CHECK:	bnor	%r0			# encoding: [0x07,0xe0]
#CHECK:	bnor	%r15			# encoding: [0x07,0xef]

	bcr	14, %r7
	bnor	%r0
	bnor	%r15

#CHECK:	bcr	15, %r7			# encoding: [0x07,0xf7]
#CHECK: br	%r0                     # encoding: [0x07,0xf0]
#CHECK: br	%r1                     # encoding: [0x07,0xf1]
#CHECK: br	%r14                    # encoding: [0x07,0xfe]
#CHECK: br	%r15                    # encoding: [0x07,0xff]

	bcr	15, %r7
	br	%r0
	br	%r1
	br	%r14
	br	%r15

#CHECK:	bcr	4, %r7			# encoding: [0x07,0x47]
#CHECK:	bmr	%r0			# encoding: [0x07,0x40]
	bcr	4, %r7
	bmr	%r0

#CHECK:	bcr	11, %r8			# encoding: [0x07,0xb8]
#CHECK:	bnmr	%r1			# encoding: [0x07,0xb1]
	bcr	11, %r8
	bnmr	%r1

#CHECK:	bcr	13, %r9			# encoding: [0x07,0xd9]
#CHECK:	bnpr	%r2			# encoding: [0x07,0xd2]
	bcr	13, %r9
	bnpr	%r2

#CHECK:	bcr	7, %r10			# encoding: [0x07,0x7a]
#CHECK:	bnzr	%r3			# encoding: [0x07,0x73]
	bcr	7, %r10
	bnzr	%r3

#CHECK:	bcr	2, %r11			# encoding: [0x07,0x2b]
#CHECK:	bpr	%r4			# encoding: [0x07,0x24]
	bcr	2, %r11
	bpr	%r4

#CHECK:	bcr	8, %r12			# encoding: [0x07,0x8c]
#CHECK:	bzr	%r5			# encoding: [0x07,0x85]
	bcr	8, %r12
	bzr	%r5

#CHECK: bras	%r0, .[[LAB:L.*]]-65536	# encoding: [0xa7,0x05,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-65536)+2, kind: FK_390_PC16DBL
#CHECK: bras	%r0, .[[LAB:L.*]]-65536	# encoding: [0xa7,0x05,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-65536)+2, kind: FK_390_PC16DBL
	bras	%r0, -0x10000
	jas	%r0, -0x10000

#CHECK: bras	%r0, .[[LAB:L.*]]-2	# encoding: [0xa7,0x05,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC16DBL
#CHECK: bras	%r0, .[[LAB:L.*]]-2	# encoding: [0xa7,0x05,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC16DBL
	bras	%r0, -2
	jas	%r0, -2

#CHECK: bras	%r0, .[[LAB:L.*]]	# encoding: [0xa7,0x05,A,A]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC16DBL
#CHECK: bras	%r0, .[[LAB:L.*]]	# encoding: [0xa7,0x05,A,A]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC16DBL
	bras	%r0, 0
	jas	%r0, 0

#CHECK: bras	%r0, .[[LAB:L.*]]+65534	# encoding: [0xa7,0x05,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+65534)+2, kind: FK_390_PC16DBL
#CHECK: bras	%r0, .[[LAB:L.*]]+65534	# encoding: [0xa7,0x05,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+65534)+2, kind: FK_390_PC16DBL
	bras	%r0, 0xfffe
	jas	%r0, 0xfffe

#CHECK: bras	%r0, foo                # encoding: [0xa7,0x05,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: bras	%r14, foo               # encoding: [0xa7,0xe5,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: bras	%r15, foo               # encoding: [0xa7,0xf5,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: bras	%r0, foo                # encoding: [0xa7,0x05,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: bras	%r14, foo               # encoding: [0xa7,0xe5,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: bras	%r15, foo               # encoding: [0xa7,0xf5,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	bras	%r0,foo
	bras	%r14,foo
	bras	%r15,foo
	jas	%r0,foo
	jas	%r14,foo
	jas	%r15,foo

#CHECK: bras	%r0, bar+100                # encoding: [0xa7,0x05,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
#CHECK: bras	%r14, bar+100               # encoding: [0xa7,0xe5,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
#CHECK: bras	%r15, bar+100               # encoding: [0xa7,0xf5,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
#CHECK: bras	%r0, bar+100                # encoding: [0xa7,0x05,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
#CHECK: bras	%r14, bar+100               # encoding: [0xa7,0xe5,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
#CHECK: bras	%r15, bar+100               # encoding: [0xa7,0xf5,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	bras	%r0,bar+100
	bras	%r14,bar+100
	bras	%r15,bar+100
	jas	%r0,bar+100
	jas	%r14,bar+100
	jas	%r15,bar+100

#CHECK: bras	%r0, bar@PLT                # encoding: [0xa7,0x05,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
#CHECK: bras	%r14, bar@PLT               # encoding: [0xa7,0xe5,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
#CHECK: bras	%r15, bar@PLT               # encoding: [0xa7,0xf5,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
#CHECK: bras	%r0, bar@PLT                # encoding: [0xa7,0x05,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
#CHECK: bras	%r14, bar@PLT               # encoding: [0xa7,0xe5,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
#CHECK: bras	%r15, bar@PLT               # encoding: [0xa7,0xf5,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	bras	%r0,bar@PLT
	bras	%r14,bar@PLT
	bras	%r15,bar@PLT
	jas	%r0,bar@PLT
	jas	%r14,bar@PLT
	jas	%r15,bar@PLT

#CHECK: brasl	%r0, .[[LAB:L.*]]-4294967296 # encoding: [0xc0,0x05,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-4294967296)+2, kind: FK_390_PC32DBL
#CHECK: brasl	%r0, .[[LAB:L.*]]-4294967296 # encoding: [0xc0,0x05,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-4294967296)+2, kind: FK_390_PC32DBL
	brasl	%r0, -0x100000000
	jasl	%r0, -0x100000000
#CHECK: brasl	%r0, .[[LAB:L.*]]-4294967296 # encoding: [0xc0,0x05,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-4294967296)+2, kind: FK_390_PC32DBL
#CHECK: brasl	%r0, .[[LAB:L.*]]-4294967296 # encoding: [0xc0,0x05,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-4294967296)+2, kind: FK_390_PC32DBL
	brasl	%r0, .-0x100000000
	jasl	%r0, .-0x100000000
#CHECK: brasl	%r0, .[[LAB:L.*]]-2	# encoding: [0xc0,0x05,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC32DBL
#CHECK: brasl	%r0, .[[LAB:L.*]]-2	# encoding: [0xc0,0x05,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC32DBL
	brasl	%r0, -2
	jasl	%r0, -2
#CHECK: brasl	%r0, .[[LAB:L.*]]	# encoding: [0xc0,0x05,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC32DBL
#CHECK: brasl	%r0, .[[LAB:L.*]]	# encoding: [0xc0,0x05,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC32DBL
	brasl	%r0, 0
	jasl	%r0, 0
#CHECK: brasl	%r0, .[[LAB:L.*]]+4294967294 # encoding: [0xc0,0x05,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+4294967294)+2, kind: FK_390_PC32DBL
#CHECK: brasl	%r0, .[[LAB:L.*]]+4294967294 # encoding: [0xc0,0x05,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+4294967294)+2, kind: FK_390_PC32DBL
	brasl	%r0, 0xfffffffe
	jasl	%r0, 0xfffffffe

#CHECK: brasl	%r0, foo                # encoding: [0xc0,0x05,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: brasl	%r14, foo               # encoding: [0xc0,0xe5,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: brasl	%r15, foo               # encoding: [0xc0,0xf5,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: brasl	%r0, foo                # encoding: [0xc0,0x05,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: brasl	%r14, foo               # encoding: [0xc0,0xe5,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: brasl	%r15, foo               # encoding: [0xc0,0xf5,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
	brasl	%r0,foo
	brasl	%r14,foo
	brasl	%r15,foo
	jasl	%r0,foo
	jasl	%r14,foo
	jasl	%r15,foo

#CHECK: brasl	%r0, bar+100                # encoding: [0xc0,0x05,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: brasl	%r14, bar+100               # encoding: [0xc0,0xe5,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: brasl	%r15, bar+100               # encoding: [0xc0,0xf5,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: brasl	%r0, bar+100                # encoding: [0xc0,0x05,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: brasl	%r14, bar+100               # encoding: [0xc0,0xe5,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: brasl	%r15, bar+100               # encoding: [0xc0,0xf5,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
	brasl	%r0,bar+100
	brasl	%r14,bar+100
	brasl	%r15,bar+100
	jasl	%r0,bar+100
	jasl	%r14,bar+100
	jasl	%r15,bar+100

#CHECK: brasl	%r0, bar@PLT                # encoding: [0xc0,0x05,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC32DBL
#CHECK: brasl	%r14, bar@PLT               # encoding: [0xc0,0xe5,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC32DBL
#CHECK: brasl	%r15, bar@PLT               # encoding: [0xc0,0xf5,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC32DBL
#CHECK: brasl	%r0, bar@PLT                # encoding: [0xc0,0x05,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC32DBL
#CHECK: brasl	%r14, bar@PLT               # encoding: [0xc0,0xe5,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC32DBL
#CHECK: brasl	%r15, bar@PLT               # encoding: [0xc0,0xf5,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC32DBL
	brasl	%r0,bar@PLT
	brasl	%r14,bar@PLT
	brasl	%r15,bar@PLT
	jasl	%r0,bar@PLT
	jasl	%r14,bar@PLT
	jasl	%r15,bar@PLT

#CHECK: brc	0, .[[LAB:L.*]]-65536	# encoding: [0xa7,0x04,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-65536)+2, kind: FK_390_PC16DBL
#CHECK: jnop	.[[LAB:L.*]]-65536	# encoding: [0xa7,0x04,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-65536)+2, kind: FK_390_PC16DBL
	brc	0, -0x10000
	jnop	-0x10000

#CHECK: brc	0, .[[LAB:L.*]]-2	# encoding: [0xa7,0x04,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC16DBL
#CHECK: jnop	.[[LAB:L.*]]-2	# encoding: [0xa7,0x04,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC16DBL
	brc	0, -2
	jnop	-2

#CHECK: brc	0, .[[LAB:L.*]]		# encoding: [0xa7,0x04,A,A]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC16DBL
#CHECK: jnop	.[[LAB:L.*]]		# encoding: [0xa7,0x04,A,A]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC16DBL
	brc	0, 0
	jnop	0

#CHECK: brc	0, .[[LAB:L.*]]+65534	# encoding: [0xa7,0x04,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+65534)+2, kind: FK_390_PC16DBL
#CHECK: jnop	.[[LAB:L.*]]+65534	# encoding: [0xa7,0x04,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+65534)+2, kind: FK_390_PC16DBL
	brc	0, 0xfffe
	jnop	0xfffe

#CHECK: brc	0, foo                  # encoding: [0xa7,0x04,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: jnop	foo                     # encoding: [0xa7,0x04,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	brc	0, foo
	jnop	foo

#CHECK: brc	1, foo                  # encoding: [0xa7,0x14,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: jo	foo                     # encoding: [0xa7,0x14,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: jo	foo                     # encoding: [0xa7,0x14,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	brc	1, foo
	jo	foo
	bro	foo

#CHECK: brc	2, foo                  # encoding: [0xa7,0x24,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: brc	2, foo                  # encoding: [0xa7,0x24,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: jh	foo                     # encoding: [0xa7,0x24,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: jp	foo                     # encoding: [0xa7,0x24,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: jh	foo                     # encoding: [0xa7,0x24,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: jp	foo                     # encoding: [0xa7,0x24,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	brc	2, foo
	jc	2, foo
	jh	foo
	jp	foo
	brh	foo
	brp	foo

#CHECK: brc	3, foo                  # encoding: [0xa7,0x34,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: jnle	foo                     # encoding: [0xa7,0x34,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: jnle	foo                     # encoding: [0xa7,0x34,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	brc	3, foo
	jnle	foo
	brnle	foo

#CHECK: brc	4, foo                  # encoding: [0xa7,0x44,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: jl	foo                     # encoding: [0xa7,0x44,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: jm	foo                     # encoding: [0xa7,0x44,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: jl	foo                     # encoding: [0xa7,0x44,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: jm	foo                     # encoding: [0xa7,0x44,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	brc	4, foo
	jl	foo
	jm	foo
	brl	foo
	brm	foo

#CHECK: brc	5, foo                  # encoding: [0xa7,0x54,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: jnhe	foo                     # encoding: [0xa7,0x54,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: jnhe	foo                     # encoding: [0xa7,0x54,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	brc	5, foo
	jnhe	foo
	brnhe	foo

#CHECK: brc	6, foo                  # encoding: [0xa7,0x64,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: jlh	foo                     # encoding: [0xa7,0x64,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: jlh	foo                     # encoding: [0xa7,0x64,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	brc	6, foo
	jlh	foo
	brlh	foo

#CHECK: brc	7, foo                  # encoding: [0xa7,0x74,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: jne	foo                     # encoding: [0xa7,0x74,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: jnz	foo                     # encoding: [0xa7,0x74,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: jne	foo                     # encoding: [0xa7,0x74,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: jnz	foo                     # encoding: [0xa7,0x74,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	brc	7, foo
	jne	foo
	jnz	foo
	brne	foo
	brnz	foo


#CHECK: brc	8, foo                  # encoding: [0xa7,0x84,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: je	foo                     # encoding: [0xa7,0x84,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: jz	foo                     # encoding: [0xa7,0x84,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: je	foo                     # encoding: [0xa7,0x84,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: jz	foo                     # encoding: [0xa7,0x84,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	brc	8, foo
	je	foo
	jz	foo
	bre	foo
	brz	foo

#CHECK: brc	9, foo                  # encoding: [0xa7,0x94,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: jnlh	foo                     # encoding: [0xa7,0x94,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: jnlh	foo                     # encoding: [0xa7,0x94,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	brc	9, foo
	jnlh	foo
	brnlh	foo

#CHECK: brc	10, foo                 # encoding: [0xa7,0xa4,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: jhe	foo                     # encoding: [0xa7,0xa4,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: jhe	foo                     # encoding: [0xa7,0xa4,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	brc	10, foo
	jhe	foo
	brhe	foo

#CHECK: brc	11, foo                 # encoding: [0xa7,0xb4,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: jnl	foo                     # encoding: [0xa7,0xb4,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: jnm	foo                     # encoding: [0xa7,0xb4,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: jnl	foo                     # encoding: [0xa7,0xb4,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: jnm	foo                     # encoding: [0xa7,0xb4,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	brc	11, foo
	jnl	foo
	jnm	foo
	brnl	foo
	brnm	foo

#CHECK: brc	12, foo                 # encoding: [0xa7,0xc4,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: jle	foo                     # encoding: [0xa7,0xc4,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: jle	foo                     # encoding: [0xa7,0xc4,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	brc	12, foo
	jle	foo
	brle	foo

#CHECK: brc	13, foo                 # encoding: [0xa7,0xd4,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: jnh	foo                     # encoding: [0xa7,0xd4,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: jnp	foo                     # encoding: [0xa7,0xd4,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: jnh	foo                     # encoding: [0xa7,0xd4,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: jnp	foo                     # encoding: [0xa7,0xd4,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	brc	13, foo
	jnh	foo
	jnp	foo
	brnh	foo
	brnp	foo

#CHECK: brc	14, foo                 # encoding: [0xa7,0xe4,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: jno	foo                     # encoding: [0xa7,0xe4,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: jno	foo                     # encoding: [0xa7,0xe4,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	brc	14, foo
	jno	foo
	brno	foo

#CHECK: brc	15, foo                 # encoding: [0xa7,0xf4,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: j	foo                     # encoding: [0xa7,0xf4,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: j	foo                     # encoding: [0xa7,0xf4,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	brc	15, foo
	j	foo
	bru	foo

#CHECK: brc	0, bar+100              # encoding: [0xa7,0x04,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
#CHECK: jnop	bar+100                 # encoding: [0xa7,0x04,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	brc	0, bar+100
	jnop	bar+100

#CHECK: jo	bar+100                 # encoding: [0xa7,0x14,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
#CHECK: jo	bar+100                 # encoding: [0xa7,0x14,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	jo	bar+100
	bro	bar+100

#CHECK: jh	bar+100                 # encoding: [0xa7,0x24,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
#CHECK: jh	bar+100                 # encoding: [0xa7,0x24,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	jh	bar+100
	brh	bar+100

#CHECK: jnle	bar+100                 # encoding: [0xa7,0x34,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
#CHECK: jnle	bar+100                 # encoding: [0xa7,0x34,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	jnle	bar+100
	brnle	bar+100

#CHECK: jl	bar+100                 # encoding: [0xa7,0x44,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
#CHECK: jl	bar+100                 # encoding: [0xa7,0x44,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	jl	bar+100
	brl	bar+100

#CHECK: jnhe	bar+100                 # encoding: [0xa7,0x54,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
#CHECK: jnhe	bar+100                 # encoding: [0xa7,0x54,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	jnhe	bar+100
	brnhe	bar+100

#CHECK: jlh	bar+100                 # encoding: [0xa7,0x64,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
#CHECK: jlh	bar+100                 # encoding: [0xa7,0x64,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	jlh	bar+100
	brlh	bar+100

#CHECK: jne	bar+100                 # encoding: [0xa7,0x74,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
#CHECK: jne	bar+100                 # encoding: [0xa7,0x74,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	jne	bar+100
	brne	bar+100

#CHECK: je	bar+100                 # encoding: [0xa7,0x84,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
#CHECK: je	bar+100                 # encoding: [0xa7,0x84,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	je	bar+100
	bre	bar+100

#CHECK: jnlh	bar+100                 # encoding: [0xa7,0x94,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
#CHECK: jnlh	bar+100                 # encoding: [0xa7,0x94,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	jnlh	bar+100
	brnlh	bar+100

#CHECK: jhe	bar+100                 # encoding: [0xa7,0xa4,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
#CHECK: jhe	bar+100                 # encoding: [0xa7,0xa4,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	jhe	bar+100
	brhe	bar+100

#CHECK: jnl	bar+100                 # encoding: [0xa7,0xb4,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
#CHECK: jnl	bar+100                 # encoding: [0xa7,0xb4,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	jnl	bar+100
	brnl	bar+100

#CHECK: jle	bar+100                 # encoding: [0xa7,0xc4,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
#CHECK: jle	bar+100                 # encoding: [0xa7,0xc4,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	jle	bar+100
	brle	bar+100

#CHECK: jnh	bar+100                 # encoding: [0xa7,0xd4,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
#CHECK: jnh	bar+100                 # encoding: [0xa7,0xd4,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	jnh	bar+100
	brnh	bar+100

#CHECK: jno	bar+100                 # encoding: [0xa7,0xe4,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
#CHECK: jno	bar+100                 # encoding: [0xa7,0xe4,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	jno	bar+100
	brno	bar+100

#CHECK: j	bar+100                 # encoding: [0xa7,0xf4,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
#CHECK: j	bar+100                 # encoding: [0xa7,0xf4,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	j	bar+100
	bru	bar+100

#CHECK: brc	0, bar@PLT              # encoding: [0xa7,0x04,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
#CHECK: jnop	bar@PLT                 # encoding: [0xa7,0x04,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	brc	0, bar@PLT
	jnop	bar@PLT

#CHECK: jo	bar@PLT                 # encoding: [0xa7,0x14,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
#CHECK: jo	bar@PLT                 # encoding: [0xa7,0x14,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	jo	bar@PLT
	bro	bar@PLT

#CHECK: jh	bar@PLT                 # encoding: [0xa7,0x24,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
#CHECK: jh	bar@PLT                 # encoding: [0xa7,0x24,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	jh	bar@PLT
	brh	bar@PLT

#CHECK: jnle	bar@PLT                 # encoding: [0xa7,0x34,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
#CHECK: jnle	bar@PLT                 # encoding: [0xa7,0x34,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	jnle	bar@PLT
	brnle	bar@PLT

#CHECK: jl	bar@PLT                 # encoding: [0xa7,0x44,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
#CHECK: jl	bar@PLT                 # encoding: [0xa7,0x44,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	jl	bar@PLT
	brl	bar@PLT

#CHECK: jnhe	bar@PLT                 # encoding: [0xa7,0x54,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
#CHECK: jnhe	bar@PLT                 # encoding: [0xa7,0x54,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	jnhe	bar@PLT
	brnhe	bar@PLT

#CHECK: jlh	bar@PLT                 # encoding: [0xa7,0x64,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
#CHECK: jlh	bar@PLT                 # encoding: [0xa7,0x64,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	jlh	bar@PLT
	brlh	bar@PLT

#CHECK: jne	bar@PLT                 # encoding: [0xa7,0x74,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
#CHECK: jne	bar@PLT                 # encoding: [0xa7,0x74,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	jne	bar@PLT
	brne	bar@PLT

#CHECK: je	bar@PLT                 # encoding: [0xa7,0x84,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
#CHECK: je	bar@PLT                 # encoding: [0xa7,0x84,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	je	bar@PLT
	bre	bar@PLT

#CHECK: jnlh	bar@PLT                 # encoding: [0xa7,0x94,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
#CHECK: jnlh	bar@PLT                 # encoding: [0xa7,0x94,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	jnlh	bar@PLT
	brnlh	bar@PLT

#CHECK: jhe	bar@PLT                 # encoding: [0xa7,0xa4,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
#CHECK: jhe	bar@PLT                 # encoding: [0xa7,0xa4,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	jhe	bar@PLT
	brhe	bar@PLT

#CHECK: jnl	bar@PLT                 # encoding: [0xa7,0xb4,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
#CHECK: jnl	bar@PLT                 # encoding: [0xa7,0xb4,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	jnl	bar@PLT
	brnl	bar@PLT

#CHECK: jle	bar@PLT                 # encoding: [0xa7,0xc4,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
#CHECK: jle	bar@PLT                 # encoding: [0xa7,0xc4,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	jle	bar@PLT
	brle	bar@PLT

#CHECK: jnh	bar@PLT                 # encoding: [0xa7,0xd4,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
#CHECK: jnh	bar@PLT                 # encoding: [0xa7,0xd4,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	jnh	bar@PLT
	brnh	bar@PLT

#CHECK: jno	bar@PLT                 # encoding: [0xa7,0xe4,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
#CHECK: jno	bar@PLT                 # encoding: [0xa7,0xe4,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	jno	bar@PLT
	brno	bar@PLT

#CHECK: j	bar@PLT                 # encoding: [0xa7,0xf4,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
#CHECK: j	bar@PLT                 # encoding: [0xa7,0xf4,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	j	bar@PLT
	bru	bar@PLT

#CHECK: brcl	0, .[[LAB:L.*]]-4294967296 # encoding: [0xc0,0x04,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-4294967296)+2, kind: FK_390_PC32DBL
#CHECK: jgnop	.[[LAB:L.*]]-4294967296    # encoding: [0xc0,0x04,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-4294967296)+2, kind: FK_390_PC32DBL
	brcl	0, -0x100000000
	jgnop	-0x100000000
#CHECK: brcl	0, .[[LAB:L.*]]-2	# encoding: [0xc0,0x04,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC32DBL
#CHECK: jgnop	.[[LAB:L.*]]-2		# encoding: [0xc0,0x04,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC32DBL
	brcl	0, -2
	jgnop	-2
#CHECK: brcl	0, .[[LAB:L.*]]		# encoding: [0xc0,0x04,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC32DBL
#CHECK: jgnop	.[[LAB:L.*]]		# encoding: [0xc0,0x04,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC32DBL
	brcl	0, 0
	jgnop	0
#CHECK: brcl	0, .[[LAB:L.*]]+4294967294 # encoding: [0xc0,0x04,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+4294967294)+2, kind: FK_390_PC32DBL
#CHECK: jgnop	.[[LAB:L.*]]+4294967294    # encoding: [0xc0,0x04,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+4294967294)+2, kind: FK_390_PC32DBL
	brcl	0, 0xfffffffe
	jgnop	0xfffffffe

#CHECK: brcl	0, foo                  # encoding: [0xc0,0x04,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: jgnop	foo                     # encoding: [0xc0,0x04,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
	brcl	0, foo
	jgnop	foo

#CHECK: brcl	1, foo                  # encoding: [0xc0,0x14,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: jgo	foo                     # encoding: [0xc0,0x14,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: jgo	foo                     # encoding: [0xc0,0x14,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
	brcl	1, foo
	jgo	foo
	brol	foo

#CHECK: brcl	2, foo                  # encoding: [0xc0,0x24,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: jgh	foo                     # encoding: [0xc0,0x24,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: jgp	foo                     # encoding: [0xc0,0x24,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: jgh	foo                     # encoding: [0xc0,0x24,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: jgp	foo                     # encoding: [0xc0,0x24,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
	brcl	2, foo
	jgh	foo
	jgp	foo
	brhl	foo
	brpl	foo

#CHECK: brcl	3, foo                  # encoding: [0xc0,0x34,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: jgnle	foo                     # encoding: [0xc0,0x34,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: jgnle	foo                     # encoding: [0xc0,0x34,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
	brcl	3, foo
	jgnle	foo
	brnlel	foo

#CHECK: brcl	4, foo                  # encoding: [0xc0,0x44,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: jgl	foo                     # encoding: [0xc0,0x44,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: jgm	foo                     # encoding: [0xc0,0x44,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: jgl	foo                     # encoding: [0xc0,0x44,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: jgm	foo                     # encoding: [0xc0,0x44,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
	brcl	4, foo
	jgl	foo
	jgm	foo
	brll	foo
	brml	foo

#CHECK: brcl	5, foo                  # encoding: [0xc0,0x54,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: jgnhe	foo                     # encoding: [0xc0,0x54,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: jgnhe	foo                     # encoding: [0xc0,0x54,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
	brcl	5, foo
	jgnhe	foo
	brnhel	foo

#CHECK: brcl	6, foo                  # encoding: [0xc0,0x64,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: jglh	foo                     # encoding: [0xc0,0x64,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: jglh	foo                     # encoding: [0xc0,0x64,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
	brcl	6, foo
	jglh	foo
	brlhl	foo

#CHECK: brcl	7, foo                  # encoding: [0xc0,0x74,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: jgne	foo                     # encoding: [0xc0,0x74,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: jgnz	foo                     # encoding: [0xc0,0x74,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: jgne	foo                     # encoding: [0xc0,0x74,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: jgnz	foo                     # encoding: [0xc0,0x74,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
	brcl	7, foo
	jgne	foo
	jgnz	foo
	brnel	foo
	brnzl	foo

#CHECK: brcl	8, foo                  # encoding: [0xc0,0x84,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: jge	foo                     # encoding: [0xc0,0x84,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: jgz	foo                     # encoding: [0xc0,0x84,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: jge	foo                     # encoding: [0xc0,0x84,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: jgz	foo                     # encoding: [0xc0,0x84,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
	brcl	8, foo
	jge	foo
	jgz	foo
	brel	foo
	brzl	foo

#CHECK: brcl	9, foo                  # encoding: [0xc0,0x94,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: jgnlh	foo                     # encoding: [0xc0,0x94,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: jgnlh	foo                     # encoding: [0xc0,0x94,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
	brcl	9, foo
	jgnlh	foo
	brnlhl	foo

#CHECK: brcl	10, foo                 # encoding: [0xc0,0xa4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: jghe	foo                     # encoding: [0xc0,0xa4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: jghe	foo                     # encoding: [0xc0,0xa4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
	brcl	10, foo
	jghe	foo
	brhel	foo

#CHECK: brcl	11, foo                 # encoding: [0xc0,0xb4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: jgnl	foo                     # encoding: [0xc0,0xb4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: jgnm	foo                     # encoding: [0xc0,0xb4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: jgnl	foo                     # encoding: [0xc0,0xb4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: jgnm	foo                     # encoding: [0xc0,0xb4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
	brcl	11, foo
	jgnl	foo
	jgnm	foo
	brnll	foo
	brnml	foo

#CHECK: brcl	12, foo                 # encoding: [0xc0,0xc4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: jgle	foo                     # encoding: [0xc0,0xc4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: jgle	foo                     # encoding: [0xc0,0xc4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
	brcl	12, foo
	jgle	foo
	brlel	foo

#CHECK: brcl	13, foo                 # encoding: [0xc0,0xd4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: jgnh	foo                     # encoding: [0xc0,0xd4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: jgnp	foo                     # encoding: [0xc0,0xd4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: jgnh	foo                     # encoding: [0xc0,0xd4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: jgnp	foo                     # encoding: [0xc0,0xd4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
	brcl	13, foo
	jgnh	foo
	jgnp	foo
	brnhl	foo
	brnpl	foo

#CHECK: brcl	14, foo                 # encoding: [0xc0,0xe4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: jgno	foo                     # encoding: [0xc0,0xe4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: jgno	foo                     # encoding: [0xc0,0xe4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
	brcl	14, foo
	jgno	foo
	brnol	foo

#CHECK: brcl	15, foo                 # encoding: [0xc0,0xf4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: jg	foo                     # encoding: [0xc0,0xf4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: jg	foo                     # encoding: [0xc0,0xf4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
	brcl	15, foo
	jg	foo
	brul	foo

#CHECK: brcl	0, bar+100              # encoding: [0xc0,0x04,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: jgnop	bar+100                 # encoding: [0xc0,0x04,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
	brcl	0, bar+100
	jgnop	bar+100

#CHECK: jgo	bar+100                 # encoding: [0xc0,0x14,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: jgo	bar+100                 # encoding: [0xc0,0x14,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
	jgo	bar+100
	brol	bar+100

#CHECK: jgh	bar+100                 # encoding: [0xc0,0x24,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: jgh	bar+100                 # encoding: [0xc0,0x24,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
	jgh	bar+100
	brhl	bar+100

#CHECK: jgnle	bar+100                 # encoding: [0xc0,0x34,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: jgnle	bar+100                 # encoding: [0xc0,0x34,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
	jgnle	bar+100
	brnlel	bar+100

#CHECK: jgl	bar+100                 # encoding: [0xc0,0x44,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: jgl	bar+100                 # encoding: [0xc0,0x44,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
	jgl	bar+100
	brll	bar+100

#CHECK: jgnhe	bar+100                 # encoding: [0xc0,0x54,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: jgnhe	bar+100                 # encoding: [0xc0,0x54,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
	jgnhe	bar+100
	brnhel	bar+100

#CHECK: jglh	bar+100                 # encoding: [0xc0,0x64,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: jglh	bar+100                 # encoding: [0xc0,0x64,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
	jglh	bar+100
	brlhl	bar+100

#CHECK: jgne	bar+100                 # encoding: [0xc0,0x74,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: jgne	bar+100                 # encoding: [0xc0,0x74,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
	jgne	bar+100
	brnel	bar+100

#CHECK: jge	bar+100                 # encoding: [0xc0,0x84,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: jge	bar+100                 # encoding: [0xc0,0x84,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
	jge	bar+100
	brel	bar+100

#CHECK: jgnlh	bar+100                 # encoding: [0xc0,0x94,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: jgnlh	bar+100                 # encoding: [0xc0,0x94,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
	jgnlh	bar+100
	brnlhl	bar+100

#CHECK: jghe	bar+100                 # encoding: [0xc0,0xa4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: jghe	bar+100                 # encoding: [0xc0,0xa4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
	jghe	bar+100
	brhel	bar+100

#CHECK: jgnl	bar+100                 # encoding: [0xc0,0xb4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: jgnl	bar+100                 # encoding: [0xc0,0xb4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
	jgnl	bar+100
	brnll	bar+100

#CHECK: jgle	bar+100                 # encoding: [0xc0,0xc4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: jgle	bar+100                 # encoding: [0xc0,0xc4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
	jgle	bar+100
	brlel	bar+100

#CHECK: jgnh	bar+100                 # encoding: [0xc0,0xd4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: jgnh	bar+100                 # encoding: [0xc0,0xd4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
	jgnh	bar+100
	brnhl	bar+100

#CHECK: jgno	bar+100                 # encoding: [0xc0,0xe4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: jgno	bar+100                 # encoding: [0xc0,0xe4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
	jgno	bar+100
	brnol	bar+100

#CHECK: jg	bar+100                 # encoding: [0xc0,0xf4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: jg	bar+100                 # encoding: [0xc0,0xf4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
	jg	bar+100
	brul	bar+100

#CHECK: brcl	0, bar@PLT              # encoding: [0xc0,0x04,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC32DBL
#CHECK: jgnop	bar@PLT                 # encoding: [0xc0,0x04,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC32DBL
	brcl	0, bar@PLT
	jgnop	bar@PLT

#CHECK: jgo	bar@PLT                 # encoding: [0xc0,0x14,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC32DBL
#CHECK: jgo	bar@PLT                 # encoding: [0xc0,0x14,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC32DBL
	jgo	bar@PLT
	brol	bar@PLT

#CHECK: jgh	bar@PLT                 # encoding: [0xc0,0x24,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC32DBL
#CHECK: jgh	bar@PLT                 # encoding: [0xc0,0x24,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC32DBL
	jgh	bar@PLT
	brhl	bar@PLT

#CHECK: jgnle	bar@PLT                 # encoding: [0xc0,0x34,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC32DBL
#CHECK: jgnle	bar@PLT                 # encoding: [0xc0,0x34,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC32DBL
	jgnle	bar@PLT
	brnlel	bar@PLT

#CHECK: jgl	bar@PLT                 # encoding: [0xc0,0x44,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC32DBL
#CHECK: jgl	bar@PLT                 # encoding: [0xc0,0x44,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC32DBL
	jgl	bar@PLT
	brll	bar@PLT

#CHECK: jgnhe	bar@PLT                 # encoding: [0xc0,0x54,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC32DBL
#CHECK: jgnhe	bar@PLT                 # encoding: [0xc0,0x54,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC32DBL
	jgnhe	bar@PLT
	brnhel	bar@PLT

#CHECK: jglh	bar@PLT                 # encoding: [0xc0,0x64,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC32DBL
#CHECK: jglh	bar@PLT                 # encoding: [0xc0,0x64,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC32DBL
	jglh	bar@PLT
	brlhl	bar@PLT

#CHECK: jgne	bar@PLT                 # encoding: [0xc0,0x74,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC32DBL
#CHECK: jgne	bar@PLT                 # encoding: [0xc0,0x74,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC32DBL
	jgne	bar@PLT
	brnel	bar@PLT

#CHECK: jge	bar@PLT                 # encoding: [0xc0,0x84,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC32DBL
#CHECK: jge	bar@PLT                 # encoding: [0xc0,0x84,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC32DBL
	jge	bar@PLT
	brel	bar@PLT

#CHECK: jgnlh	bar@PLT                 # encoding: [0xc0,0x94,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC32DBL
#CHECK: jgnlh	bar@PLT                 # encoding: [0xc0,0x94,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC32DBL
	jgnlh	bar@PLT
	brnlhl	bar@PLT

#CHECK: jghe	bar@PLT                 # encoding: [0xc0,0xa4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC32DBL
#CHECK: jghe	bar@PLT                 # encoding: [0xc0,0xa4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC32DBL
	jghe	bar@PLT
	brhel	bar@PLT

#CHECK: jgnl	bar@PLT                 # encoding: [0xc0,0xb4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC32DBL
#CHECK: jgnl	bar@PLT                 # encoding: [0xc0,0xb4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC32DBL
	jgnl	bar@PLT
	brnll	bar@PLT

#CHECK: jgle	bar@PLT                 # encoding: [0xc0,0xc4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC32DBL
#CHECK: jgle	bar@PLT                 # encoding: [0xc0,0xc4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC32DBL
	jgle	bar@PLT
	brlel	bar@PLT

#CHECK: jgnh	bar@PLT                 # encoding: [0xc0,0xd4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC32DBL
#CHECK: jgnh	bar@PLT                 # encoding: [0xc0,0xd4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC32DBL
	jgnh	bar@PLT
	brnhl	bar@PLT

#CHECK: jgno	bar@PLT                 # encoding: [0xc0,0xe4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC32DBL
#CHECK: jgno	bar@PLT                 # encoding: [0xc0,0xe4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC32DBL
	jgno	bar@PLT
	brnol	bar@PLT

#CHECK: jg	bar@PLT                 # encoding: [0xc0,0xf4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC32DBL
#CHECK: jg	bar@PLT                 # encoding: [0xc0,0xf4,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC32DBL
	jg	bar@PLT
	brul	bar@PLT

#CHECK: bct	%r0, 0                  # encoding: [0x46,0x00,0x00,0x00]
#CHECK: bct	%r0, 4095               # encoding: [0x46,0x00,0x0f,0xff]
#CHECK: bct	%r0, 0(%r1)             # encoding: [0x46,0x00,0x10,0x00]
#CHECK: bct	%r0, 0(%r15)            # encoding: [0x46,0x00,0xf0,0x00]
#CHECK: bct	%r0, 4095(%r1,%r15)     # encoding: [0x46,0x01,0xff,0xff]
#CHECK: bct	%r0, 4095(%r15,%r1)     # encoding: [0x46,0x0f,0x1f,0xff]
#CHECK: bct	%r15, 0                 # encoding: [0x46,0xf0,0x00,0x00]

	bct	%r0, 0
	bct	%r0, 4095
	bct	%r0, 0(%r1)
	bct	%r0, 0(%r15)
	bct	%r0, 4095(%r1,%r15)
	bct	%r0, 4095(%r15,%r1)
	bct	%r15, 0

#CHECK: bctr	%r0, %r9                # encoding: [0x06,0x09]
#CHECK: bctr	%r0, %r15               # encoding: [0x06,0x0f]
#CHECK: bctr	%r15, %r0               # encoding: [0x06,0xf0]
#CHECK: bctr	%r15, %r9               # encoding: [0x06,0xf9]

	bctr	%r0,%r9
	bctr	%r0,%r15
	bctr	%r15,%r0
	bctr	%r15,%r9

#CHECK: bctg	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x46]
#CHECK: bctg	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x46]
#CHECK: bctg	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x46]
#CHECK: bctg	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x46]
#CHECK: bctg	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x46]
#CHECK: bctg	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x46]
#CHECK: bctg	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x46]
#CHECK: bctg	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x46]
#CHECK: bctg	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x46]
#CHECK: bctg	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x46]

	bctg	%r0, -524288
	bctg	%r0, -1
	bctg	%r0, 0
	bctg	%r0, 1
	bctg	%r0, 524287
	bctg	%r0, 0(%r1)
	bctg	%r0, 0(%r15)
	bctg	%r0, 524287(%r1,%r15)
	bctg	%r0, 524287(%r15,%r1)
	bctg	%r15, 0

#CHECK: bctgr	%r0, %r9                # encoding: [0xb9,0x46,0x00,0x09]
#CHECK: bctgr	%r0, %r15               # encoding: [0xb9,0x46,0x00,0x0f]
#CHECK: bctgr	%r15, %r0               # encoding: [0xb9,0x46,0x00,0xf0]
#CHECK: bctgr	%r15, %r9               # encoding: [0xb9,0x46,0x00,0xf9]

	bctgr	%r0,%r9
	bctgr	%r0,%r15
	bctgr	%r15,%r0
	bctgr	%r15,%r9

#CHECK: brct	%r0, .[[LAB:L.*]]-65536	# encoding: [0xa7,0x06,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-65536)+2, kind: FK_390_PC16DBL
#CHECK: brct	%r0, .[[LAB:L.*]]-65536	# encoding: [0xa7,0x06,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-65536)+2, kind: FK_390_PC16DBL
	brct	%r0, -0x10000
	jct	%r0, -0x10000
#CHECK: brct	%r0, .[[LAB:L.*]]-2	# encoding: [0xa7,0x06,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC16DBL
#CHECK: brct	%r0, .[[LAB:L.*]]-2	# encoding: [0xa7,0x06,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC16DBL
	brct	%r0, -2
	jct	%r0, -2
#CHECK: brct	%r0, .[[LAB:L.*]]	# encoding: [0xa7,0x06,A,A]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC16DBL
#CHECK: brct	%r0, .[[LAB:L.*]]	# encoding: [0xa7,0x06,A,A]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC16DBL
	brct	%r0, 0
	jct	%r0, 0
#CHECK: brct	%r0, .[[LAB:L.*]]+65534	# encoding: [0xa7,0x06,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+65534)+2, kind: FK_390_PC16DBL
#CHECK: brct	%r0, .[[LAB:L.*]]+65534	# encoding: [0xa7,0x06,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+65534)+2, kind: FK_390_PC16DBL
	brct	%r0, 0xfffe
	jct	%r0, 0xfffe
#CHECK: brct	%r15, .[[LAB:L.*]]	# encoding: [0xa7,0xf6,A,A]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC16DBL
#CHECK: brct	%r15, .[[LAB:L.*]]	# encoding: [0xa7,0xf6,A,A]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC16DBL
	brct	%r15, 0
	jct	%r15, 0

#CHECK: brctg	%r0, .[[LAB:L.*]]-65536	# encoding: [0xa7,0x07,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-65536)+2, kind: FK_390_PC16DBL
#CHECK: brctg	%r0, .[[LAB:L.*]]-65536	# encoding: [0xa7,0x07,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-65536)+2, kind: FK_390_PC16DBL
	brctg	%r0, -0x10000
	jctg	%r0, -0x10000
#CHECK: brctg	%r0, .[[LAB:L.*]]-2	# encoding: [0xa7,0x07,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC16DBL
#CHECK: brctg	%r0, .[[LAB:L.*]]-2	# encoding: [0xa7,0x07,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC16DBL
	brctg	%r0, -2
	jctg	%r0, -2
#CHECK: brctg	%r0, .[[LAB:L.*]]	# encoding: [0xa7,0x07,A,A]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC16DBL
#CHECK: brctg	%r0, .[[LAB:L.*]]	# encoding: [0xa7,0x07,A,A]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC16DBL
	brctg	%r0, 0
	jctg	%r0, 0
#CHECK: brctg	%r0, .[[LAB:L.*]]+65534	# encoding: [0xa7,0x07,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+65534)+2, kind: FK_390_PC16DBL
#CHECK: brctg	%r0, .[[LAB:L.*]]+65534	# encoding: [0xa7,0x07,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+65534)+2, kind: FK_390_PC16DBL
	brctg	%r0, 0xfffe
	jctg	%r0, 0xfffe
#CHECK: brctg	%r15, .[[LAB:L.*]]	# encoding: [0xa7,0xf7,A,A]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC16DBL
#CHECK: brctg	%r15, .[[LAB:L.*]]	# encoding: [0xa7,0xf7,A,A]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC16DBL
	brctg	%r15, 0
	jctg	%r15, 0



#CHECK: bxh	%r0, %r0, 0             # encoding: [0x86,0x00,0x00,0x00]
#CHECK: bxh	%r0, %r15, 0            # encoding: [0x86,0x0f,0x00,0x00]
#CHECK: bxh	%r14, %r15, 0           # encoding: [0x86,0xef,0x00,0x00]
#CHECK: bxh	%r15, %r15, 0           # encoding: [0x86,0xff,0x00,0x00]
#CHECK: bxh	%r0, %r0, 4095          # encoding: [0x86,0x00,0x0f,0xff]
#CHECK: bxh	%r0, %r0, 1             # encoding: [0x86,0x00,0x00,0x01]
#CHECK: bxh	%r0, %r0, 0(%r1)        # encoding: [0x86,0x00,0x10,0x00]
#CHECK: bxh	%r0, %r0, 0(%r15)       # encoding: [0x86,0x00,0xf0,0x00]
#CHECK: bxh	%r0, %r0, 4095(%r1)     # encoding: [0x86,0x00,0x1f,0xff]
#CHECK: bxh	%r0, %r0, 4095(%r15)    # encoding: [0x86,0x00,0xff,0xff]

	bxh	%r0,%r0,0
	bxh	%r0,%r15,0
	bxh	%r14,%r15,0
	bxh	%r15,%r15,0
	bxh	%r0,%r0,4095
	bxh	%r0,%r0,1
	bxh	%r0,%r0,0(%r1)
	bxh	%r0,%r0,0(%r15)
	bxh	%r0,%r0,4095(%r1)
	bxh	%r0,%r0,4095(%r15)

#CHECK: bxhg	%r0, %r0, 0             # encoding: [0xeb,0x00,0x00,0x00,0x00,0x44]
#CHECK: bxhg	%r0, %r15, 0            # encoding: [0xeb,0x0f,0x00,0x00,0x00,0x44]
#CHECK: bxhg	%r14, %r15, 0           # encoding: [0xeb,0xef,0x00,0x00,0x00,0x44]
#CHECK: bxhg	%r15, %r15, 0           # encoding: [0xeb,0xff,0x00,0x00,0x00,0x44]
#CHECK: bxhg	%r0, %r0, -524288       # encoding: [0xeb,0x00,0x00,0x00,0x80,0x44]
#CHECK: bxhg	%r0, %r0, -1            # encoding: [0xeb,0x00,0x0f,0xff,0xff,0x44]
#CHECK: bxhg	%r0, %r0, 0             # encoding: [0xeb,0x00,0x00,0x00,0x00,0x44]
#CHECK: bxhg	%r0, %r0, 1             # encoding: [0xeb,0x00,0x00,0x01,0x00,0x44]
#CHECK: bxhg	%r0, %r0, 524287        # encoding: [0xeb,0x00,0x0f,0xff,0x7f,0x44]
#CHECK: bxhg	%r0, %r0, 0(%r1)        # encoding: [0xeb,0x00,0x10,0x00,0x00,0x44]
#CHECK: bxhg	%r0, %r0, 0(%r15)       # encoding: [0xeb,0x00,0xf0,0x00,0x00,0x44]
#CHECK: bxhg	%r0, %r0, 524287(%r1)   # encoding: [0xeb,0x00,0x1f,0xff,0x7f,0x44]
#CHECK: bxhg	%r0, %r0, 524287(%r15)  # encoding: [0xeb,0x00,0xff,0xff,0x7f,0x44]

	bxhg	%r0,%r0,0
	bxhg	%r0,%r15,0
	bxhg	%r14,%r15,0
	bxhg	%r15,%r15,0
	bxhg	%r0,%r0,-524288
	bxhg	%r0,%r0,-1
	bxhg	%r0,%r0,0
	bxhg	%r0,%r0,1
	bxhg	%r0,%r0,524287
	bxhg	%r0,%r0,0(%r1)
	bxhg	%r0,%r0,0(%r15)
	bxhg	%r0,%r0,524287(%r1)
	bxhg	%r0,%r0,524287(%r15)

#CHECK: brxh	%r0, %r2, .[[LAB:L.*]]-65536	# encoding: [0x84,0x02,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-65536)+2, kind: FK_390_PC16DBL
#CHECK: brxh	%r0, %r2, .[[LAB:L.*]]-65536	# encoding: [0x84,0x02,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-65536)+2, kind: FK_390_PC16DBL
	brxh	%r0,%r2, -0x10000
	jxh	%r0,%r2, -0x10000
#CHECK: brxh	%r0, %r2, .[[LAB:L.*]]-2	# encoding: [0x84,0x02,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC16DBL
#CHECK: brxh	%r0, %r2, .[[LAB:L.*]]-2	# encoding: [0x84,0x02,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC16DBL
	brxh	%r0, %r2, -2
	jxh	%r0, %r2, -2
#CHECK: brxh	%r0, %r2, .[[LAB:L.*]]	        # encoding: [0x84,0x02,A,A]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC16DBL
#CHECK: brxh	%r0, %r2, .[[LAB:L.*]]	        # encoding: [0x84,0x02,A,A]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC16DBL
	brxh	%r0,%r2, 0
	jxh	%r0,%r2, 0
#CHECK: brxh	%r0, %r2, .[[LAB:L.*]]+65534	# encoding: [0x84,0x02,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+65534)+2, kind: FK_390_PC16DBL
#CHECK: brxh	%r0, %r2, .[[LAB:L.*]]+65534	# encoding: [0x84,0x02,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+65534)+2, kind: FK_390_PC16DBL
	brxh	%r0,%r2, 0xfffe
	jxh	%r0,%r2, 0xfffe

#CHECK: brxh	%r0, %r2, foo                   # encoding: [0x84,0x02,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: brxh	%r14, %r2, foo                  # encoding: [0x84,0xe2,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: brxh	%r15, %r2, foo                  # encoding: [0x84,0xf2,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: brxh	%r0, %r2, foo                   # encoding: [0x84,0x02,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: brxh	%r14, %r2, foo                  # encoding: [0x84,0xe2,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: brxh	%r15, %r2, foo                  # encoding: [0x84,0xf2,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	brxh	%r0,%r2,foo
	brxh	%r14,%r2,foo
	brxh	%r15,%r2,foo
	jxh	%r0,%r2,foo
	jxh	%r14,%r2,foo
	jxh	%r15,%r2,foo

#CHECK: brxh	%r0, %r2, bar+100               # encoding: [0x84,0x02,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
#CHECK: brxh	%r14, %r2, bar+100              # encoding: [0x84,0xe2,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
#CHECK: brxh	%r15, %r2, bar+100              # encoding: [0x84,0xf2,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
#CHECK: brxh	%r0, %r2, bar+100               # encoding: [0x84,0x02,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
#CHECK: brxh	%r14, %r2, bar+100              # encoding: [0x84,0xe2,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
#CHECK: brxh	%r15, %r2, bar+100              # encoding: [0x84,0xf2,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	brxh	%r0,%r2,bar+100
	brxh	%r14,%r2,bar+100
	brxh	%r15,%r2,bar+100
	jxh	%r0,%r2,bar+100
	jxh	%r14,%r2,bar+100
	jxh	%r15,%r2,bar+100

#CHECK: brxh	%r0, %r2, bar@PLT               # encoding: [0x84,0x02,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
#CHECK: brxh	%r14, %r2, bar@PLT              # encoding: [0x84,0xe2,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
#CHECK: brxh	%r15, %r2, bar@PLT              # encoding: [0x84,0xf2,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
#CHECK: brxh	%r0, %r2, bar@PLT               # encoding: [0x84,0x02,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
#CHECK: brxh	%r14, %r2, bar@PLT              # encoding: [0x84,0xe2,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
#CHECK: brxh	%r15, %r2, bar@PLT              # encoding: [0x84,0xf2,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	brxh	%r0,%r2,bar@PLT
	brxh	%r14,%r2,bar@PLT
	brxh	%r15,%r2,bar@PLT
	jxh	%r0,%r2,bar@PLT
	jxh	%r14,%r2,bar@PLT
	jxh	%r15,%r2,bar@PLT

#CHECK: brxhg	%r0, %r2, .[[LAB:L.*]]-65536	# encoding: [0xec,0x02,A,A,0x00,0x44]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-65536)+2, kind: FK_390_PC16DBL
#CHECK: brxhg	%r0, %r2, .[[LAB:L.*]]-65536	# encoding: [0xec,0x02,A,A,0x00,0x44]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-65536)+2, kind: FK_390_PC16DBL
	brxhg	%r0,%r2, -0x10000
	jxhg	%r0,%r2, -0x10000
#CHECK: brxhg	%r0, %r2, .[[LAB:L.*]]-2	# encoding: [0xec,0x02,A,A,0x00,0x44]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC16DBL
#CHECK: brxhg	%r0, %r2, .[[LAB:L.*]]-2	# encoding: [0xec,0x02,A,A,0x00,0x44]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC16DBL
	brxhg	%r0, %r2, -2
	jxhg	%r0, %r2, -2
#CHECK: brxhg	%r0, %r2, .[[LAB:L.*]]	        # encoding: [0xec,0x02,A,A,0x00,0x44]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC16DBL
#CHECK: brxhg	%r0, %r2, .[[LAB:L.*]]	        # encoding: [0xec,0x02,A,A,0x00,0x44]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC16DBL
	brxhg	%r0,%r2, 0
	jxhg	%r0,%r2, 0
#CHECK: brxhg	%r0, %r2, .[[LAB:L.*]]+65534	# encoding: [0xec,0x02,A,A,0x00,0x44]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+65534)+2, kind: FK_390_PC16DBL
#CHECK: brxhg	%r0, %r2, .[[LAB:L.*]]+65534	# encoding: [0xec,0x02,A,A,0x00,0x44]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+65534)+2, kind: FK_390_PC16DBL
	brxhg	%r0,%r2, 0xfffe
	jxhg	%r0,%r2, 0xfffe

#CHECK: brxhg	%r0, %r2, foo                   # encoding: [0xec,0x02,A,A,0x00,0x44]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: brxhg	%r14, %r2, foo                  # encoding: [0xec,0xe2,A,A,0x00,0x44]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: brxhg	%r15, %r2, foo                  # encoding: [0xec,0xf2,A,A,0x00,0x44]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: brxhg	%r0, %r2, foo                   # encoding: [0xec,0x02,A,A,0x00,0x44]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: brxhg	%r14, %r2, foo                  # encoding: [0xec,0xe2,A,A,0x00,0x44]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: brxhg	%r15, %r2, foo                  # encoding: [0xec,0xf2,A,A,0x00,0x44]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	brxhg	%r0,%r2,foo
	brxhg	%r14,%r2,foo
	brxhg	%r15,%r2,foo
	jxhg	%r0,%r2,foo
	jxhg	%r14,%r2,foo
	jxhg	%r15,%r2,foo

#CHECK: brxhg	%r0, %r2, bar+100               # encoding: [0xec,0x02,A,A,0x00,0x44]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
#CHECK: brxhg	%r14, %r2, bar+100              # encoding: [0xec,0xe2,A,A,0x00,0x44]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
#CHECK: brxhg	%r15, %r2, bar+100              # encoding: [0xec,0xf2,A,A,0x00,0x44]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
#CHECK: brxhg	%r0, %r2, bar+100               # encoding: [0xec,0x02,A,A,0x00,0x44]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
#CHECK: brxhg	%r14, %r2, bar+100              # encoding: [0xec,0xe2,A,A,0x00,0x44]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
#CHECK: brxhg	%r15, %r2, bar+100              # encoding: [0xec,0xf2,A,A,0x00,0x44]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	brxhg	%r0,%r2,bar+100
	brxhg	%r14,%r2,bar+100
	brxhg	%r15,%r2,bar+100
	jxhg	%r0,%r2,bar+100
	jxhg	%r14,%r2,bar+100
	jxhg	%r15,%r2,bar+100

#CHECK: brxhg	%r0, %r2, bar@PLT               # encoding: [0xec,0x02,A,A,0x00,0x44]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
#CHECK: brxhg	%r14, %r2, bar@PLT              # encoding: [0xec,0xe2,A,A,0x00,0x44]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
#CHECK: brxhg	%r15, %r2, bar@PLT              # encoding: [0xec,0xf2,A,A,0x00,0x44]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
#CHECK: brxhg	%r0, %r2, bar@PLT               # encoding: [0xec,0x02,A,A,0x00,0x44]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
#CHECK: brxhg	%r14, %r2, bar@PLT              # encoding: [0xec,0xe2,A,A,0x00,0x44]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
#CHECK: brxhg	%r15, %r2, bar@PLT              # encoding: [0xec,0xf2,A,A,0x00,0x44]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	brxhg	%r0,%r2,bar@PLT
	brxhg	%r14,%r2,bar@PLT
	brxhg	%r15,%r2,bar@PLT
	jxhg	%r0,%r2,bar@PLT
	jxhg	%r14,%r2,bar@PLT
	jxhg	%r15,%r2,bar@PLT


#CHECK: bxle	%r0, %r0, 0             # encoding: [0x87,0x00,0x00,0x00]
#CHECK: bxle	%r0, %r15, 0            # encoding: [0x87,0x0f,0x00,0x00]
#CHECK: bxle	%r14, %r15, 0           # encoding: [0x87,0xef,0x00,0x00]
#CHECK: bxle	%r15, %r15, 0           # encoding: [0x87,0xff,0x00,0x00]
#CHECK: bxle	%r0, %r0, 4095          # encoding: [0x87,0x00,0x0f,0xff]
#CHECK: bxle	%r0, %r0, 1             # encoding: [0x87,0x00,0x00,0x01]
#CHECK: bxle	%r0, %r0, 0(%r1)        # encoding: [0x87,0x00,0x10,0x00]
#CHECK: bxle	%r0, %r0, 0(%r15)       # encoding: [0x87,0x00,0xf0,0x00]
#CHECK: bxle	%r0, %r0, 4095(%r1)     # encoding: [0x87,0x00,0x1f,0xff]
#CHECK: bxle	%r0, %r0, 4095(%r15)    # encoding: [0x87,0x00,0xff,0xff]

	bxle	%r0,%r0,0
	bxle	%r0,%r15,0
	bxle	%r14,%r15,0
	bxle	%r15,%r15,0
	bxle	%r0,%r0,4095
	bxle	%r0,%r0,1
	bxle	%r0,%r0,0(%r1)
	bxle	%r0,%r0,0(%r15)
	bxle	%r0,%r0,4095(%r1)
	bxle	%r0,%r0,4095(%r15)

#CHECK: bxleg	%r0, %r0, 0             # encoding: [0xeb,0x00,0x00,0x00,0x00,0x45]
#CHECK: bxleg	%r0, %r15, 0            # encoding: [0xeb,0x0f,0x00,0x00,0x00,0x45]
#CHECK: bxleg	%r14, %r15, 0           # encoding: [0xeb,0xef,0x00,0x00,0x00,0x45]
#CHECK: bxleg	%r15, %r15, 0           # encoding: [0xeb,0xff,0x00,0x00,0x00,0x45]
#CHECK: bxleg	%r0, %r0, -524288       # encoding: [0xeb,0x00,0x00,0x00,0x80,0x45]
#CHECK: bxleg	%r0, %r0, -1            # encoding: [0xeb,0x00,0x0f,0xff,0xff,0x45]
#CHECK: bxleg	%r0, %r0, 0             # encoding: [0xeb,0x00,0x00,0x00,0x00,0x45]
#CHECK: bxleg	%r0, %r0, 1             # encoding: [0xeb,0x00,0x00,0x01,0x00,0x45]
#CHECK: bxleg	%r0, %r0, 524287        # encoding: [0xeb,0x00,0x0f,0xff,0x7f,0x45]
#CHECK: bxleg	%r0, %r0, 0(%r1)        # encoding: [0xeb,0x00,0x10,0x00,0x00,0x45]
#CHECK: bxleg	%r0, %r0, 0(%r15)       # encoding: [0xeb,0x00,0xf0,0x00,0x00,0x45]
#CHECK: bxleg	%r0, %r0, 524287(%r1)   # encoding: [0xeb,0x00,0x1f,0xff,0x7f,0x45]
#CHECK: bxleg	%r0, %r0, 524287(%r15)  # encoding: [0xeb,0x00,0xff,0xff,0x7f,0x45]

	bxleg	%r0,%r0,0
	bxleg	%r0,%r15,0
	bxleg	%r14,%r15,0
	bxleg	%r15,%r15,0
	bxleg	%r0,%r0,-524288
	bxleg	%r0,%r0,-1
	bxleg	%r0,%r0,0
	bxleg	%r0,%r0,1
	bxleg	%r0,%r0,524287
	bxleg	%r0,%r0,0(%r1)
	bxleg	%r0,%r0,0(%r15)
	bxleg	%r0,%r0,524287(%r1)
	bxleg	%r0,%r0,524287(%r15)

#CHECK: brxle	%r0, %r2, .[[LAB:L.*]]-65536	# encoding: [0x85,0x02,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-65536)+2, kind: FK_390_PC16DBL
#CHECK: brxle	%r0, %r2, .[[LAB:L.*]]-65536	# encoding: [0x85,0x02,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-65536)+2, kind: FK_390_PC16DBL
	brxle	%r0,%r2, -0x10000
	jxle	%r0,%r2, -0x10000
#CHECK: brxle	%r0, %r2, .[[LAB:L.*]]-2	# encoding: [0x85,0x02,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC16DBL
#CHECK: brxle	%r0, %r2, .[[LAB:L.*]]-2	# encoding: [0x85,0x02,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC16DBL
	brxle	%r0, %r2, -2
	jxle	%r0, %r2, -2
#CHECK: brxle	%r0, %r2, .[[LAB:L.*]]	        # encoding: [0x85,0x02,A,A]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC16DBL
#CHECK: brxle	%r0, %r2, .[[LAB:L.*]]	        # encoding: [0x85,0x02,A,A]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC16DBL
	brxle	%r0,%r2, 0
	jxle	%r0,%r2, 0
#CHECK: brxle	%r0, %r2, .[[LAB:L.*]]+65534	# encoding: [0x85,0x02,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+65534)+2, kind: FK_390_PC16DBL
#CHECK: brxle	%r0, %r2, .[[LAB:L.*]]+65534	# encoding: [0x85,0x02,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+65534)+2, kind: FK_390_PC16DBL
	brxle	%r0,%r2, 0xfffe
	jxle	%r0,%r2, 0xfffe

#CHECK: brxle	%r0, %r2, foo                   # encoding: [0x85,0x02,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: brxle	%r14, %r2, foo                  # encoding: [0x85,0xe2,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: brxle	%r15, %r2, foo                  # encoding: [0x85,0xf2,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: brxle	%r0, %r2, foo                   # encoding: [0x85,0x02,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: brxle	%r14, %r2, foo                  # encoding: [0x85,0xe2,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: brxle	%r15, %r2, foo                  # encoding: [0x85,0xf2,A,A]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	brxle	%r0,%r2,foo
	brxle	%r14,%r2,foo
	brxle	%r15,%r2,foo
	jxle	%r0,%r2,foo
	jxle	%r14,%r2,foo
        jxle	%r15,%r2,foo

#CHECK: brxle	%r0, %r2, bar+100               # encoding: [0x85,0x02,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
#CHECK: brxle	%r14, %r2, bar+100              # encoding: [0x85,0xe2,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
#CHECK: brxle	%r15, %r2, bar+100              # encoding: [0x85,0xf2,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
#CHECK: brxle	%r0, %r2, bar+100               # encoding: [0x85,0x02,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
#CHECK: brxle	%r14, %r2, bar+100              # encoding: [0x85,0xe2,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
#CHECK: brxle	%r15, %r2, bar+100              # encoding: [0x85,0xf2,A,A]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	brxle	%r0,%r2,bar+100
	brxle	%r14,%r2,bar+100
	brxle	%r15,%r2,bar+100
	jxle	%r0,%r2,bar+100
	jxle	%r14,%r2,bar+100
	jxle	%r15,%r2,bar+100

#CHECK: brxle	%r0, %r2, bar@PLT               # encoding: [0x85,0x02,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
#CHECK: brxle	%r14, %r2, bar@PLT              # encoding: [0x85,0xe2,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
#CHECK: brxle	%r15, %r2, bar@PLT              # encoding: [0x85,0xf2,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
#CHECK: brxle	%r0, %r2, bar@PLT               # encoding: [0x85,0x02,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
#CHECK: brxle	%r14, %r2, bar@PLT              # encoding: [0x85,0xe2,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
#CHECK: brxle	%r15, %r2, bar@PLT              # encoding: [0x85,0xf2,A,A]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	brxle	%r0,%r2,bar@PLT
	brxle	%r14,%r2,bar@PLT
	brxle	%r15,%r2,bar@PLT
	jxle	%r0,%r2,bar@PLT
	jxle	%r14,%r2,bar@PLT
	jxle	%r15,%r2,bar@PLT

#CHECK: brxlg	%r0, %r2, .[[LAB:L.*]]-65536	# encoding: [0xec,0x02,A,A,0x00,0x45]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-65536)+2, kind: FK_390_PC16DBL
#CHECK: brxlg	%r0, %r2, .[[LAB:L.*]]-65536	# encoding: [0xec,0x02,A,A,0x00,0x45]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-65536)+2, kind: FK_390_PC16DBL
	brxlg	%r0,%r2, -0x10000
	jxleg	%r0,%r2, -0x10000
#CHECK: brxlg	%r0, %r2, .[[LAB:L.*]]-2	# encoding: [0xec,0x02,A,A,0x00,0x45]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC16DBL
#CHECK: brxlg	%r0, %r2, .[[LAB:L.*]]-2	# encoding: [0xec,0x02,A,A,0x00,0x45]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC16DBL
	brxlg	%r0, %r2, -2
	jxleg	%r0, %r2, -2
#CHECK: brxlg	%r0, %r2, .[[LAB:L.*]]	        # encoding: [0xec,0x02,A,A,0x00,0x45]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC16DBL
#CHECK: brxlg	%r0, %r2, .[[LAB:L.*]]	        # encoding: [0xec,0x02,A,A,0x00,0x45]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC16DBL
	brxlg	%r0,%r2, 0
	jxleg	%r0,%r2, 0
#CHECK: brxlg	%r0, %r2, .[[LAB:L.*]]+65534	# encoding: [0xec,0x02,A,A,0x00,0x45]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+65534)+2, kind: FK_390_PC16DBL
#CHECK: brxlg	%r0, %r2, .[[LAB:L.*]]+65534	# encoding: [0xec,0x02,A,A,0x00,0x45]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+65534)+2, kind: FK_390_PC16DBL
	brxlg	%r0,%r2, 0xfffe
	jxleg	%r0,%r2, 0xfffe

#CHECK: brxlg	%r0, %r2, foo                   # encoding: [0xec,0x02,A,A,0x00,0x45]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: brxlg	%r14, %r2, foo                  # encoding: [0xec,0xe2,A,A,0x00,0x45]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: brxlg	%r15, %r2, foo                  # encoding: [0xec,0xf2,A,A,0x00,0x45]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: brxlg	%r0, %r2, foo                   # encoding: [0xec,0x02,A,A,0x00,0x45]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: brxlg	%r14, %r2, foo                  # encoding: [0xec,0xe2,A,A,0x00,0x45]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: brxlg	%r15, %r2, foo                  # encoding: [0xec,0xf2,A,A,0x00,0x45]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	brxlg	%r0,%r2,foo
	brxlg	%r14,%r2,foo
	brxlg	%r15,%r2,foo
	jxleg	%r0,%r2,foo
	jxleg	%r14,%r2,foo
	jxleg	%r15,%r2,foo

#CHECK: brxlg	%r0, %r2, bar+100               # encoding: [0xec,0x02,A,A,0x00,0x45]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
#CHECK: brxlg	%r14, %r2, bar+100              # encoding: [0xec,0xe2,A,A,0x00,0x45]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
#CHECK: brxlg	%r15, %r2, bar+100              # encoding: [0xec,0xf2,A,A,0x00,0x45]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
#CHECK: brxlg	%r0, %r2, bar+100               # encoding: [0xec,0x02,A,A,0x00,0x45]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
#CHECK: brxlg	%r14, %r2, bar+100              # encoding: [0xec,0xe2,A,A,0x00,0x45]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
#CHECK: brxlg	%r15, %r2, bar+100              # encoding: [0xec,0xf2,A,A,0x00,0x45]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	brxlg	%r0,%r2,bar+100
	brxlg	%r14,%r2,bar+100
	brxlg	%r15,%r2,bar+100
	jxleg	%r0,%r2,bar+100
	jxleg	%r14,%r2,bar+100
	jxleg	%r15,%r2,bar+100

#CHECK: brxlg	%r0, %r2, bar@PLT               # encoding: [0xec,0x02,A,A,0x00,0x45]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
#CHECK: brxlg	%r14, %r2, bar@PLT              # encoding: [0xec,0xe2,A,A,0x00,0x45]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
#CHECK: brxlg	%r15, %r2, bar@PLT              # encoding: [0xec,0xf2,A,A,0x00,0x45]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
#CHECK: brxlg	%r0, %r2, bar@PLT               # encoding: [0xec,0x02,A,A,0x00,0x45]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
#CHECK: brxlg	%r14, %r2, bar@PLT              # encoding: [0xec,0xe2,A,A,0x00,0x45]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
#CHECK: brxlg	%r15, %r2, bar@PLT              # encoding: [0xec,0xf2,A,A,0x00,0x45]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	brxlg	%r0,%r2,bar@PLT
	brxlg	%r14,%r2,bar@PLT
	brxlg	%r15,%r2,bar@PLT
	jxleg	%r0,%r2,bar@PLT
	jxleg	%r14,%r2,bar@PLT
	jxleg	%r15,%r2,bar@PLT

#CHECK: c	%r0, 0                  # encoding: [0x59,0x00,0x00,0x00]
#CHECK: c	%r0, 4095               # encoding: [0x59,0x00,0x0f,0xff]
#CHECK: c	%r0, 0(%r1)             # encoding: [0x59,0x00,0x10,0x00]
#CHECK: c	%r0, 0(%r15)            # encoding: [0x59,0x00,0xf0,0x00]
#CHECK: c	%r0, 4095(%r1,%r15)     # encoding: [0x59,0x01,0xff,0xff]
#CHECK: c	%r0, 4095(%r15,%r1)     # encoding: [0x59,0x0f,0x1f,0xff]
#CHECK: c	%r15, 0                 # encoding: [0x59,0xf0,0x00,0x00]

	c	%r0, 0
	c	%r0, 4095
	c	%r0, 0(%r1)
	c	%r0, 0(%r15)
	c	%r0, 4095(%r1,%r15)
	c	%r0, 4095(%r15,%r1)
	c	%r15, 0

#CHECK: cd	%f0, 0                  # encoding: [0x69,0x00,0x00,0x00]
#CHECK: cd	%f0, 4095               # encoding: [0x69,0x00,0x0f,0xff]
#CHECK: cd	%f0, 0(%r1)             # encoding: [0x69,0x00,0x10,0x00]
#CHECK: cd	%f0, 0(%r15)            # encoding: [0x69,0x00,0xf0,0x00]
#CHECK: cd	%f0, 4095(%r1,%r15)     # encoding: [0x69,0x01,0xff,0xff]
#CHECK: cd	%f0, 4095(%r15,%r1)     # encoding: [0x69,0x0f,0x1f,0xff]
#CHECK: cd	%f15, 0                 # encoding: [0x69,0xf0,0x00,0x00]

	cd	%f0, 0
	cd	%f0, 4095
	cd	%f0, 0(%r1)
	cd	%f0, 0(%r15)
	cd	%f0, 4095(%r1,%r15)
	cd	%f0, 4095(%r15,%r1)
	cd	%f15, 0

#CHECK: cdb	%f0, 0                  # encoding: [0xed,0x00,0x00,0x00,0x00,0x19]
#CHECK: cdb	%f0, 4095               # encoding: [0xed,0x00,0x0f,0xff,0x00,0x19]
#CHECK: cdb	%f0, 0(%r1)             # encoding: [0xed,0x00,0x10,0x00,0x00,0x19]
#CHECK: cdb	%f0, 0(%r15)            # encoding: [0xed,0x00,0xf0,0x00,0x00,0x19]
#CHECK: cdb	%f0, 4095(%r1,%r15)     # encoding: [0xed,0x01,0xff,0xff,0x00,0x19]
#CHECK: cdb	%f0, 4095(%r15,%r1)     # encoding: [0xed,0x0f,0x1f,0xff,0x00,0x19]
#CHECK: cdb	%f15, 0                 # encoding: [0xed,0xf0,0x00,0x00,0x00,0x19]

	cdb	%f0, 0
	cdb	%f0, 4095
	cdb	%f0, 0(%r1)
	cdb	%f0, 0(%r15)
	cdb	%f0, 4095(%r1,%r15)
	cdb	%f0, 4095(%r15,%r1)
	cdb	%f15, 0

#CHECK: cdbr	%f0, %f0                # encoding: [0xb3,0x19,0x00,0x00]
#CHECK: cdbr	%f0, %f15               # encoding: [0xb3,0x19,0x00,0x0f]
#CHECK: cdbr	%f7, %f8                # encoding: [0xb3,0x19,0x00,0x78]
#CHECK: cdbr	%f15, %f0               # encoding: [0xb3,0x19,0x00,0xf0]

	cdbr	%f0, %f0
	cdbr	%f0, %f15
	cdbr	%f7, %f8
	cdbr	%f15, %f0

#CHECK: cdfbr	%f0, %r0                # encoding: [0xb3,0x95,0x00,0x00]
#CHECK: cdfbr	%f0, %r15               # encoding: [0xb3,0x95,0x00,0x0f]
#CHECK: cdfbr	%f15, %r0               # encoding: [0xb3,0x95,0x00,0xf0]
#CHECK: cdfbr	%f7, %r8                # encoding: [0xb3,0x95,0x00,0x78]
#CHECK: cdfbr	%f15, %r15              # encoding: [0xb3,0x95,0x00,0xff]

	cdfbr	%f0, %r0
	cdfbr	%f0, %r15
	cdfbr	%f15, %r0
	cdfbr	%f7, %r8
	cdfbr	%f15, %r15

#CHECK: cdfr	%f0, %r0                # encoding: [0xb3,0xb5,0x00,0x00]
#CHECK: cdfr	%f0, %r15               # encoding: [0xb3,0xb5,0x00,0x0f]
#CHECK: cdfr	%f15, %r0               # encoding: [0xb3,0xb5,0x00,0xf0]
#CHECK: cdfr	%f7, %r8                # encoding: [0xb3,0xb5,0x00,0x78]
#CHECK: cdfr	%f15, %r15              # encoding: [0xb3,0xb5,0x00,0xff]

	cdfr	%f0, %r0
	cdfr	%f0, %r15
	cdfr	%f15, %r0
	cdfr	%f7, %r8
	cdfr	%f15, %r15

#CHECK: cdgbr	%f0, %r0                # encoding: [0xb3,0xa5,0x00,0x00]
#CHECK: cdgbr	%f0, %r15               # encoding: [0xb3,0xa5,0x00,0x0f]
#CHECK: cdgbr	%f15, %r0               # encoding: [0xb3,0xa5,0x00,0xf0]
#CHECK: cdgbr	%f7, %r8                # encoding: [0xb3,0xa5,0x00,0x78]
#CHECK: cdgbr	%f15, %r15              # encoding: [0xb3,0xa5,0x00,0xff]

	cdgbr	%f0, %r0
	cdgbr	%f0, %r15
	cdgbr	%f15, %r0
	cdgbr	%f7, %r8
	cdgbr	%f15, %r15

#CHECK: cdgr	%f0, %r0                # encoding: [0xb3,0xc5,0x00,0x00]
#CHECK: cdgr	%f0, %r15               # encoding: [0xb3,0xc5,0x00,0x0f]
#CHECK: cdgr	%f15, %r0               # encoding: [0xb3,0xc5,0x00,0xf0]
#CHECK: cdgr	%f7, %r8                # encoding: [0xb3,0xc5,0x00,0x78]
#CHECK: cdgr	%f15, %r15              # encoding: [0xb3,0xc5,0x00,0xff]

	cdgr	%f0, %r0
	cdgr	%f0, %r15
	cdgr	%f15, %r0
	cdgr	%f7, %r8
	cdgr	%f15, %r15

#CHECK: cdgtr	%f0, %r0                # encoding: [0xb3,0xf1,0x00,0x00]
#CHECK: cdgtr	%f0, %r15               # encoding: [0xb3,0xf1,0x00,0x0f]
#CHECK: cdgtr	%f15, %r0               # encoding: [0xb3,0xf1,0x00,0xf0]
#CHECK: cdgtr	%f7, %r8                # encoding: [0xb3,0xf1,0x00,0x78]
#CHECK: cdgtr	%f15, %r15              # encoding: [0xb3,0xf1,0x00,0xff]

	cdgtr	%f0, %r0
	cdgtr	%f0, %r15
	cdgtr	%f15, %r0
	cdgtr	%f7, %r8
	cdgtr	%f15, %r15

#CHECK: cdr	%f0, %f0                # encoding: [0x29,0x00]
#CHECK: cdr	%f0, %f15               # encoding: [0x29,0x0f]
#CHECK: cdr	%f7, %f8                # encoding: [0x29,0x78]
#CHECK: cdr	%f15, %f0               # encoding: [0x29,0xf0]

	cdr	%f0, %f0
	cdr	%f0, %f15
	cdr	%f7, %f8
	cdr	%f15, %f0

#CHECK: cds	%r0, %r0, 0             # encoding: [0xbb,0x00,0x00,0x00]
#CHECK: cds	%r0, %r0, 4095          # encoding: [0xbb,0x00,0x0f,0xff]
#CHECK: cds	%r0, %r0, 0(%r1)        # encoding: [0xbb,0x00,0x10,0x00]
#CHECK: cds	%r0, %r0, 0(%r15)       # encoding: [0xbb,0x00,0xf0,0x00]
#CHECK: cds	%r0, %r0, 4095(%r1)     # encoding: [0xbb,0x00,0x1f,0xff]
#CHECK: cds	%r0, %r0, 4095(%r15)    # encoding: [0xbb,0x00,0xff,0xff]
#CHECK: cds	%r0, %r14, 0            # encoding: [0xbb,0x0e,0x00,0x00]
#CHECK: cds	%r14, %r0, 0            # encoding: [0xbb,0xe0,0x00,0x00]

	cds	%r0, %r0, 0
	cds	%r0, %r0, 4095
	cds	%r0, %r0, 0(%r1)
	cds	%r0, %r0, 0(%r15)
	cds	%r0, %r0, 4095(%r1)
	cds	%r0, %r0, 4095(%r15)
	cds	%r0, %r14, 0
	cds	%r14, %r0, 0

#CHECK: cdsg	%r0, %r0, -524288       # encoding: [0xeb,0x00,0x00,0x00,0x80,0x3e]
#CHECK: cdsg	%r0, %r0, -1            # encoding: [0xeb,0x00,0x0f,0xff,0xff,0x3e]
#CHECK: cdsg	%r0, %r0, 0             # encoding: [0xeb,0x00,0x00,0x00,0x00,0x3e]
#CHECK: cdsg	%r0, %r0, 1             # encoding: [0xeb,0x00,0x00,0x01,0x00,0x3e]
#CHECK: cdsg	%r0, %r0, 524287        # encoding: [0xeb,0x00,0x0f,0xff,0x7f,0x3e]
#CHECK: cdsg	%r0, %r0, 0(%r1)        # encoding: [0xeb,0x00,0x10,0x00,0x00,0x3e]
#CHECK: cdsg	%r0, %r0, 0(%r15)       # encoding: [0xeb,0x00,0xf0,0x00,0x00,0x3e]
#CHECK: cdsg	%r0, %r0, 524287(%r1)   # encoding: [0xeb,0x00,0x1f,0xff,0x7f,0x3e]
#CHECK: cdsg	%r0, %r0, 524287(%r15)  # encoding: [0xeb,0x00,0xff,0xff,0x7f,0x3e]
#CHECK: cdsg	%r0, %r14, 0            # encoding: [0xeb,0x0e,0x00,0x00,0x00,0x3e]
#CHECK: cdsg	%r14, %r0, 0            # encoding: [0xeb,0xe0,0x00,0x00,0x00,0x3e]

	cdsg	%r0, %r0, -524288
	cdsg	%r0, %r0, -1
	cdsg	%r0, %r0, 0
	cdsg	%r0, %r0, 1
	cdsg	%r0, %r0, 524287
	cdsg	%r0, %r0, 0(%r1)
	cdsg	%r0, %r0, 0(%r15)
	cdsg	%r0, %r0, 524287(%r1)
	cdsg	%r0, %r0, 524287(%r15)
	cdsg	%r0, %r14, 0
	cdsg	%r14, %r0, 0

#CHECK: cdstr	%f0, %r0                # encoding: [0xb3,0xf3,0x00,0x00]
#CHECK: cdstr	%f0, %r15               # encoding: [0xb3,0xf3,0x00,0x0f]
#CHECK: cdstr	%f15, %r0               # encoding: [0xb3,0xf3,0x00,0xf0]
#CHECK: cdstr	%f7, %r8                # encoding: [0xb3,0xf3,0x00,0x78]
#CHECK: cdstr	%f15, %r15              # encoding: [0xb3,0xf3,0x00,0xff]

	cdstr	%f0, %r0
	cdstr	%f0, %r15
	cdstr	%f15, %r0
	cdstr	%f7, %r8
	cdstr	%f15, %r15

#CHECK: cdsy	%r0, %r0, -524288       # encoding: [0xeb,0x00,0x00,0x00,0x80,0x31]
#CHECK: cdsy	%r0, %r0, -1            # encoding: [0xeb,0x00,0x0f,0xff,0xff,0x31]
#CHECK: cdsy	%r0, %r0, 0             # encoding: [0xeb,0x00,0x00,0x00,0x00,0x31]
#CHECK: cdsy	%r0, %r0, 1             # encoding: [0xeb,0x00,0x00,0x01,0x00,0x31]
#CHECK: cdsy	%r0, %r0, 524287        # encoding: [0xeb,0x00,0x0f,0xff,0x7f,0x31]
#CHECK: cdsy	%r0, %r0, 0(%r1)        # encoding: [0xeb,0x00,0x10,0x00,0x00,0x31]
#CHECK: cdsy	%r0, %r0, 0(%r15)       # encoding: [0xeb,0x00,0xf0,0x00,0x00,0x31]
#CHECK: cdsy	%r0, %r0, 524287(%r1)   # encoding: [0xeb,0x00,0x1f,0xff,0x7f,0x31]
#CHECK: cdsy	%r0, %r0, 524287(%r15)  # encoding: [0xeb,0x00,0xff,0xff,0x7f,0x31]
#CHECK: cdsy	%r0, %r14, 0            # encoding: [0xeb,0x0e,0x00,0x00,0x00,0x31]
#CHECK: cdsy	%r14, %r0, 0            # encoding: [0xeb,0xe0,0x00,0x00,0x00,0x31]

	cdsy	%r0, %r0, -524288
	cdsy	%r0, %r0, -1
	cdsy	%r0, %r0, 0
	cdsy	%r0, %r0, 1
	cdsy	%r0, %r0, 524287
	cdsy	%r0, %r0, 0(%r1)
	cdsy	%r0, %r0, 0(%r15)
	cdsy	%r0, %r0, 524287(%r1)
	cdsy	%r0, %r0, 524287(%r15)
	cdsy	%r0, %r14, 0
	cdsy	%r14, %r0, 0

#CHECK: cdtr	%f0, %f0                # encoding: [0xb3,0xe4,0x00,0x00]
#CHECK: cdtr	%f0, %f15               # encoding: [0xb3,0xe4,0x00,0x0f]
#CHECK: cdtr	%f7, %f8                # encoding: [0xb3,0xe4,0x00,0x78]
#CHECK: cdtr	%f15, %f0               # encoding: [0xb3,0xe4,0x00,0xf0]

	cdtr	%f0, %f0
	cdtr	%f0, %f15
	cdtr	%f7, %f8
	cdtr	%f15, %f0

#CHECK: cdutr	%f0, %r0                # encoding: [0xb3,0xf2,0x00,0x00]
#CHECK: cdutr	%f0, %r15               # encoding: [0xb3,0xf2,0x00,0x0f]
#CHECK: cdutr	%f15, %r0               # encoding: [0xb3,0xf2,0x00,0xf0]
#CHECK: cdutr	%f7, %r8                # encoding: [0xb3,0xf2,0x00,0x78]
#CHECK: cdutr	%f15, %r15              # encoding: [0xb3,0xf2,0x00,0xff]

	cdutr	%f0, %r0
	cdutr	%f0, %r15
	cdutr	%f15, %r0
	cdutr	%f7, %r8
	cdutr	%f15, %r15

#CHECK: ce	%f0, 0                  # encoding: [0x79,0x00,0x00,0x00]
#CHECK: ce	%f0, 4095               # encoding: [0x79,0x00,0x0f,0xff]
#CHECK: ce	%f0, 0(%r1)             # encoding: [0x79,0x00,0x10,0x00]
#CHECK: ce	%f0, 0(%r15)            # encoding: [0x79,0x00,0xf0,0x00]
#CHECK: ce	%f0, 4095(%r1,%r15)     # encoding: [0x79,0x01,0xff,0xff]
#CHECK: ce	%f0, 4095(%r15,%r1)     # encoding: [0x79,0x0f,0x1f,0xff]
#CHECK: ce	%f15, 0                 # encoding: [0x79,0xf0,0x00,0x00]

	ce	%f0, 0
	ce	%f0, 4095
	ce	%f0, 0(%r1)
	ce	%f0, 0(%r15)
	ce	%f0, 4095(%r1,%r15)
	ce	%f0, 4095(%r15,%r1)
	ce	%f15, 0

#CHECK: ceb	%f0, 0                  # encoding: [0xed,0x00,0x00,0x00,0x00,0x09]
#CHECK: ceb	%f0, 4095               # encoding: [0xed,0x00,0x0f,0xff,0x00,0x09]
#CHECK: ceb	%f0, 0(%r1)             # encoding: [0xed,0x00,0x10,0x00,0x00,0x09]
#CHECK: ceb	%f0, 0(%r15)            # encoding: [0xed,0x00,0xf0,0x00,0x00,0x09]
#CHECK: ceb	%f0, 4095(%r1,%r15)     # encoding: [0xed,0x01,0xff,0xff,0x00,0x09]
#CHECK: ceb	%f0, 4095(%r15,%r1)     # encoding: [0xed,0x0f,0x1f,0xff,0x00,0x09]
#CHECK: ceb	%f15, 0                 # encoding: [0xed,0xf0,0x00,0x00,0x00,0x09]

	ceb	%f0, 0
	ceb	%f0, 4095
	ceb	%f0, 0(%r1)
	ceb	%f0, 0(%r15)
	ceb	%f0, 4095(%r1,%r15)
	ceb	%f0, 4095(%r15,%r1)
	ceb	%f15, 0

#CHECK: cebr	%f0, %f0                # encoding: [0xb3,0x09,0x00,0x00]
#CHECK: cebr	%f0, %f15               # encoding: [0xb3,0x09,0x00,0x0f]
#CHECK: cebr	%f7, %f8                # encoding: [0xb3,0x09,0x00,0x78]
#CHECK: cebr	%f15, %f0               # encoding: [0xb3,0x09,0x00,0xf0]

	cebr	%f0, %f0
	cebr	%f0, %f15
	cebr	%f7, %f8
	cebr	%f15, %f0

#CHECK: cedtr	%f0, %f0                # encoding: [0xb3,0xf4,0x00,0x00]
#CHECK: cedtr	%f0, %f15               # encoding: [0xb3,0xf4,0x00,0x0f]
#CHECK: cedtr	%f7, %f8                # encoding: [0xb3,0xf4,0x00,0x78]
#CHECK: cedtr	%f15, %f0               # encoding: [0xb3,0xf4,0x00,0xf0]

	cedtr	%f0, %f0
	cedtr	%f0, %f15
	cedtr	%f7, %f8
	cedtr	%f15, %f0

#CHECK: cefbr	%f0, %r0                # encoding: [0xb3,0x94,0x00,0x00]
#CHECK: cefbr	%f0, %r15               # encoding: [0xb3,0x94,0x00,0x0f]
#CHECK: cefbr	%f15, %r0               # encoding: [0xb3,0x94,0x00,0xf0]
#CHECK: cefbr	%f7, %r8                # encoding: [0xb3,0x94,0x00,0x78]
#CHECK: cefbr	%f15, %r15              # encoding: [0xb3,0x94,0x00,0xff]

	cefbr	%f0, %r0
	cefbr	%f0, %r15
	cefbr	%f15, %r0
	cefbr	%f7, %r8
	cefbr	%f15, %r15

#CHECK: cefr	%f0, %r0                # encoding: [0xb3,0xb4,0x00,0x00]
#CHECK: cefr	%f0, %r15               # encoding: [0xb3,0xb4,0x00,0x0f]
#CHECK: cefr	%f15, %r0               # encoding: [0xb3,0xb4,0x00,0xf0]
#CHECK: cefr	%f7, %r8                # encoding: [0xb3,0xb4,0x00,0x78]
#CHECK: cefr	%f15, %r15              # encoding: [0xb3,0xb4,0x00,0xff]

	cefr	%f0, %r0
	cefr	%f0, %r15
	cefr	%f15, %r0
	cefr	%f7, %r8
	cefr	%f15, %r15

#CHECK: cegbr	%f0, %r0                # encoding: [0xb3,0xa4,0x00,0x00]
#CHECK: cegbr	%f0, %r15               # encoding: [0xb3,0xa4,0x00,0x0f]
#CHECK: cegbr	%f15, %r0               # encoding: [0xb3,0xa4,0x00,0xf0]
#CHECK: cegbr	%f7, %r8                # encoding: [0xb3,0xa4,0x00,0x78]
#CHECK: cegbr	%f15, %r15              # encoding: [0xb3,0xa4,0x00,0xff]

	cegbr	%f0, %r0
	cegbr	%f0, %r15
	cegbr	%f15, %r0
	cegbr	%f7, %r8
	cegbr	%f15, %r15

#CHECK: cegr	%f0, %r0                # encoding: [0xb3,0xc4,0x00,0x00]
#CHECK: cegr	%f0, %r15               # encoding: [0xb3,0xc4,0x00,0x0f]
#CHECK: cegr	%f15, %r0               # encoding: [0xb3,0xc4,0x00,0xf0]
#CHECK: cegr	%f7, %r8                # encoding: [0xb3,0xc4,0x00,0x78]
#CHECK: cegr	%f15, %r15              # encoding: [0xb3,0xc4,0x00,0xff]

	cegr	%f0, %r0
	cegr	%f0, %r15
	cegr	%f15, %r0
	cegr	%f7, %r8
	cegr	%f15, %r15

#CHECK: cer	%f0, %f0                # encoding: [0x39,0x00]
#CHECK: cer	%f0, %f15               # encoding: [0x39,0x0f]
#CHECK: cer	%f7, %f8                # encoding: [0x39,0x78]
#CHECK: cer	%f15, %f0               # encoding: [0x39,0xf0]

	cer	%f0, %f0
	cer	%f0, %f15
	cer	%f7, %f8
	cer	%f15, %f0

#CHECK: cextr	%f0, %f0                # encoding: [0xb3,0xfc,0x00,0x00]
#CHECK: cextr	%f0, %f13               # encoding: [0xb3,0xfc,0x00,0x0d]
#CHECK: cextr	%f8, %f8                # encoding: [0xb3,0xfc,0x00,0x88]
#CHECK: cextr	%f13, %f0               # encoding: [0xb3,0xfc,0x00,0xd0]

	cextr	%f0, %f0
	cextr	%f0, %f13
	cextr	%f8, %f8
	cextr	%f13, %f0

#CHECK: cfc	0                       # encoding: [0xb2,0x1a,0x00,0x00]
#CHECK: cfc	0(%r1)                  # encoding: [0xb2,0x1a,0x10,0x00]
#CHECK: cfc	0(%r15)                 # encoding: [0xb2,0x1a,0xf0,0x00]
#CHECK: cfc	4095                    # encoding: [0xb2,0x1a,0x0f,0xff]
#CHECK: cfc	4095(%r1)               # encoding: [0xb2,0x1a,0x1f,0xff]
#CHECK: cfc	4095(%r15)              # encoding: [0xb2,0x1a,0xff,0xff]

	cfc	0
	cfc	0(%r1)
	cfc	0(%r15)
	cfc	4095
	cfc	4095(%r1)
	cfc	4095(%r15)

#CHECK: cfdbr	%r0, 0, %f0             # encoding: [0xb3,0x99,0x00,0x00]
#CHECK: cfdbr	%r0, 0, %f15            # encoding: [0xb3,0x99,0x00,0x0f]
#CHECK: cfdbr	%r0, 15, %f0            # encoding: [0xb3,0x99,0xf0,0x00]
#CHECK: cfdbr	%r4, 5, %f6             # encoding: [0xb3,0x99,0x50,0x46]
#CHECK: cfdbr	%r15, 0, %f0            # encoding: [0xb3,0x99,0x00,0xf0]

	cfdbr	%r0, 0, %f0
	cfdbr	%r0, 0, %f15
	cfdbr	%r0, 15, %f0
	cfdbr	%r4, 5, %f6
	cfdbr	%r15, 0, %f0

#CHECK: cfdr	%r0, 0, %f0             # encoding: [0xb3,0xb9,0x00,0x00]
#CHECK: cfdr	%r0, 0, %f15            # encoding: [0xb3,0xb9,0x00,0x0f]
#CHECK: cfdr	%r0, 15, %f0            # encoding: [0xb3,0xb9,0xf0,0x00]
#CHECK: cfdr	%r4, 5, %f6             # encoding: [0xb3,0xb9,0x50,0x46]
#CHECK: cfdr	%r15, 0, %f0            # encoding: [0xb3,0xb9,0x00,0xf0]

	cfdr	%r0, 0, %f0
	cfdr	%r0, 0, %f15
	cfdr	%r0, 15, %f0
	cfdr	%r4, 5, %f6
	cfdr	%r15, 0, %f0

#CHECK: cfebr	%r0, 0, %f0             # encoding: [0xb3,0x98,0x00,0x00]
#CHECK: cfebr	%r0, 0, %f15            # encoding: [0xb3,0x98,0x00,0x0f]
#CHECK: cfebr	%r0, 15, %f0            # encoding: [0xb3,0x98,0xf0,0x00]
#CHECK: cfebr	%r4, 5, %f6             # encoding: [0xb3,0x98,0x50,0x46]
#CHECK: cfebr	%r15, 0, %f0            # encoding: [0xb3,0x98,0x00,0xf0]

	cfebr	%r0, 0, %f0
	cfebr	%r0, 0, %f15
	cfebr	%r0, 15, %f0
	cfebr	%r4, 5, %f6
	cfebr	%r15, 0, %f0

#CHECK: cfer	%r0, 0, %f0             # encoding: [0xb3,0xb8,0x00,0x00]
#CHECK: cfer	%r0, 0, %f15            # encoding: [0xb3,0xb8,0x00,0x0f]
#CHECK: cfer	%r0, 15, %f0            # encoding: [0xb3,0xb8,0xf0,0x00]
#CHECK: cfer	%r4, 5, %f6             # encoding: [0xb3,0xb8,0x50,0x46]
#CHECK: cfer	%r15, 0, %f0            # encoding: [0xb3,0xb8,0x00,0xf0]

	cfer	%r0, 0, %f0
	cfer	%r0, 0, %f15
	cfer	%r0, 15, %f0
	cfer	%r4, 5, %f6
	cfer	%r15, 0, %f0

#CHECK: cfi	%r0, -2147483648        # encoding: [0xc2,0x0d,0x80,0x00,0x00,0x00]
#CHECK: cfi	%r0, -1                 # encoding: [0xc2,0x0d,0xff,0xff,0xff,0xff]
#CHECK: cfi	%r0, 0                  # encoding: [0xc2,0x0d,0x00,0x00,0x00,0x00]
#CHECK: cfi	%r0, 1                  # encoding: [0xc2,0x0d,0x00,0x00,0x00,0x01]
#CHECK: cfi	%r0, 2147483647         # encoding: [0xc2,0x0d,0x7f,0xff,0xff,0xff]
#CHECK: cfi	%r15, 0                 # encoding: [0xc2,0xfd,0x00,0x00,0x00,0x00]

	cfi	%r0, -1 << 31
	cfi	%r0, -1
	cfi	%r0, 0
	cfi	%r0, 1
	cfi	%r0, (1 << 31) - 1
	cfi	%r15, 0

#CHECK: cfxbr	%r0, 0, %f0             # encoding: [0xb3,0x9a,0x00,0x00]
#CHECK: cfxbr	%r0, 0, %f13            # encoding: [0xb3,0x9a,0x00,0x0d]
#CHECK: cfxbr	%r0, 15, %f0            # encoding: [0xb3,0x9a,0xf0,0x00]
#CHECK: cfxbr	%r4, 5, %f8             # encoding: [0xb3,0x9a,0x50,0x48]
#CHECK: cfxbr	%r15, 0, %f0            # encoding: [0xb3,0x9a,0x00,0xf0]

	cfxbr	%r0, 0, %f0
	cfxbr	%r0, 0, %f13
	cfxbr	%r0, 15, %f0
	cfxbr	%r4, 5, %f8
	cfxbr	%r15, 0, %f0

#CHECK: cfxr	%r0, 0, %f0             # encoding: [0xb3,0xba,0x00,0x00]
#CHECK: cfxr	%r0, 0, %f13            # encoding: [0xb3,0xba,0x00,0x0d]
#CHECK: cfxr	%r0, 15, %f0            # encoding: [0xb3,0xba,0xf0,0x00]
#CHECK: cfxr	%r4, 5, %f8             # encoding: [0xb3,0xba,0x50,0x48]
#CHECK: cfxr	%r15, 0, %f0            # encoding: [0xb3,0xba,0x00,0xf0]

	cfxr	%r0, 0, %f0
	cfxr	%r0, 0, %f13
	cfxr	%r0, 15, %f0
	cfxr	%r4, 5, %f8
	cfxr	%r15, 0, %f0

#CHECK: cg	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x20]
#CHECK: cg	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x20]
#CHECK: cg	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x20]
#CHECK: cg	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x20]
#CHECK: cg	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x20]
#CHECK: cg	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x20]
#CHECK: cg	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x20]
#CHECK: cg	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x20]
#CHECK: cg	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x20]
#CHECK: cg	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x20]

	cg	%r0, -524288
	cg	%r0, -1
	cg	%r0, 0
	cg	%r0, 1
	cg	%r0, 524287
	cg	%r0, 0(%r1)
	cg	%r0, 0(%r15)
	cg	%r0, 524287(%r1,%r15)
	cg	%r0, 524287(%r15,%r1)
	cg	%r15, 0

#CHECK: cgdbr	%r0, 0, %f0             # encoding: [0xb3,0xa9,0x00,0x00]
#CHECK: cgdbr	%r0, 0, %f15            # encoding: [0xb3,0xa9,0x00,0x0f]
#CHECK: cgdbr	%r0, 15, %f0            # encoding: [0xb3,0xa9,0xf0,0x00]
#CHECK: cgdbr	%r4, 5, %f6             # encoding: [0xb3,0xa9,0x50,0x46]
#CHECK: cgdbr	%r15, 0, %f0            # encoding: [0xb3,0xa9,0x00,0xf0]

	cgdbr	%r0, 0, %f0
	cgdbr	%r0, 0, %f15
	cgdbr	%r0, 15, %f0
	cgdbr	%r4, 5, %f6
	cgdbr	%r15, 0, %f0

#CHECK: cgdr	%r0, 0, %f0             # encoding: [0xb3,0xc9,0x00,0x00]
#CHECK: cgdr	%r0, 0, %f15            # encoding: [0xb3,0xc9,0x00,0x0f]
#CHECK: cgdr	%r0, 15, %f0            # encoding: [0xb3,0xc9,0xf0,0x00]
#CHECK: cgdr	%r4, 5, %f6             # encoding: [0xb3,0xc9,0x50,0x46]
#CHECK: cgdr	%r15, 0, %f0            # encoding: [0xb3,0xc9,0x00,0xf0]

	cgdr	%r0, 0, %f0
	cgdr	%r0, 0, %f15
	cgdr	%r0, 15, %f0
	cgdr	%r4, 5, %f6
	cgdr	%r15, 0, %f0

#CHECK: cgdtr	%r0, 0, %f0             # encoding: [0xb3,0xe1,0x00,0x00]
#CHECK: cgdtr	%r0, 0, %f15            # encoding: [0xb3,0xe1,0x00,0x0f]
#CHECK: cgdtr	%r0, 15, %f0            # encoding: [0xb3,0xe1,0xf0,0x00]
#CHECK: cgdtr	%r4, 5, %f6             # encoding: [0xb3,0xe1,0x50,0x46]
#CHECK: cgdtr	%r15, 0, %f0            # encoding: [0xb3,0xe1,0x00,0xf0]

	cgdtr	%r0, 0, %f0
	cgdtr	%r0, 0, %f15
	cgdtr	%r0, 15, %f0
	cgdtr	%r4, 5, %f6
	cgdtr	%r15, 0, %f0

#CHECK: cgebr	%r0, 0, %f0             # encoding: [0xb3,0xa8,0x00,0x00]
#CHECK: cgebr	%r0, 0, %f15            # encoding: [0xb3,0xa8,0x00,0x0f]
#CHECK: cgebr	%r0, 15, %f0            # encoding: [0xb3,0xa8,0xf0,0x00]
#CHECK: cgebr	%r4, 5, %f6             # encoding: [0xb3,0xa8,0x50,0x46]
#CHECK: cgebr	%r15, 0, %f0            # encoding: [0xb3,0xa8,0x00,0xf0]

	cgebr	%r0, 0, %f0
	cgebr	%r0, 0, %f15
	cgebr	%r0, 15, %f0
	cgebr	%r4, 5, %f6
	cgebr	%r15, 0, %f0

#CHECK: cger	%r0, 0, %f0             # encoding: [0xb3,0xc8,0x00,0x00]
#CHECK: cger	%r0, 0, %f15            # encoding: [0xb3,0xc8,0x00,0x0f]
#CHECK: cger	%r0, 15, %f0            # encoding: [0xb3,0xc8,0xf0,0x00]
#CHECK: cger	%r4, 5, %f6             # encoding: [0xb3,0xc8,0x50,0x46]
#CHECK: cger	%r15, 0, %f0            # encoding: [0xb3,0xc8,0x00,0xf0]

	cger	%r0, 0, %f0
	cger	%r0, 0, %f15
	cger	%r0, 15, %f0
	cger	%r4, 5, %f6
	cger	%r15, 0, %f0

#CHECK: cgf	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x30]
#CHECK: cgf	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x30]
#CHECK: cgf	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x30]
#CHECK: cgf	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x30]
#CHECK: cgf	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x30]
#CHECK: cgf	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x30]
#CHECK: cgf	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x30]
#CHECK: cgf	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x30]
#CHECK: cgf	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x30]
#CHECK: cgf	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x30]

	cgf	%r0, -524288
	cgf	%r0, -1
	cgf	%r0, 0
	cgf	%r0, 1
	cgf	%r0, 524287
	cgf	%r0, 0(%r1)
	cgf	%r0, 0(%r15)
	cgf	%r0, 524287(%r1,%r15)
	cgf	%r0, 524287(%r15,%r1)
	cgf	%r15, 0

#CHECK: cgfi	%r0, -2147483648        # encoding: [0xc2,0x0c,0x80,0x00,0x00,0x00]
#CHECK: cgfi	%r0, -1                 # encoding: [0xc2,0x0c,0xff,0xff,0xff,0xff]
#CHECK: cgfi	%r0, 0                  # encoding: [0xc2,0x0c,0x00,0x00,0x00,0x00]
#CHECK: cgfi	%r0, 1                  # encoding: [0xc2,0x0c,0x00,0x00,0x00,0x01]
#CHECK: cgfi	%r0, 2147483647         # encoding: [0xc2,0x0c,0x7f,0xff,0xff,0xff]
#CHECK: cgfi	%r15, 0                 # encoding: [0xc2,0xfc,0x00,0x00,0x00,0x00]

	cgfi	%r0, -1 << 31
	cgfi	%r0, -1
	cgfi	%r0, 0
	cgfi	%r0, 1
	cgfi	%r0, (1 << 31) - 1
	cgfi	%r15, 0

#CHECK: cgfr	%r0, %r0                # encoding: [0xb9,0x30,0x00,0x00]
#CHECK: cgfr	%r0, %r15               # encoding: [0xb9,0x30,0x00,0x0f]
#CHECK: cgfr	%r15, %r0               # encoding: [0xb9,0x30,0x00,0xf0]
#CHECK: cgfr	%r7, %r8                # encoding: [0xb9,0x30,0x00,0x78]

	cgfr	%r0,%r0
	cgfr	%r0,%r15
	cgfr	%r15,%r0
	cgfr	%r7,%r8

#CHECK: cgfrl	%r0, .[[LAB:L.*]]-4294967296 # encoding: [0xc6,0x0c,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-4294967296)+2, kind: FK_390_PC32DBL
	cgfrl	%r0, -0x100000000
#CHECK: cgfrl	%r0, .[[LAB:L.*]]-2	# encoding: [0xc6,0x0c,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC32DBL
	cgfrl	%r0, -2
#CHECK: cgfrl	%r0, .[[LAB:L.*]]	# encoding: [0xc6,0x0c,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC32DBL
	cgfrl	%r0, 0
#CHECK: cgfrl	%r0, .[[LAB:L.*]]+4294967294 # encoding: [0xc6,0x0c,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+4294967294)+2, kind: FK_390_PC32DBL
	cgfrl	%r0, 0xfffffffe

#CHECK: cgfrl	%r0, foo                # encoding: [0xc6,0x0c,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: cgfrl	%r15, foo               # encoding: [0xc6,0xfc,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL

	cgfrl	%r0,foo
	cgfrl	%r15,foo

#CHECK: cgfrl	%r3, bar+100            # encoding: [0xc6,0x3c,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: cgfrl	%r4, bar+100            # encoding: [0xc6,0x4c,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL

	cgfrl	%r3,bar+100
	cgfrl	%r4,bar+100

#CHECK: cgfrl	%r7, frob@PLT           # encoding: [0xc6,0x7c,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL
#CHECK: cgfrl	%r8, frob@PLT           # encoding: [0xc6,0x8c,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL

	cgfrl	%r7,frob@PLT
	cgfrl	%r8,frob@PLT

#CHECK: cgh	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x34]
#CHECK: cgh	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x34]
#CHECK: cgh	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x34]
#CHECK: cgh	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x34]
#CHECK: cgh	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x34]
#CHECK: cgh	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x34]
#CHECK: cgh	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x34]
#CHECK: cgh	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x34]
#CHECK: cgh	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x34]
#CHECK: cgh	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x34]

	cgh	%r0, -524288
	cgh	%r0, -1
	cgh	%r0, 0
	cgh	%r0, 1
	cgh	%r0, 524287
	cgh	%r0, 0(%r1)
	cgh	%r0, 0(%r15)
	cgh	%r0, 524287(%r1,%r15)
	cgh	%r0, 524287(%r15,%r1)
	cgh	%r15, 0

#CHECK: cghi	%r0, -32768             # encoding: [0xa7,0x0f,0x80,0x00]
#CHECK: cghi	%r0, -1                 # encoding: [0xa7,0x0f,0xff,0xff]
#CHECK: cghi	%r0, 0                  # encoding: [0xa7,0x0f,0x00,0x00]
#CHECK: cghi	%r0, 1                  # encoding: [0xa7,0x0f,0x00,0x01]
#CHECK: cghi	%r0, 32767              # encoding: [0xa7,0x0f,0x7f,0xff]
#CHECK: cghi	%r15, 0                 # encoding: [0xa7,0xff,0x00,0x00]
#CHECK: cghi	%r0, foo                # encoding: [0xa7,0x0f,A,A]

	cghi	%r0, -32768
	cghi	%r0, -1
	cghi	%r0, 0
	cghi	%r0, 1
	cghi	%r0, 32767
	cghi	%r15, 0
	cghi	%r0, foo

#CHECK: cghrl	%r0, .[[LAB:L.*]]-4294967296 # encoding: [0xc6,0x04,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-4294967296)+2, kind: FK_390_PC32DBL
	cghrl	%r0, -0x100000000
#CHECK: cghrl	%r0, .[[LAB:L.*]]-2	# encoding: [0xc6,0x04,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC32DBL
	cghrl	%r0, -2
#CHECK: cghrl	%r0, .[[LAB:L.*]]	# encoding: [0xc6,0x04,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC32DBL
	cghrl	%r0, 0
#CHECK: cghrl	%r0, .[[LAB:L.*]]+4294967294 # encoding: [0xc6,0x04,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+4294967294)+2, kind: FK_390_PC32DBL
	cghrl	%r0, 0xfffffffe

#CHECK: cghrl	%r0, foo                # encoding: [0xc6,0x04,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: cghrl	%r15, foo               # encoding: [0xc6,0xf4,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL

	cghrl	%r0,foo
	cghrl	%r15,foo

#CHECK: cghrl	%r3, bar+100            # encoding: [0xc6,0x34,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: cghrl	%r4, bar+100            # encoding: [0xc6,0x44,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL

	cghrl	%r3,bar+100
	cghrl	%r4,bar+100

#CHECK: cghrl	%r7, frob@PLT           # encoding: [0xc6,0x74,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL
#CHECK: cghrl	%r8, frob@PLT           # encoding: [0xc6,0x84,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL

	cghrl	%r7,frob@PLT
	cghrl	%r8,frob@PLT

#CHECK: cghsi	0, 0                    # encoding: [0xe5,0x58,0x00,0x00,0x00,0x00]
#CHECK: cghsi	4095, 0                 # encoding: [0xe5,0x58,0x0f,0xff,0x00,0x00]
#CHECK: cghsi	0, -32768               # encoding: [0xe5,0x58,0x00,0x00,0x80,0x00]
#CHECK: cghsi	0, -1                   # encoding: [0xe5,0x58,0x00,0x00,0xff,0xff]
#CHECK: cghsi	0, 0                    # encoding: [0xe5,0x58,0x00,0x00,0x00,0x00]
#CHECK: cghsi	0, 1                    # encoding: [0xe5,0x58,0x00,0x00,0x00,0x01]
#CHECK: cghsi	0, 32767                # encoding: [0xe5,0x58,0x00,0x00,0x7f,0xff]
#CHECK: cghsi	0(%r1), 42              # encoding: [0xe5,0x58,0x10,0x00,0x00,0x2a]
#CHECK: cghsi	0(%r15), 42             # encoding: [0xe5,0x58,0xf0,0x00,0x00,0x2a]
#CHECK: cghsi	4095(%r1), 42           # encoding: [0xe5,0x58,0x1f,0xff,0x00,0x2a]
#CHECK: cghsi	4095(%r15), 42          # encoding: [0xe5,0x58,0xff,0xff,0x00,0x2a]

	cghsi	0, 0
	cghsi	4095, 0
	cghsi	0, -32768
	cghsi	0, -1
	cghsi	0, 0
	cghsi	0, 1
	cghsi	0, 32767
	cghsi	0(%r1), 42
	cghsi	0(%r15), 42
	cghsi	4095(%r1), 42
	cghsi	4095(%r15), 42

#CHECK: cgib	%r0, 0, 0, 0            # encoding: [0xec,0x00,0x00,0x00,0x00,0xfc]
#CHECK: cgib	%r0, -128, 0, 0         # encoding: [0xec,0x00,0x00,0x00,0x80,0xfc]
#CHECK: cgib	%r0, 127, 0, 0          # encoding: [0xec,0x00,0x00,0x00,0x7f,0xfc]
#CHECK: cgib	%r15, 0, 0, 0           # encoding: [0xec,0xf0,0x00,0x00,0x00,0xfc]
#CHECK: cgib	%r7, -1, 0, 0           # encoding: [0xec,0x70,0x00,0x00,0xff,0xfc]
#CHECK: cgib	%r0, 0, 1, 0            # encoding: [0xec,0x01,0x00,0x00,0x00,0xfc]
#CHECK: cgib	%r0, 0, 15, 0           # encoding: [0xec,0x0f,0x00,0x00,0x00,0xfc]
#CHECK: cgib	%r0, 0, 0, 0(%r13)      # encoding: [0xec,0x00,0xd0,0x00,0x00,0xfc]
#CHECK: cgib	%r0, 0, 0, 4095         # encoding: [0xec,0x00,0x0f,0xff,0x00,0xfc]
#CHECK: cgib	%r0, 0, 0, 4095(%r7)    # encoding: [0xec,0x00,0x7f,0xff,0x00,0xfc]
	cgib	%r0, 0, 0, 0
	cgib	%r0, -128, 0, 0
	cgib	%r0, 127, 0, 0
	cgib	%r15, 0, 0, 0
	cgib	%r7, -1, 0, 0
	cgib	%r0, 0, 1, 0
	cgib	%r0, 0, 15, 0
	cgib	%r0, 0, 0, 0(%r13)
	cgib	%r0, 0, 0, 4095
	cgib	%r0, 0, 0, 4095(%r7)

#CHECK: cgibe	%r0, 0, 0               # encoding: [0xec,0x08,0x00,0x00,0x00,0xfc]
#CHECK: cgibe	%r0, -128, 0            # encoding: [0xec,0x08,0x00,0x00,0x80,0xfc]
#CHECK: cgibe	%r0, 127, 0             # encoding: [0xec,0x08,0x00,0x00,0x7f,0xfc]
#CHECK: cgibe	%r15, 0, 0              # encoding: [0xec,0xf8,0x00,0x00,0x00,0xfc]
#CHECK: cgibe	%r7, -1, 0              # encoding: [0xec,0x78,0x00,0x00,0xff,0xfc]
#CHECK: cgibe	%r0, 0, 0(%r13)         # encoding: [0xec,0x08,0xd0,0x00,0x00,0xfc]
#CHECK: cgibe	%r0, 0, 4095            # encoding: [0xec,0x08,0x0f,0xff,0x00,0xfc]
#CHECK: cgibe	%r0, 0, 4095(%r7)       # encoding: [0xec,0x08,0x7f,0xff,0x00,0xfc]
	cgibe	%r0, 0, 0
	cgibe	%r0, -128, 0
	cgibe	%r0, 127, 0
	cgibe	%r15, 0, 0
	cgibe	%r7, -1, 0
	cgibe	%r0, 0, 0(%r13)
	cgibe	%r0, 0, 4095
	cgibe	%r0, 0, 4095(%r7)

#CHECK: cgib	%r1, 2, 2, 3(%r4)       # encoding: [0xec,0x12,0x40,0x03,0x02,0xfc]
#CHECK: cgibh	%r1, 2, 3(%r4)          # encoding: [0xec,0x12,0x40,0x03,0x02,0xfc]
#CHECK: cgibnle	%r1, 2, 3(%r4)          # encoding: [0xec,0x12,0x40,0x03,0x02,0xfc]
	cgib	%r1, 2, 2, 3(%r4)
	cgibh	%r1, 2, 3(%r4)
	cgibnle	%r1, 2, 3(%r4)

#CHECK: cgib	%r1, 2, 4, 3(%r4)       # encoding: [0xec,0x14,0x40,0x03,0x02,0xfc]
#CHECK: cgibl	%r1, 2, 3(%r4)          # encoding: [0xec,0x14,0x40,0x03,0x02,0xfc]
#CHECK: cgibnhe	%r1, 2, 3(%r4)          # encoding: [0xec,0x14,0x40,0x03,0x02,0xfc]
	cgib	%r1, 2, 4, 3(%r4)
	cgibl	%r1, 2, 3(%r4)
	cgibnhe	%r1, 2, 3(%r4)

#CHECK: cgib	%r1, 2, 6, 3(%r4)       # encoding: [0xec,0x16,0x40,0x03,0x02,0xfc]
#CHECK: cgiblh	%r1, 2, 3(%r4)          # encoding: [0xec,0x16,0x40,0x03,0x02,0xfc]
#CHECK: cgibne	%r1, 2, 3(%r4)          # encoding: [0xec,0x16,0x40,0x03,0x02,0xfc]
	cgib	%r1, 2, 6, 3(%r4)
	cgiblh	%r1, 2, 3(%r4)
	cgibne	%r1, 2, 3(%r4)

#CHECK: cgib	%r1, 2, 8, 3(%r4)       # encoding: [0xec,0x18,0x40,0x03,0x02,0xfc]
#CHECK: cgibe	%r1, 2, 3(%r4)          # encoding: [0xec,0x18,0x40,0x03,0x02,0xfc]
#CHECK: cgibnlh	%r1, 2, 3(%r4)          # encoding: [0xec,0x18,0x40,0x03,0x02,0xfc]
	cgib	%r1, 2, 8, 3(%r4)
	cgibe	%r1, 2, 3(%r4)
	cgibnlh	%r1, 2, 3(%r4)

#CHECK: cgib	%r1, 2, 10, 3(%r4)      # encoding: [0xec,0x1a,0x40,0x03,0x02,0xfc]
#CHECK: cgibhe	%r1, 2, 3(%r4)          # encoding: [0xec,0x1a,0x40,0x03,0x02,0xfc]
#CHECK: cgibnl	%r1, 2, 3(%r4)          # encoding: [0xec,0x1a,0x40,0x03,0x02,0xfc]
	cgib	%r1, 2, 10, 3(%r4)
	cgibhe	%r1, 2, 3(%r4)
	cgibnl	%r1, 2, 3(%r4)

#CHECK: cgib	%r1, 2, 12, 3(%r4)      # encoding: [0xec,0x1c,0x40,0x03,0x02,0xfc]
#CHECK: cgible	%r1, 2, 3(%r4)          # encoding: [0xec,0x1c,0x40,0x03,0x02,0xfc]
#CHECK: cgibnh	%r1, 2, 3(%r4)          # encoding: [0xec,0x1c,0x40,0x03,0x02,0xfc]
	cgib	%r1, 2, 12, 3(%r4)
	cgible	%r1, 2, 3(%r4)
	cgibnh	%r1, 2, 3(%r4)

#CHECK: cgij	%r0, 0, 0, .[[LAB:L.*]]	# encoding: [0xec,0x00,A,A,0x00,0x7c]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC16DBL
#CHECK: cgij	%r0, -128, 0, .[[LAB:L.*]]	# encoding: [0xec,0x00,A,A,0x80,0x7c]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC16DBL
#CHECK: cgij	%r0, 127, 0, .[[LAB:L.*]]	# encoding: [0xec,0x00,A,A,0x7f,0x7c]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC16DBL
#CHECK: cgij	%r15, 0, 0, .[[LAB:L.*]]	# encoding: [0xec,0xf0,A,A,0x00,0x7c]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC16DBL
#CHECK: cgij	%r7, -1, 0, .[[LAB:L.*]]	# encoding: [0xec,0x70,A,A,0xff,0x7c]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC16DBL
	cgij	%r0, 0, 0, 0
	cgij	%r0, -128, 0, 0
	cgij	%r0, 127, 0, 0
	cgij	%r15, 0, 0, 0
	cgij	%r7, -1, 0, 0

#CHECK: cgij	%r1, -66, 0, .[[LAB:L.*]]-65536	# encoding: [0xec,0x10,A,A,0xbe,0x7c]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-65536)+2, kind: FK_390_PC16DBL
	cgij	%r1, -66, 0, -0x10000
#CHECK: cgij	%r1, -66, 0, .[[LAB:L.*]]-2	# encoding: [0xec,0x10,A,A,0xbe,0x7c]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC16DBL
	cgij	%r1, -66, 0, -2
#CHECK: cgij	%r1, -66, 0, .[[LAB:L.*]]		# encoding: [0xec,0x10,A,A,0xbe,0x7c]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC16DBL
	cgij	%r1, -66, 0, 0
#CHECK: cgij	%r1, -66, 0, .[[LAB:L.*]]+65534	# encoding: [0xec,0x10,A,A,0xbe,0x7c]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+65534)+2, kind: FK_390_PC16DBL
	cgij	%r1, -66, 0, 0xfffe

#CHECK: cgij	%r1, -66, 0, foo                  # encoding: [0xec,0x10,A,A,0xbe,0x7c]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	cgij	%r1, -66, 0, foo

#CHECK: cgij	%r1, -66, 1, foo                  # encoding: [0xec,0x11,A,A,0xbe,0x7c]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	cgij	%r1, -66, 1, foo

#CHECK: cgij	%r1, -66, 2, foo                  # encoding: [0xec,0x12,A,A,0xbe,0x7c]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: cgijh	%r1, -66, foo                     # encoding: [0xec,0x12,A,A,0xbe,0x7c]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: cgijnle	%r1, -66, foo                     # encoding: [0xec,0x12,A,A,0xbe,0x7c]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	cgij	%r1, -66, 2, foo
	cgijh	%r1, -66, foo
	cgijnle	%r1, -66, foo

#CHECK: cgij	%r1, -66, 3, foo                  # encoding: [0xec,0x13,A,A,0xbe,0x7c]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	cgij	%r1, -66, 3, foo

#CHECK: cgij	%r1, -66, 4, foo                  # encoding: [0xec,0x14,A,A,0xbe,0x7c]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: cgijl	%r1, -66, foo                     # encoding: [0xec,0x14,A,A,0xbe,0x7c]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: cgijnhe	%r1, -66, foo                     # encoding: [0xec,0x14,A,A,0xbe,0x7c]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	cgij	%r1, -66, 4, foo
	cgijl	%r1, -66, foo
	cgijnhe	%r1, -66, foo

#CHECK: cgij	%r1, -66, 5, foo                  # encoding: [0xec,0x15,A,A,0xbe,0x7c]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	cgij	%r1, -66, 5, foo

#CHECK: cgij	%r1, -66, 6, foo                  # encoding: [0xec,0x16,A,A,0xbe,0x7c]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: cgijlh	%r1, -66, foo                     # encoding: [0xec,0x16,A,A,0xbe,0x7c]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: cgijne	%r1, -66, foo                     # encoding: [0xec,0x16,A,A,0xbe,0x7c]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	cgij	%r1, -66, 6, foo
	cgijlh	%r1, -66, foo
	cgijne	%r1, -66, foo

#CHECK: cgij	%r1, -66, 7, foo                  # encoding: [0xec,0x17,A,A,0xbe,0x7c]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	cgij	%r1, -66, 7, foo

#CHECK: cgij	%r1, -66, 8, foo                  # encoding: [0xec,0x18,A,A,0xbe,0x7c]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: cgije	%r1, -66, foo                     # encoding: [0xec,0x18,A,A,0xbe,0x7c]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: cgijnlh	%r1, -66, foo                     # encoding: [0xec,0x18,A,A,0xbe,0x7c]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	cgij	%r1, -66, 8, foo
	cgije	%r1, -66, foo
	cgijnlh	%r1, -66, foo

#CHECK: cgij	%r1, -66, 9, foo                  # encoding: [0xec,0x19,A,A,0xbe,0x7c]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	cgij	%r1, -66, 9, foo

#CHECK: cgij	%r1, -66, 10, foo                 # encoding: [0xec,0x1a,A,A,0xbe,0x7c]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: cgijhe	%r1, -66, foo                     # encoding: [0xec,0x1a,A,A,0xbe,0x7c]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: cgijnl	%r1, -66, foo                     # encoding: [0xec,0x1a,A,A,0xbe,0x7c]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	cgij	%r1, -66, 10, foo
	cgijhe	%r1, -66, foo
	cgijnl	%r1, -66, foo

#CHECK: cgij	%r1, -66, 11, foo                 # encoding: [0xec,0x1b,A,A,0xbe,0x7c]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	cgij	%r1, -66, 11, foo

#CHECK: cgij	%r1, -66, 12, foo                 # encoding: [0xec,0x1c,A,A,0xbe,0x7c]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: cgijle	%r1, -66, foo                     # encoding: [0xec,0x1c,A,A,0xbe,0x7c]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: cgijnh	%r1, -66, foo                     # encoding: [0xec,0x1c,A,A,0xbe,0x7c]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	cgij	%r1, -66, 12, foo
	cgijle	%r1, -66, foo
	cgijnh	%r1, -66, foo

#CHECK: cgij	%r1, -66, 13, foo                 # encoding: [0xec,0x1d,A,A,0xbe,0x7c]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	cgij	%r1, -66, 13, foo

#CHECK: cgij	%r1, -66, 14, foo                 # encoding: [0xec,0x1e,A,A,0xbe,0x7c]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	cgij	%r1, -66, 14, foo

#CHECK: cgij	%r1, -66, 15, foo                 # encoding: [0xec,0x1f,A,A,0xbe,0x7c]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	cgij	%r1, -66, 15, foo

#CHECK: cgij	%r1, -66, 0, bar+100              # encoding: [0xec,0x10,A,A,0xbe,0x7c]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	cgij	%r1, -66, 0, bar+100

#CHECK: cgijh	%r1, -66, bar+100                 # encoding: [0xec,0x12,A,A,0xbe,0x7c]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	cgijh	%r1, -66, bar+100

#CHECK: cgijnle	%r1, -66, bar+100                 # encoding: [0xec,0x12,A,A,0xbe,0x7c]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	cgijnle	%r1, -66, bar+100

#CHECK: cgijl	%r1, -66, bar+100                 # encoding: [0xec,0x14,A,A,0xbe,0x7c]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	cgijl	%r1, -66, bar+100

#CHECK: cgijnhe	%r1, -66, bar+100                 # encoding: [0xec,0x14,A,A,0xbe,0x7c]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	cgijnhe	%r1, -66, bar+100

#CHECK: cgijlh	%r1, -66, bar+100                 # encoding: [0xec,0x16,A,A,0xbe,0x7c]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	cgijlh	%r1, -66, bar+100

#CHECK: cgijne	%r1, -66, bar+100                 # encoding: [0xec,0x16,A,A,0xbe,0x7c]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	cgijne	%r1, -66, bar+100

#CHECK: cgije	%r1, -66, bar+100                 # encoding: [0xec,0x18,A,A,0xbe,0x7c]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	cgije	%r1, -66, bar+100

#CHECK: cgijnlh	%r1, -66, bar+100                 # encoding: [0xec,0x18,A,A,0xbe,0x7c]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	cgijnlh	%r1, -66, bar+100

#CHECK: cgijhe	%r1, -66, bar+100                 # encoding: [0xec,0x1a,A,A,0xbe,0x7c]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	cgijhe	%r1, -66, bar+100

#CHECK: cgijnl	%r1, -66, bar+100                 # encoding: [0xec,0x1a,A,A,0xbe,0x7c]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	cgijnl	%r1, -66, bar+100

#CHECK: cgijle	%r1, -66, bar+100                 # encoding: [0xec,0x1c,A,A,0xbe,0x7c]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	cgijle	%r1, -66, bar+100

#CHECK: cgijnh	%r1, -66, bar+100                 # encoding: [0xec,0x1c,A,A,0xbe,0x7c]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	cgijnh	%r1, -66, bar+100

#CHECK: cgij	%r1, -66, 0, bar@PLT              # encoding: [0xec,0x10,A,A,0xbe,0x7c]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	cgij	%r1, -66, 0, bar@PLT

#CHECK: cgijh	%r1, -66, bar@PLT                 # encoding: [0xec,0x12,A,A,0xbe,0x7c]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	cgijh	%r1, -66, bar@PLT

#CHECK: cgijnle	%r1, -66, bar@PLT                 # encoding: [0xec,0x12,A,A,0xbe,0x7c]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	cgijnle	%r1, -66, bar@PLT

#CHECK: cgijl	%r1, -66, bar@PLT                 # encoding: [0xec,0x14,A,A,0xbe,0x7c]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	cgijl	%r1, -66, bar@PLT

#CHECK: cgijnhe	%r1, -66, bar@PLT                 # encoding: [0xec,0x14,A,A,0xbe,0x7c]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	cgijnhe	%r1, -66, bar@PLT

#CHECK: cgijlh	%r1, -66, bar@PLT                 # encoding: [0xec,0x16,A,A,0xbe,0x7c]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	cgijlh	%r1, -66, bar@PLT

#CHECK: cgijne	%r1, -66, bar@PLT                 # encoding: [0xec,0x16,A,A,0xbe,0x7c]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	cgijne	%r1, -66, bar@PLT

#CHECK: cgije	%r1, -66, bar@PLT                 # encoding: [0xec,0x18,A,A,0xbe,0x7c]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	cgije	%r1, -66, bar@PLT

#CHECK: cgijnlh	%r1, -66, bar@PLT                 # encoding: [0xec,0x18,A,A,0xbe,0x7c]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	cgijnlh	%r1, -66, bar@PLT

#CHECK: cgijhe	%r1, -66, bar@PLT                 # encoding: [0xec,0x1a,A,A,0xbe,0x7c]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	cgijhe	%r1, -66, bar@PLT

#CHECK: cgijnl	%r1, -66, bar@PLT                 # encoding: [0xec,0x1a,A,A,0xbe,0x7c]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	cgijnl	%r1, -66, bar@PLT

#CHECK: cgijle	%r1, -66, bar@PLT                 # encoding: [0xec,0x1c,A,A,0xbe,0x7c]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	cgijle	%r1, -66, bar@PLT

#CHECK: cgijnh	%r1, -66, bar@PLT                 # encoding: [0xec,0x1c,A,A,0xbe,0x7c]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	cgijnh	%r1, -66, bar@PLT

#CHECK: cgit     %r0, 0, 12             # encoding: [0xec,0x00,0x00,0x00,0xc0,0x70]
#CHECK: cgit     %r0, -1, 12            # encoding: [0xec,0x00,0xff,0xff,0xc0,0x70]
#CHECK: cgit     %r0, -32768, 12        # encoding: [0xec,0x00,0x80,0x00,0xc0,0x70]
#CHECK: cgit     %r0, 32767, 12         # encoding: [0xec,0x00,0x7f,0xff,0xc0,0x70]
#CHECK: cgith    %r15, 1                # encoding: [0xec,0xf0,0x00,0x01,0x20,0x70]
#CHECK: cgitl    %r15, 1                # encoding: [0xec,0xf0,0x00,0x01,0x40,0x70]
#CHECK: cgite    %r15, 1                # encoding: [0xec,0xf0,0x00,0x01,0x80,0x70]
#CHECK: cgitne   %r15, 1                # encoding: [0xec,0xf0,0x00,0x01,0x60,0x70]
#CHECK: cgitnl   %r15, 1                # encoding: [0xec,0xf0,0x00,0x01,0xa0,0x70]
#CHECK: cgitnh   %r15, 1                # encoding: [0xec,0xf0,0x00,0x01,0xc0,0x70]
#CHECK: cgitnle  %r15, 1                # encoding: [0xec,0xf0,0x00,0x01,0x20,0x70]
#CHECK: cgitnhe  %r15, 1                # encoding: [0xec,0xf0,0x00,0x01,0x40,0x70]
#CHECK: cgitnlh  %r15, 1                # encoding: [0xec,0xf0,0x00,0x01,0x80,0x70]
#CHECK: cgitlh   %r15, 1                # encoding: [0xec,0xf0,0x00,0x01,0x60,0x70]
#CHECK: cgithe   %r15, 1                # encoding: [0xec,0xf0,0x00,0x01,0xa0,0x70]
#CHECK: cgitle   %r15, 1                # encoding: [0xec,0xf0,0x00,0x01,0xc0,0x70]

        cgit     %r0, 0, 12
        cgit     %r0, -1, 12
        cgit     %r0, -32768, 12
        cgit     %r0, 32767, 12
        cgith    %r15, 1
        cgitl    %r15, 1
        cgite    %r15, 1
        cgitne   %r15, 1
        cgitnl   %r15, 1
        cgitnh   %r15, 1
        cgitnle  %r15, 1
        cgitnhe  %r15, 1
        cgitnlh  %r15, 1
        cgitlh   %r15, 1
        cgithe   %r15, 1
        cgitle   %r15, 1

#CHECK: cgr	%r0, %r0                # encoding: [0xb9,0x20,0x00,0x00]
#CHECK: cgr	%r0, %r15               # encoding: [0xb9,0x20,0x00,0x0f]
#CHECK: cgr	%r15, %r0               # encoding: [0xb9,0x20,0x00,0xf0]
#CHECK: cgr	%r7, %r8                # encoding: [0xb9,0x20,0x00,0x78]

	cgr	%r0,%r0
	cgr	%r0,%r15
	cgr	%r15,%r0
	cgr	%r7,%r8

#CHECK: cgrb	%r0, %r0, 0, 0          # encoding: [0xec,0x00,0x00,0x00,0x00,0xe4]
#CHECK: cgrb	%r0, %r15, 0, 0         # encoding: [0xec,0x0f,0x00,0x00,0x00,0xe4]
#CHECK: cgrb	%r15, %r0, 0, 0         # encoding: [0xec,0xf0,0x00,0x00,0x00,0xe4]
#CHECK: cgrb	%r7, %r2, 0, 0          # encoding: [0xec,0x72,0x00,0x00,0x00,0xe4]
#CHECK: cgrb	%r0, %r0, 1, 0          # encoding: [0xec,0x00,0x00,0x00,0x10,0xe4]
#CHECK: cgrb	%r0, %r0, 15, 0         # encoding: [0xec,0x00,0x00,0x00,0xf0,0xe4]
#CHECK: cgrb	%r0, %r0, 0, 0(%r13)    # encoding: [0xec,0x00,0xd0,0x00,0x00,0xe4]
#CHECK: cgrb	%r0, %r0, 0, 4095       # encoding: [0xec,0x00,0x0f,0xff,0x00,0xe4]
#CHECK: cgrb	%r0, %r0, 0, 4095(%r7)  # encoding: [0xec,0x00,0x7f,0xff,0x00,0xe4]
	cgrb	%r0, %r0, 0, 0
	cgrb	%r0, %r15, 0, 0
	cgrb	%r15, %r0, 0, 0
	cgrb	%r7, %r2, 0, 0
	cgrb	%r0, %r0, 1, 0
	cgrb	%r0, %r0, 15, 0
	cgrb	%r0, %r0, 0, 0(%r13)
	cgrb	%r0, %r0, 0, 4095
	cgrb	%r0, %r0, 0, 4095(%r7)

#CHECK: cgrbe	%r0, %r0, 0             # encoding: [0xec,0x00,0x00,0x00,0x80,0xe4]
#CHECK: cgrbe	%r0, %r15, 0            # encoding: [0xec,0x0f,0x00,0x00,0x80,0xe4]
#CHECK: cgrbe	%r15, %r0, 0            # encoding: [0xec,0xf0,0x00,0x00,0x80,0xe4]
#CHECK: cgrbe	%r7, %r2, 0             # encoding: [0xec,0x72,0x00,0x00,0x80,0xe4]
#CHECK: cgrbe	%r0, %r0, 0(%r13)       # encoding: [0xec,0x00,0xd0,0x00,0x80,0xe4]
#CHECK: cgrbe	%r0, %r0, 4095          # encoding: [0xec,0x00,0x0f,0xff,0x80,0xe4]
#CHECK: cgrbe	%r0, %r0, 4095(%r7)     # encoding: [0xec,0x00,0x7f,0xff,0x80,0xe4]
	cgrbe	%r0, %r0, 0
	cgrbe	%r0, %r15, 0
	cgrbe	%r15, %r0, 0
	cgrbe	%r7, %r2, 0
	cgrbe	%r0, %r0, 0(%r13)
	cgrbe	%r0, %r0, 4095
	cgrbe	%r0, %r0, 4095(%r7)

#CHECK: cgrb	%r1, %r2, 2, 3(%r4)     # encoding: [0xec,0x12,0x40,0x03,0x20,0xe4]
#CHECK: cgrbh	%r1, %r2, 3(%r4)        # encoding: [0xec,0x12,0x40,0x03,0x20,0xe4]
#CHECK: cgrbnle	%r1, %r2, 3(%r4)        # encoding: [0xec,0x12,0x40,0x03,0x20,0xe4]
	cgrb	%r1, %r2, 2, 3(%r4)
	cgrbh	%r1, %r2, 3(%r4)
	cgrbnle	%r1, %r2, 3(%r4)

#CHECK: cgrb	%r1, %r2, 4, 3(%r4)     # encoding: [0xec,0x12,0x40,0x03,0x40,0xe4]
#CHECK: cgrbl	%r1, %r2, 3(%r4)        # encoding: [0xec,0x12,0x40,0x03,0x40,0xe4]
#CHECK: cgrbnhe	%r1, %r2, 3(%r4)        # encoding: [0xec,0x12,0x40,0x03,0x40,0xe4]
	cgrb	%r1, %r2, 4, 3(%r4)
	cgrbl	%r1, %r2, 3(%r4)
	cgrbnhe	%r1, %r2, 3(%r4)

#CHECK: cgrb	%r1, %r2, 6, 3(%r4)     # encoding: [0xec,0x12,0x40,0x03,0x60,0xe4]
#CHECK: cgrblh	%r1, %r2, 3(%r4)        # encoding: [0xec,0x12,0x40,0x03,0x60,0xe4]
#CHECK: cgrbne	%r1, %r2, 3(%r4)        # encoding: [0xec,0x12,0x40,0x03,0x60,0xe4]
	cgrb	%r1, %r2, 6, 3(%r4)
	cgrblh	%r1, %r2, 3(%r4)
	cgrbne	%r1, %r2, 3(%r4)

#CHECK: cgrb	%r1, %r2, 8, 3(%r4)     # encoding: [0xec,0x12,0x40,0x03,0x80,0xe4]
#CHECK: cgrbe	%r1, %r2, 3(%r4)        # encoding: [0xec,0x12,0x40,0x03,0x80,0xe4]
#CHECK: cgrbnlh	%r1, %r2, 3(%r4)        # encoding: [0xec,0x12,0x40,0x03,0x80,0xe4]
	cgrb	%r1, %r2, 8, 3(%r4)
	cgrbe	%r1, %r2, 3(%r4)
	cgrbnlh	%r1, %r2, 3(%r4)

#CHECK: cgrb	%r1, %r2, 10, 3(%r4)    # encoding: [0xec,0x12,0x40,0x03,0xa0,0xe4]
#CHECK: cgrbhe	%r1, %r2, 3(%r4)        # encoding: [0xec,0x12,0x40,0x03,0xa0,0xe4]
#CHECK: cgrbnl	%r1, %r2, 3(%r4)        # encoding: [0xec,0x12,0x40,0x03,0xa0,0xe4]
	cgrb	%r1, %r2, 10, 3(%r4)
	cgrbhe	%r1, %r2, 3(%r4)
	cgrbnl	%r1, %r2, 3(%r4)

#CHECK: cgrb	%r1, %r2, 12, 3(%r4)    # encoding: [0xec,0x12,0x40,0x03,0xc0,0xe4]
#CHECK: cgrble	%r1, %r2, 3(%r4)        # encoding: [0xec,0x12,0x40,0x03,0xc0,0xe4]
#CHECK: cgrbnh	%r1, %r2, 3(%r4)        # encoding: [0xec,0x12,0x40,0x03,0xc0,0xe4]
	cgrb	%r1, %r2, 12, 3(%r4)
	cgrble	%r1, %r2, 3(%r4)
	cgrbnh	%r1, %r2, 3(%r4)

#CHECK: cgrj	%r0, %r0, 0, .[[LAB:L.*]]	# encoding: [0xec,0x00,A,A,0x00,0x64]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC16DBL
#CHECK: cgrj	%r0, %r15, 0, .[[LAB:L.*]]	# encoding: [0xec,0x0f,A,A,0x00,0x64]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC16DBL
#CHECK: cgrj	%r15, %r0, 0, .[[LAB:L.*]]	# encoding: [0xec,0xf0,A,A,0x00,0x64]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC16DBL
#CHECK: cgrj	%r7, %r8, 0, .[[LAB:L.*]]	# encoding: [0xec,0x78,A,A,0x00,0x64]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC16DBL
	cgrj	%r0,%r0,0,0
	cgrj	%r0,%r15,0,0
	cgrj	%r15,%r0,0,0
	cgrj	%r7,%r8,0,0

#CHECK: cgrj	%r1, %r2, 0, .[[LAB:L.*]]-65536	# encoding: [0xec,0x12,A,A,0x00,0x64]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-65536)+2, kind: FK_390_PC16DBL
	cgrj	%r1, %r2, 0, -0x10000
#CHECK: cgrj	%r1, %r2, 0, .[[LAB:L.*]]-2	# encoding: [0xec,0x12,A,A,0x00,0x64]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC16DBL
	cgrj	%r1, %r2, 0, -2
#CHECK: cgrj	%r1, %r2, 0, .[[LAB:L.*]]		# encoding: [0xec,0x12,A,A,0x00,0x64]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC16DBL
	cgrj	%r1, %r2, 0, 0
#CHECK: cgrj	%r1, %r2, 0, .[[LAB:L.*]]+65534	# encoding: [0xec,0x12,A,A,0x00,0x64]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+65534)+2, kind: FK_390_PC16DBL
	cgrj	%r1, %r2, 0, 0xfffe

#CHECK: cgrj	%r1, %r2, 0, foo                  # encoding: [0xec,0x12,A,A,0x00,0x64]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	cgrj	%r1, %r2, 0, foo

#CHECK: cgrj	%r1, %r2, 1, foo                  # encoding: [0xec,0x12,A,A,0x10,0x64]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	cgrj	%r1, %r2, 1, foo

#CHECK: cgrj	%r1, %r2, 2, foo                  # encoding: [0xec,0x12,A,A,0x20,0x64]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: cgrjh	%r1, %r2, foo                     # encoding: [0xec,0x12,A,A,0x20,0x64]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: cgrjnle	%r1, %r2, foo                     # encoding: [0xec,0x12,A,A,0x20,0x64]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	cgrj	%r1, %r2, 2, foo
	cgrjh	%r1, %r2, foo
	cgrjnle	%r1, %r2, foo

#CHECK: cgrj	%r1, %r2, 3, foo                  # encoding: [0xec,0x12,A,A,0x30,0x64]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	cgrj	%r1, %r2, 3, foo

#CHECK: cgrj	%r1, %r2, 4, foo                  # encoding: [0xec,0x12,A,A,0x40,0x64]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: cgrjl	%r1, %r2, foo                     # encoding: [0xec,0x12,A,A,0x40,0x64]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: cgrjnhe	%r1, %r2, foo                     # encoding: [0xec,0x12,A,A,0x40,0x64]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	cgrj	%r1, %r2, 4, foo
	cgrjl	%r1, %r2, foo
	cgrjnhe	%r1, %r2, foo

#CHECK: cgrj	%r1, %r2, 5, foo                  # encoding: [0xec,0x12,A,A,0x50,0x64]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	cgrj	%r1, %r2, 5, foo

#CHECK: cgrj	%r1, %r2, 6, foo                  # encoding: [0xec,0x12,A,A,0x60,0x64]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: cgrjlh	%r1, %r2, foo                     # encoding: [0xec,0x12,A,A,0x60,0x64]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: cgrjne	%r1, %r2, foo                     # encoding: [0xec,0x12,A,A,0x60,0x64]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	cgrj	%r1, %r2, 6, foo
	cgrjlh	%r1, %r2, foo
	cgrjne	%r1, %r2, foo

#CHECK: cgrj	%r1, %r2, 7, foo                  # encoding: [0xec,0x12,A,A,0x70,0x64]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	cgrj	%r1, %r2, 7, foo

#CHECK: cgrj	%r1, %r2, 8, foo                  # encoding: [0xec,0x12,A,A,0x80,0x64]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: cgrje	%r1, %r2, foo                     # encoding: [0xec,0x12,A,A,0x80,0x64]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: cgrjnlh	%r1, %r2, foo                     # encoding: [0xec,0x12,A,A,0x80,0x64]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	cgrj	%r1, %r2, 8, foo
	cgrje	%r1, %r2, foo
	cgrjnlh	%r1, %r2, foo

#CHECK: cgrj	%r1, %r2, 9, foo                  # encoding: [0xec,0x12,A,A,0x90,0x64]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	cgrj	%r1, %r2, 9, foo

#CHECK: cgrj	%r1, %r2, 10, foo                 # encoding: [0xec,0x12,A,A,0xa0,0x64]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: cgrjhe	%r1, %r2, foo                     # encoding: [0xec,0x12,A,A,0xa0,0x64]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: cgrjnl	%r1, %r2, foo                     # encoding: [0xec,0x12,A,A,0xa0,0x64]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	cgrj	%r1, %r2, 10, foo
	cgrjhe	%r1, %r2, foo
	cgrjnl	%r1, %r2, foo

#CHECK: cgrj	%r1, %r2, 11, foo                 # encoding: [0xec,0x12,A,A,0xb0,0x64]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	cgrj	%r1, %r2, 11, foo

#CHECK: cgrj	%r1, %r2, 12, foo                 # encoding: [0xec,0x12,A,A,0xc0,0x64]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: cgrjle	%r1, %r2, foo                     # encoding: [0xec,0x12,A,A,0xc0,0x64]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: cgrjnh	%r1, %r2, foo                     # encoding: [0xec,0x12,A,A,0xc0,0x64]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	cgrj	%r1, %r2, 12, foo
	cgrjle	%r1, %r2, foo
	cgrjnh	%r1, %r2, foo

#CHECK: cgrj	%r1, %r2, 13, foo                 # encoding: [0xec,0x12,A,A,0xd0,0x64]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	cgrj	%r1, %r2, 13, foo

#CHECK: cgrj	%r1, %r2, 14, foo                 # encoding: [0xec,0x12,A,A,0xe0,0x64]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	cgrj	%r1, %r2, 14, foo

#CHECK: cgrj	%r1, %r2, 15, foo                 # encoding: [0xec,0x12,A,A,0xf0,0x64]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	cgrj	%r1, %r2, 15, foo

#CHECK: cgrj	%r1, %r2, 0, bar+100              # encoding: [0xec,0x12,A,A,0x00,0x64]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	cgrj	%r1, %r2, 0, bar+100

#CHECK: cgrjh	%r1, %r2, bar+100                 # encoding: [0xec,0x12,A,A,0x20,0x64]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	cgrjh	%r1, %r2, bar+100

#CHECK: cgrjnle	%r1, %r2, bar+100                 # encoding: [0xec,0x12,A,A,0x20,0x64]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	cgrjnle	%r1, %r2, bar+100

#CHECK: cgrjl	%r1, %r2, bar+100                 # encoding: [0xec,0x12,A,A,0x40,0x64]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	cgrjl	%r1, %r2, bar+100

#CHECK: cgrjnhe	%r1, %r2, bar+100                 # encoding: [0xec,0x12,A,A,0x40,0x64]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	cgrjnhe	%r1, %r2, bar+100

#CHECK: cgrjlh	%r1, %r2, bar+100                 # encoding: [0xec,0x12,A,A,0x60,0x64]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	cgrjlh	%r1, %r2, bar+100

#CHECK: cgrjne	%r1, %r2, bar+100                 # encoding: [0xec,0x12,A,A,0x60,0x64]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	cgrjne	%r1, %r2, bar+100

#CHECK: cgrje	%r1, %r2, bar+100                 # encoding: [0xec,0x12,A,A,0x80,0x64]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	cgrje	%r1, %r2, bar+100

#CHECK: cgrjnlh	%r1, %r2, bar+100                 # encoding: [0xec,0x12,A,A,0x80,0x64]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	cgrjnlh	%r1, %r2, bar+100

#CHECK: cgrjhe	%r1, %r2, bar+100                 # encoding: [0xec,0x12,A,A,0xa0,0x64]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	cgrjhe	%r1, %r2, bar+100

#CHECK: cgrjnl	%r1, %r2, bar+100                 # encoding: [0xec,0x12,A,A,0xa0,0x64]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	cgrjnl	%r1, %r2, bar+100

#CHECK: cgrjle	%r1, %r2, bar+100                 # encoding: [0xec,0x12,A,A,0xc0,0x64]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	cgrjle	%r1, %r2, bar+100

#CHECK: cgrjnh	%r1, %r2, bar+100                 # encoding: [0xec,0x12,A,A,0xc0,0x64]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	cgrjnh	%r1, %r2, bar+100

#CHECK: cgrj	%r1, %r2, 0, bar@PLT              # encoding: [0xec,0x12,A,A,0x00,0x64]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	cgrj	%r1, %r2, 0, bar@PLT

#CHECK: cgrjh	%r1, %r2, bar@PLT                 # encoding: [0xec,0x12,A,A,0x20,0x64]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	cgrjh	%r1, %r2, bar@PLT

#CHECK: cgrjnle	%r1, %r2, bar@PLT                 # encoding: [0xec,0x12,A,A,0x20,0x64]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	cgrjnle	%r1, %r2, bar@PLT

#CHECK: cgrjl	%r1, %r2, bar@PLT                 # encoding: [0xec,0x12,A,A,0x40,0x64]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	cgrjl	%r1, %r2, bar@PLT

#CHECK: cgrjnhe	%r1, %r2, bar@PLT                 # encoding: [0xec,0x12,A,A,0x40,0x64]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	cgrjnhe	%r1, %r2, bar@PLT

#CHECK: cgrjlh	%r1, %r2, bar@PLT                 # encoding: [0xec,0x12,A,A,0x60,0x64]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	cgrjlh	%r1, %r2, bar@PLT

#CHECK: cgrjne	%r1, %r2, bar@PLT                 # encoding: [0xec,0x12,A,A,0x60,0x64]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	cgrjne	%r1, %r2, bar@PLT

#CHECK: cgrje	%r1, %r2, bar@PLT                 # encoding: [0xec,0x12,A,A,0x80,0x64]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	cgrje	%r1, %r2, bar@PLT

#CHECK: cgrjnlh	%r1, %r2, bar@PLT                 # encoding: [0xec,0x12,A,A,0x80,0x64]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	cgrjnlh	%r1, %r2, bar@PLT

#CHECK: cgrjhe	%r1, %r2, bar@PLT                 # encoding: [0xec,0x12,A,A,0xa0,0x64]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	cgrjhe	%r1, %r2, bar@PLT

#CHECK: cgrjnl	%r1, %r2, bar@PLT                 # encoding: [0xec,0x12,A,A,0xa0,0x64]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	cgrjnl	%r1, %r2, bar@PLT

#CHECK: cgrjle	%r1, %r2, bar@PLT                 # encoding: [0xec,0x12,A,A,0xc0,0x64]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	cgrjle	%r1, %r2, bar@PLT

#CHECK: cgrjnh	%r1, %r2, bar@PLT                 # encoding: [0xec,0x12,A,A,0xc0,0x64]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	cgrjnh	%r1, %r2, bar@PLT

#CHECK: cgrl	%r0, .[[LAB:L.*]]-4294967296 # encoding: [0xc6,0x08,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-4294967296)+2, kind: FK_390_PC32DBL
	cgrl	%r0, -0x100000000
#CHECK: cgrl	%r0, .[[LAB:L.*]]-2	# encoding: [0xc6,0x08,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC32DBL
	cgrl	%r0, -2
#CHECK: cgrl	%r0, .[[LAB:L.*]]	# encoding: [0xc6,0x08,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC32DBL
	cgrl	%r0, 0
#CHECK: cgrl	%r0, .[[LAB:L.*]]+4294967294 # encoding: [0xc6,0x08,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+4294967294)+2, kind: FK_390_PC32DBL
	cgrl	%r0, 0xfffffffe

#CHECK: cgrl	%r0, foo                # encoding: [0xc6,0x08,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: cgrl	%r15, foo               # encoding: [0xc6,0xf8,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL

	cgrl	%r0,foo
	cgrl	%r15,foo

#CHECK: cgrl	%r3, bar+100            # encoding: [0xc6,0x38,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: cgrl	%r4, bar+100            # encoding: [0xc6,0x48,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL

	cgrl	%r3,bar+100
	cgrl	%r4,bar+100

#CHECK: cgrl	%r7, frob@PLT           # encoding: [0xc6,0x78,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL
#CHECK: cgrl	%r8, frob@PLT           # encoding: [0xc6,0x88,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL

	cgrl	%r7,frob@PLT
	cgrl	%r8,frob@PLT

#CHECK: cgrt     %r0, %r1, 12           # encoding: [0xb9,0x60,0xc0,0x01]
#CHECK: cgrt     %r0, %r1, 12           # encoding: [0xb9,0x60,0xc0,0x01]
#CHECK: cgrt     %r0, %r1, 12           # encoding: [0xb9,0x60,0xc0,0x01]
#CHECK: cgrt     %r0, %r1, 12           # encoding: [0xb9,0x60,0xc0,0x01]
#CHECK: cgrth    %r0, %r15              # encoding: [0xb9,0x60,0x20,0x0f]
#CHECK: cgrtl    %r0, %r15              # encoding: [0xb9,0x60,0x40,0x0f]
#CHECK: cgrte    %r0, %r15              # encoding: [0xb9,0x60,0x80,0x0f]
#CHECK: cgrtne   %r0, %r15              # encoding: [0xb9,0x60,0x60,0x0f]
#CHECK: cgrtnl   %r0, %r15              # encoding: [0xb9,0x60,0xa0,0x0f]
#CHECK: cgrtnh   %r0, %r15              # encoding: [0xb9,0x60,0xc0,0x0f]
#CHECK: cgrtnle  %r0, %r15              # encoding: [0xb9,0x60,0x20,0x0f]
#CHECK: cgrtnhe  %r0, %r15              # encoding: [0xb9,0x60,0x40,0x0f]
#CHECK: cgrtnlh  %r0, %r15              # encoding: [0xb9,0x60,0x80,0x0f]
#CHECK: cgrtlh   %r0, %r15              # encoding: [0xb9,0x60,0x60,0x0f]
#CHECK: cgrthe   %r0, %r15              # encoding: [0xb9,0x60,0xa0,0x0f]
#CHECK: cgrtle   %r0, %r15              # encoding: [0xb9,0x60,0xc0,0x0f]

        cgrt     %r0, %r1, 12
        cgrt     %r0, %r1, 12
        cgrt     %r0, %r1, 12
        cgrt     %r0, %r1, 12
        cgrth    %r0, %r15
        cgrtl    %r0, %r15
        cgrte    %r0, %r15
        cgrtne   %r0, %r15
        cgrtnl   %r0, %r15
        cgrtnh   %r0, %r15
        cgrtnle  %r0, %r15
        cgrtnhe  %r0, %r15
        cgrtnlh  %r0, %r15
        cgrtlh   %r0, %r15
        cgrthe   %r0, %r15
        cgrtle   %r0, %r15

#CHECK: cgxbr	%r0, 0, %f0             # encoding: [0xb3,0xaa,0x00,0x00]
#CHECK: cgxbr	%r0, 0, %f13            # encoding: [0xb3,0xaa,0x00,0x0d]
#CHECK: cgxbr	%r0, 15, %f0            # encoding: [0xb3,0xaa,0xf0,0x00]
#CHECK: cgxbr	%r4, 5, %f8             # encoding: [0xb3,0xaa,0x50,0x48]
#CHECK: cgxbr	%r15, 0, %f0            # encoding: [0xb3,0xaa,0x00,0xf0]

	cgxbr	%r0, 0, %f0
	cgxbr	%r0, 0, %f13
	cgxbr	%r0, 15, %f0
	cgxbr	%r4, 5, %f8
	cgxbr	%r15, 0, %f0

#CHECK: cgxr	%r0, 0, %f0             # encoding: [0xb3,0xca,0x00,0x00]
#CHECK: cgxr	%r0, 0, %f13            # encoding: [0xb3,0xca,0x00,0x0d]
#CHECK: cgxr	%r0, 15, %f0            # encoding: [0xb3,0xca,0xf0,0x00]
#CHECK: cgxr	%r4, 5, %f8             # encoding: [0xb3,0xca,0x50,0x48]
#CHECK: cgxr	%r15, 0, %f0            # encoding: [0xb3,0xca,0x00,0xf0]

	cgxr	%r0, 0, %f0
	cgxr	%r0, 0, %f13
	cgxr	%r0, 15, %f0
	cgxr	%r4, 5, %f8
	cgxr	%r15, 0, %f0

#CHECK: cgxtr	%r0, 0, %f0             # encoding: [0xb3,0xe9,0x00,0x00]
#CHECK: cgxtr	%r0, 0, %f13            # encoding: [0xb3,0xe9,0x00,0x0d]
#CHECK: cgxtr	%r0, 15, %f0            # encoding: [0xb3,0xe9,0xf0,0x00]
#CHECK: cgxtr	%r4, 5, %f8             # encoding: [0xb3,0xe9,0x50,0x48]
#CHECK: cgxtr	%r15, 0, %f0            # encoding: [0xb3,0xe9,0x00,0xf0]

	cgxtr	%r0, 0, %f0
	cgxtr	%r0, 0, %f13
	cgxtr	%r0, 15, %f0
	cgxtr	%r4, 5, %f8
	cgxtr	%r15, 0, %f0

#CHECK: ch	%r0, 0                  # encoding: [0x49,0x00,0x00,0x00]
#CHECK: ch	%r0, 4095               # encoding: [0x49,0x00,0x0f,0xff]
#CHECK: ch	%r0, 0(%r1)             # encoding: [0x49,0x00,0x10,0x00]
#CHECK: ch	%r0, 0(%r15)            # encoding: [0x49,0x00,0xf0,0x00]
#CHECK: ch	%r0, 4095(%r1,%r15)     # encoding: [0x49,0x01,0xff,0xff]
#CHECK: ch	%r0, 4095(%r15,%r1)     # encoding: [0x49,0x0f,0x1f,0xff]
#CHECK: ch	%r15, 0                 # encoding: [0x49,0xf0,0x00,0x00]

	ch	%r0, 0
	ch	%r0, 4095
	ch	%r0, 0(%r1)
	ch	%r0, 0(%r15)
	ch	%r0, 4095(%r1,%r15)
	ch	%r0, 4095(%r15,%r1)
	ch	%r15, 0

#CHECK: chhsi	0, 0                    # encoding: [0xe5,0x54,0x00,0x00,0x00,0x00]
#CHECK: chhsi	4095, 0                 # encoding: [0xe5,0x54,0x0f,0xff,0x00,0x00]
#CHECK: chhsi	0, -32768               # encoding: [0xe5,0x54,0x00,0x00,0x80,0x00]
#CHECK: chhsi	0, -1                   # encoding: [0xe5,0x54,0x00,0x00,0xff,0xff]
#CHECK: chhsi	0, 0                    # encoding: [0xe5,0x54,0x00,0x00,0x00,0x00]
#CHECK: chhsi	0, 1                    # encoding: [0xe5,0x54,0x00,0x00,0x00,0x01]
#CHECK: chhsi	0, 32767                # encoding: [0xe5,0x54,0x00,0x00,0x7f,0xff]
#CHECK: chhsi	0(%r1), 42              # encoding: [0xe5,0x54,0x10,0x00,0x00,0x2a]
#CHECK: chhsi	0(%r15), 42             # encoding: [0xe5,0x54,0xf0,0x00,0x00,0x2a]
#CHECK: chhsi	4095(%r1), 42           # encoding: [0xe5,0x54,0x1f,0xff,0x00,0x2a]
#CHECK: chhsi	4095(%r15), 42          # encoding: [0xe5,0x54,0xff,0xff,0x00,0x2a]

	chhsi	0, 0
	chhsi	4095, 0
	chhsi	0, -32768
	chhsi	0, -1
	chhsi	0, 0
	chhsi	0, 1
	chhsi	0, 32767
	chhsi	0(%r1), 42
	chhsi	0(%r15), 42
	chhsi	4095(%r1), 42
	chhsi	4095(%r15), 42

#CHECK: chi	%r0, -32768             # encoding: [0xa7,0x0e,0x80,0x00]
#CHECK: chi	%r0, -1                 # encoding: [0xa7,0x0e,0xff,0xff]
#CHECK: chi	%r0, 0                  # encoding: [0xa7,0x0e,0x00,0x00]
#CHECK: chi	%r0, 1                  # encoding: [0xa7,0x0e,0x00,0x01]
#CHECK: chi	%r0, 32767              # encoding: [0xa7,0x0e,0x7f,0xff]
#CHECK: chi	%r15, 0                 # encoding: [0xa7,0xfe,0x00,0x00]
#CHECK: chi	%r0, foo                # encoding: [0xa7,0x0e,A,A]

	chi	%r0, -32768
	chi	%r0, -1
	chi	%r0, 0
	chi	%r0, 1
	chi	%r0, 32767
	chi	%r15, 0
	chi	%r0, foo

#CHECK: chrl	%r0, .[[LAB:L.*]]-4294967296 # encoding: [0xc6,0x05,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-4294967296)+2, kind: FK_390_PC32DBL
	chrl	%r0, -0x100000000
#CHECK: chrl	%r0, .[[LAB:L.*]]-2	# encoding: [0xc6,0x05,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC32DBL
	chrl	%r0, -2
#CHECK: chrl	%r0, .[[LAB:L.*]]	# encoding: [0xc6,0x05,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC32DBL
	chrl	%r0, 0
#CHECK: chrl	%r0, .[[LAB:L.*]]+4294967294 # encoding: [0xc6,0x05,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+4294967294)+2, kind: FK_390_PC32DBL
	chrl	%r0, 0xfffffffe

#CHECK: chrl	%r0, foo                # encoding: [0xc6,0x05,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: chrl	%r15, foo               # encoding: [0xc6,0xf5,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL

	chrl	%r0,foo
	chrl	%r15,foo

#CHECK: chrl	%r3, bar+100            # encoding: [0xc6,0x35,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: chrl	%r4, bar+100            # encoding: [0xc6,0x45,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL

	chrl	%r3,bar+100
	chrl	%r4,bar+100

#CHECK: chrl	%r7, frob@PLT           # encoding: [0xc6,0x75,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL
#CHECK: chrl	%r8, frob@PLT           # encoding: [0xc6,0x85,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL

	chrl	%r7,frob@PLT
	chrl	%r8,frob@PLT

#CHECK: chsi	0, 0                    # encoding: [0xe5,0x5c,0x00,0x00,0x00,0x00]
#CHECK: chsi	4095, 0                 # encoding: [0xe5,0x5c,0x0f,0xff,0x00,0x00]
#CHECK: chsi	0, -32768               # encoding: [0xe5,0x5c,0x00,0x00,0x80,0x00]
#CHECK: chsi	0, -1                   # encoding: [0xe5,0x5c,0x00,0x00,0xff,0xff]
#CHECK: chsi	0, 0                    # encoding: [0xe5,0x5c,0x00,0x00,0x00,0x00]
#CHECK: chsi	0, 1                    # encoding: [0xe5,0x5c,0x00,0x00,0x00,0x01]
#CHECK: chsi	0, 32767                # encoding: [0xe5,0x5c,0x00,0x00,0x7f,0xff]
#CHECK: chsi	0(%r1), 42              # encoding: [0xe5,0x5c,0x10,0x00,0x00,0x2a]
#CHECK: chsi	0(%r15), 42             # encoding: [0xe5,0x5c,0xf0,0x00,0x00,0x2a]
#CHECK: chsi	4095(%r1), 42           # encoding: [0xe5,0x5c,0x1f,0xff,0x00,0x2a]
#CHECK: chsi	4095(%r15), 42          # encoding: [0xe5,0x5c,0xff,0xff,0x00,0x2a]

	chsi	0, 0
	chsi	4095, 0
	chsi	0, -32768
	chsi	0, -1
	chsi	0, 0
	chsi	0, 1
	chsi	0, 32767
	chsi	0(%r1), 42
	chsi	0(%r15), 42
	chsi	4095(%r1), 42
	chsi	4095(%r15), 42

#CHECK: chy	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x79]
#CHECK: chy	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x79]
#CHECK: chy	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x79]
#CHECK: chy	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x79]
#CHECK: chy	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x79]
#CHECK: chy	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x79]
#CHECK: chy	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x79]
#CHECK: chy	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x79]
#CHECK: chy	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x79]
#CHECK: chy	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x79]

	chy	%r0, -524288
	chy	%r0, -1
	chy	%r0, 0
	chy	%r0, 1
	chy	%r0, 524287
	chy	%r0, 0(%r1)
	chy	%r0, 0(%r15)
	chy	%r0, 524287(%r1,%r15)
	chy	%r0, 524287(%r15,%r1)
	chy	%r15, 0

#CHECK: cib	%r0, 0, 0, 0            # encoding: [0xec,0x00,0x00,0x00,0x00,0xfe]
#CHECK: cib	%r0, -128, 0, 0         # encoding: [0xec,0x00,0x00,0x00,0x80,0xfe]
#CHECK: cib	%r0, 127, 0, 0          # encoding: [0xec,0x00,0x00,0x00,0x7f,0xfe]
#CHECK: cib	%r15, 0, 0, 0           # encoding: [0xec,0xf0,0x00,0x00,0x00,0xfe]
#CHECK: cib	%r7, -1, 0, 0           # encoding: [0xec,0x70,0x00,0x00,0xff,0xfe]
#CHECK: cib	%r0, 0, 1, 0            # encoding: [0xec,0x01,0x00,0x00,0x00,0xfe]
#CHECK: cib	%r0, 0, 15, 0           # encoding: [0xec,0x0f,0x00,0x00,0x00,0xfe]
#CHECK: cib	%r0, 0, 0, 0(%r13)      # encoding: [0xec,0x00,0xd0,0x00,0x00,0xfe]
#CHECK: cib	%r0, 0, 0, 4095         # encoding: [0xec,0x00,0x0f,0xff,0x00,0xfe]
#CHECK: cib	%r0, 0, 0, 4095(%r7)    # encoding: [0xec,0x00,0x7f,0xff,0x00,0xfe]
	cib	%r0, 0, 0, 0
	cib	%r0, -128, 0, 0
	cib	%r0, 127, 0, 0
	cib	%r15, 0, 0, 0
	cib	%r7, -1, 0, 0
	cib	%r0, 0, 1, 0
	cib	%r0, 0, 15, 0
	cib	%r0, 0, 0, 0(%r13)
	cib	%r0, 0, 0, 4095
	cib	%r0, 0, 0, 4095(%r7)

#CHECK: cibe	%r0, 0, 0               # encoding: [0xec,0x08,0x00,0x00,0x00,0xfe]
#CHECK: cibe	%r0, -128, 0            # encoding: [0xec,0x08,0x00,0x00,0x80,0xfe]
#CHECK: cibe	%r0, 127, 0             # encoding: [0xec,0x08,0x00,0x00,0x7f,0xfe]
#CHECK: cibe	%r15, 0, 0              # encoding: [0xec,0xf8,0x00,0x00,0x00,0xfe]
#CHECK: cibe	%r7, -1, 0              # encoding: [0xec,0x78,0x00,0x00,0xff,0xfe]
#CHECK: cibe	%r0, 0, 0(%r13)         # encoding: [0xec,0x08,0xd0,0x00,0x00,0xfe]
#CHECK: cibe	%r0, 0, 4095            # encoding: [0xec,0x08,0x0f,0xff,0x00,0xfe]
#CHECK: cibe	%r0, 0, 4095(%r7)       # encoding: [0xec,0x08,0x7f,0xff,0x00,0xfe]
	cibe	%r0, 0, 0
	cibe	%r0, -128, 0
	cibe	%r0, 127, 0
	cibe	%r15, 0, 0
	cibe	%r7, -1, 0
	cibe	%r0, 0, 0(%r13)
	cibe	%r0, 0, 4095
	cibe	%r0, 0, 4095(%r7)

#CHECK: cib	%r1, 2, 2, 3(%r4)       # encoding: [0xec,0x12,0x40,0x03,0x02,0xfe]
#CHECK: cibh	%r1, 2, 3(%r4)          # encoding: [0xec,0x12,0x40,0x03,0x02,0xfe]
#CHECK: cibnle	%r1, 2, 3(%r4)          # encoding: [0xec,0x12,0x40,0x03,0x02,0xfe]
	cib	%r1, 2, 2, 3(%r4)
	cibh	%r1, 2, 3(%r4)
	cibnle	%r1, 2, 3(%r4)

#CHECK: cib	%r1, 2, 4, 3(%r4)       # encoding: [0xec,0x14,0x40,0x03,0x02,0xfe]
#CHECK: cibl	%r1, 2, 3(%r4)          # encoding: [0xec,0x14,0x40,0x03,0x02,0xfe]
#CHECK: cibnhe	%r1, 2, 3(%r4)          # encoding: [0xec,0x14,0x40,0x03,0x02,0xfe]
	cib	%r1, 2, 4, 3(%r4)
	cibl	%r1, 2, 3(%r4)
	cibnhe	%r1, 2, 3(%r4)

#CHECK: cib	%r1, 2, 6, 3(%r4)       # encoding: [0xec,0x16,0x40,0x03,0x02,0xfe]
#CHECK: ciblh	%r1, 2, 3(%r4)          # encoding: [0xec,0x16,0x40,0x03,0x02,0xfe]
#CHECK: cibne	%r1, 2, 3(%r4)          # encoding: [0xec,0x16,0x40,0x03,0x02,0xfe]
	cib	%r1, 2, 6, 3(%r4)
	ciblh	%r1, 2, 3(%r4)
	cibne	%r1, 2, 3(%r4)

#CHECK: cib	%r1, 2, 8, 3(%r4)       # encoding: [0xec,0x18,0x40,0x03,0x02,0xfe]
#CHECK: cibe	%r1, 2, 3(%r4)          # encoding: [0xec,0x18,0x40,0x03,0x02,0xfe]
#CHECK: cibnlh	%r1, 2, 3(%r4)          # encoding: [0xec,0x18,0x40,0x03,0x02,0xfe]
	cib	%r1, 2, 8, 3(%r4)
	cibe	%r1, 2, 3(%r4)
	cibnlh	%r1, 2, 3(%r4)

#CHECK: cib	%r1, 2, 10, 3(%r4)      # encoding: [0xec,0x1a,0x40,0x03,0x02,0xfe]
#CHECK: cibhe	%r1, 2, 3(%r4)          # encoding: [0xec,0x1a,0x40,0x03,0x02,0xfe]
#CHECK: cibnl	%r1, 2, 3(%r4)          # encoding: [0xec,0x1a,0x40,0x03,0x02,0xfe]
	cib	%r1, 2, 10, 3(%r4)
	cibhe	%r1, 2, 3(%r4)
	cibnl	%r1, 2, 3(%r4)

#CHECK: cib	%r1, 2, 12, 3(%r4)      # encoding: [0xec,0x1c,0x40,0x03,0x02,0xfe]
#CHECK: cible	%r1, 2, 3(%r4)          # encoding: [0xec,0x1c,0x40,0x03,0x02,0xfe]
#CHECK: cibnh	%r1, 2, 3(%r4)          # encoding: [0xec,0x1c,0x40,0x03,0x02,0xfe]
	cib	%r1, 2, 12, 3(%r4)
	cible	%r1, 2, 3(%r4)
	cibnh	%r1, 2, 3(%r4)

#CHECK: cij	%r0, 0, 0, .[[LAB:L.*]]	# encoding: [0xec,0x00,A,A,0x00,0x7e]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC16DBL
#CHECK: cij	%r0, -128, 0, .[[LAB:L.*]]	# encoding: [0xec,0x00,A,A,0x80,0x7e]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC16DBL
#CHECK: cij	%r0, 127, 0, .[[LAB:L.*]]	# encoding: [0xec,0x00,A,A,0x7f,0x7e]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC16DBL
#CHECK: cij	%r15, 0, 0, .[[LAB:L.*]]	# encoding: [0xec,0xf0,A,A,0x00,0x7e]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC16DBL
#CHECK: cij	%r7, -1, 0, .[[LAB:L.*]]	# encoding: [0xec,0x70,A,A,0xff,0x7e]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC16DBL
	cij	%r0, 0, 0, 0
	cij	%r0, -128, 0, 0
	cij	%r0, 127, 0, 0
	cij	%r15, 0, 0, 0
	cij	%r7, -1, 0, 0

#CHECK: cij	%r1, -66, 0, .[[LAB:L.*]]-65536	# encoding: [0xec,0x10,A,A,0xbe,0x7e]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-65536)+2, kind: FK_390_PC16DBL
	cij	%r1, -66, 0, -0x10000
#CHECK: cij	%r1, -66, 0, .[[LAB:L.*]]-2	# encoding: [0xec,0x10,A,A,0xbe,0x7e]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC16DBL
	cij	%r1, -66, 0, -2
#CHECK: cij	%r1, -66, 0, .[[LAB:L.*]]		# encoding: [0xec,0x10,A,A,0xbe,0x7e]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC16DBL
	cij	%r1, -66, 0, 0
#CHECK: cij	%r1, -66, 0, .[[LAB:L.*]]+65534	# encoding: [0xec,0x10,A,A,0xbe,0x7e]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+65534)+2, kind: FK_390_PC16DBL
	cij	%r1, -66, 0, 0xfffe

#CHECK: cij	%r1, -66, 0, foo                  # encoding: [0xec,0x10,A,A,0xbe,0x7e]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	cij	%r1, -66, 0, foo

#CHECK: cij	%r1, -66, 1, foo                  # encoding: [0xec,0x11,A,A,0xbe,0x7e]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	cij	%r1, -66, 1, foo

#CHECK: cij	%r1, -66, 2, foo                  # encoding: [0xec,0x12,A,A,0xbe,0x7e]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: cijh	%r1, -66, foo                     # encoding: [0xec,0x12,A,A,0xbe,0x7e]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: cijnle	%r1, -66, foo                     # encoding: [0xec,0x12,A,A,0xbe,0x7e]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	cij	%r1, -66, 2, foo
	cijh	%r1, -66, foo
	cijnle	%r1, -66, foo

#CHECK: cij	%r1, -66, 3, foo                  # encoding: [0xec,0x13,A,A,0xbe,0x7e]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	cij	%r1, -66, 3, foo

#CHECK: cij	%r1, -66, 4, foo                  # encoding: [0xec,0x14,A,A,0xbe,0x7e]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: cijl	%r1, -66, foo                     # encoding: [0xec,0x14,A,A,0xbe,0x7e]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: cijnhe	%r1, -66, foo                     # encoding: [0xec,0x14,A,A,0xbe,0x7e]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	cij	%r1, -66, 4, foo
	cijl	%r1, -66, foo
	cijnhe	%r1, -66, foo

#CHECK: cij	%r1, -66, 5, foo                  # encoding: [0xec,0x15,A,A,0xbe,0x7e]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	cij	%r1, -66, 5, foo

#CHECK: cij	%r1, -66, 6, foo                  # encoding: [0xec,0x16,A,A,0xbe,0x7e]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: cijlh	%r1, -66, foo                     # encoding: [0xec,0x16,A,A,0xbe,0x7e]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: cijne	%r1, -66, foo                     # encoding: [0xec,0x16,A,A,0xbe,0x7e]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	cij	%r1, -66, 6, foo
	cijlh	%r1, -66, foo
	cijne	%r1, -66, foo

#CHECK: cij	%r1, -66, 7, foo                  # encoding: [0xec,0x17,A,A,0xbe,0x7e]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	cij	%r1, -66, 7, foo

#CHECK: cij	%r1, -66, 8, foo                  # encoding: [0xec,0x18,A,A,0xbe,0x7e]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: cije	%r1, -66, foo                     # encoding: [0xec,0x18,A,A,0xbe,0x7e]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: cijnlh	%r1, -66, foo                     # encoding: [0xec,0x18,A,A,0xbe,0x7e]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	cij	%r1, -66, 8, foo
	cije	%r1, -66, foo
	cijnlh	%r1, -66, foo

#CHECK: cij	%r1, -66, 9, foo                  # encoding: [0xec,0x19,A,A,0xbe,0x7e]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	cij	%r1, -66, 9, foo

#CHECK: cij	%r1, -66, 10, foo                 # encoding: [0xec,0x1a,A,A,0xbe,0x7e]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: cijhe	%r1, -66, foo                     # encoding: [0xec,0x1a,A,A,0xbe,0x7e]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: cijnl	%r1, -66, foo                     # encoding: [0xec,0x1a,A,A,0xbe,0x7e]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	cij	%r1, -66, 10, foo
	cijhe	%r1, -66, foo
	cijnl	%r1, -66, foo

#CHECK: cij	%r1, -66, 11, foo                 # encoding: [0xec,0x1b,A,A,0xbe,0x7e]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	cij	%r1, -66, 11, foo

#CHECK: cij	%r1, -66, 12, foo                 # encoding: [0xec,0x1c,A,A,0xbe,0x7e]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: cijle	%r1, -66, foo                     # encoding: [0xec,0x1c,A,A,0xbe,0x7e]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: cijnh	%r1, -66, foo                     # encoding: [0xec,0x1c,A,A,0xbe,0x7e]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	cij	%r1, -66, 12, foo
	cijle	%r1, -66, foo
	cijnh	%r1, -66, foo

#CHECK: cij	%r1, -66, 13, foo                 # encoding: [0xec,0x1d,A,A,0xbe,0x7e]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	cij	%r1, -66, 13, foo

#CHECK: cij	%r1, -66, 14, foo                 # encoding: [0xec,0x1e,A,A,0xbe,0x7e]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	cij	%r1, -66, 14, foo

#CHECK: cij	%r1, -66, 15, foo                 # encoding: [0xec,0x1f,A,A,0xbe,0x7e]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	cij	%r1, -66, 15, foo

#CHECK: cij	%r1, -66, 0, bar+100              # encoding: [0xec,0x10,A,A,0xbe,0x7e]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	cij	%r1, -66, 0, bar+100

#CHECK: cijh	%r1, -66, bar+100                 # encoding: [0xec,0x12,A,A,0xbe,0x7e]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	cijh	%r1, -66, bar+100

#CHECK: cijnle	%r1, -66, bar+100                 # encoding: [0xec,0x12,A,A,0xbe,0x7e]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	cijnle	%r1, -66, bar+100

#CHECK: cijl	%r1, -66, bar+100                 # encoding: [0xec,0x14,A,A,0xbe,0x7e]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	cijl	%r1, -66, bar+100

#CHECK: cijnhe	%r1, -66, bar+100                 # encoding: [0xec,0x14,A,A,0xbe,0x7e]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	cijnhe	%r1, -66, bar+100

#CHECK: cijlh	%r1, -66, bar+100                 # encoding: [0xec,0x16,A,A,0xbe,0x7e]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	cijlh	%r1, -66, bar+100

#CHECK: cijne	%r1, -66, bar+100                 # encoding: [0xec,0x16,A,A,0xbe,0x7e]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	cijne	%r1, -66, bar+100

#CHECK: cije	%r1, -66, bar+100                 # encoding: [0xec,0x18,A,A,0xbe,0x7e]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	cije	%r1, -66, bar+100

#CHECK: cijnlh	%r1, -66, bar+100                 # encoding: [0xec,0x18,A,A,0xbe,0x7e]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	cijnlh	%r1, -66, bar+100

#CHECK: cijhe	%r1, -66, bar+100                 # encoding: [0xec,0x1a,A,A,0xbe,0x7e]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	cijhe	%r1, -66, bar+100

#CHECK: cijnl	%r1, -66, bar+100                 # encoding: [0xec,0x1a,A,A,0xbe,0x7e]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	cijnl	%r1, -66, bar+100

#CHECK: cijle	%r1, -66, bar+100                 # encoding: [0xec,0x1c,A,A,0xbe,0x7e]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	cijle	%r1, -66, bar+100

#CHECK: cijnh	%r1, -66, bar+100                 # encoding: [0xec,0x1c,A,A,0xbe,0x7e]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	cijnh	%r1, -66, bar+100

#CHECK: cij	%r1, -66, 0, bar@PLT              # encoding: [0xec,0x10,A,A,0xbe,0x7e]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	cij	%r1, -66, 0, bar@PLT

#CHECK: cijh	%r1, -66, bar@PLT                 # encoding: [0xec,0x12,A,A,0xbe,0x7e]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	cijh	%r1, -66, bar@PLT

#CHECK: cijnle	%r1, -66, bar@PLT                 # encoding: [0xec,0x12,A,A,0xbe,0x7e]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	cijnle	%r1, -66, bar@PLT

#CHECK: cijl	%r1, -66, bar@PLT                 # encoding: [0xec,0x14,A,A,0xbe,0x7e]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	cijl	%r1, -66, bar@PLT

#CHECK: cijnhe	%r1, -66, bar@PLT                 # encoding: [0xec,0x14,A,A,0xbe,0x7e]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	cijnhe	%r1, -66, bar@PLT

#CHECK: cijlh	%r1, -66, bar@PLT                 # encoding: [0xec,0x16,A,A,0xbe,0x7e]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	cijlh	%r1, -66, bar@PLT

#CHECK: cijne	%r1, -66, bar@PLT                 # encoding: [0xec,0x16,A,A,0xbe,0x7e]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	cijne	%r1, -66, bar@PLT

#CHECK: cije	%r1, -66, bar@PLT                 # encoding: [0xec,0x18,A,A,0xbe,0x7e]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	cije	%r1, -66, bar@PLT

#CHECK: cijnlh	%r1, -66, bar@PLT                 # encoding: [0xec,0x18,A,A,0xbe,0x7e]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	cijnlh	%r1, -66, bar@PLT

#CHECK: cijhe	%r1, -66, bar@PLT                 # encoding: [0xec,0x1a,A,A,0xbe,0x7e]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	cijhe	%r1, -66, bar@PLT

#CHECK: cijnl	%r1, -66, bar@PLT                 # encoding: [0xec,0x1a,A,A,0xbe,0x7e]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	cijnl	%r1, -66, bar@PLT

#CHECK: cijle	%r1, -66, bar@PLT                 # encoding: [0xec,0x1c,A,A,0xbe,0x7e]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	cijle	%r1, -66, bar@PLT

#CHECK: cijnh	%r1, -66, bar@PLT                 # encoding: [0xec,0x1c,A,A,0xbe,0x7e]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	cijnh	%r1, -66, bar@PLT

#CHECK: cit     %r0, 0, 12              # encoding: [0xec,0x00,0x00,0x00,0xc0,0x72]
#CHECK: cit     %r0, -1, 12             # encoding: [0xec,0x00,0xff,0xff,0xc0,0x72]
#CHECK: cit     %r0, -32768, 12         # encoding: [0xec,0x00,0x80,0x00,0xc0,0x72]
#CHECK: cit     %r0, 32767, 12          # encoding: [0xec,0x00,0x7f,0xff,0xc0,0x72]
#CHECK: cith    %r15, 1                 # encoding: [0xec,0xf0,0x00,0x01,0x20,0x72]
#CHECK: citl    %r15, 1                 # encoding: [0xec,0xf0,0x00,0x01,0x40,0x72]
#CHECK: cite    %r15, 1                 # encoding: [0xec,0xf0,0x00,0x01,0x80,0x72]
#CHECK: citne   %r15, 1                 # encoding: [0xec,0xf0,0x00,0x01,0x60,0x72]
#CHECK: citnl   %r15, 1                 # encoding: [0xec,0xf0,0x00,0x01,0xa0,0x72]
#CHECK: citnh   %r15, 1                 # encoding: [0xec,0xf0,0x00,0x01,0xc0,0x72]
#CHECK: citnle  %r15, 1                 # encoding: [0xec,0xf0,0x00,0x01,0x20,0x72]
#CHECK: citnhe  %r15, 1                 # encoding: [0xec,0xf0,0x00,0x01,0x40,0x72]
#CHECK: citnlh  %r15, 1                 # encoding: [0xec,0xf0,0x00,0x01,0x80,0x72]
#CHECK: citlh   %r15, 1                 # encoding: [0xec,0xf0,0x00,0x01,0x60,0x72]
#CHECK: cithe   %r15, 1                 # encoding: [0xec,0xf0,0x00,0x01,0xa0,0x72]
#CHECK: citle   %r15, 1                 # encoding: [0xec,0xf0,0x00,0x01,0xc0,0x72]

        cit     %r0, 0, 12
        cit     %r0, -1, 12
        cit     %r0, -32768, 12
        cit     %r0, 32767, 12
        cith    %r15, 1
        citl    %r15, 1
        cite    %r15, 1
        citne   %r15, 1
        citnl   %r15, 1
        citnh   %r15, 1
        citnle  %r15, 1
        citnhe  %r15, 1
        citnlh  %r15, 1
        citlh   %r15, 1
        cithe   %r15, 1
        citle   %r15, 1

#CHECK: cksm	%r0, %r8                # encoding: [0xb2,0x41,0x00,0x08]
#CHECK: cksm	%r0, %r14               # encoding: [0xb2,0x41,0x00,0x0e]
#CHECK: cksm	%r15, %r0               # encoding: [0xb2,0x41,0x00,0xf0]
#CHECK: cksm	%r15, %r8               # encoding: [0xb2,0x41,0x00,0xf8]

	cksm	%r0, %r8
	cksm	%r0, %r14
	cksm	%r15, %r0
	cksm	%r15, %r8

#CHECK: cl	%r0, 0                  # encoding: [0x55,0x00,0x00,0x00]
#CHECK: cl	%r0, 4095               # encoding: [0x55,0x00,0x0f,0xff]
#CHECK: cl	%r0, 0(%r1)             # encoding: [0x55,0x00,0x10,0x00]
#CHECK: cl	%r0, 0(%r15)            # encoding: [0x55,0x00,0xf0,0x00]
#CHECK: cl	%r0, 4095(%r1,%r15)     # encoding: [0x55,0x01,0xff,0xff]
#CHECK: cl	%r0, 4095(%r15,%r1)     # encoding: [0x55,0x0f,0x1f,0xff]
#CHECK: cl	%r15, 0                 # encoding: [0x55,0xf0,0x00,0x00]

	cl	%r0, 0
	cl	%r0, 4095
	cl	%r0, 0(%r1)
	cl	%r0, 0(%r15)
	cl	%r0, 4095(%r1,%r15)
	cl	%r0, 4095(%r15,%r1)
	cl	%r15, 0

#CHECK: clc	0(1), 0                 # encoding: [0xd5,0x00,0x00,0x00,0x00,0x00]
#CHECK: clc	0(1), 0(%r1)            # encoding: [0xd5,0x00,0x00,0x00,0x10,0x00]
#CHECK: clc	0(1), 0(%r15)           # encoding: [0xd5,0x00,0x00,0x00,0xf0,0x00]
#CHECK: clc	0(1), 4095              # encoding: [0xd5,0x00,0x00,0x00,0x0f,0xff]
#CHECK: clc	0(1), 4095(%r1)         # encoding: [0xd5,0x00,0x00,0x00,0x1f,0xff]
#CHECK: clc	0(1), 4095(%r15)        # encoding: [0xd5,0x00,0x00,0x00,0xff,0xff]
#CHECK: clc	0(1,%r1), 0             # encoding: [0xd5,0x00,0x10,0x00,0x00,0x00]
#CHECK: clc	0(1,%r15), 0            # encoding: [0xd5,0x00,0xf0,0x00,0x00,0x00]
#CHECK: clc	4095(1,%r1), 0          # encoding: [0xd5,0x00,0x1f,0xff,0x00,0x00]
#CHECK: clc	4095(1,%r15), 0         # encoding: [0xd5,0x00,0xff,0xff,0x00,0x00]
#CHECK: clc	0(256,%r1), 0           # encoding: [0xd5,0xff,0x10,0x00,0x00,0x00]
#CHECK: clc	0(256,%r15), 0          # encoding: [0xd5,0xff,0xf0,0x00,0x00,0x00]

	clc	0(1), 0
	clc	0(1), 0(%r1)
	clc	0(1), 0(%r15)
	clc	0(1), 4095
	clc	0(1), 4095(%r1)
	clc	0(1), 4095(%r15)
	clc	0(1,%r1), 0
	clc	0(1,%r15), 0
	clc	4095(1,%r1), 0
	clc	4095(1,%r15), 0
	clc	0(256,%r1), 0
	clc	0(256,%r15), 0

#CHECK: clcl	%r0, %r8                # encoding: [0x0f,0x08]
#CHECK: clcl	%r0, %r14               # encoding: [0x0f,0x0e]
#CHECK: clcl	%r14, %r0               # encoding: [0x0f,0xe0]
#CHECK: clcl	%r14, %r8               # encoding: [0x0f,0xe8]

	clcl	%r0, %r8
	clcl	%r0, %r14
	clcl	%r14, %r0
	clcl	%r14, %r8

#CHECK: clcle	%r0, %r0, 0             # encoding: [0xa9,0x00,0x00,0x00]
#CHECK: clcle	%r0, %r14, 4095         # encoding: [0xa9,0x0e,0x0f,0xff]
#CHECK: clcle	%r0, %r0, 0(%r1)        # encoding: [0xa9,0x00,0x10,0x00]
#CHECK: clcle	%r0, %r0, 0(%r15)       # encoding: [0xa9,0x00,0xf0,0x00]
#CHECK: clcle	%r14, %r14, 4095(%r1)   # encoding: [0xa9,0xee,0x1f,0xff]
#CHECK: clcle	%r0, %r0, 4095(%r15)    # encoding: [0xa9,0x00,0xff,0xff]
#CHECK: clcle	%r14, %r0, 0            # encoding: [0xa9,0xe0,0x00,0x00]

	clcle	%r0, %r0, 0
	clcle	%r0, %r14, 4095
	clcle	%r0, %r0, 0(%r1)
	clcle	%r0, %r0, 0(%r15)
	clcle	%r14, %r14, 4095(%r1)
	clcle	%r0, %r0, 4095(%r15)
	clcle	%r14, %r0, 0

#CHECK: clclu	%r0, %r0, -524288       # encoding: [0xeb,0x00,0x00,0x00,0x80,0x8f]
#CHECK: clclu	%r0, %r0, -1            # encoding: [0xeb,0x00,0x0f,0xff,0xff,0x8f]
#CHECK: clclu	%r0, %r14, 0            # encoding: [0xeb,0x0e,0x00,0x00,0x00,0x8f]
#CHECK: clclu	%r0, %r14, 1            # encoding: [0xeb,0x0e,0x00,0x01,0x00,0x8f]
#CHECK: clclu	%r0, %r8, 524287        # encoding: [0xeb,0x08,0x0f,0xff,0x7f,0x8f]
#CHECK: clclu	%r0, %r8, 0(%r1)        # encoding: [0xeb,0x08,0x10,0x00,0x00,0x8f]
#CHECK: clclu	%r0, %r4, 0(%r15)       # encoding: [0xeb,0x04,0xf0,0x00,0x00,0x8f]
#CHECK: clclu	%r0, %r4, 524287(%r15)  # encoding: [0xeb,0x04,0xff,0xff,0x7f,0x8f]
#CHECK: clclu	%r0, %r0, 524287(%r1)   # encoding: [0xeb,0x00,0x1f,0xff,0x7f,0x8f]
#CHECK: clclu	%r14, %r0, 0            # encoding: [0xeb,0xe0,0x00,0x00,0x00,0x8f]

	clclu	%r0, %r0, -524288
	clclu	%r0, %r0, -1
	clclu	%r0, %r14, 0
	clclu	%r0, %r14, 1
	clclu	%r0, %r8, 524287
	clclu	%r0, %r8, 0(%r1)
	clclu	%r0, %r4, 0(%r15)
	clclu	%r0, %r4, 524287(%r15)
	clclu	%r0, %r0, 524287(%r1)
	clclu	%r14, %r0, 0

#CHECK: clfhsi	0, 0                    # encoding: [0xe5,0x5d,0x00,0x00,0x00,0x00]
#CHECK: clfhsi	4095, 0                 # encoding: [0xe5,0x5d,0x0f,0xff,0x00,0x00]
#CHECK: clfhsi	0, 65535                # encoding: [0xe5,0x5d,0x00,0x00,0xff,0xff]
#CHECK: clfhsi	0(%r1), 42              # encoding: [0xe5,0x5d,0x10,0x00,0x00,0x2a]
#CHECK: clfhsi	0(%r15), 42             # encoding: [0xe5,0x5d,0xf0,0x00,0x00,0x2a]
#CHECK: clfhsi	4095(%r1), 42           # encoding: [0xe5,0x5d,0x1f,0xff,0x00,0x2a]
#CHECK: clfhsi	4095(%r15), 42          # encoding: [0xe5,0x5d,0xff,0xff,0x00,0x2a]

	clfhsi	0, 0
	clfhsi	4095, 0
	clfhsi	0, 65535
	clfhsi	0(%r1), 42
	clfhsi	0(%r15), 42
	clfhsi	4095(%r1), 42
	clfhsi	4095(%r15), 42

#CHECK: clfi	%r0, 0                  # encoding: [0xc2,0x0f,0x00,0x00,0x00,0x00]
#CHECK: clfi	%r0, 4294967295         # encoding: [0xc2,0x0f,0xff,0xff,0xff,0xff]
#CHECK: clfi	%r15, 0                 # encoding: [0xc2,0xff,0x00,0x00,0x00,0x00]

	clfi	%r0, 0
	clfi	%r0, (1 << 32) - 1
	clfi	%r15, 0

#CHECK: clfit     %r0, 0, 12             # encoding: [0xec,0x00,0x00,0x00,0xc0,0x73]
#CHECK: clfit     %r0, 65535, 12         # encoding: [0xec,0x00,0xff,0xff,0xc0,0x73]
#CHECK: clfit     %r0, 32768, 12         # encoding: [0xec,0x00,0x80,0x00,0xc0,0x73]
#CHECK: clfith    %r15, 1                # encoding: [0xec,0xf0,0x00,0x01,0x20,0x73]
#CHECK: clfitl    %r15, 1                # encoding: [0xec,0xf0,0x00,0x01,0x40,0x73]
#CHECK: clfite    %r15, 1                # encoding: [0xec,0xf0,0x00,0x01,0x80,0x73]
#CHECK: clfitne   %r15, 1                # encoding: [0xec,0xf0,0x00,0x01,0x60,0x73]
#CHECK: clfitnl   %r15, 1                # encoding: [0xec,0xf0,0x00,0x01,0xa0,0x73]
#CHECK: clfitnh   %r15, 1                # encoding: [0xec,0xf0,0x00,0x01,0xc0,0x73]
#CHECK: clfitnle  %r15, 1                # encoding: [0xec,0xf0,0x00,0x01,0x20,0x73]
#CHECK: clfitnhe  %r15, 1                # encoding: [0xec,0xf0,0x00,0x01,0x40,0x73]
#CHECK: clfitnlh  %r15, 1                # encoding: [0xec,0xf0,0x00,0x01,0x80,0x73]
#CHECK: clfitlh   %r15, 1                # encoding: [0xec,0xf0,0x00,0x01,0x60,0x73]
#CHECK: clfithe   %r15, 1                # encoding: [0xec,0xf0,0x00,0x01,0xa0,0x73]
#CHECK: clfitle   %r15, 1                # encoding: [0xec,0xf0,0x00,0x01,0xc0,0x73]

        clfit     %r0, 0, 12
        clfit     %r0, 65535, 12
        clfit     %r0, 32768, 12
        clfith    %r15, 1
        clfitl    %r15, 1
        clfite    %r15, 1
        clfitne   %r15, 1
        clfitnl   %r15, 1
        clfitnh   %r15, 1
        clfitnle  %r15, 1
        clfitnhe  %r15, 1
        clfitnlh  %r15, 1
        clfitlh   %r15, 1
        clfithe   %r15, 1
        clfitle   %r15, 1

#CHECK: clg	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x21]
#CHECK: clg	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x21]
#CHECK: clg	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x21]
#CHECK: clg	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x21]
#CHECK: clg	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x21]
#CHECK: clg	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x21]
#CHECK: clg	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x21]
#CHECK: clg	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x21]
#CHECK: clg	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x21]
#CHECK: clg	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x21]

	clg	%r0, -524288
	clg	%r0, -1
	clg	%r0, 0
	clg	%r0, 1
	clg	%r0, 524287
	clg	%r0, 0(%r1)
	clg	%r0, 0(%r15)
	clg	%r0, 524287(%r1,%r15)
	clg	%r0, 524287(%r15,%r1)
	clg	%r15, 0

#CHECK: clgf	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x31]
#CHECK: clgf	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x31]
#CHECK: clgf	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x31]
#CHECK: clgf	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x31]
#CHECK: clgf	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x31]
#CHECK: clgf	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x31]
#CHECK: clgf	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x31]
#CHECK: clgf	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x31]
#CHECK: clgf	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x31]
#CHECK: clgf	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x31]

	clgf	%r0, -524288
	clgf	%r0, -1
	clgf	%r0, 0
	clgf	%r0, 1
	clgf	%r0, 524287
	clgf	%r0, 0(%r1)
	clgf	%r0, 0(%r15)
	clgf	%r0, 524287(%r1,%r15)
	clgf	%r0, 524287(%r15,%r1)
	clgf	%r15, 0

#CHECK: clgfi	%r0, 0                  # encoding: [0xc2,0x0e,0x00,0x00,0x00,0x00]
#CHECK: clgfi	%r0, 4294967295         # encoding: [0xc2,0x0e,0xff,0xff,0xff,0xff]
#CHECK: clgfi	%r15, 0                 # encoding: [0xc2,0xfe,0x00,0x00,0x00,0x00]

	clgfi	%r0, 0
	clgfi	%r0, (1 << 32) - 1
	clgfi	%r15, 0

#CHECK: clgfr	%r0, %r0                # encoding: [0xb9,0x31,0x00,0x00]
#CHECK: clgfr	%r0, %r15               # encoding: [0xb9,0x31,0x00,0x0f]
#CHECK: clgfr	%r15, %r0               # encoding: [0xb9,0x31,0x00,0xf0]
#CHECK: clgfr	%r7, %r8                # encoding: [0xb9,0x31,0x00,0x78]

	clgfr	%r0,%r0
	clgfr	%r0,%r15
	clgfr	%r15,%r0
	clgfr	%r7,%r8

#CHECK: clgfrl	%r0, .[[LAB:L.*]]-4294967296 # encoding: [0xc6,0x0e,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-4294967296)+2, kind: FK_390_PC32DBL
	clgfrl	%r0, -0x100000000
#CHECK: clgfrl	%r0, .[[LAB:L.*]]-2	# encoding: [0xc6,0x0e,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC32DBL
	clgfrl	%r0, -2
#CHECK: clgfrl	%r0, .[[LAB:L.*]]	# encoding: [0xc6,0x0e,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC32DBL
	clgfrl	%r0, 0
#CHECK: clgfrl	%r0, .[[LAB:L.*]]+4294967294 # encoding: [0xc6,0x0e,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+4294967294)+2, kind: FK_390_PC32DBL
	clgfrl	%r0, 0xfffffffe

#CHECK: clgfrl	%r0, foo                # encoding: [0xc6,0x0e,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: clgfrl	%r15, foo               # encoding: [0xc6,0xfe,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL

	clgfrl	%r0,foo
	clgfrl	%r15,foo

#CHECK: clgfrl	%r3, bar+100            # encoding: [0xc6,0x3e,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: clgfrl	%r4, bar+100            # encoding: [0xc6,0x4e,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL

	clgfrl	%r3,bar+100
	clgfrl	%r4,bar+100

#CHECK: clgfrl	%r7, frob@PLT           # encoding: [0xc6,0x7e,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL
#CHECK: clgfrl	%r8, frob@PLT           # encoding: [0xc6,0x8e,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL

	clgfrl	%r7,frob@PLT
	clgfrl	%r8,frob@PLT

#CHECK: clghrl	%r0, .[[LAB:L.*]]-4294967296 # encoding: [0xc6,0x06,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-4294967296)+2, kind: FK_390_PC32DBL
	clghrl	%r0, -0x100000000
#CHECK: clghrl	%r0, .[[LAB:L.*]]-2	# encoding: [0xc6,0x06,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC32DBL
	clghrl	%r0, -2
#CHECK: clghrl	%r0, .[[LAB:L.*]]	# encoding: [0xc6,0x06,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC32DBL
	clghrl	%r0, 0
#CHECK: clghrl	%r0, .[[LAB:L.*]]+4294967294 # encoding: [0xc6,0x06,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+4294967294)+2, kind: FK_390_PC32DBL
	clghrl	%r0, 0xfffffffe

#CHECK: clghrl	%r0, foo                # encoding: [0xc6,0x06,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: clghrl	%r15, foo               # encoding: [0xc6,0xf6,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL

	clghrl	%r0,foo
	clghrl	%r15,foo

#CHECK: clghrl	%r3, bar+100            # encoding: [0xc6,0x36,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: clghrl	%r4, bar+100            # encoding: [0xc6,0x46,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL

	clghrl	%r3,bar+100
	clghrl	%r4,bar+100

#CHECK: clghrl	%r7, frob@PLT           # encoding: [0xc6,0x76,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL
#CHECK: clghrl	%r8, frob@PLT           # encoding: [0xc6,0x86,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL

	clghrl	%r7,frob@PLT
	clghrl	%r8,frob@PLT

#CHECK: clghsi	0, 0                    # encoding: [0xe5,0x59,0x00,0x00,0x00,0x00]
#CHECK: clghsi	4095, 0                 # encoding: [0xe5,0x59,0x0f,0xff,0x00,0x00]
#CHECK: clghsi	0, 65535                # encoding: [0xe5,0x59,0x00,0x00,0xff,0xff]
#CHECK: clghsi	0(%r1), 42              # encoding: [0xe5,0x59,0x10,0x00,0x00,0x2a]
#CHECK: clghsi	0(%r15), 42             # encoding: [0xe5,0x59,0xf0,0x00,0x00,0x2a]
#CHECK: clghsi	4095(%r1), 42           # encoding: [0xe5,0x59,0x1f,0xff,0x00,0x2a]
#CHECK: clghsi	4095(%r15), 42          # encoding: [0xe5,0x59,0xff,0xff,0x00,0x2a]

	clghsi	0, 0
	clghsi	4095, 0
	clghsi	0, 65535
	clghsi	0(%r1), 42
	clghsi	0(%r15), 42
	clghsi	4095(%r1), 42
	clghsi	4095(%r15), 42

#CHECK: clgib	%r0, 0, 0, 0            # encoding: [0xec,0x00,0x00,0x00,0x00,0xfd]
#CHECK: clgib	%r0, 128, 0, 0          # encoding: [0xec,0x00,0x00,0x00,0x80,0xfd]
#CHECK: clgib	%r0, 127, 0, 0          # encoding: [0xec,0x00,0x00,0x00,0x7f,0xfd]
#CHECK: clgib	%r15, 0, 0, 0           # encoding: [0xec,0xf0,0x00,0x00,0x00,0xfd]
#CHECK: clgib	%r7, 255, 0, 0          # encoding: [0xec,0x70,0x00,0x00,0xff,0xfd]
#CHECK: clgib	%r0, 0, 1, 0            # encoding: [0xec,0x01,0x00,0x00,0x00,0xfd]
#CHECK: clgib	%r0, 0, 15, 0           # encoding: [0xec,0x0f,0x00,0x00,0x00,0xfd]
#CHECK: clgib	%r0, 0, 0, 0(%r13)      # encoding: [0xec,0x00,0xd0,0x00,0x00,0xfd]
#CHECK: clgib	%r0, 0, 0, 4095         # encoding: [0xec,0x00,0x0f,0xff,0x00,0xfd]
#CHECK: clgib	%r0, 0, 0, 4095(%r7)    # encoding: [0xec,0x00,0x7f,0xff,0x00,0xfd]
	clgib	%r0, 0, 0, 0
	clgib	%r0, 128, 0, 0
	clgib	%r0, 127, 0, 0
	clgib	%r15, 0, 0, 0
	clgib	%r7, 255, 0, 0
	clgib	%r0, 0, 1, 0
	clgib	%r0, 0, 15, 0
	clgib	%r0, 0, 0, 0(%r13)
	clgib	%r0, 0, 0, 4095
	clgib	%r0, 0, 0, 4095(%r7)

#CHECK: clgibe	%r0, 0, 0               # encoding: [0xec,0x08,0x00,0x00,0x00,0xfd]
#CHECK: clgibe	%r0, 128, 0             # encoding: [0xec,0x08,0x00,0x00,0x80,0xfd]
#CHECK: clgibe	%r0, 127, 0             # encoding: [0xec,0x08,0x00,0x00,0x7f,0xfd]
#CHECK: clgibe	%r15, 0, 0              # encoding: [0xec,0xf8,0x00,0x00,0x00,0xfd]
#CHECK: clgibe	%r7, 255, 0             # encoding: [0xec,0x78,0x00,0x00,0xff,0xfd]
#CHECK: clgibe	%r0, 0, 0(%r13)         # encoding: [0xec,0x08,0xd0,0x00,0x00,0xfd]
#CHECK: clgibe	%r0, 0, 4095            # encoding: [0xec,0x08,0x0f,0xff,0x00,0xfd]
#CHECK: clgibe	%r0, 0, 4095(%r7)       # encoding: [0xec,0x08,0x7f,0xff,0x00,0xfd]
	clgibe	%r0, 0, 0
	clgibe	%r0, 128, 0
	clgibe	%r0, 127, 0
	clgibe	%r15, 0, 0
	clgibe	%r7, 255, 0
	clgibe	%r0, 0, 0(%r13)
	clgibe	%r0, 0, 4095
	clgibe	%r0, 0, 4095(%r7)

#CHECK: clgib	%r1, 2, 2, 3(%r4)       # encoding: [0xec,0x12,0x40,0x03,0x02,0xfd]
#CHECK: clgibh	%r1, 2, 3(%r4)          # encoding: [0xec,0x12,0x40,0x03,0x02,0xfd]
#CHECK: clgibnle	%r1, 2, 3(%r4)  # encoding: [0xec,0x12,0x40,0x03,0x02,0xfd]
	clgib	%r1, 2, 2, 3(%r4)
	clgibh	%r1, 2, 3(%r4)
	clgibnle	%r1, 2, 3(%r4)

#CHECK: clgib	%r1, 2, 4, 3(%r4)       # encoding: [0xec,0x14,0x40,0x03,0x02,0xfd]
#CHECK: clgibl	%r1, 2, 3(%r4)          # encoding: [0xec,0x14,0x40,0x03,0x02,0xfd]
#CHECK: clgibnhe	%r1, 2, 3(%r4)  # encoding: [0xec,0x14,0x40,0x03,0x02,0xfd]
	clgib	%r1, 2, 4, 3(%r4)
	clgibl	%r1, 2, 3(%r4)
	clgibnhe	%r1, 2, 3(%r4)

#CHECK: clgib	%r1, 2, 6, 3(%r4)       # encoding: [0xec,0x16,0x40,0x03,0x02,0xfd]
#CHECK: clgiblh	%r1, 2, 3(%r4)          # encoding: [0xec,0x16,0x40,0x03,0x02,0xfd]
#CHECK: clgibne	%r1, 2, 3(%r4)          # encoding: [0xec,0x16,0x40,0x03,0x02,0xfd]
	clgib	%r1, 2, 6, 3(%r4)
	clgiblh	%r1, 2, 3(%r4)
	clgibne	%r1, 2, 3(%r4)

#CHECK: clgib	%r1, 2, 8, 3(%r4)       # encoding: [0xec,0x18,0x40,0x03,0x02,0xfd]
#CHECK: clgibe	%r1, 2, 3(%r4)          # encoding: [0xec,0x18,0x40,0x03,0x02,0xfd]
#CHECK: clgibnlh	%r1, 2, 3(%r4)  # encoding: [0xec,0x18,0x40,0x03,0x02,0xfd]
	clgib	%r1, 2, 8, 3(%r4)
	clgibe	%r1, 2, 3(%r4)
	clgibnlh	%r1, 2, 3(%r4)

#CHECK: clgib	%r1, 2, 10, 3(%r4)      # encoding: [0xec,0x1a,0x40,0x03,0x02,0xfd]
#CHECK: clgibhe	%r1, 2, 3(%r4)          # encoding: [0xec,0x1a,0x40,0x03,0x02,0xfd]
#CHECK: clgibnl	%r1, 2, 3(%r4)          # encoding: [0xec,0x1a,0x40,0x03,0x02,0xfd]
	clgib	%r1, 2, 10, 3(%r4)
	clgibhe	%r1, 2, 3(%r4)
	clgibnl	%r1, 2, 3(%r4)

#CHECK: clgib	%r1, 2, 12, 3(%r4)      # encoding: [0xec,0x1c,0x40,0x03,0x02,0xfd]
#CHECK: clgible	%r1, 2, 3(%r4)          # encoding: [0xec,0x1c,0x40,0x03,0x02,0xfd]
#CHECK: clgibnh	%r1, 2, 3(%r4)          # encoding: [0xec,0x1c,0x40,0x03,0x02,0xfd]
	clgib	%r1, 2, 12, 3(%r4)
	clgible	%r1, 2, 3(%r4)
	clgibnh	%r1, 2, 3(%r4)

#CHECK: clgij	%r0, 0, 0, .[[LAB:L.*]]	# encoding: [0xec,0x00,A,A,0x00,0x7d]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC16DBL
#CHECK: clgij	%r0, 255, 0, .[[LAB:L.*]]	# encoding: [0xec,0x00,A,A,0xff,0x7d]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC16DBL
#CHECK: clgij	%r15, 0, 0, .[[LAB:L.*]]	# encoding: [0xec,0xf0,A,A,0x00,0x7d]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC16DBL
	clgij	%r0, 0, 0, 0
	clgij	%r0, 255, 0, 0
	clgij	%r15, 0, 0, 0

#CHECK: clgij	%r1, 193, 0, .[[LAB:L.*]]-65536	# encoding: [0xec,0x10,A,A,0xc1,0x7d]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-65536)+2, kind: FK_390_PC16DBL
	clgij	%r1, 193, 0, -0x10000
#CHECK: clgij	%r1, 193, 0, .[[LAB:L.*]]-2	# encoding: [0xec,0x10,A,A,0xc1,0x7d]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC16DBL
	clgij	%r1, 193, 0, -2
#CHECK: clgij	%r1, 193, 0, .[[LAB:L.*]]		# encoding: [0xec,0x10,A,A,0xc1,0x7d]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC16DBL
	clgij	%r1, 193, 0, 0
#CHECK: clgij	%r1, 193, 0, .[[LAB:L.*]]+65534	# encoding: [0xec,0x10,A,A,0xc1,0x7d]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+65534)+2, kind: FK_390_PC16DBL
	clgij	%r1, 193, 0, 0xfffe

#CHECK: clgij	%r1, 193, 0, foo                  # encoding: [0xec,0x10,A,A,0xc1,0x7d]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clgij	%r1, 193, 0, foo

#CHECK: clgij	%r1, 193, 1, foo                  # encoding: [0xec,0x11,A,A,0xc1,0x7d]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clgij	%r1, 193, 1, foo

#CHECK: clgij	%r1, 193, 2, foo                  # encoding: [0xec,0x12,A,A,0xc1,0x7d]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: clgijh	%r1, 193, foo                     # encoding: [0xec,0x12,A,A,0xc1,0x7d]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: clgijnle	%r1, 193, foo                     # encoding: [0xec,0x12,A,A,0xc1,0x7d]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clgij	%r1, 193, 2, foo
	clgijh	%r1, 193, foo
	clgijnle	%r1, 193, foo

#CHECK: clgij	%r1, 193, 3, foo                  # encoding: [0xec,0x13,A,A,0xc1,0x7d]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clgij	%r1, 193, 3, foo

#CHECK: clgij	%r1, 193, 4, foo                  # encoding: [0xec,0x14,A,A,0xc1,0x7d]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: clgijl	%r1, 193, foo                     # encoding: [0xec,0x14,A,A,0xc1,0x7d]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: clgijnhe	%r1, 193, foo                     # encoding: [0xec,0x14,A,A,0xc1,0x7d]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clgij	%r1, 193, 4, foo
	clgijl	%r1, 193, foo
	clgijnhe	%r1, 193, foo

#CHECK: clgij	%r1, 193, 5, foo                  # encoding: [0xec,0x15,A,A,0xc1,0x7d]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clgij	%r1, 193, 5, foo

#CHECK: clgij	%r1, 193, 6, foo                  # encoding: [0xec,0x16,A,A,0xc1,0x7d]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: clgijlh	%r1, 193, foo                     # encoding: [0xec,0x16,A,A,0xc1,0x7d]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: clgijne	%r1, 193, foo                     # encoding: [0xec,0x16,A,A,0xc1,0x7d]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clgij	%r1, 193, 6, foo
	clgijlh	%r1, 193, foo
	clgijne	%r1, 193, foo

#CHECK: clgij	%r1, 193, 7, foo                  # encoding: [0xec,0x17,A,A,0xc1,0x7d]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clgij	%r1, 193, 7, foo

#CHECK: clgij	%r1, 193, 8, foo                  # encoding: [0xec,0x18,A,A,0xc1,0x7d]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: clgije	%r1, 193, foo                     # encoding: [0xec,0x18,A,A,0xc1,0x7d]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: clgijnlh	%r1, 193, foo                     # encoding: [0xec,0x18,A,A,0xc1,0x7d]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clgij	%r1, 193, 8, foo
	clgije	%r1, 193, foo
	clgijnlh	%r1, 193, foo

#CHECK: clgij	%r1, 193, 9, foo                  # encoding: [0xec,0x19,A,A,0xc1,0x7d]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clgij	%r1, 193, 9, foo

#CHECK: clgij	%r1, 193, 10, foo                 # encoding: [0xec,0x1a,A,A,0xc1,0x7d]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: clgijhe	%r1, 193, foo                     # encoding: [0xec,0x1a,A,A,0xc1,0x7d]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: clgijnl	%r1, 193, foo                     # encoding: [0xec,0x1a,A,A,0xc1,0x7d]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clgij	%r1, 193, 10, foo
	clgijhe	%r1, 193, foo
	clgijnl	%r1, 193, foo

#CHECK: clgij	%r1, 193, 11, foo                 # encoding: [0xec,0x1b,A,A,0xc1,0x7d]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clgij	%r1, 193, 11, foo

#CHECK: clgij	%r1, 193, 12, foo                 # encoding: [0xec,0x1c,A,A,0xc1,0x7d]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: clgijle	%r1, 193, foo                     # encoding: [0xec,0x1c,A,A,0xc1,0x7d]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: clgijnh	%r1, 193, foo                     # encoding: [0xec,0x1c,A,A,0xc1,0x7d]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clgij	%r1, 193, 12, foo
	clgijle	%r1, 193, foo
	clgijnh	%r1, 193, foo

#CHECK: clgij	%r1, 193, 13, foo                 # encoding: [0xec,0x1d,A,A,0xc1,0x7d]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clgij	%r1, 193, 13, foo

#CHECK: clgij	%r1, 193, 14, foo                 # encoding: [0xec,0x1e,A,A,0xc1,0x7d]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clgij	%r1, 193, 14, foo

#CHECK: clgij	%r1, 193, 15, foo                 # encoding: [0xec,0x1f,A,A,0xc1,0x7d]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clgij	%r1, 193, 15, foo

#CHECK: clgij	%r1, 193, 0, bar+100              # encoding: [0xec,0x10,A,A,0xc1,0x7d]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	clgij	%r1, 193, 0, bar+100

#CHECK: clgijh	%r1, 193, bar+100                 # encoding: [0xec,0x12,A,A,0xc1,0x7d]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	clgijh	%r1, 193, bar+100

#CHECK: clgijnle	%r1, 193, bar+100                 # encoding: [0xec,0x12,A,A,0xc1,0x7d]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	clgijnle	%r1, 193, bar+100

#CHECK: clgijl	%r1, 193, bar+100                 # encoding: [0xec,0x14,A,A,0xc1,0x7d]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	clgijl	%r1, 193, bar+100

#CHECK: clgijnhe	%r1, 193, bar+100                 # encoding: [0xec,0x14,A,A,0xc1,0x7d]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	clgijnhe	%r1, 193, bar+100

#CHECK: clgijlh	%r1, 193, bar+100                 # encoding: [0xec,0x16,A,A,0xc1,0x7d]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	clgijlh	%r1, 193, bar+100

#CHECK: clgijne	%r1, 193, bar+100                 # encoding: [0xec,0x16,A,A,0xc1,0x7d]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	clgijne	%r1, 193, bar+100

#CHECK: clgije	%r1, 193, bar+100                 # encoding: [0xec,0x18,A,A,0xc1,0x7d]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	clgije	%r1, 193, bar+100

#CHECK: clgijnlh	%r1, 193, bar+100                 # encoding: [0xec,0x18,A,A,0xc1,0x7d]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	clgijnlh	%r1, 193, bar+100

#CHECK: clgijhe	%r1, 193, bar+100                 # encoding: [0xec,0x1a,A,A,0xc1,0x7d]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	clgijhe	%r1, 193, bar+100

#CHECK: clgijnl	%r1, 193, bar+100                 # encoding: [0xec,0x1a,A,A,0xc1,0x7d]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	clgijnl	%r1, 193, bar+100

#CHECK: clgijle	%r1, 193, bar+100                 # encoding: [0xec,0x1c,A,A,0xc1,0x7d]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	clgijle	%r1, 193, bar+100

#CHECK: clgijnh	%r1, 193, bar+100                 # encoding: [0xec,0x1c,A,A,0xc1,0x7d]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	clgijnh	%r1, 193, bar+100

#CHECK: clgij	%r1, 193, 0, bar@PLT              # encoding: [0xec,0x10,A,A,0xc1,0x7d]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	clgij	%r1, 193, 0, bar@PLT

#CHECK: clgijh	%r1, 193, bar@PLT                 # encoding: [0xec,0x12,A,A,0xc1,0x7d]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	clgijh	%r1, 193, bar@PLT

#CHECK: clgijnle	%r1, 193, bar@PLT                 # encoding: [0xec,0x12,A,A,0xc1,0x7d]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	clgijnle	%r1, 193, bar@PLT

#CHECK: clgijl	%r1, 193, bar@PLT                 # encoding: [0xec,0x14,A,A,0xc1,0x7d]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	clgijl	%r1, 193, bar@PLT

#CHECK: clgijnhe	%r1, 193, bar@PLT                 # encoding: [0xec,0x14,A,A,0xc1,0x7d]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	clgijnhe	%r1, 193, bar@PLT

#CHECK: clgijlh	%r1, 193, bar@PLT                 # encoding: [0xec,0x16,A,A,0xc1,0x7d]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	clgijlh	%r1, 193, bar@PLT

#CHECK: clgijne	%r1, 193, bar@PLT                 # encoding: [0xec,0x16,A,A,0xc1,0x7d]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	clgijne	%r1, 193, bar@PLT

#CHECK: clgije	%r1, 193, bar@PLT                 # encoding: [0xec,0x18,A,A,0xc1,0x7d]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	clgije	%r1, 193, bar@PLT

#CHECK: clgijnlh	%r1, 193, bar@PLT                 # encoding: [0xec,0x18,A,A,0xc1,0x7d]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	clgijnlh	%r1, 193, bar@PLT

#CHECK: clgijhe	%r1, 193, bar@PLT                 # encoding: [0xec,0x1a,A,A,0xc1,0x7d]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	clgijhe	%r1, 193, bar@PLT

#CHECK: clgijnl	%r1, 193, bar@PLT                 # encoding: [0xec,0x1a,A,A,0xc1,0x7d]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	clgijnl	%r1, 193, bar@PLT

#CHECK: clgijle	%r1, 193, bar@PLT                 # encoding: [0xec,0x1c,A,A,0xc1,0x7d]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	clgijle	%r1, 193, bar@PLT

#CHECK: clgijnh	%r1, 193, bar@PLT                 # encoding: [0xec,0x1c,A,A,0xc1,0x7d]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	clgijnh	%r1, 193, bar@PLT

#CHECK: clgit     %r0, 0, 12             # encoding: [0xec,0x00,0x00,0x00,0xc0,0x71]
#CHECK: clgit     %r0, 65535, 12         # encoding: [0xec,0x00,0xff,0xff,0xc0,0x71]
#CHECK: clgit     %r0, 32768, 12         # encoding: [0xec,0x00,0x80,0x00,0xc0,0x71]
#CHECK: clgith    %r15, 1                # encoding: [0xec,0xf0,0x00,0x01,0x20,0x71]
#CHECK: clgitl    %r15, 1                # encoding: [0xec,0xf0,0x00,0x01,0x40,0x71]
#CHECK: clgite    %r15, 1                # encoding: [0xec,0xf0,0x00,0x01,0x80,0x71]
#CHECK: clgitne   %r15, 1                # encoding: [0xec,0xf0,0x00,0x01,0x60,0x71]
#CHECK: clgitnl   %r15, 1                # encoding: [0xec,0xf0,0x00,0x01,0xa0,0x71]
#CHECK: clgitnh   %r15, 1                # encoding: [0xec,0xf0,0x00,0x01,0xc0,0x71]
#CHECK: clgitnle  %r15, 1                # encoding: [0xec,0xf0,0x00,0x01,0x20,0x71]
#CHECK: clgitnhe  %r15, 1                # encoding: [0xec,0xf0,0x00,0x01,0x40,0x71]
#CHECK: clgitnlh  %r15, 1                # encoding: [0xec,0xf0,0x00,0x01,0x80,0x71]
#CHECK: clgitlh   %r15, 1                # encoding: [0xec,0xf0,0x00,0x01,0x60,0x71]
#CHECK: clgithe   %r15, 1                # encoding: [0xec,0xf0,0x00,0x01,0xa0,0x71]
#CHECK: clgitle   %r15, 1                # encoding: [0xec,0xf0,0x00,0x01,0xc0,0x71]

        clgit     %r0, 0, 12
        clgit     %r0, 65535, 12
        clgit     %r0, 32768, 12
        clgith    %r15, 1
        clgitl    %r15, 1
        clgite    %r15, 1
        clgitne   %r15, 1
        clgitnl   %r15, 1
        clgitnh   %r15, 1
        clgitnle  %r15, 1
        clgitnhe  %r15, 1
        clgitnlh  %r15, 1
        clgitlh   %r15, 1
        clgithe   %r15, 1
        clgitle   %r15, 1

#CHECK: clgr	%r0, %r0                # encoding: [0xb9,0x21,0x00,0x00]
#CHECK: clgr	%r0, %r15               # encoding: [0xb9,0x21,0x00,0x0f]
#CHECK: clgr	%r15, %r0               # encoding: [0xb9,0x21,0x00,0xf0]
#CHECK: clgr	%r7, %r8                # encoding: [0xb9,0x21,0x00,0x78]

	clgr	%r0,%r0
	clgr	%r0,%r15
	clgr	%r15,%r0
	clgr	%r7,%r8

#CHECK: clgrb	%r0, %r0, 0, 0          # encoding: [0xec,0x00,0x00,0x00,0x00,0xe5]
#CHECK: clgrb	%r0, %r15, 0, 0         # encoding: [0xec,0x0f,0x00,0x00,0x00,0xe5]
#CHECK: clgrb	%r15, %r0, 0, 0         # encoding: [0xec,0xf0,0x00,0x00,0x00,0xe5]
#CHECK: clgrb	%r7, %r2, 0, 0          # encoding: [0xec,0x72,0x00,0x00,0x00,0xe5]
#CHECK: clgrb	%r0, %r0, 1, 0          # encoding: [0xec,0x00,0x00,0x00,0x10,0xe5]
#CHECK: clgrb	%r0, %r0, 15, 0         # encoding: [0xec,0x00,0x00,0x00,0xf0,0xe5]
#CHECK: clgrb	%r0, %r0, 0, 0(%r13)    # encoding: [0xec,0x00,0xd0,0x00,0x00,0xe5]
#CHECK: clgrb	%r0, %r0, 0, 4095       # encoding: [0xec,0x00,0x0f,0xff,0x00,0xe5]
#CHECK: clgrb	%r0, %r0, 0, 4095(%r7)  # encoding: [0xec,0x00,0x7f,0xff,0x00,0xe5]
	clgrb	%r0, %r0, 0, 0
	clgrb	%r0, %r15, 0, 0
	clgrb	%r15, %r0, 0, 0
	clgrb	%r7, %r2, 0, 0
	clgrb	%r0, %r0, 1, 0
	clgrb	%r0, %r0, 15, 0
	clgrb	%r0, %r0, 0, 0(%r13)
	clgrb	%r0, %r0, 0, 4095
	clgrb	%r0, %r0, 0, 4095(%r7)

#CHECK: clgrbe	%r0, %r0, 0             # encoding: [0xec,0x00,0x00,0x00,0x80,0xe5]
#CHECK: clgrbe	%r0, %r15, 0            # encoding: [0xec,0x0f,0x00,0x00,0x80,0xe5]
#CHECK: clgrbe	%r15, %r0, 0            # encoding: [0xec,0xf0,0x00,0x00,0x80,0xe5]
#CHECK: clgrbe	%r7, %r2, 0             # encoding: [0xec,0x72,0x00,0x00,0x80,0xe5]
#CHECK: clgrbe	%r0, %r0, 0(%r13)       # encoding: [0xec,0x00,0xd0,0x00,0x80,0xe5]
#CHECK: clgrbe	%r0, %r0, 4095          # encoding: [0xec,0x00,0x0f,0xff,0x80,0xe5]
#CHECK: clgrbe	%r0, %r0, 4095(%r7)     # encoding: [0xec,0x00,0x7f,0xff,0x80,0xe5]
	clgrbe	%r0, %r0, 0
	clgrbe	%r0, %r15, 0
	clgrbe	%r15, %r0, 0
	clgrbe	%r7, %r2, 0
	clgrbe	%r0, %r0, 0(%r13)
	clgrbe	%r0, %r0, 4095
	clgrbe	%r0, %r0, 4095(%r7)

#CHECK: clgrb	%r1, %r2, 2, 3(%r4)     # encoding: [0xec,0x12,0x40,0x03,0x20,0xe5]
#CHECK: clgrbh	%r1, %r2, 3(%r4)        # encoding: [0xec,0x12,0x40,0x03,0x20,0xe5]
#CHECK: clgrbnle	%r1, %r2, 3(%r4)        # encoding: [0xec,0x12,0x40,0x03,0x20,0xe5]
	clgrb	%r1, %r2, 2, 3(%r4)
	clgrbh	%r1, %r2, 3(%r4)
	clgrbnle	%r1, %r2, 3(%r4)

#CHECK: clgrb	%r1, %r2, 4, 3(%r4)     # encoding: [0xec,0x12,0x40,0x03,0x40,0xe5]
#CHECK: clgrbl	%r1, %r2, 3(%r4)        # encoding: [0xec,0x12,0x40,0x03,0x40,0xe5]
#CHECK: clgrbnhe	%r1, %r2, 3(%r4)        # encoding: [0xec,0x12,0x40,0x03,0x40,0xe5]
	clgrb	%r1, %r2, 4, 3(%r4)
	clgrbl	%r1, %r2, 3(%r4)
	clgrbnhe	%r1, %r2, 3(%r4)

#CHECK: clgrb	%r1, %r2, 6, 3(%r4)     # encoding: [0xec,0x12,0x40,0x03,0x60,0xe5]
#CHECK: clgrblh	%r1, %r2, 3(%r4)        # encoding: [0xec,0x12,0x40,0x03,0x60,0xe5]
#CHECK: clgrbne	%r1, %r2, 3(%r4)        # encoding: [0xec,0x12,0x40,0x03,0x60,0xe5]
	clgrb	%r1, %r2, 6, 3(%r4)
	clgrblh	%r1, %r2, 3(%r4)
	clgrbne	%r1, %r2, 3(%r4)

#CHECK: clgrb	%r1, %r2, 8, 3(%r4)     # encoding: [0xec,0x12,0x40,0x03,0x80,0xe5]
#CHECK: clgrbe	%r1, %r2, 3(%r4)        # encoding: [0xec,0x12,0x40,0x03,0x80,0xe5]
#CHECK: clgrbnlh	%r1, %r2, 3(%r4)        # encoding: [0xec,0x12,0x40,0x03,0x80,0xe5]
	clgrb	%r1, %r2, 8, 3(%r4)
	clgrbe	%r1, %r2, 3(%r4)
	clgrbnlh	%r1, %r2, 3(%r4)

#CHECK: clgrb	%r1, %r2, 10, 3(%r4)    # encoding: [0xec,0x12,0x40,0x03,0xa0,0xe5]
#CHECK: clgrbhe	%r1, %r2, 3(%r4)        # encoding: [0xec,0x12,0x40,0x03,0xa0,0xe5]
#CHECK: clgrbnl	%r1, %r2, 3(%r4)        # encoding: [0xec,0x12,0x40,0x03,0xa0,0xe5]
	clgrb	%r1, %r2, 10, 3(%r4)
	clgrbhe	%r1, %r2, 3(%r4)
	clgrbnl	%r1, %r2, 3(%r4)

#CHECK: clgrb	%r1, %r2, 12, 3(%r4)    # encoding: [0xec,0x12,0x40,0x03,0xc0,0xe5]
#CHECK: clgrble	%r1, %r2, 3(%r4)        # encoding: [0xec,0x12,0x40,0x03,0xc0,0xe5]
#CHECK: clgrbnh	%r1, %r2, 3(%r4)        # encoding: [0xec,0x12,0x40,0x03,0xc0,0xe5]
	clgrb	%r1, %r2, 12, 3(%r4)
	clgrble	%r1, %r2, 3(%r4)
	clgrbnh	%r1, %r2, 3(%r4)

#CHECK: clgrj	%r0, %r0, 0, .[[LAB:L.*]]	# encoding: [0xec,0x00,A,A,0x00,0x65]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC16DBL
#CHECK: clgrj	%r0, %r15, 0, .[[LAB:L.*]]	# encoding: [0xec,0x0f,A,A,0x00,0x65]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC16DBL
#CHECK: clgrj	%r15, %r0, 0, .[[LAB:L.*]]	# encoding: [0xec,0xf0,A,A,0x00,0x65]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC16DBL
#CHECK: clgrj	%r7, %r8, 0, .[[LAB:L.*]]	# encoding: [0xec,0x78,A,A,0x00,0x65]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC16DBL
	clgrj	%r0,%r0,0,0
	clgrj	%r0,%r15,0,0
	clgrj	%r15,%r0,0,0
	clgrj	%r7,%r8,0,0

#CHECK: clgrj	%r1, %r2, 0, .[[LAB:L.*]]-65536	# encoding: [0xec,0x12,A,A,0x00,0x65]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-65536)+2, kind: FK_390_PC16DBL
	clgrj	%r1, %r2, 0, -0x10000
#CHECK: clgrj	%r1, %r2, 0, .[[LAB:L.*]]-2	# encoding: [0xec,0x12,A,A,0x00,0x65]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC16DBL
	clgrj	%r1, %r2, 0, -2
#CHECK: clgrj	%r1, %r2, 0, .[[LAB:L.*]]		# encoding: [0xec,0x12,A,A,0x00,0x65]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC16DBL
	clgrj	%r1, %r2, 0, 0
#CHECK: clgrj	%r1, %r2, 0, .[[LAB:L.*]]+65534	# encoding: [0xec,0x12,A,A,0x00,0x65]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+65534)+2, kind: FK_390_PC16DBL
	clgrj	%r1, %r2, 0, 0xfffe

#CHECK: clgrj	%r1, %r2, 0, foo                  # encoding: [0xec,0x12,A,A,0x00,0x65]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clgrj	%r1, %r2, 0, foo

#CHECK: clgrj	%r1, %r2, 1, foo                  # encoding: [0xec,0x12,A,A,0x10,0x65]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clgrj	%r1, %r2, 1, foo

#CHECK: clgrj	%r1, %r2, 2, foo                  # encoding: [0xec,0x12,A,A,0x20,0x65]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: clgrjh	%r1, %r2, foo                     # encoding: [0xec,0x12,A,A,0x20,0x65]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: clgrjnle	%r1, %r2, foo                     # encoding: [0xec,0x12,A,A,0x20,0x65]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clgrj	%r1, %r2, 2, foo
	clgrjh	%r1, %r2, foo
	clgrjnle	%r1, %r2, foo

#CHECK: clgrj	%r1, %r2, 3, foo                  # encoding: [0xec,0x12,A,A,0x30,0x65]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clgrj	%r1, %r2, 3, foo

#CHECK: clgrj	%r1, %r2, 4, foo                  # encoding: [0xec,0x12,A,A,0x40,0x65]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: clgrjl	%r1, %r2, foo                     # encoding: [0xec,0x12,A,A,0x40,0x65]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: clgrjnhe	%r1, %r2, foo                     # encoding: [0xec,0x12,A,A,0x40,0x65]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clgrj	%r1, %r2, 4, foo
	clgrjl	%r1, %r2, foo
	clgrjnhe	%r1, %r2, foo

#CHECK: clgrj	%r1, %r2, 5, foo                  # encoding: [0xec,0x12,A,A,0x50,0x65]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clgrj	%r1, %r2, 5, foo

#CHECK: clgrj	%r1, %r2, 6, foo                  # encoding: [0xec,0x12,A,A,0x60,0x65]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: clgrjlh	%r1, %r2, foo                     # encoding: [0xec,0x12,A,A,0x60,0x65]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: clgrjne	%r1, %r2, foo                     # encoding: [0xec,0x12,A,A,0x60,0x65]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clgrj	%r1, %r2, 6, foo
	clgrjlh	%r1, %r2, foo
	clgrjne	%r1, %r2, foo

#CHECK: clgrj	%r1, %r2, 7, foo                  # encoding: [0xec,0x12,A,A,0x70,0x65]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clgrj	%r1, %r2, 7, foo

#CHECK: clgrj	%r1, %r2, 8, foo                  # encoding: [0xec,0x12,A,A,0x80,0x65]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: clgrje	%r1, %r2, foo                     # encoding: [0xec,0x12,A,A,0x80,0x65]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: clgrjnlh	%r1, %r2, foo                     # encoding: [0xec,0x12,A,A,0x80,0x65]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clgrj	%r1, %r2, 8, foo
	clgrje	%r1, %r2, foo
	clgrjnlh	%r1, %r2, foo

#CHECK: clgrj	%r1, %r2, 9, foo                  # encoding: [0xec,0x12,A,A,0x90,0x65]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clgrj	%r1, %r2, 9, foo

#CHECK: clgrj	%r1, %r2, 10, foo                 # encoding: [0xec,0x12,A,A,0xa0,0x65]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: clgrjhe	%r1, %r2, foo                     # encoding: [0xec,0x12,A,A,0xa0,0x65]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: clgrjnl	%r1, %r2, foo                     # encoding: [0xec,0x12,A,A,0xa0,0x65]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clgrj	%r1, %r2, 10, foo
	clgrjhe	%r1, %r2, foo
	clgrjnl	%r1, %r2, foo

#CHECK: clgrj	%r1, %r2, 11, foo                 # encoding: [0xec,0x12,A,A,0xb0,0x65]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clgrj	%r1, %r2, 11, foo

#CHECK: clgrj	%r1, %r2, 12, foo                 # encoding: [0xec,0x12,A,A,0xc0,0x65]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: clgrjle	%r1, %r2, foo                     # encoding: [0xec,0x12,A,A,0xc0,0x65]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: clgrjnh	%r1, %r2, foo                     # encoding: [0xec,0x12,A,A,0xc0,0x65]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clgrj	%r1, %r2, 12, foo
	clgrjle	%r1, %r2, foo
	clgrjnh	%r1, %r2, foo

#CHECK: clgrj	%r1, %r2, 13, foo                 # encoding: [0xec,0x12,A,A,0xd0,0x65]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clgrj	%r1, %r2, 13, foo

#CHECK: clgrj	%r1, %r2, 14, foo                 # encoding: [0xec,0x12,A,A,0xe0,0x65]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clgrj	%r1, %r2, 14, foo

#CHECK: clgrj	%r1, %r2, 15, foo                 # encoding: [0xec,0x12,A,A,0xf0,0x65]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clgrj	%r1, %r2, 15, foo

#CHECK: clgrj	%r1, %r2, 0, bar+100              # encoding: [0xec,0x12,A,A,0x00,0x65]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	clgrj	%r1, %r2, 0, bar+100

#CHECK: clgrjh	%r1, %r2, bar+100                 # encoding: [0xec,0x12,A,A,0x20,0x65]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	clgrjh	%r1, %r2, bar+100

#CHECK: clgrjnle	%r1, %r2, bar+100                 # encoding: [0xec,0x12,A,A,0x20,0x65]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	clgrjnle	%r1, %r2, bar+100

#CHECK: clgrjl	%r1, %r2, bar+100                 # encoding: [0xec,0x12,A,A,0x40,0x65]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	clgrjl	%r1, %r2, bar+100

#CHECK: clgrjnhe	%r1, %r2, bar+100                 # encoding: [0xec,0x12,A,A,0x40,0x65]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	clgrjnhe	%r1, %r2, bar+100

#CHECK: clgrjlh	%r1, %r2, bar+100                 # encoding: [0xec,0x12,A,A,0x60,0x65]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	clgrjlh	%r1, %r2, bar+100

#CHECK: clgrjne	%r1, %r2, bar+100                 # encoding: [0xec,0x12,A,A,0x60,0x65]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	clgrjne	%r1, %r2, bar+100

#CHECK: clgrje	%r1, %r2, bar+100                 # encoding: [0xec,0x12,A,A,0x80,0x65]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	clgrje	%r1, %r2, bar+100

#CHECK: clgrjnlh	%r1, %r2, bar+100                 # encoding: [0xec,0x12,A,A,0x80,0x65]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	clgrjnlh	%r1, %r2, bar+100

#CHECK: clgrjhe	%r1, %r2, bar+100                 # encoding: [0xec,0x12,A,A,0xa0,0x65]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	clgrjhe	%r1, %r2, bar+100

#CHECK: clgrjnl	%r1, %r2, bar+100                 # encoding: [0xec,0x12,A,A,0xa0,0x65]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	clgrjnl	%r1, %r2, bar+100

#CHECK: clgrjle	%r1, %r2, bar+100                 # encoding: [0xec,0x12,A,A,0xc0,0x65]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	clgrjle	%r1, %r2, bar+100

#CHECK: clgrjnh	%r1, %r2, bar+100                 # encoding: [0xec,0x12,A,A,0xc0,0x65]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	clgrjnh	%r1, %r2, bar+100

#CHECK: clgrj	%r1, %r2, 0, bar@PLT              # encoding: [0xec,0x12,A,A,0x00,0x65]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	clgrj	%r1, %r2, 0, bar@PLT

#CHECK: clgrjh	%r1, %r2, bar@PLT                 # encoding: [0xec,0x12,A,A,0x20,0x65]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	clgrjh	%r1, %r2, bar@PLT

#CHECK: clgrjnle	%r1, %r2, bar@PLT                 # encoding: [0xec,0x12,A,A,0x20,0x65]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	clgrjnle	%r1, %r2, bar@PLT

#CHECK: clgrjl	%r1, %r2, bar@PLT                 # encoding: [0xec,0x12,A,A,0x40,0x65]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	clgrjl	%r1, %r2, bar@PLT

#CHECK: clgrjnhe	%r1, %r2, bar@PLT                 # encoding: [0xec,0x12,A,A,0x40,0x65]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	clgrjnhe	%r1, %r2, bar@PLT

#CHECK: clgrjlh	%r1, %r2, bar@PLT                 # encoding: [0xec,0x12,A,A,0x60,0x65]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	clgrjlh	%r1, %r2, bar@PLT

#CHECK: clgrjne	%r1, %r2, bar@PLT                 # encoding: [0xec,0x12,A,A,0x60,0x65]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	clgrjne	%r1, %r2, bar@PLT

#CHECK: clgrje	%r1, %r2, bar@PLT                 # encoding: [0xec,0x12,A,A,0x80,0x65]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	clgrje	%r1, %r2, bar@PLT

#CHECK: clgrjnlh	%r1, %r2, bar@PLT                 # encoding: [0xec,0x12,A,A,0x80,0x65]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	clgrjnlh	%r1, %r2, bar@PLT

#CHECK: clgrjhe	%r1, %r2, bar@PLT                 # encoding: [0xec,0x12,A,A,0xa0,0x65]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	clgrjhe	%r1, %r2, bar@PLT

#CHECK: clgrjnl	%r1, %r2, bar@PLT                 # encoding: [0xec,0x12,A,A,0xa0,0x65]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	clgrjnl	%r1, %r2, bar@PLT

#CHECK: clgrjle	%r1, %r2, bar@PLT                 # encoding: [0xec,0x12,A,A,0xc0,0x65]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	clgrjle	%r1, %r2, bar@PLT

#CHECK: clgrjnh	%r1, %r2, bar@PLT                 # encoding: [0xec,0x12,A,A,0xc0,0x65]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	clgrjnh	%r1, %r2, bar@PLT

#CHECK: clgrl	%r0, .[[LAB:L.*]]-4294967296 # encoding: [0xc6,0x0a,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-4294967296)+2, kind: FK_390_PC32DBL
	clgrl	%r0, -0x100000000
#CHECK: clgrl	%r0, .[[LAB:L.*]]-2	# encoding: [0xc6,0x0a,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC32DBL
	clgrl	%r0, -2
#CHECK: clgrl	%r0, .[[LAB:L.*]]	# encoding: [0xc6,0x0a,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC32DBL
	clgrl	%r0, 0
#CHECK: clgrl	%r0, .[[LAB:L.*]]+4294967294 # encoding: [0xc6,0x0a,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+4294967294)+2, kind: FK_390_PC32DBL
	clgrl	%r0, 0xfffffffe

#CHECK: clgrl	%r0, foo                # encoding: [0xc6,0x0a,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: clgrl	%r15, foo               # encoding: [0xc6,0xfa,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL

	clgrl	%r0,foo
	clgrl	%r15,foo

#CHECK: clgrl	%r3, bar+100            # encoding: [0xc6,0x3a,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: clgrl	%r4, bar+100            # encoding: [0xc6,0x4a,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL

	clgrl	%r3,bar+100
	clgrl	%r4,bar+100

#CHECK: clgrl	%r7, frob@PLT           # encoding: [0xc6,0x7a,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL
#CHECK: clgrl	%r8, frob@PLT           # encoding: [0xc6,0x8a,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL

	clgrl	%r7,frob@PLT
	clgrl	%r8,frob@PLT

#CHECK: clhhsi	0, 0                    # encoding: [0xe5,0x55,0x00,0x00,0x00,0x00]
#CHECK: clhhsi	4095, 0                 # encoding: [0xe5,0x55,0x0f,0xff,0x00,0x00]
#CHECK: clhhsi	0, 65535                # encoding: [0xe5,0x55,0x00,0x00,0xff,0xff]
#CHECK: clhhsi	0(%r1), 42              # encoding: [0xe5,0x55,0x10,0x00,0x00,0x2a]
#CHECK: clhhsi	0(%r15), 42             # encoding: [0xe5,0x55,0xf0,0x00,0x00,0x2a]
#CHECK: clhhsi	4095(%r1), 42           # encoding: [0xe5,0x55,0x1f,0xff,0x00,0x2a]
#CHECK: clhhsi	4095(%r15), 42          # encoding: [0xe5,0x55,0xff,0xff,0x00,0x2a]

	clhhsi	0, 0
	clhhsi	4095, 0
	clhhsi	0, 65535
	clhhsi	0(%r1), 42
	clhhsi	0(%r15), 42
	clhhsi	4095(%r1), 42
	clhhsi	4095(%r15), 42

#CHECK: clhrl	%r0, .[[LAB:L.*]]-4294967296 # encoding: [0xc6,0x07,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-4294967296)+2, kind: FK_390_PC32DBL
	clhrl	%r0, -0x100000000
#CHECK: clhrl	%r0, .[[LAB:L.*]]-2	# encoding: [0xc6,0x07,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC32DBL
	clhrl	%r0, -2
#CHECK: clhrl	%r0, .[[LAB:L.*]]	# encoding: [0xc6,0x07,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC32DBL
	clhrl	%r0, 0
#CHECK: clhrl	%r0, .[[LAB:L.*]]+4294967294 # encoding: [0xc6,0x07,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+4294967294)+2, kind: FK_390_PC32DBL
	clhrl	%r0, 0xfffffffe

#CHECK: clhrl	%r0, foo                # encoding: [0xc6,0x07,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: clhrl	%r15, foo               # encoding: [0xc6,0xf7,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL

	clhrl	%r0,foo
	clhrl	%r15,foo

#CHECK: clhrl	%r3, bar+100            # encoding: [0xc6,0x37,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: clhrl	%r4, bar+100            # encoding: [0xc6,0x47,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL

	clhrl	%r3,bar+100
	clhrl	%r4,bar+100

#CHECK: clhrl	%r7, frob@PLT           # encoding: [0xc6,0x77,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL
#CHECK: clhrl	%r8, frob@PLT           # encoding: [0xc6,0x87,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL

	clhrl	%r7,frob@PLT
	clhrl	%r8,frob@PLT

#CHECK: cli	0, 0                    # encoding: [0x95,0x00,0x00,0x00]
#CHECK: cli	4095, 0                 # encoding: [0x95,0x00,0x0f,0xff]
#CHECK: cli	0, 255                  # encoding: [0x95,0xff,0x00,0x00]
#CHECK: cli	0(%r1), 42              # encoding: [0x95,0x2a,0x10,0x00]
#CHECK: cli	0(%r15), 42             # encoding: [0x95,0x2a,0xf0,0x00]
#CHECK: cli	4095(%r1), 42           # encoding: [0x95,0x2a,0x1f,0xff]
#CHECK: cli	4095(%r15), 42          # encoding: [0x95,0x2a,0xff,0xff]

	cli	0, 0
	cli	4095, 0
	cli	0, 255
	cli	0(%r1), 42
	cli	0(%r15), 42
	cli	4095(%r1), 42
	cli	4095(%r15), 42

#CHECK: clib	%r0, 0, 0, 0            # encoding: [0xec,0x00,0x00,0x00,0x00,0xff]
#CHECK: clib	%r0, 128, 0, 0          # encoding: [0xec,0x00,0x00,0x00,0x80,0xff]
#CHECK: clib	%r0, 127, 0, 0          # encoding: [0xec,0x00,0x00,0x00,0x7f,0xff]
#CHECK: clib	%r15, 0, 0, 0           # encoding: [0xec,0xf0,0x00,0x00,0x00,0xff]
#CHECK: clib	%r7, 255, 0, 0          # encoding: [0xec,0x70,0x00,0x00,0xff,0xff]
#CHECK: clib	%r0, 0, 1, 0            # encoding: [0xec,0x01,0x00,0x00,0x00,0xff]
#CHECK: clib	%r0, 0, 15, 0           # encoding: [0xec,0x0f,0x00,0x00,0x00,0xff]
#CHECK: clib	%r0, 0, 0, 0(%r13)      # encoding: [0xec,0x00,0xd0,0x00,0x00,0xff]
#CHECK: clib	%r0, 0, 0, 4095         # encoding: [0xec,0x00,0x0f,0xff,0x00,0xff]
#CHECK: clib	%r0, 0, 0, 4095(%r7)    # encoding: [0xec,0x00,0x7f,0xff,0x00,0xff]
	clib	%r0, 0, 0, 0
	clib	%r0, 128, 0, 0
	clib	%r0, 127, 0, 0
	clib	%r15, 0, 0, 0
	clib	%r7, 255, 0, 0
	clib	%r0, 0, 1, 0
	clib	%r0, 0, 15, 0
	clib	%r0, 0, 0, 0(%r13)
	clib	%r0, 0, 0, 4095
	clib	%r0, 0, 0, 4095(%r7)

#CHECK: clibe	%r0, 0, 0               # encoding: [0xec,0x08,0x00,0x00,0x00,0xff]
#CHECK: clibe	%r0, 128, 0             # encoding: [0xec,0x08,0x00,0x00,0x80,0xff]
#CHECK: clibe	%r0, 127, 0             # encoding: [0xec,0x08,0x00,0x00,0x7f,0xff]
#CHECK: clibe	%r15, 0, 0              # encoding: [0xec,0xf8,0x00,0x00,0x00,0xff]
#CHECK: clibe	%r7, 255, 0             # encoding: [0xec,0x78,0x00,0x00,0xff,0xff]
#CHECK: clibe	%r0, 0, 0(%r13)         # encoding: [0xec,0x08,0xd0,0x00,0x00,0xff]
#CHECK: clibe	%r0, 0, 4095            # encoding: [0xec,0x08,0x0f,0xff,0x00,0xff]
#CHECK: clibe	%r0, 0, 4095(%r7)       # encoding: [0xec,0x08,0x7f,0xff,0x00,0xff]
	clibe	%r0, 0, 0
	clibe	%r0, 128, 0
	clibe	%r0, 127, 0
	clibe	%r15, 0, 0
	clibe	%r7, 255, 0
	clibe	%r0, 0, 0(%r13)
	clibe	%r0, 0, 4095
	clibe	%r0, 0, 4095(%r7)

#CHECK: clib	%r1, 2, 2, 3(%r4)       # encoding: [0xec,0x12,0x40,0x03,0x02,0xff]
#CHECK: clibh	%r1, 2, 3(%r4)          # encoding: [0xec,0x12,0x40,0x03,0x02,0xff]
#CHECK: clibnle	%r1, 2, 3(%r4)          # encoding: [0xec,0x12,0x40,0x03,0x02,0xff]
	clib	%r1, 2, 2, 3(%r4)
	clibh	%r1, 2, 3(%r4)
	clibnle	%r1, 2, 3(%r4)

#CHECK: clib	%r1, 2, 4, 3(%r4)       # encoding: [0xec,0x14,0x40,0x03,0x02,0xff]
#CHECK: clibl	%r1, 2, 3(%r4)          # encoding: [0xec,0x14,0x40,0x03,0x02,0xff]
#CHECK: clibnhe	%r1, 2, 3(%r4)          # encoding: [0xec,0x14,0x40,0x03,0x02,0xff]
	clib	%r1, 2, 4, 3(%r4)
	clibl	%r1, 2, 3(%r4)
	clibnhe	%r1, 2, 3(%r4)

#CHECK: clib	%r1, 2, 6, 3(%r4)       # encoding: [0xec,0x16,0x40,0x03,0x02,0xff]
#CHECK: cliblh	%r1, 2, 3(%r4)          # encoding: [0xec,0x16,0x40,0x03,0x02,0xff]
#CHECK: clibne	%r1, 2, 3(%r4)          # encoding: [0xec,0x16,0x40,0x03,0x02,0xff]
	clib	%r1, 2, 6, 3(%r4)
	cliblh	%r1, 2, 3(%r4)
	clibne	%r1, 2, 3(%r4)

#CHECK: clib	%r1, 2, 8, 3(%r4)       # encoding: [0xec,0x18,0x40,0x03,0x02,0xff]
#CHECK: clibe	%r1, 2, 3(%r4)          # encoding: [0xec,0x18,0x40,0x03,0x02,0xff]
#CHECK: clibnlh	%r1, 2, 3(%r4)          # encoding: [0xec,0x18,0x40,0x03,0x02,0xff]
	clib	%r1, 2, 8, 3(%r4)
	clibe	%r1, 2, 3(%r4)
	clibnlh	%r1, 2, 3(%r4)

#CHECK: clib	%r1, 2, 10, 3(%r4)      # encoding: [0xec,0x1a,0x40,0x03,0x02,0xff]
#CHECK: clibhe	%r1, 2, 3(%r4)          # encoding: [0xec,0x1a,0x40,0x03,0x02,0xff]
#CHECK: clibnl	%r1, 2, 3(%r4)          # encoding: [0xec,0x1a,0x40,0x03,0x02,0xff]
	clib	%r1, 2, 10, 3(%r4)
	clibhe	%r1, 2, 3(%r4)
	clibnl	%r1, 2, 3(%r4)

#CHECK: clib	%r1, 2, 12, 3(%r4)      # encoding: [0xec,0x1c,0x40,0x03,0x02,0xff]
#CHECK: clible	%r1, 2, 3(%r4)          # encoding: [0xec,0x1c,0x40,0x03,0x02,0xff]
#CHECK: clibnh	%r1, 2, 3(%r4)          # encoding: [0xec,0x1c,0x40,0x03,0x02,0xff]
	clib	%r1, 2, 12, 3(%r4)
	clible	%r1, 2, 3(%r4)
	clibnh	%r1, 2, 3(%r4)

#CHECK: clij	%r0, 0, 0, .[[LAB:L.*]]	# encoding: [0xec,0x00,A,A,0x00,0x7f]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC16DBL
#CHECK: clij	%r0, 255, 0, .[[LAB:L.*]]	# encoding: [0xec,0x00,A,A,0xff,0x7f]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC16DBL
#CHECK: clij	%r15, 0, 0, .[[LAB:L.*]]	# encoding: [0xec,0xf0,A,A,0x00,0x7f]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC16DBL
	clij	%r0, 0, 0, 0
	clij	%r0, 255, 0, 0
	clij	%r15, 0, 0, 0

#CHECK: clij	%r1, 193, 0, .[[LAB:L.*]]-65536	# encoding: [0xec,0x10,A,A,0xc1,0x7f]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-65536)+2, kind: FK_390_PC16DBL
	clij	%r1, 193, 0, -0x10000
#CHECK: clij	%r1, 193, 0, .[[LAB:L.*]]-2	# encoding: [0xec,0x10,A,A,0xc1,0x7f]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC16DBL
	clij	%r1, 193, 0, -2
#CHECK: clij	%r1, 193, 0, .[[LAB:L.*]]		# encoding: [0xec,0x10,A,A,0xc1,0x7f]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC16DBL
	clij	%r1, 193, 0, 0
#CHECK: clij	%r1, 193, 0, .[[LAB:L.*]]+65534	# encoding: [0xec,0x10,A,A,0xc1,0x7f]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+65534)+2, kind: FK_390_PC16DBL
	clij	%r1, 193, 0, 0xfffe

#CHECK: clij	%r1, 193, 0, foo                  # encoding: [0xec,0x10,A,A,0xc1,0x7f]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clij	%r1, 193, 0, foo

#CHECK: clij	%r1, 193, 1, foo                  # encoding: [0xec,0x11,A,A,0xc1,0x7f]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clij	%r1, 193, 1, foo

#CHECK: clij	%r1, 193, 2, foo                  # encoding: [0xec,0x12,A,A,0xc1,0x7f]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: clijh	%r1, 193, foo                     # encoding: [0xec,0x12,A,A,0xc1,0x7f]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: clijnle	%r1, 193, foo                     # encoding: [0xec,0x12,A,A,0xc1,0x7f]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clij	%r1, 193, 2, foo
	clijh	%r1, 193, foo
	clijnle	%r1, 193, foo

#CHECK: clij	%r1, 193, 3, foo                  # encoding: [0xec,0x13,A,A,0xc1,0x7f]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clij	%r1, 193, 3, foo

#CHECK: clij	%r1, 193, 4, foo                  # encoding: [0xec,0x14,A,A,0xc1,0x7f]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: clijl	%r1, 193, foo                     # encoding: [0xec,0x14,A,A,0xc1,0x7f]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: clijnhe	%r1, 193, foo                     # encoding: [0xec,0x14,A,A,0xc1,0x7f]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clij	%r1, 193, 4, foo
	clijl	%r1, 193, foo
	clijnhe	%r1, 193, foo

#CHECK: clij	%r1, 193, 5, foo                  # encoding: [0xec,0x15,A,A,0xc1,0x7f]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clij	%r1, 193, 5, foo

#CHECK: clij	%r1, 193, 6, foo                  # encoding: [0xec,0x16,A,A,0xc1,0x7f]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: clijlh	%r1, 193, foo                     # encoding: [0xec,0x16,A,A,0xc1,0x7f]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: clijne	%r1, 193, foo                     # encoding: [0xec,0x16,A,A,0xc1,0x7f]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clij	%r1, 193, 6, foo
	clijlh	%r1, 193, foo
	clijne	%r1, 193, foo

#CHECK: clij	%r1, 193, 7, foo                  # encoding: [0xec,0x17,A,A,0xc1,0x7f]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clij	%r1, 193, 7, foo

#CHECK: clij	%r1, 193, 8, foo                  # encoding: [0xec,0x18,A,A,0xc1,0x7f]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: clije	%r1, 193, foo                     # encoding: [0xec,0x18,A,A,0xc1,0x7f]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: clijnlh	%r1, 193, foo                     # encoding: [0xec,0x18,A,A,0xc1,0x7f]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clij	%r1, 193, 8, foo
	clije	%r1, 193, foo
	clijnlh	%r1, 193, foo

#CHECK: clij	%r1, 193, 9, foo                  # encoding: [0xec,0x19,A,A,0xc1,0x7f]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clij	%r1, 193, 9, foo

#CHECK: clij	%r1, 193, 10, foo                 # encoding: [0xec,0x1a,A,A,0xc1,0x7f]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: clijhe	%r1, 193, foo                     # encoding: [0xec,0x1a,A,A,0xc1,0x7f]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: clijnl	%r1, 193, foo                     # encoding: [0xec,0x1a,A,A,0xc1,0x7f]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clij	%r1, 193, 10, foo
	clijhe	%r1, 193, foo
	clijnl	%r1, 193, foo

#CHECK: clij	%r1, 193, 11, foo                 # encoding: [0xec,0x1b,A,A,0xc1,0x7f]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clij	%r1, 193, 11, foo

#CHECK: clij	%r1, 193, 12, foo                 # encoding: [0xec,0x1c,A,A,0xc1,0x7f]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: clijle	%r1, 193, foo                     # encoding: [0xec,0x1c,A,A,0xc1,0x7f]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: clijnh	%r1, 193, foo                     # encoding: [0xec,0x1c,A,A,0xc1,0x7f]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clij	%r1, 193, 12, foo
	clijle	%r1, 193, foo
	clijnh	%r1, 193, foo

#CHECK: clij	%r1, 193, 13, foo                 # encoding: [0xec,0x1d,A,A,0xc1,0x7f]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clij	%r1, 193, 13, foo

#CHECK: clij	%r1, 193, 14, foo                 # encoding: [0xec,0x1e,A,A,0xc1,0x7f]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clij	%r1, 193, 14, foo

#CHECK: clij	%r1, 193, 15, foo                 # encoding: [0xec,0x1f,A,A,0xc1,0x7f]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clij	%r1, 193, 15, foo

#CHECK: clij	%r1, 193, 0, bar+100              # encoding: [0xec,0x10,A,A,0xc1,0x7f]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	clij	%r1, 193, 0, bar+100

#CHECK: clijh	%r1, 193, bar+100                 # encoding: [0xec,0x12,A,A,0xc1,0x7f]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	clijh	%r1, 193, bar+100

#CHECK: clijnle	%r1, 193, bar+100                 # encoding: [0xec,0x12,A,A,0xc1,0x7f]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	clijnle	%r1, 193, bar+100

#CHECK: clijl	%r1, 193, bar+100                 # encoding: [0xec,0x14,A,A,0xc1,0x7f]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	clijl	%r1, 193, bar+100

#CHECK: clijnhe	%r1, 193, bar+100                 # encoding: [0xec,0x14,A,A,0xc1,0x7f]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	clijnhe	%r1, 193, bar+100

#CHECK: clijlh	%r1, 193, bar+100                 # encoding: [0xec,0x16,A,A,0xc1,0x7f]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	clijlh	%r1, 193, bar+100

#CHECK: clijne	%r1, 193, bar+100                 # encoding: [0xec,0x16,A,A,0xc1,0x7f]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	clijne	%r1, 193, bar+100

#CHECK: clije	%r1, 193, bar+100                 # encoding: [0xec,0x18,A,A,0xc1,0x7f]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	clije	%r1, 193, bar+100

#CHECK: clijnlh	%r1, 193, bar+100                 # encoding: [0xec,0x18,A,A,0xc1,0x7f]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	clijnlh	%r1, 193, bar+100

#CHECK: clijhe	%r1, 193, bar+100                 # encoding: [0xec,0x1a,A,A,0xc1,0x7f]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	clijhe	%r1, 193, bar+100

#CHECK: clijnl	%r1, 193, bar+100                 # encoding: [0xec,0x1a,A,A,0xc1,0x7f]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	clijnl	%r1, 193, bar+100

#CHECK: clijle	%r1, 193, bar+100                 # encoding: [0xec,0x1c,A,A,0xc1,0x7f]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	clijle	%r1, 193, bar+100

#CHECK: clijnh	%r1, 193, bar+100                 # encoding: [0xec,0x1c,A,A,0xc1,0x7f]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	clijnh	%r1, 193, bar+100

#CHECK: clij	%r1, 193, 0, bar@PLT              # encoding: [0xec,0x10,A,A,0xc1,0x7f]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	clij	%r1, 193, 0, bar@PLT

#CHECK: clijh	%r1, 193, bar@PLT                 # encoding: [0xec,0x12,A,A,0xc1,0x7f]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	clijh	%r1, 193, bar@PLT

#CHECK: clijnle	%r1, 193, bar@PLT                 # encoding: [0xec,0x12,A,A,0xc1,0x7f]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	clijnle	%r1, 193, bar@PLT

#CHECK: clijl	%r1, 193, bar@PLT                 # encoding: [0xec,0x14,A,A,0xc1,0x7f]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	clijl	%r1, 193, bar@PLT

#CHECK: clijnhe	%r1, 193, bar@PLT                 # encoding: [0xec,0x14,A,A,0xc1,0x7f]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	clijnhe	%r1, 193, bar@PLT

#CHECK: clijlh	%r1, 193, bar@PLT                 # encoding: [0xec,0x16,A,A,0xc1,0x7f]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	clijlh	%r1, 193, bar@PLT

#CHECK: clijne	%r1, 193, bar@PLT                 # encoding: [0xec,0x16,A,A,0xc1,0x7f]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	clijne	%r1, 193, bar@PLT

#CHECK: clije	%r1, 193, bar@PLT                 # encoding: [0xec,0x18,A,A,0xc1,0x7f]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	clije	%r1, 193, bar@PLT

#CHECK: clijnlh	%r1, 193, bar@PLT                 # encoding: [0xec,0x18,A,A,0xc1,0x7f]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	clijnlh	%r1, 193, bar@PLT

#CHECK: clijhe	%r1, 193, bar@PLT                 # encoding: [0xec,0x1a,A,A,0xc1,0x7f]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	clijhe	%r1, 193, bar@PLT

#CHECK: clijnl	%r1, 193, bar@PLT                 # encoding: [0xec,0x1a,A,A,0xc1,0x7f]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	clijnl	%r1, 193, bar@PLT

#CHECK: clijle	%r1, 193, bar@PLT                 # encoding: [0xec,0x1c,A,A,0xc1,0x7f]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	clijle	%r1, 193, bar@PLT

#CHECK: clijnh	%r1, 193, bar@PLT                 # encoding: [0xec,0x1c,A,A,0xc1,0x7f]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	clijnh	%r1, 193, bar@PLT

#CHECK: cliy	-524288, 0              # encoding: [0xeb,0x00,0x00,0x00,0x80,0x55]
#CHECK: cliy	-1, 0                   # encoding: [0xeb,0x00,0x0f,0xff,0xff,0x55]
#CHECK: cliy	0, 0                    # encoding: [0xeb,0x00,0x00,0x00,0x00,0x55]
#CHECK: cliy	1, 0                    # encoding: [0xeb,0x00,0x00,0x01,0x00,0x55]
#CHECK: cliy	524287, 0               # encoding: [0xeb,0x00,0x0f,0xff,0x7f,0x55]
#CHECK: cliy	0, 255                  # encoding: [0xeb,0xff,0x00,0x00,0x00,0x55]
#CHECK: cliy	0(%r1), 42              # encoding: [0xeb,0x2a,0x10,0x00,0x00,0x55]
#CHECK: cliy	0(%r15), 42             # encoding: [0xeb,0x2a,0xf0,0x00,0x00,0x55]
#CHECK: cliy	524287(%r1), 42         # encoding: [0xeb,0x2a,0x1f,0xff,0x7f,0x55]
#CHECK: cliy	524287(%r15), 42        # encoding: [0xeb,0x2a,0xff,0xff,0x7f,0x55]

	cliy	-524288, 0
	cliy	-1, 0
	cliy	0, 0
	cliy	1, 0
	cliy	524287, 0
	cliy	0, 255
	cliy	0(%r1), 42
	cliy	0(%r15), 42
	cliy	524287(%r1), 42
	cliy	524287(%r15), 42

#CHECK: clm	%r0, 0, 0               # encoding: [0xbd,0x00,0x00,0x00]
#CHECK: clm	%r0, 15, 4095           # encoding: [0xbd,0x0f,0x0f,0xff]
#CHECK: clm	%r0, 0, 0(%r1)          # encoding: [0xbd,0x00,0x10,0x00]
#CHECK: clm	%r0, 0, 0(%r15)         # encoding: [0xbd,0x00,0xf0,0x00]
#CHECK: clm	%r15, 15, 4095(%r1)     # encoding: [0xbd,0xff,0x1f,0xff]
#CHECK: clm	%r0, 0, 4095(%r15)      # encoding: [0xbd,0x00,0xff,0xff]
#CHECK: clm	%r15, 0, 0              # encoding: [0xbd,0xf0,0x00,0x00]

	clm	%r0, 0, 0
	clm	%r0, 15, 4095
	clm	%r0, 0, 0(%r1)
	clm	%r0, 0, 0(%r15)
	clm	%r15, 15, 4095(%r1)
	clm	%r0, 0, 4095(%r15)
	clm	%r15, 0, 0

#CHECK: clmh	%r0, 0, -524288            # encoding: [0xeb,0x00,0x00,0x00,0x80,0x20]
#CHECK: clmh	%r0, 0, -1                 # encoding: [0xeb,0x00,0x0f,0xff,0xff,0x20]
#CHECK: clmh	%r0, 15, 0                 # encoding: [0xeb,0x0f,0x00,0x00,0x00,0x20]
#CHECK: clmh	%r0, 15, 1                 # encoding: [0xeb,0x0f,0x00,0x01,0x00,0x20]
#CHECK: clmh	%r0, 8, 524287             # encoding: [0xeb,0x08,0x0f,0xff,0x7f,0x20]
#CHECK: clmh	%r0, 8, 0(%r1)             # encoding: [0xeb,0x08,0x10,0x00,0x00,0x20]
#CHECK: clmh	%r0, 4, 0(%r15)            # encoding: [0xeb,0x04,0xf0,0x00,0x00,0x20]
#CHECK: clmh	%r0, 4, 524287(%r15)       # encoding: [0xeb,0x04,0xff,0xff,0x7f,0x20]
#CHECK: clmh	%r0, 0, 524287(%r1)        # encoding: [0xeb,0x00,0x1f,0xff,0x7f,0x20]
#CHECK: clmh	%r15, 0, 0                 # encoding: [0xeb,0xf0,0x00,0x00,0x00,0x20]

	clmh	%r0, 0, -524288
	clmh	%r0, 0, -1
	clmh	%r0, 15, 0
	clmh	%r0, 15, 1
	clmh	%r0, 8, 524287
	clmh	%r0, 8, 0(%r1)
	clmh	%r0, 4, 0(%r15)
	clmh	%r0, 4, 524287(%r15)
	clmh	%r0, 0, 524287(%r1)
	clmh	%r15, 0, 0

#CHECK: clmy	%r0, 0, -524288            # encoding: [0xeb,0x00,0x00,0x00,0x80,0x21]
#CHECK: clmy	%r0, 0, -1                 # encoding: [0xeb,0x00,0x0f,0xff,0xff,0x21]
#CHECK: clmy	%r0, 15, 0                 # encoding: [0xeb,0x0f,0x00,0x00,0x00,0x21]
#CHECK: clmy	%r0, 15, 1                 # encoding: [0xeb,0x0f,0x00,0x01,0x00,0x21]
#CHECK: clmy	%r0, 8, 524287             # encoding: [0xeb,0x08,0x0f,0xff,0x7f,0x21]
#CHECK: clmy	%r0, 8, 0(%r1)             # encoding: [0xeb,0x08,0x10,0x00,0x00,0x21]
#CHECK: clmy	%r0, 4, 0(%r15)            # encoding: [0xeb,0x04,0xf0,0x00,0x00,0x21]
#CHECK: clmy	%r0, 4, 524287(%r15)       # encoding: [0xeb,0x04,0xff,0xff,0x7f,0x21]
#CHECK: clmy	%r0, 0, 524287(%r1)        # encoding: [0xeb,0x00,0x1f,0xff,0x7f,0x21]
#CHECK: clmy	%r15, 0, 0                 # encoding: [0xeb,0xf0,0x00,0x00,0x00,0x21]

	clmy	%r0, 0, -524288
	clmy	%r0, 0, -1
	clmy	%r0, 15, 0
	clmy	%r0, 15, 1
	clmy	%r0, 8, 524287
	clmy	%r0, 8, 0(%r1)
	clmy	%r0, 4, 0(%r15)
	clmy	%r0, 4, 524287(%r15)
	clmy	%r0, 0, 524287(%r1)
	clmy	%r15, 0, 0

#CHECK: clr	%r0, %r0                # encoding: [0x15,0x00]
#CHECK: clr	%r0, %r15               # encoding: [0x15,0x0f]
#CHECK: clr	%r15, %r0               # encoding: [0x15,0xf0]
#CHECK: clr	%r7, %r8                # encoding: [0x15,0x78]

	clr	%r0,%r0
	clr	%r0,%r15
	clr	%r15,%r0
	clr	%r7,%r8

#CHECK: clrb	%r0, %r0, 0, 0          # encoding: [0xec,0x00,0x00,0x00,0x00,0xf7]
#CHECK: clrb	%r0, %r15, 0, 0         # encoding: [0xec,0x0f,0x00,0x00,0x00,0xf7]
#CHECK: clrb	%r15, %r0, 0, 0         # encoding: [0xec,0xf0,0x00,0x00,0x00,0xf7]
#CHECK: clrb	%r7, %r2, 0, 0          # encoding: [0xec,0x72,0x00,0x00,0x00,0xf7]
#CHECK: clrb	%r0, %r0, 1, 0          # encoding: [0xec,0x00,0x00,0x00,0x10,0xf7]
#CHECK: clrb	%r0, %r0, 15, 0         # encoding: [0xec,0x00,0x00,0x00,0xf0,0xf7]
#CHECK: clrb	%r0, %r0, 0, 0(%r13)    # encoding: [0xec,0x00,0xd0,0x00,0x00,0xf7]
#CHECK: clrb	%r0, %r0, 0, 4095       # encoding: [0xec,0x00,0x0f,0xff,0x00,0xf7]
#CHECK: clrb	%r0, %r0, 0, 4095(%r7)  # encoding: [0xec,0x00,0x7f,0xff,0x00,0xf7]
	clrb	%r0, %r0, 0, 0
	clrb	%r0, %r15, 0, 0
	clrb	%r15, %r0, 0, 0
	clrb	%r7, %r2, 0, 0
	clrb	%r0, %r0, 1, 0
	clrb	%r0, %r0, 15, 0
	clrb	%r0, %r0, 0, 0(%r13)
	clrb	%r0, %r0, 0, 4095
	clrb	%r0, %r0, 0, 4095(%r7)

#CHECK: clrbe	%r0, %r0, 0             # encoding: [0xec,0x00,0x00,0x00,0x80,0xf7]
#CHECK: clrbe	%r0, %r15, 0            # encoding: [0xec,0x0f,0x00,0x00,0x80,0xf7]
#CHECK: clrbe	%r15, %r0, 0            # encoding: [0xec,0xf0,0x00,0x00,0x80,0xf7]
#CHECK: clrbe	%r7, %r2, 0             # encoding: [0xec,0x72,0x00,0x00,0x80,0xf7]
#CHECK: clrbe	%r0, %r0, 0(%r13)       # encoding: [0xec,0x00,0xd0,0x00,0x80,0xf7]
#CHECK: clrbe	%r0, %r0, 4095          # encoding: [0xec,0x00,0x0f,0xff,0x80,0xf7]
#CHECK: clrbe	%r0, %r0, 4095(%r7)     # encoding: [0xec,0x00,0x7f,0xff,0x80,0xf7]
	clrbe	%r0, %r0, 0
	clrbe	%r0, %r15, 0
	clrbe	%r15, %r0, 0
	clrbe	%r7, %r2, 0
	clrbe	%r0, %r0, 0(%r13)
	clrbe	%r0, %r0, 4095
	clrbe	%r0, %r0, 4095(%r7)

#CHECK: clrb	%r1, %r2, 2, 3(%r4)     # encoding: [0xec,0x12,0x40,0x03,0x20,0xf7]
#CHECK: clrbh	%r1, %r2, 3(%r4)        # encoding: [0xec,0x12,0x40,0x03,0x20,0xf7]
#CHECK: clrbnle	%r1, %r2, 3(%r4)        # encoding: [0xec,0x12,0x40,0x03,0x20,0xf7]
	clrb	%r1, %r2, 2, 3(%r4)
	clrbh	%r1, %r2, 3(%r4)
	clrbnle	%r1, %r2, 3(%r4)

#CHECK: clrb	%r1, %r2, 4, 3(%r4)     # encoding: [0xec,0x12,0x40,0x03,0x40,0xf7]
#CHECK: clrbl	%r1, %r2, 3(%r4)        # encoding: [0xec,0x12,0x40,0x03,0x40,0xf7]
#CHECK: clrbnhe	%r1, %r2, 3(%r4)        # encoding: [0xec,0x12,0x40,0x03,0x40,0xf7]
	clrb	%r1, %r2, 4, 3(%r4)
	clrbl	%r1, %r2, 3(%r4)
	clrbnhe	%r1, %r2, 3(%r4)

#CHECK: clrb	%r1, %r2, 6, 3(%r4)     # encoding: [0xec,0x12,0x40,0x03,0x60,0xf7]
#CHECK: clrblh	%r1, %r2, 3(%r4)        # encoding: [0xec,0x12,0x40,0x03,0x60,0xf7]
#CHECK: clrbne	%r1, %r2, 3(%r4)        # encoding: [0xec,0x12,0x40,0x03,0x60,0xf7]
	clrb	%r1, %r2, 6, 3(%r4)
	clrblh	%r1, %r2, 3(%r4)
	clrbne	%r1, %r2, 3(%r4)

#CHECK: clrb	%r1, %r2, 8, 3(%r4)     # encoding: [0xec,0x12,0x40,0x03,0x80,0xf7]
#CHECK: clrbe	%r1, %r2, 3(%r4)        # encoding: [0xec,0x12,0x40,0x03,0x80,0xf7]
#CHECK: clrbnlh	%r1, %r2, 3(%r4)        # encoding: [0xec,0x12,0x40,0x03,0x80,0xf7]
	clrb	%r1, %r2, 8, 3(%r4)
	clrbe	%r1, %r2, 3(%r4)
	clrbnlh	%r1, %r2, 3(%r4)

#CHECK: clrb	%r1, %r2, 10, 3(%r4)    # encoding: [0xec,0x12,0x40,0x03,0xa0,0xf7]
#CHECK: clrbhe	%r1, %r2, 3(%r4)        # encoding: [0xec,0x12,0x40,0x03,0xa0,0xf7]
#CHECK: clrbnl	%r1, %r2, 3(%r4)        # encoding: [0xec,0x12,0x40,0x03,0xa0,0xf7]
	clrb	%r1, %r2, 10, 3(%r4)
	clrbhe	%r1, %r2, 3(%r4)
	clrbnl	%r1, %r2, 3(%r4)

#CHECK: clrb	%r1, %r2, 12, 3(%r4)    # encoding: [0xec,0x12,0x40,0x03,0xc0,0xf7]
#CHECK: clrble	%r1, %r2, 3(%r4)        # encoding: [0xec,0x12,0x40,0x03,0xc0,0xf7]
#CHECK: clrbnh	%r1, %r2, 3(%r4)        # encoding: [0xec,0x12,0x40,0x03,0xc0,0xf7]
	clrb	%r1, %r2, 12, 3(%r4)
	clrble	%r1, %r2, 3(%r4)
	clrbnh	%r1, %r2, 3(%r4)

#CHECK: clgrt     %r0, %r1, 12          # encoding: [0xb9,0x61,0xc0,0x01]
#CHECK: clgrt     %r0, %r1, 12          # encoding: [0xb9,0x61,0xc0,0x01]
#CHECK: clgrt     %r0, %r1, 12          # encoding: [0xb9,0x61,0xc0,0x01]
#CHECK: clgrt     %r0, %r1, 12          # encoding: [0xb9,0x61,0xc0,0x01]
#CHECK: clgrth    %r0, %r15             # encoding: [0xb9,0x61,0x20,0x0f]
#CHECK: clgrtl    %r0, %r15             # encoding: [0xb9,0x61,0x40,0x0f]
#CHECK: clgrte    %r0, %r15             # encoding: [0xb9,0x61,0x80,0x0f]
#CHECK: clgrtne   %r0, %r15             # encoding: [0xb9,0x61,0x60,0x0f]
#CHECK: clgrtnl   %r0, %r15             # encoding: [0xb9,0x61,0xa0,0x0f]
#CHECK: clgrtnh   %r0, %r15             # encoding: [0xb9,0x61,0xc0,0x0f]
#CHECK: clgrtnle  %r0, %r15             # encoding: [0xb9,0x61,0x20,0x0f]
#CHECK: clgrtnhe  %r0, %r15             # encoding: [0xb9,0x61,0x40,0x0f]
#CHECK: clgrtnlh  %r0, %r15             # encoding: [0xb9,0x61,0x80,0x0f]
#CHECK: clgrtlh   %r0, %r15             # encoding: [0xb9,0x61,0x60,0x0f]
#CHECK: clgrthe   %r0, %r15             # encoding: [0xb9,0x61,0xa0,0x0f]
#CHECK: clgrtle   %r0, %r15             # encoding: [0xb9,0x61,0xc0,0x0f]

        clgrt     %r0, %r1, 12
        clgrt     %r0, %r1, 12
        clgrt     %r0, %r1, 12
        clgrt     %r0, %r1, 12
        clgrth    %r0, %r15
        clgrtl    %r0, %r15
        clgrte    %r0, %r15
        clgrtne   %r0, %r15
        clgrtnl   %r0, %r15
        clgrtnh   %r0, %r15
        clgrtnle  %r0, %r15
        clgrtnhe  %r0, %r15
        clgrtnlh  %r0, %r15
        clgrtlh   %r0, %r15
        clgrthe   %r0, %r15
        clgrtle   %r0, %r15

#CHECK: clrj	%r0, %r0, 0, .[[LAB:L.*]]	# encoding: [0xec,0x00,A,A,0x00,0x77]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC16DBL
#CHECK: clrj	%r0, %r15, 0, .[[LAB:L.*]]	# encoding: [0xec,0x0f,A,A,0x00,0x77]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC16DBL
#CHECK: clrj	%r15, %r0, 0, .[[LAB:L.*]]	# encoding: [0xec,0xf0,A,A,0x00,0x77]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC16DBL
#CHECK: clrj	%r7, %r8, 0, .[[LAB:L.*]]	# encoding: [0xec,0x78,A,A,0x00,0x77]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC16DBL
	clrj	%r0,%r0,0,0
	clrj	%r0,%r15,0,0
	clrj	%r15,%r0,0,0
	clrj	%r7,%r8,0,0

#CHECK: clrj	%r1, %r2, 0, .[[LAB:L.*]]-65536	# encoding: [0xec,0x12,A,A,0x00,0x77]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-65536)+2, kind: FK_390_PC16DBL
	clrj	%r1, %r2, 0, -0x10000
#CHECK: clrj	%r1, %r2, 0, .[[LAB:L.*]]-2	# encoding: [0xec,0x12,A,A,0x00,0x77]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC16DBL
	clrj	%r1, %r2, 0, -2
#CHECK: clrj	%r1, %r2, 0, .[[LAB:L.*]]		# encoding: [0xec,0x12,A,A,0x00,0x77]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC16DBL
	clrj	%r1, %r2, 0, 0
#CHECK: clrj	%r1, %r2, 0, .[[LAB:L.*]]+65534	# encoding: [0xec,0x12,A,A,0x00,0x77]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+65534)+2, kind: FK_390_PC16DBL
	clrj	%r1, %r2, 0, 0xfffe

#CHECK: clrj	%r1, %r2, 0, foo                  # encoding: [0xec,0x12,A,A,0x00,0x77]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clrj	%r1, %r2, 0, foo

#CHECK: clrj	%r1, %r2, 1, foo                  # encoding: [0xec,0x12,A,A,0x10,0x77]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clrj	%r1, %r2, 1, foo

#CHECK: clrj	%r1, %r2, 2, foo                  # encoding: [0xec,0x12,A,A,0x20,0x77]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: clrjh	%r1, %r2, foo                     # encoding: [0xec,0x12,A,A,0x20,0x77]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: clrjnle	%r1, %r2, foo                     # encoding: [0xec,0x12,A,A,0x20,0x77]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clrj	%r1, %r2, 2, foo
	clrjh	%r1, %r2, foo
	clrjnle	%r1, %r2, foo

#CHECK: clrj	%r1, %r2, 3, foo                  # encoding: [0xec,0x12,A,A,0x30,0x77]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clrj	%r1, %r2, 3, foo

#CHECK: clrj	%r1, %r2, 4, foo                  # encoding: [0xec,0x12,A,A,0x40,0x77]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: clrjl	%r1, %r2, foo                     # encoding: [0xec,0x12,A,A,0x40,0x77]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: clrjnhe	%r1, %r2, foo                     # encoding: [0xec,0x12,A,A,0x40,0x77]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clrj	%r1, %r2, 4, foo
	clrjl	%r1, %r2, foo
	clrjnhe	%r1, %r2, foo

#CHECK: clrj	%r1, %r2, 5, foo                  # encoding: [0xec,0x12,A,A,0x50,0x77]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clrj	%r1, %r2, 5, foo

#CHECK: clrj	%r1, %r2, 6, foo                  # encoding: [0xec,0x12,A,A,0x60,0x77]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: clrjlh	%r1, %r2, foo                     # encoding: [0xec,0x12,A,A,0x60,0x77]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: clrjne	%r1, %r2, foo                     # encoding: [0xec,0x12,A,A,0x60,0x77]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clrj	%r1, %r2, 6, foo
	clrjlh	%r1, %r2, foo
	clrjne	%r1, %r2, foo

#CHECK: clrj	%r1, %r2, 7, foo                  # encoding: [0xec,0x12,A,A,0x70,0x77]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clrj	%r1, %r2, 7, foo

#CHECK: clrj	%r1, %r2, 8, foo                  # encoding: [0xec,0x12,A,A,0x80,0x77]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: clrje	%r1, %r2, foo                     # encoding: [0xec,0x12,A,A,0x80,0x77]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: clrjnlh	%r1, %r2, foo                     # encoding: [0xec,0x12,A,A,0x80,0x77]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clrj	%r1, %r2, 8, foo
	clrje	%r1, %r2, foo
	clrjnlh	%r1, %r2, foo

#CHECK: clrj	%r1, %r2, 9, foo                  # encoding: [0xec,0x12,A,A,0x90,0x77]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clrj	%r1, %r2, 9, foo

#CHECK: clrj	%r1, %r2, 10, foo                 # encoding: [0xec,0x12,A,A,0xa0,0x77]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: clrjhe	%r1, %r2, foo                     # encoding: [0xec,0x12,A,A,0xa0,0x77]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: clrjnl	%r1, %r2, foo                     # encoding: [0xec,0x12,A,A,0xa0,0x77]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clrj	%r1, %r2, 10, foo
	clrjhe	%r1, %r2, foo
	clrjnl	%r1, %r2, foo

#CHECK: clrj	%r1, %r2, 11, foo                 # encoding: [0xec,0x12,A,A,0xb0,0x77]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clrj	%r1, %r2, 11, foo

#CHECK: clrj	%r1, %r2, 12, foo                 # encoding: [0xec,0x12,A,A,0xc0,0x77]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: clrjle	%r1, %r2, foo                     # encoding: [0xec,0x12,A,A,0xc0,0x77]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: clrjnh	%r1, %r2, foo                     # encoding: [0xec,0x12,A,A,0xc0,0x77]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clrj	%r1, %r2, 12, foo
	clrjle	%r1, %r2, foo
	clrjnh	%r1, %r2, foo

#CHECK: clrj	%r1, %r2, 13, foo                 # encoding: [0xec,0x12,A,A,0xd0,0x77]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clrj	%r1, %r2, 13, foo

#CHECK: clrj	%r1, %r2, 14, foo                 # encoding: [0xec,0x12,A,A,0xe0,0x77]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clrj	%r1, %r2, 14, foo

#CHECK: clrj	%r1, %r2, 15, foo                 # encoding: [0xec,0x12,A,A,0xf0,0x77]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	clrj	%r1, %r2, 15, foo

#CHECK: clrj	%r1, %r2, 0, bar+100              # encoding: [0xec,0x12,A,A,0x00,0x77]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	clrj	%r1, %r2, 0, bar+100

#CHECK: clrjh	%r1, %r2, bar+100                 # encoding: [0xec,0x12,A,A,0x20,0x77]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	clrjh	%r1, %r2, bar+100

#CHECK: clrjnle	%r1, %r2, bar+100                 # encoding: [0xec,0x12,A,A,0x20,0x77]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	clrjnle	%r1, %r2, bar+100

#CHECK: clrjl	%r1, %r2, bar+100                 # encoding: [0xec,0x12,A,A,0x40,0x77]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	clrjl	%r1, %r2, bar+100

#CHECK: clrjnhe	%r1, %r2, bar+100                 # encoding: [0xec,0x12,A,A,0x40,0x77]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	clrjnhe	%r1, %r2, bar+100

#CHECK: clrjlh	%r1, %r2, bar+100                 # encoding: [0xec,0x12,A,A,0x60,0x77]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	clrjlh	%r1, %r2, bar+100

#CHECK: clrjne	%r1, %r2, bar+100                 # encoding: [0xec,0x12,A,A,0x60,0x77]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	clrjne	%r1, %r2, bar+100

#CHECK: clrje	%r1, %r2, bar+100                 # encoding: [0xec,0x12,A,A,0x80,0x77]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	clrje	%r1, %r2, bar+100

#CHECK: clrjnlh	%r1, %r2, bar+100                 # encoding: [0xec,0x12,A,A,0x80,0x77]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	clrjnlh	%r1, %r2, bar+100

#CHECK: clrjhe	%r1, %r2, bar+100                 # encoding: [0xec,0x12,A,A,0xa0,0x77]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	clrjhe	%r1, %r2, bar+100

#CHECK: clrjnl	%r1, %r2, bar+100                 # encoding: [0xec,0x12,A,A,0xa0,0x77]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	clrjnl	%r1, %r2, bar+100

#CHECK: clrjle	%r1, %r2, bar+100                 # encoding: [0xec,0x12,A,A,0xc0,0x77]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	clrjle	%r1, %r2, bar+100

#CHECK: clrjnh	%r1, %r2, bar+100                 # encoding: [0xec,0x12,A,A,0xc0,0x77]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	clrjnh	%r1, %r2, bar+100

#CHECK: clrj	%r1, %r2, 0, bar@PLT              # encoding: [0xec,0x12,A,A,0x00,0x77]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	clrj	%r1, %r2, 0, bar@PLT

#CHECK: clrjh	%r1, %r2, bar@PLT                 # encoding: [0xec,0x12,A,A,0x20,0x77]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	clrjh	%r1, %r2, bar@PLT

#CHECK: clrjnle	%r1, %r2, bar@PLT                 # encoding: [0xec,0x12,A,A,0x20,0x77]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	clrjnle	%r1, %r2, bar@PLT

#CHECK: clrjl	%r1, %r2, bar@PLT                 # encoding: [0xec,0x12,A,A,0x40,0x77]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	clrjl	%r1, %r2, bar@PLT

#CHECK: clrjnhe	%r1, %r2, bar@PLT                 # encoding: [0xec,0x12,A,A,0x40,0x77]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	clrjnhe	%r1, %r2, bar@PLT

#CHECK: clrjlh	%r1, %r2, bar@PLT                 # encoding: [0xec,0x12,A,A,0x60,0x77]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	clrjlh	%r1, %r2, bar@PLT

#CHECK: clrjne	%r1, %r2, bar@PLT                 # encoding: [0xec,0x12,A,A,0x60,0x77]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	clrjne	%r1, %r2, bar@PLT

#CHECK: clrje	%r1, %r2, bar@PLT                 # encoding: [0xec,0x12,A,A,0x80,0x77]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	clrje	%r1, %r2, bar@PLT

#CHECK: clrjnlh	%r1, %r2, bar@PLT                 # encoding: [0xec,0x12,A,A,0x80,0x77]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	clrjnlh	%r1, %r2, bar@PLT

#CHECK: clrjhe	%r1, %r2, bar@PLT                 # encoding: [0xec,0x12,A,A,0xa0,0x77]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	clrjhe	%r1, %r2, bar@PLT

#CHECK: clrjnl	%r1, %r2, bar@PLT                 # encoding: [0xec,0x12,A,A,0xa0,0x77]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	clrjnl	%r1, %r2, bar@PLT

#CHECK: clrjle	%r1, %r2, bar@PLT                 # encoding: [0xec,0x12,A,A,0xc0,0x77]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	clrjle	%r1, %r2, bar@PLT

#CHECK: clrjnh	%r1, %r2, bar@PLT                 # encoding: [0xec,0x12,A,A,0xc0,0x77]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	clrjnh	%r1, %r2, bar@PLT

#CHECK: clrl	%r0, .[[LAB:L.*]]-4294967296 # encoding: [0xc6,0x0f,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-4294967296)+2, kind: FK_390_PC32DBL
	clrl	%r0, -0x100000000
#CHECK: clrl	%r0, .[[LAB:L.*]]-2	# encoding: [0xc6,0x0f,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC32DBL
	clrl	%r0, -2
#CHECK: clrl	%r0, .[[LAB:L.*]]	# encoding: [0xc6,0x0f,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC32DBL
	clrl	%r0, 0
#CHECK: clrl	%r0, .[[LAB:L.*]]+4294967294 # encoding: [0xc6,0x0f,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+4294967294)+2, kind: FK_390_PC32DBL
	clrl	%r0, 0xfffffffe

#CHECK: clrl	%r0, foo                # encoding: [0xc6,0x0f,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: clrl	%r15, foo               # encoding: [0xc6,0xff,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL

	clrl	%r0,foo
	clrl	%r15,foo

#CHECK: clrl	%r3, bar+100            # encoding: [0xc6,0x3f,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: clrl	%r4, bar+100            # encoding: [0xc6,0x4f,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL

	clrl	%r3,bar+100
	clrl	%r4,bar+100

#CHECK: clrl	%r7, frob@PLT           # encoding: [0xc6,0x7f,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL
#CHECK: clrl	%r8, frob@PLT           # encoding: [0xc6,0x8f,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL

	clrl	%r7,frob@PLT
	clrl	%r8,frob@PLT

#CHECK: clrt     %r0, %r1, 12           # encoding: [0xb9,0x73,0xc0,0x01]
#CHECK: clrt     %r0, %r1, 12           # encoding: [0xb9,0x73,0xc0,0x01]
#CHECK: clrt     %r0, %r1, 12           # encoding: [0xb9,0x73,0xc0,0x01]
#CHECK: clrt     %r0, %r1, 12           # encoding: [0xb9,0x73,0xc0,0x01]
#CHECK: clrth    %r0, %r15              # encoding: [0xb9,0x73,0x20,0x0f]
#CHECK: clrtl    %r0, %r15              # encoding: [0xb9,0x73,0x40,0x0f]
#CHECK: clrte    %r0, %r15              # encoding: [0xb9,0x73,0x80,0x0f]
#CHECK: clrtne   %r0, %r15              # encoding: [0xb9,0x73,0x60,0x0f]
#CHECK: clrtnl   %r0, %r15              # encoding: [0xb9,0x73,0xa0,0x0f]
#CHECK: clrtnh   %r0, %r15              # encoding: [0xb9,0x73,0xc0,0x0f]
#CHECK: clrtnle  %r0, %r15              # encoding: [0xb9,0x73,0x20,0x0f]
#CHECK: clrtnhe  %r0, %r15              # encoding: [0xb9,0x73,0x40,0x0f]
#CHECK: clrtnlh  %r0, %r15              # encoding: [0xb9,0x73,0x80,0x0f]
#CHECK: clrtlh   %r0, %r15              # encoding: [0xb9,0x73,0x60,0x0f]
#CHECK: clrthe   %r0, %r15              # encoding: [0xb9,0x73,0xa0,0x0f]
#CHECK: clrtle   %r0, %r15              # encoding: [0xb9,0x73,0xc0,0x0f]

        clrt     %r0, %r1, 12
        clrt     %r0, %r1, 12
        clrt     %r0, %r1, 12
        clrt     %r0, %r1, 12
        clrth    %r0, %r15
        clrtl    %r0, %r15
        clrte    %r0, %r15
        clrtne   %r0, %r15
        clrtnl   %r0, %r15
        clrtnh   %r0, %r15
        clrtnle  %r0, %r15
        clrtnhe  %r0, %r15
        clrtnlh  %r0, %r15
        clrtlh   %r0, %r15
        clrthe   %r0, %r15
        clrtle   %r0, %r15

#CHECK: clst	%r0, %r0                # encoding: [0xb2,0x5d,0x00,0x00]
#CHECK: clst	%r0, %r15               # encoding: [0xb2,0x5d,0x00,0x0f]
#CHECK: clst	%r15, %r0               # encoding: [0xb2,0x5d,0x00,0xf0]
#CHECK: clst	%r7, %r8                # encoding: [0xb2,0x5d,0x00,0x78]

	clst	%r0,%r0
	clst	%r0,%r15
	clst	%r15,%r0
	clst	%r7,%r8

#CHECK: cly	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x55]
#CHECK: cly	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x55]
#CHECK: cly	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x55]
#CHECK: cly	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x55]
#CHECK: cly	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x55]
#CHECK: cly	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x55]
#CHECK: cly	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x55]
#CHECK: cly	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x55]
#CHECK: cly	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x55]
#CHECK: cly	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x55]

	cly	%r0, -524288
	cly	%r0, -1
	cly	%r0, 0
	cly	%r0, 1
	cly	%r0, 524287
	cly	%r0, 0(%r1)
	cly	%r0, 0(%r15)
	cly	%r0, 524287(%r1,%r15)
	cly	%r0, 524287(%r15,%r1)
	cly	%r15, 0

#CHECK: cmpsc	%r0, %r8                # encoding: [0xb2,0x63,0x00,0x08]
#CHECK: cmpsc	%r0, %r14               # encoding: [0xb2,0x63,0x00,0x0e]
#CHECK: cmpsc	%r14, %r0               # encoding: [0xb2,0x63,0x00,0xe0]
#CHECK: cmpsc	%r14, %r8               # encoding: [0xb2,0x63,0x00,0xe8]

	cmpsc	%r0, %r8
	cmpsc	%r0, %r14
	cmpsc	%r14, %r0
	cmpsc	%r14, %r8

#CHECK: cp	0(1), 0(1)              # encoding: [0xf9,0x00,0x00,0x00,0x00,0x00]
#CHECK: cp	0(1), 0(1,%r1)          # encoding: [0xf9,0x00,0x00,0x00,0x10,0x00]
#CHECK: cp	0(1), 0(1,%r15)         # encoding: [0xf9,0x00,0x00,0x00,0xf0,0x00]
#CHECK: cp	0(1), 4095(1)           # encoding: [0xf9,0x00,0x00,0x00,0x0f,0xff]
#CHECK: cp	0(1), 4095(1,%r1)       # encoding: [0xf9,0x00,0x00,0x00,0x1f,0xff]
#CHECK: cp	0(1), 4095(1,%r15)      # encoding: [0xf9,0x00,0x00,0x00,0xff,0xff]
#CHECK: cp	0(1,%r1), 0(1)          # encoding: [0xf9,0x00,0x10,0x00,0x00,0x00]
#CHECK: cp	0(1,%r15), 0(1)         # encoding: [0xf9,0x00,0xf0,0x00,0x00,0x00]
#CHECK: cp	4095(1,%r1), 0(1)       # encoding: [0xf9,0x00,0x1f,0xff,0x00,0x00]
#CHECK: cp	4095(1,%r15), 0(1)      # encoding: [0xf9,0x00,0xff,0xff,0x00,0x00]
#CHECK: cp	0(16,%r1), 0(1)         # encoding: [0xf9,0xf0,0x10,0x00,0x00,0x00]
#CHECK: cp	0(16,%r15), 0(1)        # encoding: [0xf9,0xf0,0xf0,0x00,0x00,0x00]
#CHECK: cp	0(1), 0(16,%r1)         # encoding: [0xf9,0x0f,0x00,0x00,0x10,0x00]
#CHECK: cp	0(1), 0(16,%r15)        # encoding: [0xf9,0x0f,0x00,0x00,0xf0,0x00]

	cp	0(1), 0(1)
	cp	0(1), 0(1,%r1)
	cp	0(1), 0(1,%r15)
	cp	0(1), 4095(1)
	cp	0(1), 4095(1,%r1)
	cp	0(1), 4095(1,%r15)
	cp	0(1,%r1), 0(1)
	cp	0(1,%r15), 0(1)
	cp	4095(1,%r1), 0(1)
	cp	4095(1,%r15), 0(1)
	cp	0(16,%r1), 0(1)
	cp	0(16,%r15), 0(1)
	cp	0(1), 0(16,%r1)
	cp	0(1), 0(16,%r15)

#CHECK: cpsdr	%f0, %f0, %f0           # encoding: [0xb3,0x72,0x00,0x00]
#CHECK: cpsdr	%f0, %f0, %f15          # encoding: [0xb3,0x72,0x00,0x0f]
#CHECK: cpsdr	%f0, %f15, %f0          # encoding: [0xb3,0x72,0xf0,0x00]
#CHECK: cpsdr	%f15, %f0, %f0          # encoding: [0xb3,0x72,0x00,0xf0]
#CHECK: cpsdr	%f1, %f2, %f3           # encoding: [0xb3,0x72,0x20,0x13]
#CHECK: cpsdr	%f15, %f15, %f15        # encoding: [0xb3,0x72,0xf0,0xff]

	cpsdr	%f0, %f0, %f0
	cpsdr	%f0, %f0, %f15
	cpsdr	%f0, %f15, %f0
	cpsdr	%f15, %f0, %f0
	cpsdr	%f1, %f2, %f3
	cpsdr	%f15, %f15, %f15

#CHECK: cpya	%a0, %a0                # encoding: [0xb2,0x4d,0x00,0x00]
#CHECK: cpya	%a0, %a15               # encoding: [0xb2,0x4d,0x00,0x0f]
#CHECK: cpya	%a15, %a0               # encoding: [0xb2,0x4d,0x00,0xf0]
#CHECK: cpya	%a7, %a8                # encoding: [0xb2,0x4d,0x00,0x78]
#CHECK: cpya	%a15, %a15              # encoding: [0xb2,0x4d,0x00,0xff]

	cpya	%a0, %a0
	cpya	%a0, %a15
	cpya	%a15, %a0
	cpya	%a7, %a8
	cpya	%a15, %a15

#CHECK: cr	%r0, %r0                # encoding: [0x19,0x00]
#CHECK: cr	%r0, %r15               # encoding: [0x19,0x0f]
#CHECK: cr	%r15, %r0               # encoding: [0x19,0xf0]
#CHECK: cr	%r7, %r8                # encoding: [0x19,0x78]

	cr	%r0,%r0
	cr	%r0,%r15
	cr	%r15,%r0
	cr	%r7,%r8

#CHECK: crb	%r0, %r0, 0, 0          # encoding: [0xec,0x00,0x00,0x00,0x00,0xf6]
#CHECK: crb	%r0, %r15, 0, 0         # encoding: [0xec,0x0f,0x00,0x00,0x00,0xf6]
#CHECK: crb	%r15, %r0, 0, 0         # encoding: [0xec,0xf0,0x00,0x00,0x00,0xf6]
#CHECK: crb	%r7, %r2, 0, 0          # encoding: [0xec,0x72,0x00,0x00,0x00,0xf6]
#CHECK: crb	%r0, %r0, 1, 0          # encoding: [0xec,0x00,0x00,0x00,0x10,0xf6]
#CHECK: crb	%r0, %r0, 15, 0         # encoding: [0xec,0x00,0x00,0x00,0xf0,0xf6]
#CHECK: crb	%r0, %r0, 0, 0(%r13)    # encoding: [0xec,0x00,0xd0,0x00,0x00,0xf6]
#CHECK: crb	%r0, %r0, 0, 4095       # encoding: [0xec,0x00,0x0f,0xff,0x00,0xf6]
#CHECK: crb	%r0, %r0, 0, 4095(%r7)  # encoding: [0xec,0x00,0x7f,0xff,0x00,0xf6]
	crb	%r0, %r0, 0, 0
	crb	%r0, %r15, 0, 0
	crb	%r15, %r0, 0, 0
	crb	%r7, %r2, 0, 0
	crb	%r0, %r0, 1, 0
	crb	%r0, %r0, 15, 0
	crb	%r0, %r0, 0, 0(%r13)
	crb	%r0, %r0, 0, 4095
	crb	%r0, %r0, 0, 4095(%r7)

#CHECK: crbe	%r0, %r0, 0             # encoding: [0xec,0x00,0x00,0x00,0x80,0xf6]
#CHECK: crbe	%r0, %r15, 0            # encoding: [0xec,0x0f,0x00,0x00,0x80,0xf6]
#CHECK: crbe	%r15, %r0, 0            # encoding: [0xec,0xf0,0x00,0x00,0x80,0xf6]
#CHECK: crbe	%r7, %r2, 0             # encoding: [0xec,0x72,0x00,0x00,0x80,0xf6]
#CHECK: crbe	%r0, %r0, 0(%r13)       # encoding: [0xec,0x00,0xd0,0x00,0x80,0xf6]
#CHECK: crbe	%r0, %r0, 4095          # encoding: [0xec,0x00,0x0f,0xff,0x80,0xf6]
#CHECK: crbe	%r0, %r0, 4095(%r7)     # encoding: [0xec,0x00,0x7f,0xff,0x80,0xf6]
	crbe	%r0, %r0, 0
	crbe	%r0, %r15, 0
	crbe	%r15, %r0, 0
	crbe	%r7, %r2, 0
	crbe	%r0, %r0, 0(%r13)
	crbe	%r0, %r0, 4095
	crbe	%r0, %r0, 4095(%r7)

#CHECK: crb	%r1, %r2, 2, 3(%r4)     # encoding: [0xec,0x12,0x40,0x03,0x20,0xf6]
#CHECK: crbh	%r1, %r2, 3(%r4)        # encoding: [0xec,0x12,0x40,0x03,0x20,0xf6]
#CHECK: crbnle	%r1, %r2, 3(%r4)        # encoding: [0xec,0x12,0x40,0x03,0x20,0xf6]
	crb	%r1, %r2, 2, 3(%r4)
	crbh	%r1, %r2, 3(%r4)
	crbnle	%r1, %r2, 3(%r4)

#CHECK: crb	%r1, %r2, 4, 3(%r4)     # encoding: [0xec,0x12,0x40,0x03,0x40,0xf6]
#CHECK: crbl	%r1, %r2, 3(%r4)        # encoding: [0xec,0x12,0x40,0x03,0x40,0xf6]
#CHECK: crbnhe	%r1, %r2, 3(%r4)        # encoding: [0xec,0x12,0x40,0x03,0x40,0xf6]
	crb	%r1, %r2, 4, 3(%r4)
	crbl	%r1, %r2, 3(%r4)
	crbnhe	%r1, %r2, 3(%r4)

#CHECK: crb	%r1, %r2, 6, 3(%r4)     # encoding: [0xec,0x12,0x40,0x03,0x60,0xf6]
#CHECK: crblh	%r1, %r2, 3(%r4)        # encoding: [0xec,0x12,0x40,0x03,0x60,0xf6]
#CHECK: crbne	%r1, %r2, 3(%r4)        # encoding: [0xec,0x12,0x40,0x03,0x60,0xf6]
	crb	%r1, %r2, 6, 3(%r4)
	crblh	%r1, %r2, 3(%r4)
	crbne	%r1, %r2, 3(%r4)

#CHECK: crb	%r1, %r2, 8, 3(%r4)     # encoding: [0xec,0x12,0x40,0x03,0x80,0xf6]
#CHECK: crbe	%r1, %r2, 3(%r4)        # encoding: [0xec,0x12,0x40,0x03,0x80,0xf6]
#CHECK: crbnlh	%r1, %r2, 3(%r4)        # encoding: [0xec,0x12,0x40,0x03,0x80,0xf6]
	crb	%r1, %r2, 8, 3(%r4)
	crbe	%r1, %r2, 3(%r4)
	crbnlh	%r1, %r2, 3(%r4)

#CHECK: crb	%r1, %r2, 10, 3(%r4)    # encoding: [0xec,0x12,0x40,0x03,0xa0,0xf6]
#CHECK: crbhe	%r1, %r2, 3(%r4)        # encoding: [0xec,0x12,0x40,0x03,0xa0,0xf6]
#CHECK: crbnl	%r1, %r2, 3(%r4)        # encoding: [0xec,0x12,0x40,0x03,0xa0,0xf6]
	crb	%r1, %r2, 10, 3(%r4)
	crbhe	%r1, %r2, 3(%r4)
	crbnl	%r1, %r2, 3(%r4)

#CHECK: crb	%r1, %r2, 12, 3(%r4)    # encoding: [0xec,0x12,0x40,0x03,0xc0,0xf6]
#CHECK: crble	%r1, %r2, 3(%r4)        # encoding: [0xec,0x12,0x40,0x03,0xc0,0xf6]
#CHECK: crbnh	%r1, %r2, 3(%r4)        # encoding: [0xec,0x12,0x40,0x03,0xc0,0xf6]
	crb	%r1, %r2, 12, 3(%r4)
	crble	%r1, %r2, 3(%r4)
	crbnh	%r1, %r2, 3(%r4)

#CHECK: crj	%r0, %r0, 0, .[[LAB:L.*]]	# encoding: [0xec,0x00,A,A,0x00,0x76]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC16DBL
#CHECK: crj	%r0, %r15, 0, .[[LAB:L.*]]	# encoding: [0xec,0x0f,A,A,0x00,0x76]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC16DBL
#CHECK: crj	%r15, %r0, 0, .[[LAB:L.*]]	# encoding: [0xec,0xf0,A,A,0x00,0x76]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC16DBL
#CHECK: crj	%r7, %r8, 0, .[[LAB:L.*]]	# encoding: [0xec,0x78,A,A,0x00,0x76]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC16DBL
	crj	%r0,%r0,0,0
	crj	%r0,%r15,0,0
	crj	%r15,%r0,0,0
	crj	%r7,%r8,0,0

#CHECK: crj	%r1, %r2, 0, .[[LAB:L.*]]-65536	# encoding: [0xec,0x12,A,A,0x00,0x76]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-65536)+2, kind: FK_390_PC16DBL
	crj	%r1, %r2, 0, -0x10000
#CHECK: crj	%r1, %r2, 0, .[[LAB:L.*]]-2	# encoding: [0xec,0x12,A,A,0x00,0x76]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC16DBL
	crj	%r1, %r2, 0, -2
#CHECK: crj	%r1, %r2, 0, .[[LAB:L.*]]		# encoding: [0xec,0x12,A,A,0x00,0x76]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC16DBL
	crj	%r1, %r2, 0, 0
#CHECK: crj	%r1, %r2, 0, .[[LAB:L.*]]+65534	# encoding: [0xec,0x12,A,A,0x00,0x76]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+65534)+2, kind: FK_390_PC16DBL
	crj	%r1, %r2, 0, 0xfffe

#CHECK: crj	%r1, %r2, 0, foo                  # encoding: [0xec,0x12,A,A,0x00,0x76]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	crj	%r1, %r2, 0, foo

#CHECK: crj	%r1, %r2, 1, foo                  # encoding: [0xec,0x12,A,A,0x10,0x76]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	crj	%r1, %r2, 1, foo

#CHECK: crj	%r1, %r2, 2, foo                  # encoding: [0xec,0x12,A,A,0x20,0x76]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: crjh	%r1, %r2, foo                     # encoding: [0xec,0x12,A,A,0x20,0x76]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: crjnle	%r1, %r2, foo                     # encoding: [0xec,0x12,A,A,0x20,0x76]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	crj	%r1, %r2, 2, foo
	crjh	%r1, %r2, foo
	crjnle	%r1, %r2, foo

#CHECK: crj	%r1, %r2, 3, foo                  # encoding: [0xec,0x12,A,A,0x30,0x76]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	crj	%r1, %r2, 3, foo

#CHECK: crj	%r1, %r2, 4, foo                  # encoding: [0xec,0x12,A,A,0x40,0x76]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: crjl	%r1, %r2, foo                     # encoding: [0xec,0x12,A,A,0x40,0x76]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: crjnhe	%r1, %r2, foo                     # encoding: [0xec,0x12,A,A,0x40,0x76]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	crj	%r1, %r2, 4, foo
	crjl	%r1, %r2, foo
	crjnhe	%r1, %r2, foo

#CHECK: crj	%r1, %r2, 5, foo                  # encoding: [0xec,0x12,A,A,0x50,0x76]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	crj	%r1, %r2, 5, foo

#CHECK: crj	%r1, %r2, 6, foo                  # encoding: [0xec,0x12,A,A,0x60,0x76]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: crjlh	%r1, %r2, foo                     # encoding: [0xec,0x12,A,A,0x60,0x76]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: crjne	%r1, %r2, foo                     # encoding: [0xec,0x12,A,A,0x60,0x76]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	crj	%r1, %r2, 6, foo
	crjlh	%r1, %r2, foo
	crjne	%r1, %r2, foo

#CHECK: crj	%r1, %r2, 7, foo                  # encoding: [0xec,0x12,A,A,0x70,0x76]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	crj	%r1, %r2, 7, foo

#CHECK: crj	%r1, %r2, 8, foo                  # encoding: [0xec,0x12,A,A,0x80,0x76]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: crje	%r1, %r2, foo                     # encoding: [0xec,0x12,A,A,0x80,0x76]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: crjnlh	%r1, %r2, foo                     # encoding: [0xec,0x12,A,A,0x80,0x76]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	crj	%r1, %r2, 8, foo
	crje	%r1, %r2, foo
	crjnlh	%r1, %r2, foo

#CHECK: crj	%r1, %r2, 9, foo                  # encoding: [0xec,0x12,A,A,0x90,0x76]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	crj	%r1, %r2, 9, foo

#CHECK: crj	%r1, %r2, 10, foo                 # encoding: [0xec,0x12,A,A,0xa0,0x76]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: crjhe	%r1, %r2, foo                     # encoding: [0xec,0x12,A,A,0xa0,0x76]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: crjnl	%r1, %r2, foo                     # encoding: [0xec,0x12,A,A,0xa0,0x76]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	crj	%r1, %r2, 10, foo
	crjhe	%r1, %r2, foo
	crjnl	%r1, %r2, foo

#CHECK: crj	%r1, %r2, 11, foo                 # encoding: [0xec,0x12,A,A,0xb0,0x76]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	crj	%r1, %r2, 11, foo

#CHECK: crj	%r1, %r2, 12, foo                 # encoding: [0xec,0x12,A,A,0xc0,0x76]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: crjle	%r1, %r2, foo                     # encoding: [0xec,0x12,A,A,0xc0,0x76]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
#CHECK: crjnh	%r1, %r2, foo                     # encoding: [0xec,0x12,A,A,0xc0,0x76]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	crj	%r1, %r2, 12, foo
	crjle	%r1, %r2, foo
	crjnh	%r1, %r2, foo

#CHECK: crj	%r1, %r2, 13, foo                 # encoding: [0xec,0x12,A,A,0xd0,0x76]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	crj	%r1, %r2, 13, foo

#CHECK: crj	%r1, %r2, 14, foo                 # encoding: [0xec,0x12,A,A,0xe0,0x76]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	crj	%r1, %r2, 14, foo

#CHECK: crj	%r1, %r2, 15, foo                 # encoding: [0xec,0x12,A,A,0xf0,0x76]
#CHECK:  fixup A - offset: 2, value: foo+2, kind: FK_390_PC16DBL
	crj	%r1, %r2, 15, foo

#CHECK: crj	%r1, %r2, 0, bar+100              # encoding: [0xec,0x12,A,A,0x00,0x76]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	crj	%r1, %r2, 0, bar+100

#CHECK: crjh	%r1, %r2, bar+100                 # encoding: [0xec,0x12,A,A,0x20,0x76]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	crjh	%r1, %r2, bar+100

#CHECK: crjnle	%r1, %r2, bar+100                 # encoding: [0xec,0x12,A,A,0x20,0x76]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	crjnle	%r1, %r2, bar+100

#CHECK: crjl	%r1, %r2, bar+100                 # encoding: [0xec,0x12,A,A,0x40,0x76]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	crjl	%r1, %r2, bar+100

#CHECK: crjnhe	%r1, %r2, bar+100                 # encoding: [0xec,0x12,A,A,0x40,0x76]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	crjnhe	%r1, %r2, bar+100

#CHECK: crjlh	%r1, %r2, bar+100                 # encoding: [0xec,0x12,A,A,0x60,0x76]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	crjlh	%r1, %r2, bar+100

#CHECK: crjne	%r1, %r2, bar+100                 # encoding: [0xec,0x12,A,A,0x60,0x76]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	crjne	%r1, %r2, bar+100

#CHECK: crje	%r1, %r2, bar+100                 # encoding: [0xec,0x12,A,A,0x80,0x76]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	crje	%r1, %r2, bar+100

#CHECK: crjnlh	%r1, %r2, bar+100                 # encoding: [0xec,0x12,A,A,0x80,0x76]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	crjnlh	%r1, %r2, bar+100

#CHECK: crjhe	%r1, %r2, bar+100                 # encoding: [0xec,0x12,A,A,0xa0,0x76]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	crjhe	%r1, %r2, bar+100

#CHECK: crjnl	%r1, %r2, bar+100                 # encoding: [0xec,0x12,A,A,0xa0,0x76]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	crjnl	%r1, %r2, bar+100

#CHECK: crjle	%r1, %r2, bar+100                 # encoding: [0xec,0x12,A,A,0xc0,0x76]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	crjle	%r1, %r2, bar+100

#CHECK: crjnh	%r1, %r2, bar+100                 # encoding: [0xec,0x12,A,A,0xc0,0x76]
#CHECK:  fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC16DBL
	crjnh	%r1, %r2, bar+100

#CHECK: crj	%r1, %r2, 0, bar@PLT              # encoding: [0xec,0x12,A,A,0x00,0x76]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	crj	%r1, %r2, 0, bar@PLT

#CHECK: crjh	%r1, %r2, bar@PLT                 # encoding: [0xec,0x12,A,A,0x20,0x76]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	crjh	%r1, %r2, bar@PLT

#CHECK: crjnle	%r1, %r2, bar@PLT                 # encoding: [0xec,0x12,A,A,0x20,0x76]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	crjnle	%r1, %r2, bar@PLT

#CHECK: crjl	%r1, %r2, bar@PLT                 # encoding: [0xec,0x12,A,A,0x40,0x76]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	crjl	%r1, %r2, bar@PLT

#CHECK: crjnhe	%r1, %r2, bar@PLT                 # encoding: [0xec,0x12,A,A,0x40,0x76]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	crjnhe	%r1, %r2, bar@PLT

#CHECK: crjlh	%r1, %r2, bar@PLT                 # encoding: [0xec,0x12,A,A,0x60,0x76]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	crjlh	%r1, %r2, bar@PLT

#CHECK: crjne	%r1, %r2, bar@PLT                 # encoding: [0xec,0x12,A,A,0x60,0x76]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	crjne	%r1, %r2, bar@PLT

#CHECK: crje	%r1, %r2, bar@PLT                 # encoding: [0xec,0x12,A,A,0x80,0x76]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	crje	%r1, %r2, bar@PLT

#CHECK: crjnlh	%r1, %r2, bar@PLT                 # encoding: [0xec,0x12,A,A,0x80,0x76]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	crjnlh	%r1, %r2, bar@PLT

#CHECK: crjhe	%r1, %r2, bar@PLT                 # encoding: [0xec,0x12,A,A,0xa0,0x76]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	crjhe	%r1, %r2, bar@PLT

#CHECK: crjnl	%r1, %r2, bar@PLT                 # encoding: [0xec,0x12,A,A,0xa0,0x76]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	crjnl	%r1, %r2, bar@PLT

#CHECK: crjle	%r1, %r2, bar@PLT                 # encoding: [0xec,0x12,A,A,0xc0,0x76]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	crjle	%r1, %r2, bar@PLT

#CHECK: crjnh	%r1, %r2, bar@PLT                 # encoding: [0xec,0x12,A,A,0xc0,0x76]
#CHECK:  fixup A - offset: 2, value: bar@PLT+2, kind: FK_390_PC16DBL
	crjnh	%r1, %r2, bar@PLT

#CHECK: crl	%r0, .[[LAB:L.*]]-4294967296 # encoding: [0xc6,0x0d,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-4294967296)+2, kind: FK_390_PC32DBL
	crl	%r0, -0x100000000
#CHECK: crl	%r0, .[[LAB:L.*]]-2	# encoding: [0xc6,0x0d,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC32DBL
	crl	%r0, -2
#CHECK: crl	%r0, .[[LAB:L.*]]	# encoding: [0xc6,0x0d,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC32DBL
	crl	%r0, 0
#CHECK: crl	%r0, .[[LAB:L.*]]+4294967294 # encoding: [0xc6,0x0d,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+4294967294)+2, kind: FK_390_PC32DBL
	crl	%r0, 0xfffffffe

#CHECK: crl	%r0, foo                # encoding: [0xc6,0x0d,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: crl	%r15, foo               # encoding: [0xc6,0xfd,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL

	crl	%r0,foo
	crl	%r15,foo

#CHECK: crl	%r3, bar+100            # encoding: [0xc6,0x3d,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: crl	%r4, bar+100            # encoding: [0xc6,0x4d,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL

	crl	%r3,bar+100
	crl	%r4,bar+100

#CHECK: crl	%r7, frob@PLT           # encoding: [0xc6,0x7d,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL
#CHECK: crl	%r8, frob@PLT           # encoding: [0xc6,0x8d,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL

	crl	%r7,frob@PLT
	crl	%r8,frob@PLT

#CHECK: crt     %r0, %r1, 12            # encoding: [0xb9,0x72,0xc0,0x01]
#CHECK: crt     %r0, %r1, 12            # encoding: [0xb9,0x72,0xc0,0x01]
#CHECK: crt     %r0, %r1, 12            # encoding: [0xb9,0x72,0xc0,0x01]
#CHECK: crt     %r0, %r1, 12            # encoding: [0xb9,0x72,0xc0,0x01]
#CHECK: crth    %r0, %r15               # encoding: [0xb9,0x72,0x20,0x0f]
#CHECK: crtl    %r0, %r15               # encoding: [0xb9,0x72,0x40,0x0f]
#CHECK: crte    %r0, %r15               # encoding: [0xb9,0x72,0x80,0x0f]
#CHECK: crtne   %r0, %r15               # encoding: [0xb9,0x72,0x60,0x0f]
#CHECK: crtnl   %r0, %r15               # encoding: [0xb9,0x72,0xa0,0x0f]
#CHECK: crtnh   %r0, %r15               # encoding: [0xb9,0x72,0xc0,0x0f]
#CHECK: crtnle  %r0, %r15               # encoding: [0xb9,0x72,0x20,0x0f]
#CHECK: crtnhe  %r0, %r15               # encoding: [0xb9,0x72,0x40,0x0f]
#CHECK: crtnlh  %r0, %r15               # encoding: [0xb9,0x72,0x80,0x0f]
#CHECK: crtlh   %r0, %r15               # encoding: [0xb9,0x72,0x60,0x0f]
#CHECK: crthe   %r0, %r15               # encoding: [0xb9,0x72,0xa0,0x0f]
#CHECK: crtle   %r0, %r15               # encoding: [0xb9,0x72,0xc0,0x0f]

        crt     %r0, %r1, 12
        crt     %r0, %r1, 12
        crt     %r0, %r1, 12
        crt     %r0, %r1, 12
        crth    %r0, %r15
        crtl    %r0, %r15
        crte    %r0, %r15
        crtne   %r0, %r15
        crtnl   %r0, %r15
        crtnh   %r0, %r15
        crtnle  %r0, %r15
        crtnhe  %r0, %r15
        crtnlh  %r0, %r15
        crtlh   %r0, %r15
        crthe   %r0, %r15
        crtle   %r0, %r15

#CHECK: cs	%r0, %r0, 0             # encoding: [0xba,0x00,0x00,0x00]
#CHECK: cs	%r0, %r0, 4095          # encoding: [0xba,0x00,0x0f,0xff]
#CHECK: cs	%r0, %r0, 0(%r1)        # encoding: [0xba,0x00,0x10,0x00]
#CHECK: cs	%r0, %r0, 0(%r15)       # encoding: [0xba,0x00,0xf0,0x00]
#CHECK: cs	%r0, %r0, 4095(%r1)     # encoding: [0xba,0x00,0x1f,0xff]
#CHECK: cs	%r0, %r0, 4095(%r15)    # encoding: [0xba,0x00,0xff,0xff]
#CHECK: cs	%r0, %r15, 0            # encoding: [0xba,0x0f,0x00,0x00]
#CHECK: cs	%r15, %r0, 0            # encoding: [0xba,0xf0,0x00,0x00]

	cs	%r0, %r0, 0
	cs	%r0, %r0, 4095
	cs	%r0, %r0, 0(%r1)
	cs	%r0, %r0, 0(%r15)
	cs	%r0, %r0, 4095(%r1)
	cs	%r0, %r0, 4095(%r15)
	cs	%r0, %r15, 0
	cs	%r15, %r0, 0

#CHECK: csch                            # encoding: [0xb2,0x30,0x00,0x00]

	csch

#CHECK: csdtr	%r0, %f0, 0             # encoding: [0xb3,0xe3,0x00,0x00]
#CHECK: csdtr	%r0, %f15, 0            # encoding: [0xb3,0xe3,0x00,0x0f]
#CHECK: csdtr	%r0, %f0, 15            # encoding: [0xb3,0xe3,0x0f,0x00]
#CHECK: csdtr	%r4, %f5, 6             # encoding: [0xb3,0xe3,0x06,0x45]
#CHECK: csdtr	%r15, %f0, 0            # encoding: [0xb3,0xe3,0x00,0xf0]

	csdtr	%r0, %f0, 0
	csdtr	%r0, %f15, 0
	csdtr	%r0, %f0, 15
	csdtr	%r4, %f5, 6
	csdtr	%r15, %f0, 0

#CHECK: csg	%r0, %r0, -524288       # encoding: [0xeb,0x00,0x00,0x00,0x80,0x30]
#CHECK: csg	%r0, %r0, -1            # encoding: [0xeb,0x00,0x0f,0xff,0xff,0x30]
#CHECK: csg	%r0, %r0, 0             # encoding: [0xeb,0x00,0x00,0x00,0x00,0x30]
#CHECK: csg	%r0, %r0, 1             # encoding: [0xeb,0x00,0x00,0x01,0x00,0x30]
#CHECK: csg	%r0, %r0, 524287        # encoding: [0xeb,0x00,0x0f,0xff,0x7f,0x30]
#CHECK: csg	%r0, %r0, 0(%r1)        # encoding: [0xeb,0x00,0x10,0x00,0x00,0x30]
#CHECK: csg	%r0, %r0, 0(%r15)       # encoding: [0xeb,0x00,0xf0,0x00,0x00,0x30]
#CHECK: csg	%r0, %r0, 524287(%r1)   # encoding: [0xeb,0x00,0x1f,0xff,0x7f,0x30]
#CHECK: csg	%r0, %r0, 524287(%r15)  # encoding: [0xeb,0x00,0xff,0xff,0x7f,0x30]
#CHECK: csg	%r0, %r15, 0            # encoding: [0xeb,0x0f,0x00,0x00,0x00,0x30]
#CHECK: csg	%r15, %r0, 0            # encoding: [0xeb,0xf0,0x00,0x00,0x00,0x30]

	csg	%r0, %r0, -524288
	csg	%r0, %r0, -1
	csg	%r0, %r0, 0
	csg	%r0, %r0, 1
	csg	%r0, %r0, 524287
	csg	%r0, %r0, 0(%r1)
	csg	%r0, %r0, 0(%r15)
	csg	%r0, %r0, 524287(%r1)
	csg	%r0, %r0, 524287(%r15)
	csg	%r0, %r15, 0
	csg	%r15, %r0, 0

#CHECK: csp	%r0, %r0                # encoding: [0xb2,0x50,0x00,0x00]
#CHECK: csp	%r0, %r15               # encoding: [0xb2,0x50,0x00,0x0f]
#CHECK: csp	%r14, %r0               # encoding: [0xb2,0x50,0x00,0xe0]
#CHECK: csp	%r6, %r8                # encoding: [0xb2,0x50,0x00,0x68]

	csp	%r0,%r0
	csp	%r0,%r15
	csp	%r14,%r0
	csp	%r6,%r8

#CHECK: cspg	%r0, %r0                # encoding: [0xb9,0x8a,0x00,0x00]
#CHECK: cspg	%r0, %r15               # encoding: [0xb9,0x8a,0x00,0x0f]
#CHECK: cspg	%r14, %r0               # encoding: [0xb9,0x8a,0x00,0xe0]
#CHECK: cspg	%r6, %r8                # encoding: [0xb9,0x8a,0x00,0x68]

	cspg	%r0,%r0
	cspg	%r0,%r15
	cspg	%r14,%r0
	cspg	%r6,%r8

#CHECK: csst	0, 0, %r0               # encoding: [0xc8,0x02,0x00,0x00,0x00,0x00]
#CHECK: csst	0(%r1), 0(%r15), %r2    # encoding: [0xc8,0x22,0x10,0x00,0xf0,0x00]
#CHECK: csst	1(%r1), 0(%r15), %r2    # encoding: [0xc8,0x22,0x10,0x01,0xf0,0x00]
#CHECK: csst	4095(%r1), 0(%r15), %r2 # encoding: [0xc8,0x22,0x1f,0xff,0xf0,0x00]
#CHECK: csst	0(%r1), 1(%r15), %r2    # encoding: [0xc8,0x22,0x10,0x00,0xf0,0x01]
#CHECK: csst	0(%r1), 4095(%r15), %r2 # encoding: [0xc8,0x22,0x10,0x00,0xff,0xff]

        csst	0, 0, %r0
        csst	0(%r1), 0(%r15), %r2
        csst	1(%r1), 0(%r15), %r2
        csst	4095(%r1), 0(%r15), %r2
        csst	0(%r1), 1(%r15), %r2
        csst	0(%r1), 4095(%r15), %r2

#CHECK: csxtr	%r0, %f0, 0             # encoding: [0xb3,0xeb,0x00,0x00]
#CHECK: csxtr	%r0, %f13, 0            # encoding: [0xb3,0xeb,0x00,0x0d]
#CHECK: csxtr	%r0, %f0, 15            # encoding: [0xb3,0xeb,0x0f,0x00]
#CHECK: csxtr	%r4, %f5, 6             # encoding: [0xb3,0xeb,0x06,0x45]
#CHECK: csxtr	%r14, %f0, 0            # encoding: [0xb3,0xeb,0x00,0xe0]

	csxtr	%r0, %f0, 0
	csxtr	%r0, %f13, 0
	csxtr	%r0, %f0, 15
	csxtr	%r4, %f5, 6
	csxtr	%r14, %f0, 0

#CHECK: csy	%r0, %r0, -524288       # encoding: [0xeb,0x00,0x00,0x00,0x80,0x14]
#CHECK: csy	%r0, %r0, -1            # encoding: [0xeb,0x00,0x0f,0xff,0xff,0x14]
#CHECK: csy	%r0, %r0, 0             # encoding: [0xeb,0x00,0x00,0x00,0x00,0x14]
#CHECK: csy	%r0, %r0, 1             # encoding: [0xeb,0x00,0x00,0x01,0x00,0x14]
#CHECK: csy	%r0, %r0, 524287        # encoding: [0xeb,0x00,0x0f,0xff,0x7f,0x14]
#CHECK: csy	%r0, %r0, 0(%r1)        # encoding: [0xeb,0x00,0x10,0x00,0x00,0x14]
#CHECK: csy	%r0, %r0, 0(%r15)       # encoding: [0xeb,0x00,0xf0,0x00,0x00,0x14]
#CHECK: csy	%r0, %r0, 524287(%r1)   # encoding: [0xeb,0x00,0x1f,0xff,0x7f,0x14]
#CHECK: csy	%r0, %r0, 524287(%r15)  # encoding: [0xeb,0x00,0xff,0xff,0x7f,0x14]
#CHECK: csy	%r0, %r15, 0            # encoding: [0xeb,0x0f,0x00,0x00,0x00,0x14]
#CHECK: csy	%r15, %r0, 0            # encoding: [0xeb,0xf0,0x00,0x00,0x00,0x14]

	csy	%r0, %r0, -524288
	csy	%r0, %r0, -1
	csy	%r0, %r0, 0
	csy	%r0, %r0, 1
	csy	%r0, %r0, 524287
	csy	%r0, %r0, 0(%r1)
	csy	%r0, %r0, 0(%r15)
	csy	%r0, %r0, 524287(%r1)
	csy	%r0, %r0, 524287(%r15)
	csy	%r0, %r15, 0
	csy	%r15, %r0, 0

#CHECK: cu12	%r0, %r0                # encoding: [0xb2,0xa7,0x00,0x00]
#CHECK: cu12	%r0, %r14               # encoding: [0xb2,0xa7,0x00,0x0e]
#CHECK: cu12	%r14, %r0               # encoding: [0xb2,0xa7,0x00,0xe0]
#CHECK: cu12	%r6, %r8                # encoding: [0xb2,0xa7,0x00,0x68]
#CHECK: cu12	%r4, %r12, 0            # encoding: [0xb2,0xa7,0x00,0x4c]
#CHECK: cu12	%r4, %r12, 15           # encoding: [0xb2,0xa7,0xf0,0x4c]

	cu12	%r0, %r0
	cu12	%r0, %r14
	cu12	%r14, %r0
	cu12	%r6, %r8
	cu12	%r4, %r12, 0
	cu12	%r4, %r12, 15

#CHECK: cu14	%r0, %r0                # encoding: [0xb9,0xb0,0x00,0x00]
#CHECK: cu14	%r0, %r14               # encoding: [0xb9,0xb0,0x00,0x0e]
#CHECK: cu14	%r14, %r0               # encoding: [0xb9,0xb0,0x00,0xe0]
#CHECK: cu14	%r6, %r8                # encoding: [0xb9,0xb0,0x00,0x68]
#CHECK: cu14	%r4, %r12, 0            # encoding: [0xb9,0xb0,0x00,0x4c]
#CHECK: cu14	%r4, %r12, 15           # encoding: [0xb9,0xb0,0xf0,0x4c]

	cu14	%r0, %r0
	cu14	%r0, %r14
	cu14	%r14, %r0
	cu14	%r6, %r8
	cu14	%r4, %r12, 0
	cu14	%r4, %r12, 15

#CHECK: cu21	%r0, %r0                # encoding: [0xb2,0xa6,0x00,0x00]
#CHECK: cu21	%r0, %r14               # encoding: [0xb2,0xa6,0x00,0x0e]
#CHECK: cu21	%r14, %r0               # encoding: [0xb2,0xa6,0x00,0xe0]
#CHECK: cu21	%r6, %r8                # encoding: [0xb2,0xa6,0x00,0x68]
#CHECK: cu21	%r4, %r12, 0            # encoding: [0xb2,0xa6,0x00,0x4c]
#CHECK: cu21	%r4, %r12, 15           # encoding: [0xb2,0xa6,0xf0,0x4c]

	cu21	%r0, %r0
	cu21	%r0, %r14
	cu21	%r14, %r0
	cu21	%r6, %r8
	cu21	%r4, %r12, 0
	cu21	%r4, %r12, 15

#CHECK: cu24	%r0, %r0                # encoding: [0xb9,0xb1,0x00,0x00]
#CHECK: cu24	%r0, %r14               # encoding: [0xb9,0xb1,0x00,0x0e]
#CHECK: cu24	%r14, %r0               # encoding: [0xb9,0xb1,0x00,0xe0]
#CHECK: cu24	%r6, %r8                # encoding: [0xb9,0xb1,0x00,0x68]
#CHECK: cu24	%r4, %r12, 0            # encoding: [0xb9,0xb1,0x00,0x4c]
#CHECK: cu24	%r4, %r12, 15           # encoding: [0xb9,0xb1,0xf0,0x4c]

	cu24	%r0, %r0
	cu24	%r0, %r14
	cu24	%r14, %r0
	cu24	%r6, %r8
	cu24	%r4, %r12, 0
	cu24	%r4, %r12, 15

#CHECK: cu41	%r0, %r0                # encoding: [0xb9,0xb2,0x00,0x00]
#CHECK: cu41	%r0, %r14               # encoding: [0xb9,0xb2,0x00,0x0e]
#CHECK: cu41	%r14, %r0               # encoding: [0xb9,0xb2,0x00,0xe0]
#CHECK: cu41	%r6, %r8                # encoding: [0xb9,0xb2,0x00,0x68]

	cu41	%r0, %r0
	cu41	%r0, %r14
	cu41	%r14, %r0
	cu41	%r6, %r8

#CHECK: cu42	%r0, %r0                # encoding: [0xb9,0xb3,0x00,0x00]
#CHECK: cu42	%r0, %r14               # encoding: [0xb9,0xb3,0x00,0x0e]
#CHECK: cu42	%r14, %r0               # encoding: [0xb9,0xb3,0x00,0xe0]
#CHECK: cu42	%r6, %r8                # encoding: [0xb9,0xb3,0x00,0x68]

	cu42	%r0, %r0
	cu42	%r0, %r14
	cu42	%r14, %r0
	cu42	%r6, %r8

#CHECK: cudtr	%r0, %f0                # encoding: [0xb3,0xe2,0x00,0x00]
#CHECK: cudtr	%r0, %f15               # encoding: [0xb3,0xe2,0x00,0x0f]
#CHECK: cudtr	%r15, %f0               # encoding: [0xb3,0xe2,0x00,0xf0]
#CHECK: cudtr	%r7, %f8                # encoding: [0xb3,0xe2,0x00,0x78]
#CHECK: cudtr	%r15, %f15              # encoding: [0xb3,0xe2,0x00,0xff]

	cudtr	%r0, %f0
	cudtr	%r0, %f15
	cudtr	%r15, %f0
	cudtr	%r7, %f8
	cudtr	%r15, %f15

#CHECK: cuse	%r0, %r8                # encoding: [0xb2,0x57,0x00,0x08]
#CHECK: cuse	%r0, %r14               # encoding: [0xb2,0x57,0x00,0x0e]
#CHECK: cuse	%r14, %r0               # encoding: [0xb2,0x57,0x00,0xe0]
#CHECK: cuse	%r14, %r8               # encoding: [0xb2,0x57,0x00,0xe8]

	cuse	%r0, %r8
	cuse	%r0, %r14
	cuse	%r14, %r0
	cuse	%r14, %r8

#CHECK: cutfu	%r0, %r0                # encoding: [0xb2,0xa7,0x00,0x00]
#CHECK: cutfu	%r0, %r14               # encoding: [0xb2,0xa7,0x00,0x0e]
#CHECK: cutfu	%r14, %r0               # encoding: [0xb2,0xa7,0x00,0xe0]
#CHECK: cutfu	%r6, %r8                # encoding: [0xb2,0xa7,0x00,0x68]
#CHECK: cutfu	%r4, %r12, 0            # encoding: [0xb2,0xa7,0x00,0x4c]
#CHECK: cutfu	%r4, %r12, 15           # encoding: [0xb2,0xa7,0xf0,0x4c]

	cutfu	%r0, %r0
	cutfu	%r0, %r14
	cutfu	%r14, %r0
	cutfu	%r6, %r8
	cutfu	%r4, %r12, 0
	cutfu	%r4, %r12, 15

#CHECK: cuutf	%r0, %r0                # encoding: [0xb2,0xa6,0x00,0x00]
#CHECK: cuutf	%r0, %r14               # encoding: [0xb2,0xa6,0x00,0x0e]
#CHECK: cuutf	%r14, %r0               # encoding: [0xb2,0xa6,0x00,0xe0]
#CHECK: cuutf	%r6, %r8                # encoding: [0xb2,0xa6,0x00,0x68]
#CHECK: cuutf	%r4, %r12, 0            # encoding: [0xb2,0xa6,0x00,0x4c]
#CHECK: cuutf	%r4, %r12, 15           # encoding: [0xb2,0xa6,0xf0,0x4c]

	cuutf	%r0, %r0
	cuutf	%r0, %r14
	cuutf	%r14, %r0
	cuutf	%r6, %r8
	cuutf	%r4, %r12, 0
	cuutf	%r4, %r12, 15

#CHECK: cuxtr	%r0, %f0                # encoding: [0xb3,0xea,0x00,0x00]
#CHECK: cuxtr	%r0, %f13               # encoding: [0xb3,0xea,0x00,0x0d]
#CHECK: cuxtr	%r14, %f0               # encoding: [0xb3,0xea,0x00,0xe0]
#CHECK: cuxtr	%r6, %f8                # encoding: [0xb3,0xea,0x00,0x68]
#CHECK: cuxtr	%r14, %f13              # encoding: [0xb3,0xea,0x00,0xed]

	cuxtr	%r0, %f0
	cuxtr	%r0, %f13
	cuxtr	%r14, %f0
	cuxtr	%r6, %f8
	cuxtr	%r14, %f13

#CHECK: cvb	%r0, 0                  # encoding: [0x4f,0x00,0x00,0x00]
#CHECK: cvb	%r0, 4095               # encoding: [0x4f,0x00,0x0f,0xff]
#CHECK: cvb	%r0, 0(%r1)             # encoding: [0x4f,0x00,0x10,0x00]
#CHECK: cvb	%r0, 0(%r15)            # encoding: [0x4f,0x00,0xf0,0x00]
#CHECK: cvb	%r0, 4095(%r1,%r15)     # encoding: [0x4f,0x01,0xff,0xff]
#CHECK: cvb	%r0, 4095(%r15,%r1)     # encoding: [0x4f,0x0f,0x1f,0xff]
#CHECK: cvb	%r15, 0                 # encoding: [0x4f,0xf0,0x00,0x00]

	cvb	%r0, 0
	cvb	%r0, 4095
	cvb	%r0, 0(%r1)
	cvb	%r0, 0(%r15)
	cvb	%r0, 4095(%r1,%r15)
	cvb	%r0, 4095(%r15,%r1)
	cvb	%r15, 0

#CHECK: cvbg	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x0e]
#CHECK: cvbg	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x0e]
#CHECK: cvbg	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x0e]
#CHECK: cvbg	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x0e]
#CHECK: cvbg	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x0e]
#CHECK: cvbg	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x0e]
#CHECK: cvbg	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x0e]
#CHECK: cvbg	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x0e]
#CHECK: cvbg	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x0e]
#CHECK: cvbg	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x0e]

	cvbg	%r0, -524288
	cvbg	%r0, -1
	cvbg	%r0, 0
	cvbg	%r0, 1
	cvbg	%r0, 524287
	cvbg	%r0, 0(%r1)
	cvbg	%r0, 0(%r15)
	cvbg	%r0, 524287(%r1,%r15)
	cvbg	%r0, 524287(%r15,%r1)
	cvbg	%r15, 0

#CHECK: cvby	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x06]
#CHECK: cvby	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x06]
#CHECK: cvby	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x06]
#CHECK: cvby	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x06]
#CHECK: cvby	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x06]
#CHECK: cvby	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x06]
#CHECK: cvby	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x06]
#CHECK: cvby	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x06]
#CHECK: cvby	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x06]
#CHECK: cvby	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x06]

	cvby	%r0, -524288
	cvby	%r0, -1
	cvby	%r0, 0
	cvby	%r0, 1
	cvby	%r0, 524287
	cvby	%r0, 0(%r1)
	cvby	%r0, 0(%r15)
	cvby	%r0, 524287(%r1,%r15)
	cvby	%r0, 524287(%r15,%r1)
	cvby	%r15, 0

#CHECK: cvd	%r0, 0                  # encoding: [0x4e,0x00,0x00,0x00]
#CHECK: cvd	%r0, 4095               # encoding: [0x4e,0x00,0x0f,0xff]
#CHECK: cvd	%r0, 0(%r1)             # encoding: [0x4e,0x00,0x10,0x00]
#CHECK: cvd	%r0, 0(%r15)            # encoding: [0x4e,0x00,0xf0,0x00]
#CHECK: cvd	%r0, 4095(%r1,%r15)     # encoding: [0x4e,0x01,0xff,0xff]
#CHECK: cvd	%r0, 4095(%r15,%r1)     # encoding: [0x4e,0x0f,0x1f,0xff]
#CHECK: cvd	%r15, 0                 # encoding: [0x4e,0xf0,0x00,0x00]

	cvd	%r0, 0
	cvd	%r0, 4095
	cvd	%r0, 0(%r1)
	cvd	%r0, 0(%r15)
	cvd	%r0, 4095(%r1,%r15)
	cvd	%r0, 4095(%r15,%r1)
	cvd	%r15, 0

#CHECK: cvdg	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x2e]
#CHECK: cvdg	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x2e]
#CHECK: cvdg	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x2e]
#CHECK: cvdg	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x2e]
#CHECK: cvdg	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x2e]
#CHECK: cvdg	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x2e]
#CHECK: cvdg	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x2e]
#CHECK: cvdg	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x2e]
#CHECK: cvdg	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x2e]
#CHECK: cvdg	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x2e]

	cvdg	%r0, -524288
	cvdg	%r0, -1
	cvdg	%r0, 0
	cvdg	%r0, 1
	cvdg	%r0, 524287
	cvdg	%r0, 0(%r1)
	cvdg	%r0, 0(%r15)
	cvdg	%r0, 524287(%r1,%r15)
	cvdg	%r0, 524287(%r15,%r1)
	cvdg	%r15, 0

#CHECK: cvdy	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x26]
#CHECK: cvdy	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x26]
#CHECK: cvdy	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x26]
#CHECK: cvdy	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x26]
#CHECK: cvdy	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x26]
#CHECK: cvdy	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x26]
#CHECK: cvdy	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x26]
#CHECK: cvdy	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x26]
#CHECK: cvdy	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x26]
#CHECK: cvdy	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x26]

	cvdy	%r0, -524288
	cvdy	%r0, -1
	cvdy	%r0, 0
	cvdy	%r0, 1
	cvdy	%r0, 524287
	cvdy	%r0, 0(%r1)
	cvdy	%r0, 0(%r15)
	cvdy	%r0, 524287(%r1,%r15)
	cvdy	%r0, 524287(%r15,%r1)
	cvdy	%r15, 0

#CHECK: cxbr	%f0, %f0                # encoding: [0xb3,0x49,0x00,0x00]
#CHECK: cxbr	%f0, %f13               # encoding: [0xb3,0x49,0x00,0x0d]
#CHECK: cxbr	%f8, %f8                # encoding: [0xb3,0x49,0x00,0x88]
#CHECK: cxbr	%f13, %f0               # encoding: [0xb3,0x49,0x00,0xd0]

	cxbr	%f0, %f0
	cxbr	%f0, %f13
	cxbr	%f8, %f8
	cxbr	%f13, %f0

#CHECK: cxfbr	%f0, %r0                # encoding: [0xb3,0x96,0x00,0x00]
#CHECK: cxfbr	%f0, %r15               # encoding: [0xb3,0x96,0x00,0x0f]
#CHECK: cxfbr	%f13, %r0               # encoding: [0xb3,0x96,0x00,0xd0]
#CHECK: cxfbr	%f8, %r7                # encoding: [0xb3,0x96,0x00,0x87]
#CHECK: cxfbr	%f13, %r15              # encoding: [0xb3,0x96,0x00,0xdf]

	cxfbr	%f0, %r0
	cxfbr	%f0, %r15
	cxfbr	%f13, %r0
	cxfbr	%f8, %r7
	cxfbr	%f13, %r15

#CHECK: cxfr	%f0, %r0                # encoding: [0xb3,0xb6,0x00,0x00]
#CHECK: cxfr	%f0, %r15               # encoding: [0xb3,0xb6,0x00,0x0f]
#CHECK: cxfr	%f13, %r0               # encoding: [0xb3,0xb6,0x00,0xd0]
#CHECK: cxfr	%f8, %r7                # encoding: [0xb3,0xb6,0x00,0x87]
#CHECK: cxfr	%f13, %r15              # encoding: [0xb3,0xb6,0x00,0xdf]

	cxfr	%f0, %r0
	cxfr	%f0, %r15
	cxfr	%f13, %r0
	cxfr	%f8, %r7
	cxfr	%f13, %r15

#CHECK: cxgbr	%f0, %r0                # encoding: [0xb3,0xa6,0x00,0x00]
#CHECK: cxgbr	%f0, %r15               # encoding: [0xb3,0xa6,0x00,0x0f]
#CHECK: cxgbr	%f13, %r0               # encoding: [0xb3,0xa6,0x00,0xd0]
#CHECK: cxgbr	%f8, %r7                # encoding: [0xb3,0xa6,0x00,0x87]
#CHECK: cxgbr	%f13, %r15              # encoding: [0xb3,0xa6,0x00,0xdf]

	cxgbr	%f0, %r0
	cxgbr	%f0, %r15
	cxgbr	%f13, %r0
	cxgbr	%f8, %r7
	cxgbr	%f13, %r15

#CHECK: cxgr	%f0, %r0                # encoding: [0xb3,0xc6,0x00,0x00]
#CHECK: cxgr	%f0, %r15               # encoding: [0xb3,0xc6,0x00,0x0f]
#CHECK: cxgr	%f13, %r0               # encoding: [0xb3,0xc6,0x00,0xd0]
#CHECK: cxgr	%f8, %r7                # encoding: [0xb3,0xc6,0x00,0x87]
#CHECK: cxgr	%f13, %r15              # encoding: [0xb3,0xc6,0x00,0xdf]

	cxgr	%f0, %r0
	cxgr	%f0, %r15
	cxgr	%f13, %r0
	cxgr	%f8, %r7
	cxgr	%f13, %r15

#CHECK: cxgtr	%f0, %r0                # encoding: [0xb3,0xf9,0x00,0x00]
#CHECK: cxgtr	%f0, %r15               # encoding: [0xb3,0xf9,0x00,0x0f]
#CHECK: cxgtr	%f13, %r0               # encoding: [0xb3,0xf9,0x00,0xd0]
#CHECK: cxgtr	%f8, %r7                # encoding: [0xb3,0xf9,0x00,0x87]
#CHECK: cxgtr	%f13, %r15              # encoding: [0xb3,0xf9,0x00,0xdf]

	cxgtr	%f0, %r0
	cxgtr	%f0, %r15
	cxgtr	%f13, %r0
	cxgtr	%f8, %r7
	cxgtr	%f13, %r15

#CHECK: cxr	%f0, %f0                # encoding: [0xb3,0x69,0x00,0x00]
#CHECK: cxr	%f0, %f13               # encoding: [0xb3,0x69,0x00,0x0d]
#CHECK: cxr	%f8, %f8                # encoding: [0xb3,0x69,0x00,0x88]
#CHECK: cxr	%f13, %f0               # encoding: [0xb3,0x69,0x00,0xd0]

	cxr	%f0, %f0
	cxr	%f0, %f13
	cxr	%f8, %f8
	cxr	%f13, %f0

#CHECK: cxstr	%f0, %r0                # encoding: [0xb3,0xfb,0x00,0x00]
#CHECK: cxstr	%f0, %r14               # encoding: [0xb3,0xfb,0x00,0x0e]
#CHECK: cxstr	%f13, %r0               # encoding: [0xb3,0xfb,0x00,0xd0]
#CHECK: cxstr	%f8, %r6                # encoding: [0xb3,0xfb,0x00,0x86]
#CHECK: cxstr	%f13, %r14              # encoding: [0xb3,0xfb,0x00,0xde]

	cxstr	%f0, %r0
	cxstr	%f0, %r14
	cxstr	%f13, %r0
	cxstr	%f8, %r6
	cxstr	%f13, %r14

#CHECK: cxtr	%f0, %f0                # encoding: [0xb3,0xec,0x00,0x00]
#CHECK: cxtr	%f0, %f13               # encoding: [0xb3,0xec,0x00,0x0d]
#CHECK: cxtr	%f8, %f8                # encoding: [0xb3,0xec,0x00,0x88]
#CHECK: cxtr	%f13, %f0               # encoding: [0xb3,0xec,0x00,0xd0]

	cxtr	%f0, %f0
	cxtr	%f0, %f13
	cxtr	%f8, %f8
	cxtr	%f13, %f0

#CHECK: cxutr	%f0, %r0                # encoding: [0xb3,0xfa,0x00,0x00]
#CHECK: cxutr	%f0, %r14               # encoding: [0xb3,0xfa,0x00,0x0e]
#CHECK: cxutr	%f13, %r0               # encoding: [0xb3,0xfa,0x00,0xd0]
#CHECK: cxutr	%f8, %r6                # encoding: [0xb3,0xfa,0x00,0x86]
#CHECK: cxutr	%f13, %r14              # encoding: [0xb3,0xfa,0x00,0xde]

	cxutr	%f0, %r0
	cxutr	%f0, %r14
	cxutr	%f13, %r0
	cxutr	%f8, %r6
	cxutr	%f13, %r14

#CHECK: cy	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x59]
#CHECK: cy	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x59]
#CHECK: cy	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x59]
#CHECK: cy	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x59]
#CHECK: cy	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x59]
#CHECK: cy	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x59]
#CHECK: cy	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x59]
#CHECK: cy	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x59]
#CHECK: cy	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x59]
#CHECK: cy	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x59]

	cy	%r0, -524288
	cy	%r0, -1
	cy	%r0, 0
	cy	%r0, 1
	cy	%r0, 524287
	cy	%r0, 0(%r1)
	cy	%r0, 0(%r15)
	cy	%r0, 524287(%r1,%r15)
	cy	%r0, 524287(%r15,%r1)
	cy	%r15, 0

#CHECK: d	%r0, 0                  # encoding: [0x5d,0x00,0x00,0x00]
#CHECK: d	%r0, 4095               # encoding: [0x5d,0x00,0x0f,0xff]
#CHECK: d	%r0, 0(%r1)             # encoding: [0x5d,0x00,0x10,0x00]
#CHECK: d	%r0, 0(%r15)            # encoding: [0x5d,0x00,0xf0,0x00]
#CHECK: d	%r0, 4095(%r1,%r15)     # encoding: [0x5d,0x01,0xff,0xff]
#CHECK: d	%r0, 4095(%r15,%r1)     # encoding: [0x5d,0x0f,0x1f,0xff]
#CHECK: d	%r14, 0                 # encoding: [0x5d,0xe0,0x00,0x00]

	d	%r0, 0
	d	%r0, 4095
	d	%r0, 0(%r1)
	d	%r0, 0(%r15)
	d	%r0, 4095(%r1,%r15)
	d	%r0, 4095(%r15,%r1)
	d	%r14, 0

#CHECK: dd	%f0, 0                  # encoding: [0x6d,0x00,0x00,0x00]
#CHECK: dd	%f0, 4095               # encoding: [0x6d,0x00,0x0f,0xff]
#CHECK: dd	%f0, 0(%r1)             # encoding: [0x6d,0x00,0x10,0x00]
#CHECK: dd	%f0, 0(%r15)            # encoding: [0x6d,0x00,0xf0,0x00]
#CHECK: dd	%f0, 4095(%r1,%r15)     # encoding: [0x6d,0x01,0xff,0xff]
#CHECK: dd	%f0, 4095(%r15,%r1)     # encoding: [0x6d,0x0f,0x1f,0xff]
#CHECK: dd	%f15, 0                 # encoding: [0x6d,0xf0,0x00,0x00]

	dd	%f0, 0
	dd	%f0, 4095
	dd	%f0, 0(%r1)
	dd	%f0, 0(%r15)
	dd	%f0, 4095(%r1,%r15)
	dd	%f0, 4095(%r15,%r1)
	dd	%f15, 0

#CHECK: ddb	%f0, 0                  # encoding: [0xed,0x00,0x00,0x00,0x00,0x1d]
#CHECK: ddb	%f0, 4095               # encoding: [0xed,0x00,0x0f,0xff,0x00,0x1d]
#CHECK: ddb	%f0, 0(%r1)             # encoding: [0xed,0x00,0x10,0x00,0x00,0x1d]
#CHECK: ddb	%f0, 0(%r15)            # encoding: [0xed,0x00,0xf0,0x00,0x00,0x1d]
#CHECK: ddb	%f0, 4095(%r1,%r15)     # encoding: [0xed,0x01,0xff,0xff,0x00,0x1d]
#CHECK: ddb	%f0, 4095(%r15,%r1)     # encoding: [0xed,0x0f,0x1f,0xff,0x00,0x1d]
#CHECK: ddb	%f15, 0                 # encoding: [0xed,0xf0,0x00,0x00,0x00,0x1d]

	ddb	%f0, 0
	ddb	%f0, 4095
	ddb	%f0, 0(%r1)
	ddb	%f0, 0(%r15)
	ddb	%f0, 4095(%r1,%r15)
	ddb	%f0, 4095(%r15,%r1)
	ddb	%f15, 0

#CHECK: ddbr	%f0, %f0                # encoding: [0xb3,0x1d,0x00,0x00]
#CHECK: ddbr	%f0, %f15               # encoding: [0xb3,0x1d,0x00,0x0f]
#CHECK: ddbr	%f7, %f8                # encoding: [0xb3,0x1d,0x00,0x78]
#CHECK: ddbr	%f15, %f0               # encoding: [0xb3,0x1d,0x00,0xf0]

	ddbr	%f0, %f0
	ddbr	%f0, %f15
	ddbr	%f7, %f8
	ddbr	%f15, %f0

#CHECK: ddr	%f0, %f0                # encoding: [0x2d,0x00]
#CHECK: ddr	%f0, %f15               # encoding: [0x2d,0x0f]
#CHECK: ddr	%f7, %f8                # encoding: [0x2d,0x78]
#CHECK: ddr	%f15, %f0               # encoding: [0x2d,0xf0]

	ddr	%f0, %f0
	ddr	%f0, %f15
	ddr	%f7, %f8
	ddr	%f15, %f0

#CHECK: ddtr	%f0, %f0, %f0           # encoding: [0xb3,0xd1,0x00,0x00]
#CHECK: ddtr	%f0, %f0, %f15          # encoding: [0xb3,0xd1,0xf0,0x00]
#CHECK: ddtr	%f0, %f15, %f0          # encoding: [0xb3,0xd1,0x00,0x0f]
#CHECK: ddtr	%f15, %f0, %f0          # encoding: [0xb3,0xd1,0x00,0xf0]
#CHECK: ddtr	%f7, %f8, %f9           # encoding: [0xb3,0xd1,0x90,0x78]

	ddtr	%f0, %f0, %f0
	ddtr	%f0, %f0, %f15
	ddtr	%f0, %f15, %f0
	ddtr	%f15, %f0, %f0
	ddtr	%f7, %f8, %f9

#CHECK: de	%f0, 0                  # encoding: [0x7d,0x00,0x00,0x00]
#CHECK: de	%f0, 4095               # encoding: [0x7d,0x00,0x0f,0xff]
#CHECK: de	%f0, 0(%r1)             # encoding: [0x7d,0x00,0x10,0x00]
#CHECK: de	%f0, 0(%r15)            # encoding: [0x7d,0x00,0xf0,0x00]
#CHECK: de	%f0, 4095(%r1,%r15)     # encoding: [0x7d,0x01,0xff,0xff]
#CHECK: de	%f0, 4095(%r15,%r1)     # encoding: [0x7d,0x0f,0x1f,0xff]
#CHECK: de	%f15, 0                 # encoding: [0x7d,0xf0,0x00,0x00]

	de	%f0, 0
	de	%f0, 4095
	de	%f0, 0(%r1)
	de	%f0, 0(%r15)
	de	%f0, 4095(%r1,%r15)
	de	%f0, 4095(%r15,%r1)
	de	%f15, 0

#CHECK: deb	%f0, 0                  # encoding: [0xed,0x00,0x00,0x00,0x00,0x0d]
#CHECK: deb	%f0, 4095               # encoding: [0xed,0x00,0x0f,0xff,0x00,0x0d]
#CHECK: deb	%f0, 0(%r1)             # encoding: [0xed,0x00,0x10,0x00,0x00,0x0d]
#CHECK: deb	%f0, 0(%r15)            # encoding: [0xed,0x00,0xf0,0x00,0x00,0x0d]
#CHECK: deb	%f0, 4095(%r1,%r15)     # encoding: [0xed,0x01,0xff,0xff,0x00,0x0d]
#CHECK: deb	%f0, 4095(%r15,%r1)     # encoding: [0xed,0x0f,0x1f,0xff,0x00,0x0d]
#CHECK: deb	%f15, 0                 # encoding: [0xed,0xf0,0x00,0x00,0x00,0x0d]

	deb	%f0, 0
	deb	%f0, 4095
	deb	%f0, 0(%r1)
	deb	%f0, 0(%r15)
	deb	%f0, 4095(%r1,%r15)
	deb	%f0, 4095(%r15,%r1)
	deb	%f15, 0

#CHECK: debr	%f0, %f0                # encoding: [0xb3,0x0d,0x00,0x00]
#CHECK: debr	%f0, %f15               # encoding: [0xb3,0x0d,0x00,0x0f]
#CHECK: debr	%f7, %f8                # encoding: [0xb3,0x0d,0x00,0x78]
#CHECK: debr	%f15, %f0               # encoding: [0xb3,0x0d,0x00,0xf0]

	debr	%f0, %f0
	debr	%f0, %f15
	debr	%f7, %f8
	debr	%f15, %f0

#CHECK: der	%f0, %f0                # encoding: [0x3d,0x00]
#CHECK: der	%f0, %f15               # encoding: [0x3d,0x0f]
#CHECK: der	%f7, %f8                # encoding: [0x3d,0x78]
#CHECK: der	%f15, %f0               # encoding: [0x3d,0xf0]

	der	%f0, %f0
	der	%f0, %f15
	der	%f7, %f8
	der	%f15, %f0

#CHECK: diag	%r0, %r0, 0             # encoding: [0x83,0x00,0x00,0x00]
#CHECK: diag	%r0, %r15, 0            # encoding: [0x83,0x0f,0x00,0x00]
#CHECK: diag	%r14, %r15, 0           # encoding: [0x83,0xef,0x00,0x00]
#CHECK: diag	%r15, %r15, 0           # encoding: [0x83,0xff,0x00,0x00]
#CHECK: diag	%r0, %r0, 4095          # encoding: [0x83,0x00,0x0f,0xff]
#CHECK: diag	%r0, %r0, 1             # encoding: [0x83,0x00,0x00,0x01]
#CHECK: diag	%r0, %r0, 0(%r1)        # encoding: [0x83,0x00,0x10,0x00]
#CHECK: diag	%r0, %r0, 0(%r15)       # encoding: [0x83,0x00,0xf0,0x00]
#CHECK: diag	%r0, %r0, 4095(%r1)     # encoding: [0x83,0x00,0x1f,0xff]
#CHECK: diag	%r0, %r0, 4095(%r15)    # encoding: [0x83,0x00,0xff,0xff]

	diag	%r0,%r0,0
	diag	%r0,%r15,0
	diag	%r14,%r15,0
	diag	%r15,%r15,0
	diag	%r0,%r0,4095
	diag	%r0,%r0,1
	diag	%r0,%r0,0(%r1)
	diag	%r0,%r0,0(%r15)
	diag	%r0,%r0,4095(%r1)
	diag	%r0,%r0,4095(%r15)

#CHECK: didbr	%f0, %f0, %f0, 0        # encoding: [0xb3,0x5b,0x00,0x00]
#CHECK: didbr	%f0, %f0, %f0, 15       # encoding: [0xb3,0x5b,0x0f,0x00]
#CHECK: didbr	%f0, %f0, %f15, 0       # encoding: [0xb3,0x5b,0x00,0x0f]
#CHECK: didbr	%f0, %f15, %f0, 0       # encoding: [0xb3,0x5b,0xf0,0x00]
#CHECK: didbr	%f4, %f5, %f6, 7        # encoding: [0xb3,0x5b,0x57,0x46]
#CHECK: didbr	%f15, %f0, %f0, 0       # encoding: [0xb3,0x5b,0x00,0xf0]

	didbr	%f0, %f0, %f0, 0
	didbr	%f0, %f0, %f0, 15
	didbr	%f0, %f0, %f15, 0
	didbr	%f0, %f15, %f0, 0
	didbr	%f4, %f5, %f6, 7
	didbr	%f15, %f0, %f0, 0

#CHECK: diebr	%f0, %f0, %f0, 0        # encoding: [0xb3,0x53,0x00,0x00]
#CHECK: diebr	%f0, %f0, %f0, 15       # encoding: [0xb3,0x53,0x0f,0x00]
#CHECK: diebr	%f0, %f0, %f15, 0       # encoding: [0xb3,0x53,0x00,0x0f]
#CHECK: diebr	%f0, %f15, %f0, 0       # encoding: [0xb3,0x53,0xf0,0x00]
#CHECK: diebr	%f4, %f5, %f6, 7        # encoding: [0xb3,0x53,0x57,0x46]
#CHECK: diebr	%f15, %f0, %f0, 0       # encoding: [0xb3,0x53,0x00,0xf0]

	diebr	%f0, %f0, %f0, 0
	diebr	%f0, %f0, %f0, 15
	diebr	%f0, %f0, %f15, 0
	diebr	%f0, %f15, %f0, 0
	diebr	%f4, %f5, %f6, 7
	diebr	%f15, %f0, %f0, 0

#CHECK: dl	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x97]
#CHECK: dl	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x97]
#CHECK: dl	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x97]
#CHECK: dl	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x97]
#CHECK: dl	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x97]
#CHECK: dl	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x97]
#CHECK: dl	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x97]
#CHECK: dl	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x97]
#CHECK: dl	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x97]
#CHECK: dl	%r14, 0                 # encoding: [0xe3,0xe0,0x00,0x00,0x00,0x97]

	dl	%r0, -524288
	dl	%r0, -1
	dl	%r0, 0
	dl	%r0, 1
	dl	%r0, 524287
	dl	%r0, 0(%r1)
	dl	%r0, 0(%r15)
	dl	%r0, 524287(%r1,%r15)
	dl	%r0, 524287(%r15,%r1)
	dl	%r14, 0

#CHECK: dlg	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x87]
#CHECK: dlg	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x87]
#CHECK: dlg	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x87]
#CHECK: dlg	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x87]
#CHECK: dlg	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x87]
#CHECK: dlg	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x87]
#CHECK: dlg	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x87]
#CHECK: dlg	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x87]
#CHECK: dlg	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x87]
#CHECK: dlg	%r14, 0                 # encoding: [0xe3,0xe0,0x00,0x00,0x00,0x87]

	dlg	%r0, -524288
	dlg	%r0, -1
	dlg	%r0, 0
	dlg	%r0, 1
	dlg	%r0, 524287
	dlg	%r0, 0(%r1)
	dlg	%r0, 0(%r15)
	dlg	%r0, 524287(%r1,%r15)
	dlg	%r0, 524287(%r15,%r1)
	dlg	%r14, 0

#CHECK: dlgr	%r0, %r0                # encoding: [0xb9,0x87,0x00,0x00]
#CHECK: dlgr	%r0, %r15               # encoding: [0xb9,0x87,0x00,0x0f]
#CHECK: dlgr	%r14, %r0               # encoding: [0xb9,0x87,0x00,0xe0]
#CHECK: dlgr	%r6, %r9                # encoding: [0xb9,0x87,0x00,0x69]

	dlgr	%r0,%r0
	dlgr	%r0,%r15
	dlgr	%r14,%r0
	dlgr	%r6,%r9

#CHECK: dlr	%r0, %r0                # encoding: [0xb9,0x97,0x00,0x00]
#CHECK: dlr	%r0, %r15               # encoding: [0xb9,0x97,0x00,0x0f]
#CHECK: dlr	%r14, %r0               # encoding: [0xb9,0x97,0x00,0xe0]
#CHECK: dlr	%r6, %r9                # encoding: [0xb9,0x97,0x00,0x69]

	dlr	%r0,%r0
	dlr	%r0,%r15
	dlr	%r14,%r0
	dlr	%r6,%r9

#CHECK: dp	0(1), 0(1)              # encoding: [0xfd,0x00,0x00,0x00,0x00,0x00]
#CHECK: dp	0(1), 0(1,%r1)          # encoding: [0xfd,0x00,0x00,0x00,0x10,0x00]
#CHECK: dp	0(1), 0(1,%r15)         # encoding: [0xfd,0x00,0x00,0x00,0xf0,0x00]
#CHECK: dp	0(1), 4095(1)           # encoding: [0xfd,0x00,0x00,0x00,0x0f,0xff]
#CHECK: dp	0(1), 4095(1,%r1)       # encoding: [0xfd,0x00,0x00,0x00,0x1f,0xff]
#CHECK: dp	0(1), 4095(1,%r15)      # encoding: [0xfd,0x00,0x00,0x00,0xff,0xff]
#CHECK: dp	0(1,%r1), 0(1)          # encoding: [0xfd,0x00,0x10,0x00,0x00,0x00]
#CHECK: dp	0(1,%r15), 0(1)         # encoding: [0xfd,0x00,0xf0,0x00,0x00,0x00]
#CHECK: dp	4095(1,%r1), 0(1)       # encoding: [0xfd,0x00,0x1f,0xff,0x00,0x00]
#CHECK: dp	4095(1,%r15), 0(1)      # encoding: [0xfd,0x00,0xff,0xff,0x00,0x00]
#CHECK: dp	0(16,%r1), 0(1)         # encoding: [0xfd,0xf0,0x10,0x00,0x00,0x00]
#CHECK: dp	0(16,%r15), 0(1)        # encoding: [0xfd,0xf0,0xf0,0x00,0x00,0x00]
#CHECK: dp	0(1), 0(16,%r1)         # encoding: [0xfd,0x0f,0x00,0x00,0x10,0x00]
#CHECK: dp	0(1), 0(16,%r15)        # encoding: [0xfd,0x0f,0x00,0x00,0xf0,0x00]

	dp	0(1), 0(1)
	dp	0(1), 0(1,%r1)
	dp	0(1), 0(1,%r15)
	dp	0(1), 4095(1)
	dp	0(1), 4095(1,%r1)
	dp	0(1), 4095(1,%r15)
	dp	0(1,%r1), 0(1)
	dp	0(1,%r15), 0(1)
	dp	4095(1,%r1), 0(1)
	dp	4095(1,%r15), 0(1)
	dp	0(16,%r1), 0(1)
	dp	0(16,%r15), 0(1)
	dp	0(1), 0(16,%r1)
	dp	0(1), 0(16,%r15)

#CHECK: dr	%r0, %r0                # encoding: [0x1d,0x00]
#CHECK: dr	%r0, %r15               # encoding: [0x1d,0x0f]
#CHECK: dr	%r14, %r0               # encoding: [0x1d,0xe0]
#CHECK: dr	%r6, %r9                # encoding: [0x1d,0x69]

	dr	%r0,%r0
	dr	%r0,%r15
	dr	%r14,%r0
	dr	%r6,%r9

#CHECK: dsg	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x0d]
#CHECK: dsg	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x0d]
#CHECK: dsg	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x0d]
#CHECK: dsg	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x0d]
#CHECK: dsg	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x0d]
#CHECK: dsg	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x0d]
#CHECK: dsg	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x0d]
#CHECK: dsg	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x0d]
#CHECK: dsg	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x0d]
#CHECK: dsg	%r14, 0                 # encoding: [0xe3,0xe0,0x00,0x00,0x00,0x0d]

	dsg	%r0, -524288
	dsg	%r0, -1
	dsg	%r0, 0
	dsg	%r0, 1
	dsg	%r0, 524287
	dsg	%r0, 0(%r1)
	dsg	%r0, 0(%r15)
	dsg	%r0, 524287(%r1,%r15)
	dsg	%r0, 524287(%r15,%r1)
	dsg	%r14, 0

#CHECK: dsgf	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x1d]
#CHECK: dsgf	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x1d]
#CHECK: dsgf	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x1d]
#CHECK: dsgf	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x1d]
#CHECK: dsgf	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x1d]
#CHECK: dsgf	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x1d]
#CHECK: dsgf	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x1d]
#CHECK: dsgf	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x1d]
#CHECK: dsgf	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x1d]
#CHECK: dsgf	%r14, 0                 # encoding: [0xe3,0xe0,0x00,0x00,0x00,0x1d]

	dsgf	%r0, -524288
	dsgf	%r0, -1
	dsgf	%r0, 0
	dsgf	%r0, 1
	dsgf	%r0, 524287
	dsgf	%r0, 0(%r1)
	dsgf	%r0, 0(%r15)
	dsgf	%r0, 524287(%r1,%r15)
	dsgf	%r0, 524287(%r15,%r1)
	dsgf	%r14, 0

#CHECK: dsgfr	%r0, %r0                # encoding: [0xb9,0x1d,0x00,0x00]
#CHECK: dsgfr	%r0, %r15               # encoding: [0xb9,0x1d,0x00,0x0f]
#CHECK: dsgfr	%r14, %r0               # encoding: [0xb9,0x1d,0x00,0xe0]
#CHECK: dsgfr	%r6, %r9                # encoding: [0xb9,0x1d,0x00,0x69]

	dsgfr	%r0,%r0
	dsgfr	%r0,%r15
	dsgfr	%r14,%r0
	dsgfr	%r6,%r9

#CHECK: dsgr	%r0, %r0                # encoding: [0xb9,0x0d,0x00,0x00]
#CHECK: dsgr	%r0, %r15               # encoding: [0xb9,0x0d,0x00,0x0f]
#CHECK: dsgr	%r14, %r0               # encoding: [0xb9,0x0d,0x00,0xe0]
#CHECK: dsgr	%r6, %r9                # encoding: [0xb9,0x0d,0x00,0x69]

	dsgr	%r0,%r0
	dsgr	%r0,%r15
	dsgr	%r14,%r0
	dsgr	%r6,%r9

#CHECK: dxbr	%f0, %f0                # encoding: [0xb3,0x4d,0x00,0x00]
#CHECK: dxbr	%f0, %f13               # encoding: [0xb3,0x4d,0x00,0x0d]
#CHECK: dxbr	%f8, %f8                # encoding: [0xb3,0x4d,0x00,0x88]
#CHECK: dxbr	%f13, %f0               # encoding: [0xb3,0x4d,0x00,0xd0]

	dxbr	%f0, %f0
	dxbr	%f0, %f13
	dxbr	%f8, %f8
	dxbr	%f13, %f0

#CHECK: dxr	%f0, %f0                # encoding: [0xb2,0x2d,0x00,0x00]
#CHECK: dxr	%f0, %f13               # encoding: [0xb2,0x2d,0x00,0x0d]
#CHECK: dxr	%f8, %f8                # encoding: [0xb2,0x2d,0x00,0x88]
#CHECK: dxr	%f13, %f0               # encoding: [0xb2,0x2d,0x00,0xd0]

	dxr	%f0, %f0
	dxr	%f0, %f13
	dxr	%f8, %f8
	dxr	%f13, %f0

#CHECK: dxtr	%f0, %f0, %f0           # encoding: [0xb3,0xd9,0x00,0x00]
#CHECK: dxtr	%f0, %f0, %f13          # encoding: [0xb3,0xd9,0xd0,0x00]
#CHECK: dxtr	%f0, %f13, %f0          # encoding: [0xb3,0xd9,0x00,0x0d]
#CHECK: dxtr	%f13, %f0, %f0          # encoding: [0xb3,0xd9,0x00,0xd0]
#CHECK: dxtr	%f8, %f8, %f8           # encoding: [0xb3,0xd9,0x80,0x88]

	dxtr	%f0, %f0, %f0
	dxtr	%f0, %f0, %f13
	dxtr	%f0, %f13, %f0
	dxtr	%f13, %f0, %f0
	dxtr	%f8, %f8, %f8

#CHECK: ear	%r0, %a0                # encoding: [0xb2,0x4f,0x00,0x00]
#CHECK: ear	%r0, %a15               # encoding: [0xb2,0x4f,0x00,0x0f]
#CHECK: ear	%r15, %a0               # encoding: [0xb2,0x4f,0x00,0xf0]
#CHECK: ear	%r7, %a8                # encoding: [0xb2,0x4f,0x00,0x78]
#CHECK: ear	%r15, %a15              # encoding: [0xb2,0x4f,0x00,0xff]

	ear	%r0, %a0
	ear	%r0, %a15
	ear	%r15, %a0
	ear	%r7, %a8
	ear	%r15, %a15

#CHECK: ecag	%r0, %r0, 0             # encoding: [0xeb,0x00,0x00,0x00,0x00,0x4c]
#CHECK: ecag	%r0, %r15, 0            # encoding: [0xeb,0x0f,0x00,0x00,0x00,0x4c]
#CHECK: ecag	%r14, %r15, 0           # encoding: [0xeb,0xef,0x00,0x00,0x00,0x4c]
#CHECK: ecag	%r15, %r15, 0           # encoding: [0xeb,0xff,0x00,0x00,0x00,0x4c]
#CHECK: ecag	%r0, %r0, -524288       # encoding: [0xeb,0x00,0x00,0x00,0x80,0x4c]
#CHECK: ecag	%r0, %r0, -1            # encoding: [0xeb,0x00,0x0f,0xff,0xff,0x4c]
#CHECK: ecag	%r0, %r0, 0             # encoding: [0xeb,0x00,0x00,0x00,0x00,0x4c]
#CHECK: ecag	%r0, %r0, 1             # encoding: [0xeb,0x00,0x00,0x01,0x00,0x4c]
#CHECK: ecag	%r0, %r0, 524287        # encoding: [0xeb,0x00,0x0f,0xff,0x7f,0x4c]
#CHECK: ecag	%r0, %r0, 0(%r1)        # encoding: [0xeb,0x00,0x10,0x00,0x00,0x4c]
#CHECK: ecag	%r0, %r0, 0(%r15)       # encoding: [0xeb,0x00,0xf0,0x00,0x00,0x4c]
#CHECK: ecag	%r0, %r0, 524287(%r1)   # encoding: [0xeb,0x00,0x1f,0xff,0x7f,0x4c]
#CHECK: ecag	%r0, %r0, 524287(%r15)  # encoding: [0xeb,0x00,0xff,0xff,0x7f,0x4c]

	ecag	%r0,%r0,0
	ecag	%r0,%r15,0
	ecag	%r14,%r15,0
	ecag	%r15,%r15,0
	ecag	%r0,%r0,-524288
	ecag	%r0,%r0,-1
	ecag	%r0,%r0,0
	ecag	%r0,%r0,1
	ecag	%r0,%r0,524287
	ecag	%r0,%r0,0(%r1)
	ecag	%r0,%r0,0(%r15)
	ecag	%r0,%r0,524287(%r1)
	ecag	%r0,%r0,524287(%r15)

#CHECK: ecctr	%r0, %r0                # encoding: [0xb2,0xe4,0x00,0x00]
#CHECK: ecctr	%r0, %r15               # encoding: [0xb2,0xe4,0x00,0x0f]
#CHECK: ecctr	%r15, %r0               # encoding: [0xb2,0xe4,0x00,0xf0]
#CHECK: ecctr	%r7, %r8                # encoding: [0xb2,0xe4,0x00,0x78]

	ecctr	%r0,%r0
	ecctr	%r0,%r15
	ecctr	%r15,%r0
	ecctr	%r7,%r8

#CHECK: ecpga	%r0, %r0                # encoding: [0xb2,0xed,0x00,0x00]
#CHECK: ecpga	%r0, %r15               # encoding: [0xb2,0xed,0x00,0x0f]
#CHECK: ecpga	%r15, %r0               # encoding: [0xb2,0xed,0x00,0xf0]
#CHECK: ecpga	%r7, %r8                # encoding: [0xb2,0xed,0x00,0x78]

	ecpga	%r0,%r0
	ecpga	%r0,%r15
	ecpga	%r15,%r0
	ecpga	%r7,%r8

#CHECK: ectg    0, 0, %r0               # encoding: [0xc8,0x01,0x00,0x00,0x00,0x00]
#CHECK: ectg    0(%r1), 0(%r15), %r2    # encoding: [0xc8,0x21,0x10,0x00,0xf0,0x00]
#CHECK: ectg    1(%r1), 0(%r15), %r2    # encoding: [0xc8,0x21,0x10,0x01,0xf0,0x00]
#CHECK: ectg    4095(%r1), 0(%r15), %r2 # encoding: [0xc8,0x21,0x1f,0xff,0xf0,0x00]
#CHECK: ectg    0(%r1), 1(%r15), %r2    # encoding: [0xc8,0x21,0x10,0x00,0xf0,0x01]
#CHECK: ectg    0(%r1), 4095(%r15), %r2 # encoding: [0xc8,0x21,0x10,0x00,0xff,0xff]

        ectg    0,0,%r0
        ectg    0(%r1),0(%r15),%r2
        ectg    1(%r1),0(%r15),%r2
        ectg    4095(%r1),0(%r15),%r2
        ectg    0(%r1),1(%r15),%r2
        ectg    0(%r1),4095(%r15),%r2

#CHECK: ed	0(1), 0                 # encoding: [0xde,0x00,0x00,0x00,0x00,0x00]
#CHECK: ed	0(1), 0(%r1)            # encoding: [0xde,0x00,0x00,0x00,0x10,0x00]
#CHECK: ed	0(1), 0(%r15)           # encoding: [0xde,0x00,0x00,0x00,0xf0,0x00]
#CHECK: ed	0(1), 4095              # encoding: [0xde,0x00,0x00,0x00,0x0f,0xff]
#CHECK: ed	0(1), 4095(%r1)         # encoding: [0xde,0x00,0x00,0x00,0x1f,0xff]
#CHECK: ed	0(1), 4095(%r15)        # encoding: [0xde,0x00,0x00,0x00,0xff,0xff]
#CHECK: ed	0(1,%r1), 0             # encoding: [0xde,0x00,0x10,0x00,0x00,0x00]
#CHECK: ed	0(1,%r15), 0            # encoding: [0xde,0x00,0xf0,0x00,0x00,0x00]
#CHECK: ed	4095(1,%r1), 0          # encoding: [0xde,0x00,0x1f,0xff,0x00,0x00]
#CHECK: ed	4095(1,%r15), 0         # encoding: [0xde,0x00,0xff,0xff,0x00,0x00]
#CHECK: ed	0(256,%r1), 0           # encoding: [0xde,0xff,0x10,0x00,0x00,0x00]
#CHECK: ed	0(256,%r15), 0          # encoding: [0xde,0xff,0xf0,0x00,0x00,0x00]

	ed	0(1), 0
	ed	0(1), 0(%r1)
	ed	0(1), 0(%r15)
	ed	0(1), 4095
	ed	0(1), 4095(%r1)
	ed	0(1), 4095(%r15)
	ed	0(1,%r1), 0
	ed	0(1,%r15), 0
	ed	4095(1,%r1), 0
	ed	4095(1,%r15), 0
	ed	0(256,%r1), 0
	ed	0(256,%r15), 0

#CHECK: edmk	0(1), 0                 # encoding: [0xdf,0x00,0x00,0x00,0x00,0x00]
#CHECK: edmk	0(1), 0(%r1)            # encoding: [0xdf,0x00,0x00,0x00,0x10,0x00]
#CHECK: edmk	0(1), 0(%r15)           # encoding: [0xdf,0x00,0x00,0x00,0xf0,0x00]
#CHECK: edmk	0(1), 4095              # encoding: [0xdf,0x00,0x00,0x00,0x0f,0xff]
#CHECK: edmk	0(1), 4095(%r1)         # encoding: [0xdf,0x00,0x00,0x00,0x1f,0xff]
#CHECK: edmk	0(1), 4095(%r15)        # encoding: [0xdf,0x00,0x00,0x00,0xff,0xff]
#CHECK: edmk	0(1,%r1), 0             # encoding: [0xdf,0x00,0x10,0x00,0x00,0x00]
#CHECK: edmk	0(1,%r15), 0            # encoding: [0xdf,0x00,0xf0,0x00,0x00,0x00]
#CHECK: edmk	4095(1,%r1), 0          # encoding: [0xdf,0x00,0x1f,0xff,0x00,0x00]
#CHECK: edmk	4095(1,%r15), 0         # encoding: [0xdf,0x00,0xff,0xff,0x00,0x00]
#CHECK: edmk	0(256,%r1), 0           # encoding: [0xdf,0xff,0x10,0x00,0x00,0x00]
#CHECK: edmk	0(256,%r15), 0          # encoding: [0xdf,0xff,0xf0,0x00,0x00,0x00]

	edmk	0(1), 0
	edmk	0(1), 0(%r1)
	edmk	0(1), 0(%r15)
	edmk	0(1), 4095
	edmk	0(1), 4095(%r1)
	edmk	0(1), 4095(%r15)
	edmk	0(1,%r1), 0
	edmk	0(1,%r15), 0
	edmk	4095(1,%r1), 0
	edmk	4095(1,%r15), 0
	edmk	0(256,%r1), 0
	edmk	0(256,%r15), 0

#CHECK: eedtr	%r0, %f9                # encoding: [0xb3,0xe5,0x00,0x09]
#CHECK: eedtr	%r0, %f15               # encoding: [0xb3,0xe5,0x00,0x0f]
#CHECK: eedtr	%r15, %f0               # encoding: [0xb3,0xe5,0x00,0xf0]
#CHECK: eedtr	%r15, %f9               # encoding: [0xb3,0xe5,0x00,0xf9]

	eedtr	%r0,%f9
	eedtr	%r0,%f15
	eedtr	%r15,%f0
	eedtr	%r15,%f9

#CHECK: eextr	%r0, %f8                # encoding: [0xb3,0xed,0x00,0x08]
#CHECK: eextr	%r0, %f13               # encoding: [0xb3,0xed,0x00,0x0d]
#CHECK: eextr	%r13, %f0               # encoding: [0xb3,0xed,0x00,0xd0]
#CHECK: eextr	%r13, %f9               # encoding: [0xb3,0xed,0x00,0xd9]

	eextr	%r0,%f8
	eextr	%r0,%f13
	eextr	%r13,%f0
	eextr	%r13,%f9

#CHECK: efpc	%r0                     # encoding: [0xb3,0x8c,0x00,0x00]
#CHECK: efpc	%r1                     # encoding: [0xb3,0x8c,0x00,0x10]
#CHECK: efpc	%r15                    # encoding: [0xb3,0x8c,0x00,0xf0]

	efpc	%r0
	efpc	%r1
	efpc	%r15

#CHECK: epar	%r0                     # encoding: [0xb2,0x26,0x00,0x00]
#CHECK: epar	%r1                     # encoding: [0xb2,0x26,0x00,0x10]
#CHECK: epar	%r15                    # encoding: [0xb2,0x26,0x00,0xf0]

	epar	%r0
	epar	%r1
	epar	%r15

#CHECK: epair	%r0                     # encoding: [0xb9,0x9a,0x00,0x00]
#CHECK: epair	%r1                     # encoding: [0xb9,0x9a,0x00,0x10]
#CHECK: epair	%r15                    # encoding: [0xb9,0x9a,0x00,0xf0]

	epair	%r0
	epair	%r1
	epair	%r15

#CHECK: epctr	%r0, %r0                # encoding: [0xb2,0xe5,0x00,0x00]
#CHECK: epctr	%r0, %r15               # encoding: [0xb2,0xe5,0x00,0x0f]
#CHECK: epctr	%r15, %r0               # encoding: [0xb2,0xe5,0x00,0xf0]
#CHECK: epctr	%r7, %r8                # encoding: [0xb2,0xe5,0x00,0x78]

	epctr	%r0,%r0
	epctr	%r0,%r15
	epctr	%r15,%r0
	epctr	%r7,%r8

#CHECK: epsw	%r0, %r8                # encoding: [0xb9,0x8d,0x00,0x08]
#CHECK: epsw	%r0, %r15               # encoding: [0xb9,0x8d,0x00,0x0f]
#CHECK: epsw	%r15, %r0               # encoding: [0xb9,0x8d,0x00,0xf0]
#CHECK: epsw	%r15, %r8               # encoding: [0xb9,0x8d,0x00,0xf8]

	epsw	%r0, %r8
	epsw	%r0, %r15
	epsw	%r15, %r0
	epsw	%r15, %r8

#CHECK: ereg	%r0, %r0                # encoding: [0xb2,0x49,0x00,0x00]
#CHECK: ereg	%r0, %r15               # encoding: [0xb2,0x49,0x00,0x0f]
#CHECK: ereg	%r15, %r0               # encoding: [0xb2,0x49,0x00,0xf0]
#CHECK: ereg	%r7, %r8                # encoding: [0xb2,0x49,0x00,0x78]

	ereg	%r0,%r0
	ereg	%r0,%r15
	ereg	%r15,%r0
	ereg	%r7,%r8

#CHECK: eregg	%r0, %r0                # encoding: [0xb9,0x0e,0x00,0x00]
#CHECK: eregg	%r0, %r15               # encoding: [0xb9,0x0e,0x00,0x0f]
#CHECK: eregg	%r15, %r0               # encoding: [0xb9,0x0e,0x00,0xf0]
#CHECK: eregg	%r7, %r8                # encoding: [0xb9,0x0e,0x00,0x78]

	eregg	%r0,%r0
	eregg	%r0,%r15
	eregg	%r15,%r0
	eregg	%r7,%r8

#CHECK: esar	%r0                     # encoding: [0xb2,0x27,0x00,0x00]
#CHECK: esar	%r1                     # encoding: [0xb2,0x27,0x00,0x10]
#CHECK: esar	%r15                    # encoding: [0xb2,0x27,0x00,0xf0]

	esar	%r0
	esar	%r1
	esar	%r15

#CHECK: esair	%r0                     # encoding: [0xb9,0x9b,0x00,0x00]
#CHECK: esair	%r1                     # encoding: [0xb9,0x9b,0x00,0x10]
#CHECK: esair	%r15                    # encoding: [0xb9,0x9b,0x00,0xf0]

	esair	%r0
	esair	%r1
	esair	%r15

#CHECK: esdtr	%r0, %f9                # encoding: [0xb3,0xe7,0x00,0x09]
#CHECK: esdtr	%r0, %f15               # encoding: [0xb3,0xe7,0x00,0x0f]
#CHECK: esdtr	%r15, %f0               # encoding: [0xb3,0xe7,0x00,0xf0]
#CHECK: esdtr	%r15, %f9               # encoding: [0xb3,0xe7,0x00,0xf9]

	esdtr	%r0,%f9
	esdtr	%r0,%f15
	esdtr	%r15,%f0
	esdtr	%r15,%f9

#CHECK: esea	%r0                     # encoding: [0xb9,0x9d,0x00,0x00]
#CHECK: esea	%r1                     # encoding: [0xb9,0x9d,0x00,0x10]
#CHECK: esea	%r15                    # encoding: [0xb9,0x9d,0x00,0xf0]

	esea	%r0
	esea	%r1
	esea	%r15

#CHECK: esta	%r0, %r0                # encoding: [0xb2,0x4a,0x00,0x00]
#CHECK: esta	%r0, %r15               # encoding: [0xb2,0x4a,0x00,0x0f]
#CHECK: esta	%r14, %r0               # encoding: [0xb2,0x4a,0x00,0xe0]
#CHECK: esta	%r6, %r8                # encoding: [0xb2,0x4a,0x00,0x68]

	esta	%r0,%r0
	esta	%r0,%r15
	esta	%r14,%r0
	esta	%r6,%r8

#CHECK: esxtr	%r0, %f8                # encoding: [0xb3,0xef,0x00,0x08]
#CHECK: esxtr	%r0, %f13               # encoding: [0xb3,0xef,0x00,0x0d]
#CHECK: esxtr	%r13, %f0               # encoding: [0xb3,0xef,0x00,0xd0]
#CHECK: esxtr	%r13, %f9               # encoding: [0xb3,0xef,0x00,0xd9]

	esxtr	%r0,%f8
	esxtr	%r0,%f13
	esxtr	%r13,%f0
	esxtr	%r13,%f9

#CHECK: ex	%r0, 0                  # encoding: [0x44,0x00,0x00,0x00]
#CHECK: ex	%r0, 4095               # encoding: [0x44,0x00,0x0f,0xff]
#CHECK: ex	%r0, 0(%r1)             # encoding: [0x44,0x00,0x10,0x00]
#CHECK: ex	%r0, 0(%r15)            # encoding: [0x44,0x00,0xf0,0x00]
#CHECK: ex	%r0, 4095(%r1,%r15)     # encoding: [0x44,0x01,0xff,0xff]
#CHECK: ex	%r0, 4095(%r15,%r1)     # encoding: [0x44,0x0f,0x1f,0xff]
#CHECK: ex	%r15, 0                 # encoding: [0x44,0xf0,0x00,0x00]

	ex	%r0, 0
	ex	%r0, 4095
	ex	%r0, 0(%r1)
	ex	%r0, 0(%r15)
	ex	%r0, 4095(%r1,%r15)
	ex	%r0, 4095(%r15,%r1)
	ex	%r15, 0

#CHECK: exrl	%r0, .[[LAB:L.*]]-4294967296 # encoding: [0xc6,0x00,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-4294967296)+2, kind: FK_390_PC32DBL
	exrl	%r0, -0x100000000
#CHECK: exrl	%r0, .[[LAB:L.*]]-2	# encoding: [0xc6,0x00,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC32DBL
	exrl	%r0, -2
#CHECK: exrl	%r0, .[[LAB:L.*]]	# encoding: [0xc6,0x00,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC32DBL
	exrl	%r0, 0
#CHECK: exrl	%r0, .[[LAB:L.*]]+4294967294 # encoding: [0xc6,0x00,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+4294967294)+2, kind: FK_390_PC32DBL
	exrl	%r0, 0xfffffffe

#CHECK: exrl	%r0, foo                # encoding: [0xc6,0x00,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: exrl	%r15, foo               # encoding: [0xc6,0xf0,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL

	exrl	%r0,foo
	exrl	%r15,foo

#CHECK: exrl	%r3, bar+100            # encoding: [0xc6,0x30,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: exrl	%r4, bar+100            # encoding: [0xc6,0x40,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL

	exrl	%r3,bar+100
	exrl	%r4,bar+100

#CHECK: exrl	%r7, frob@PLT           # encoding: [0xc6,0x70,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL
#CHECK: exrl	%r8, frob@PLT           # encoding: [0xc6,0x80,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL

	exrl	%r7,frob@PLT
	exrl	%r8,frob@PLT

#CHECK: fidbr	%f0, 0, %f0             # encoding: [0xb3,0x5f,0x00,0x00]
#CHECK: fidbr	%f0, 0, %f15            # encoding: [0xb3,0x5f,0x00,0x0f]
#CHECK: fidbr	%f0, 15, %f0            # encoding: [0xb3,0x5f,0xf0,0x00]
#CHECK: fidbr	%f4, 5, %f6             # encoding: [0xb3,0x5f,0x50,0x46]
#CHECK: fidbr	%f15, 0, %f0            # encoding: [0xb3,0x5f,0x00,0xf0]

	fidbr	%f0, 0, %f0
	fidbr	%f0, 0, %f15
	fidbr	%f0, 15, %f0
	fidbr	%f4, 5, %f6
	fidbr	%f15, 0, %f0

#CHECK: fidr	%f0, %f0                # encoding: [0xb3,0x7f,0x00,0x00]
#CHECK: fidr	%f0, %f15               # encoding: [0xb3,0x7f,0x00,0x0f]
#CHECK: fidr	%f4, %f6                # encoding: [0xb3,0x7f,0x00,0x46]
#CHECK: fidr	%f15, %f0               # encoding: [0xb3,0x7f,0x00,0xf0]

	fidr	%f0, %f0
	fidr	%f0, %f15
	fidr	%f4, %f6
	fidr	%f15, %f0

#CHECK: fidtr	%f0, 0, %f0, 0          # encoding: [0xb3,0xd7,0x00,0x00]
#CHECK: fidtr	%f0, 0, %f0, 15         # encoding: [0xb3,0xd7,0x0f,0x00]
#CHECK: fidtr	%f0, 0, %f15, 0         # encoding: [0xb3,0xd7,0x00,0x0f]
#CHECK: fidtr	%f0, 15, %f0, 0         # encoding: [0xb3,0xd7,0xf0,0x00]
#CHECK: fidtr	%f4, 5, %f6, 7          # encoding: [0xb3,0xd7,0x57,0x46]
#CHECK: fidtr	%f15, 0, %f0, 0         # encoding: [0xb3,0xd7,0x00,0xf0]

	fidtr	%f0, 0, %f0, 0
	fidtr	%f0, 0, %f0, 15
	fidtr	%f0, 0, %f15, 0
	fidtr	%f0, 15, %f0, 0
	fidtr	%f4, 5, %f6, 7
	fidtr	%f15, 0, %f0, 0

#CHECK: fiebr	%f0, 0, %f0             # encoding: [0xb3,0x57,0x00,0x00]
#CHECK: fiebr	%f0, 0, %f15            # encoding: [0xb3,0x57,0x00,0x0f]
#CHECK: fiebr	%f0, 15, %f0            # encoding: [0xb3,0x57,0xf0,0x00]
#CHECK: fiebr	%f4, 5, %f6             # encoding: [0xb3,0x57,0x50,0x46]
#CHECK: fiebr	%f15, 0, %f0            # encoding: [0xb3,0x57,0x00,0xf0]

	fiebr	%f0, 0, %f0
	fiebr	%f0, 0, %f15
	fiebr	%f0, 15, %f0
	fiebr	%f4, 5, %f6
	fiebr	%f15, 0, %f0

#CHECK: fier	%f0, %f0                # encoding: [0xb3,0x77,0x00,0x00]
#CHECK: fier	%f0, %f15               # encoding: [0xb3,0x77,0x00,0x0f]
#CHECK: fier	%f4, %f6                # encoding: [0xb3,0x77,0x00,0x46]
#CHECK: fier	%f15, %f0               # encoding: [0xb3,0x77,0x00,0xf0]

	fier	%f0, %f0
	fier	%f0, %f15
	fier	%f4, %f6
	fier	%f15, %f0

#CHECK: fixbr	%f0, 0, %f0             # encoding: [0xb3,0x47,0x00,0x00]
#CHECK: fixbr	%f0, 0, %f13            # encoding: [0xb3,0x47,0x00,0x0d]
#CHECK: fixbr	%f0, 15, %f0            # encoding: [0xb3,0x47,0xf0,0x00]
#CHECK: fixbr	%f4, 5, %f8             # encoding: [0xb3,0x47,0x50,0x48]
#CHECK: fixbr	%f13, 0, %f0            # encoding: [0xb3,0x47,0x00,0xd0]

	fixbr	%f0, 0, %f0
	fixbr	%f0, 0, %f13
	fixbr	%f0, 15, %f0
	fixbr	%f4, 5, %f8
	fixbr	%f13, 0, %f0

#CHECK: fixr	%f0, %f0                # encoding: [0xb3,0x67,0x00,0x00]
#CHECK: fixr	%f0, %f13               # encoding: [0xb3,0x67,0x00,0x0d]
#CHECK: fixr	%f4, %f8                # encoding: [0xb3,0x67,0x00,0x48]
#CHECK: fixr	%f13, %f0               # encoding: [0xb3,0x67,0x00,0xd0]

	fixr	%f0, %f0
	fixr	%f0, %f13
	fixr	%f4, %f8
	fixr	%f13, %f0

#CHECK: fixtr	%f0, 0, %f0, 0          # encoding: [0xb3,0xdf,0x00,0x00]
#CHECK: fixtr	%f0, 0, %f0, 15         # encoding: [0xb3,0xdf,0x0f,0x00]
#CHECK: fixtr	%f0, 0, %f13, 0         # encoding: [0xb3,0xdf,0x00,0x0d]
#CHECK: fixtr	%f0, 15, %f0, 0         # encoding: [0xb3,0xdf,0xf0,0x00]
#CHECK: fixtr	%f4, 5, %f8, 9          # encoding: [0xb3,0xdf,0x59,0x48]
#CHECK: fixtr	%f13, 0, %f0, 0         # encoding: [0xb3,0xdf,0x00,0xd0]

	fixtr	%f0, 0, %f0, 0
	fixtr	%f0, 0, %f0, 15
	fixtr	%f0, 0, %f13, 0
	fixtr	%f0, 15, %f0, 0
	fixtr	%f4, 5, %f8, 9
	fixtr	%f13, 0, %f0, 0

#CHECK: flogr	%r0, %r0                # encoding: [0xb9,0x83,0x00,0x00]
#CHECK: flogr	%r0, %r15               # encoding: [0xb9,0x83,0x00,0x0f]
#CHECK: flogr	%r10, %r9               # encoding: [0xb9,0x83,0x00,0xa9]
#CHECK: flogr	%r14, %r0               # encoding: [0xb9,0x83,0x00,0xe0]

	flogr	%r0, %r0
	flogr	%r0, %r15
	flogr	%r10, %r9
	flogr	%r14, %r0

#CHECK: hdr	%f0, %f0                # encoding: [0x24,0x00]
#CHECK: hdr	%f0, %f15               # encoding: [0x24,0x0f]
#CHECK: hdr	%f7, %f8                # encoding: [0x24,0x78]
#CHECK: hdr	%f15, %f0               # encoding: [0x24,0xf0]

	hdr	%f0, %f0
	hdr	%f0, %f15
	hdr	%f7, %f8
	hdr	%f15, %f0

#CHECK: her	%f0, %f0                # encoding: [0x34,0x00]
#CHECK: her	%f0, %f15               # encoding: [0x34,0x0f]
#CHECK: her	%f7, %f8                # encoding: [0x34,0x78]
#CHECK: her	%f15, %f0               # encoding: [0x34,0xf0]

	her	%f0, %f0
	her	%f0, %f15
	her	%f7, %f8
	her	%f15, %f0

#CHECK: hsch                            # encoding: [0xb2,0x31,0x00,0x00]

	hsch

#CHECK: iac	%r0                     # encoding: [0xb2,0x24,0x00,0x00]
#CHECK: iac	%r1                     # encoding: [0xb2,0x24,0x00,0x10]
#CHECK: iac	%r15                    # encoding: [0xb2,0x24,0x00,0xf0]

	iac	%r0
	iac	%r1
	iac	%r15

#CHECK: ic	%r0, 0                  # encoding: [0x43,0x00,0x00,0x00]
#CHECK: ic	%r0, 4095               # encoding: [0x43,0x00,0x0f,0xff]
#CHECK: ic	%r0, 0(%r1)             # encoding: [0x43,0x00,0x10,0x00]
#CHECK: ic	%r0, 0(%r15)            # encoding: [0x43,0x00,0xf0,0x00]
#CHECK: ic	%r0, 4095(%r1,%r15)     # encoding: [0x43,0x01,0xff,0xff]
#CHECK: ic	%r0, 4095(%r15,%r1)     # encoding: [0x43,0x0f,0x1f,0xff]
#CHECK: ic	%r15, 0                 # encoding: [0x43,0xf0,0x00,0x00]

	ic	%r0, 0
	ic	%r0, 4095
	ic	%r0, 0(%r1)
	ic	%r0, 0(%r15)
	ic	%r0, 4095(%r1,%r15)
	ic	%r0, 4095(%r15,%r1)
	ic	%r15, 0

#CHECK: icm	%r0, 0, 0               # encoding: [0xbf,0x00,0x00,0x00]
#CHECK: icm	%r0, 15, 4095           # encoding: [0xbf,0x0f,0x0f,0xff]
#CHECK: icm	%r0, 0, 0(%r1)          # encoding: [0xbf,0x00,0x10,0x00]
#CHECK: icm	%r0, 0, 0(%r15)         # encoding: [0xbf,0x00,0xf0,0x00]
#CHECK: icm	%r15, 15, 4095(%r1)     # encoding: [0xbf,0xff,0x1f,0xff]
#CHECK: icm	%r0, 0, 4095(%r15)      # encoding: [0xbf,0x00,0xff,0xff]
#CHECK: icm	%r15, 0, 0              # encoding: [0xbf,0xf0,0x00,0x00]

	icm	%r0, 0, 0
	icm	%r0, 15, 4095
	icm	%r0, 0, 0(%r1)
	icm	%r0, 0, 0(%r15)
	icm	%r15, 15, 4095(%r1)
	icm	%r0, 0, 4095(%r15)
	icm	%r15, 0, 0

#CHECK: icmh	%r0, 0, -524288            # encoding: [0xeb,0x00,0x00,0x00,0x80,0x80]
#CHECK: icmh	%r0, 0, -1                 # encoding: [0xeb,0x00,0x0f,0xff,0xff,0x80]
#CHECK: icmh	%r0, 15, 0                 # encoding: [0xeb,0x0f,0x00,0x00,0x00,0x80]
#CHECK: icmh	%r0, 15, 1                 # encoding: [0xeb,0x0f,0x00,0x01,0x00,0x80]
#CHECK: icmh	%r0, 8, 524287             # encoding: [0xeb,0x08,0x0f,0xff,0x7f,0x80]
#CHECK: icmh	%r0, 8, 0(%r1)             # encoding: [0xeb,0x08,0x10,0x00,0x00,0x80]
#CHECK: icmh	%r0, 4, 0(%r15)            # encoding: [0xeb,0x04,0xf0,0x00,0x00,0x80]
#CHECK: icmh	%r0, 4, 524287(%r15)       # encoding: [0xeb,0x04,0xff,0xff,0x7f,0x80]
#CHECK: icmh	%r0, 0, 524287(%r1)        # encoding: [0xeb,0x00,0x1f,0xff,0x7f,0x80]
#CHECK: icmh	%r15, 0, 0                 # encoding: [0xeb,0xf0,0x00,0x00,0x00,0x80]

	icmh	%r0, 0, -524288
	icmh	%r0, 0, -1
	icmh	%r0, 15, 0
	icmh	%r0, 15, 1
	icmh	%r0, 8, 524287
	icmh	%r0, 8, 0(%r1)
	icmh	%r0, 4, 0(%r15)
	icmh	%r0, 4, 524287(%r15)
	icmh	%r0, 0, 524287(%r1)
	icmh	%r15, 0, 0

#CHECK: icmy	%r0, 0, -524288            # encoding: [0xeb,0x00,0x00,0x00,0x80,0x81]
#CHECK: icmy	%r0, 0, -1                 # encoding: [0xeb,0x00,0x0f,0xff,0xff,0x81]
#CHECK: icmy	%r0, 15, 0                 # encoding: [0xeb,0x0f,0x00,0x00,0x00,0x81]
#CHECK: icmy	%r0, 15, 1                 # encoding: [0xeb,0x0f,0x00,0x01,0x00,0x81]
#CHECK: icmy	%r0, 8, 524287             # encoding: [0xeb,0x08,0x0f,0xff,0x7f,0x81]
#CHECK: icmy	%r0, 8, 0(%r1)             # encoding: [0xeb,0x08,0x10,0x00,0x00,0x81]
#CHECK: icmy	%r0, 4, 0(%r15)            # encoding: [0xeb,0x04,0xf0,0x00,0x00,0x81]
#CHECK: icmy	%r0, 4, 524287(%r15)       # encoding: [0xeb,0x04,0xff,0xff,0x7f,0x81]
#CHECK: icmy	%r0, 0, 524287(%r1)        # encoding: [0xeb,0x00,0x1f,0xff,0x7f,0x81]
#CHECK: icmy	%r15, 0, 0                 # encoding: [0xeb,0xf0,0x00,0x00,0x00,0x81]

	icmy	%r0, 0, -524288
	icmy	%r0, 0, -1
	icmy	%r0, 15, 0
	icmy	%r0, 15, 1
	icmy	%r0, 8, 524287
	icmy	%r0, 8, 0(%r1)
	icmy	%r0, 4, 0(%r15)
	icmy	%r0, 4, 524287(%r15)
	icmy	%r0, 0, 524287(%r1)
	icmy	%r15, 0, 0

#CHECK: icy	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x73]
#CHECK: icy	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x73]
#CHECK: icy	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x73]
#CHECK: icy	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x73]
#CHECK: icy	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x73]
#CHECK: icy	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x73]
#CHECK: icy	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x73]
#CHECK: icy	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x73]
#CHECK: icy	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x73]
#CHECK: icy	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x73]

	icy	%r0, -524288
	icy	%r0, -1
	icy	%r0, 0
	icy	%r0, 1
	icy	%r0, 524287
	icy	%r0, 0(%r1)
	icy	%r0, 0(%r15)
	icy	%r0, 524287(%r1,%r15)
	icy	%r0, 524287(%r15,%r1)
	icy	%r15, 0

#CHECK: idte	%r0, %r0, %r0           # encoding: [0xb9,0x8e,0x00,0x00]
#CHECK: idte	%r0, %r0, %r15          # encoding: [0xb9,0x8e,0x00,0x0f]
#CHECK: idte	%r0, %r15, %r0          # encoding: [0xb9,0x8e,0xf0,0x00]
#CHECK: idte	%r15, %r0, %r0          # encoding: [0xb9,0x8e,0x00,0xf0]
#CHECK: idte	%r0, %r0, %r0, 15       # encoding: [0xb9,0x8e,0x0f,0x00]
#CHECK: idte	%r4, %r5, %r6, 7        # encoding: [0xb9,0x8e,0x57,0x46]

	idte	%r0, %r0, %r0
	idte	%r0, %r0, %r15
	idte	%r0, %r15, %r0
	idte	%r15, %r0, %r0
	idte	%r0, %r0, %r0, 15
	idte	%r4, %r5, %r6, 7

#CHECK: iedtr	%f0, %f0, %r0           # encoding: [0xb3,0xf6,0x00,0x00]
#CHECK: iedtr	%f0, %f0, %r15          # encoding: [0xb3,0xf6,0x00,0x0f]
#CHECK: iedtr	%f0, %f15, %r0          # encoding: [0xb3,0xf6,0xf0,0x00]
#CHECK: iedtr	%f15, %f0, %r0          # encoding: [0xb3,0xf6,0x00,0xf0]
#CHECK: iedtr	%f1, %f2, %r3           # encoding: [0xb3,0xf6,0x20,0x13]
#CHECK: iedtr	%f15, %f15, %r15        # encoding: [0xb3,0xf6,0xf0,0xff]

	iedtr	%f0, %f0, %r0
	iedtr	%f0, %f0, %r15
	iedtr	%f0, %f15, %r0
	iedtr	%f15, %f0, %r0
	iedtr	%f1, %f2, %r3
	iedtr	%f15, %f15, %r15

#CHECK: iextr	%f0, %f0, %r0           # encoding: [0xb3,0xfe,0x00,0x00]
#CHECK: iextr	%f0, %f0, %r13          # encoding: [0xb3,0xfe,0x00,0x0d]
#CHECK: iextr	%f0, %f13, %r0          # encoding: [0xb3,0xfe,0xd0,0x00]
#CHECK: iextr	%f13, %f0, %r0          # encoding: [0xb3,0xfe,0x00,0xd0]
#CHECK: iextr	%f1, %f8, %r4           # encoding: [0xb3,0xfe,0x80,0x14]
#CHECK: iextr	%f13, %f13, %r13        # encoding: [0xb3,0xfe,0xd0,0xdd]

	iextr	%f0, %f0, %r0
	iextr	%f0, %f0, %r13
	iextr	%f0, %f13, %r0
	iextr	%f13, %f0, %r0
	iextr	%f1, %f8, %r4
	iextr	%f13, %f13, %r13

#CHECK: iihf	%r0, 0                  # encoding: [0xc0,0x08,0x00,0x00,0x00,0x00]
#CHECK: iihf	%r0, 4294967295         # encoding: [0xc0,0x08,0xff,0xff,0xff,0xff]
#CHECK: iihf	%r15, 0                 # encoding: [0xc0,0xf8,0x00,0x00,0x00,0x00]

	iihf	%r0, 0
	iihf	%r0, 0xffffffff
	iihf	%r15, 0

#CHECK: iihh	%r0, 0                  # encoding: [0xa5,0x00,0x00,0x00]
#CHECK: iihh	%r0, 32768              # encoding: [0xa5,0x00,0x80,0x00]
#CHECK: iihh	%r0, 65535              # encoding: [0xa5,0x00,0xff,0xff]
#CHECK: iihh	%r15, 0                 # encoding: [0xa5,0xf0,0x00,0x00]

	iihh	%r0, 0
	iihh	%r0, 0x8000
	iihh	%r0, 0xffff
	iihh	%r15, 0

#CHECK: iihl	%r0, 0                  # encoding: [0xa5,0x01,0x00,0x00]
#CHECK: iihl	%r0, 32768              # encoding: [0xa5,0x01,0x80,0x00]
#CHECK: iihl	%r0, 65535              # encoding: [0xa5,0x01,0xff,0xff]
#CHECK: iihl	%r15, 0                 # encoding: [0xa5,0xf1,0x00,0x00]

	iihl	%r0, 0
	iihl	%r0, 0x8000
	iihl	%r0, 0xffff
	iihl	%r15, 0

#CHECK: iilf	%r0, 0                  # encoding: [0xc0,0x09,0x00,0x00,0x00,0x00]
#CHECK: iilf	%r0, 4294967295         # encoding: [0xc0,0x09,0xff,0xff,0xff,0xff]
#CHECK: iilf	%r15, 0                 # encoding: [0xc0,0xf9,0x00,0x00,0x00,0x00]
#CHECK: iilf	%r15, 0                 # encoding: [0xc0,0xf9,0x00,0x00,0x00,0x00]

	iilf	%r0, 0
	iilf	%r0, 0xffffffff
	iilf	%r15, 0
	lfi	%r15, 0

#CHECK: iilh	%r0, 0                  # encoding: [0xa5,0x02,0x00,0x00]
#CHECK: iilh	%r0, 32768              # encoding: [0xa5,0x02,0x80,0x00]
#CHECK: iilh	%r0, 65535              # encoding: [0xa5,0x02,0xff,0xff]
#CHECK: iilh	%r15, 0                 # encoding: [0xa5,0xf2,0x00,0x00]

	iilh	%r0, 0
	iilh	%r0, 0x8000
	iilh	%r0, 0xffff
	iilh	%r15, 0

#CHECK: iill	%r0, 0                  # encoding: [0xa5,0x03,0x00,0x00]
#CHECK: iill	%r0, 32768              # encoding: [0xa5,0x03,0x80,0x00]
#CHECK: iill	%r0, 65535              # encoding: [0xa5,0x03,0xff,0xff]
#CHECK: iill	%r15, 0                 # encoding: [0xa5,0xf3,0x00,0x00]

	iill	%r0, 0
	iill	%r0, 0x8000
	iill	%r0, 0xffff
	iill	%r15, 0

#CHECK: ipk                             # encoding: [0xb2,0x0b,0x00,0x00]

	ipk

#CHECK: ipm	%r0                     # encoding: [0xb2,0x22,0x00,0x00]
#CHECK: ipm	%r1                     # encoding: [0xb2,0x22,0x00,0x10]
#CHECK: ipm	%r15                    # encoding: [0xb2,0x22,0x00,0xf0]

	ipm	%r0
	ipm	%r1
	ipm	%r15

#CHECK: ipte	%r0, %r0                # encoding: [0xb2,0x21,0x00,0x00]
#CHECK: ipte	%r0, %r15               # encoding: [0xb2,0x21,0x00,0x0f]
#CHECK: ipte	%r15, %r0               # encoding: [0xb2,0x21,0x00,0xf0]
#CHECK: ipte	%r0, %r0, %r15          # encoding: [0xb2,0x21,0xf0,0x00]
#CHECK: ipte	%r0, %r0, %r0, 15       # encoding: [0xb2,0x21,0x0f,0x00]
#CHECK: ipte	%r7, %r8, %r9, 10       # encoding: [0xb2,0x21,0x9a,0x78]

	ipte	%r0, %r0
	ipte	%r0, %r15
	ipte	%r15, %r0
	ipte	%r0, %r0, %r15
	ipte	%r0, %r0, %r0, 15
	ipte	%r7, %r8, %r9, 10

#CHECK: iske	%r0, %r0                # encoding: [0xb2,0x29,0x00,0x00]
#CHECK: iske	%r0, %r15               # encoding: [0xb2,0x29,0x00,0x0f]
#CHECK: iske	%r15, %r0               # encoding: [0xb2,0x29,0x00,0xf0]
#CHECK: iske	%r7, %r8                # encoding: [0xb2,0x29,0x00,0x78]

	iske	%r0,%r0
	iske	%r0,%r15
	iske	%r15,%r0
	iske	%r7,%r8

#CHECK: ivsk	%r0, %r0                # encoding: [0xb2,0x23,0x00,0x00]
#CHECK: ivsk	%r0, %r15               # encoding: [0xb2,0x23,0x00,0x0f]
#CHECK: ivsk	%r15, %r0               # encoding: [0xb2,0x23,0x00,0xf0]
#CHECK: ivsk	%r7, %r8                # encoding: [0xb2,0x23,0x00,0x78]

	ivsk	%r0,%r0
	ivsk	%r0,%r15
	ivsk	%r15,%r0
	ivsk	%r7,%r8

#CHECK: kdb	%f0, 0                  # encoding: [0xed,0x00,0x00,0x00,0x00,0x18]
#CHECK: kdb	%f0, 4095               # encoding: [0xed,0x00,0x0f,0xff,0x00,0x18]
#CHECK: kdb	%f0, 0(%r1)             # encoding: [0xed,0x00,0x10,0x00,0x00,0x18]
#CHECK: kdb	%f0, 0(%r15)            # encoding: [0xed,0x00,0xf0,0x00,0x00,0x18]
#CHECK: kdb	%f0, 4095(%r1,%r15)     # encoding: [0xed,0x01,0xff,0xff,0x00,0x18]
#CHECK: kdb	%f0, 4095(%r15,%r1)     # encoding: [0xed,0x0f,0x1f,0xff,0x00,0x18]
#CHECK: kdb	%f15, 0                 # encoding: [0xed,0xf0,0x00,0x00,0x00,0x18]

	kdb	%f0, 0
	kdb	%f0, 4095
	kdb	%f0, 0(%r1)
	kdb	%f0, 0(%r15)
	kdb	%f0, 4095(%r1,%r15)
	kdb	%f0, 4095(%r15,%r1)
	kdb	%f15, 0

#CHECK: kdbr	%f0, %f0                # encoding: [0xb3,0x18,0x00,0x00]
#CHECK: kdbr	%f0, %f15               # encoding: [0xb3,0x18,0x00,0x0f]
#CHECK: kdbr	%f7, %f8                # encoding: [0xb3,0x18,0x00,0x78]
#CHECK: kdbr	%f15, %f0               # encoding: [0xb3,0x18,0x00,0xf0]

	kdbr	%f0, %f0
	kdbr	%f0, %f15
	kdbr	%f7, %f8
	kdbr	%f15, %f0

#CHECK: kdtr	%f0, %f0                # encoding: [0xb3,0xe0,0x00,0x00]
#CHECK: kdtr	%f0, %f15               # encoding: [0xb3,0xe0,0x00,0x0f]
#CHECK: kdtr	%f7, %f8                # encoding: [0xb3,0xe0,0x00,0x78]
#CHECK: kdtr	%f15, %f0               # encoding: [0xb3,0xe0,0x00,0xf0]

	kdtr	%f0, %f0
	kdtr	%f0, %f15
	kdtr	%f7, %f8
	kdtr	%f15, %f0

#CHECK: keb	%f0, 0                  # encoding: [0xed,0x00,0x00,0x00,0x00,0x08]
#CHECK: keb	%f0, 4095               # encoding: [0xed,0x00,0x0f,0xff,0x00,0x08]
#CHECK: keb	%f0, 0(%r1)             # encoding: [0xed,0x00,0x10,0x00,0x00,0x08]
#CHECK: keb	%f0, 0(%r15)            # encoding: [0xed,0x00,0xf0,0x00,0x00,0x08]
#CHECK: keb	%f0, 4095(%r1,%r15)     # encoding: [0xed,0x01,0xff,0xff,0x00,0x08]
#CHECK: keb	%f0, 4095(%r15,%r1)     # encoding: [0xed,0x0f,0x1f,0xff,0x00,0x08]
#CHECK: keb	%f15, 0                 # encoding: [0xed,0xf0,0x00,0x00,0x00,0x08]

	keb	%f0, 0
	keb	%f0, 4095
	keb	%f0, 0(%r1)
	keb	%f0, 0(%r15)
	keb	%f0, 4095(%r1,%r15)
	keb	%f0, 4095(%r15,%r1)
	keb	%f15, 0

#CHECK: kebr	%f0, %f0                # encoding: [0xb3,0x08,0x00,0x00]
#CHECK: kebr	%f0, %f15               # encoding: [0xb3,0x08,0x00,0x0f]
#CHECK: kebr	%f7, %f8                # encoding: [0xb3,0x08,0x00,0x78]
#CHECK: kebr	%f15, %f0               # encoding: [0xb3,0x08,0x00,0xf0]

	kebr	%f0, %f0
	kebr	%f0, %f15
	kebr	%f7, %f8
	kebr	%f15, %f0

#CHECK: kimd	%r0, %r2                # encoding: [0xb9,0x3e,0x00,0x02]
#CHECK: kimd	%r0, %r14               # encoding: [0xb9,0x3e,0x00,0x0e]
#CHECK: kimd	%r15, %r2               # encoding: [0xb9,0x3e,0x00,0xf2]
#CHECK: kimd	%r7, %r10               # encoding: [0xb9,0x3e,0x00,0x7a]

	kimd	%r0, %r2
	kimd	%r0, %r14
	kimd	%r15, %r2
	kimd	%r7, %r10

#CHECK: klmd	%r0, %r2                # encoding: [0xb9,0x3f,0x00,0x02]
#CHECK: klmd	%r0, %r14               # encoding: [0xb9,0x3f,0x00,0x0e]
#CHECK: klmd	%r15, %r2               # encoding: [0xb9,0x3f,0x00,0xf2]
#CHECK: klmd	%r7, %r10               # encoding: [0xb9,0x3f,0x00,0x7a]

	klmd	%r0, %r2
	klmd	%r0, %r14
	klmd	%r15, %r2
	klmd	%r7, %r10

#CHECK: km	%r2, %r2                # encoding: [0xb9,0x2e,0x00,0x22]
#CHECK: km	%r2, %r14               # encoding: [0xb9,0x2e,0x00,0x2e]
#CHECK: km	%r14, %r2               # encoding: [0xb9,0x2e,0x00,0xe2]
#CHECK: km	%r6, %r10               # encoding: [0xb9,0x2e,0x00,0x6a]

	km	%r2, %r2
	km	%r2, %r14
	km	%r14, %r2
	km	%r6, %r10

#CHECK: kmac	%r0, %r2                # encoding: [0xb9,0x1e,0x00,0x02]
#CHECK: kmac	%r0, %r14               # encoding: [0xb9,0x1e,0x00,0x0e]
#CHECK: kmac	%r15, %r2               # encoding: [0xb9,0x1e,0x00,0xf2]
#CHECK: kmac	%r7, %r10               # encoding: [0xb9,0x1e,0x00,0x7a]

	kmac	%r0, %r2
	kmac	%r0, %r14
	kmac	%r15, %r2
	kmac	%r7, %r10

#CHECK: kmc	%r2, %r2                # encoding: [0xb9,0x2f,0x00,0x22]
#CHECK: kmc	%r2, %r14               # encoding: [0xb9,0x2f,0x00,0x2e]
#CHECK: kmc	%r14, %r2               # encoding: [0xb9,0x2f,0x00,0xe2]
#CHECK: kmc	%r6, %r10               # encoding: [0xb9,0x2f,0x00,0x6a]

	kmc	%r2, %r2
	kmc	%r2, %r14
	kmc	%r14, %r2
	kmc	%r6, %r10

#CHECK: kxbr	%f0, %f0                # encoding: [0xb3,0x48,0x00,0x00]
#CHECK: kxbr	%f0, %f13               # encoding: [0xb3,0x48,0x00,0x0d]
#CHECK: kxbr	%f8, %f8                # encoding: [0xb3,0x48,0x00,0x88]
#CHECK: kxbr	%f13, %f0               # encoding: [0xb3,0x48,0x00,0xd0]

	kxbr	%f0, %f0
	kxbr	%f0, %f13
	kxbr	%f8, %f8
	kxbr	%f13, %f0

#CHECK: kxtr	%f0, %f0                # encoding: [0xb3,0xe8,0x00,0x00]
#CHECK: kxtr	%f0, %f13               # encoding: [0xb3,0xe8,0x00,0x0d]
#CHECK: kxtr	%f8, %f8                # encoding: [0xb3,0xe8,0x00,0x88]
#CHECK: kxtr	%f13, %f0               # encoding: [0xb3,0xe8,0x00,0xd0]

	kxtr	%f0, %f0
	kxtr	%f0, %f13
	kxtr	%f8, %f8
	kxtr	%f13, %f0

#CHECK: l	%r0, 0                  # encoding: [0x58,0x00,0x00,0x00]
#CHECK: l	%r0, 4095               # encoding: [0x58,0x00,0x0f,0xff]
#CHECK: l	%r0, 0                  # encoding: [0x58,0x00,0x00,0x00]
#CHECK: l	%r0, 0(%r1)             # encoding: [0x58,0x00,0x10,0x00]
#CHECK: l	%r0, 0(%r15)            # encoding: [0x58,0x00,0xf0,0x00]
#CHECK: l	%r0, 0(%r1)             # encoding: [0x58,0x00,0x10,0x00]
#CHECK: l	%r0, 0(%r15)            # encoding: [0x58,0x00,0xf0,0x00]
#CHECK: l	%r0, 0(%r1)             # encoding: [0x58,0x00,0x10,0x00]
#CHECK: l	%r0, 0(%r15)            # encoding: [0x58,0x00,0xf0,0x00]
#CHECK: l	%r0, 0(%r1)             # encoding: [0x58,0x00,0x10,0x00]
#CHECK: l	%r0, 0(%r15)            # encoding: [0x58,0x00,0xf0,0x00]
#CHECK: l	%r0, 0(%r1,0)           # encoding: [0x58,0x01,0x00,0x00]
#CHECK: l	%r0, 0(%r15,0)          # encoding: [0x58,0x0f,0x00,0x00]
#CHECK: l	%r0, 0(%r1,0)           # encoding: [0x58,0x01,0x00,0x00]
#CHECK: l	%r0, 0(%r15,0)          # encoding: [0x58,0x0f,0x00,0x00]
#CHECK: l	%r0, 0(%r1,0)           # encoding: [0x58,0x01,0x00,0x00]
#CHECK: l	%r0, 0(%r15,0)          # encoding: [0x58,0x0f,0x00,0x00]
#CHECK: l	%r0, 4095(%r15)         # encoding: [0x58,0x00,0xff,0xff]
#CHECK: l	%r0, 4095(%r1,%r15)     # encoding: [0x58,0x01,0xff,0xff]
#CHECK: l	%r0, 4095(%r15,0)       # encoding: [0x58,0x0f,0x0f,0xff]
#CHECK: l	%r0, 4095(%r15,%r1)     # encoding: [0x58,0x0f,0x1f,0xff]
#CHECK: l	%r15, 0                 # encoding: [0x58,0xf0,0x00,0x00]

	l	%r0, 0
	l	%r0, 4095
	l	%r0, 0(%r0)
	l	%r0, 0(%r1)
	l	%r0, 0(%r15)
	l	%r0, 0(,%r1)
	l	%r0, 0(,%r15)
	l	%r0, 0(0,%r1)
	l	%r0, 0(0,%r15)
	l	%r0, 0(%r0,%r1)
	l	%r0, 0(%r0,%r15)
	l	%r0, 0(0,%r1)
	l	%r0, 0(0,%r15)
	l	%r0, 0(%r1,)
	l	%r0, 0(%r15,)
	l	%r0, 0(%r1,0)
	l	%r0, 0(%r15,0)
	l	%r0, 0(%r1,%r0)
	l	%r0, 0(%r15,%r0)
	l	%r0, 4095(%r0,%r15)
	l	%r0, 4095(%r1,%r15)
	l	%r0, 4095(%r15,%r0)
	l	%r0, 4095(%r15,%r1)
	l	%r15, 0

#CHECK: la	%r0, 0                  # encoding: [0x41,0x00,0x00,0x00]
#CHECK: la	%r0, 4095               # encoding: [0x41,0x00,0x0f,0xff]
#CHECK: la	%r0, 0(%r1)             # encoding: [0x41,0x00,0x10,0x00]
#CHECK: la	%r0, 0(%r15)            # encoding: [0x41,0x00,0xf0,0x00]
#CHECK: la	%r0, 4095(%r1,%r15)     # encoding: [0x41,0x01,0xff,0xff]
#CHECK: la	%r0, 4095(%r15,%r1)     # encoding: [0x41,0x0f,0x1f,0xff]
#CHECK: la	%r15, 0                 # encoding: [0x41,0xf0,0x00,0x00]

	la	%r0, 0
	la	%r0, 4095
	la	%r0, 0(%r1)
	la	%r0, 0(%r15)
	la	%r0, 4095(%r1,%r15)
	la	%r0, 4095(%r15,%r1)
	la	%r15, 0

#CHECK: lae	%r0, 0                  # encoding: [0x51,0x00,0x00,0x00]
#CHECK: lae	%r0, 4095               # encoding: [0x51,0x00,0x0f,0xff]
#CHECK: lae	%r0, 0(%r1)             # encoding: [0x51,0x00,0x10,0x00]
#CHECK: lae	%r0, 0(%r15)            # encoding: [0x51,0x00,0xf0,0x00]
#CHECK: lae	%r0, 4095(%r1,%r15)     # encoding: [0x51,0x01,0xff,0xff]
#CHECK: lae	%r0, 4095(%r15,%r1)     # encoding: [0x51,0x0f,0x1f,0xff]
#CHECK: lae	%r15, 0                 # encoding: [0x51,0xf0,0x00,0x00]

	lae	%r0, 0
	lae	%r0, 4095
	lae	%r0, 0(%r1)
	lae	%r0, 0(%r15)
	lae	%r0, 4095(%r1,%r15)
	lae	%r0, 4095(%r15,%r1)
	lae	%r15, 0

#CHECK: laey	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x75]
#CHECK: laey	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x75]
#CHECK: laey	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x75]
#CHECK: laey	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x75]
#CHECK: laey	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x75]
#CHECK: laey	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x75]
#CHECK: laey	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x75]
#CHECK: laey	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x75]
#CHECK: laey	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x75]
#CHECK: laey	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x75]

	laey	%r0, -524288
	laey	%r0, -1
	laey	%r0, 0
	laey	%r0, 1
	laey	%r0, 524287
	laey	%r0, 0(%r1)
	laey	%r0, 0(%r15)
	laey	%r0, 524287(%r1,%r15)
	laey	%r0, 524287(%r15,%r1)
	laey	%r15, 0

#CHECK: lam	%a0, %a0, 0             # encoding: [0x9a,0x00,0x00,0x00]
#CHECK: lam	%a0, %a15, 0            # encoding: [0x9a,0x0f,0x00,0x00]
#CHECK: lam	%a14, %a15, 0           # encoding: [0x9a,0xef,0x00,0x00]
#CHECK: lam	%a15, %a15, 0           # encoding: [0x9a,0xff,0x00,0x00]
#CHECK: lam	%a0, %a0, 4095          # encoding: [0x9a,0x00,0x0f,0xff]
#CHECK: lam	%a0, %a0, 1             # encoding: [0x9a,0x00,0x00,0x01]
#CHECK: lam	%a0, %a0, 0(%r1)        # encoding: [0x9a,0x00,0x10,0x00]
#CHECK: lam	%a0, %a0, 0(%r15)       # encoding: [0x9a,0x00,0xf0,0x00]
#CHECK: lam	%a0, %a0, 4095(%r1)     # encoding: [0x9a,0x00,0x1f,0xff]
#CHECK: lam	%a0, %a0, 4095(%r15)    # encoding: [0x9a,0x00,0xff,0xff]

	lam	%a0,%a0,0
	lam	%a0,%a15,0
	lam	%a14,%a15,0
	lam	%a15,%a15,0
	lam	%a0,%a0,4095
	lam	%a0,%a0,1
	lam	%a0,%a0,0(%r1)
	lam	%a0,%a0,0(%r15)
	lam	%a0,%a0,4095(%r1)
	lam	%a0,%a0,4095(%r15)

#CHECK: lamy	%a0, %a0, 0             # encoding: [0xeb,0x00,0x00,0x00,0x00,0x9a]
#CHECK: lamy	%a0, %a15, 0            # encoding: [0xeb,0x0f,0x00,0x00,0x00,0x9a]
#CHECK: lamy	%a14, %a15, 0           # encoding: [0xeb,0xef,0x00,0x00,0x00,0x9a]
#CHECK: lamy	%a15, %a15, 0           # encoding: [0xeb,0xff,0x00,0x00,0x00,0x9a]
#CHECK: lamy	%a0, %a0, -524288       # encoding: [0xeb,0x00,0x00,0x00,0x80,0x9a]
#CHECK: lamy	%a0, %a0, -1            # encoding: [0xeb,0x00,0x0f,0xff,0xff,0x9a]
#CHECK: lamy	%a0, %a0, 0             # encoding: [0xeb,0x00,0x00,0x00,0x00,0x9a]
#CHECK: lamy	%a0, %a0, 1             # encoding: [0xeb,0x00,0x00,0x01,0x00,0x9a]
#CHECK: lamy	%a0, %a0, 524287        # encoding: [0xeb,0x00,0x0f,0xff,0x7f,0x9a]
#CHECK: lamy	%a0, %a0, 0(%r1)        # encoding: [0xeb,0x00,0x10,0x00,0x00,0x9a]
#CHECK: lamy	%a0, %a0, 0(%r15)       # encoding: [0xeb,0x00,0xf0,0x00,0x00,0x9a]
#CHECK: lamy	%a0, %a0, 524287(%r1)   # encoding: [0xeb,0x00,0x1f,0xff,0x7f,0x9a]
#CHECK: lamy	%a0, %a0, 524287(%r15)  # encoding: [0xeb,0x00,0xff,0xff,0x7f,0x9a]

	lamy	%a0,%a0,0
	lamy	%a0,%a15,0
	lamy	%a14,%a15,0
	lamy	%a15,%a15,0
	lamy	%a0,%a0,-524288
	lamy	%a0,%a0,-1
	lamy	%a0,%a0,0
	lamy	%a0,%a0,1
	lamy	%a0,%a0,524287
	lamy	%a0,%a0,0(%r1)
	lamy	%a0,%a0,0(%r15)
	lamy	%a0,%a0,524287(%r1)
	lamy	%a0,%a0,524287(%r15)

#CHECK: larl	%r0, .[[LAB:L.*]]-4294967296 # encoding: [0xc0,0x00,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-4294967296)+2, kind: FK_390_PC32DBL
	larl	%r0, -0x100000000
#CHECK: larl	%r0, .[[LAB:L.*]]-2	# encoding: [0xc0,0x00,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC32DBL
	larl	%r0, -2
#CHECK: larl	%r0, .[[LAB:L.*]]	# encoding: [0xc0,0x00,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC32DBL
	larl	%r0, 0
#CHECK: larl	%r0, .[[LAB:L.*]]+4294967294 # encoding: [0xc0,0x00,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+4294967294)+2, kind: FK_390_PC32DBL
	larl	%r0, 0xfffffffe

#CHECK: larl	%r0, foo                # encoding: [0xc0,0x00,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: larl	%r15, foo               # encoding: [0xc0,0xf0,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL

	larl	%r0,foo
	larl	%r15,foo

#CHECK: larl	%r3, bar+100            # encoding: [0xc0,0x30,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: larl	%r4, bar+100            # encoding: [0xc0,0x40,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL

	larl	%r3,bar+100
	larl	%r4,bar+100

#CHECK: larl	%r7, frob@PLT           # encoding: [0xc0,0x70,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL
#CHECK: larl	%r8, frob@PLT           # encoding: [0xc0,0x80,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL

	larl	%r7,frob@PLT
	larl	%r8,frob@PLT

#CHECK: lasp	0, 0                    # encoding: [0xe5,0x00,0x00,0x00,0x00,0x00]
#CHECK: lasp	0(%r1), 0(%r2)          # encoding: [0xe5,0x00,0x10,0x00,0x20,0x00]
#CHECK: lasp	160(%r1), 320(%r15)     # encoding: [0xe5,0x00,0x10,0xa0,0xf1,0x40]
#CHECK: lasp	0(%r1), 4095            # encoding: [0xe5,0x00,0x10,0x00,0x0f,0xff]
#CHECK: lasp	0(%r1), 4095(%r2)       # encoding: [0xe5,0x00,0x10,0x00,0x2f,0xff]
#CHECK: lasp	0(%r1), 4095(%r15)      # encoding: [0xe5,0x00,0x10,0x00,0xff,0xff]
#CHECK: lasp	0(%r1), 0               # encoding: [0xe5,0x00,0x10,0x00,0x00,0x00]
#CHECK: lasp	0(%r15), 0              # encoding: [0xe5,0x00,0xf0,0x00,0x00,0x00]
#CHECK: lasp	4095(%r1), 0            # encoding: [0xe5,0x00,0x1f,0xff,0x00,0x00]
#CHECK: lasp	4095(%r15), 0           # encoding: [0xe5,0x00,0xff,0xff,0x00,0x00]

	lasp	0, 0
	lasp	0(%r1), 0(%r2)
	lasp	160(%r1), 320(%r15)
	lasp	0(%r1), 4095
	lasp	0(%r1), 4095(%r2)
	lasp	0(%r1), 4095(%r15)
	lasp	0(%r1), 0
	lasp	0(%r15), 0
	lasp	4095(%r1), 0
	lasp	4095(%r15), 0

#CHECK: lay	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x71]
#CHECK: lay	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x71]
#CHECK: lay	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x71]
#CHECK: lay	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x71]
#CHECK: lay	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x71]
#CHECK: lay	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x71]
#CHECK: lay	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x71]
#CHECK: lay	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x71]
#CHECK: lay	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x71]
#CHECK: lay	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x71]

	lay	%r0, -524288
	lay	%r0, -1
	lay	%r0, 0
	lay	%r0, 1
	lay	%r0, 524287
	lay	%r0, 0(%r1)
	lay	%r0, 0(%r15)
	lay	%r0, 524287(%r1,%r15)
	lay	%r0, 524287(%r15,%r1)
	lay	%r15, 0

#CHECK: lb	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x76]
#CHECK: lb	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x76]
#CHECK: lb	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x76]
#CHECK: lb	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x76]
#CHECK: lb	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x76]
#CHECK: lb	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x76]
#CHECK: lb	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x76]
#CHECK: lb	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x76]
#CHECK: lb	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x76]
#CHECK: lb	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x76]

	lb	%r0, -524288
	lb	%r0, -1
	lb	%r0, 0
	lb	%r0, 1
	lb	%r0, 524287
	lb	%r0, 0(%r1)
	lb	%r0, 0(%r15)
	lb	%r0, 524287(%r1,%r15)
	lb	%r0, 524287(%r15,%r1)
	lb	%r15, 0

#CHECK: lbr	%r0, %r15               # encoding: [0xb9,0x26,0x00,0x0f]
#CHECK: lbr	%r7, %r8                # encoding: [0xb9,0x26,0x00,0x78]
#CHECK: lbr	%r15, %r0               # encoding: [0xb9,0x26,0x00,0xf0]

	lbr	%r0, %r15
	lbr	%r7, %r8
	lbr	%r15, %r0

#CHECK: lcctl	0                       # encoding: [0xb2,0x84,0x00,0x00]
#CHECK: lcctl	0(%r1)                  # encoding: [0xb2,0x84,0x10,0x00]
#CHECK: lcctl	0(%r15)                 # encoding: [0xb2,0x84,0xf0,0x00]
#CHECK: lcctl	4095                    # encoding: [0xb2,0x84,0x0f,0xff]
#CHECK: lcctl	4095(%r1)               # encoding: [0xb2,0x84,0x1f,0xff]
#CHECK: lcctl	4095(%r15)              # encoding: [0xb2,0x84,0xff,0xff]

	lcctl	0
	lcctl	0(%r1)
	lcctl	0(%r15)
	lcctl	4095
	lcctl	4095(%r1)
	lcctl	4095(%r15)

#CHECK: lcdbr	%f0, %f9                # encoding: [0xb3,0x13,0x00,0x09]
#CHECK: lcdbr	%f0, %f15               # encoding: [0xb3,0x13,0x00,0x0f]
#CHECK: lcdbr	%f15, %f0               # encoding: [0xb3,0x13,0x00,0xf0]
#CHECK: lcdbr	%f15, %f9               # encoding: [0xb3,0x13,0x00,0xf9]

	lcdbr	%f0,%f9
	lcdbr	%f0,%f15
	lcdbr	%f15,%f0
	lcdbr	%f15,%f9

#CHECK: lcdfr	%f0, %f9                # encoding: [0xb3,0x73,0x00,0x09]
#CHECK: lcdfr	%f0, %f15               # encoding: [0xb3,0x73,0x00,0x0f]
#CHECK: lcdfr	%f15, %f0               # encoding: [0xb3,0x73,0x00,0xf0]
#CHECK: lcdfr	%f15, %f9               # encoding: [0xb3,0x73,0x00,0xf9]

	lcdfr	%f0,%f9
	lcdfr	%f0,%f15
	lcdfr	%f15,%f0
	lcdfr	%f15,%f9

#CHECK: lndfr	%f0, %f9                # encoding: [0xb3,0x71,0x00,0x09]
#CHECK: lndfr	%f0, %f15               # encoding: [0xb3,0x71,0x00,0x0f]
#CHECK: lndfr	%f15, %f0               # encoding: [0xb3,0x71,0x00,0xf0]
#CHECK: lndfr	%f15, %f9               # encoding: [0xb3,0x71,0x00,0xf9]

	lndfr	%f0,%f9
	lndfr	%f0,%f15
	lndfr	%f15,%f0
	lndfr	%f15,%f9

#CHECK: lpdfr	%f0, %f9                # encoding: [0xb3,0x70,0x00,0x09]
#CHECK: lpdfr	%f0, %f15               # encoding: [0xb3,0x70,0x00,0x0f]
#CHECK: lpdfr	%f15, %f0               # encoding: [0xb3,0x70,0x00,0xf0]
#CHECK: lpdfr	%f15, %f9               # encoding: [0xb3,0x70,0x00,0xf9]

	lpdfr	%f0,%f9
	lpdfr	%f0,%f15
	lpdfr	%f15,%f0
	lpdfr	%f15,%f9

#CHECK: lcdr	%f0, %f9                # encoding: [0x23,0x09]
#CHECK: lcdr	%f0, %f15               # encoding: [0x23,0x0f]
#CHECK: lcdr	%f15, %f0               # encoding: [0x23,0xf0]
#CHECK: lcdr	%f15, %f9               # encoding: [0x23,0xf9]

	lcdr	%f0,%f9
	lcdr	%f0,%f15
	lcdr	%f15,%f0
	lcdr	%f15,%f9

#CHECK: lcebr	%f0, %f9                # encoding: [0xb3,0x03,0x00,0x09]
#CHECK: lcebr	%f0, %f15               # encoding: [0xb3,0x03,0x00,0x0f]
#CHECK: lcebr	%f15, %f0               # encoding: [0xb3,0x03,0x00,0xf0]
#CHECK: lcebr	%f15, %f9               # encoding: [0xb3,0x03,0x00,0xf9]

	lcebr	%f0,%f9
	lcebr	%f0,%f15
	lcebr	%f15,%f0
	lcebr	%f15,%f9

#CHECK: lcer	%f0, %f9                # encoding: [0x33,0x09]
#CHECK: lcer	%f0, %f15               # encoding: [0x33,0x0f]
#CHECK: lcer	%f15, %f0               # encoding: [0x33,0xf0]
#CHECK: lcer	%f15, %f9               # encoding: [0x33,0xf9]

	lcer	%f0,%f9
	lcer	%f0,%f15
	lcer	%f15,%f0
	lcer	%f15,%f9

#CHECK: lcgfr	%r0, %r0                # encoding: [0xb9,0x13,0x00,0x00]
#CHECK: lcgfr	%r0, %r15               # encoding: [0xb9,0x13,0x00,0x0f]
#CHECK: lcgfr	%r15, %r0               # encoding: [0xb9,0x13,0x00,0xf0]
#CHECK: lcgfr	%r7, %r8                # encoding: [0xb9,0x13,0x00,0x78]

	lcgfr	%r0,%r0
	lcgfr	%r0,%r15
	lcgfr	%r15,%r0
	lcgfr	%r7,%r8

#CHECK: lcgr	%r0, %r0                # encoding: [0xb9,0x03,0x00,0x00]
#CHECK: lcgr	%r0, %r15               # encoding: [0xb9,0x03,0x00,0x0f]
#CHECK: lcgr	%r15, %r0               # encoding: [0xb9,0x03,0x00,0xf0]
#CHECK: lcgr	%r7, %r8                # encoding: [0xb9,0x03,0x00,0x78]

	lcgr	%r0,%r0
	lcgr	%r0,%r15
	lcgr	%r15,%r0
	lcgr	%r7,%r8

#CHECK: lcr	%r0, %r0                # encoding: [0x13,0x00]
#CHECK: lcr	%r0, %r15               # encoding: [0x13,0x0f]
#CHECK: lcr	%r15, %r0               # encoding: [0x13,0xf0]
#CHECK: lcr	%r7, %r8                # encoding: [0x13,0x78]

	lcr	%r0,%r0
	lcr	%r0,%r15
	lcr	%r15,%r0
	lcr	%r7,%r8

#CHECK: lctl	%c0, %c0, 0             # encoding: [0xb7,0x00,0x00,0x00]
#CHECK: lctl	%c0, %c15, 0            # encoding: [0xb7,0x0f,0x00,0x00]
#CHECK: lctl	%c14, %c15, 0           # encoding: [0xb7,0xef,0x00,0x00]
#CHECK: lctl	%c15, %c15, 0           # encoding: [0xb7,0xff,0x00,0x00]
#CHECK: lctl	%c0, %c0, 4095          # encoding: [0xb7,0x00,0x0f,0xff]
#CHECK: lctl	%c0, %c0, 1             # encoding: [0xb7,0x00,0x00,0x01]
#CHECK: lctl	%c0, %c0, 0(%r1)        # encoding: [0xb7,0x00,0x10,0x00]
#CHECK: lctl	%c0, %c0, 0(%r15)       # encoding: [0xb7,0x00,0xf0,0x00]
#CHECK: lctl	%c0, %c0, 4095(%r1)     # encoding: [0xb7,0x00,0x1f,0xff]
#CHECK: lctl	%c0, %c0, 4095(%r15)    # encoding: [0xb7,0x00,0xff,0xff]

	lctl	%c0,%c0,0
	lctl	%c0,%c15,0
	lctl	%c14,%c15,0
	lctl	%c15,%c15,0
	lctl	%c0,%c0,4095
	lctl	%c0,%c0,1
	lctl	%c0,%c0,0(%r1)
	lctl	%c0,%c0,0(%r15)
	lctl	%c0,%c0,4095(%r1)
	lctl	%c0,%c0,4095(%r15)

#CHECK: lctlg	%c0, %c0, 0             # encoding: [0xeb,0x00,0x00,0x00,0x00,0x2f]
#CHECK: lctlg	%c0, %c15, 0            # encoding: [0xeb,0x0f,0x00,0x00,0x00,0x2f]
#CHECK: lctlg	%c14, %c15, 0           # encoding: [0xeb,0xef,0x00,0x00,0x00,0x2f]
#CHECK: lctlg	%c15, %c15, 0           # encoding: [0xeb,0xff,0x00,0x00,0x00,0x2f]
#CHECK: lctlg	%c0, %c0, -524288       # encoding: [0xeb,0x00,0x00,0x00,0x80,0x2f]
#CHECK: lctlg	%c0, %c0, -1            # encoding: [0xeb,0x00,0x0f,0xff,0xff,0x2f]
#CHECK: lctlg	%c0, %c0, 0             # encoding: [0xeb,0x00,0x00,0x00,0x00,0x2f]
#CHECK: lctlg	%c0, %c0, 1             # encoding: [0xeb,0x00,0x00,0x01,0x00,0x2f]
#CHECK: lctlg	%c0, %c0, 524287        # encoding: [0xeb,0x00,0x0f,0xff,0x7f,0x2f]
#CHECK: lctlg	%c0, %c0, 0(%r1)        # encoding: [0xeb,0x00,0x10,0x00,0x00,0x2f]
#CHECK: lctlg	%c0, %c0, 0(%r15)       # encoding: [0xeb,0x00,0xf0,0x00,0x00,0x2f]
#CHECK: lctlg	%c0, %c0, 524287(%r1)   # encoding: [0xeb,0x00,0x1f,0xff,0x7f,0x2f]
#CHECK: lctlg	%c0, %c0, 524287(%r15)  # encoding: [0xeb,0x00,0xff,0xff,0x7f,0x2f]

	lctlg	%c0,%c0,0
	lctlg	%c0,%c15,0
	lctlg	%c14,%c15,0
	lctlg	%c15,%c15,0
	lctlg	%c0,%c0,-524288
	lctlg	%c0,%c0,-1
	lctlg	%c0,%c0,0
	lctlg	%c0,%c0,1
	lctlg	%c0,%c0,524287
	lctlg	%c0,%c0,0(%r1)
	lctlg	%c0,%c0,0(%r15)
	lctlg	%c0,%c0,524287(%r1)
	lctlg	%c0,%c0,524287(%r15)

#CHECK: lcxbr	%f0, %f8                # encoding: [0xb3,0x43,0x00,0x08]
#CHECK: lcxbr	%f0, %f13               # encoding: [0xb3,0x43,0x00,0x0d]
#CHECK: lcxbr	%f13, %f0               # encoding: [0xb3,0x43,0x00,0xd0]
#CHECK: lcxbr	%f13, %f9               # encoding: [0xb3,0x43,0x00,0xd9]

	lcxbr	%f0,%f8
	lcxbr	%f0,%f13
	lcxbr	%f13,%f0
	lcxbr	%f13,%f9

#CHECK: lcxr	%f0, %f8                # encoding: [0xb3,0x63,0x00,0x08]
#CHECK: lcxr	%f0, %f13               # encoding: [0xb3,0x63,0x00,0x0d]
#CHECK: lcxr	%f13, %f0               # encoding: [0xb3,0x63,0x00,0xd0]
#CHECK: lcxr	%f13, %f9               # encoding: [0xb3,0x63,0x00,0xd9]

	lcxr	%f0,%f8
	lcxr	%f0,%f13
	lcxr	%f13,%f0
	lcxr	%f13,%f9

#CHECK: ld	%f0, 0                  # encoding: [0x68,0x00,0x00,0x00]
#CHECK: ld	%f0, 4095               # encoding: [0x68,0x00,0x0f,0xff]
#CHECK: ld	%f0, 0(%r1)             # encoding: [0x68,0x00,0x10,0x00]
#CHECK: ld	%f0, 0(%r15)            # encoding: [0x68,0x00,0xf0,0x00]
#CHECK: ld	%f0, 4095(%r1,%r15)     # encoding: [0x68,0x01,0xff,0xff]
#CHECK: ld	%f0, 4095(%r15,%r1)     # encoding: [0x68,0x0f,0x1f,0xff]
#CHECK: ld	%f15, 0                 # encoding: [0x68,0xf0,0x00,0x00]

	ld	%f0, 0
	ld	%f0, 4095
	ld	%f0, 0(%r1)
	ld	%f0, 0(%r15)
	ld	%f0, 4095(%r1,%r15)
	ld	%f0, 4095(%r15,%r1)
	ld	%f15, 0

#CHECK: lde	%f0, 0                  # encoding: [0xed,0x00,0x00,0x00,0x00,0x24]
#CHECK: lde	%f0, 4095               # encoding: [0xed,0x00,0x0f,0xff,0x00,0x24]
#CHECK: lde	%f0, 0(%r1)             # encoding: [0xed,0x00,0x10,0x00,0x00,0x24]
#CHECK: lde	%f0, 0(%r15)            # encoding: [0xed,0x00,0xf0,0x00,0x00,0x24]
#CHECK: lde	%f0, 4095(%r1,%r15)     # encoding: [0xed,0x01,0xff,0xff,0x00,0x24]
#CHECK: lde	%f0, 4095(%r15,%r1)     # encoding: [0xed,0x0f,0x1f,0xff,0x00,0x24]
#CHECK: lde	%f15, 0                 # encoding: [0xed,0xf0,0x00,0x00,0x00,0x24]

	lde	%f0, 0
	lde	%f0, 4095
	lde	%f0, 0(%r1)
	lde	%f0, 0(%r15)
	lde	%f0, 4095(%r1,%r15)
	lde	%f0, 4095(%r15,%r1)
	lde	%f15, 0

#CHECK: ldeb	%f0, 0                  # encoding: [0xed,0x00,0x00,0x00,0x00,0x04]
#CHECK: ldeb	%f0, 4095               # encoding: [0xed,0x00,0x0f,0xff,0x00,0x04]
#CHECK: ldeb	%f0, 0(%r1)             # encoding: [0xed,0x00,0x10,0x00,0x00,0x04]
#CHECK: ldeb	%f0, 0(%r15)            # encoding: [0xed,0x00,0xf0,0x00,0x00,0x04]
#CHECK: ldeb	%f0, 4095(%r1,%r15)     # encoding: [0xed,0x01,0xff,0xff,0x00,0x04]
#CHECK: ldeb	%f0, 4095(%r15,%r1)     # encoding: [0xed,0x0f,0x1f,0xff,0x00,0x04]
#CHECK: ldeb	%f15, 0                 # encoding: [0xed,0xf0,0x00,0x00,0x00,0x04]

	ldeb	%f0, 0
	ldeb	%f0, 4095
	ldeb	%f0, 0(%r1)
	ldeb	%f0, 0(%r15)
	ldeb	%f0, 4095(%r1,%r15)
	ldeb	%f0, 4095(%r15,%r1)
	ldeb	%f15, 0

#CHECK: ldebr	%f0, %f15               # encoding: [0xb3,0x04,0x00,0x0f]
#CHECK: ldebr	%f7, %f8                # encoding: [0xb3,0x04,0x00,0x78]
#CHECK: ldebr	%f15, %f0               # encoding: [0xb3,0x04,0x00,0xf0]

	ldebr	%f0, %f15
	ldebr	%f7, %f8
	ldebr	%f15, %f0

#CHECK: lder	%f0, %f15               # encoding: [0xb3,0x24,0x00,0x0f]
#CHECK: lder	%f7, %f8                # encoding: [0xb3,0x24,0x00,0x78]
#CHECK: lder	%f15, %f0               # encoding: [0xb3,0x24,0x00,0xf0]

	lder	%f0, %f15
	lder	%f7, %f8
	lder	%f15, %f0

#CHECK: ldetr	%f0, %f0, 15            # encoding: [0xb3,0xd4,0x0f,0x00]
#CHECK: ldetr	%f0, %f15, 0            # encoding: [0xb3,0xd4,0x00,0x0f]
#CHECK: ldetr	%f7, %f8, 9             # encoding: [0xb3,0xd4,0x09,0x78]
#CHECK: ldetr	%f15, %f0, 0            # encoding: [0xb3,0xd4,0x00,0xf0]

	ldetr	%f0, %f0, 15
	ldetr	%f0, %f15, 0
	ldetr	%f7, %f8, 9
	ldetr	%f15, %f0, 0

#CHECK: ldgr	%f0, %r0                # encoding: [0xb3,0xc1,0x00,0x00]
#CHECK: ldgr	%f0, %r15               # encoding: [0xb3,0xc1,0x00,0x0f]
#CHECK: ldgr	%f15, %r0               # encoding: [0xb3,0xc1,0x00,0xf0]
#CHECK: ldgr	%f7, %r9                # encoding: [0xb3,0xc1,0x00,0x79]
#CHECK: ldgr	%f15, %r15              # encoding: [0xb3,0xc1,0x00,0xff]

	ldgr	%f0,%r0
	ldgr	%f0,%r15
	ldgr	%f15,%r0
	ldgr	%f7,%r9
	ldgr	%f15,%r15

#CHECK: ldr	%f0, %f9                # encoding: [0x28,0x09]
#CHECK: ldr	%f0, %f15               # encoding: [0x28,0x0f]
#CHECK: ldr	%f15, %f0               # encoding: [0x28,0xf0]
#CHECK: ldr	%f15, %f9               # encoding: [0x28,0xf9]

	ldr	%f0,%f9
	ldr	%f0,%f15
	ldr	%f15,%f0
	ldr	%f15,%f9

#CHECK: ldxbr	%f0, %f0                # encoding: [0xb3,0x45,0x00,0x00]
#CHECK: ldxbr	%f0, %f13               # encoding: [0xb3,0x45,0x00,0x0d]
#CHECK: ldxbr	%f8, %f12               # encoding: [0xb3,0x45,0x00,0x8c]
#CHECK: ldxbr	%f13, %f0               # encoding: [0xb3,0x45,0x00,0xd0]
#CHECK: ldxbr	%f13, %f13              # encoding: [0xb3,0x45,0x00,0xdd]

	ldxbr	%f0, %f0
	ldxbr	%f0, %f13
	ldxbr	%f8, %f12
	ldxbr	%f13, %f0
	ldxbr	%f13, %f13

#CHECK: ldxr	%f0, %f0                # encoding: [0x25,0x00]
#CHECK: ldxr	%f0, %f13               # encoding: [0x25,0x0d]
#CHECK: ldxr	%f7, %f8                # encoding: [0x25,0x78]
#CHECK: ldxr	%f15, %f0               # encoding: [0x25,0xf0]
#CHECK: ldxr	%f15, %f13              # encoding: [0x25,0xfd]

	ldxr	%f0, %f0
	ldxr	%f0, %f13
	ldxr	%f7, %f8
	ldxr	%f15, %f0
	ldxr	%f15, %f13

#CHECK: ldxtr	%f0, 0, %f0, 0          # encoding: [0xb3,0xdd,0x00,0x00]
#CHECK: ldxtr	%f0, 0, %f0, 15         # encoding: [0xb3,0xdd,0x0f,0x00]
#CHECK: ldxtr	%f0, 0, %f13, 0         # encoding: [0xb3,0xdd,0x00,0x0d]
#CHECK: ldxtr	%f0, 15, %f0, 0         # encoding: [0xb3,0xdd,0xf0,0x00]
#CHECK: ldxtr	%f4, 5, %f8, 9          # encoding: [0xb3,0xdd,0x59,0x48]
#CHECK: ldxtr	%f13, 0, %f0, 0         # encoding: [0xb3,0xdd,0x00,0xd0]

	ldxtr	%f0, 0, %f0, 0
	ldxtr	%f0, 0, %f0, 15
	ldxtr	%f0, 0, %f13, 0
	ldxtr	%f0, 15, %f0, 0
	ldxtr	%f4, 5, %f8, 9
	ldxtr	%f13, 0, %f0, 0

#CHECK: ldy	%f0, -524288            # encoding: [0xed,0x00,0x00,0x00,0x80,0x65]
#CHECK: ldy	%f0, -1                 # encoding: [0xed,0x00,0x0f,0xff,0xff,0x65]
#CHECK: ldy	%f0, 0                  # encoding: [0xed,0x00,0x00,0x00,0x00,0x65]
#CHECK: ldy	%f0, 1                  # encoding: [0xed,0x00,0x00,0x01,0x00,0x65]
#CHECK: ldy	%f0, 524287             # encoding: [0xed,0x00,0x0f,0xff,0x7f,0x65]
#CHECK: ldy	%f0, 0(%r1)             # encoding: [0xed,0x00,0x10,0x00,0x00,0x65]
#CHECK: ldy	%f0, 0(%r15)            # encoding: [0xed,0x00,0xf0,0x00,0x00,0x65]
#CHECK: ldy	%f0, 524287(%r1,%r15)   # encoding: [0xed,0x01,0xff,0xff,0x7f,0x65]
#CHECK: ldy	%f0, 524287(%r15,%r1)   # encoding: [0xed,0x0f,0x1f,0xff,0x7f,0x65]
#CHECK: ldy	%f15, 0                 # encoding: [0xed,0xf0,0x00,0x00,0x00,0x65]

	ldy	%f0, -524288
	ldy	%f0, -1
	ldy	%f0, 0
	ldy	%f0, 1
	ldy	%f0, 524287
	ldy	%f0, 0(%r1)
	ldy	%f0, 0(%r15)
	ldy	%f0, 524287(%r1,%r15)
	ldy	%f0, 524287(%r15,%r1)
	ldy	%f15, 0

#CHECK: le	%f0, 0                  # encoding: [0x78,0x00,0x00,0x00]
#CHECK: le	%f0, 4095               # encoding: [0x78,0x00,0x0f,0xff]
#CHECK: le	%f0, 0(%r1)             # encoding: [0x78,0x00,0x10,0x00]
#CHECK: le	%f0, 0(%r15)            # encoding: [0x78,0x00,0xf0,0x00]
#CHECK: le	%f0, 4095(%r1,%r15)     # encoding: [0x78,0x01,0xff,0xff]
#CHECK: le	%f0, 4095(%r15,%r1)     # encoding: [0x78,0x0f,0x1f,0xff]
#CHECK: le	%f15, 0                 # encoding: [0x78,0xf0,0x00,0x00]

	le	%f0, 0
	le	%f0, 4095
	le	%f0, 0(%r1)
	le	%f0, 0(%r15)
	le	%f0, 4095(%r1,%r15)
	le	%f0, 4095(%r15,%r1)
	le	%f15, 0

#CHECK: ledbr	%f0, %f0                # encoding: [0xb3,0x44,0x00,0x00]
#CHECK: ledbr	%f0, %f15               # encoding: [0xb3,0x44,0x00,0x0f]
#CHECK: ledbr	%f7, %f8                # encoding: [0xb3,0x44,0x00,0x78]
#CHECK: ledbr	%f15, %f0               # encoding: [0xb3,0x44,0x00,0xf0]
#CHECK: ledbr	%f15, %f15              # encoding: [0xb3,0x44,0x00,0xff]

	ledbr	%f0, %f0
	ledbr	%f0, %f15
	ledbr	%f7, %f8
	ledbr	%f15, %f0
	ledbr	%f15, %f15

#CHECK: ledr	%f0, %f0                # encoding: [0x35,0x00]
#CHECK: ledr	%f0, %f15               # encoding: [0x35,0x0f]
#CHECK: ledr	%f7, %f8                # encoding: [0x35,0x78]
#CHECK: ledr	%f15, %f0               # encoding: [0x35,0xf0]
#CHECK: ledr	%f15, %f15              # encoding: [0x35,0xff]

	ledr	%f0, %f0
	ledr	%f0, %f15
	ledr	%f7, %f8
	ledr	%f15, %f0
	ledr	%f15, %f15

#CHECK: ledtr	%f0, 0, %f0, 0          # encoding: [0xb3,0xd5,0x00,0x00]
#CHECK: ledtr	%f0, 0, %f0, 15         # encoding: [0xb3,0xd5,0x0f,0x00]
#CHECK: ledtr	%f0, 0, %f15, 0         # encoding: [0xb3,0xd5,0x00,0x0f]
#CHECK: ledtr	%f0, 15, %f0, 0         # encoding: [0xb3,0xd5,0xf0,0x00]
#CHECK: ledtr	%f4, 5, %f6, 7          # encoding: [0xb3,0xd5,0x57,0x46]
#CHECK: ledtr	%f15, 0, %f0, 0         # encoding: [0xb3,0xd5,0x00,0xf0]

	ledtr	%f0, 0, %f0, 0
	ledtr	%f0, 0, %f0, 15
	ledtr	%f0, 0, %f15, 0
	ledtr	%f0, 15, %f0, 0
	ledtr	%f4, 5, %f6, 7
	ledtr	%f15, 0, %f0, 0

#CHECK: ler	%f0, %f9                # encoding: [0x38,0x09]
#CHECK: ler	%f0, %f15               # encoding: [0x38,0x0f]
#CHECK: ler	%f15, %f0               # encoding: [0x38,0xf0]
#CHECK: ler	%f15, %f9               # encoding: [0x38,0xf9]

	ler	%f0,%f9
	ler	%f0,%f15
	ler	%f15,%f0
	ler	%f15,%f9

#CHECK: lexbr	%f0, %f0                # encoding: [0xb3,0x46,0x00,0x00]
#CHECK: lexbr	%f0, %f13               # encoding: [0xb3,0x46,0x00,0x0d]
#CHECK: lexbr	%f8, %f12               # encoding: [0xb3,0x46,0x00,0x8c]
#CHECK: lexbr	%f13, %f0               # encoding: [0xb3,0x46,0x00,0xd0]
#CHECK: lexbr	%f13, %f13              # encoding: [0xb3,0x46,0x00,0xdd]

	lexbr	%f0, %f0
	lexbr	%f0, %f13
	lexbr	%f8, %f12
	lexbr	%f13, %f0
	lexbr	%f13, %f13

#CHECK: lexr	%f0, %f0                # encoding: [0xb3,0x66,0x00,0x00]
#CHECK: lexr	%f0, %f13               # encoding: [0xb3,0x66,0x00,0x0d]
#CHECK: lexr	%f7, %f8                # encoding: [0xb3,0x66,0x00,0x78]
#CHECK: lexr	%f15, %f0               # encoding: [0xb3,0x66,0x00,0xf0]
#CHECK: lexr	%f15, %f13              # encoding: [0xb3,0x66,0x00,0xfd]

	lexr	%f0, %f0
	lexr	%f0, %f13
	lexr	%f7, %f8
	lexr	%f15, %f0
	lexr	%f15, %f13

#CHECK: ley	%f0, -524288            # encoding: [0xed,0x00,0x00,0x00,0x80,0x64]
#CHECK: ley	%f0, -1                 # encoding: [0xed,0x00,0x0f,0xff,0xff,0x64]
#CHECK: ley	%f0, 0                  # encoding: [0xed,0x00,0x00,0x00,0x00,0x64]
#CHECK: ley	%f0, 1                  # encoding: [0xed,0x00,0x00,0x01,0x00,0x64]
#CHECK: ley	%f0, 524287             # encoding: [0xed,0x00,0x0f,0xff,0x7f,0x64]
#CHECK: ley	%f0, 0(%r1)             # encoding: [0xed,0x00,0x10,0x00,0x00,0x64]
#CHECK: ley	%f0, 0(%r15)            # encoding: [0xed,0x00,0xf0,0x00,0x00,0x64]
#CHECK: ley	%f0, 524287(%r1,%r15)   # encoding: [0xed,0x01,0xff,0xff,0x7f,0x64]
#CHECK: ley	%f0, 524287(%r15,%r1)   # encoding: [0xed,0x0f,0x1f,0xff,0x7f,0x64]
#CHECK: ley	%f15, 0                 # encoding: [0xed,0xf0,0x00,0x00,0x00,0x64]

	ley	%f0, -524288
	ley	%f0, -1
	ley	%f0, 0
	ley	%f0, 1
	ley	%f0, 524287
	ley	%f0, 0(%r1)
	ley	%f0, 0(%r15)
	ley	%f0, 524287(%r1,%r15)
	ley	%f0, 524287(%r15,%r1)
	ley	%f15, 0

#CHECK: lfas	0                       # encoding: [0xb2,0xbd,0x00,0x00]
#CHECK: lfas	0(%r1)                  # encoding: [0xb2,0xbd,0x10,0x00]
#CHECK: lfas	0(%r15)                 # encoding: [0xb2,0xbd,0xf0,0x00]
#CHECK: lfas	4095                    # encoding: [0xb2,0xbd,0x0f,0xff]
#CHECK: lfas	4095(%r1)               # encoding: [0xb2,0xbd,0x1f,0xff]
#CHECK: lfas	4095(%r15)              # encoding: [0xb2,0xbd,0xff,0xff]

	lfas	0
	lfas	0(%r1)
	lfas	0(%r15)
	lfas	4095
	lfas	4095(%r1)
	lfas	4095(%r15)

#CHECK: lfpc	0                       # encoding: [0xb2,0x9d,0x00,0x00]
#CHECK: lfpc	0(%r1)                  # encoding: [0xb2,0x9d,0x10,0x00]
#CHECK: lfpc	0(%r15)                 # encoding: [0xb2,0x9d,0xf0,0x00]
#CHECK: lfpc	4095                    # encoding: [0xb2,0x9d,0x0f,0xff]
#CHECK: lfpc	4095(%r1)               # encoding: [0xb2,0x9d,0x1f,0xff]
#CHECK: lfpc	4095(%r15)              # encoding: [0xb2,0x9d,0xff,0xff]

	lfpc	0
	lfpc	0(%r1)
	lfpc	0(%r15)
	lfpc	4095
	lfpc	4095(%r1)
	lfpc	4095(%r15)

#CHECK: lg	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x04]
#CHECK: lg	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x04]
#CHECK: lg	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x04]
#CHECK: lg	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x04]
#CHECK: lg	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x04]
#CHECK: lg	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x04]
#CHECK: lg	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x04]
#CHECK: lg	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x04]
#CHECK: lg	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x04]
#CHECK: lg	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x04]

	lg	%r0, -524288
	lg	%r0, -1
	lg	%r0, 0
	lg	%r0, 1
	lg	%r0, 524287
	lg	%r0, 0(%r1)
	lg	%r0, 0(%r15)
	lg	%r0, 524287(%r1,%r15)
	lg	%r0, 524287(%r15,%r1)
	lg	%r15, 0

#CHECK: lgb	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x77]
#CHECK: lgb	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x77]
#CHECK: lgb	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x77]
#CHECK: lgb	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x77]
#CHECK: lgb	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x77]
#CHECK: lgb	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x77]
#CHECK: lgb	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x77]
#CHECK: lgb	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x77]
#CHECK: lgb	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x77]
#CHECK: lgb	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x77]

	lgb	%r0, -524288
	lgb	%r0, -1
	lgb	%r0, 0
	lgb	%r0, 1
	lgb	%r0, 524287
	lgb	%r0, 0(%r1)
	lgb	%r0, 0(%r15)
	lgb	%r0, 524287(%r1,%r15)
	lgb	%r0, 524287(%r15,%r1)
	lgb	%r15, 0


#CHECK: lgbr	%r0, %r15               # encoding: [0xb9,0x06,0x00,0x0f]
#CHECK: lgbr	%r7, %r8                # encoding: [0xb9,0x06,0x00,0x78]
#CHECK: lgbr	%r15, %r0               # encoding: [0xb9,0x06,0x00,0xf0]

	lgbr	%r0, %r15
	lgbr	%r7, %r8
	lgbr	%r15, %r0

#CHECK: lgdr	%r0, %f0                # encoding: [0xb3,0xcd,0x00,0x00]
#CHECK: lgdr	%r0, %f15               # encoding: [0xb3,0xcd,0x00,0x0f]
#CHECK: lgdr	%r15, %f0               # encoding: [0xb3,0xcd,0x00,0xf0]
#CHECK: lgdr	%r8, %f8                # encoding: [0xb3,0xcd,0x00,0x88]
#CHECK: lgdr	%r15, %f15              # encoding: [0xb3,0xcd,0x00,0xff]

	lgdr	%r0,%f0
	lgdr	%r0,%f15
	lgdr	%r15,%f0
	lgdr	%r8,%f8
	lgdr	%r15,%f15

#CHECK: lgf	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x14]
#CHECK: lgf	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x14]
#CHECK: lgf	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x14]
#CHECK: lgf	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x14]
#CHECK: lgf	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x14]
#CHECK: lgf	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x14]
#CHECK: lgf	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x14]
#CHECK: lgf	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x14]
#CHECK: lgf	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x14]
#CHECK: lgf	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x14]

	lgf	%r0, -524288
	lgf	%r0, -1
	lgf	%r0, 0
	lgf	%r0, 1
	lgf	%r0, 524287
	lgf	%r0, 0(%r1)
	lgf	%r0, 0(%r15)
	lgf	%r0, 524287(%r1,%r15)
	lgf	%r0, 524287(%r15,%r1)
	lgf	%r15, 0


#CHECK: lgfi	%r0, -2147483648        # encoding: [0xc0,0x01,0x80,0x00,0x00,0x00]
#CHECK: lgfi	%r0, -1                 # encoding: [0xc0,0x01,0xff,0xff,0xff,0xff]
#CHECK: lgfi	%r0, 0                  # encoding: [0xc0,0x01,0x00,0x00,0x00,0x00]
#CHECK: lgfi	%r0, 1                  # encoding: [0xc0,0x01,0x00,0x00,0x00,0x01]
#CHECK: lgfi	%r0, 2147483647         # encoding: [0xc0,0x01,0x7f,0xff,0xff,0xff]
#CHECK: lgfi	%r15, 0                 # encoding: [0xc0,0xf1,0x00,0x00,0x00,0x00]

	lgfi	%r0, -1 << 31
	lgfi	%r0, -1
	lgfi	%r0, 0
	lgfi	%r0, 1
	lgfi	%r0, (1 << 31) - 1
	lgfi	%r15, 0

#CHECK: lgfr	%r0, %r15               # encoding: [0xb9,0x14,0x00,0x0f]
#CHECK: lgfr	%r7, %r8                # encoding: [0xb9,0x14,0x00,0x78]
#CHECK: lgfr	%r15, %r0               # encoding: [0xb9,0x14,0x00,0xf0]

	lgfr	%r0, %r15
	lgfr	%r7, %r8
	lgfr	%r15, %r0

#CHECK: lgfrl	%r0, .[[LAB:L.*]]-4294967296 # encoding: [0xc4,0x0c,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-4294967296)+2, kind: FK_390_PC32DBL
	lgfrl	%r0, -0x100000000
#CHECK: lgfrl	%r0, .[[LAB:L.*]]-2	# encoding: [0xc4,0x0c,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC32DBL
	lgfrl	%r0, -2
#CHECK: lgfrl	%r0, .[[LAB:L.*]]	# encoding: [0xc4,0x0c,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC32DBL
	lgfrl	%r0, 0
#CHECK: lgfrl	%r0, .[[LAB:L.*]]+4294967294 # encoding: [0xc4,0x0c,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+4294967294)+2, kind: FK_390_PC32DBL
	lgfrl	%r0, 0xfffffffe

#CHECK: lgfrl	%r0, foo                # encoding: [0xc4,0x0c,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: lgfrl	%r15, foo               # encoding: [0xc4,0xfc,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL

	lgfrl	%r0,foo
	lgfrl	%r15,foo

#CHECK: lgfrl	%r3, bar+100            # encoding: [0xc4,0x3c,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: lgfrl	%r4, bar+100            # encoding: [0xc4,0x4c,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL

	lgfrl	%r3,bar+100
	lgfrl	%r4,bar+100

#CHECK: lgfrl	%r7, frob@PLT           # encoding: [0xc4,0x7c,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL
#CHECK: lgfrl	%r8, frob@PLT           # encoding: [0xc4,0x8c,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL

	lgfrl	%r7,frob@PLT
	lgfrl	%r8,frob@PLT

#CHECK: lgh	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x15]
#CHECK: lgh	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x15]
#CHECK: lgh	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x15]
#CHECK: lgh	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x15]
#CHECK: lgh	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x15]
#CHECK: lgh	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x15]
#CHECK: lgh	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x15]
#CHECK: lgh	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x15]
#CHECK: lgh	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x15]
#CHECK: lgh	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x15]

	lgh	%r0, -524288
	lgh	%r0, -1
	lgh	%r0, 0
	lgh	%r0, 1
	lgh	%r0, 524287
	lgh	%r0, 0(%r1)
	lgh	%r0, 0(%r15)
	lgh	%r0, 524287(%r1,%r15)
	lgh	%r0, 524287(%r15,%r1)
	lgh	%r15, 0


#CHECK: lghi	%r0, -32768             # encoding: [0xa7,0x09,0x80,0x00]
#CHECK: lghi	%r0, -1                 # encoding: [0xa7,0x09,0xff,0xff]
#CHECK: lghi	%r0, 0                  # encoding: [0xa7,0x09,0x00,0x00]
#CHECK: lghi	%r0, 1                  # encoding: [0xa7,0x09,0x00,0x01]
#CHECK: lghi	%r0, 32767              # encoding: [0xa7,0x09,0x7f,0xff]
#CHECK: lghi	%r15, 0                 # encoding: [0xa7,0xf9,0x00,0x00]
#CHECK: lghi	%r14, foo               # encoding: [0xa7,0xe9,A,A]

	lghi	%r0, -32768
	lghi	%r0, -1
	lghi	%r0, 0
	lghi	%r0, 1
	lghi	%r0, 32767
	lghi	%r15, 0
	lghi	%r14, foo

#CHECK: lghr	%r0, %r15               # encoding: [0xb9,0x07,0x00,0x0f]
#CHECK: lghr	%r7, %r8                # encoding: [0xb9,0x07,0x00,0x78]
#CHECK: lghr	%r15, %r0               # encoding: [0xb9,0x07,0x00,0xf0]

	lghr	%r0, %r15
	lghr	%r7, %r8
	lghr	%r15, %r0

#CHECK: lghrl	%r0, .[[LAB:L.*]]-4294967296 # encoding: [0xc4,0x04,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-4294967296)+2, kind: FK_390_PC32DBL
	lghrl	%r0, -0x100000000
#CHECK: lghrl	%r0, .[[LAB:L.*]]-2	# encoding: [0xc4,0x04,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC32DBL
	lghrl	%r0, -2
#CHECK: lghrl	%r0, .[[LAB:L.*]]	# encoding: [0xc4,0x04,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC32DBL
	lghrl	%r0, 0
#CHECK: lghrl	%r0, .[[LAB:L.*]]+4294967294 # encoding: [0xc4,0x04,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+4294967294)+2, kind: FK_390_PC32DBL
	lghrl	%r0, 0xfffffffe

#CHECK: lghrl	%r0, foo                # encoding: [0xc4,0x04,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: lghrl	%r15, foo               # encoding: [0xc4,0xf4,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL

	lghrl	%r0,foo
	lghrl	%r15,foo

#CHECK: lghrl	%r3, bar+100            # encoding: [0xc4,0x34,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: lghrl	%r4, bar+100            # encoding: [0xc4,0x44,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL

	lghrl	%r3,bar+100
	lghrl	%r4,bar+100

#CHECK: lghrl	%r7, frob@PLT           # encoding: [0xc4,0x74,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL
#CHECK: lghrl	%r8, frob@PLT           # encoding: [0xc4,0x84,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL

	lghrl	%r7,frob@PLT
	lghrl	%r8,frob@PLT

#CHECK: lgr	%r0, %r9                # encoding: [0xb9,0x04,0x00,0x09]
#CHECK: lgr	%r0, %r15               # encoding: [0xb9,0x04,0x00,0x0f]
#CHECK: lgr	%r15, %r0               # encoding: [0xb9,0x04,0x00,0xf0]
#CHECK: lgr	%r15, %r9               # encoding: [0xb9,0x04,0x00,0xf9]

	lgr	%r0,%r9
	lgr	%r0,%r15
	lgr	%r15,%r0
	lgr	%r15,%r9

#CHECK: lgrl	%r0, .[[LAB:L.*]]-4294967296 # encoding: [0xc4,0x08,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-4294967296)+2, kind: FK_390_PC32DBL
	lgrl	%r0, -0x100000000
#CHECK: lgrl	%r0, .[[LAB:L.*]]-2	# encoding: [0xc4,0x08,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC32DBL
	lgrl	%r0, -2
#CHECK: lgrl	%r0, .[[LAB:L.*]]	# encoding: [0xc4,0x08,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC32DBL
	lgrl	%r0, 0
#CHECK: lgrl	%r0, .[[LAB:L.*]]+4294967294 # encoding: [0xc4,0x08,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+4294967294)+2, kind: FK_390_PC32DBL
	lgrl	%r0, 0xfffffffe

#CHECK: lgrl	%r0, foo                # encoding: [0xc4,0x08,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: lgrl	%r15, foo               # encoding: [0xc4,0xf8,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL

	lgrl	%r0,foo
	lgrl	%r15,foo

#CHECK: lgrl	%r3, bar+100            # encoding: [0xc4,0x38,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: lgrl	%r4, bar+100            # encoding: [0xc4,0x48,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL

	lgrl	%r3,bar+100
	lgrl	%r4,bar+100

#CHECK: lgrl	%r7, frob@PLT           # encoding: [0xc4,0x78,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL
#CHECK: lgrl	%r8, frob@PLT           # encoding: [0xc4,0x88,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL

	lgrl	%r7,frob@PLT
	lgrl	%r8,frob@PLT

#CHECK: lh	%r0, 0                  # encoding: [0x48,0x00,0x00,0x00]
#CHECK: lh	%r0, 4095               # encoding: [0x48,0x00,0x0f,0xff]
#CHECK: lh	%r0, 0(%r1)             # encoding: [0x48,0x00,0x10,0x00]
#CHECK: lh	%r0, 0(%r15)            # encoding: [0x48,0x00,0xf0,0x00]
#CHECK: lh	%r0, 4095(%r1,%r15)     # encoding: [0x48,0x01,0xff,0xff]
#CHECK: lh	%r0, 4095(%r15,%r1)     # encoding: [0x48,0x0f,0x1f,0xff]
#CHECK: lh	%r15, 0                 # encoding: [0x48,0xf0,0x00,0x00]

	lh	%r0, 0
	lh	%r0, 4095
	lh	%r0, 0(%r1)
	lh	%r0, 0(%r15)
	lh	%r0, 4095(%r1,%r15)
	lh	%r0, 4095(%r15,%r1)
	lh	%r15, 0

#CHECK: lhi	%r0, -32768             # encoding: [0xa7,0x08,0x80,0x00]
#CHECK: lhi	%r0, -1                 # encoding: [0xa7,0x08,0xff,0xff]
#CHECK: lhi	%r0, 0                  # encoding: [0xa7,0x08,0x00,0x00]
#CHECK: lhi	%r0, 1                  # encoding: [0xa7,0x08,0x00,0x01]
#CHECK: lhi	%r0, 32767              # encoding: [0xa7,0x08,0x7f,0xff]
#CHECK: lhi	%r15, 0                 # encoding: [0xa7,0xf8,0x00,0x00]
#CHECK: lhi	%r0, foo                # encoding: [0xa7,0x08,A,A]

	lhi	%r0, -32768
	lhi	%r0, -1
	lhi	%r0, 0
	lhi	%r0, 1
	lhi	%r0, 32767
	lhi	%r15, 0
	lhi	%r0, foo

#CHECK: lhr	%r0, %r15               # encoding: [0xb9,0x27,0x00,0x0f]
#CHECK: lhr	%r7, %r8                # encoding: [0xb9,0x27,0x00,0x78]
#CHECK: lhr	%r15, %r0               # encoding: [0xb9,0x27,0x00,0xf0]

	lhr	%r0, %r15
	lhr	%r7, %r8
	lhr	%r15, %r0

#CHECK: lhrl	%r0, .[[LAB:L.*]]-4294967296 # encoding: [0xc4,0x05,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-4294967296)+2, kind: FK_390_PC32DBL
	lhrl	%r0, -0x100000000
#CHECK: lhrl	%r0, .[[LAB:L.*]]-2	# encoding: [0xc4,0x05,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC32DBL
	lhrl	%r0, -2
#CHECK: lhrl	%r0, .[[LAB:L.*]]	# encoding: [0xc4,0x05,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC32DBL
	lhrl	%r0, 0
#CHECK: lhrl	%r0, .[[LAB:L.*]]+4294967294 # encoding: [0xc4,0x05,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+4294967294)+2, kind: FK_390_PC32DBL
	lhrl	%r0, 0xfffffffe

#CHECK: lhrl	%r0, foo                # encoding: [0xc4,0x05,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: lhrl	%r15, foo               # encoding: [0xc4,0xf5,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL

	lhrl	%r0,foo
	lhrl	%r15,foo

#CHECK: lhrl	%r3, bar+100            # encoding: [0xc4,0x35,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: lhrl	%r4, bar+100            # encoding: [0xc4,0x45,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL

	lhrl	%r3,bar+100
	lhrl	%r4,bar+100

#CHECK: lhrl	%r7, frob@PLT           # encoding: [0xc4,0x75,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL
#CHECK: lhrl	%r8, frob@PLT           # encoding: [0xc4,0x85,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL

	lhrl	%r7,frob@PLT
	lhrl	%r8,frob@PLT

#CHECK: lhy	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x78]
#CHECK: lhy	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x78]
#CHECK: lhy	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x78]
#CHECK: lhy	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x78]
#CHECK: lhy	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x78]
#CHECK: lhy	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x78]
#CHECK: lhy	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x78]
#CHECK: lhy	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x78]
#CHECK: lhy	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x78]
#CHECK: lhy	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x78]

	lhy	%r0, -524288
	lhy	%r0, -1
	lhy	%r0, 0
	lhy	%r0, 1
	lhy	%r0, 524287
	lhy	%r0, 0(%r1)
	lhy	%r0, 0(%r15)
	lhy	%r0, 524287(%r1,%r15)
	lhy	%r0, 524287(%r15,%r1)
	lhy	%r15, 0

#CHECK: llc	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x94]
#CHECK: llc	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x94]
#CHECK: llc	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x94]
#CHECK: llc	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x94]
#CHECK: llc	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x94]
#CHECK: llc	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x94]
#CHECK: llc	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x94]
#CHECK: llc	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x94]
#CHECK: llc	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x94]
#CHECK: llc	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x94]

	llc	%r0, -524288
	llc	%r0, -1
	llc	%r0, 0
	llc	%r0, 1
	llc	%r0, 524287
	llc	%r0, 0(%r1)
	llc	%r0, 0(%r15)
	llc	%r0, 524287(%r1,%r15)
	llc	%r0, 524287(%r15,%r1)
	llc	%r15, 0

#CHECK: llcr	%r0, %r15               # encoding: [0xb9,0x94,0x00,0x0f]
#CHECK: llcr	%r7, %r8                # encoding: [0xb9,0x94,0x00,0x78]
#CHECK: llcr	%r15, %r0               # encoding: [0xb9,0x94,0x00,0xf0]

	llcr	%r0, %r15
	llcr	%r7, %r8
	llcr	%r15, %r0

#CHECK: llgc	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x90]
#CHECK: llgc	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x90]
#CHECK: llgc	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x90]
#CHECK: llgc	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x90]
#CHECK: llgc	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x90]
#CHECK: llgc	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x90]
#CHECK: llgc	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x90]
#CHECK: llgc	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x90]
#CHECK: llgc	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x90]
#CHECK: llgc	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x90]

	llgc	%r0, -524288
	llgc	%r0, -1
	llgc	%r0, 0
	llgc	%r0, 1
	llgc	%r0, 524287
	llgc	%r0, 0(%r1)
	llgc	%r0, 0(%r15)
	llgc	%r0, 524287(%r1,%r15)
	llgc	%r0, 524287(%r15,%r1)
	llgc	%r15, 0


#CHECK: llgcr	%r0, %r15               # encoding: [0xb9,0x84,0x00,0x0f]
#CHECK: llgcr	%r7, %r8                # encoding: [0xb9,0x84,0x00,0x78]
#CHECK: llgcr	%r15, %r0               # encoding: [0xb9,0x84,0x00,0xf0]

	llgcr	%r0, %r15
	llgcr	%r7, %r8
	llgcr	%r15, %r0

#CHECK: llgf	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x16]
#CHECK: llgf	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x16]
#CHECK: llgf	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x16]
#CHECK: llgf	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x16]
#CHECK: llgf	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x16]
#CHECK: llgf	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x16]
#CHECK: llgf	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x16]
#CHECK: llgf	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x16]
#CHECK: llgf	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x16]
#CHECK: llgf	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x16]

	llgf	%r0, -524288
	llgf	%r0, -1
	llgf	%r0, 0
	llgf	%r0, 1
	llgf	%r0, 524287
	llgf	%r0, 0(%r1)
	llgf	%r0, 0(%r15)
	llgf	%r0, 524287(%r1,%r15)
	llgf	%r0, 524287(%r15,%r1)
	llgf	%r15, 0


#CHECK: llgfr	%r0, %r15               # encoding: [0xb9,0x16,0x00,0x0f]
#CHECK: llgfr	%r7, %r8                # encoding: [0xb9,0x16,0x00,0x78]
#CHECK: llgfr	%r15, %r0               # encoding: [0xb9,0x16,0x00,0xf0]

	llgfr	%r0, %r15
	llgfr	%r7, %r8
	llgfr	%r15, %r0

#CHECK: llgfrl	%r0, .[[LAB:L.*]]-4294967296 # encoding: [0xc4,0x0e,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-4294967296)+2, kind: FK_390_PC32DBL
	llgfrl	%r0, -0x100000000
#CHECK: llgfrl	%r0, .[[LAB:L.*]]-2	# encoding: [0xc4,0x0e,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC32DBL
	llgfrl	%r0, -2
#CHECK: llgfrl	%r0, .[[LAB:L.*]]	# encoding: [0xc4,0x0e,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC32DBL
	llgfrl	%r0, 0
#CHECK: llgfrl	%r0, .[[LAB:L.*]]+4294967294 # encoding: [0xc4,0x0e,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+4294967294)+2, kind: FK_390_PC32DBL
	llgfrl	%r0, 0xfffffffe

#CHECK: llgfrl	%r0, foo                # encoding: [0xc4,0x0e,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: llgfrl	%r15, foo               # encoding: [0xc4,0xfe,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL

	llgfrl	%r0,foo
	llgfrl	%r15,foo

#CHECK: llgfrl	%r3, bar+100            # encoding: [0xc4,0x3e,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: llgfrl	%r4, bar+100            # encoding: [0xc4,0x4e,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL

	llgfrl	%r3,bar+100
	llgfrl	%r4,bar+100

#CHECK: llgfrl	%r7, frob@PLT           # encoding: [0xc4,0x7e,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL
#CHECK: llgfrl	%r8, frob@PLT           # encoding: [0xc4,0x8e,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL

	llgfrl	%r7,frob@PLT
	llgfrl	%r8,frob@PLT

#CHECK: llgh	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x91]
#CHECK: llgh	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x91]
#CHECK: llgh	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x91]
#CHECK: llgh	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x91]
#CHECK: llgh	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x91]
#CHECK: llgh	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x91]
#CHECK: llgh	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x91]
#CHECK: llgh	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x91]
#CHECK: llgh	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x91]
#CHECK: llgh	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x91]

	llgh	%r0, -524288
	llgh	%r0, -1
	llgh	%r0, 0
	llgh	%r0, 1
	llgh	%r0, 524287
	llgh	%r0, 0(%r1)
	llgh	%r0, 0(%r15)
	llgh	%r0, 524287(%r1,%r15)
	llgh	%r0, 524287(%r15,%r1)
	llgh	%r15, 0

#CHECK: llghr	%r0, %r15               # encoding: [0xb9,0x85,0x00,0x0f]
#CHECK: llghr	%r7, %r8                # encoding: [0xb9,0x85,0x00,0x78]
#CHECK: llghr	%r15, %r0               # encoding: [0xb9,0x85,0x00,0xf0]

	llghr	%r0, %r15
	llghr	%r7, %r8
	llghr	%r15, %r0

#CHECK: llghrl	%r0, .[[LAB:L.*]]-4294967296 # encoding: [0xc4,0x06,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-4294967296)+2, kind: FK_390_PC32DBL
	llghrl	%r0, -0x100000000
#CHECK: llghrl	%r0, .[[LAB:L.*]]-2	# encoding: [0xc4,0x06,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC32DBL
	llghrl	%r0, -2
#CHECK: llghrl	%r0, .[[LAB:L.*]]	# encoding: [0xc4,0x06,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC32DBL
	llghrl	%r0, 0
#CHECK: llghrl	%r0, .[[LAB:L.*]]+4294967294 # encoding: [0xc4,0x06,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+4294967294)+2, kind: FK_390_PC32DBL
	llghrl	%r0, 0xfffffffe

#CHECK: llghrl	%r0, foo                # encoding: [0xc4,0x06,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: llghrl	%r15, foo               # encoding: [0xc4,0xf6,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL

	llghrl	%r0,foo
	llghrl	%r15,foo

#CHECK: llghrl	%r3, bar+100            # encoding: [0xc4,0x36,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: llghrl	%r4, bar+100            # encoding: [0xc4,0x46,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL

	llghrl	%r3,bar+100
	llghrl	%r4,bar+100

#CHECK: llghrl	%r7, frob@PLT           # encoding: [0xc4,0x76,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL
#CHECK: llghrl	%r8, frob@PLT           # encoding: [0xc4,0x86,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL

	llghrl	%r7,frob@PLT
	llghrl	%r8,frob@PLT

#CHECK: llgt	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x17]
#CHECK: llgt	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x17]
#CHECK: llgt	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x17]
#CHECK: llgt	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x17]
#CHECK: llgt	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x17]
#CHECK: llgt	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x17]
#CHECK: llgt	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x17]
#CHECK: llgt	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x17]
#CHECK: llgt	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x17]
#CHECK: llgt	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x17]

	llgt	%r0, -524288
	llgt	%r0, -1
	llgt	%r0, 0
	llgt	%r0, 1
	llgt	%r0, 524287
	llgt	%r0, 0(%r1)
	llgt	%r0, 0(%r15)
	llgt	%r0, 524287(%r1,%r15)
	llgt	%r0, 524287(%r15,%r1)
	llgt	%r15, 0

#CHECK: llgtr	%r0, %r15               # encoding: [0xb9,0x17,0x00,0x0f]
#CHECK: llgtr	%r7, %r8                # encoding: [0xb9,0x17,0x00,0x78]
#CHECK: llgtr	%r15, %r0               # encoding: [0xb9,0x17,0x00,0xf0]

	llgtr	%r0, %r15
	llgtr	%r7, %r8
	llgtr	%r15, %r0

#CHECK: llh	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x95]
#CHECK: llh	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x95]
#CHECK: llh	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x95]
#CHECK: llh	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x95]
#CHECK: llh	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x95]
#CHECK: llh	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x95]
#CHECK: llh	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x95]
#CHECK: llh	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x95]
#CHECK: llh	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x95]
#CHECK: llh	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x95]

	llh	%r0, -524288
	llh	%r0, -1
	llh	%r0, 0
	llh	%r0, 1
	llh	%r0, 524287
	llh	%r0, 0(%r1)
	llh	%r0, 0(%r15)
	llh	%r0, 524287(%r1,%r15)
	llh	%r0, 524287(%r15,%r1)
	llh	%r15, 0

#CHECK: llhr	%r0, %r15               # encoding: [0xb9,0x95,0x00,0x0f]
#CHECK: llhr	%r7, %r8                # encoding: [0xb9,0x95,0x00,0x78]
#CHECK: llhr	%r15, %r0               # encoding: [0xb9,0x95,0x00,0xf0]

	llhr	%r0, %r15
	llhr	%r7, %r8
	llhr	%r15, %r0

#CHECK: llhrl	%r0, .[[LAB:L.*]]-4294967296 # encoding: [0xc4,0x02,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-4294967296)+2, kind: FK_390_PC32DBL
	llhrl	%r0, -0x100000000
#CHECK: llhrl	%r0, .[[LAB:L.*]]-2	# encoding: [0xc4,0x02,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC32DBL
	llhrl	%r0, -2
#CHECK: llhrl	%r0, .[[LAB:L.*]]	# encoding: [0xc4,0x02,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC32DBL
	llhrl	%r0, 0
#CHECK: llhrl	%r0, .[[LAB:L.*]]+4294967294 # encoding: [0xc4,0x02,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+4294967294)+2, kind: FK_390_PC32DBL
	llhrl	%r0, 0xfffffffe

#CHECK: llhrl	%r0, foo                # encoding: [0xc4,0x02,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: llhrl	%r15, foo               # encoding: [0xc4,0xf2,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL

	llhrl	%r0,foo
	llhrl	%r15,foo

#CHECK: llhrl	%r3, bar+100            # encoding: [0xc4,0x32,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: llhrl	%r4, bar+100            # encoding: [0xc4,0x42,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL

	llhrl	%r3,bar+100
	llhrl	%r4,bar+100

#CHECK: llhrl	%r7, frob@PLT           # encoding: [0xc4,0x72,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL
#CHECK: llhrl	%r8, frob@PLT           # encoding: [0xc4,0x82,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL

	llhrl	%r7,frob@PLT
	llhrl	%r8,frob@PLT

#CHECK: llihf	%r0, 0                  # encoding: [0xc0,0x0e,0x00,0x00,0x00,0x00]
#CHECK: llihf	%r0, 4294967295         # encoding: [0xc0,0x0e,0xff,0xff,0xff,0xff]
#CHECK: llihf	%r15, 0                 # encoding: [0xc0,0xfe,0x00,0x00,0x00,0x00]

	llihf	%r0, 0
	llihf	%r0, 0xffffffff
	llihf	%r15, 0

#CHECK: llihh	%r0, 0                  # encoding: [0xa5,0x0c,0x00,0x00]
#CHECK: llihh	%r0, 32768              # encoding: [0xa5,0x0c,0x80,0x00]
#CHECK: llihh	%r0, 65535              # encoding: [0xa5,0x0c,0xff,0xff]
#CHECK: llihh	%r15, 0                 # encoding: [0xa5,0xfc,0x00,0x00]

	llihh	%r0, 0
	llihh	%r0, 0x8000
	llihh	%r0, 0xffff
	llihh	%r15, 0

#CHECK: llihl	%r0, 0                  # encoding: [0xa5,0x0d,0x00,0x00]
#CHECK: llihl	%r0, 32768              # encoding: [0xa5,0x0d,0x80,0x00]
#CHECK: llihl	%r0, 65535              # encoding: [0xa5,0x0d,0xff,0xff]
#CHECK: llihl	%r15, 0                 # encoding: [0xa5,0xfd,0x00,0x00]

	llihl	%r0, 0
	llihl	%r0, 0x8000
	llihl	%r0, 0xffff
	llihl	%r15, 0

#CHECK: llilf	%r0, 0                  # encoding: [0xc0,0x0f,0x00,0x00,0x00,0x00]
#CHECK: llilf	%r0, 4294967295         # encoding: [0xc0,0x0f,0xff,0xff,0xff,0xff]
#CHECK: llilf	%r15, 0                 # encoding: [0xc0,0xff,0x00,0x00,0x00,0x00]
#CHECK: llilf	%r15, 0                 # encoding: [0xc0,0xff,0x00,0x00,0x00,0x00]

	llilf	%r0, 0
	llilf	%r0, 0xffffffff
	llilf	%r15, 0
	llgfi	%r15, 0

#CHECK: llilh	%r0, 0                  # encoding: [0xa5,0x0e,0x00,0x00]
#CHECK: llilh	%r0, 32768              # encoding: [0xa5,0x0e,0x80,0x00]
#CHECK: llilh	%r0, 65535              # encoding: [0xa5,0x0e,0xff,0xff]
#CHECK: llilh	%r15, 0                 # encoding: [0xa5,0xfe,0x00,0x00]

	llilh	%r0, 0
	llilh	%r0, 0x8000
	llilh	%r0, 0xffff
	llilh	%r15, 0

#CHECK: llill	%r0, 0                  # encoding: [0xa5,0x0f,0x00,0x00]
#CHECK: llill	%r0, 32768              # encoding: [0xa5,0x0f,0x80,0x00]
#CHECK: llill	%r0, 65535              # encoding: [0xa5,0x0f,0xff,0xff]
#CHECK: llill	%r15, 0                 # encoding: [0xa5,0xff,0x00,0x00]
#CHECK: llill	%r15, 0                 # encoding: [0xa5,0xff,0x00,0x00]

	llill	%r0, 0
	llill	%r0, 0x8000
	llill	%r0, 0xffff
	llill	%r15, 0
	llghi	%r15, 0

#CHECK: lm	%r0, %r0, 0             # encoding: [0x98,0x00,0x00,0x00]
#CHECK: lm	%r0, %r15, 0            # encoding: [0x98,0x0f,0x00,0x00]
#CHECK: lm	%r14, %r15, 0           # encoding: [0x98,0xef,0x00,0x00]
#CHECK: lm	%r15, %r15, 0           # encoding: [0x98,0xff,0x00,0x00]
#CHECK: lm	%r0, %r0, 4095          # encoding: [0x98,0x00,0x0f,0xff]
#CHECK: lm	%r0, %r0, 1             # encoding: [0x98,0x00,0x00,0x01]
#CHECK: lm	%r0, %r0, 0(%r1)        # encoding: [0x98,0x00,0x10,0x00]
#CHECK: lm	%r0, %r0, 0(%r15)       # encoding: [0x98,0x00,0xf0,0x00]
#CHECK: lm	%r0, %r0, 4095(%r1)     # encoding: [0x98,0x00,0x1f,0xff]
#CHECK: lm	%r0, %r0, 4095(%r15)    # encoding: [0x98,0x00,0xff,0xff]

	lm	%r0,%r0,0
	lm	%r0,%r15,0
	lm	%r14,%r15,0
	lm	%r15,%r15,0
	lm	%r0,%r0,4095
	lm	%r0,%r0,1
	lm	%r0,%r0,0(%r1)
	lm	%r0,%r0,0(%r15)
	lm	%r0,%r0,4095(%r1)
	lm	%r0,%r0,4095(%r15)

#CHECK: lmd	%r0, %r0, 0, 0               # encoding: [0xef,0x00,0x00,0x00,0x00,0x00]
#CHECK: lmd	%r0, %r15, 0, 0              # encoding: [0xef,0x0f,0x00,0x00,0x00,0x00]
#CHECK: lmd	%r14, %r15, 0, 0             # encoding: [0xef,0xef,0x00,0x00,0x00,0x00]
#CHECK: lmd	%r15, %r15, 0, 0             # encoding: [0xef,0xff,0x00,0x00,0x00,0x00]
#CHECK: lmd	%r2, %r4, 0(%r1), 0(%r15)    # encoding: [0xef,0x24,0x10,0x00,0xf0,0x00]
#CHECK: lmd	%r2, %r4, 1(%r1), 0(%r15)    # encoding: [0xef,0x24,0x10,0x01,0xf0,0x00]
#CHECK: lmd	%r2, %r4, 4095(%r1), 0(%r15) # encoding: [0xef,0x24,0x1f,0xff,0xf0,0x00]
#CHECK: lmd	%r2, %r4, 0(%r1), 1(%r15)    # encoding: [0xef,0x24,0x10,0x00,0xf0,0x01]
#CHECK: lmd	%r2, %r4, 0(%r1), 4095(%r15) # encoding: [0xef,0x24,0x10,0x00,0xff,0xff]

        lmd	%r0, %r0, 0, 0
        lmd	%r0, %r15, 0, 0
        lmd	%r14, %r15, 0, 0
        lmd	%r15, %r15, 0, 0
        lmd	%r2, %r4, 0(%r1), 0(%r15)
        lmd	%r2, %r4, 1(%r1), 0(%r15)
        lmd	%r2, %r4, 4095(%r1), 0(%r15)
        lmd	%r2, %r4, 0(%r1), 1(%r15)
        lmd	%r2, %r4, 0(%r1), 4095(%r15)

#CHECK: lmg	%r0, %r0, 0             # encoding: [0xeb,0x00,0x00,0x00,0x00,0x04]
#CHECK: lmg	%r0, %r15, 0            # encoding: [0xeb,0x0f,0x00,0x00,0x00,0x04]
#CHECK: lmg	%r14, %r15, 0           # encoding: [0xeb,0xef,0x00,0x00,0x00,0x04]
#CHECK: lmg	%r15, %r15, 0           # encoding: [0xeb,0xff,0x00,0x00,0x00,0x04]
#CHECK: lmg	%r0, %r0, -524288       # encoding: [0xeb,0x00,0x00,0x00,0x80,0x04]
#CHECK: lmg	%r0, %r0, -1            # encoding: [0xeb,0x00,0x0f,0xff,0xff,0x04]
#CHECK: lmg	%r0, %r0, 0             # encoding: [0xeb,0x00,0x00,0x00,0x00,0x04]
#CHECK: lmg	%r0, %r0, 1             # encoding: [0xeb,0x00,0x00,0x01,0x00,0x04]
#CHECK: lmg	%r0, %r0, 524287        # encoding: [0xeb,0x00,0x0f,0xff,0x7f,0x04]
#CHECK: lmg	%r0, %r0, 0(%r1)        # encoding: [0xeb,0x00,0x10,0x00,0x00,0x04]
#CHECK: lmg	%r0, %r0, 0(%r15)       # encoding: [0xeb,0x00,0xf0,0x00,0x00,0x04]
#CHECK: lmg	%r0, %r0, 524287(%r1)   # encoding: [0xeb,0x00,0x1f,0xff,0x7f,0x04]
#CHECK: lmg	%r0, %r0, 524287(%r15)  # encoding: [0xeb,0x00,0xff,0xff,0x7f,0x04]

	lmg	%r0,%r0,0
	lmg	%r0,%r15,0
	lmg	%r14,%r15,0
	lmg	%r15,%r15,0
	lmg	%r0,%r0,-524288
	lmg	%r0,%r0,-1
	lmg	%r0,%r0,0
	lmg	%r0,%r0,1
	lmg	%r0,%r0,524287
	lmg	%r0,%r0,0(%r1)
	lmg	%r0,%r0,0(%r15)
	lmg	%r0,%r0,524287(%r1)
	lmg	%r0,%r0,524287(%r15)

#CHECK: lmh	%r0, %r0, 0             # encoding: [0xeb,0x00,0x00,0x00,0x00,0x96]
#CHECK: lmh	%r0, %r15, 0            # encoding: [0xeb,0x0f,0x00,0x00,0x00,0x96]
#CHECK: lmh	%r14, %r15, 0           # encoding: [0xeb,0xef,0x00,0x00,0x00,0x96]
#CHECK: lmh	%r15, %r15, 0           # encoding: [0xeb,0xff,0x00,0x00,0x00,0x96]
#CHECK: lmh	%r0, %r0, -524288       # encoding: [0xeb,0x00,0x00,0x00,0x80,0x96]
#CHECK: lmh	%r0, %r0, -1            # encoding: [0xeb,0x00,0x0f,0xff,0xff,0x96]
#CHECK: lmh	%r0, %r0, 0             # encoding: [0xeb,0x00,0x00,0x00,0x00,0x96]
#CHECK: lmh	%r0, %r0, 1             # encoding: [0xeb,0x00,0x00,0x01,0x00,0x96]
#CHECK: lmh	%r0, %r0, 524287        # encoding: [0xeb,0x00,0x0f,0xff,0x7f,0x96]
#CHECK: lmh	%r0, %r0, 0(%r1)        # encoding: [0xeb,0x00,0x10,0x00,0x00,0x96]
#CHECK: lmh	%r0, %r0, 0(%r15)       # encoding: [0xeb,0x00,0xf0,0x00,0x00,0x96]
#CHECK: lmh	%r0, %r0, 524287(%r1)   # encoding: [0xeb,0x00,0x1f,0xff,0x7f,0x96]
#CHECK: lmh	%r0, %r0, 524287(%r15)  # encoding: [0xeb,0x00,0xff,0xff,0x7f,0x96]

	lmh	%r0,%r0,0
	lmh	%r0,%r15,0
	lmh	%r14,%r15,0
	lmh	%r15,%r15,0
	lmh	%r0,%r0,-524288
	lmh	%r0,%r0,-1
	lmh	%r0,%r0,0
	lmh	%r0,%r0,1
	lmh	%r0,%r0,524287
	lmh	%r0,%r0,0(%r1)
	lmh	%r0,%r0,0(%r15)
	lmh	%r0,%r0,524287(%r1)
	lmh	%r0,%r0,524287(%r15)

#CHECK: lmy	%r0, %r0, 0             # encoding: [0xeb,0x00,0x00,0x00,0x00,0x98]
#CHECK: lmy	%r0, %r15, 0            # encoding: [0xeb,0x0f,0x00,0x00,0x00,0x98]
#CHECK: lmy	%r14, %r15, 0           # encoding: [0xeb,0xef,0x00,0x00,0x00,0x98]
#CHECK: lmy	%r15, %r15, 0           # encoding: [0xeb,0xff,0x00,0x00,0x00,0x98]
#CHECK: lmy	%r0, %r0, -524288       # encoding: [0xeb,0x00,0x00,0x00,0x80,0x98]
#CHECK: lmy	%r0, %r0, -1            # encoding: [0xeb,0x00,0x0f,0xff,0xff,0x98]
#CHECK: lmy	%r0, %r0, 0             # encoding: [0xeb,0x00,0x00,0x00,0x00,0x98]
#CHECK: lmy	%r0, %r0, 1             # encoding: [0xeb,0x00,0x00,0x01,0x00,0x98]
#CHECK: lmy	%r0, %r0, 524287        # encoding: [0xeb,0x00,0x0f,0xff,0x7f,0x98]
#CHECK: lmy	%r0, %r0, 0(%r1)        # encoding: [0xeb,0x00,0x10,0x00,0x00,0x98]
#CHECK: lmy	%r0, %r0, 0(%r15)       # encoding: [0xeb,0x00,0xf0,0x00,0x00,0x98]
#CHECK: lmy	%r0, %r0, 524287(%r1)   # encoding: [0xeb,0x00,0x1f,0xff,0x7f,0x98]
#CHECK: lmy	%r0, %r0, 524287(%r15)  # encoding: [0xeb,0x00,0xff,0xff,0x7f,0x98]

	lmy	%r0,%r0,0
	lmy	%r0,%r15,0
	lmy	%r14,%r15,0
	lmy	%r15,%r15,0
	lmy	%r0,%r0,-524288
	lmy	%r0,%r0,-1
	lmy	%r0,%r0,0
	lmy	%r0,%r0,1
	lmy	%r0,%r0,524287
	lmy	%r0,%r0,0(%r1)
	lmy	%r0,%r0,0(%r15)
	lmy	%r0,%r0,524287(%r1)
	lmy	%r0,%r0,524287(%r15)

#CHECK: lndbr	%f0, %f9                # encoding: [0xb3,0x11,0x00,0x09]
#CHECK: lndbr	%f0, %f15               # encoding: [0xb3,0x11,0x00,0x0f]
#CHECK: lndbr	%f15, %f0               # encoding: [0xb3,0x11,0x00,0xf0]
#CHECK: lndbr	%f15, %f9               # encoding: [0xb3,0x11,0x00,0xf9]

	lndbr	%f0,%f9
	lndbr	%f0,%f15
	lndbr	%f15,%f0
	lndbr	%f15,%f9

#CHECK: lndr	%f0, %f9                # encoding: [0x21,0x09]
#CHECK: lndr	%f0, %f15               # encoding: [0x21,0x0f]
#CHECK: lndr	%f15, %f0               # encoding: [0x21,0xf0]
#CHECK: lndr	%f15, %f9               # encoding: [0x21,0xf9]

	lndr	%f0,%f9
	lndr	%f0,%f15
	lndr	%f15,%f0
	lndr	%f15,%f9

#CHECK: lnebr	%f0, %f9                # encoding: [0xb3,0x01,0x00,0x09]
#CHECK: lnebr	%f0, %f15               # encoding: [0xb3,0x01,0x00,0x0f]
#CHECK: lnebr	%f15, %f0               # encoding: [0xb3,0x01,0x00,0xf0]
#CHECK: lnebr	%f15, %f9               # encoding: [0xb3,0x01,0x00,0xf9]

	lnebr	%f0,%f9
	lnebr	%f0,%f15
	lnebr	%f15,%f0
	lnebr	%f15,%f9

#CHECK: lner	%f0, %f9                # encoding: [0x31,0x09]
#CHECK: lner	%f0, %f15               # encoding: [0x31,0x0f]
#CHECK: lner	%f15, %f0               # encoding: [0x31,0xf0]
#CHECK: lner	%f15, %f9               # encoding: [0x31,0xf9]

	lner	%f0,%f9
	lner	%f0,%f15
	lner	%f15,%f0
	lner	%f15,%f9

#CHECK: lngfr	%r0, %r0                # encoding: [0xb9,0x11,0x00,0x00]
#CHECK: lngfr	%r0, %r15               # encoding: [0xb9,0x11,0x00,0x0f]
#CHECK: lngfr	%r15, %r0               # encoding: [0xb9,0x11,0x00,0xf0]
#CHECK: lngfr	%r7, %r8                # encoding: [0xb9,0x11,0x00,0x78]

	lngfr	%r0,%r0
	lngfr	%r0,%r15
	lngfr	%r15,%r0
	lngfr	%r7,%r8

#CHECK: lngr	%r0, %r0                # encoding: [0xb9,0x01,0x00,0x00]
#CHECK: lngr	%r0, %r15               # encoding: [0xb9,0x01,0x00,0x0f]
#CHECK: lngr	%r15, %r0               # encoding: [0xb9,0x01,0x00,0xf0]
#CHECK: lngr	%r7, %r8                # encoding: [0xb9,0x01,0x00,0x78]

	lngr	%r0,%r0
	lngr	%r0,%r15
	lngr	%r15,%r0
	lngr	%r7,%r8

#CHECK: lnr	%r0, %r0                # encoding: [0x11,0x00]
#CHECK: lnr	%r0, %r15               # encoding: [0x11,0x0f]
#CHECK: lnr	%r15, %r0               # encoding: [0x11,0xf0]
#CHECK: lnr	%r7, %r8                # encoding: [0x11,0x78]

	lnr	%r0,%r0
	lnr	%r0,%r15
	lnr	%r15,%r0
	lnr	%r7,%r8

#CHECK: lnxbr	%f0, %f8                # encoding: [0xb3,0x41,0x00,0x08]
#CHECK: lnxbr	%f0, %f13               # encoding: [0xb3,0x41,0x00,0x0d]
#CHECK: lnxbr	%f13, %f0               # encoding: [0xb3,0x41,0x00,0xd0]
#CHECK: lnxbr	%f13, %f9               # encoding: [0xb3,0x41,0x00,0xd9]

	lnxbr	%f0,%f8
	lnxbr	%f0,%f13
	lnxbr	%f13,%f0
	lnxbr	%f13,%f9

#CHECK: lnxr	%f0, %f8                # encoding: [0xb3,0x61,0x00,0x08]
#CHECK: lnxr	%f0, %f13               # encoding: [0xb3,0x61,0x00,0x0d]
#CHECK: lnxr	%f13, %f0               # encoding: [0xb3,0x61,0x00,0xd0]
#CHECK: lnxr	%f13, %f9               # encoding: [0xb3,0x61,0x00,0xd9]

	lnxr	%f0,%f8
	lnxr	%f0,%f13
	lnxr	%f13,%f0
	lnxr	%f13,%f9

#CHECK: lpctl	0                       # encoding: [0xb2,0x85,0x00,0x00]
#CHECK: lpctl	0(%r1)                  # encoding: [0xb2,0x85,0x10,0x00]
#CHECK: lpctl	0(%r15)                 # encoding: [0xb2,0x85,0xf0,0x00]
#CHECK: lpctl	4095                    # encoding: [0xb2,0x85,0x0f,0xff]
#CHECK: lpctl	4095(%r1)               # encoding: [0xb2,0x85,0x1f,0xff]
#CHECK: lpctl	4095(%r15)              # encoding: [0xb2,0x85,0xff,0xff]

	lpctl	0
	lpctl	0(%r1)
	lpctl	0(%r15)
	lpctl	4095
	lpctl	4095(%r1)
	lpctl	4095(%r15)

#CHECK: lpdbr	%f0, %f9                # encoding: [0xb3,0x10,0x00,0x09]
#CHECK: lpdbr	%f0, %f15               # encoding: [0xb3,0x10,0x00,0x0f]
#CHECK: lpdbr	%f15, %f0               # encoding: [0xb3,0x10,0x00,0xf0]
#CHECK: lpdbr	%f15, %f9               # encoding: [0xb3,0x10,0x00,0xf9]

	lpdbr	%f0,%f9
	lpdbr	%f0,%f15
	lpdbr	%f15,%f0
	lpdbr	%f15,%f9

#CHECK: lpdr	%f0, %f9                # encoding: [0x20,0x09]
#CHECK: lpdr	%f0, %f15               # encoding: [0x20,0x0f]
#CHECK: lpdr	%f15, %f0               # encoding: [0x20,0xf0]
#CHECK: lpdr	%f15, %f9               # encoding: [0x20,0xf9]

	lpdr	%f0,%f9
	lpdr	%f0,%f15
	lpdr	%f15,%f0
	lpdr	%f15,%f9

#CHECK: lpebr	%f0, %f9                # encoding: [0xb3,0x00,0x00,0x09]
#CHECK: lpebr	%f0, %f15               # encoding: [0xb3,0x00,0x00,0x0f]
#CHECK: lpebr	%f15, %f0               # encoding: [0xb3,0x00,0x00,0xf0]
#CHECK: lpebr	%f15, %f9               # encoding: [0xb3,0x00,0x00,0xf9]

	lpebr	%f0,%f9
	lpebr	%f0,%f15
	lpebr	%f15,%f0
	lpebr	%f15,%f9

#CHECK: lper	%f0, %f9                # encoding: [0x30,0x09]
#CHECK: lper	%f0, %f15               # encoding: [0x30,0x0f]
#CHECK: lper	%f15, %f0               # encoding: [0x30,0xf0]
#CHECK: lper	%f15, %f9               # encoding: [0x30,0xf9]

	lper	%f0,%f9
	lper	%f0,%f15
	lper	%f15,%f0
	lper	%f15,%f9

#CHECK: lpgfr	%r0, %r0                # encoding: [0xb9,0x10,0x00,0x00]
#CHECK: lpgfr	%r0, %r15               # encoding: [0xb9,0x10,0x00,0x0f]
#CHECK: lpgfr	%r15, %r0               # encoding: [0xb9,0x10,0x00,0xf0]
#CHECK: lpgfr	%r7, %r8                # encoding: [0xb9,0x10,0x00,0x78]

	lpgfr	%r0,%r0
	lpgfr	%r0,%r15
	lpgfr	%r15,%r0
	lpgfr	%r7,%r8

#CHECK: lpgr	%r0, %r0                # encoding: [0xb9,0x00,0x00,0x00]
#CHECK: lpgr	%r0, %r15               # encoding: [0xb9,0x00,0x00,0x0f]
#CHECK: lpgr	%r15, %r0               # encoding: [0xb9,0x00,0x00,0xf0]
#CHECK: lpgr	%r7, %r8                # encoding: [0xb9,0x00,0x00,0x78]

	lpgr	%r0,%r0
	lpgr	%r0,%r15
	lpgr	%r15,%r0
	lpgr	%r7,%r8

#CHECK: lpp	0                       # encoding: [0xb2,0x80,0x00,0x00]
#CHECK: lpp	0(%r1)                  # encoding: [0xb2,0x80,0x10,0x00]
#CHECK: lpp	0(%r15)                 # encoding: [0xb2,0x80,0xf0,0x00]
#CHECK: lpp	4095                    # encoding: [0xb2,0x80,0x0f,0xff]
#CHECK: lpp	4095(%r1)               # encoding: [0xb2,0x80,0x1f,0xff]
#CHECK: lpp	4095(%r15)              # encoding: [0xb2,0x80,0xff,0xff]

	lpp	0
	lpp	0(%r1)
	lpp	0(%r15)
	lpp	4095
	lpp	4095(%r1)
	lpp	4095(%r15)

#CHECK: lpq	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x8f]
#CHECK: lpq	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x8f]
#CHECK: lpq	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x8f]
#CHECK: lpq	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x8f]
#CHECK: lpq	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x8f]
#CHECK: lpq	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x8f]
#CHECK: lpq	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x8f]
#CHECK: lpq	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x8f]
#CHECK: lpq	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x8f]
#CHECK: lpq	%r14, 0                 # encoding: [0xe3,0xe0,0x00,0x00,0x00,0x8f]

	lpq	%r0, -524288
	lpq	%r0, -1
	lpq	%r0, 0
	lpq	%r0, 1
	lpq	%r0, 524287
	lpq	%r0, 0(%r1)
	lpq	%r0, 0(%r15)
	lpq	%r0, 524287(%r1,%r15)
	lpq	%r0, 524287(%r15,%r1)
	lpq	%r14, 0

#CHECK: lpr	%r0, %r0                # encoding: [0x10,0x00]
#CHECK: lpr	%r0, %r15               # encoding: [0x10,0x0f]
#CHECK: lpr	%r15, %r0               # encoding: [0x10,0xf0]
#CHECK: lpr	%r7, %r8                # encoding: [0x10,0x78]

	lpr	%r0,%r0
	lpr	%r0,%r15
	lpr	%r15,%r0
	lpr	%r7,%r8

#CHECK: lpsw	0                       # encoding: [0x82,0x00,0x00,0x00]
#CHECK: lpsw	0(%r1)                  # encoding: [0x82,0x00,0x10,0x00]
#CHECK: lpsw	0(%r15)                 # encoding: [0x82,0x00,0xf0,0x00]
#CHECK: lpsw	4095                    # encoding: [0x82,0x00,0x0f,0xff]
#CHECK: lpsw	4095(%r1)               # encoding: [0x82,0x00,0x1f,0xff]
#CHECK: lpsw	4095(%r15)              # encoding: [0x82,0x00,0xff,0xff]

	lpsw	0
	lpsw	0(%r1)
	lpsw	0(%r15)
	lpsw	4095
	lpsw	4095(%r1)
	lpsw	4095(%r15)

#CHECK: lpswe	0                       # encoding: [0xb2,0xb2,0x00,0x00]
#CHECK: lpswe	0(%r1)                  # encoding: [0xb2,0xb2,0x10,0x00]
#CHECK: lpswe	0(%r15)                 # encoding: [0xb2,0xb2,0xf0,0x00]
#CHECK: lpswe	4095                    # encoding: [0xb2,0xb2,0x0f,0xff]
#CHECK: lpswe	4095(%r1)               # encoding: [0xb2,0xb2,0x1f,0xff]
#CHECK: lpswe	4095(%r15)              # encoding: [0xb2,0xb2,0xff,0xff]

	lpswe	0
	lpswe	0(%r1)
	lpswe	0(%r15)
	lpswe	4095
	lpswe	4095(%r1)
	lpswe	4095(%r15)

#CHECK: lptea	%r0, %r0, %r0, 0        # encoding: [0xb9,0xaa,0x00,0x00]
#CHECK: lptea	%r0, %r0, %r0, 15       # encoding: [0xb9,0xaa,0x0f,0x00]
#CHECK: lptea	%r0, %r0, %r15, 0       # encoding: [0xb9,0xaa,0x00,0x0f]
#CHECK: lptea	%r0, %r15, %r0, 0       # encoding: [0xb9,0xaa,0xf0,0x00]
#CHECK: lptea	%r4, %r5, %r6, 7        # encoding: [0xb9,0xaa,0x57,0x46]
#CHECK: lptea	%r15, %r0, %r0, 0       # encoding: [0xb9,0xaa,0x00,0xf0]

	lptea	%r0, %r0, %r0, 0
	lptea	%r0, %r0, %r0, 15
	lptea	%r0, %r0, %r15, 0
	lptea	%r0, %r15, %r0, 0
	lptea	%r4, %r5, %r6, 7
	lptea	%r15, %r0, %r0, 0

#CHECK: lpxbr	%f0, %f8                # encoding: [0xb3,0x40,0x00,0x08]
#CHECK: lpxbr	%f0, %f13               # encoding: [0xb3,0x40,0x00,0x0d]
#CHECK: lpxbr	%f13, %f0               # encoding: [0xb3,0x40,0x00,0xd0]
#CHECK: lpxbr	%f13, %f9               # encoding: [0xb3,0x40,0x00,0xd9]

	lpxbr	%f0,%f8
	lpxbr	%f0,%f13
	lpxbr	%f13,%f0
	lpxbr	%f13,%f9

#CHECK: lpxr	%f0, %f8                # encoding: [0xb3,0x60,0x00,0x08]
#CHECK: lpxr	%f0, %f13               # encoding: [0xb3,0x60,0x00,0x0d]
#CHECK: lpxr	%f13, %f0               # encoding: [0xb3,0x60,0x00,0xd0]
#CHECK: lpxr	%f13, %f9               # encoding: [0xb3,0x60,0x00,0xd9]

	lpxr	%f0,%f8
	lpxr	%f0,%f13
	lpxr	%f13,%f0
	lpxr	%f13,%f9

#CHECK: lr	%r0, %r9                # encoding: [0x18,0x09]
#CHECK: lr	%r0, %r15               # encoding: [0x18,0x0f]
#CHECK: lr	%r15, %r0               # encoding: [0x18,0xf0]
#CHECK: lr	%r15, %r9               # encoding: [0x18,0xf9]

	lr	%r0,%r9
	lr	%r0,%r15
	lr	%r15,%r0
	lr	%r15,%r9

#CHECK: lra	%r0, 0                  # encoding: [0xb1,0x00,0x00,0x00]
#CHECK: lra	%r0, 4095               # encoding: [0xb1,0x00,0x0f,0xff]
#CHECK: lra	%r0, 0(%r1)             # encoding: [0xb1,0x00,0x10,0x00]
#CHECK: lra	%r0, 0(%r15)            # encoding: [0xb1,0x00,0xf0,0x00]
#CHECK: lra	%r0, 4095(%r1,%r15)     # encoding: [0xb1,0x01,0xff,0xff]
#CHECK: lra	%r0, 4095(%r15,%r1)     # encoding: [0xb1,0x0f,0x1f,0xff]
#CHECK: lra	%r15, 0                 # encoding: [0xb1,0xf0,0x00,0x00]

	lra	%r0, 0
	lra	%r0, 4095
	lra	%r0, 0(%r1)
	lra	%r0, 0(%r15)
	lra	%r0, 4095(%r1,%r15)
	lra	%r0, 4095(%r15,%r1)
	lra	%r15, 0

#CHECK: lrag	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x03]
#CHECK: lrag	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x03]
#CHECK: lrag	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x03]
#CHECK: lrag	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x03]
#CHECK: lrag	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x03]
#CHECK: lrag	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x03]
#CHECK: lrag	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x03]
#CHECK: lrag	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x03]
#CHECK: lrag	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x03]
#CHECK: lrag	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x03]

	lrag	%r0, -524288
	lrag	%r0, -1
	lrag	%r0, 0
	lrag	%r0, 1
	lrag	%r0, 524287
	lrag	%r0, 0(%r1)
	lrag	%r0, 0(%r15)
	lrag	%r0, 524287(%r1,%r15)
	lrag	%r0, 524287(%r15,%r1)
	lrag	%r15, 0

#CHECK: lray	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x13]
#CHECK: lray	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x13]
#CHECK: lray	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x13]
#CHECK: lray	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x13]
#CHECK: lray	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x13]
#CHECK: lray	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x13]
#CHECK: lray	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x13]
#CHECK: lray	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x13]
#CHECK: lray	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x13]
#CHECK: lray	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x13]

	lray	%r0, -524288
	lray	%r0, -1
	lray	%r0, 0
	lray	%r0, 1
	lray	%r0, 524287
	lray	%r0, 0(%r1)
	lray	%r0, 0(%r15)
	lray	%r0, 524287(%r1,%r15)
	lray	%r0, 524287(%r15,%r1)
	lray	%r15, 0

#CHECK: lrdr	%f0, %f0                # encoding: [0x25,0x00]
#CHECK: lrdr	%f0, %f13               # encoding: [0x25,0x0d]
#CHECK: lrdr	%f7, %f8                # encoding: [0x25,0x78]
#CHECK: lrdr	%f15, %f0               # encoding: [0x25,0xf0]
#CHECK: lrdr	%f15, %f13              # encoding: [0x25,0xfd]

	lrdr	%f0, %f0
	lrdr	%f0, %f13
	lrdr	%f7, %f8
	lrdr	%f15, %f0
	lrdr	%f15, %f13

#CHECK: lrer	%f0, %f0                # encoding: [0x35,0x00]
#CHECK: lrer	%f0, %f15               # encoding: [0x35,0x0f]
#CHECK: lrer	%f7, %f8                # encoding: [0x35,0x78]
#CHECK: lrer	%f15, %f0               # encoding: [0x35,0xf0]
#CHECK: lrer	%f15, %f15              # encoding: [0x35,0xff]

	lrer	%f0, %f0
	lrer	%f0, %f15
	lrer	%f7, %f8
	lrer	%f15, %f0
	lrer	%f15, %f15

#CHECK: lrl	%r0, .[[LAB:L.*]]-4294967296 # encoding: [0xc4,0x0d,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-4294967296)+2, kind: FK_390_PC32DBL
	lrl	%r0, -0x100000000
#CHECK: lrl	%r0, .[[LAB:L.*]]-2	# encoding: [0xc4,0x0d,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC32DBL
	lrl	%r0, -2
#CHECK: lrl	%r0, .[[LAB:L.*]]	# encoding: [0xc4,0x0d,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC32DBL
	lrl	%r0, 0
#CHECK: lrl	%r0, .[[LAB:L.*]]+4294967294 # encoding: [0xc4,0x0d,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+4294967294)+2, kind: FK_390_PC32DBL
	lrl	%r0, 0xfffffffe

#CHECK: lrl	%r0, foo                # encoding: [0xc4,0x0d,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: lrl	%r15, foo               # encoding: [0xc4,0xfd,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL

	lrl	%r0,foo
	lrl	%r15,foo

#CHECK: lrl	%r3, bar+100            # encoding: [0xc4,0x3d,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: lrl	%r4, bar+100            # encoding: [0xc4,0x4d,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL

	lrl	%r3,bar+100
	lrl	%r4,bar+100

#CHECK: lrl	%r7, frob@PLT           # encoding: [0xc4,0x7d,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL
#CHECK: lrl	%r8, frob@PLT           # encoding: [0xc4,0x8d,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL

	lrl	%r7,frob@PLT
	lrl	%r8,frob@PLT

#CHECK: lrv	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x1e]
#CHECK: lrv	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x1e]
#CHECK: lrv	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x1e]
#CHECK: lrv	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x1e]
#CHECK: lrv	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x1e]
#CHECK: lrv	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x1e]
#CHECK: lrv	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x1e]
#CHECK: lrv	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x1e]
#CHECK: lrv	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x1e]
#CHECK: lrv	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x1e]

	lrv	%r0,-524288
	lrv	%r0,-1
	lrv	%r0,0
	lrv	%r0,1
	lrv	%r0,524287
	lrv	%r0,0(%r1)
	lrv	%r0,0(%r15)
	lrv	%r0,524287(%r1,%r15)
	lrv	%r0,524287(%r15,%r1)
	lrv	%r15,0

#CHECK: lrvg	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x0f]
#CHECK: lrvg	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x0f]
#CHECK: lrvg	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x0f]
#CHECK: lrvg	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x0f]
#CHECK: lrvg	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x0f]
#CHECK: lrvg	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x0f]
#CHECK: lrvg	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x0f]
#CHECK: lrvg	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x0f]
#CHECK: lrvg	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x0f]
#CHECK: lrvg	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x0f]

	lrvg	%r0,-524288
	lrvg	%r0,-1
	lrvg	%r0,0
	lrvg	%r0,1
	lrvg	%r0,524287
	lrvg	%r0,0(%r1)
	lrvg	%r0,0(%r15)
	lrvg	%r0,524287(%r1,%r15)
	lrvg	%r0,524287(%r15,%r1)
	lrvg	%r15,0

#CHECK: lrvgr	%r0, %r0                # encoding: [0xb9,0x0f,0x00,0x00]
#CHECK: lrvgr	%r0, %r15               # encoding: [0xb9,0x0f,0x00,0x0f]
#CHECK: lrvgr	%r15, %r0               # encoding: [0xb9,0x0f,0x00,0xf0]
#CHECK: lrvgr	%r7, %r8                # encoding: [0xb9,0x0f,0x00,0x78]
#CHECK: lrvgr	%r15, %r15              # encoding: [0xb9,0x0f,0x00,0xff]

	lrvgr	%r0,%r0
	lrvgr	%r0,%r15
	lrvgr	%r15,%r0
	lrvgr	%r7,%r8
	lrvgr	%r15,%r15

#CHECK: lrvh	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x1f]
#CHECK: lrvh	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x1f]
#CHECK: lrvh	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x1f]
#CHECK: lrvh	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x1f]
#CHECK: lrvh	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x1f]
#CHECK: lrvh	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x1f]
#CHECK: lrvh	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x1f]
#CHECK: lrvh	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x1f]
#CHECK: lrvh	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x1f]
#CHECK: lrvh	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x1f]

	lrvh	%r0,-524288
	lrvh	%r0,-1
	lrvh	%r0,0
	lrvh	%r0,1
	lrvh	%r0,524287
	lrvh	%r0,0(%r1)
	lrvh	%r0,0(%r15)
	lrvh	%r0,524287(%r1,%r15)
	lrvh	%r0,524287(%r15,%r1)
	lrvh	%r15,0

#CHECK: lrvr	%r0, %r0                # encoding: [0xb9,0x1f,0x00,0x00]
#CHECK: lrvr	%r0, %r15               # encoding: [0xb9,0x1f,0x00,0x0f]
#CHECK: lrvr	%r15, %r0               # encoding: [0xb9,0x1f,0x00,0xf0]
#CHECK: lrvr	%r7, %r8                # encoding: [0xb9,0x1f,0x00,0x78]
#CHECK: lrvr	%r15, %r15              # encoding: [0xb9,0x1f,0x00,0xff]

	lrvr	%r0,%r0
	lrvr	%r0,%r15
	lrvr	%r15,%r0
	lrvr	%r7,%r8
	lrvr	%r15,%r15

#CHECK: lsctl	0                       # encoding: [0xb2,0x87,0x00,0x00]
#CHECK: lsctl	0(%r1)                  # encoding: [0xb2,0x87,0x10,0x00]
#CHECK: lsctl	0(%r15)                 # encoding: [0xb2,0x87,0xf0,0x00]
#CHECK: lsctl	4095                    # encoding: [0xb2,0x87,0x0f,0xff]
#CHECK: lsctl	4095(%r1)               # encoding: [0xb2,0x87,0x1f,0xff]
#CHECK: lsctl	4095(%r15)              # encoding: [0xb2,0x87,0xff,0xff]

	lsctl	0
	lsctl	0(%r1)
	lsctl	0(%r15)
	lsctl	4095
	lsctl	4095(%r1)
	lsctl	4095(%r15)

#CHECK: lt	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x12]
#CHECK: lt	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x12]
#CHECK: lt	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x12]
#CHECK: lt	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x12]
#CHECK: lt	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x12]
#CHECK: lt	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x12]
#CHECK: lt	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x12]
#CHECK: lt	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x12]
#CHECK: lt	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x12]
#CHECK: lt	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x12]

	lt	%r0, -524288
	lt	%r0, -1
	lt	%r0, 0
	lt	%r0, 1
	lt	%r0, 524287
	lt	%r0, 0(%r1)
	lt	%r0, 0(%r15)
	lt	%r0, 524287(%r1,%r15)
	lt	%r0, 524287(%r15,%r1)
	lt	%r15, 0

#CHECK: ltdbr	%f0, %f9                # encoding: [0xb3,0x12,0x00,0x09]
#CHECK: ltdbr	%f0, %f15               # encoding: [0xb3,0x12,0x00,0x0f]
#CHECK: ltdbr	%f15, %f0               # encoding: [0xb3,0x12,0x00,0xf0]
#CHECK: ltdbr	%f15, %f9               # encoding: [0xb3,0x12,0x00,0xf9]

	ltdbr	%f0,%f9
	ltdbr	%f0,%f15
	ltdbr	%f15,%f0
	ltdbr	%f15,%f9

#CHECK: ltdr	%f0, %f9                # encoding: [0x22,0x09]
#CHECK: ltdr	%f0, %f15               # encoding: [0x22,0x0f]
#CHECK: ltdr	%f15, %f0               # encoding: [0x22,0xf0]
#CHECK: ltdr	%f15, %f9               # encoding: [0x22,0xf9]

	ltdr	%f0,%f9
	ltdr	%f0,%f15
	ltdr	%f15,%f0
	ltdr	%f15,%f9

#CHECK: ltdtr	%f0, %f9                # encoding: [0xb3,0xd6,0x00,0x09]
#CHECK: ltdtr	%f0, %f15               # encoding: [0xb3,0xd6,0x00,0x0f]
#CHECK: ltdtr	%f15, %f0               # encoding: [0xb3,0xd6,0x00,0xf0]
#CHECK: ltdtr	%f15, %f9               # encoding: [0xb3,0xd6,0x00,0xf9]

	ltdtr	%f0,%f9
	ltdtr	%f0,%f15
	ltdtr	%f15,%f0
	ltdtr	%f15,%f9

#CHECK: ltebr	%f0, %f9                # encoding: [0xb3,0x02,0x00,0x09]
#CHECK: ltebr	%f0, %f15               # encoding: [0xb3,0x02,0x00,0x0f]
#CHECK: ltebr	%f15, %f0               # encoding: [0xb3,0x02,0x00,0xf0]
#CHECK: ltebr	%f15, %f9               # encoding: [0xb3,0x02,0x00,0xf9]

	ltebr	%f0,%f9
	ltebr	%f0,%f15
	ltebr	%f15,%f0
	ltebr	%f15,%f9

#CHECK: lter	%f0, %f9                # encoding: [0x32,0x09]
#CHECK: lter	%f0, %f15               # encoding: [0x32,0x0f]
#CHECK: lter	%f15, %f0               # encoding: [0x32,0xf0]
#CHECK: lter	%f15, %f9               # encoding: [0x32,0xf9]

	lter	%f0,%f9
	lter	%f0,%f15
	lter	%f15,%f0
	lter	%f15,%f9

#CHECK: ltg	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x02]
#CHECK: ltg	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x02]
#CHECK: ltg	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x02]
#CHECK: ltg	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x02]
#CHECK: ltg	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x02]
#CHECK: ltg	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x02]
#CHECK: ltg	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x02]
#CHECK: ltg	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x02]
#CHECK: ltg	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x02]
#CHECK: ltg	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x02]

	ltg	%r0, -524288
	ltg	%r0, -1
	ltg	%r0, 0
	ltg	%r0, 1
	ltg	%r0, 524287
	ltg	%r0, 0(%r1)
	ltg	%r0, 0(%r15)
	ltg	%r0, 524287(%r1,%r15)
	ltg	%r0, 524287(%r15,%r1)
	ltg	%r15, 0

#CHECK: ltgf	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x32]
#CHECK: ltgf	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x32]
#CHECK: ltgf	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x32]
#CHECK: ltgf	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x32]
#CHECK: ltgf	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x32]
#CHECK: ltgf	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x32]
#CHECK: ltgf	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x32]
#CHECK: ltgf	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x32]
#CHECK: ltgf	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x32]
#CHECK: ltgf	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x32]

	ltgf	%r0, -524288
	ltgf	%r0, -1
	ltgf	%r0, 0
	ltgf	%r0, 1
	ltgf	%r0, 524287
	ltgf	%r0, 0(%r1)
	ltgf	%r0, 0(%r15)
	ltgf	%r0, 524287(%r1,%r15)
	ltgf	%r0, 524287(%r15,%r1)
	ltgf	%r15, 0

#CHECK: ltgfr	%r0, %r9                # encoding: [0xb9,0x12,0x00,0x09]
#CHECK: ltgfr	%r0, %r15               # encoding: [0xb9,0x12,0x00,0x0f]
#CHECK: ltgfr	%r15, %r0               # encoding: [0xb9,0x12,0x00,0xf0]
#CHECK: ltgfr	%r15, %r9               # encoding: [0xb9,0x12,0x00,0xf9]

	ltgfr	%r0,%r9
	ltgfr	%r0,%r15
	ltgfr	%r15,%r0
	ltgfr	%r15,%r9

#CHECK: ltgr	%r0, %r9                # encoding: [0xb9,0x02,0x00,0x09]
#CHECK: ltgr	%r0, %r15               # encoding: [0xb9,0x02,0x00,0x0f]
#CHECK: ltgr	%r15, %r0               # encoding: [0xb9,0x02,0x00,0xf0]
#CHECK: ltgr	%r15, %r9               # encoding: [0xb9,0x02,0x00,0xf9]

	ltgr	%r0,%r9
	ltgr	%r0,%r15
	ltgr	%r15,%r0
	ltgr	%r15,%r9

#CHECK: ltr	%r0, %r9                # encoding: [0x12,0x09]
#CHECK: ltr	%r0, %r15               # encoding: [0x12,0x0f]
#CHECK: ltr	%r15, %r0               # encoding: [0x12,0xf0]
#CHECK: ltr	%r15, %r9               # encoding: [0x12,0xf9]

	ltr	%r0,%r9
	ltr	%r0,%r15
	ltr	%r15,%r0
	ltr	%r15,%r9

#CHECK: ltxbr	%f0, %f9                # encoding: [0xb3,0x42,0x00,0x09]
#CHECK: ltxbr	%f0, %f13               # encoding: [0xb3,0x42,0x00,0x0d]
#CHECK: ltxbr	%f13, %f0               # encoding: [0xb3,0x42,0x00,0xd0]
#CHECK: ltxbr	%f13, %f9               # encoding: [0xb3,0x42,0x00,0xd9]

	ltxbr	%f0,%f9
	ltxbr	%f0,%f13
	ltxbr	%f13,%f0
	ltxbr	%f13,%f9

#CHECK: ltxr	%f0, %f9                # encoding: [0xb3,0x62,0x00,0x09]
#CHECK: ltxr	%f0, %f13               # encoding: [0xb3,0x62,0x00,0x0d]
#CHECK: ltxr	%f13, %f0               # encoding: [0xb3,0x62,0x00,0xd0]
#CHECK: ltxr	%f13, %f9               # encoding: [0xb3,0x62,0x00,0xd9]

	ltxr	%f0,%f9
	ltxr	%f0,%f13
	ltxr	%f13,%f0
	ltxr	%f13,%f9

#CHECK: ltxtr	%f0, %f9                # encoding: [0xb3,0xde,0x00,0x09]
#CHECK: ltxtr	%f0, %f13               # encoding: [0xb3,0xde,0x00,0x0d]
#CHECK: ltxtr	%f13, %f0               # encoding: [0xb3,0xde,0x00,0xd0]
#CHECK: ltxtr	%f13, %f9               # encoding: [0xb3,0xde,0x00,0xd9]

	ltxtr	%f0,%f9
	ltxtr	%f0,%f13
	ltxtr	%f13,%f0
	ltxtr	%f13,%f9

#CHECK: lura	%r0, %r0                # encoding: [0xb2,0x4b,0x00,0x00]
#CHECK: lura	%r0, %r15               # encoding: [0xb2,0x4b,0x00,0x0f]
#CHECK: lura	%r15, %r0               # encoding: [0xb2,0x4b,0x00,0xf0]
#CHECK: lura	%r7, %r8                # encoding: [0xb2,0x4b,0x00,0x78]

	lura	%r0,%r0
	lura	%r0,%r15
	lura	%r15,%r0
	lura	%r7,%r8

#CHECK: lurag	%r0, %r0                # encoding: [0xb9,0x05,0x00,0x00]
#CHECK: lurag	%r0, %r15               # encoding: [0xb9,0x05,0x00,0x0f]
#CHECK: lurag	%r15, %r0               # encoding: [0xb9,0x05,0x00,0xf0]
#CHECK: lurag	%r7, %r8                # encoding: [0xb9,0x05,0x00,0x78]

	lurag	%r0,%r0
	lurag	%r0,%r15
	lurag	%r15,%r0
	lurag	%r7,%r8

#CHECK: lxd	%f0, 4095               # encoding: [0xed,0x00,0x0f,0xff,0x00,0x25]
#CHECK: lxd	%f0, 0(%r1)             # encoding: [0xed,0x00,0x10,0x00,0x00,0x25]
#CHECK: lxd	%f0, 0(%r15)            # encoding: [0xed,0x00,0xf0,0x00,0x00,0x25]
#CHECK: lxd	%f0, 4095(%r1,%r15)     # encoding: [0xed,0x01,0xff,0xff,0x00,0x25]
#CHECK: lxd	%f0, 4095(%r15,%r1)     # encoding: [0xed,0x0f,0x1f,0xff,0x00,0x25]
#CHECK: lxd	%f13, 0                 # encoding: [0xed,0xd0,0x00,0x00,0x00,0x25]

	lxd	%f0, 0
	lxd	%f0, 4095
	lxd	%f0, 0(%r1)
	lxd	%f0, 0(%r15)
	lxd	%f0, 4095(%r1,%r15)
	lxd	%f0, 4095(%r15,%r1)
	lxd	%f13, 0

#CHECK: lxdb	%f0, 4095               # encoding: [0xed,0x00,0x0f,0xff,0x00,0x05]
#CHECK: lxdb	%f0, 0(%r1)             # encoding: [0xed,0x00,0x10,0x00,0x00,0x05]
#CHECK: lxdb	%f0, 0(%r15)            # encoding: [0xed,0x00,0xf0,0x00,0x00,0x05]
#CHECK: lxdb	%f0, 4095(%r1,%r15)     # encoding: [0xed,0x01,0xff,0xff,0x00,0x05]
#CHECK: lxdb	%f0, 4095(%r15,%r1)     # encoding: [0xed,0x0f,0x1f,0xff,0x00,0x05]
#CHECK: lxdb	%f13, 0                 # encoding: [0xed,0xd0,0x00,0x00,0x00,0x05]

	lxdb	%f0, 0
	lxdb	%f0, 4095
	lxdb	%f0, 0(%r1)
	lxdb	%f0, 0(%r15)
	lxdb	%f0, 4095(%r1,%r15)
	lxdb	%f0, 4095(%r15,%r1)
	lxdb	%f13, 0

#CHECK: lxdbr	%f0, %f8                # encoding: [0xb3,0x05,0x00,0x08]
#CHECK: lxdbr	%f0, %f13               # encoding: [0xb3,0x05,0x00,0x0d]
#CHECK: lxdbr	%f13, %f0               # encoding: [0xb3,0x05,0x00,0xd0]
#CHECK: lxdbr	%f13, %f15              # encoding: [0xb3,0x05,0x00,0xdf]

	lxdbr	%f0,%f8
	lxdbr	%f0,%f13
	lxdbr	%f13,%f0
	lxdbr	%f13,%f15

#CHECK: lxdr	%f0, %f8                # encoding: [0xb3,0x25,0x00,0x08]
#CHECK: lxdr	%f0, %f13               # encoding: [0xb3,0x25,0x00,0x0d]
#CHECK: lxdr	%f13, %f0               # encoding: [0xb3,0x25,0x00,0xd0]
#CHECK: lxdr	%f13, %f15              # encoding: [0xb3,0x25,0x00,0xdf]

	lxdr	%f0,%f8
	lxdr	%f0,%f13
	lxdr	%f13,%f0
	lxdr	%f13,%f15

#CHECK: lxdtr	%f0, %f0, 15            # encoding: [0xb3,0xdc,0x0f,0x00]
#CHECK: lxdtr	%f0, %f15, 0            # encoding: [0xb3,0xdc,0x00,0x0f]
#CHECK: lxdtr	%f5, %f8, 9             # encoding: [0xb3,0xdc,0x09,0x58]
#CHECK: lxdtr	%f13, %f0, 0            # encoding: [0xb3,0xdc,0x00,0xd0]

	lxdtr	%f0, %f0, 15
	lxdtr	%f0, %f15, 0
	lxdtr	%f5, %f8, 9
	lxdtr	%f13, %f0, 0

#CHECK: lxe	%f0, 4095               # encoding: [0xed,0x00,0x0f,0xff,0x00,0x26]
#CHECK: lxe	%f0, 0(%r1)             # encoding: [0xed,0x00,0x10,0x00,0x00,0x26]
#CHECK: lxe	%f0, 0(%r15)            # encoding: [0xed,0x00,0xf0,0x00,0x00,0x26]
#CHECK: lxe	%f0, 4095(%r1,%r15)     # encoding: [0xed,0x01,0xff,0xff,0x00,0x26]
#CHECK: lxe	%f0, 4095(%r15,%r1)     # encoding: [0xed,0x0f,0x1f,0xff,0x00,0x26]
#CHECK: lxe	%f13, 0                 # encoding: [0xed,0xd0,0x00,0x00,0x00,0x26]

	lxe	%f0, 0
	lxe	%f0, 4095
	lxe	%f0, 0(%r1)
	lxe	%f0, 0(%r15)
	lxe	%f0, 4095(%r1,%r15)
	lxe	%f0, 4095(%r15,%r1)
	lxe	%f13, 0

#CHECK: lxeb	%f0, 4095               # encoding: [0xed,0x00,0x0f,0xff,0x00,0x06]
#CHECK: lxeb	%f0, 0(%r1)             # encoding: [0xed,0x00,0x10,0x00,0x00,0x06]
#CHECK: lxeb	%f0, 0(%r15)            # encoding: [0xed,0x00,0xf0,0x00,0x00,0x06]
#CHECK: lxeb	%f0, 4095(%r1,%r15)     # encoding: [0xed,0x01,0xff,0xff,0x00,0x06]
#CHECK: lxeb	%f0, 4095(%r15,%r1)     # encoding: [0xed,0x0f,0x1f,0xff,0x00,0x06]
#CHECK: lxeb	%f13, 0                 # encoding: [0xed,0xd0,0x00,0x00,0x00,0x06]

	lxeb	%f0, 0
	lxeb	%f0, 4095
	lxeb	%f0, 0(%r1)
	lxeb	%f0, 0(%r15)
	lxeb	%f0, 4095(%r1,%r15)
	lxeb	%f0, 4095(%r15,%r1)
	lxeb	%f13, 0

#CHECK: lxebr	%f0, %f8                # encoding: [0xb3,0x06,0x00,0x08]
#CHECK: lxebr	%f0, %f13               # encoding: [0xb3,0x06,0x00,0x0d]
#CHECK: lxebr	%f13, %f0               # encoding: [0xb3,0x06,0x00,0xd0]
#CHECK: lxebr	%f13, %f15              # encoding: [0xb3,0x06,0x00,0xdf]

	lxebr	%f0,%f8
	lxebr	%f0,%f13
	lxebr	%f13,%f0
	lxebr	%f13,%f15

#CHECK: lxer	%f0, %f8                # encoding: [0xb3,0x26,0x00,0x08]
#CHECK: lxer	%f0, %f13               # encoding: [0xb3,0x26,0x00,0x0d]
#CHECK: lxer	%f13, %f0               # encoding: [0xb3,0x26,0x00,0xd0]
#CHECK: lxer	%f13, %f15              # encoding: [0xb3,0x26,0x00,0xdf]

	lxer	%f0,%f8
	lxer	%f0,%f13
	lxer	%f13,%f0
	lxer	%f13,%f15

#CHECK: lxr	%f0, %f8                # encoding: [0xb3,0x65,0x00,0x08]
#CHECK: lxr	%f0, %f13               # encoding: [0xb3,0x65,0x00,0x0d]
#CHECK: lxr	%f13, %f0               # encoding: [0xb3,0x65,0x00,0xd0]
#CHECK: lxr	%f13, %f9               # encoding: [0xb3,0x65,0x00,0xd9]

	lxr	%f0,%f8
	lxr	%f0,%f13
	lxr	%f13,%f0
	lxr	%f13,%f9

#CHECK: ly	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x58]
#CHECK: ly	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x58]
#CHECK: ly	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x58]
#CHECK: ly	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x58]
#CHECK: ly	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x58]
#CHECK: ly	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x58]
#CHECK: ly	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x58]
#CHECK: ly	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x58]
#CHECK: ly	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x58]
#CHECK: ly	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x58]

	ly	%r0, -524288
	ly	%r0, -1
	ly	%r0, 0
	ly	%r0, 1
	ly	%r0, 524287
	ly	%r0, 0(%r1)
	ly	%r0, 0(%r15)
	ly	%r0, 524287(%r1,%r15)
	ly	%r0, 524287(%r15,%r1)
	ly	%r15, 0

#CHECK: lzdr	%f0                     # encoding: [0xb3,0x75,0x00,0x00]
#CHECK: lzdr	%f7                     # encoding: [0xb3,0x75,0x00,0x70]
#CHECK: lzdr	%f15                    # encoding: [0xb3,0x75,0x00,0xf0]

	lzdr	%f0
	lzdr	%f7
	lzdr	%f15

#CHECK: lzer	%f0                     # encoding: [0xb3,0x74,0x00,0x00]
#CHECK: lzer	%f7                     # encoding: [0xb3,0x74,0x00,0x70]
#CHECK: lzer	%f15                    # encoding: [0xb3,0x74,0x00,0xf0]

	lzer	%f0
	lzer	%f7
	lzer	%f15

#CHECK: lzxr	%f0                     # encoding: [0xb3,0x76,0x00,0x00]
#CHECK: lzxr	%f8                     # encoding: [0xb3,0x76,0x00,0x80]
#CHECK: lzxr	%f13                    # encoding: [0xb3,0x76,0x00,0xd0]

	lzxr	%f0
	lzxr	%f8
	lzxr	%f13

#CHECK: m	%r0, 0                  # encoding: [0x5c,0x00,0x00,0x00]
#CHECK: m	%r0, 4095               # encoding: [0x5c,0x00,0x0f,0xff]
#CHECK: m	%r0, 0(%r1)             # encoding: [0x5c,0x00,0x10,0x00]
#CHECK: m	%r0, 0(%r15)            # encoding: [0x5c,0x00,0xf0,0x00]
#CHECK: m	%r0, 4095(%r1,%r15)     # encoding: [0x5c,0x01,0xff,0xff]
#CHECK: m	%r0, 4095(%r15,%r1)     # encoding: [0x5c,0x0f,0x1f,0xff]
#CHECK: m	%r14, 0                 # encoding: [0x5c,0xe0,0x00,0x00]

	m	%r0, 0
	m	%r0, 4095
	m	%r0, 0(%r1)
	m	%r0, 0(%r15)
	m	%r0, 4095(%r1,%r15)
	m	%r0, 4095(%r15,%r1)
	m	%r14, 0

#CHECK: mad	%f0, %f0, 0             # encoding: [0xed,0x00,0x00,0x00,0x00,0x3e]
#CHECK: mad	%f0, %f0, 4095          # encoding: [0xed,0x00,0x0f,0xff,0x00,0x3e]
#CHECK: mad	%f0, %f0, 0(%r1)        # encoding: [0xed,0x00,0x10,0x00,0x00,0x3e]
#CHECK: mad	%f0, %f0, 0(%r15)       # encoding: [0xed,0x00,0xf0,0x00,0x00,0x3e]
#CHECK: mad	%f0, %f0, 4095(%r1,%r15) # encoding: [0xed,0x01,0xff,0xff,0x00,0x3e]
#CHECK: mad	%f0, %f0, 4095(%r15,%r1) # encoding: [0xed,0x0f,0x1f,0xff,0x00,0x3e]
#CHECK: mad	%f0, %f15, 0            # encoding: [0xed,0xf0,0x00,0x00,0x00,0x3e]
#CHECK: mad	%f15, %f0, 0            # encoding: [0xed,0x00,0x00,0x00,0xf0,0x3e]
#CHECK: mad	%f15, %f15, 0           # encoding: [0xed,0xf0,0x00,0x00,0xf0,0x3e]

	mad	%f0, %f0, 0
	mad	%f0, %f0, 4095
	mad	%f0, %f0, 0(%r1)
	mad	%f0, %f0, 0(%r15)
	mad	%f0, %f0, 4095(%r1,%r15)
	mad	%f0, %f0, 4095(%r15,%r1)
	mad	%f0, %f15, 0
	mad	%f15, %f0, 0
	mad	%f15, %f15, 0

#CHECK: madb	%f0, %f0, 0             # encoding: [0xed,0x00,0x00,0x00,0x00,0x1e]
#CHECK: madb	%f0, %f0, 4095          # encoding: [0xed,0x00,0x0f,0xff,0x00,0x1e]
#CHECK: madb	%f0, %f0, 0(%r1)        # encoding: [0xed,0x00,0x10,0x00,0x00,0x1e]
#CHECK: madb	%f0, %f0, 0(%r15)       # encoding: [0xed,0x00,0xf0,0x00,0x00,0x1e]
#CHECK: madb	%f0, %f0, 4095(%r1,%r15) # encoding: [0xed,0x01,0xff,0xff,0x00,0x1e]
#CHECK: madb	%f0, %f0, 4095(%r15,%r1) # encoding: [0xed,0x0f,0x1f,0xff,0x00,0x1e]
#CHECK: madb	%f0, %f15, 0            # encoding: [0xed,0xf0,0x00,0x00,0x00,0x1e]
#CHECK: madb	%f15, %f0, 0            # encoding: [0xed,0x00,0x00,0x00,0xf0,0x1e]
#CHECK: madb	%f15, %f15, 0           # encoding: [0xed,0xf0,0x00,0x00,0xf0,0x1e]

	madb	%f0, %f0, 0
	madb	%f0, %f0, 4095
	madb	%f0, %f0, 0(%r1)
	madb	%f0, %f0, 0(%r15)
	madb	%f0, %f0, 4095(%r1,%r15)
	madb	%f0, %f0, 4095(%r15,%r1)
	madb	%f0, %f15, 0
	madb	%f15, %f0, 0
	madb	%f15, %f15, 0

#CHECK: madbr	%f0, %f0, %f0           # encoding: [0xb3,0x1e,0x00,0x00]
#CHECK: madbr	%f0, %f0, %f15          # encoding: [0xb3,0x1e,0x00,0x0f]
#CHECK: madbr	%f0, %f15, %f0          # encoding: [0xb3,0x1e,0x00,0xf0]
#CHECK: madbr	%f15, %f0, %f0          # encoding: [0xb3,0x1e,0xf0,0x00]
#CHECK: madbr	%f7, %f8, %f9           # encoding: [0xb3,0x1e,0x70,0x89]
#CHECK: madbr	%f15, %f15, %f15        # encoding: [0xb3,0x1e,0xf0,0xff]

	madbr	%f0, %f0, %f0
	madbr	%f0, %f0, %f15
	madbr	%f0, %f15, %f0
	madbr	%f15, %f0, %f0
	madbr	%f7, %f8, %f9
	madbr	%f15, %f15, %f15

#CHECK: madr	%f0, %f0, %f0           # encoding: [0xb3,0x3e,0x00,0x00]
#CHECK: madr	%f0, %f0, %f15          # encoding: [0xb3,0x3e,0x00,0x0f]
#CHECK: madr	%f0, %f15, %f0          # encoding: [0xb3,0x3e,0x00,0xf0]
#CHECK: madr	%f15, %f0, %f0          # encoding: [0xb3,0x3e,0xf0,0x00]
#CHECK: madr	%f7, %f8, %f9           # encoding: [0xb3,0x3e,0x70,0x89]
#CHECK: madr	%f15, %f15, %f15        # encoding: [0xb3,0x3e,0xf0,0xff]

	madr	%f0, %f0, %f0
	madr	%f0, %f0, %f15
	madr	%f0, %f15, %f0
	madr	%f15, %f0, %f0
	madr	%f7, %f8, %f9
	madr	%f15, %f15, %f15

#CHECK: mae	%f0, %f0, 0             # encoding: [0xed,0x00,0x00,0x00,0x00,0x2e]
#CHECK: mae	%f0, %f0, 4095          # encoding: [0xed,0x00,0x0f,0xff,0x00,0x2e]
#CHECK: mae	%f0, %f0, 0(%r1)        # encoding: [0xed,0x00,0x10,0x00,0x00,0x2e]
#CHECK: mae	%f0, %f0, 0(%r15)       # encoding: [0xed,0x00,0xf0,0x00,0x00,0x2e]
#CHECK: mae	%f0, %f0, 4095(%r1,%r15) # encoding: [0xed,0x01,0xff,0xff,0x00,0x2e]
#CHECK: mae	%f0, %f0, 4095(%r15,%r1) # encoding: [0xed,0x0f,0x1f,0xff,0x00,0x2e]
#CHECK: mae	%f0, %f15, 0            # encoding: [0xed,0xf0,0x00,0x00,0x00,0x2e]
#CHECK: mae	%f15, %f0, 0            # encoding: [0xed,0x00,0x00,0x00,0xf0,0x2e]
#CHECK: mae	%f15, %f15, 0           # encoding: [0xed,0xf0,0x00,0x00,0xf0,0x2e]

	mae	%f0, %f0, 0
	mae	%f0, %f0, 4095
	mae	%f0, %f0, 0(%r1)
	mae	%f0, %f0, 0(%r15)
	mae	%f0, %f0, 4095(%r1,%r15)
	mae	%f0, %f0, 4095(%r15,%r1)
	mae	%f0, %f15, 0
	mae	%f15, %f0, 0
	mae	%f15, %f15, 0

#CHECK: maeb	%f0, %f0, 0             # encoding: [0xed,0x00,0x00,0x00,0x00,0x0e]
#CHECK: maeb	%f0, %f0, 4095          # encoding: [0xed,0x00,0x0f,0xff,0x00,0x0e]
#CHECK: maeb	%f0, %f0, 0(%r1)        # encoding: [0xed,0x00,0x10,0x00,0x00,0x0e]
#CHECK: maeb	%f0, %f0, 0(%r15)       # encoding: [0xed,0x00,0xf0,0x00,0x00,0x0e]
#CHECK: maeb	%f0, %f0, 4095(%r1,%r15) # encoding: [0xed,0x01,0xff,0xff,0x00,0x0e]
#CHECK: maeb	%f0, %f0, 4095(%r15,%r1) # encoding: [0xed,0x0f,0x1f,0xff,0x00,0x0e]
#CHECK: maeb	%f0, %f15, 0            # encoding: [0xed,0xf0,0x00,0x00,0x00,0x0e]
#CHECK: maeb	%f15, %f0, 0            # encoding: [0xed,0x00,0x00,0x00,0xf0,0x0e]
#CHECK: maeb	%f15, %f15, 0           # encoding: [0xed,0xf0,0x00,0x00,0xf0,0x0e]

	maeb	%f0, %f0, 0
	maeb	%f0, %f0, 4095
	maeb	%f0, %f0, 0(%r1)
	maeb	%f0, %f0, 0(%r15)
	maeb	%f0, %f0, 4095(%r1,%r15)
	maeb	%f0, %f0, 4095(%r15,%r1)
	maeb	%f0, %f15, 0
	maeb	%f15, %f0, 0
	maeb	%f15, %f15, 0

#CHECK: maebr	%f0, %f0, %f0           # encoding: [0xb3,0x0e,0x00,0x00]
#CHECK: maebr	%f0, %f0, %f15          # encoding: [0xb3,0x0e,0x00,0x0f]
#CHECK: maebr	%f0, %f15, %f0          # encoding: [0xb3,0x0e,0x00,0xf0]
#CHECK: maebr	%f15, %f0, %f0          # encoding: [0xb3,0x0e,0xf0,0x00]
#CHECK: maebr	%f7, %f8, %f9           # encoding: [0xb3,0x0e,0x70,0x89]
#CHECK: maebr	%f15, %f15, %f15        # encoding: [0xb3,0x0e,0xf0,0xff]

	maebr	%f0, %f0, %f0
	maebr	%f0, %f0, %f15
	maebr	%f0, %f15, %f0
	maebr	%f15, %f0, %f0
	maebr	%f7, %f8, %f9
	maebr	%f15, %f15, %f15

#CHECK: maer	%f0, %f0, %f0           # encoding: [0xb3,0x2e,0x00,0x00]
#CHECK: maer	%f0, %f0, %f15          # encoding: [0xb3,0x2e,0x00,0x0f]
#CHECK: maer	%f0, %f15, %f0          # encoding: [0xb3,0x2e,0x00,0xf0]
#CHECK: maer	%f15, %f0, %f0          # encoding: [0xb3,0x2e,0xf0,0x00]
#CHECK: maer	%f7, %f8, %f9           # encoding: [0xb3,0x2e,0x70,0x89]
#CHECK: maer	%f15, %f15, %f15        # encoding: [0xb3,0x2e,0xf0,0xff]

	maer	%f0, %f0, %f0
	maer	%f0, %f0, %f15
	maer	%f0, %f15, %f0
	maer	%f15, %f0, %f0
	maer	%f7, %f8, %f9
	maer	%f15, %f15, %f15

#CHECK: may	%f0, %f0, 0             # encoding: [0xed,0x00,0x00,0x00,0x00,0x3a]
#CHECK: may	%f0, %f0, 4095          # encoding: [0xed,0x00,0x0f,0xff,0x00,0x3a]
#CHECK: may	%f0, %f0, 0(%r1)        # encoding: [0xed,0x00,0x10,0x00,0x00,0x3a]
#CHECK: may	%f0, %f0, 0(%r15)       # encoding: [0xed,0x00,0xf0,0x00,0x00,0x3a]
#CHECK: may	%f0, %f0, 4095(%r1,%r15) # encoding: [0xed,0x01,0xff,0xff,0x00,0x3a]
#CHECK: may	%f0, %f0, 4095(%r15,%r1) # encoding: [0xed,0x0f,0x1f,0xff,0x00,0x3a]
#CHECK: may	%f0, %f15, 0            # encoding: [0xed,0xf0,0x00,0x00,0x00,0x3a]
#CHECK: may	%f13, %f0, 0            # encoding: [0xed,0x00,0x00,0x00,0xd0,0x3a]
#CHECK: may	%f13, %f15, 0           # encoding: [0xed,0xf0,0x00,0x00,0xd0,0x3a]
#CHECK: may	%f2, %f0, 0             # encoding: [0xed,0x00,0x00,0x00,0x20,0x3a]

	may	%f0, %f0, 0
	may	%f0, %f0, 4095
	may	%f0, %f0, 0(%r1)
	may	%f0, %f0, 0(%r15)
	may	%f0, %f0, 4095(%r1,%r15)
	may	%f0, %f0, 4095(%r15,%r1)
	may	%f0, %f15, 0
	may	%f13, %f0, 0
	may	%f13, %f15, 0
	may	%f2, %f0, 0

#CHECK: mayh	%f0, %f0, 0             # encoding: [0xed,0x00,0x00,0x00,0x00,0x3c]
#CHECK: mayh	%f0, %f0, 4095          # encoding: [0xed,0x00,0x0f,0xff,0x00,0x3c]
#CHECK: mayh	%f0, %f0, 0(%r1)        # encoding: [0xed,0x00,0x10,0x00,0x00,0x3c]
#CHECK: mayh	%f0, %f0, 0(%r15)       # encoding: [0xed,0x00,0xf0,0x00,0x00,0x3c]
#CHECK: mayh	%f0, %f0, 4095(%r1,%r15) # encoding: [0xed,0x01,0xff,0xff,0x00,0x3c]
#CHECK: mayh	%f0, %f0, 4095(%r15,%r1) # encoding: [0xed,0x0f,0x1f,0xff,0x00,0x3c]
#CHECK: mayh	%f0, %f15, 0            # encoding: [0xed,0xf0,0x00,0x00,0x00,0x3c]
#CHECK: mayh	%f15, %f0, 0            # encoding: [0xed,0x00,0x00,0x00,0xf0,0x3c]
#CHECK: mayh	%f15, %f15, 0           # encoding: [0xed,0xf0,0x00,0x00,0xf0,0x3c]

	mayh	%f0, %f0, 0
	mayh	%f0, %f0, 4095
	mayh	%f0, %f0, 0(%r1)
	mayh	%f0, %f0, 0(%r15)
	mayh	%f0, %f0, 4095(%r1,%r15)
	mayh	%f0, %f0, 4095(%r15,%r1)
	mayh	%f0, %f15, 0
	mayh	%f15, %f0, 0
	mayh	%f15, %f15, 0

#CHECK: mayhr	%f0, %f0, %f0           # encoding: [0xb3,0x3c,0x00,0x00]
#CHECK: mayhr	%f0, %f0, %f15          # encoding: [0xb3,0x3c,0x00,0x0f]
#CHECK: mayhr	%f0, %f15, %f0          # encoding: [0xb3,0x3c,0x00,0xf0]
#CHECK: mayhr	%f15, %f0, %f0          # encoding: [0xb3,0x3c,0xf0,0x00]
#CHECK: mayhr	%f7, %f8, %f9           # encoding: [0xb3,0x3c,0x70,0x89]
#CHECK: mayhr	%f15, %f15, %f15        # encoding: [0xb3,0x3c,0xf0,0xff]

	mayhr	%f0, %f0, %f0
	mayhr	%f0, %f0, %f15
	mayhr	%f0, %f15, %f0
	mayhr	%f15, %f0, %f0
	mayhr	%f7, %f8, %f9
	mayhr	%f15, %f15, %f15

#CHECK: mayl	%f0, %f0, 0             # encoding: [0xed,0x00,0x00,0x00,0x00,0x38]
#CHECK: mayl	%f0, %f0, 4095          # encoding: [0xed,0x00,0x0f,0xff,0x00,0x38]
#CHECK: mayl	%f0, %f0, 0(%r1)        # encoding: [0xed,0x00,0x10,0x00,0x00,0x38]
#CHECK: mayl	%f0, %f0, 0(%r15)       # encoding: [0xed,0x00,0xf0,0x00,0x00,0x38]
#CHECK: mayl	%f0, %f0, 4095(%r1,%r15) # encoding: [0xed,0x01,0xff,0xff,0x00,0x38]
#CHECK: mayl	%f0, %f0, 4095(%r15,%r1) # encoding: [0xed,0x0f,0x1f,0xff,0x00,0x38]
#CHECK: mayl	%f0, %f15, 0            # encoding: [0xed,0xf0,0x00,0x00,0x00,0x38]
#CHECK: mayl	%f15, %f0, 0            # encoding: [0xed,0x00,0x00,0x00,0xf0,0x38]
#CHECK: mayl	%f15, %f15, 0           # encoding: [0xed,0xf0,0x00,0x00,0xf0,0x38]

	mayl	%f0, %f0, 0
	mayl	%f0, %f0, 4095
	mayl	%f0, %f0, 0(%r1)
	mayl	%f0, %f0, 0(%r15)
	mayl	%f0, %f0, 4095(%r1,%r15)
	mayl	%f0, %f0, 4095(%r15,%r1)
	mayl	%f0, %f15, 0
	mayl	%f15, %f0, 0
	mayl	%f15, %f15, 0

#CHECK: maylr	%f0, %f0, %f0           # encoding: [0xb3,0x38,0x00,0x00]
#CHECK: maylr	%f0, %f0, %f15          # encoding: [0xb3,0x38,0x00,0x0f]
#CHECK: maylr	%f0, %f15, %f0          # encoding: [0xb3,0x38,0x00,0xf0]
#CHECK: maylr	%f15, %f0, %f0          # encoding: [0xb3,0x38,0xf0,0x00]
#CHECK: maylr	%f7, %f8, %f9           # encoding: [0xb3,0x38,0x70,0x89]
#CHECK: maylr	%f15, %f15, %f15        # encoding: [0xb3,0x38,0xf0,0xff]

	maylr	%f0, %f0, %f0
	maylr	%f0, %f0, %f15
	maylr	%f0, %f15, %f0
	maylr	%f15, %f0, %f0
	maylr	%f7, %f8, %f9
	maylr	%f15, %f15, %f15

#CHECK: mayr	%f0, %f0, %f0           # encoding: [0xb3,0x3a,0x00,0x00]
#CHECK: mayr	%f0, %f0, %f15          # encoding: [0xb3,0x3a,0x00,0x0f]
#CHECK: mayr	%f0, %f15, %f0          # encoding: [0xb3,0x3a,0x00,0xf0]
#CHECK: mayr	%f13, %f0, %f0          # encoding: [0xb3,0x3a,0xd0,0x00]
#CHECK: mayr	%f5, %f8, %f9           # encoding: [0xb3,0x3a,0x50,0x89]
#CHECK: mayr	%f13, %f15, %f15        # encoding: [0xb3,0x3a,0xd0,0xff]
#CHECK: mayr	%f2, %f0, %f0           # encoding: [0xb3,0x3a,0x20,0x00]

	mayr	%f0, %f0, %f0
	mayr	%f0, %f0, %f15
	mayr	%f0, %f15, %f0
	mayr	%f13, %f0, %f0
	mayr	%f5, %f8, %f9
	mayr	%f13, %f15, %f15
	mayr	%f2, %f0, %f0

#CHECK: mc	0, 0                    # encoding: [0xaf,0x00,0x00,0x00]
#CHECK: mc	4095, 0                 # encoding: [0xaf,0x00,0x0f,0xff]
#CHECK: mc	0, 255                  # encoding: [0xaf,0xff,0x00,0x00]
#CHECK: mc	0(%r1), 42              # encoding: [0xaf,0x2a,0x10,0x00]
#CHECK: mc	0(%r15), 42             # encoding: [0xaf,0x2a,0xf0,0x00]
#CHECK: mc	4095(%r1), 42           # encoding: [0xaf,0x2a,0x1f,0xff]
#CHECK: mc	4095(%r15), 42          # encoding: [0xaf,0x2a,0xff,0xff]

	mc	0, 0
	mc	4095, 0
	mc	0, 255
	mc	0(%r1), 42
	mc	0(%r15), 42
	mc	4095(%r1), 42
	mc	4095(%r15), 42

#CHECK: md	%f0, 0                  # encoding: [0x6c,0x00,0x00,0x00]
#CHECK: md	%f0, 4095               # encoding: [0x6c,0x00,0x0f,0xff]
#CHECK: md	%f0, 0(%r1)             # encoding: [0x6c,0x00,0x10,0x00]
#CHECK: md	%f0, 0(%r15)            # encoding: [0x6c,0x00,0xf0,0x00]
#CHECK: md	%f0, 4095(%r1,%r15)     # encoding: [0x6c,0x01,0xff,0xff]
#CHECK: md	%f0, 4095(%r15,%r1)     # encoding: [0x6c,0x0f,0x1f,0xff]
#CHECK: md	%f15, 0                 # encoding: [0x6c,0xf0,0x00,0x00]

	md	%f0, 0
	md	%f0, 4095
	md	%f0, 0(%r1)
	md	%f0, 0(%r15)
	md	%f0, 4095(%r1,%r15)
	md	%f0, 4095(%r15,%r1)
	md	%f15, 0

#CHECK: mdb	%f0, 0                  # encoding: [0xed,0x00,0x00,0x00,0x00,0x1c]
#CHECK: mdb	%f0, 4095               # encoding: [0xed,0x00,0x0f,0xff,0x00,0x1c]
#CHECK: mdb	%f0, 0(%r1)             # encoding: [0xed,0x00,0x10,0x00,0x00,0x1c]
#CHECK: mdb	%f0, 0(%r15)            # encoding: [0xed,0x00,0xf0,0x00,0x00,0x1c]
#CHECK: mdb	%f0, 4095(%r1,%r15)     # encoding: [0xed,0x01,0xff,0xff,0x00,0x1c]
#CHECK: mdb	%f0, 4095(%r15,%r1)     # encoding: [0xed,0x0f,0x1f,0xff,0x00,0x1c]
#CHECK: mdb	%f15, 0                 # encoding: [0xed,0xf0,0x00,0x00,0x00,0x1c]

	mdb	%f0, 0
	mdb	%f0, 4095
	mdb	%f0, 0(%r1)
	mdb	%f0, 0(%r15)
	mdb	%f0, 4095(%r1,%r15)
	mdb	%f0, 4095(%r15,%r1)
	mdb	%f15, 0

#CHECK: mdbr	%f0, %f0                # encoding: [0xb3,0x1c,0x00,0x00]
#CHECK: mdbr	%f0, %f15               # encoding: [0xb3,0x1c,0x00,0x0f]
#CHECK: mdbr	%f7, %f8                # encoding: [0xb3,0x1c,0x00,0x78]
#CHECK: mdbr	%f15, %f0               # encoding: [0xb3,0x1c,0x00,0xf0]

	mdbr	%f0, %f0
	mdbr	%f0, %f15
	mdbr	%f7, %f8
	mdbr	%f15, %f0

#CHECK: mde	%f0, 0                  # encoding: [0x7c,0x00,0x00,0x00]
#CHECK: mde	%f0, 4095               # encoding: [0x7c,0x00,0x0f,0xff]
#CHECK: mde	%f0, 0(%r1)             # encoding: [0x7c,0x00,0x10,0x00]
#CHECK: mde	%f0, 0(%r15)            # encoding: [0x7c,0x00,0xf0,0x00]
#CHECK: mde	%f0, 4095(%r1,%r15)     # encoding: [0x7c,0x01,0xff,0xff]
#CHECK: mde	%f0, 4095(%r15,%r1)     # encoding: [0x7c,0x0f,0x1f,0xff]
#CHECK: mde	%f15, 0                 # encoding: [0x7c,0xf0,0x00,0x00]

	mde	%f0, 0
	mde	%f0, 4095
	mde	%f0, 0(%r1)
	mde	%f0, 0(%r15)
	mde	%f0, 4095(%r1,%r15)
	mde	%f0, 4095(%r15,%r1)
	mde	%f15, 0

#CHECK: mdeb	%f0, 0                  # encoding: [0xed,0x00,0x00,0x00,0x00,0x0c]
#CHECK: mdeb	%f0, 4095               # encoding: [0xed,0x00,0x0f,0xff,0x00,0x0c]
#CHECK: mdeb	%f0, 0(%r1)             # encoding: [0xed,0x00,0x10,0x00,0x00,0x0c]
#CHECK: mdeb	%f0, 0(%r15)            # encoding: [0xed,0x00,0xf0,0x00,0x00,0x0c]
#CHECK: mdeb	%f0, 4095(%r1,%r15)     # encoding: [0xed,0x01,0xff,0xff,0x00,0x0c]
#CHECK: mdeb	%f0, 4095(%r15,%r1)     # encoding: [0xed,0x0f,0x1f,0xff,0x00,0x0c]
#CHECK: mdeb	%f15, 0                 # encoding: [0xed,0xf0,0x00,0x00,0x00,0x0c]

	mdeb	%f0, 0
	mdeb	%f0, 4095
	mdeb	%f0, 0(%r1)
	mdeb	%f0, 0(%r15)
	mdeb	%f0, 4095(%r1,%r15)
	mdeb	%f0, 4095(%r15,%r1)
	mdeb	%f15, 0

#CHECK: mdebr	%f0, %f0                # encoding: [0xb3,0x0c,0x00,0x00]
#CHECK: mdebr	%f0, %f15               # encoding: [0xb3,0x0c,0x00,0x0f]
#CHECK: mdebr	%f7, %f8                # encoding: [0xb3,0x0c,0x00,0x78]
#CHECK: mdebr	%f15, %f0               # encoding: [0xb3,0x0c,0x00,0xf0]

	mdebr	%f0, %f0
	mdebr	%f0, %f15
	mdebr	%f7, %f8
	mdebr	%f15, %f0

#CHECK: mder	%f0, %f0                # encoding: [0x3c,0x00]
#CHECK: mder	%f0, %f15               # encoding: [0x3c,0x0f]
#CHECK: mder	%f7, %f8                # encoding: [0x3c,0x78]
#CHECK: mder	%f15, %f0               # encoding: [0x3c,0xf0]

	mder	%f0, %f0
	mder	%f0, %f15
	mder	%f7, %f8
	mder	%f15, %f0

#CHECK: mdr	%f0, %f0                # encoding: [0x2c,0x00]
#CHECK: mdr	%f0, %f15               # encoding: [0x2c,0x0f]
#CHECK: mdr	%f7, %f8                # encoding: [0x2c,0x78]
#CHECK: mdr	%f15, %f0               # encoding: [0x2c,0xf0]

	mdr	%f0, %f0
	mdr	%f0, %f15
	mdr	%f7, %f8
	mdr	%f15, %f0

#CHECK: mdtr	%f0, %f0, %f0           # encoding: [0xb3,0xd0,0x00,0x00]
#CHECK: mdtr	%f0, %f0, %f15          # encoding: [0xb3,0xd0,0xf0,0x00]
#CHECK: mdtr	%f0, %f15, %f0          # encoding: [0xb3,0xd0,0x00,0x0f]
#CHECK: mdtr	%f15, %f0, %f0          # encoding: [0xb3,0xd0,0x00,0xf0]
#CHECK: mdtr	%f7, %f8, %f9           # encoding: [0xb3,0xd0,0x90,0x78]

	mdtr	%f0, %f0, %f0
	mdtr	%f0, %f0, %f15
	mdtr	%f0, %f15, %f0
	mdtr	%f15, %f0, %f0
	mdtr	%f7, %f8, %f9

#CHECK: me	%f0, 0                  # encoding: [0x7c,0x00,0x00,0x00]
#CHECK: me	%f0, 4095               # encoding: [0x7c,0x00,0x0f,0xff]
#CHECK: me	%f0, 0(%r1)             # encoding: [0x7c,0x00,0x10,0x00]
#CHECK: me	%f0, 0(%r15)            # encoding: [0x7c,0x00,0xf0,0x00]
#CHECK: me	%f0, 4095(%r1,%r15)     # encoding: [0x7c,0x01,0xff,0xff]
#CHECK: me	%f0, 4095(%r15,%r1)     # encoding: [0x7c,0x0f,0x1f,0xff]
#CHECK: me	%f15, 0                 # encoding: [0x7c,0xf0,0x00,0x00]

	me	%f0, 0
	me	%f0, 4095
	me	%f0, 0(%r1)
	me	%f0, 0(%r15)
	me	%f0, 4095(%r1,%r15)
	me	%f0, 4095(%r15,%r1)
	me	%f15, 0

#CHECK: mee	%f0, 0                  # encoding: [0xed,0x00,0x00,0x00,0x00,0x37]
#CHECK: mee	%f0, 4095               # encoding: [0xed,0x00,0x0f,0xff,0x00,0x37]
#CHECK: mee	%f0, 0(%r1)             # encoding: [0xed,0x00,0x10,0x00,0x00,0x37]
#CHECK: mee	%f0, 0(%r15)            # encoding: [0xed,0x00,0xf0,0x00,0x00,0x37]
#CHECK: mee	%f0, 4095(%r1,%r15)     # encoding: [0xed,0x01,0xff,0xff,0x00,0x37]
#CHECK: mee	%f0, 4095(%r15,%r1)     # encoding: [0xed,0x0f,0x1f,0xff,0x00,0x37]
#CHECK: mee	%f15, 0                 # encoding: [0xed,0xf0,0x00,0x00,0x00,0x37]

	mee	%f0, 0
	mee	%f0, 4095
	mee	%f0, 0(%r1)
	mee	%f0, 0(%r15)
	mee	%f0, 4095(%r1,%r15)
	mee	%f0, 4095(%r15,%r1)
	mee	%f15, 0

#CHECK: meeb	%f0, 0                  # encoding: [0xed,0x00,0x00,0x00,0x00,0x17]
#CHECK: meeb	%f0, 4095               # encoding: [0xed,0x00,0x0f,0xff,0x00,0x17]
#CHECK: meeb	%f0, 0(%r1)             # encoding: [0xed,0x00,0x10,0x00,0x00,0x17]
#CHECK: meeb	%f0, 0(%r15)            # encoding: [0xed,0x00,0xf0,0x00,0x00,0x17]
#CHECK: meeb	%f0, 4095(%r1,%r15)     # encoding: [0xed,0x01,0xff,0xff,0x00,0x17]
#CHECK: meeb	%f0, 4095(%r15,%r1)     # encoding: [0xed,0x0f,0x1f,0xff,0x00,0x17]
#CHECK: meeb	%f15, 0                 # encoding: [0xed,0xf0,0x00,0x00,0x00,0x17]

	meeb	%f0, 0
	meeb	%f0, 4095
	meeb	%f0, 0(%r1)
	meeb	%f0, 0(%r15)
	meeb	%f0, 4095(%r1,%r15)
	meeb	%f0, 4095(%r15,%r1)
	meeb	%f15, 0

#CHECK: meebr	%f0, %f0                # encoding: [0xb3,0x17,0x00,0x00]
#CHECK: meebr	%f0, %f15               # encoding: [0xb3,0x17,0x00,0x0f]
#CHECK: meebr	%f7, %f8                # encoding: [0xb3,0x17,0x00,0x78]
#CHECK: meebr	%f15, %f0               # encoding: [0xb3,0x17,0x00,0xf0]

	meebr	%f0, %f0
	meebr	%f0, %f15
	meebr	%f7, %f8
	meebr	%f15, %f0

#CHECK: meer	%f0, %f0                # encoding: [0xb3,0x37,0x00,0x00]
#CHECK: meer	%f0, %f15               # encoding: [0xb3,0x37,0x00,0x0f]
#CHECK: meer	%f7, %f8                # encoding: [0xb3,0x37,0x00,0x78]
#CHECK: meer	%f15, %f0               # encoding: [0xb3,0x37,0x00,0xf0]

	meer	%f0, %f0
	meer	%f0, %f15
	meer	%f7, %f8
	meer	%f15, %f0

#CHECK: mer	%f0, %f0                # encoding: [0x3c,0x00]
#CHECK: mer	%f0, %f15               # encoding: [0x3c,0x0f]
#CHECK: mer	%f7, %f8                # encoding: [0x3c,0x78]
#CHECK: mer	%f15, %f0               # encoding: [0x3c,0xf0]

	mer	%f0, %f0
	mer	%f0, %f15
	mer	%f7, %f8
	mer	%f15, %f0

#CHECK: mfy	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x5c]
#CHECK: mfy	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x5c]
#CHECK: mfy	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x5c]
#CHECK: mfy	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x5c]
#CHECK: mfy	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x5c]
#CHECK: mfy	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x5c]
#CHECK: mfy	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x5c]
#CHECK: mfy	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x5c]
#CHECK: mfy	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x5c]
#CHECK: mfy	%r14, 0                 # encoding: [0xe3,0xe0,0x00,0x00,0x00,0x5c]

	mfy	%r0, -524288
	mfy	%r0, -1
	mfy	%r0, 0
	mfy	%r0, 1
	mfy	%r0, 524287
	mfy	%r0, 0(%r1)
	mfy	%r0, 0(%r15)
	mfy	%r0, 524287(%r1,%r15)
	mfy	%r0, 524287(%r15,%r1)
	mfy	%r14, 0

#CHECK: mghi	%r0, -32768             # encoding: [0xa7,0x0d,0x80,0x00]
#CHECK: mghi	%r0, -1                 # encoding: [0xa7,0x0d,0xff,0xff]
#CHECK: mghi	%r0, 0                  # encoding: [0xa7,0x0d,0x00,0x00]
#CHECK: mghi	%r0, 1                  # encoding: [0xa7,0x0d,0x00,0x01]
#CHECK: mghi	%r0, 32767              # encoding: [0xa7,0x0d,0x7f,0xff]
#CHECK: mghi	%r15, 0                 # encoding: [0xa7,0xfd,0x00,0x00]
#CHECK: mghi	%r0, foo                # encoding: [0xa7,0x0d,A,A]

	mghi	%r0, -32768
	mghi	%r0, -1
	mghi	%r0, 0
	mghi	%r0, 1
	mghi	%r0, 32767
	mghi	%r15, 0
	mghi	%r0, foo

#CHECK: mh	%r0, 0                  # encoding: [0x4c,0x00,0x00,0x00]
#CHECK: mh	%r0, 4095               # encoding: [0x4c,0x00,0x0f,0xff]
#CHECK: mh	%r0, 0(%r1)             # encoding: [0x4c,0x00,0x10,0x00]
#CHECK: mh	%r0, 0(%r15)            # encoding: [0x4c,0x00,0xf0,0x00]
#CHECK: mh	%r0, 4095(%r1,%r15)     # encoding: [0x4c,0x01,0xff,0xff]
#CHECK: mh	%r0, 4095(%r15,%r1)     # encoding: [0x4c,0x0f,0x1f,0xff]
#CHECK: mh	%r15, 0                 # encoding: [0x4c,0xf0,0x00,0x00]

	mh	%r0, 0
	mh	%r0, 4095
	mh	%r0, 0(%r1)
	mh	%r0, 0(%r15)
	mh	%r0, 4095(%r1,%r15)
	mh	%r0, 4095(%r15,%r1)
	mh	%r15, 0

#CHECK: mhi	%r0, -32768             # encoding: [0xa7,0x0c,0x80,0x00]
#CHECK: mhi	%r0, -1                 # encoding: [0xa7,0x0c,0xff,0xff]
#CHECK: mhi	%r0, 0                  # encoding: [0xa7,0x0c,0x00,0x00]
#CHECK: mhi	%r0, 1                  # encoding: [0xa7,0x0c,0x00,0x01]
#CHECK: mhi	%r0, 32767              # encoding: [0xa7,0x0c,0x7f,0xff]
#CHECK: mhi	%r15, 0                 # encoding: [0xa7,0xfc,0x00,0x00]
#CHECK: mhi	%r0, foo                # encoding: [0xa7,0x0c,A,A]

	mhi	%r0, -32768
	mhi	%r0, -1
	mhi	%r0, 0
	mhi	%r0, 1
	mhi	%r0, 32767
	mhi	%r15, 0
	mhi	%r0, foo

#CHECK: mhy	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x7c]
#CHECK: mhy	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x7c]
#CHECK: mhy	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x7c]
#CHECK: mhy	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x7c]
#CHECK: mhy	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x7c]
#CHECK: mhy	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x7c]
#CHECK: mhy	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x7c]
#CHECK: mhy	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x7c]
#CHECK: mhy	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x7c]
#CHECK: mhy	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x7c]

	mhy	%r0, -524288
	mhy	%r0, -1
	mhy	%r0, 0
	mhy	%r0, 1
	mhy	%r0, 524287
	mhy	%r0, 0(%r1)
	mhy	%r0, 0(%r15)
	mhy	%r0, 524287(%r1,%r15)
	mhy	%r0, 524287(%r15,%r1)
	mhy	%r15, 0

#CHECK: ml	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x96]
#CHECK: ml	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x96]
#CHECK: ml	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x96]
#CHECK: ml	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x96]
#CHECK: ml	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x96]
#CHECK: ml	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x96]
#CHECK: ml	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x96]
#CHECK: ml	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x96]
#CHECK: ml	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x96]
#CHECK: ml	%r14, 0                 # encoding: [0xe3,0xe0,0x00,0x00,0x00,0x96]

	ml	%r0, -524288
	ml	%r0, -1
	ml	%r0, 0
	ml	%r0, 1
	ml	%r0, 524287
	ml	%r0, 0(%r1)
	ml	%r0, 0(%r15)
	ml	%r0, 524287(%r1,%r15)
	ml	%r0, 524287(%r15,%r1)
	ml	%r14, 0

#CHECK: mlg	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x86]
#CHECK: mlg	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x86]
#CHECK: mlg	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x86]
#CHECK: mlg	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x86]
#CHECK: mlg	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x86]
#CHECK: mlg	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x86]
#CHECK: mlg	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x86]
#CHECK: mlg	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x86]
#CHECK: mlg	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x86]
#CHECK: mlg	%r14, 0                 # encoding: [0xe3,0xe0,0x00,0x00,0x00,0x86]

	mlg	%r0, -524288
	mlg	%r0, -1
	mlg	%r0, 0
	mlg	%r0, 1
	mlg	%r0, 524287
	mlg	%r0, 0(%r1)
	mlg	%r0, 0(%r15)
	mlg	%r0, 524287(%r1,%r15)
	mlg	%r0, 524287(%r15,%r1)
	mlg	%r14, 0

#CHECK: mlgr	%r0, %r0                # encoding: [0xb9,0x86,0x00,0x00]
#CHECK: mlgr	%r0, %r15               # encoding: [0xb9,0x86,0x00,0x0f]
#CHECK: mlgr	%r14, %r0               # encoding: [0xb9,0x86,0x00,0xe0]
#CHECK: mlgr	%r6, %r9                # encoding: [0xb9,0x86,0x00,0x69]

	mlgr	%r0,%r0
	mlgr	%r0,%r15
	mlgr	%r14,%r0
	mlgr	%r6,%r9

#CHECK: mlr	%r0, %r0                # encoding: [0xb9,0x96,0x00,0x00]
#CHECK: mlr	%r0, %r15               # encoding: [0xb9,0x96,0x00,0x0f]
#CHECK: mlr	%r14, %r0               # encoding: [0xb9,0x96,0x00,0xe0]
#CHECK: mlr	%r6, %r9                # encoding: [0xb9,0x96,0x00,0x69]

	mlr	%r0,%r0
	mlr	%r0,%r15
	mlr	%r14,%r0
	mlr	%r6,%r9

#CHECK: mp	0(1), 0(1)              # encoding: [0xfc,0x00,0x00,0x00,0x00,0x00]
#CHECK: mp	0(1), 0(1,%r1)          # encoding: [0xfc,0x00,0x00,0x00,0x10,0x00]
#CHECK: mp	0(1), 0(1,%r15)         # encoding: [0xfc,0x00,0x00,0x00,0xf0,0x00]
#CHECK: mp	0(1), 4095(1)           # encoding: [0xfc,0x00,0x00,0x00,0x0f,0xff]
#CHECK: mp	0(1), 4095(1,%r1)       # encoding: [0xfc,0x00,0x00,0x00,0x1f,0xff]
#CHECK: mp	0(1), 4095(1,%r15)      # encoding: [0xfc,0x00,0x00,0x00,0xff,0xff]
#CHECK: mp	0(1,%r1), 0(1)          # encoding: [0xfc,0x00,0x10,0x00,0x00,0x00]
#CHECK: mp	0(1,%r15), 0(1)         # encoding: [0xfc,0x00,0xf0,0x00,0x00,0x00]
#CHECK: mp	4095(1,%r1), 0(1)       # encoding: [0xfc,0x00,0x1f,0xff,0x00,0x00]
#CHECK: mp	4095(1,%r15), 0(1)      # encoding: [0xfc,0x00,0xff,0xff,0x00,0x00]
#CHECK: mp	0(16,%r1), 0(1)         # encoding: [0xfc,0xf0,0x10,0x00,0x00,0x00]
#CHECK: mp	0(16,%r15), 0(1)        # encoding: [0xfc,0xf0,0xf0,0x00,0x00,0x00]
#CHECK: mp	0(1), 0(16,%r1)         # encoding: [0xfc,0x0f,0x00,0x00,0x10,0x00]
#CHECK: mp	0(1), 0(16,%r15)        # encoding: [0xfc,0x0f,0x00,0x00,0xf0,0x00]

	mp	0(1), 0(1)
	mp	0(1), 0(1,%r1)
	mp	0(1), 0(1,%r15)
	mp	0(1), 4095(1)
	mp	0(1), 4095(1,%r1)
	mp	0(1), 4095(1,%r15)
	mp	0(1,%r1), 0(1)
	mp	0(1,%r15), 0(1)
	mp	4095(1,%r1), 0(1)
	mp	4095(1,%r15), 0(1)
	mp	0(16,%r1), 0(1)
	mp	0(16,%r15), 0(1)
	mp	0(1), 0(16,%r1)
	mp	0(1), 0(16,%r15)

#CHECK: mr	%r0, %r0                # encoding: [0x1c,0x00]
#CHECK: mr	%r0, %r15               # encoding: [0x1c,0x0f]
#CHECK: mr	%r14, %r0               # encoding: [0x1c,0xe0]
#CHECK: mr	%r6, %r9                # encoding: [0x1c,0x69]

	mr	%r0,%r0
	mr	%r0,%r15
	mr	%r14,%r0
	mr	%r6,%r9

#CHECK: ms	%r0, 0                  # encoding: [0x71,0x00,0x00,0x00]
#CHECK: ms	%r0, 4095               # encoding: [0x71,0x00,0x0f,0xff]
#CHECK: ms	%r0, 0(%r1)             # encoding: [0x71,0x00,0x10,0x00]
#CHECK: ms	%r0, 0(%r15)            # encoding: [0x71,0x00,0xf0,0x00]
#CHECK: ms	%r0, 4095(%r1,%r15)     # encoding: [0x71,0x01,0xff,0xff]
#CHECK: ms	%r0, 4095(%r15,%r1)     # encoding: [0x71,0x0f,0x1f,0xff]
#CHECK: ms	%r15, 0                 # encoding: [0x71,0xf0,0x00,0x00]

	ms	%r0, 0
	ms	%r0, 4095
	ms	%r0, 0(%r1)
	ms	%r0, 0(%r15)
	ms	%r0, 4095(%r1,%r15)
	ms	%r0, 4095(%r15,%r1)
	ms	%r15, 0

#CHECK: msch	0                       # encoding: [0xb2,0x32,0x00,0x00]
#CHECK: msch	0(%r1)                  # encoding: [0xb2,0x32,0x10,0x00]
#CHECK: msch	0(%r15)                 # encoding: [0xb2,0x32,0xf0,0x00]
#CHECK: msch	4095                    # encoding: [0xb2,0x32,0x0f,0xff]
#CHECK: msch	4095(%r1)               # encoding: [0xb2,0x32,0x1f,0xff]
#CHECK: msch	4095(%r15)              # encoding: [0xb2,0x32,0xff,0xff]

	msch	0
	msch	0(%r1)
	msch	0(%r15)
	msch	4095
	msch	4095(%r1)
	msch	4095(%r15)

#CHECK: msd	%f0, %f0, 0             # encoding: [0xed,0x00,0x00,0x00,0x00,0x3f]
#CHECK: msd	%f0, %f0, 4095          # encoding: [0xed,0x00,0x0f,0xff,0x00,0x3f]
#CHECK: msd	%f0, %f0, 0(%r1)        # encoding: [0xed,0x00,0x10,0x00,0x00,0x3f]
#CHECK: msd	%f0, %f0, 0(%r15)       # encoding: [0xed,0x00,0xf0,0x00,0x00,0x3f]
#CHECK: msd	%f0, %f0, 4095(%r1,%r15) # encoding: [0xed,0x01,0xff,0xff,0x00,0x3f]
#CHECK: msd	%f0, %f0, 4095(%r15,%r1) # encoding: [0xed,0x0f,0x1f,0xff,0x00,0x3f]
#CHECK: msd	%f0, %f15, 0            # encoding: [0xed,0xf0,0x00,0x00,0x00,0x3f]
#CHECK: msd	%f15, %f0, 0            # encoding: [0xed,0x00,0x00,0x00,0xf0,0x3f]
#CHECK: msd	%f15, %f15, 0           # encoding: [0xed,0xf0,0x00,0x00,0xf0,0x3f]

	msd	%f0, %f0, 0
	msd	%f0, %f0, 4095
	msd	%f0, %f0, 0(%r1)
	msd	%f0, %f0, 0(%r15)
	msd	%f0, %f0, 4095(%r1,%r15)
	msd	%f0, %f0, 4095(%r15,%r1)
	msd	%f0, %f15, 0
	msd	%f15, %f0, 0
	msd	%f15, %f15, 0

#CHECK: msdb	%f0, %f0, 0             # encoding: [0xed,0x00,0x00,0x00,0x00,0x1f]
#CHECK: msdb	%f0, %f0, 4095          # encoding: [0xed,0x00,0x0f,0xff,0x00,0x1f]
#CHECK: msdb	%f0, %f0, 0(%r1)        # encoding: [0xed,0x00,0x10,0x00,0x00,0x1f]
#CHECK: msdb	%f0, %f0, 0(%r15)       # encoding: [0xed,0x00,0xf0,0x00,0x00,0x1f]
#CHECK: msdb	%f0, %f0, 4095(%r1,%r15) # encoding: [0xed,0x01,0xff,0xff,0x00,0x1f]
#CHECK: msdb	%f0, %f0, 4095(%r15,%r1) # encoding: [0xed,0x0f,0x1f,0xff,0x00,0x1f]
#CHECK: msdb	%f0, %f15, 0            # encoding: [0xed,0xf0,0x00,0x00,0x00,0x1f]
#CHECK: msdb	%f15, %f0, 0            # encoding: [0xed,0x00,0x00,0x00,0xf0,0x1f]
#CHECK: msdb	%f15, %f15, 0           # encoding: [0xed,0xf0,0x00,0x00,0xf0,0x1f]

	msdb	%f0, %f0, 0
	msdb	%f0, %f0, 4095
	msdb	%f0, %f0, 0(%r1)
	msdb	%f0, %f0, 0(%r15)
	msdb	%f0, %f0, 4095(%r1,%r15)
	msdb	%f0, %f0, 4095(%r15,%r1)
	msdb	%f0, %f15, 0
	msdb	%f15, %f0, 0
	msdb	%f15, %f15, 0

#CHECK: msdbr	%f0, %f0, %f0           # encoding: [0xb3,0x1f,0x00,0x00]
#CHECK: msdbr	%f0, %f0, %f15          # encoding: [0xb3,0x1f,0x00,0x0f]
#CHECK: msdbr	%f0, %f15, %f0          # encoding: [0xb3,0x1f,0x00,0xf0]
#CHECK: msdbr	%f15, %f0, %f0          # encoding: [0xb3,0x1f,0xf0,0x00]
#CHECK: msdbr	%f7, %f8, %f9           # encoding: [0xb3,0x1f,0x70,0x89]
#CHECK: msdbr	%f15, %f15, %f15        # encoding: [0xb3,0x1f,0xf0,0xff]

	msdbr	%f0, %f0, %f0
	msdbr	%f0, %f0, %f15
	msdbr	%f0, %f15, %f0
	msdbr	%f15, %f0, %f0
	msdbr	%f7, %f8, %f9
	msdbr	%f15, %f15, %f15

#CHECK: msdr	%f0, %f0, %f0           # encoding: [0xb3,0x3f,0x00,0x00]
#CHECK: msdr	%f0, %f0, %f15          # encoding: [0xb3,0x3f,0x00,0x0f]
#CHECK: msdr	%f0, %f15, %f0          # encoding: [0xb3,0x3f,0x00,0xf0]
#CHECK: msdr	%f15, %f0, %f0          # encoding: [0xb3,0x3f,0xf0,0x00]
#CHECK: msdr	%f7, %f8, %f9           # encoding: [0xb3,0x3f,0x70,0x89]
#CHECK: msdr	%f15, %f15, %f15        # encoding: [0xb3,0x3f,0xf0,0xff]

	msdr	%f0, %f0, %f0
	msdr	%f0, %f0, %f15
	msdr	%f0, %f15, %f0
	msdr	%f15, %f0, %f0
	msdr	%f7, %f8, %f9
	msdr	%f15, %f15, %f15

#CHECK: mse	%f0, %f0, 0             # encoding: [0xed,0x00,0x00,0x00,0x00,0x2f]
#CHECK: mse	%f0, %f0, 4095          # encoding: [0xed,0x00,0x0f,0xff,0x00,0x2f]
#CHECK: mse	%f0, %f0, 0(%r1)        # encoding: [0xed,0x00,0x10,0x00,0x00,0x2f]
#CHECK: mse	%f0, %f0, 0(%r15)       # encoding: [0xed,0x00,0xf0,0x00,0x00,0x2f]
#CHECK: mse	%f0, %f0, 4095(%r1,%r15) # encoding: [0xed,0x01,0xff,0xff,0x00,0x2f]
#CHECK: mse	%f0, %f0, 4095(%r15,%r1) # encoding: [0xed,0x0f,0x1f,0xff,0x00,0x2f]
#CHECK: mse	%f0, %f15, 0            # encoding: [0xed,0xf0,0x00,0x00,0x00,0x2f]
#CHECK: mse	%f15, %f0, 0            # encoding: [0xed,0x00,0x00,0x00,0xf0,0x2f]
#CHECK: mse	%f15, %f15, 0           # encoding: [0xed,0xf0,0x00,0x00,0xf0,0x2f]

	mse	%f0, %f0, 0
	mse	%f0, %f0, 4095
	mse	%f0, %f0, 0(%r1)
	mse	%f0, %f0, 0(%r15)
	mse	%f0, %f0, 4095(%r1,%r15)
	mse	%f0, %f0, 4095(%r15,%r1)
	mse	%f0, %f15, 0
	mse	%f15, %f0, 0
	mse	%f15, %f15, 0

#CHECK: mseb	%f0, %f0, 0             # encoding: [0xed,0x00,0x00,0x00,0x00,0x0f]
#CHECK: mseb	%f0, %f0, 4095          # encoding: [0xed,0x00,0x0f,0xff,0x00,0x0f]
#CHECK: mseb	%f0, %f0, 0(%r1)        # encoding: [0xed,0x00,0x10,0x00,0x00,0x0f]
#CHECK: mseb	%f0, %f0, 0(%r15)       # encoding: [0xed,0x00,0xf0,0x00,0x00,0x0f]
#CHECK: mseb	%f0, %f0, 4095(%r1,%r15) # encoding: [0xed,0x01,0xff,0xff,0x00,0x0f]
#CHECK: mseb	%f0, %f0, 4095(%r15,%r1) # encoding: [0xed,0x0f,0x1f,0xff,0x00,0x0f]
#CHECK: mseb	%f0, %f15, 0            # encoding: [0xed,0xf0,0x00,0x00,0x00,0x0f]
#CHECK: mseb	%f15, %f0, 0            # encoding: [0xed,0x00,0x00,0x00,0xf0,0x0f]
#CHECK: mseb	%f15, %f15, 0           # encoding: [0xed,0xf0,0x00,0x00,0xf0,0x0f]

	mseb	%f0, %f0, 0
	mseb	%f0, %f0, 4095
	mseb	%f0, %f0, 0(%r1)
	mseb	%f0, %f0, 0(%r15)
	mseb	%f0, %f0, 4095(%r1,%r15)
	mseb	%f0, %f0, 4095(%r15,%r1)
	mseb	%f0, %f15, 0
	mseb	%f15, %f0, 0
	mseb	%f15, %f15, 0

#CHECK: msebr	%f0, %f0, %f0           # encoding: [0xb3,0x0f,0x00,0x00]
#CHECK: msebr	%f0, %f0, %f15          # encoding: [0xb3,0x0f,0x00,0x0f]
#CHECK: msebr	%f0, %f15, %f0          # encoding: [0xb3,0x0f,0x00,0xf0]
#CHECK: msebr	%f15, %f0, %f0          # encoding: [0xb3,0x0f,0xf0,0x00]
#CHECK: msebr	%f7, %f8, %f9           # encoding: [0xb3,0x0f,0x70,0x89]
#CHECK: msebr	%f15, %f15, %f15        # encoding: [0xb3,0x0f,0xf0,0xff]

	msebr	%f0, %f0, %f0
	msebr	%f0, %f0, %f15
	msebr	%f0, %f15, %f0
	msebr	%f15, %f0, %f0
	msebr	%f7, %f8, %f9
	msebr	%f15, %f15, %f15

#CHECK: mser	%f0, %f0, %f0           # encoding: [0xb3,0x2f,0x00,0x00]
#CHECK: mser	%f0, %f0, %f15          # encoding: [0xb3,0x2f,0x00,0x0f]
#CHECK: mser	%f0, %f15, %f0          # encoding: [0xb3,0x2f,0x00,0xf0]
#CHECK: mser	%f15, %f0, %f0          # encoding: [0xb3,0x2f,0xf0,0x00]
#CHECK: mser	%f7, %f8, %f9           # encoding: [0xb3,0x2f,0x70,0x89]
#CHECK: mser	%f15, %f15, %f15        # encoding: [0xb3,0x2f,0xf0,0xff]

	mser	%f0, %f0, %f0
	mser	%f0, %f0, %f15
	mser	%f0, %f15, %f0
	mser	%f15, %f0, %f0
	mser	%f7, %f8, %f9
	mser	%f15, %f15, %f15

#CHECK: msfi	%r0, -2147483648        # encoding: [0xc2,0x01,0x80,0x00,0x00,0x00]
#CHECK: msfi	%r0, -1                 # encoding: [0xc2,0x01,0xff,0xff,0xff,0xff]
#CHECK: msfi	%r0, 0                  # encoding: [0xc2,0x01,0x00,0x00,0x00,0x00]
#CHECK: msfi	%r0, 1                  # encoding: [0xc2,0x01,0x00,0x00,0x00,0x01]
#CHECK: msfi	%r0, 2147483647         # encoding: [0xc2,0x01,0x7f,0xff,0xff,0xff]
#CHECK: msfi	%r15, 0                 # encoding: [0xc2,0xf1,0x00,0x00,0x00,0x00]

	msfi	%r0, -1 << 31
	msfi	%r0, -1
	msfi	%r0, 0
	msfi	%r0, 1
	msfi	%r0, (1 << 31) - 1
	msfi	%r15, 0

#CHECK: msg	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x0c]
#CHECK: msg	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x0c]
#CHECK: msg	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x0c]
#CHECK: msg	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x0c]
#CHECK: msg	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x0c]
#CHECK: msg	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x0c]
#CHECK: msg	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x0c]
#CHECK: msg	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x0c]
#CHECK: msg	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x0c]
#CHECK: msg	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x0c]

	msg	%r0, -524288
	msg	%r0, -1
	msg	%r0, 0
	msg	%r0, 1
	msg	%r0, 524287
	msg	%r0, 0(%r1)
	msg	%r0, 0(%r15)
	msg	%r0, 524287(%r1,%r15)
	msg	%r0, 524287(%r15,%r1)
	msg	%r15, 0

#CHECK: msgf	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x1c]
#CHECK: msgf	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x1c]
#CHECK: msgf	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x1c]
#CHECK: msgf	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x1c]
#CHECK: msgf	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x1c]
#CHECK: msgf	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x1c]
#CHECK: msgf	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x1c]
#CHECK: msgf	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x1c]
#CHECK: msgf	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x1c]
#CHECK: msgf	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x1c]

	msgf	%r0, -524288
	msgf	%r0, -1
	msgf	%r0, 0
	msgf	%r0, 1
	msgf	%r0, 524287
	msgf	%r0, 0(%r1)
	msgf	%r0, 0(%r15)
	msgf	%r0, 524287(%r1,%r15)
	msgf	%r0, 524287(%r15,%r1)
	msgf	%r15, 0

#CHECK: msgfi	%r0, -2147483648        # encoding: [0xc2,0x00,0x80,0x00,0x00,0x00]
#CHECK: msgfi	%r0, -1                 # encoding: [0xc2,0x00,0xff,0xff,0xff,0xff]
#CHECK: msgfi	%r0, 0                  # encoding: [0xc2,0x00,0x00,0x00,0x00,0x00]
#CHECK: msgfi	%r0, 1                  # encoding: [0xc2,0x00,0x00,0x00,0x00,0x01]
#CHECK: msgfi	%r0, 2147483647         # encoding: [0xc2,0x00,0x7f,0xff,0xff,0xff]
#CHECK: msgfi	%r15, 0                 # encoding: [0xc2,0xf0,0x00,0x00,0x00,0x00]

	msgfi	%r0, -1 << 31
	msgfi	%r0, -1
	msgfi	%r0, 0
	msgfi	%r0, 1
	msgfi	%r0, (1 << 31) - 1
	msgfi	%r15, 0

#CHECK: msgfr	%r0, %r0                # encoding: [0xb9,0x1c,0x00,0x00]
#CHECK: msgfr	%r0, %r15               # encoding: [0xb9,0x1c,0x00,0x0f]
#CHECK: msgfr	%r15, %r0               # encoding: [0xb9,0x1c,0x00,0xf0]
#CHECK: msgfr	%r7, %r8                # encoding: [0xb9,0x1c,0x00,0x78]

	msgfr	%r0,%r0
	msgfr	%r0,%r15
	msgfr	%r15,%r0
	msgfr	%r7,%r8

#CHECK: msgr	%r0, %r0                # encoding: [0xb9,0x0c,0x00,0x00]
#CHECK: msgr	%r0, %r15               # encoding: [0xb9,0x0c,0x00,0x0f]
#CHECK: msgr	%r15, %r0               # encoding: [0xb9,0x0c,0x00,0xf0]
#CHECK: msgr	%r7, %r8                # encoding: [0xb9,0x0c,0x00,0x78]

	msgr	%r0,%r0
	msgr	%r0,%r15
	msgr	%r15,%r0
	msgr	%r7,%r8

#CHECK: msr	%r0, %r0                # encoding: [0xb2,0x52,0x00,0x00]
#CHECK: msr	%r0, %r15               # encoding: [0xb2,0x52,0x00,0x0f]
#CHECK: msr	%r15, %r0               # encoding: [0xb2,0x52,0x00,0xf0]
#CHECK: msr	%r7, %r8                # encoding: [0xb2,0x52,0x00,0x78]

	msr	%r0,%r0
	msr	%r0,%r15
	msr	%r15,%r0
	msr	%r7,%r8

#CHECK: msta	%r0                     # encoding: [0xb2,0x47,0x00,0x00]
#CHECK: msta	%r2                     # encoding: [0xb2,0x47,0x00,0x20]
#CHECK: msta	%r14                    # encoding: [0xb2,0x47,0x00,0xe0]

	msta	%r0
	msta	%r2
	msta	%r14

#CHECK: msy	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x51]
#CHECK: msy	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x51]
#CHECK: msy	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x51]
#CHECK: msy	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x51]
#CHECK: msy	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x51]
#CHECK: msy	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x51]
#CHECK: msy	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x51]
#CHECK: msy	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x51]
#CHECK: msy	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x51]
#CHECK: msy	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x51]

	msy	%r0, -524288
	msy	%r0, -1
	msy	%r0, 0
	msy	%r0, 1
	msy	%r0, 524287
	msy	%r0, 0(%r1)
	msy	%r0, 0(%r15)
	msy	%r0, 524287(%r1,%r15)
	msy	%r0, 524287(%r15,%r1)
	msy	%r15, 0

#CHECK: mvc	0(1), 0                 # encoding: [0xd2,0x00,0x00,0x00,0x00,0x00]
#CHECK: mvc	0(1), 0(%r1)            # encoding: [0xd2,0x00,0x00,0x00,0x10,0x00]
#CHECK: mvc	0(1), 0(%r15)           # encoding: [0xd2,0x00,0x00,0x00,0xf0,0x00]
#CHECK: mvc	0(1), 4095              # encoding: [0xd2,0x00,0x00,0x00,0x0f,0xff]
#CHECK: mvc	0(1), 4095(%r1)         # encoding: [0xd2,0x00,0x00,0x00,0x1f,0xff]
#CHECK: mvc	0(1), 4095(%r15)        # encoding: [0xd2,0x00,0x00,0x00,0xff,0xff]
#CHECK: mvc	0(1,%r1), 0             # encoding: [0xd2,0x00,0x10,0x00,0x00,0x00]
#CHECK: mvc	0(1,%r15), 0            # encoding: [0xd2,0x00,0xf0,0x00,0x00,0x00]
#CHECK: mvc	4095(1,%r1), 0          # encoding: [0xd2,0x00,0x1f,0xff,0x00,0x00]
#CHECK: mvc	4095(1,%r15), 0         # encoding: [0xd2,0x00,0xff,0xff,0x00,0x00]
#CHECK: mvc	0(256,%r1), 0           # encoding: [0xd2,0xff,0x10,0x00,0x00,0x00]
#CHECK: mvc	0(256,%r15), 0          # encoding: [0xd2,0xff,0xf0,0x00,0x00,0x00]

	mvc	0(1), 0
	mvc	0(1), 0(%r1)
	mvc	0(1), 0(%r15)
	mvc	0(1), 4095
	mvc	0(1), 4095(%r1)
	mvc	0(1), 4095(%r15)
	mvc	0(1,%r1), 0
	mvc	0(1,%r15), 0
	mvc	4095(1,%r1), 0
	mvc	4095(1,%r15), 0
	mvc	0(256,%r1), 0
	mvc	0(256,%r15), 0

#CHECK: mvcdk	0, 0                    # encoding: [0xe5,0x0f,0x00,0x00,0x00,0x00]
#CHECK: mvcdk	0(%r1), 0(%r2)          # encoding: [0xe5,0x0f,0x10,0x00,0x20,0x00]
#CHECK: mvcdk	160(%r1), 320(%r15)     # encoding: [0xe5,0x0f,0x10,0xa0,0xf1,0x40]
#CHECK: mvcdk	0(%r1), 4095            # encoding: [0xe5,0x0f,0x10,0x00,0x0f,0xff]
#CHECK: mvcdk	0(%r1), 4095(%r2)       # encoding: [0xe5,0x0f,0x10,0x00,0x2f,0xff]
#CHECK: mvcdk	0(%r1), 4095(%r15)      # encoding: [0xe5,0x0f,0x10,0x00,0xff,0xff]
#CHECK: mvcdk	0(%r1), 0               # encoding: [0xe5,0x0f,0x10,0x00,0x00,0x00]
#CHECK: mvcdk	0(%r15), 0              # encoding: [0xe5,0x0f,0xf0,0x00,0x00,0x00]
#CHECK: mvcdk	4095(%r1), 0            # encoding: [0xe5,0x0f,0x1f,0xff,0x00,0x00]
#CHECK: mvcdk	4095(%r15), 0           # encoding: [0xe5,0x0f,0xff,0xff,0x00,0x00]

	mvcdk	0, 0
	mvcdk	0(%r1), 0(%r2)
	mvcdk	160(%r1), 320(%r15)
	mvcdk	0(%r1), 4095
	mvcdk	0(%r1), 4095(%r2)
	mvcdk	0(%r1), 4095(%r15)
	mvcdk	0(%r1), 0
	mvcdk	0(%r15), 0
	mvcdk	4095(%r1), 0
	mvcdk	4095(%r15), 0

#CHECK: mvcin	0(1), 0                 # encoding: [0xe8,0x00,0x00,0x00,0x00,0x00]
#CHECK: mvcin	0(1), 0(%r1)            # encoding: [0xe8,0x00,0x00,0x00,0x10,0x00]
#CHECK: mvcin	0(1), 0(%r15)           # encoding: [0xe8,0x00,0x00,0x00,0xf0,0x00]
#CHECK: mvcin	0(1), 4095              # encoding: [0xe8,0x00,0x00,0x00,0x0f,0xff]
#CHECK: mvcin	0(1), 4095(%r1)         # encoding: [0xe8,0x00,0x00,0x00,0x1f,0xff]
#CHECK: mvcin	0(1), 4095(%r15)        # encoding: [0xe8,0x00,0x00,0x00,0xff,0xff]
#CHECK: mvcin	0(1,%r1), 0             # encoding: [0xe8,0x00,0x10,0x00,0x00,0x00]
#CHECK: mvcin	0(1,%r15), 0            # encoding: [0xe8,0x00,0xf0,0x00,0x00,0x00]
#CHECK: mvcin	4095(1,%r1), 0          # encoding: [0xe8,0x00,0x1f,0xff,0x00,0x00]
#CHECK: mvcin	4095(1,%r15), 0         # encoding: [0xe8,0x00,0xff,0xff,0x00,0x00]
#CHECK: mvcin	0(256,%r1), 0           # encoding: [0xe8,0xff,0x10,0x00,0x00,0x00]
#CHECK: mvcin	0(256,%r15), 0          # encoding: [0xe8,0xff,0xf0,0x00,0x00,0x00]

	mvcin	0(1), 0
	mvcin	0(1), 0(%r1)
	mvcin	0(1), 0(%r15)
	mvcin	0(1), 4095
	mvcin	0(1), 4095(%r1)
	mvcin	0(1), 4095(%r15)
	mvcin	0(1,%r1), 0
	mvcin	0(1,%r15), 0
	mvcin	4095(1,%r1), 0
	mvcin	4095(1,%r15), 0
	mvcin	0(256,%r1), 0
	mvcin	0(256,%r15), 0

#CHECK: mvck	0(%r0), 0, %r3          # encoding: [0xd9,0x03,0x00,0x00,0x00,0x00]
#CHECK: mvck	0(%r1), 0, %r3          # encoding: [0xd9,0x13,0x00,0x00,0x00,0x00]
#CHECK: mvck	0(%r1), 0(%r1), %r3     # encoding: [0xd9,0x13,0x00,0x00,0x10,0x00]
#CHECK: mvck	0(%r1), 0(%r15), %r3    # encoding: [0xd9,0x13,0x00,0x00,0xf0,0x00]
#CHECK: mvck	0(%r1), 4095, %r3       # encoding: [0xd9,0x13,0x00,0x00,0x0f,0xff]
#CHECK: mvck	0(%r1), 4095(%r1), %r3  # encoding: [0xd9,0x13,0x00,0x00,0x1f,0xff]
#CHECK: mvck	0(%r1), 4095(%r15), %r3 # encoding: [0xd9,0x13,0x00,0x00,0xff,0xff]
#CHECK: mvck	0(%r2,%r1), 0, %r3      # encoding: [0xd9,0x23,0x10,0x00,0x00,0x00]
#CHECK: mvck	0(%r2,%r15), 0, %r3     # encoding: [0xd9,0x23,0xf0,0x00,0x00,0x00]
#CHECK: mvck	4095(%r2,%r1), 0, %r3   # encoding: [0xd9,0x23,0x1f,0xff,0x00,0x00]
#CHECK: mvck	4095(%r2,%r15), 0, %r3  # encoding: [0xd9,0x23,0xff,0xff,0x00,0x00]
#CHECK: mvck	0(%r2,%r1), 0, %r3      # encoding: [0xd9,0x23,0x10,0x00,0x00,0x00]
#CHECK: mvck	0(%r2,%r15), 0, %r3     # encoding: [0xd9,0x23,0xf0,0x00,0x00,0x00]

        mvck	0(%r0), 0, %r3
        mvck	0(%r1), 0, %r3
        mvck	0(%r1), 0(%r1), %r3
        mvck	0(%r1), 0(%r15), %r3
        mvck	0(%r1), 4095, %r3
        mvck	0(%r1), 4095(%r1), %r3
        mvck	0(%r1), 4095(%r15), %r3
        mvck	0(%r2,%r1), 0, %r3
        mvck	0(%r2,%r15), 0, %r3
        mvck	4095(%r2,%r1), 0, %r3
        mvck	4095(%r2,%r15), 0, %r3
        mvck	0(%r2,%r1), 0, %r3
        mvck	0(%r2,%r15), 0, %r3

#CHECK: mvcl	%r0, %r8                # encoding: [0x0e,0x08]
#CHECK: mvcl	%r0, %r14               # encoding: [0x0e,0x0e]
#CHECK: mvcl	%r14, %r0               # encoding: [0x0e,0xe0]
#CHECK: mvcl	%r14, %r8               # encoding: [0x0e,0xe8]

	mvcl	%r0, %r8
	mvcl	%r0, %r14
	mvcl	%r14, %r0
	mvcl	%r14, %r8

#CHECK: mvcle	%r0, %r0, 0             # encoding: [0xa8,0x00,0x00,0x00]
#CHECK: mvcle	%r0, %r14, 4095         # encoding: [0xa8,0x0e,0x0f,0xff]
#CHECK: mvcle	%r0, %r0, 0(%r1)        # encoding: [0xa8,0x00,0x10,0x00]
#CHECK: mvcle	%r0, %r0, 0(%r15)       # encoding: [0xa8,0x00,0xf0,0x00]
#CHECK: mvcle	%r14, %r14, 4095(%r1)   # encoding: [0xa8,0xee,0x1f,0xff]
#CHECK: mvcle	%r0, %r0, 4095(%r15)    # encoding: [0xa8,0x00,0xff,0xff]
#CHECK: mvcle	%r14, %r0, 0            # encoding: [0xa8,0xe0,0x00,0x00]

	mvcle	%r0, %r0, 0
	mvcle	%r0, %r14, 4095
	mvcle	%r0, %r0, 0(%r1)
	mvcle	%r0, %r0, 0(%r15)
	mvcle	%r14, %r14, 4095(%r1)
	mvcle	%r0, %r0, 4095(%r15)
	mvcle	%r14, %r0, 0

#CHECK: mvclu	%r0, %r0, -524288       # encoding: [0xeb,0x00,0x00,0x00,0x80,0x8e]
#CHECK: mvclu	%r0, %r0, -1            # encoding: [0xeb,0x00,0x0f,0xff,0xff,0x8e]
#CHECK: mvclu	%r0, %r14, 0            # encoding: [0xeb,0x0e,0x00,0x00,0x00,0x8e]
#CHECK: mvclu	%r0, %r14, 1            # encoding: [0xeb,0x0e,0x00,0x01,0x00,0x8e]
#CHECK: mvclu	%r0, %r8, 524287        # encoding: [0xeb,0x08,0x0f,0xff,0x7f,0x8e]
#CHECK: mvclu	%r0, %r8, 0(%r1)        # encoding: [0xeb,0x08,0x10,0x00,0x00,0x8e]
#CHECK: mvclu	%r0, %r4, 0(%r15)       # encoding: [0xeb,0x04,0xf0,0x00,0x00,0x8e]
#CHECK: mvclu	%r0, %r4, 524287(%r15)  # encoding: [0xeb,0x04,0xff,0xff,0x7f,0x8e]
#CHECK: mvclu	%r0, %r0, 524287(%r1)   # encoding: [0xeb,0x00,0x1f,0xff,0x7f,0x8e]
#CHECK: mvclu	%r14, %r0, 0            # encoding: [0xeb,0xe0,0x00,0x00,0x00,0x8e]

	mvclu	%r0, %r0, -524288
	mvclu	%r0, %r0, -1
	mvclu	%r0, %r14, 0
	mvclu	%r0, %r14, 1
	mvclu	%r0, %r8, 524287
	mvclu	%r0, %r8, 0(%r1)
	mvclu	%r0, %r4, 0(%r15)
	mvclu	%r0, %r4, 524287(%r15)
	mvclu	%r0, %r0, 524287(%r1)
	mvclu	%r14, %r0, 0

#CHECK: mvcos	0, 0, %r0               # encoding: [0xc8,0x00,0x00,0x00,0x00,0x00]
#CHECK: mvcos	0(%r1), 0(%r15), %r2    # encoding: [0xc8,0x20,0x10,0x00,0xf0,0x00]
#CHECK: mvcos	1(%r1), 0(%r15), %r2    # encoding: [0xc8,0x20,0x10,0x01,0xf0,0x00]
#CHECK: mvcos	4095(%r1), 0(%r15), %r2 # encoding: [0xc8,0x20,0x1f,0xff,0xf0,0x00]
#CHECK: mvcos	0(%r1), 1(%r15), %r2    # encoding: [0xc8,0x20,0x10,0x00,0xf0,0x01]
#CHECK: mvcos	0(%r1), 4095(%r15), %r2 # encoding: [0xc8,0x20,0x10,0x00,0xff,0xff]

        mvcos	0, 0, %r0
        mvcos	0(%r1), 0(%r15), %r2
        mvcos	1(%r1), 0(%r15), %r2
        mvcos	4095(%r1), 0(%r15), %r2
        mvcos	0(%r1), 1(%r15), %r2
        mvcos	0(%r1), 4095(%r15), %r2

#CHECK: mvcp	0(%r0), 0, %r3          # encoding: [0xda,0x03,0x00,0x00,0x00,0x00]
#CHECK: mvcp	0(%r1), 0, %r3          # encoding: [0xda,0x13,0x00,0x00,0x00,0x00]
#CHECK: mvcp	0(%r1), 0(%r1), %r3     # encoding: [0xda,0x13,0x00,0x00,0x10,0x00]
#CHECK: mvcp	0(%r1), 0(%r15), %r3    # encoding: [0xda,0x13,0x00,0x00,0xf0,0x00]
#CHECK: mvcp	0(%r1), 4095, %r3       # encoding: [0xda,0x13,0x00,0x00,0x0f,0xff]
#CHECK: mvcp	0(%r1), 4095(%r1), %r3  # encoding: [0xda,0x13,0x00,0x00,0x1f,0xff]
#CHECK: mvcp	0(%r1), 4095(%r15), %r3 # encoding: [0xda,0x13,0x00,0x00,0xff,0xff]
#CHECK: mvcp	0(%r2,%r1), 0, %r3      # encoding: [0xda,0x23,0x10,0x00,0x00,0x00]
#CHECK: mvcp	0(%r2,%r15), 0, %r3     # encoding: [0xda,0x23,0xf0,0x00,0x00,0x00]
#CHECK: mvcp	4095(%r2,%r1), 0, %r3   # encoding: [0xda,0x23,0x1f,0xff,0x00,0x00]
#CHECK: mvcp	4095(%r2,%r15), 0, %r3  # encoding: [0xda,0x23,0xff,0xff,0x00,0x00]
#CHECK: mvcp	0(%r2,%r1), 0, %r3      # encoding: [0xda,0x23,0x10,0x00,0x00,0x00]
#CHECK: mvcp	0(%r2,%r15), 0, %r3     # encoding: [0xda,0x23,0xf0,0x00,0x00,0x00]

        mvcp	0(%r0), 0, %r3
        mvcp	0(%r1), 0, %r3
        mvcp	0(%r1), 0(%r1), %r3
        mvcp	0(%r1), 0(%r15), %r3
        mvcp	0(%r1), 4095, %r3
        mvcp	0(%r1), 4095(%r1), %r3
        mvcp	0(%r1), 4095(%r15), %r3
        mvcp	0(%r2,%r1), 0, %r3
        mvcp	0(%r2,%r15), 0, %r3
        mvcp	4095(%r2,%r1), 0, %r3
        mvcp	4095(%r2,%r15), 0, %r3
        mvcp	0(%r2,%r1), 0, %r3
        mvcp	0(%r2,%r15), 0, %r3

#CHECK: mvcs	0(%r0), 0, %r3          # encoding: [0xdb,0x03,0x00,0x00,0x00,0x00]
#CHECK: mvcs	0(%r1), 0, %r3          # encoding: [0xdb,0x13,0x00,0x00,0x00,0x00]
#CHECK: mvcs	0(%r1), 0(%r1), %r3     # encoding: [0xdb,0x13,0x00,0x00,0x10,0x00]
#CHECK: mvcs	0(%r1), 0(%r15), %r3    # encoding: [0xdb,0x13,0x00,0x00,0xf0,0x00]
#CHECK: mvcs	0(%r1), 4095, %r3       # encoding: [0xdb,0x13,0x00,0x00,0x0f,0xff]
#CHECK: mvcs	0(%r1), 4095(%r1), %r3  # encoding: [0xdb,0x13,0x00,0x00,0x1f,0xff]
#CHECK: mvcs	0(%r1), 4095(%r15), %r3 # encoding: [0xdb,0x13,0x00,0x00,0xff,0xff]
#CHECK: mvcs	0(%r2,%r1), 0, %r3      # encoding: [0xdb,0x23,0x10,0x00,0x00,0x00]
#CHECK: mvcs	0(%r2,%r15), 0, %r3     # encoding: [0xdb,0x23,0xf0,0x00,0x00,0x00]
#CHECK: mvcs	4095(%r2,%r1), 0, %r3   # encoding: [0xdb,0x23,0x1f,0xff,0x00,0x00]
#CHECK: mvcs	4095(%r2,%r15), 0, %r3  # encoding: [0xdb,0x23,0xff,0xff,0x00,0x00]
#CHECK: mvcs	0(%r2,%r1), 0, %r3      # encoding: [0xdb,0x23,0x10,0x00,0x00,0x00]
#CHECK: mvcs	0(%r2,%r15), 0, %r3     # encoding: [0xdb,0x23,0xf0,0x00,0x00,0x00]

        mvcs	0(%r0), 0, %r3
        mvcs	0(%r1), 0, %r3
        mvcs	0(%r1), 0(%r1), %r3
        mvcs	0(%r1), 0(%r15), %r3
        mvcs	0(%r1), 4095, %r3
        mvcs	0(%r1), 4095(%r1), %r3
        mvcs	0(%r1), 4095(%r15), %r3
        mvcs	0(%r2,%r1), 0, %r3
        mvcs	0(%r2,%r15), 0, %r3
        mvcs	4095(%r2,%r1), 0, %r3
        mvcs	4095(%r2,%r15), 0, %r3
        mvcs	0(%r2,%r1), 0, %r3
        mvcs	0(%r2,%r15), 0, %r3

#CHECK: mvcsk	0, 0                    # encoding: [0xe5,0x0e,0x00,0x00,0x00,0x00]
#CHECK: mvcsk	0(%r1), 0(%r2)          # encoding: [0xe5,0x0e,0x10,0x00,0x20,0x00]
#CHECK: mvcsk	160(%r1), 320(%r15)     # encoding: [0xe5,0x0e,0x10,0xa0,0xf1,0x40]
#CHECK: mvcsk	0(%r1), 4095            # encoding: [0xe5,0x0e,0x10,0x00,0x0f,0xff]
#CHECK: mvcsk	0(%r1), 4095(%r2)       # encoding: [0xe5,0x0e,0x10,0x00,0x2f,0xff]
#CHECK: mvcsk	0(%r1), 4095(%r15)      # encoding: [0xe5,0x0e,0x10,0x00,0xff,0xff]
#CHECK: mvcsk	0(%r1), 0               # encoding: [0xe5,0x0e,0x10,0x00,0x00,0x00]
#CHECK: mvcsk	0(%r15), 0              # encoding: [0xe5,0x0e,0xf0,0x00,0x00,0x00]
#CHECK: mvcsk	4095(%r1), 0            # encoding: [0xe5,0x0e,0x1f,0xff,0x00,0x00]
#CHECK: mvcsk	4095(%r15), 0           # encoding: [0xe5,0x0e,0xff,0xff,0x00,0x00]

	mvcsk	0, 0
	mvcsk	0(%r1), 0(%r2)
	mvcsk	160(%r1), 320(%r15)
	mvcsk	0(%r1), 4095
	mvcsk	0(%r1), 4095(%r2)
	mvcsk	0(%r1), 4095(%r15)
	mvcsk	0(%r1), 0
	mvcsk	0(%r15), 0
	mvcsk	4095(%r1), 0
	mvcsk	4095(%r15), 0

#CHECK: mvghi	0, 0                    # encoding: [0xe5,0x48,0x00,0x00,0x00,0x00]
#CHECK: mvghi	4095, 0                 # encoding: [0xe5,0x48,0x0f,0xff,0x00,0x00]
#CHECK: mvghi	0, -32768               # encoding: [0xe5,0x48,0x00,0x00,0x80,0x00]
#CHECK: mvghi	0, -1                   # encoding: [0xe5,0x48,0x00,0x00,0xff,0xff]
#CHECK: mvghi	0, 0                    # encoding: [0xe5,0x48,0x00,0x00,0x00,0x00]
#CHECK: mvghi	0, 1                    # encoding: [0xe5,0x48,0x00,0x00,0x00,0x01]
#CHECK: mvghi	0, 32767                # encoding: [0xe5,0x48,0x00,0x00,0x7f,0xff]
#CHECK: mvghi	0(%r1), 42              # encoding: [0xe5,0x48,0x10,0x00,0x00,0x2a]
#CHECK: mvghi	0(%r15), 42             # encoding: [0xe5,0x48,0xf0,0x00,0x00,0x2a]
#CHECK: mvghi	4095(%r1), 42           # encoding: [0xe5,0x48,0x1f,0xff,0x00,0x2a]
#CHECK: mvghi	4095(%r15), 42          # encoding: [0xe5,0x48,0xff,0xff,0x00,0x2a]

	mvghi	0, 0
	mvghi	4095, 0
	mvghi	0, -32768
	mvghi	0, -1
	mvghi	0, 0
	mvghi	0, 1
	mvghi	0, 32767
	mvghi	0(%r1), 42
	mvghi	0(%r15), 42
	mvghi	4095(%r1), 42
	mvghi	4095(%r15), 42

#CHECK: mvhhi	0, 0                    # encoding: [0xe5,0x44,0x00,0x00,0x00,0x00]
#CHECK: mvhhi	4095, 0                 # encoding: [0xe5,0x44,0x0f,0xff,0x00,0x00]
#CHECK: mvhhi	0, -32768               # encoding: [0xe5,0x44,0x00,0x00,0x80,0x00]
#CHECK: mvhhi	0, -1                   # encoding: [0xe5,0x44,0x00,0x00,0xff,0xff]
#CHECK: mvhhi	0, 0                    # encoding: [0xe5,0x44,0x00,0x00,0x00,0x00]
#CHECK: mvhhi	0, 1                    # encoding: [0xe5,0x44,0x00,0x00,0x00,0x01]
#CHECK: mvhhi	0, 32767                # encoding: [0xe5,0x44,0x00,0x00,0x7f,0xff]
#CHECK: mvhhi	0(%r1), 42              # encoding: [0xe5,0x44,0x10,0x00,0x00,0x2a]
#CHECK: mvhhi	0(%r15), 42             # encoding: [0xe5,0x44,0xf0,0x00,0x00,0x2a]
#CHECK: mvhhi	4095(%r1), 42           # encoding: [0xe5,0x44,0x1f,0xff,0x00,0x2a]
#CHECK: mvhhi	4095(%r15), 42          # encoding: [0xe5,0x44,0xff,0xff,0x00,0x2a]

	mvhhi	0, 0
	mvhhi	4095, 0
	mvhhi	0, -32768
	mvhhi	0, -1
	mvhhi	0, 0
	mvhhi	0, 1
	mvhhi	0, 32767
	mvhhi	0(%r1), 42
	mvhhi	0(%r15), 42
	mvhhi	4095(%r1), 42
	mvhhi	4095(%r15), 42

#CHECK: mvhi	0, 0                    # encoding: [0xe5,0x4c,0x00,0x00,0x00,0x00]
#CHECK: mvhi	4095, 0                 # encoding: [0xe5,0x4c,0x0f,0xff,0x00,0x00]
#CHECK: mvhi	0, -32768               # encoding: [0xe5,0x4c,0x00,0x00,0x80,0x00]
#CHECK: mvhi	0, -1                   # encoding: [0xe5,0x4c,0x00,0x00,0xff,0xff]
#CHECK: mvhi	0, 0                    # encoding: [0xe5,0x4c,0x00,0x00,0x00,0x00]
#CHECK: mvhi	0, 1                    # encoding: [0xe5,0x4c,0x00,0x00,0x00,0x01]
#CHECK: mvhi	0, 32767                # encoding: [0xe5,0x4c,0x00,0x00,0x7f,0xff]
#CHECK: mvhi	0(%r1), 42              # encoding: [0xe5,0x4c,0x10,0x00,0x00,0x2a]
#CHECK: mvhi	0(%r15), 42             # encoding: [0xe5,0x4c,0xf0,0x00,0x00,0x2a]
#CHECK: mvhi	4095(%r1), 42           # encoding: [0xe5,0x4c,0x1f,0xff,0x00,0x2a]
#CHECK: mvhi	4095(%r15), 42          # encoding: [0xe5,0x4c,0xff,0xff,0x00,0x2a]

	mvhi	0, 0
	mvhi	4095, 0
	mvhi	0, -32768
	mvhi	0, -1
	mvhi	0, 0
	mvhi	0, 1
	mvhi	0, 32767
	mvhi	0(%r1), 42
	mvhi	0(%r15), 42
	mvhi	4095(%r1), 42
	mvhi	4095(%r15), 42

#CHECK: mvi	0, 0                    # encoding: [0x92,0x00,0x00,0x00]
#CHECK: mvi	4095, 0                 # encoding: [0x92,0x00,0x0f,0xff]
#CHECK: mvi	0, 255                  # encoding: [0x92,0xff,0x00,0x00]
#CHECK: mvi	0(%r1), 42              # encoding: [0x92,0x2a,0x10,0x00]
#CHECK: mvi	0(%r15), 42             # encoding: [0x92,0x2a,0xf0,0x00]
#CHECK: mvi	4095(%r1), 42           # encoding: [0x92,0x2a,0x1f,0xff]
#CHECK: mvi	4095(%r15), 42          # encoding: [0x92,0x2a,0xff,0xff]

	mvi	0, 0
	mvi	4095, 0
	mvi	0, 255
	mvi	0(%r1), 42
	mvi	0(%r15), 42
	mvi	4095(%r1), 42
	mvi	4095(%r15), 42

#CHECK: mviy	-524288, 0              # encoding: [0xeb,0x00,0x00,0x00,0x80,0x52]
#CHECK: mviy	-1, 0                   # encoding: [0xeb,0x00,0x0f,0xff,0xff,0x52]
#CHECK: mviy	0, 0                    # encoding: [0xeb,0x00,0x00,0x00,0x00,0x52]
#CHECK: mviy	1, 0                    # encoding: [0xeb,0x00,0x00,0x01,0x00,0x52]
#CHECK: mviy	524287, 0               # encoding: [0xeb,0x00,0x0f,0xff,0x7f,0x52]
#CHECK: mviy	0, 255                  # encoding: [0xeb,0xff,0x00,0x00,0x00,0x52]
#CHECK: mviy	0(%r1), 42              # encoding: [0xeb,0x2a,0x10,0x00,0x00,0x52]
#CHECK: mviy	0(%r15), 42             # encoding: [0xeb,0x2a,0xf0,0x00,0x00,0x52]
#CHECK: mviy	524287(%r1), 42         # encoding: [0xeb,0x2a,0x1f,0xff,0x7f,0x52]
#CHECK: mviy	524287(%r15), 42        # encoding: [0xeb,0x2a,0xff,0xff,0x7f,0x52]

	mviy	-524288, 0
	mviy	-1, 0
	mviy	0, 0
	mviy	1, 0
	mviy	524287, 0
	mviy	0, 255
	mviy	0(%r1), 42
	mviy	0(%r15), 42
	mviy	524287(%r1), 42
	mviy	524287(%r15), 42

#CHECK: mvn	0(1), 0                 # encoding: [0xd1,0x00,0x00,0x00,0x00,0x00]
#CHECK: mvn	0(1), 0(%r1)            # encoding: [0xd1,0x00,0x00,0x00,0x10,0x00]
#CHECK: mvn	0(1), 0(%r15)           # encoding: [0xd1,0x00,0x00,0x00,0xf0,0x00]
#CHECK: mvn	0(1), 4095              # encoding: [0xd1,0x00,0x00,0x00,0x0f,0xff]
#CHECK: mvn	0(1), 4095(%r1)         # encoding: [0xd1,0x00,0x00,0x00,0x1f,0xff]
#CHECK: mvn	0(1), 4095(%r15)        # encoding: [0xd1,0x00,0x00,0x00,0xff,0xff]
#CHECK: mvn	0(1,%r1), 0             # encoding: [0xd1,0x00,0x10,0x00,0x00,0x00]
#CHECK: mvn	0(1,%r15), 0            # encoding: [0xd1,0x00,0xf0,0x00,0x00,0x00]
#CHECK: mvn	4095(1,%r1), 0          # encoding: [0xd1,0x00,0x1f,0xff,0x00,0x00]
#CHECK: mvn	4095(1,%r15), 0         # encoding: [0xd1,0x00,0xff,0xff,0x00,0x00]
#CHECK: mvn	0(256,%r1), 0           # encoding: [0xd1,0xff,0x10,0x00,0x00,0x00]
#CHECK: mvn	0(256,%r15), 0          # encoding: [0xd1,0xff,0xf0,0x00,0x00,0x00]

	mvn	0(1), 0
	mvn	0(1), 0(%r1)
	mvn	0(1), 0(%r15)
	mvn	0(1), 4095
	mvn	0(1), 4095(%r1)
	mvn	0(1), 4095(%r15)
	mvn	0(1,%r1), 0
	mvn	0(1,%r15), 0
	mvn	4095(1,%r1), 0
	mvn	4095(1,%r15), 0
	mvn	0(256,%r1), 0
	mvn	0(256,%r15), 0

#CHECK: mvo	0(1), 0(1)              # encoding: [0xf1,0x00,0x00,0x00,0x00,0x00]
#CHECK: mvo	0(1), 0(1,%r1)          # encoding: [0xf1,0x00,0x00,0x00,0x10,0x00]
#CHECK: mvo	0(1), 0(1,%r15)         # encoding: [0xf1,0x00,0x00,0x00,0xf0,0x00]
#CHECK: mvo	0(1), 4095(1)           # encoding: [0xf1,0x00,0x00,0x00,0x0f,0xff]
#CHECK: mvo	0(1), 4095(1,%r1)       # encoding: [0xf1,0x00,0x00,0x00,0x1f,0xff]
#CHECK: mvo	0(1), 4095(1,%r15)      # encoding: [0xf1,0x00,0x00,0x00,0xff,0xff]
#CHECK: mvo	0(1,%r1), 0(1)          # encoding: [0xf1,0x00,0x10,0x00,0x00,0x00]
#CHECK: mvo	0(1,%r15), 0(1)         # encoding: [0xf1,0x00,0xf0,0x00,0x00,0x00]
#CHECK: mvo	4095(1,%r1), 0(1)       # encoding: [0xf1,0x00,0x1f,0xff,0x00,0x00]
#CHECK: mvo	4095(1,%r15), 0(1)      # encoding: [0xf1,0x00,0xff,0xff,0x00,0x00]
#CHECK: mvo	0(16,%r1), 0(1)         # encoding: [0xf1,0xf0,0x10,0x00,0x00,0x00]
#CHECK: mvo	0(16,%r15), 0(1)        # encoding: [0xf1,0xf0,0xf0,0x00,0x00,0x00]
#CHECK: mvo	0(1), 0(16,%r1)         # encoding: [0xf1,0x0f,0x00,0x00,0x10,0x00]
#CHECK: mvo	0(1), 0(16,%r15)        # encoding: [0xf1,0x0f,0x00,0x00,0xf0,0x00]

	mvo	0(1), 0(1)
	mvo	0(1), 0(1,%r1)
	mvo	0(1), 0(1,%r15)
	mvo	0(1), 4095(1)
	mvo	0(1), 4095(1,%r1)
	mvo	0(1), 4095(1,%r15)
	mvo	0(1,%r1), 0(1)
	mvo	0(1,%r15), 0(1)
	mvo	4095(1,%r1), 0(1)
	mvo	4095(1,%r15), 0(1)
	mvo	0(16,%r1), 0(1)
	mvo	0(16,%r15), 0(1)
	mvo	0(1), 0(16,%r1)
	mvo	0(1), 0(16,%r15)

#CHECK: mvpg	%r0, %r0                # encoding: [0xb2,0x54,0x00,0x00]
#CHECK: mvpg	%r0, %r15               # encoding: [0xb2,0x54,0x00,0x0f]
#CHECK: mvpg	%r15, %r0               # encoding: [0xb2,0x54,0x00,0xf0]
#CHECK: mvpg	%r7, %r8                # encoding: [0xb2,0x54,0x00,0x78]

	mvpg	%r0,%r0
	mvpg	%r0,%r15
	mvpg	%r15,%r0
	mvpg	%r7,%r8

#CHECK: mvst	%r0, %r0                # encoding: [0xb2,0x55,0x00,0x00]
#CHECK: mvst	%r0, %r15               # encoding: [0xb2,0x55,0x00,0x0f]
#CHECK: mvst	%r15, %r0               # encoding: [0xb2,0x55,0x00,0xf0]
#CHECK: mvst	%r7, %r8                # encoding: [0xb2,0x55,0x00,0x78]

	mvst	%r0,%r0
	mvst	%r0,%r15
	mvst	%r15,%r0
	mvst	%r7,%r8

#CHECK: mvz	0(1), 0                 # encoding: [0xd3,0x00,0x00,0x00,0x00,0x00]
#CHECK: mvz	0(1), 0(%r1)            # encoding: [0xd3,0x00,0x00,0x00,0x10,0x00]
#CHECK: mvz	0(1), 0(%r15)           # encoding: [0xd3,0x00,0x00,0x00,0xf0,0x00]
#CHECK: mvz	0(1), 4095              # encoding: [0xd3,0x00,0x00,0x00,0x0f,0xff]
#CHECK: mvz	0(1), 4095(%r1)         # encoding: [0xd3,0x00,0x00,0x00,0x1f,0xff]
#CHECK: mvz	0(1), 4095(%r15)        # encoding: [0xd3,0x00,0x00,0x00,0xff,0xff]
#CHECK: mvz	0(1,%r1), 0             # encoding: [0xd3,0x00,0x10,0x00,0x00,0x00]
#CHECK: mvz	0(1,%r15), 0            # encoding: [0xd3,0x00,0xf0,0x00,0x00,0x00]
#CHECK: mvz	4095(1,%r1), 0          # encoding: [0xd3,0x00,0x1f,0xff,0x00,0x00]
#CHECK: mvz	4095(1,%r15), 0         # encoding: [0xd3,0x00,0xff,0xff,0x00,0x00]
#CHECK: mvz	0(256,%r1), 0           # encoding: [0xd3,0xff,0x10,0x00,0x00,0x00]
#CHECK: mvz	0(256,%r15), 0          # encoding: [0xd3,0xff,0xf0,0x00,0x00,0x00]

	mvz	0(1), 0
	mvz	0(1), 0(%r1)
	mvz	0(1), 0(%r15)
	mvz	0(1), 4095
	mvz	0(1), 4095(%r1)
	mvz	0(1), 4095(%r15)
	mvz	0(1,%r1), 0
	mvz	0(1,%r15), 0
	mvz	4095(1,%r1), 0
	mvz	4095(1,%r15), 0
	mvz	0(256,%r1), 0
	mvz	0(256,%r15), 0

#CHECK: mxbr	%f0, %f0                # encoding: [0xb3,0x4c,0x00,0x00]
#CHECK: mxbr	%f0, %f13               # encoding: [0xb3,0x4c,0x00,0x0d]
#CHECK: mxbr	%f8, %f5                # encoding: [0xb3,0x4c,0x00,0x85]
#CHECK: mxbr	%f13, %f13              # encoding: [0xb3,0x4c,0x00,0xdd]

	mxbr	%f0, %f0
	mxbr	%f0, %f13
	mxbr	%f8, %f5
	mxbr	%f13, %f13

#CHECK: mxd	%f0, 0                  # encoding: [0x67,0x00,0x00,0x00]
#CHECK: mxd	%f0, 4095               # encoding: [0x67,0x00,0x0f,0xff]
#CHECK: mxd	%f0, 0(%r1)             # encoding: [0x67,0x00,0x10,0x00]
#CHECK: mxd	%f0, 0(%r15)            # encoding: [0x67,0x00,0xf0,0x00]
#CHECK: mxd	%f0, 4095(%r1,%r15)     # encoding: [0x67,0x01,0xff,0xff]
#CHECK: mxd	%f0, 4095(%r15,%r1)     # encoding: [0x67,0x0f,0x1f,0xff]
#CHECK: mxd	%f13, 0                 # encoding: [0x67,0xd0,0x00,0x00]

	mxd	%f0, 0
	mxd	%f0, 4095
	mxd	%f0, 0(%r1)
	mxd	%f0, 0(%r15)
	mxd	%f0, 4095(%r1,%r15)
	mxd	%f0, 4095(%r15,%r1)
	mxd	%f13, 0

#CHECK: mxdb	%f0, 0                  # encoding: [0xed,0x00,0x00,0x00,0x00,0x07]
#CHECK: mxdb	%f0, 4095               # encoding: [0xed,0x00,0x0f,0xff,0x00,0x07]
#CHECK: mxdb	%f0, 0(%r1)             # encoding: [0xed,0x00,0x10,0x00,0x00,0x07]
#CHECK: mxdb	%f0, 0(%r15)            # encoding: [0xed,0x00,0xf0,0x00,0x00,0x07]
#CHECK: mxdb	%f0, 4095(%r1,%r15)     # encoding: [0xed,0x01,0xff,0xff,0x00,0x07]
#CHECK: mxdb	%f0, 4095(%r15,%r1)     # encoding: [0xed,0x0f,0x1f,0xff,0x00,0x07]
#CHECK: mxdb	%f13, 0                 # encoding: [0xed,0xd0,0x00,0x00,0x00,0x07]

	mxdb	%f0, 0
	mxdb	%f0, 4095
	mxdb	%f0, 0(%r1)
	mxdb	%f0, 0(%r15)
	mxdb	%f0, 4095(%r1,%r15)
	mxdb	%f0, 4095(%r15,%r1)
	mxdb	%f13, 0

#CHECK: mxdbr	%f0, %f0                # encoding: [0xb3,0x07,0x00,0x00]
#CHECK: mxdbr	%f0, %f15               # encoding: [0xb3,0x07,0x00,0x0f]
#CHECK: mxdbr	%f8, %f8                # encoding: [0xb3,0x07,0x00,0x88]
#CHECK: mxdbr	%f13, %f0               # encoding: [0xb3,0x07,0x00,0xd0]

	mxdbr	%f0, %f0
	mxdbr	%f0, %f15
	mxdbr	%f8, %f8
	mxdbr	%f13, %f0

#CHECK: mxdr	%f0, %f0                # encoding: [0x27,0x00]
#CHECK: mxdr	%f0, %f15               # encoding: [0x27,0x0f]
#CHECK: mxdr	%f8, %f8                # encoding: [0x27,0x88]
#CHECK: mxdr	%f13, %f0               # encoding: [0x27,0xd0]

	mxdr	%f0, %f0
	mxdr	%f0, %f15
	mxdr	%f8, %f8
	mxdr	%f13, %f0

#CHECK: mxr	%f0, %f0                # encoding: [0x26,0x00]
#CHECK: mxr	%f0, %f13               # encoding: [0x26,0x0d]
#CHECK: mxr	%f8, %f5                # encoding: [0x26,0x85]
#CHECK: mxr	%f13, %f13              # encoding: [0x26,0xdd]

	mxr	%f0, %f0
	mxr	%f0, %f13
	mxr	%f8, %f5
	mxr	%f13, %f13

#CHECK: mxtr	%f0, %f0, %f0           # encoding: [0xb3,0xd8,0x00,0x00]
#CHECK: mxtr	%f0, %f0, %f13          # encoding: [0xb3,0xd8,0xd0,0x00]
#CHECK: mxtr	%f0, %f13, %f0          # encoding: [0xb3,0xd8,0x00,0x0d]
#CHECK: mxtr	%f13, %f0, %f0          # encoding: [0xb3,0xd8,0x00,0xd0]
#CHECK: mxtr	%f8, %f8, %f8           # encoding: [0xb3,0xd8,0x80,0x88]

	mxtr	%f0, %f0, %f0
	mxtr	%f0, %f0, %f13
	mxtr	%f0, %f13, %f0
	mxtr	%f13, %f0, %f0
	mxtr	%f8, %f8, %f8

#CHECK: my	%f0, %f0, 0             # encoding: [0xed,0x00,0x00,0x00,0x00,0x3b]
#CHECK: my	%f0, %f0, 4095          # encoding: [0xed,0x00,0x0f,0xff,0x00,0x3b]
#CHECK: my	%f0, %f0, 0(%r1)        # encoding: [0xed,0x00,0x10,0x00,0x00,0x3b]
#CHECK: my	%f0, %f0, 0(%r15)       # encoding: [0xed,0x00,0xf0,0x00,0x00,0x3b]
#CHECK: my	%f0, %f0, 4095(%r1,%r15) # encoding: [0xed,0x01,0xff,0xff,0x00,0x3b]
#CHECK: my	%f0, %f0, 4095(%r15,%r1) # encoding: [0xed,0x0f,0x1f,0xff,0x00,0x3b]
#CHECK: my	%f0, %f15, 0            # encoding: [0xed,0xf0,0x00,0x00,0x00,0x3b]
#CHECK: my	%f13, %f0, 0            # encoding: [0xed,0x00,0x00,0x00,0xd0,0x3b]
#CHECK: my	%f13, %f15, 0           # encoding: [0xed,0xf0,0x00,0x00,0xd0,0x3b]

	my	%f0, %f0, 0
	my	%f0, %f0, 4095
	my	%f0, %f0, 0(%r1)
	my	%f0, %f0, 0(%r15)
	my	%f0, %f0, 4095(%r1,%r15)
	my	%f0, %f0, 4095(%r15,%r1)
	my	%f0, %f15, 0
	my	%f13, %f0, 0
	my	%f13, %f15, 0

#CHECK: myh	%f0, %f0, 0             # encoding: [0xed,0x00,0x00,0x00,0x00,0x3d]
#CHECK: myh	%f0, %f0, 4095          # encoding: [0xed,0x00,0x0f,0xff,0x00,0x3d]
#CHECK: myh	%f0, %f0, 0(%r1)        # encoding: [0xed,0x00,0x10,0x00,0x00,0x3d]
#CHECK: myh	%f0, %f0, 0(%r15)       # encoding: [0xed,0x00,0xf0,0x00,0x00,0x3d]
#CHECK: myh	%f0, %f0, 4095(%r1,%r15) # encoding: [0xed,0x01,0xff,0xff,0x00,0x3d]
#CHECK: myh	%f0, %f0, 4095(%r15,%r1) # encoding: [0xed,0x0f,0x1f,0xff,0x00,0x3d]
#CHECK: myh	%f0, %f15, 0            # encoding: [0xed,0xf0,0x00,0x00,0x00,0x3d]
#CHECK: myh	%f15, %f0, 0            # encoding: [0xed,0x00,0x00,0x00,0xf0,0x3d]
#CHECK: myh	%f15, %f15, 0           # encoding: [0xed,0xf0,0x00,0x00,0xf0,0x3d]

	myh	%f0, %f0, 0
	myh	%f0, %f0, 4095
	myh	%f0, %f0, 0(%r1)
	myh	%f0, %f0, 0(%r15)
	myh	%f0, %f0, 4095(%r1,%r15)
	myh	%f0, %f0, 4095(%r15,%r1)
	myh	%f0, %f15, 0
	myh	%f15, %f0, 0
	myh	%f15, %f15, 0

#CHECK: myhr	%f0, %f0, %f0           # encoding: [0xb3,0x3d,0x00,0x00]
#CHECK: myhr	%f0, %f0, %f15          # encoding: [0xb3,0x3d,0x00,0x0f]
#CHECK: myhr	%f0, %f15, %f0          # encoding: [0xb3,0x3d,0x00,0xf0]
#CHECK: myhr	%f15, %f0, %f0          # encoding: [0xb3,0x3d,0xf0,0x00]
#CHECK: myhr	%f7, %f8, %f9           # encoding: [0xb3,0x3d,0x70,0x89]
#CHECK: myhr	%f15, %f15, %f15        # encoding: [0xb3,0x3d,0xf0,0xff]

	myhr	%f0, %f0, %f0
	myhr	%f0, %f0, %f15
	myhr	%f0, %f15, %f0
	myhr	%f15, %f0, %f0
	myhr	%f7, %f8, %f9
	myhr	%f15, %f15, %f15

#CHECK: myl	%f0, %f0, 0             # encoding: [0xed,0x00,0x00,0x00,0x00,0x39]
#CHECK: myl	%f0, %f0, 4095          # encoding: [0xed,0x00,0x0f,0xff,0x00,0x39]
#CHECK: myl	%f0, %f0, 0(%r1)        # encoding: [0xed,0x00,0x10,0x00,0x00,0x39]
#CHECK: myl	%f0, %f0, 0(%r15)       # encoding: [0xed,0x00,0xf0,0x00,0x00,0x39]
#CHECK: myl	%f0, %f0, 4095(%r1,%r15) # encoding: [0xed,0x01,0xff,0xff,0x00,0x39]
#CHECK: myl	%f0, %f0, 4095(%r15,%r1) # encoding: [0xed,0x0f,0x1f,0xff,0x00,0x39]
#CHECK: myl	%f0, %f15, 0            # encoding: [0xed,0xf0,0x00,0x00,0x00,0x39]
#CHECK: myl	%f15, %f0, 0            # encoding: [0xed,0x00,0x00,0x00,0xf0,0x39]
#CHECK: myl	%f15, %f15, 0           # encoding: [0xed,0xf0,0x00,0x00,0xf0,0x39]

	myl	%f0, %f0, 0
	myl	%f0, %f0, 4095
	myl	%f0, %f0, 0(%r1)
	myl	%f0, %f0, 0(%r15)
	myl	%f0, %f0, 4095(%r1,%r15)
	myl	%f0, %f0, 4095(%r15,%r1)
	myl	%f0, %f15, 0
	myl	%f15, %f0, 0
	myl	%f15, %f15, 0

#CHECK: mylr	%f0, %f0, %f0           # encoding: [0xb3,0x39,0x00,0x00]
#CHECK: mylr	%f0, %f0, %f15          # encoding: [0xb3,0x39,0x00,0x0f]
#CHECK: mylr	%f0, %f15, %f0          # encoding: [0xb3,0x39,0x00,0xf0]
#CHECK: mylr	%f15, %f0, %f0          # encoding: [0xb3,0x39,0xf0,0x00]
#CHECK: mylr	%f7, %f8, %f9           # encoding: [0xb3,0x39,0x70,0x89]
#CHECK: mylr	%f15, %f15, %f15        # encoding: [0xb3,0x39,0xf0,0xff]

	mylr	%f0, %f0, %f0
	mylr	%f0, %f0, %f15
	mylr	%f0, %f15, %f0
	mylr	%f15, %f0, %f0
	mylr	%f7, %f8, %f9
	mylr	%f15, %f15, %f15

#CHECK: myr	%f0, %f0, %f0           # encoding: [0xb3,0x3b,0x00,0x00]
#CHECK: myr	%f0, %f0, %f15          # encoding: [0xb3,0x3b,0x00,0x0f]
#CHECK: myr	%f0, %f15, %f0          # encoding: [0xb3,0x3b,0x00,0xf0]
#CHECK: myr	%f13, %f0, %f0          # encoding: [0xb3,0x3b,0xd0,0x00]
#CHECK: myr	%f5, %f8, %f9           # encoding: [0xb3,0x3b,0x50,0x89]
#CHECK: myr	%f13, %f15, %f15        # encoding: [0xb3,0x3b,0xd0,0xff]

	myr	%f0, %f0, %f0
	myr	%f0, %f0, %f15
	myr	%f0, %f15, %f0
	myr	%f13, %f0, %f0
	myr	%f5, %f8, %f9
	myr	%f13, %f15, %f15

#CHECK: n	%r0, 0                  # encoding: [0x54,0x00,0x00,0x00]
#CHECK: n	%r0, 4095               # encoding: [0x54,0x00,0x0f,0xff]
#CHECK: n	%r0, 0(%r1)             # encoding: [0x54,0x00,0x10,0x00]
#CHECK: n	%r0, 0(%r15)            # encoding: [0x54,0x00,0xf0,0x00]
#CHECK: n	%r0, 4095(%r1,%r15)     # encoding: [0x54,0x01,0xff,0xff]
#CHECK: n	%r0, 4095(%r15,%r1)     # encoding: [0x54,0x0f,0x1f,0xff]
#CHECK: n	%r15, 0                 # encoding: [0x54,0xf0,0x00,0x00]

	n	%r0, 0
	n	%r0, 4095
	n	%r0, 0(%r1)
	n	%r0, 0(%r15)
	n	%r0, 4095(%r1,%r15)
	n	%r0, 4095(%r15,%r1)
	n	%r15, 0

#CHECK: nc	0(1), 0                 # encoding: [0xd4,0x00,0x00,0x00,0x00,0x00]
#CHECK: nc	0(1), 0(%r1)            # encoding: [0xd4,0x00,0x00,0x00,0x10,0x00]
#CHECK: nc	0(1), 0(%r15)           # encoding: [0xd4,0x00,0x00,0x00,0xf0,0x00]
#CHECK: nc	0(1), 4095              # encoding: [0xd4,0x00,0x00,0x00,0x0f,0xff]
#CHECK: nc	0(1), 4095(%r1)         # encoding: [0xd4,0x00,0x00,0x00,0x1f,0xff]
#CHECK: nc	0(1), 4095(%r15)        # encoding: [0xd4,0x00,0x00,0x00,0xff,0xff]
#CHECK: nc	0(1,%r1), 0             # encoding: [0xd4,0x00,0x10,0x00,0x00,0x00]
#CHECK: nc	0(1,%r15), 0            # encoding: [0xd4,0x00,0xf0,0x00,0x00,0x00]
#CHECK: nc	4095(1,%r1), 0          # encoding: [0xd4,0x00,0x1f,0xff,0x00,0x00]
#CHECK: nc	4095(1,%r15), 0         # encoding: [0xd4,0x00,0xff,0xff,0x00,0x00]
#CHECK: nc	0(256,%r1), 0           # encoding: [0xd4,0xff,0x10,0x00,0x00,0x00]
#CHECK: nc	0(256,%r15), 0          # encoding: [0xd4,0xff,0xf0,0x00,0x00,0x00]

	nc	0(1), 0
	nc	0(1), 0(%r1)
	nc	0(1), 0(%r15)
	nc	0(1), 4095
	nc	0(1), 4095(%r1)
	nc	0(1), 4095(%r15)
	nc	0(1,%r1), 0
	nc	0(1,%r15), 0
	nc	4095(1,%r1), 0
	nc	4095(1,%r15), 0
	nc	0(256,%r1), 0
	nc	0(256,%r15), 0

#CHECK: ng	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x80]
#CHECK: ng	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x80]
#CHECK: ng	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x80]
#CHECK: ng	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x80]
#CHECK: ng	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x80]
#CHECK: ng	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x80]
#CHECK: ng	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x80]
#CHECK: ng	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x80]
#CHECK: ng	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x80]
#CHECK: ng	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x80]

	ng	%r0, -524288
	ng	%r0, -1
	ng	%r0, 0
	ng	%r0, 1
	ng	%r0, 524287
	ng	%r0, 0(%r1)
	ng	%r0, 0(%r15)
	ng	%r0, 524287(%r1,%r15)
	ng	%r0, 524287(%r15,%r1)
	ng	%r15, 0

#CHECK: ngr	%r0, %r0                # encoding: [0xb9,0x80,0x00,0x00]
#CHECK: ngr	%r0, %r15               # encoding: [0xb9,0x80,0x00,0x0f]
#CHECK: ngr	%r15, %r0               # encoding: [0xb9,0x80,0x00,0xf0]
#CHECK: ngr	%r7, %r8                # encoding: [0xb9,0x80,0x00,0x78]

	ngr	%r0,%r0
	ngr	%r0,%r15
	ngr	%r15,%r0
	ngr	%r7,%r8

#CHECK: ni	0, 0                    # encoding: [0x94,0x00,0x00,0x00]
#CHECK: ni	4095, 0                 # encoding: [0x94,0x00,0x0f,0xff]
#CHECK: ni	0, 255                  # encoding: [0x94,0xff,0x00,0x00]
#CHECK: ni	0(%r1), 42              # encoding: [0x94,0x2a,0x10,0x00]
#CHECK: ni	0(%r15), 42             # encoding: [0x94,0x2a,0xf0,0x00]
#CHECK: ni	4095(%r1), 42           # encoding: [0x94,0x2a,0x1f,0xff]
#CHECK: ni	4095(%r15), 42          # encoding: [0x94,0x2a,0xff,0xff]

	ni	0, 0
	ni	4095, 0
	ni	0, 255
	ni	0(%r1), 42
	ni	0(%r15), 42
	ni	4095(%r1), 42
	ni	4095(%r15), 42

#CHECK: nihf	%r0, 0                  # encoding: [0xc0,0x0a,0x00,0x00,0x00,0x00]
#CHECK: nihf	%r0, 4294967295         # encoding: [0xc0,0x0a,0xff,0xff,0xff,0xff]
#CHECK: nihf	%r15, 0                 # encoding: [0xc0,0xfa,0x00,0x00,0x00,0x00]

	nihf	%r0, 0
	nihf	%r0, 0xffffffff
	nihf	%r15, 0

#CHECK: nihh	%r0, 0                  # encoding: [0xa5,0x04,0x00,0x00]
#CHECK: nihh	%r0, 32768              # encoding: [0xa5,0x04,0x80,0x00]
#CHECK: nihh	%r0, 65535              # encoding: [0xa5,0x04,0xff,0xff]
#CHECK: nihh	%r15, 0                 # encoding: [0xa5,0xf4,0x00,0x00]

	nihh	%r0, 0
	nihh	%r0, 0x8000
	nihh	%r0, 0xffff
	nihh	%r15, 0

#CHECK: nihl	%r0, 0                  # encoding: [0xa5,0x05,0x00,0x00]
#CHECK: nihl	%r0, 32768              # encoding: [0xa5,0x05,0x80,0x00]
#CHECK: nihl	%r0, 65535              # encoding: [0xa5,0x05,0xff,0xff]
#CHECK: nihl	%r15, 0                 # encoding: [0xa5,0xf5,0x00,0x00]

	nihl	%r0, 0
	nihl	%r0, 0x8000
	nihl	%r0, 0xffff
	nihl	%r15, 0

#CHECK: nilf	%r0, 0                  # encoding: [0xc0,0x0b,0x00,0x00,0x00,0x00]
#CHECK: nilf	%r0, 4294967295         # encoding: [0xc0,0x0b,0xff,0xff,0xff,0xff]
#CHECK: nilf	%r15, 0                 # encoding: [0xc0,0xfb,0x00,0x00,0x00,0x00]

	nilf	%r0, 0
	nilf	%r0, 0xffffffff
	nilf	%r15, 0

#CHECK: nilh	%r0, 0                  # encoding: [0xa5,0x06,0x00,0x00]
#CHECK: nilh	%r0, 32768              # encoding: [0xa5,0x06,0x80,0x00]
#CHECK: nilh	%r0, 65535              # encoding: [0xa5,0x06,0xff,0xff]
#CHECK: nilh	%r15, 0                 # encoding: [0xa5,0xf6,0x00,0x00]

	nilh	%r0, 0
	nilh	%r0, 0x8000
	nilh	%r0, 0xffff
	nilh	%r15, 0

#CHECK: nill	%r0, 0                  # encoding: [0xa5,0x07,0x00,0x00]
#CHECK: nill	%r0, 32768              # encoding: [0xa5,0x07,0x80,0x00]
#CHECK: nill	%r0, 65535              # encoding: [0xa5,0x07,0xff,0xff]
#CHECK: nill	%r15, 0                 # encoding: [0xa5,0xf7,0x00,0x00]

	nill	%r0, 0
	nill	%r0, 0x8000
	nill	%r0, 0xffff
	nill	%r15, 0

#CHECK: niy	-524288, 0              # encoding: [0xeb,0x00,0x00,0x00,0x80,0x54]
#CHECK: niy	-1, 0                   # encoding: [0xeb,0x00,0x0f,0xff,0xff,0x54]
#CHECK: niy	0, 0                    # encoding: [0xeb,0x00,0x00,0x00,0x00,0x54]
#CHECK: niy	1, 0                    # encoding: [0xeb,0x00,0x00,0x01,0x00,0x54]
#CHECK: niy	524287, 0               # encoding: [0xeb,0x00,0x0f,0xff,0x7f,0x54]
#CHECK: niy	0, 255                  # encoding: [0xeb,0xff,0x00,0x00,0x00,0x54]
#CHECK: niy	0(%r1), 42              # encoding: [0xeb,0x2a,0x10,0x00,0x00,0x54]
#CHECK: niy	0(%r15), 42             # encoding: [0xeb,0x2a,0xf0,0x00,0x00,0x54]
#CHECK: niy	524287(%r1), 42         # encoding: [0xeb,0x2a,0x1f,0xff,0x7f,0x54]
#CHECK: niy	524287(%r15), 42        # encoding: [0xeb,0x2a,0xff,0xff,0x7f,0x54]

	niy	-524288, 0
	niy	-1, 0
	niy	0, 0
	niy	1, 0
	niy	524287, 0
	niy	0, 255
	niy	0(%r1), 42
	niy	0(%r15), 42
	niy	524287(%r1), 42
	niy	524287(%r15), 42

#CHECK: nop	0                       # encoding: [0x47,0x00,0x00,0x00]
#CHECK: nop                             # encoding: [0x47,0x00,0x00,0x00]
#CHECK: nopr	%r7                     # encoding: [0x07,0x07]
#CHECK: nopr	                        # encoding: [0x07,0x00]

	nop	0
	nop
	nopr	%r7
	nopr

#CHECK: nr	%r0, %r0                # encoding: [0x14,0x00]
#CHECK: nr	%r0, %r15               # encoding: [0x14,0x0f]
#CHECK: nr	%r15, %r0               # encoding: [0x14,0xf0]
#CHECK: nr	%r7, %r8                # encoding: [0x14,0x78]

	nr	%r0,%r0
	nr	%r0,%r15
	nr	%r15,%r0
	nr	%r7,%r8

#CHECK: ny	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x54]
#CHECK: ny	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x54]
#CHECK: ny	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x54]
#CHECK: ny	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x54]
#CHECK: ny	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x54]
#CHECK: ny	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x54]
#CHECK: ny	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x54]
#CHECK: ny	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x54]
#CHECK: ny	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x54]
#CHECK: ny	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x54]

	ny	%r0, -524288
	ny	%r0, -1
	ny	%r0, 0
	ny	%r0, 1
	ny	%r0, 524287
	ny	%r0, 0(%r1)
	ny	%r0, 0(%r15)
	ny	%r0, 524287(%r1,%r15)
	ny	%r0, 524287(%r15,%r1)
	ny	%r15, 0

#CHECK: o	%r0, 0                  # encoding: [0x56,0x00,0x00,0x00]
#CHECK: o	%r0, 4095               # encoding: [0x56,0x00,0x0f,0xff]
#CHECK: o	%r0, 0(%r1)             # encoding: [0x56,0x00,0x10,0x00]
#CHECK: o	%r0, 0(%r15)            # encoding: [0x56,0x00,0xf0,0x00]
#CHECK: o	%r0, 4095(%r1,%r15)     # encoding: [0x56,0x01,0xff,0xff]
#CHECK: o	%r0, 4095(%r15,%r1)     # encoding: [0x56,0x0f,0x1f,0xff]
#CHECK: o	%r15, 0                 # encoding: [0x56,0xf0,0x00,0x00]

	o	%r0, 0
	o	%r0, 4095
	o	%r0, 0(%r1)
	o	%r0, 0(%r15)
	o	%r0, 4095(%r1,%r15)
	o	%r0, 4095(%r15,%r1)
	o	%r15, 0

#CHECK: oc	0(1), 0                 # encoding: [0xd6,0x00,0x00,0x00,0x00,0x00]
#CHECK: oc	0(1), 0(%r1)            # encoding: [0xd6,0x00,0x00,0x00,0x10,0x00]
#CHECK: oc	0(1), 0(%r15)           # encoding: [0xd6,0x00,0x00,0x00,0xf0,0x00]
#CHECK: oc	0(1), 4095              # encoding: [0xd6,0x00,0x00,0x00,0x0f,0xff]
#CHECK: oc	0(1), 4095(%r1)         # encoding: [0xd6,0x00,0x00,0x00,0x1f,0xff]
#CHECK: oc	0(1), 4095(%r15)        # encoding: [0xd6,0x00,0x00,0x00,0xff,0xff]
#CHECK: oc	0(1,%r1), 0             # encoding: [0xd6,0x00,0x10,0x00,0x00,0x00]
#CHECK: oc	0(1,%r15), 0            # encoding: [0xd6,0x00,0xf0,0x00,0x00,0x00]
#CHECK: oc	4095(1,%r1), 0          # encoding: [0xd6,0x00,0x1f,0xff,0x00,0x00]
#CHECK: oc	4095(1,%r15), 0         # encoding: [0xd6,0x00,0xff,0xff,0x00,0x00]
#CHECK: oc	0(256,%r1), 0           # encoding: [0xd6,0xff,0x10,0x00,0x00,0x00]
#CHECK: oc	0(256,%r15), 0          # encoding: [0xd6,0xff,0xf0,0x00,0x00,0x00]

	oc	0(1), 0
	oc	0(1), 0(%r1)
	oc	0(1), 0(%r15)
	oc	0(1), 4095
	oc	0(1), 4095(%r1)
	oc	0(1), 4095(%r15)
	oc	0(1,%r1), 0
	oc	0(1,%r15), 0
	oc	4095(1,%r1), 0
	oc	4095(1,%r15), 0
	oc	0(256,%r1), 0
	oc	0(256,%r15), 0

#CHECK: og	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x81]
#CHECK: og	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x81]
#CHECK: og	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x81]
#CHECK: og	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x81]
#CHECK: og	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x81]
#CHECK: og	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x81]
#CHECK: og	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x81]
#CHECK: og	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x81]
#CHECK: og	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x81]
#CHECK: og	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x81]

	og	%r0, -524288
	og	%r0, -1
	og	%r0, 0
	og	%r0, 1
	og	%r0, 524287
	og	%r0, 0(%r1)
	og	%r0, 0(%r15)
	og	%r0, 524287(%r1,%r15)
	og	%r0, 524287(%r15,%r1)
	og	%r15, 0

#CHECK: ogr	%r0, %r0                # encoding: [0xb9,0x81,0x00,0x00]
#CHECK: ogr	%r0, %r15               # encoding: [0xb9,0x81,0x00,0x0f]
#CHECK: ogr	%r15, %r0               # encoding: [0xb9,0x81,0x00,0xf0]
#CHECK: ogr	%r7, %r8                # encoding: [0xb9,0x81,0x00,0x78]

	ogr	%r0,%r0
	ogr	%r0,%r15
	ogr	%r15,%r0
	ogr	%r7,%r8

#CHECK: oi	0, 0                    # encoding: [0x96,0x00,0x00,0x00]
#CHECK: oi	4095, 0                 # encoding: [0x96,0x00,0x0f,0xff]
#CHECK: oi	0, 255                  # encoding: [0x96,0xff,0x00,0x00]
#CHECK: oi	0(%r1), 42              # encoding: [0x96,0x2a,0x10,0x00]
#CHECK: oi	0(%r15), 42             # encoding: [0x96,0x2a,0xf0,0x00]
#CHECK: oi	4095(%r1), 42           # encoding: [0x96,0x2a,0x1f,0xff]
#CHECK: oi	4095(%r15), 42          # encoding: [0x96,0x2a,0xff,0xff]

	oi	0, 0
	oi	4095, 0
	oi	0, 255
	oi	0(%r1), 42
	oi	0(%r15), 42
	oi	4095(%r1), 42
	oi	4095(%r15), 42

#CHECK: oihf	%r0, 0                  # encoding: [0xc0,0x0c,0x00,0x00,0x00,0x00]
#CHECK: oihf	%r0, 4294967295         # encoding: [0xc0,0x0c,0xff,0xff,0xff,0xff]
#CHECK: oihf	%r15, 0                 # encoding: [0xc0,0xfc,0x00,0x00,0x00,0x00]

	oihf	%r0, 0
	oihf	%r0, 0xffffffff
	oihf	%r15, 0

#CHECK: oihh	%r0, 0                  # encoding: [0xa5,0x08,0x00,0x00]
#CHECK: oihh	%r0, 32768              # encoding: [0xa5,0x08,0x80,0x00]
#CHECK: oihh	%r0, 65535              # encoding: [0xa5,0x08,0xff,0xff]
#CHECK: oihh	%r15, 0                 # encoding: [0xa5,0xf8,0x00,0x00]

	oihh	%r0, 0
	oihh	%r0, 0x8000
	oihh	%r0, 0xffff
	oihh	%r15, 0

#CHECK: oihl	%r0, 0                  # encoding: [0xa5,0x09,0x00,0x00]
#CHECK: oihl	%r0, 32768              # encoding: [0xa5,0x09,0x80,0x00]
#CHECK: oihl	%r0, 65535              # encoding: [0xa5,0x09,0xff,0xff]
#CHECK: oihl	%r15, 0                 # encoding: [0xa5,0xf9,0x00,0x00]

	oihl	%r0, 0
	oihl	%r0, 0x8000
	oihl	%r0, 0xffff
	oihl	%r15, 0

#CHECK: oilf	%r0, 0                  # encoding: [0xc0,0x0d,0x00,0x00,0x00,0x00]
#CHECK: oilf	%r0, 4294967295         # encoding: [0xc0,0x0d,0xff,0xff,0xff,0xff]
#CHECK: oilf	%r15, 0                 # encoding: [0xc0,0xfd,0x00,0x00,0x00,0x00]

	oilf	%r0, 0
	oilf	%r0, 0xffffffff
	oilf	%r15, 0

#CHECK: oilh	%r0, 0                  # encoding: [0xa5,0x0a,0x00,0x00]
#CHECK: oilh	%r0, 32768              # encoding: [0xa5,0x0a,0x80,0x00]
#CHECK: oilh	%r0, 65535              # encoding: [0xa5,0x0a,0xff,0xff]
#CHECK: oilh	%r15, 0                 # encoding: [0xa5,0xfa,0x00,0x00]

	oilh	%r0, 0
	oilh	%r0, 0x8000
	oilh	%r0, 0xffff
	oilh	%r15, 0

#CHECK: oill	%r0, 0                  # encoding: [0xa5,0x0b,0x00,0x00]
#CHECK: oill	%r0, 32768              # encoding: [0xa5,0x0b,0x80,0x00]
#CHECK: oill	%r0, 65535              # encoding: [0xa5,0x0b,0xff,0xff]
#CHECK: oill	%r15, 0                 # encoding: [0xa5,0xfb,0x00,0x00]

	oill	%r0, 0
	oill	%r0, 0x8000
	oill	%r0, 0xffff
	oill	%r15, 0

#CHECK: oiy	-524288, 0              # encoding: [0xeb,0x00,0x00,0x00,0x80,0x56]
#CHECK: oiy	-1, 0                   # encoding: [0xeb,0x00,0x0f,0xff,0xff,0x56]
#CHECK: oiy	0, 0                    # encoding: [0xeb,0x00,0x00,0x00,0x00,0x56]
#CHECK: oiy	1, 0                    # encoding: [0xeb,0x00,0x00,0x01,0x00,0x56]
#CHECK: oiy	524287, 0               # encoding: [0xeb,0x00,0x0f,0xff,0x7f,0x56]
#CHECK: oiy	0, 255                  # encoding: [0xeb,0xff,0x00,0x00,0x00,0x56]
#CHECK: oiy	0(%r1), 42              # encoding: [0xeb,0x2a,0x10,0x00,0x00,0x56]
#CHECK: oiy	0(%r15), 42             # encoding: [0xeb,0x2a,0xf0,0x00,0x00,0x56]
#CHECK: oiy	524287(%r1), 42         # encoding: [0xeb,0x2a,0x1f,0xff,0x7f,0x56]
#CHECK: oiy	524287(%r15), 42        # encoding: [0xeb,0x2a,0xff,0xff,0x7f,0x56]

	oiy	-524288, 0
	oiy	-1, 0
	oiy	0, 0
	oiy	1, 0
	oiy	524287, 0
	oiy	0, 255
	oiy	0(%r1), 42
	oiy	0(%r15), 42
	oiy	524287(%r1), 42
	oiy	524287(%r15), 42

#CHECK: or	%r0, %r0                # encoding: [0x16,0x00]
#CHECK: or	%r0, %r15               # encoding: [0x16,0x0f]
#CHECK: or	%r15, %r0               # encoding: [0x16,0xf0]
#CHECK: or	%r7, %r8                # encoding: [0x16,0x78]

	or	%r0,%r0
	or	%r0,%r15
	or	%r15,%r0
	or	%r7,%r8

#CHECK: oy	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x56]
#CHECK: oy	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x56]
#CHECK: oy	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x56]
#CHECK: oy	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x56]
#CHECK: oy	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x56]
#CHECK: oy	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x56]
#CHECK: oy	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x56]
#CHECK: oy	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x56]
#CHECK: oy	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x56]
#CHECK: oy	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x56]

	oy	%r0, -524288
	oy	%r0, -1
	oy	%r0, 0
	oy	%r0, 1
	oy	%r0, 524287
	oy	%r0, 0(%r1)
	oy	%r0, 0(%r15)
	oy	%r0, 524287(%r1,%r15)
	oy	%r0, 524287(%r15,%r1)
	oy	%r15, 0

#CHECK: pack	0(1), 0(1)              # encoding: [0xf2,0x00,0x00,0x00,0x00,0x00]
#CHECK: pack	0(1), 0(1,%r1)          # encoding: [0xf2,0x00,0x00,0x00,0x10,0x00]
#CHECK: pack	0(1), 0(1,%r15)         # encoding: [0xf2,0x00,0x00,0x00,0xf0,0x00]
#CHECK: pack	0(1), 4095(1)           # encoding: [0xf2,0x00,0x00,0x00,0x0f,0xff]
#CHECK: pack	0(1), 4095(1,%r1)       # encoding: [0xf2,0x00,0x00,0x00,0x1f,0xff]
#CHECK: pack	0(1), 4095(1,%r15)      # encoding: [0xf2,0x00,0x00,0x00,0xff,0xff]
#CHECK: pack	0(1,%r1), 0(1)          # encoding: [0xf2,0x00,0x10,0x00,0x00,0x00]
#CHECK: pack	0(1,%r15), 0(1)         # encoding: [0xf2,0x00,0xf0,0x00,0x00,0x00]
#CHECK: pack	4095(1,%r1), 0(1)       # encoding: [0xf2,0x00,0x1f,0xff,0x00,0x00]
#CHECK: pack	4095(1,%r15), 0(1)      # encoding: [0xf2,0x00,0xff,0xff,0x00,0x00]
#CHECK: pack	0(16,%r1), 0(1)         # encoding: [0xf2,0xf0,0x10,0x00,0x00,0x00]
#CHECK: pack	0(16,%r15), 0(1)        # encoding: [0xf2,0xf0,0xf0,0x00,0x00,0x00]
#CHECK: pack	0(1), 0(16,%r1)         # encoding: [0xf2,0x0f,0x00,0x00,0x10,0x00]
#CHECK: pack	0(1), 0(16,%r15)        # encoding: [0xf2,0x0f,0x00,0x00,0xf0,0x00]

	pack	0(1), 0(1)
	pack	0(1), 0(1,%r1)
	pack	0(1), 0(1,%r15)
	pack	0(1), 4095(1)
	pack	0(1), 4095(1,%r1)
	pack	0(1), 4095(1,%r15)
	pack	0(1,%r1), 0(1)
	pack	0(1,%r15), 0(1)
	pack	4095(1,%r1), 0(1)
	pack	4095(1,%r15), 0(1)
	pack	0(16,%r1), 0(1)
	pack	0(16,%r15), 0(1)
	pack	0(1), 0(16,%r1)
	pack	0(1), 0(16,%r15)

#CHECK: palb	                        # encoding: [0xb2,0x48,0x00,0x00]

	palb

#CHECK: pc	0                       # encoding: [0xb2,0x18,0x00,0x00]
#CHECK: pc	0(%r1)                  # encoding: [0xb2,0x18,0x10,0x00]
#CHECK: pc	0(%r15)                 # encoding: [0xb2,0x18,0xf0,0x00]
#CHECK: pc	4095                    # encoding: [0xb2,0x18,0x0f,0xff]
#CHECK: pc	4095(%r1)               # encoding: [0xb2,0x18,0x1f,0xff]
#CHECK: pc	4095(%r15)              # encoding: [0xb2,0x18,0xff,0xff]

	pc	0
	pc	0(%r1)
	pc	0(%r15)
	pc	4095
	pc	4095(%r1)
	pc	4095(%r15)

#CHECK: pfd	0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x36]
#CHECK: pfd	0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x36]
#CHECK: pfd	0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x36]
#CHECK: pfd	0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x36]
#CHECK: pfd	0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x36]
#CHECK: pfd	0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x36]
#CHECK: pfd	0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x36]
#CHECK: pfd	0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x36]
#CHECK: pfd	0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x36]
#CHECK: pfd	15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x36]

	pfd	0, -524288
	pfd	0, -1
	pfd	0, 0
	pfd	0, 1
	pfd	0, 524287
	pfd	0, 0(%r1)
	pfd	0, 0(%r15)
	pfd	0, 524287(%r1,%r15)
	pfd	0, 524287(%r15,%r1)
	pfd	15, 0

#CHECK: pfdrl	0, .[[LAB:L.*]]-4294967296 # encoding: [0xc6,0x02,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-4294967296)+2, kind: FK_390_PC32DBL
	pfdrl	0, -0x100000000
#CHECK: pfdrl	0, .[[LAB:L.*]]-2	# encoding: [0xc6,0x02,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC32DBL
	pfdrl	0, -2
#CHECK: pfdrl	0, .[[LAB:L.*]]	# encoding: [0xc6,0x02,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC32DBL
	pfdrl	0, 0
#CHECK: pfdrl	0, .[[LAB:L.*]]+4294967294 # encoding: [0xc6,0x02,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+4294967294)+2, kind: FK_390_PC32DBL
	pfdrl	0, 0xfffffffe

#CHECK: pfdrl	0, foo                # encoding: [0xc6,0x02,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: pfdrl	15, foo               # encoding: [0xc6,0xf2,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL

	pfdrl	0, foo
	pfdrl	15, foo

#CHECK: pfdrl	3, bar+100            # encoding: [0xc6,0x32,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: pfdrl	4, bar+100            # encoding: [0xc6,0x42,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL

	pfdrl	3, bar+100
	pfdrl	4, bar+100

#CHECK: pfdrl	7, frob@PLT           # encoding: [0xc6,0x72,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL
#CHECK: pfdrl	8, frob@PLT           # encoding: [0xc6,0x82,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL

	pfdrl	7, frob@PLT
	pfdrl	8, frob@PLT

#CHECK: pfmf	%r0, %r0                # encoding: [0xb9,0xaf,0x00,0x00]
#CHECK: pfmf	%r0, %r15               # encoding: [0xb9,0xaf,0x00,0x0f]
#CHECK: pfmf	%r15, %r0               # encoding: [0xb9,0xaf,0x00,0xf0]
#CHECK: pfmf	%r7, %r8                # encoding: [0xb9,0xaf,0x00,0x78]
#CHECK: pfmf	%r15, %r15              # encoding: [0xb9,0xaf,0x00,0xff]

	pfmf	%r0,%r0
	pfmf	%r0,%r15
	pfmf	%r15,%r0
	pfmf	%r7,%r8
	pfmf	%r15,%r15

#CHECK: pfpo                            # encoding: [0x01,0x0a]
        pfpo

#CHECK: pgin	%r0, %r0                # encoding: [0xb2,0x2e,0x00,0x00]
#CHECK: pgin	%r0, %r15               # encoding: [0xb2,0x2e,0x00,0x0f]
#CHECK: pgin	%r15, %r0               # encoding: [0xb2,0x2e,0x00,0xf0]
#CHECK: pgin	%r7, %r8                # encoding: [0xb2,0x2e,0x00,0x78]
#CHECK: pgin	%r15, %r15              # encoding: [0xb2,0x2e,0x00,0xff]

	pgin	%r0,%r0
	pgin	%r0,%r15
	pgin	%r15,%r0
	pgin	%r7,%r8
	pgin	%r15,%r15

#CHECK: pgout	%r0, %r0                # encoding: [0xb2,0x2f,0x00,0x00]
#CHECK: pgout	%r0, %r15               # encoding: [0xb2,0x2f,0x00,0x0f]
#CHECK: pgout	%r15, %r0               # encoding: [0xb2,0x2f,0x00,0xf0]
#CHECK: pgout	%r7, %r8                # encoding: [0xb2,0x2f,0x00,0x78]
#CHECK: pgout	%r15, %r15              # encoding: [0xb2,0x2f,0x00,0xff]

	pgout	%r0,%r0
	pgout	%r0,%r15
	pgout	%r15,%r0
	pgout	%r7,%r8
	pgout	%r15,%r15

#CHECK: pka	0, 0(1)                 # encoding: [0xe9,0x00,0x00,0x00,0x00,0x00]
#CHECK: pka	0, 0(1,%r1)             # encoding: [0xe9,0x00,0x00,0x00,0x10,0x00]
#CHECK: pka	0, 0(1,%r15)            # encoding: [0xe9,0x00,0x00,0x00,0xf0,0x00]
#CHECK: pka	0, 4095(1)              # encoding: [0xe9,0x00,0x00,0x00,0x0f,0xff]
#CHECK: pka	0, 4095(1,%r1)          # encoding: [0xe9,0x00,0x00,0x00,0x1f,0xff]
#CHECK: pka	0, 4095(1,%r15)         # encoding: [0xe9,0x00,0x00,0x00,0xff,0xff]
#CHECK: pka	0(%r1), 0(1)            # encoding: [0xe9,0x00,0x10,0x00,0x00,0x00]
#CHECK: pka	0(%r15), 0(1)           # encoding: [0xe9,0x00,0xf0,0x00,0x00,0x00]
#CHECK: pka	4095(%r1), 0(1)         # encoding: [0xe9,0x00,0x1f,0xff,0x00,0x00]
#CHECK: pka	4095(%r15), 0(1)        # encoding: [0xe9,0x00,0xff,0xff,0x00,0x00]
#CHECK: pka	0, 0(256,%r1)           # encoding: [0xe9,0xff,0x00,0x00,0x10,0x00]
#CHECK: pka	0, 0(256,%r15)          # encoding: [0xe9,0xff,0x00,0x00,0xf0,0x00]

	pka	0, 0(1)
	pka	0, 0(1,%r1)
	pka	0, 0(1,%r15)
	pka	0, 4095(1)
	pka	0, 4095(1,%r1)
	pka	0, 4095(1,%r15)
	pka	0(%r1), 0(1)
	pka	0(%r15), 0(1)
	pka	4095(%r1), 0(1)
	pka	4095(%r15), 0(1)
	pka	0, 0(256,%r1)
	pka	0, 0(256,%r15)

#CHECK: pku	0, 0(1)                 # encoding: [0xe1,0x00,0x00,0x00,0x00,0x00]
#CHECK: pku	0, 0(1,%r1)             # encoding: [0xe1,0x00,0x00,0x00,0x10,0x00]
#CHECK: pku	0, 0(1,%r15)            # encoding: [0xe1,0x00,0x00,0x00,0xf0,0x00]
#CHECK: pku	0, 4095(1)              # encoding: [0xe1,0x00,0x00,0x00,0x0f,0xff]
#CHECK: pku	0, 4095(1,%r1)          # encoding: [0xe1,0x00,0x00,0x00,0x1f,0xff]
#CHECK: pku	0, 4095(1,%r15)         # encoding: [0xe1,0x00,0x00,0x00,0xff,0xff]
#CHECK: pku	0(%r1), 0(1)            # encoding: [0xe1,0x00,0x10,0x00,0x00,0x00]
#CHECK: pku	0(%r15), 0(1)           # encoding: [0xe1,0x00,0xf0,0x00,0x00,0x00]
#CHECK: pku	4095(%r1), 0(1)         # encoding: [0xe1,0x00,0x1f,0xff,0x00,0x00]
#CHECK: pku	4095(%r15), 0(1)        # encoding: [0xe1,0x00,0xff,0xff,0x00,0x00]
#CHECK: pku	0, 0(256,%r1)           # encoding: [0xe1,0xff,0x00,0x00,0x10,0x00]
#CHECK: pku	0, 0(256,%r15)          # encoding: [0xe1,0xff,0x00,0x00,0xf0,0x00]

	pku	0, 0(1)
	pku	0, 0(1,%r1)
	pku	0, 0(1,%r15)
	pku	0, 4095(1)
	pku	0, 4095(1,%r1)
	pku	0, 4095(1,%r15)
	pku	0(%r1), 0(1)
	pku	0(%r15), 0(1)
	pku	4095(%r1), 0(1)
	pku	4095(%r15), 0(1)
	pku	0, 0(256,%r1)
	pku	0, 0(256,%r15)

#CHECK: plo	%r0, 0, %r0, 0               # encoding: [0xee,0x00,0x00,0x00,0x00,0x00]
#CHECK: plo	%r2, 0(%r1), %r4, 0(%r15)    # encoding: [0xee,0x24,0x10,0x00,0xf0,0x00]
#CHECK: plo	%r2, 1(%r1), %r4, 0(%r15)    # encoding: [0xee,0x24,0x10,0x01,0xf0,0x00]
#CHECK: plo	%r2, 4095(%r1), %r4, 0(%r15) # encoding: [0xee,0x24,0x1f,0xff,0xf0,0x00]
#CHECK: plo	%r2, 0(%r1), %r4, 1(%r15)    # encoding: [0xee,0x24,0x10,0x00,0xf0,0x01]
#CHECK: plo	%r2, 0(%r1), %r4, 4095(%r15) # encoding: [0xee,0x24,0x10,0x00,0xff,0xff]

        plo	%r0, 0, %r0, 0
        plo	%r2, 0(%r1), %r4, 0(%r15)
        plo	%r2, 1(%r1), %r4, 0(%r15)
        plo	%r2, 4095(%r1), %r4, 0(%r15)
        plo	%r2, 0(%r1), %r4, 1(%r15)
        plo	%r2, 0(%r1), %r4, 4095(%r15)

#CHECK: pr                              # encoding: [0x01,0x01]

        pr

#CHECK: pt	%r0, %r0                # encoding: [0xb2,0x28,0x00,0x00]
#CHECK: pt	%r0, %r15               # encoding: [0xb2,0x28,0x00,0x0f]
#CHECK: pt	%r15, %r0               # encoding: [0xb2,0x28,0x00,0xf0]
#CHECK: pt	%r7, %r8                # encoding: [0xb2,0x28,0x00,0x78]

	pt	%r0,%r0
	pt	%r0,%r15
	pt	%r15,%r0
	pt	%r7,%r8

#CHECK: ptf	%r0                     # encoding: [0xb9,0xa2,0x00,0x00]
#CHECK: ptf	%r1                     # encoding: [0xb9,0xa2,0x00,0x10]
#CHECK: ptf	%r15                    # encoding: [0xb9,0xa2,0x00,0xf0]

	ptf	%r0
	ptf	%r1
	ptf	%r15

#CHECK: ptff                            # encoding: [0x01,0x04]

        ptff

#CHECK: pti	%r0, %r0                # encoding: [0xb9,0x9e,0x00,0x00]
#CHECK: pti	%r0, %r15               # encoding: [0xb9,0x9e,0x00,0x0f]
#CHECK: pti	%r15, %r0               # encoding: [0xb9,0x9e,0x00,0xf0]
#CHECK: pti	%r7, %r8                # encoding: [0xb9,0x9e,0x00,0x78]

	pti	%r0,%r0
	pti	%r0,%r15
	pti	%r15,%r0
	pti	%r7,%r8

#CHECK: ptlb                            # encoding: [0xb2,0x0d,0x00,0x00]

	ptlb

#CHECK: qadtr	%f0, %f0, %f0, 0        # encoding: [0xb3,0xf5,0x00,0x00]
#CHECK: qadtr	%f0, %f0, %f0, 15       # encoding: [0xb3,0xf5,0x0f,0x00]
#CHECK: qadtr	%f0, %f0, %f15, 0       # encoding: [0xb3,0xf5,0x00,0x0f]
#CHECK: qadtr	%f0, %f15, %f0, 0       # encoding: [0xb3,0xf5,0xf0,0x00]
#CHECK: qadtr	%f4, %f5, %f6, 7        # encoding: [0xb3,0xf5,0x57,0x46]
#CHECK: qadtr	%f15, %f0, %f0, 0       # encoding: [0xb3,0xf5,0x00,0xf0]

	qadtr	%f0, %f0, %f0, 0
	qadtr	%f0, %f0, %f0, 15
	qadtr	%f0, %f0, %f15, 0
	qadtr	%f0, %f15, %f0, 0
	qadtr	%f4, %f5, %f6, 7
	qadtr	%f15, %f0, %f0, 0

#CHECK: qaxtr	%f0, %f0, %f0, 0        # encoding: [0xb3,0xfd,0x00,0x00]
#CHECK: qaxtr	%f0, %f0, %f0, 15       # encoding: [0xb3,0xfd,0x0f,0x00]
#CHECK: qaxtr	%f0, %f0, %f13, 0       # encoding: [0xb3,0xfd,0x00,0x0d]
#CHECK: qaxtr	%f0, %f13, %f0, 0       # encoding: [0xb3,0xfd,0xd0,0x00]
#CHECK: qaxtr	%f8, %f8, %f8, 8        # encoding: [0xb3,0xfd,0x88,0x88]
#CHECK: qaxtr	%f13, %f0, %f0, 0       # encoding: [0xb3,0xfd,0x00,0xd0]

	qaxtr	%f0, %f0, %f0, 0
	qaxtr	%f0, %f0, %f0, 15
	qaxtr	%f0, %f0, %f13, 0
	qaxtr	%f0, %f13, %f0, 0
	qaxtr	%f8, %f8, %f8, 8
	qaxtr	%f13, %f0, %f0, 0

#CHECK: qctri	0                       # encoding: [0xb2,0x8e,0x00,0x00]
#CHECK: qctri	0(%r1)                  # encoding: [0xb2,0x8e,0x10,0x00]
#CHECK: qctri	0(%r15)                 # encoding: [0xb2,0x8e,0xf0,0x00]
#CHECK: qctri	4095                    # encoding: [0xb2,0x8e,0x0f,0xff]
#CHECK: qctri	4095(%r1)               # encoding: [0xb2,0x8e,0x1f,0xff]
#CHECK: qctri	4095(%r15)              # encoding: [0xb2,0x8e,0xff,0xff]

	qctri	0
	qctri	0(%r1)
	qctri	0(%r15)
	qctri	4095
	qctri	4095(%r1)
	qctri	4095(%r15)

#CHECK: qsi	0                       # encoding: [0xb2,0x86,0x00,0x00]
#CHECK: qsi	0(%r1)                  # encoding: [0xb2,0x86,0x10,0x00]
#CHECK: qsi	0(%r15)                 # encoding: [0xb2,0x86,0xf0,0x00]
#CHECK: qsi	4095                    # encoding: [0xb2,0x86,0x0f,0xff]
#CHECK: qsi	4095(%r1)               # encoding: [0xb2,0x86,0x1f,0xff]
#CHECK: qsi	4095(%r15)              # encoding: [0xb2,0x86,0xff,0xff]

	qsi	0
	qsi	0(%r1)
	qsi	0(%r15)
	qsi	4095
	qsi	4095(%r1)
	qsi	4095(%r15)

#CHECK: rchp                            # encoding: [0xb2,0x3b,0x00,0x00]

	rchp

#CHECK: risbg	%r0, %r0, 0, 0, 0       # encoding: [0xec,0x00,0x00,0x00,0x00,0x55]
#CHECK: risbg	%r0, %r0, 0, 0, 63      # encoding: [0xec,0x00,0x00,0x00,0x3f,0x55]
#CHECK: risbg	%r0, %r0, 0, 0, 64      # encoding: [0xec,0x00,0x00,0x00,0x40,0x55]
#CHECK: risbg	%r0, %r0, 0, 0, 255     # encoding: [0xec,0x00,0x00,0x00,0xff,0x55]
#CHECK: risbgz	%r0, %r0, 0, 255, 0     # encoding: [0xec,0x00,0x00,0xff,0x00,0x55]
#CHECK: risbg 	%r0, %r0, 0, 255, 0     # encoding: [0xec,0x00,0x00,0xff,0x00,0x55]
#CHECK: risbg	%r0, %r0, 255, 0, 0     # encoding: [0xec,0x00,0xff,0x00,0x00,0x55]
#CHECK: risbg	%r0, %r0, 0, 0, 127     # encoding: [0xec,0x00,0x00,0x00,0x7f,0x55]
#CHECK: risbgz	%r0, %r0, 0, 127, 0     # encoding: [0xec,0x00,0x00,0xff,0x00,0x55]
#CHECK: risbg 	%r0, %r0, 0, 127, 0     # encoding: [0xec,0x00,0x00,0x7f,0x00,0x55]
#CHECK: risbg	%r0, %r0, 127, 0, 0     # encoding: [0xec,0x00,0x7f,0x00,0x00,0x55]
#CHECK: risbg	%r0, %r15, 0, 0, 0      # encoding: [0xec,0x0f,0x00,0x00,0x00,0x55]
#CHECK: risbg	%r15, %r0, 0, 0, 0      # encoding: [0xec,0xf0,0x00,0x00,0x00,0x55]
#CHECK: risbg	%r4, %r5, 6, 7, 8       # encoding: [0xec,0x45,0x06,0x07,0x08,0x55]
#CHECK: risbgz	%r4, %r5, 6, 7, 8       # encoding: [0xec,0x45,0x06,0x87,0x08,0x55]

	risbg	%r0,%r0,0,0,0
	risbg	%r0,%r0,0,0,63
	risbg	%r0,%r0,0,0,64
	risbg	%r0,%r0,0,0,255
	risbgz	%r0,%r0,0,255,0
	risbg	%r0,%r0,0,255,0
	risbg	%r0,%r0,255,0,0
	risbg	%r0,%r0,0,0,127
	risbgz	%r0,%r0,0,127,0
	risbg	%r0,%r0,0,127,0
	risbg	%r0,%r0,127,0,0
	risbg	%r0,%r15,0,0,0
	risbg	%r15,%r0,0,0,0
	risbg	%r4,%r5,6,7,8
	risbgz	%r4,%r5,6,7,8

#CHECK: rll	%r0, %r0, 0             # encoding: [0xeb,0x00,0x00,0x00,0x00,0x1d]
#CHECK: rll	%r15, %r1, 0            # encoding: [0xeb,0xf1,0x00,0x00,0x00,0x1d]
#CHECK: rll	%r1, %r15, 0            # encoding: [0xeb,0x1f,0x00,0x00,0x00,0x1d]
#CHECK: rll	%r15, %r15, 0           # encoding: [0xeb,0xff,0x00,0x00,0x00,0x1d]
#CHECK: rll	%r0, %r0, -524288       # encoding: [0xeb,0x00,0x00,0x00,0x80,0x1d]
#CHECK: rll	%r0, %r0, -1            # encoding: [0xeb,0x00,0x0f,0xff,0xff,0x1d]
#CHECK: rll	%r0, %r0, 1             # encoding: [0xeb,0x00,0x00,0x01,0x00,0x1d]
#CHECK: rll	%r0, %r0, 524287        # encoding: [0xeb,0x00,0x0f,0xff,0x7f,0x1d]
#CHECK: rll	%r0, %r0, 0(%r1)        # encoding: [0xeb,0x00,0x10,0x00,0x00,0x1d]
#CHECK: rll	%r0, %r0, 0(%r15)       # encoding: [0xeb,0x00,0xf0,0x00,0x00,0x1d]
#CHECK: rll	%r0, %r0, 524287(%r1)   # encoding: [0xeb,0x00,0x1f,0xff,0x7f,0x1d]
#CHECK: rll	%r0, %r0, 524287(%r15)  # encoding: [0xeb,0x00,0xff,0xff,0x7f,0x1d]

	rll	%r0,%r0,0
	rll	%r15,%r1,0
	rll	%r1,%r15,0
	rll	%r15,%r15,0
	rll	%r0,%r0,-524288
	rll	%r0,%r0,-1
	rll	%r0,%r0,1
	rll	%r0,%r0,524287
	rll	%r0,%r0,0(%r1)
	rll	%r0,%r0,0(%r15)
	rll	%r0,%r0,524287(%r1)
	rll	%r0,%r0,524287(%r15)

#CHECK: rllg	%r0, %r0, 0             # encoding: [0xeb,0x00,0x00,0x00,0x00,0x1c]
#CHECK: rllg	%r15, %r1, 0            # encoding: [0xeb,0xf1,0x00,0x00,0x00,0x1c]
#CHECK: rllg	%r1, %r15, 0            # encoding: [0xeb,0x1f,0x00,0x00,0x00,0x1c]
#CHECK: rllg	%r15, %r15, 0           # encoding: [0xeb,0xff,0x00,0x00,0x00,0x1c]
#CHECK: rllg	%r0, %r0, -524288       # encoding: [0xeb,0x00,0x00,0x00,0x80,0x1c]
#CHECK: rllg	%r0, %r0, -1            # encoding: [0xeb,0x00,0x0f,0xff,0xff,0x1c]
#CHECK: rllg	%r0, %r0, 1             # encoding: [0xeb,0x00,0x00,0x01,0x00,0x1c]
#CHECK: rllg	%r0, %r0, 524287        # encoding: [0xeb,0x00,0x0f,0xff,0x7f,0x1c]
#CHECK: rllg	%r0, %r0, 0(%r1)        # encoding: [0xeb,0x00,0x10,0x00,0x00,0x1c]
#CHECK: rllg	%r0, %r0, 0(%r15)       # encoding: [0xeb,0x00,0xf0,0x00,0x00,0x1c]
#CHECK: rllg	%r0, %r0, 524287(%r1)   # encoding: [0xeb,0x00,0x1f,0xff,0x7f,0x1c]
#CHECK: rllg	%r0, %r0, 524287(%r15)  # encoding: [0xeb,0x00,0xff,0xff,0x7f,0x1c]

	rllg	%r0,%r0,0
	rllg	%r15,%r1,0
	rllg	%r1,%r15,0
	rllg	%r15,%r15,0
	rllg	%r0,%r0,-524288
	rllg	%r0,%r0,-1
	rllg	%r0,%r0,1
	rllg	%r0,%r0,524287
	rllg	%r0,%r0,0(%r1)
	rllg	%r0,%r0,0(%r15)
	rllg	%r0,%r0,524287(%r1)
	rllg	%r0,%r0,524287(%r15)

#CHECK: rnsbg	%r0, %r0, 0, 0, 0       # encoding: [0xec,0x00,0x00,0x00,0x00,0x54]
#CHECK: rnsbg	%r0, %r0, 0, 0, 63      # encoding: [0xec,0x00,0x00,0x00,0x3f,0x54]
#CHECK: rnsbg	%r0, %r0, 0, 0, 64      # encoding: [0xec,0x00,0x00,0x00,0x40,0x54]
#CHECK: rnsbg	%r0, %r0, 0, 0, 255     # encoding: [0xec,0x00,0x00,0x00,0xff,0x54]
#CHECK: rnsbg	%r0, %r0, 0, 255, 0     # encoding: [0xec,0x00,0x00,0xff,0x00,0x54]
#CHECK: rnsbg	%r0, %r0, 255, 0, 0     # encoding: [0xec,0x00,0xff,0x00,0x00,0x54]
#CHECK: rnsbg	%r0, %r15, 0, 0, 0      # encoding: [0xec,0x0f,0x00,0x00,0x00,0x54]
#CHECK: rnsbg	%r15, %r0, 0, 0, 0      # encoding: [0xec,0xf0,0x00,0x00,0x00,0x54]
#CHECK: rnsbg	%r4, %r5, 6, 7, 8       # encoding: [0xec,0x45,0x06,0x07,0x08,0x54]

	rnsbg	%r0,%r0,0,0,0
	rnsbg	%r0,%r0,0,0,63
	rnsbg	%r0,%r0,0,0,64
	rnsbg	%r0,%r0,0,0,255
	rnsbg	%r0,%r0,0,255,0
	rnsbg	%r0,%r0,255,0,0
	rnsbg	%r0,%r15,0,0,0
	rnsbg	%r15,%r0,0,0,0
	rnsbg	%r4,%r5,6,7,8

#CHECK: rosbg	%r0, %r0, 0, 0, 0       # encoding: [0xec,0x00,0x00,0x00,0x00,0x56]
#CHECK: rosbg	%r0, %r0, 0, 0, 63      # encoding: [0xec,0x00,0x00,0x00,0x3f,0x56]
#CHECK: rosbg	%r0, %r0, 0, 0, 64      # encoding: [0xec,0x00,0x00,0x00,0x40,0x56]
#CHECK: rosbg	%r0, %r0, 0, 0, 255     # encoding: [0xec,0x00,0x00,0x00,0xff,0x56]
#CHECK: rosbg	%r0, %r0, 0, 255, 0     # encoding: [0xec,0x00,0x00,0xff,0x00,0x56]
#CHECK: rosbg	%r0, %r0, 255, 0, 0     # encoding: [0xec,0x00,0xff,0x00,0x00,0x56]
#CHECK: rosbg	%r0, %r15, 0, 0, 0      # encoding: [0xec,0x0f,0x00,0x00,0x00,0x56]
#CHECK: rosbg	%r15, %r0, 0, 0, 0      # encoding: [0xec,0xf0,0x00,0x00,0x00,0x56]
#CHECK: rosbg	%r4, %r5, 6, 7, 8       # encoding: [0xec,0x45,0x06,0x07,0x08,0x56]

	rosbg	%r0,%r0,0,0,0
	rosbg	%r0,%r0,0,0,63
	rosbg	%r0,%r0,0,0,64
	rosbg	%r0,%r0,0,0,255
	rosbg	%r0,%r0,0,255,0
	rosbg	%r0,%r0,255,0,0
	rosbg	%r0,%r15,0,0,0
	rosbg	%r15,%r0,0,0,0
	rosbg	%r4,%r5,6,7,8

#CHECK: rp	0                       # encoding: [0xb2,0x77,0x00,0x00]
#CHECK: rp	0(%r1)                  # encoding: [0xb2,0x77,0x10,0x00]
#CHECK: rp	0(%r15)                 # encoding: [0xb2,0x77,0xf0,0x00]
#CHECK: rp	4095                    # encoding: [0xb2,0x77,0x0f,0xff]
#CHECK: rp	4095(%r1)               # encoding: [0xb2,0x77,0x1f,0xff]
#CHECK: rp	4095(%r15)              # encoding: [0xb2,0x77,0xff,0xff]

	rp	0
	rp	0(%r1)
	rp	0(%r15)
	rp	4095
	rp	4095(%r1)
	rp	4095(%r15)

#CHECK: rrbe	%r0, %r0                # encoding: [0xb2,0x2a,0x00,0x00]
#CHECK: rrbe	%r0, %r15               # encoding: [0xb2,0x2a,0x00,0x0f]
#CHECK: rrbe	%r15, %r0               # encoding: [0xb2,0x2a,0x00,0xf0]
#CHECK: rrbe	%r7, %r8                # encoding: [0xb2,0x2a,0x00,0x78]
#CHECK: rrbe	%r15, %r15              # encoding: [0xb2,0x2a,0x00,0xff]

	rrbe	%r0,%r0
	rrbe	%r0,%r15
	rrbe	%r15,%r0
	rrbe	%r7,%r8
	rrbe	%r15,%r15

#CHECK: rrdtr	%f0, %f0, %r0, 0        # encoding: [0xb3,0xf7,0x00,0x00]
#CHECK: rrdtr	%f0, %f0, %r0, 15       # encoding: [0xb3,0xf7,0x0f,0x00]
#CHECK: rrdtr	%f0, %f0, %r15, 0       # encoding: [0xb3,0xf7,0x00,0x0f]
#CHECK: rrdtr	%f0, %f15, %r0, 0       # encoding: [0xb3,0xf7,0xf0,0x00]
#CHECK: rrdtr	%f4, %f5, %r6, 7        # encoding: [0xb3,0xf7,0x57,0x46]
#CHECK: rrdtr	%f15, %f0, %r0, 0       # encoding: [0xb3,0xf7,0x00,0xf0]

	rrdtr	%f0, %f0, %r0, 0
	rrdtr	%f0, %f0, %r0, 15
	rrdtr	%f0, %f0, %r15, 0
	rrdtr	%f0, %f15, %r0, 0
	rrdtr	%f4, %f5, %r6, 7
	rrdtr	%f15, %f0, %r0, 0

#CHECK: rrxtr	%f0, %f0, %r0, 0        # encoding: [0xb3,0xff,0x00,0x00]
#CHECK: rrxtr	%f0, %f0, %r0, 15       # encoding: [0xb3,0xff,0x0f,0x00]
#CHECK: rrxtr	%f0, %f0, %r13, 0       # encoding: [0xb3,0xff,0x00,0x0d]
#CHECK: rrxtr	%f0, %f13, %r0, 0       # encoding: [0xb3,0xff,0xd0,0x00]
#CHECK: rrxtr	%f8, %f8, %r8, 8        # encoding: [0xb3,0xff,0x88,0x88]
#CHECK: rrxtr	%f13, %f0, %r0, 0       # encoding: [0xb3,0xff,0x00,0xd0]

	rrxtr	%f0, %f0, %r0, 0
	rrxtr	%f0, %f0, %r0, 15
	rrxtr	%f0, %f0, %r13, 0
	rrxtr	%f0, %f13, %r0, 0
	rrxtr	%f8, %f8, %r8, 8
	rrxtr	%f13, %f0, %r0, 0

#CHECK: rsch                            # encoding: [0xb2,0x38,0x00,0x00]

	rsch

#CHECK: rxsbg	%r0, %r0, 0, 0, 0       # encoding: [0xec,0x00,0x00,0x00,0x00,0x57]
#CHECK: rxsbg	%r0, %r0, 0, 0, 63      # encoding: [0xec,0x00,0x00,0x00,0x3f,0x57]
#CHECK: rxsbg	%r0, %r0, 0, 0, 64      # encoding: [0xec,0x00,0x00,0x00,0x40,0x57]
#CHECK: rxsbg	%r0, %r0, 0, 0, 255     # encoding: [0xec,0x00,0x00,0x00,0xff,0x57]
#CHECK: rxsbg	%r0, %r0, 0, 255, 0     # encoding: [0xec,0x00,0x00,0xff,0x00,0x57]
#CHECK: rxsbg	%r0, %r0, 255, 0, 0     # encoding: [0xec,0x00,0xff,0x00,0x00,0x57]
#CHECK: rxsbg	%r0, %r15, 0, 0, 0      # encoding: [0xec,0x0f,0x00,0x00,0x00,0x57]
#CHECK: rxsbg	%r15, %r0, 0, 0, 0      # encoding: [0xec,0xf0,0x00,0x00,0x00,0x57]
#CHECK: rxsbg	%r4, %r5, 6, 7, 8       # encoding: [0xec,0x45,0x06,0x07,0x08,0x57]

	rxsbg	%r0,%r0,0,0,0
	rxsbg	%r0,%r0,0,0,63
	rxsbg	%r0,%r0,0,0,64
	rxsbg	%r0,%r0,0,0,255
	rxsbg	%r0,%r0,0,255,0
	rxsbg	%r0,%r0,255,0,0
	rxsbg	%r0,%r15,0,0,0
	rxsbg	%r15,%r0,0,0,0
	rxsbg	%r4,%r5,6,7,8

#CHECK: s	%r0, 0                  # encoding: [0x5b,0x00,0x00,0x00]
#CHECK: s	%r0, 4095               # encoding: [0x5b,0x00,0x0f,0xff]
#CHECK: s	%r0, 0(%r1)             # encoding: [0x5b,0x00,0x10,0x00]
#CHECK: s	%r0, 0(%r15)            # encoding: [0x5b,0x00,0xf0,0x00]
#CHECK: s	%r0, 4095(%r1,%r15)     # encoding: [0x5b,0x01,0xff,0xff]
#CHECK: s	%r0, 4095(%r15,%r1)     # encoding: [0x5b,0x0f,0x1f,0xff]
#CHECK: s	%r15, 0                 # encoding: [0x5b,0xf0,0x00,0x00]

	s	%r0, 0
	s	%r0, 4095
	s	%r0, 0(%r1)
	s	%r0, 0(%r15)
	s	%r0, 4095(%r1,%r15)
	s	%r0, 4095(%r15,%r1)
	s	%r15, 0

#CHECK: sac	0                       # encoding: [0xb2,0x19,0x00,0x00]
#CHECK: sac	0(%r1)                  # encoding: [0xb2,0x19,0x10,0x00]
#CHECK: sac	0(%r15)                 # encoding: [0xb2,0x19,0xf0,0x00]
#CHECK: sac	4095                    # encoding: [0xb2,0x19,0x0f,0xff]
#CHECK: sac	4095(%r1)               # encoding: [0xb2,0x19,0x1f,0xff]
#CHECK: sac	4095(%r15)              # encoding: [0xb2,0x19,0xff,0xff]

	sac	0
	sac	0(%r1)
	sac	0(%r15)
	sac	4095
	sac	4095(%r1)
	sac	4095(%r15)

#CHECK: sacf	0                       # encoding: [0xb2,0x79,0x00,0x00]
#CHECK: sacf	0(%r1)                  # encoding: [0xb2,0x79,0x10,0x00]
#CHECK: sacf	0(%r15)                 # encoding: [0xb2,0x79,0xf0,0x00]
#CHECK: sacf	4095                    # encoding: [0xb2,0x79,0x0f,0xff]
#CHECK: sacf	4095(%r1)               # encoding: [0xb2,0x79,0x1f,0xff]
#CHECK: sacf	4095(%r15)              # encoding: [0xb2,0x79,0xff,0xff]

	sacf	0
	sacf	0(%r1)
	sacf	0(%r15)
	sacf	4095
	sacf	4095(%r1)
	sacf	4095(%r15)

#CHECK: sal                             # encoding: [0xb2,0x37,0x00,0x00]

	sal

#CHECK: sam24                           # encoding: [0x01,0x0c]
#CHECK: sam31                           # encoding: [0x01,0x0d]
#CHECK: sam64                           # encoding: [0x01,0x0e]

	sam24
	sam31
	sam64

#CHECK: sar	%a0, %r0                # encoding: [0xb2,0x4e,0x00,0x00]
#CHECK: sar	%a0, %r15               # encoding: [0xb2,0x4e,0x00,0x0f]
#CHECK: sar	%a15, %r0               # encoding: [0xb2,0x4e,0x00,0xf0]
#CHECK: sar	%a7, %r8                # encoding: [0xb2,0x4e,0x00,0x78]
#CHECK: sar	%a15, %r15              # encoding: [0xb2,0x4e,0x00,0xff]

	sar	%a0, %r0
	sar	%a0, %r15
	sar	%a15, %r0
	sar	%a7, %r8
	sar	%a15, %r15

#CHECK: scctr	%r0, %r0                # encoding: [0xb2,0xe0,0x00,0x00]
#CHECK: scctr	%r0, %r15               # encoding: [0xb2,0xe0,0x00,0x0f]
#CHECK: scctr	%r15, %r0               # encoding: [0xb2,0xe0,0x00,0xf0]
#CHECK: scctr	%r7, %r8                # encoding: [0xb2,0xe0,0x00,0x78]

	scctr	%r0,%r0
	scctr	%r0,%r15
	scctr	%r15,%r0
	scctr	%r7,%r8

#CHECK: schm                            # encoding: [0xb2,0x3c,0x00,0x00]

	schm

#CHECK: sck	0                  	# encoding: [0xb2,0x04,0x00,0x00]
#CHECK: sck	0(%r1)             	# encoding: [0xb2,0x04,0x10,0x00]
#CHECK: sck	0(%r15)            	# encoding: [0xb2,0x04,0xf0,0x00]
#CHECK: sck	4095                 	# encoding: [0xb2,0x04,0x0f,0xff]
#CHECK: sck	4095(%r1)             	# encoding: [0xb2,0x04,0x1f,0xff]
#CHECK: sck	4095(%r15)             	# encoding: [0xb2,0x04,0xff,0xff]

	sck	0
	sck	0(%r1)
	sck	0(%r15)
	sck	4095
	sck	4095(%r1)
	sck	4095(%r15)

#CHECK: sckc	0                  	# encoding: [0xb2,0x06,0x00,0x00]
#CHECK: sckc	0(%r1)             	# encoding: [0xb2,0x06,0x10,0x00]
#CHECK: sckc	0(%r15)            	# encoding: [0xb2,0x06,0xf0,0x00]
#CHECK: sckc	4095                 	# encoding: [0xb2,0x06,0x0f,0xff]
#CHECK: sckc	4095(%r1)             	# encoding: [0xb2,0x06,0x1f,0xff]
#CHECK: sckc	4095(%r15)             	# encoding: [0xb2,0x06,0xff,0xff]

	sckc	0
	sckc	0(%r1)
	sckc	0(%r15)
	sckc	4095
	sckc	4095(%r1)
	sckc	4095(%r15)

#CHECK: sckpf                           # encoding: [0x01,0x07]

	sckpf

#CHECK: sd	%f0, 0                  # encoding: [0x6b,0x00,0x00,0x00]
#CHECK: sd	%f0, 4095               # encoding: [0x6b,0x00,0x0f,0xff]
#CHECK: sd	%f0, 0(%r1)             # encoding: [0x6b,0x00,0x10,0x00]
#CHECK: sd	%f0, 0(%r15)            # encoding: [0x6b,0x00,0xf0,0x00]
#CHECK: sd	%f0, 4095(%r1,%r15)     # encoding: [0x6b,0x01,0xff,0xff]
#CHECK: sd	%f0, 4095(%r15,%r1)     # encoding: [0x6b,0x0f,0x1f,0xff]
#CHECK: sd	%f15, 0                 # encoding: [0x6b,0xf0,0x00,0x00]

	sd	%f0, 0
	sd	%f0, 4095
	sd	%f0, 0(%r1)
	sd	%f0, 0(%r15)
	sd	%f0, 4095(%r1,%r15)
	sd	%f0, 4095(%r15,%r1)
	sd	%f15, 0

#CHECK: sdb	%f0, 0                  # encoding: [0xed,0x00,0x00,0x00,0x00,0x1b]
#CHECK: sdb	%f0, 4095               # encoding: [0xed,0x00,0x0f,0xff,0x00,0x1b]
#CHECK: sdb	%f0, 0(%r1)             # encoding: [0xed,0x00,0x10,0x00,0x00,0x1b]
#CHECK: sdb	%f0, 0(%r15)            # encoding: [0xed,0x00,0xf0,0x00,0x00,0x1b]
#CHECK: sdb	%f0, 4095(%r1,%r15)     # encoding: [0xed,0x01,0xff,0xff,0x00,0x1b]
#CHECK: sdb	%f0, 4095(%r15,%r1)     # encoding: [0xed,0x0f,0x1f,0xff,0x00,0x1b]
#CHECK: sdb	%f15, 0                 # encoding: [0xed,0xf0,0x00,0x00,0x00,0x1b]

	sdb	%f0, 0
	sdb	%f0, 4095
	sdb	%f0, 0(%r1)
	sdb	%f0, 0(%r15)
	sdb	%f0, 4095(%r1,%r15)
	sdb	%f0, 4095(%r15,%r1)
	sdb	%f15, 0

#CHECK: sdbr	%f0, %f0                # encoding: [0xb3,0x1b,0x00,0x00]
#CHECK: sdbr	%f0, %f15               # encoding: [0xb3,0x1b,0x00,0x0f]
#CHECK: sdbr	%f7, %f8                # encoding: [0xb3,0x1b,0x00,0x78]
#CHECK: sdbr	%f15, %f0               # encoding: [0xb3,0x1b,0x00,0xf0]

	sdbr	%f0, %f0
	sdbr	%f0, %f15
	sdbr	%f7, %f8
	sdbr	%f15, %f0

#CHECK: sdr	%f0, %f0                # encoding: [0x2b,0x00]
#CHECK: sdr	%f0, %f15               # encoding: [0x2b,0x0f]
#CHECK: sdr	%f7, %f8                # encoding: [0x2b,0x78]
#CHECK: sdr	%f15, %f0               # encoding: [0x2b,0xf0]

	sdr	%f0, %f0
	sdr	%f0, %f15
	sdr	%f7, %f8
	sdr	%f15, %f0

#CHECK: sdtr	%f0, %f0, %f0           # encoding: [0xb3,0xd3,0x00,0x00]
#CHECK: sdtr	%f0, %f0, %f15          # encoding: [0xb3,0xd3,0xf0,0x00]
#CHECK: sdtr	%f0, %f15, %f0          # encoding: [0xb3,0xd3,0x00,0x0f]
#CHECK: sdtr	%f15, %f0, %f0          # encoding: [0xb3,0xd3,0x00,0xf0]
#CHECK: sdtr	%f7, %f8, %f9           # encoding: [0xb3,0xd3,0x90,0x78]

	sdtr	%f0, %f0, %f0
	sdtr	%f0, %f0, %f15
	sdtr	%f0, %f15, %f0
	sdtr	%f15, %f0, %f0
	sdtr	%f7, %f8, %f9

#CHECK: se	%f0, 0                  # encoding: [0x7b,0x00,0x00,0x00]
#CHECK: se	%f0, 4095               # encoding: [0x7b,0x00,0x0f,0xff]
#CHECK: se	%f0, 0(%r1)             # encoding: [0x7b,0x00,0x10,0x00]
#CHECK: se	%f0, 0(%r15)            # encoding: [0x7b,0x00,0xf0,0x00]
#CHECK: se	%f0, 4095(%r1,%r15)     # encoding: [0x7b,0x01,0xff,0xff]
#CHECK: se	%f0, 4095(%r15,%r1)     # encoding: [0x7b,0x0f,0x1f,0xff]
#CHECK: se	%f15, 0                 # encoding: [0x7b,0xf0,0x00,0x00]

	se	%f0, 0
	se	%f0, 4095
	se	%f0, 0(%r1)
	se	%f0, 0(%r15)
	se	%f0, 4095(%r1,%r15)
	se	%f0, 4095(%r15,%r1)
	se	%f15, 0

#CHECK: seb	%f0, 0                  # encoding: [0xed,0x00,0x00,0x00,0x00,0x0b]
#CHECK: seb	%f0, 4095               # encoding: [0xed,0x00,0x0f,0xff,0x00,0x0b]
#CHECK: seb	%f0, 0(%r1)             # encoding: [0xed,0x00,0x10,0x00,0x00,0x0b]
#CHECK: seb	%f0, 0(%r15)            # encoding: [0xed,0x00,0xf0,0x00,0x00,0x0b]
#CHECK: seb	%f0, 4095(%r1,%r15)     # encoding: [0xed,0x01,0xff,0xff,0x00,0x0b]
#CHECK: seb	%f0, 4095(%r15,%r1)     # encoding: [0xed,0x0f,0x1f,0xff,0x00,0x0b]
#CHECK: seb	%f15, 0                 # encoding: [0xed,0xf0,0x00,0x00,0x00,0x0b]

	seb	%f0, 0
	seb	%f0, 4095
	seb	%f0, 0(%r1)
	seb	%f0, 0(%r15)
	seb	%f0, 4095(%r1,%r15)
	seb	%f0, 4095(%r15,%r1)
	seb	%f15, 0

#CHECK: sebr	%f0, %f0                # encoding: [0xb3,0x0b,0x00,0x00]
#CHECK: sebr	%f0, %f15               # encoding: [0xb3,0x0b,0x00,0x0f]
#CHECK: sebr	%f7, %f8                # encoding: [0xb3,0x0b,0x00,0x78]
#CHECK: sebr	%f15, %f0               # encoding: [0xb3,0x0b,0x00,0xf0]

	sebr	%f0, %f0
	sebr	%f0, %f15
	sebr	%f7, %f8
	sebr	%f15, %f0

#CHECK: ser	%f0, %f0                # encoding: [0x3b,0x00]
#CHECK: ser	%f0, %f15               # encoding: [0x3b,0x0f]
#CHECK: ser	%f7, %f8                # encoding: [0x3b,0x78]
#CHECK: ser	%f15, %f0               # encoding: [0x3b,0xf0]

	ser	%f0, %f0
	ser	%f0, %f15
	ser	%f7, %f8
	ser	%f15, %f0

#CHECK: sfasr	%r0                     # encoding: [0xb3,0x85,0x00,0x00]
#CHECK: sfasr	%r1                     # encoding: [0xb3,0x85,0x00,0x10]
#CHECK: sfasr	%r15                    # encoding: [0xb3,0x85,0x00,0xf0]

	sfasr	%r0
	sfasr	%r1
	sfasr	%r15

#CHECK: sfpc	%r0                     # encoding: [0xb3,0x84,0x00,0x00]
#CHECK: sfpc	%r1                     # encoding: [0xb3,0x84,0x00,0x10]
#CHECK: sfpc	%r15                    # encoding: [0xb3,0x84,0x00,0xf0]

	sfpc	%r0
	sfpc	%r1
	sfpc	%r15

#CHECK: sg	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x09]
#CHECK: sg	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x09]
#CHECK: sg	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x09]
#CHECK: sg	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x09]
#CHECK: sg	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x09]
#CHECK: sg	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x09]
#CHECK: sg	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x09]
#CHECK: sg	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x09]
#CHECK: sg	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x09]
#CHECK: sg	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x09]

	sg	%r0, -524288
	sg	%r0, -1
	sg	%r0, 0
	sg	%r0, 1
	sg	%r0, 524287
	sg	%r0, 0(%r1)
	sg	%r0, 0(%r15)
	sg	%r0, 524287(%r1,%r15)
	sg	%r0, 524287(%r15,%r1)
	sg	%r15, 0

#CHECK: sgf	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x19]
#CHECK: sgf	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x19]
#CHECK: sgf	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x19]
#CHECK: sgf	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x19]
#CHECK: sgf	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x19]
#CHECK: sgf	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x19]
#CHECK: sgf	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x19]
#CHECK: sgf	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x19]
#CHECK: sgf	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x19]
#CHECK: sgf	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x19]

	sgf	%r0, -524288
	sgf	%r0, -1
	sgf	%r0, 0
	sgf	%r0, 1
	sgf	%r0, 524287
	sgf	%r0, 0(%r1)
	sgf	%r0, 0(%r15)
	sgf	%r0, 524287(%r1,%r15)
	sgf	%r0, 524287(%r15,%r1)
	sgf	%r15, 0

#CHECK: sgfr	%r0, %r0                # encoding: [0xb9,0x19,0x00,0x00]
#CHECK: sgfr	%r0, %r15               # encoding: [0xb9,0x19,0x00,0x0f]
#CHECK: sgfr	%r15, %r0               # encoding: [0xb9,0x19,0x00,0xf0]
#CHECK: sgfr	%r7, %r8                # encoding: [0xb9,0x19,0x00,0x78]

	sgfr	%r0,%r0
	sgfr	%r0,%r15
	sgfr	%r15,%r0
	sgfr	%r7,%r8

#CHECK: sgr	%r0, %r0                # encoding: [0xb9,0x09,0x00,0x00]
#CHECK: sgr	%r0, %r15               # encoding: [0xb9,0x09,0x00,0x0f]
#CHECK: sgr	%r15, %r0               # encoding: [0xb9,0x09,0x00,0xf0]
#CHECK: sgr	%r7, %r8                # encoding: [0xb9,0x09,0x00,0x78]

	sgr	%r0,%r0
	sgr	%r0,%r15
	sgr	%r15,%r0
	sgr	%r7,%r8

#CHECK: sh	%r0, 0                  # encoding: [0x4b,0x00,0x00,0x00]
#CHECK: sh	%r0, 4095               # encoding: [0x4b,0x00,0x0f,0xff]
#CHECK: sh	%r0, 0(%r1)             # encoding: [0x4b,0x00,0x10,0x00]
#CHECK: sh	%r0, 0(%r15)            # encoding: [0x4b,0x00,0xf0,0x00]
#CHECK: sh	%r0, 4095(%r1,%r15)     # encoding: [0x4b,0x01,0xff,0xff]
#CHECK: sh	%r0, 4095(%r15,%r1)     # encoding: [0x4b,0x0f,0x1f,0xff]
#CHECK: sh	%r15, 0                 # encoding: [0x4b,0xf0,0x00,0x00]

	sh	%r0, 0
	sh	%r0, 4095
	sh	%r0, 0(%r1)
	sh	%r0, 0(%r15)
	sh	%r0, 4095(%r1,%r15)
	sh	%r0, 4095(%r15,%r1)
	sh	%r15, 0

#CHECK: shy	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x7b]
#CHECK: shy	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x7b]
#CHECK: shy	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x7b]
#CHECK: shy	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x7b]
#CHECK: shy	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x7b]
#CHECK: shy	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x7b]
#CHECK: shy	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x7b]
#CHECK: shy	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x7b]
#CHECK: shy	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x7b]
#CHECK: shy	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x7b]

	shy	%r0, -524288
	shy	%r0, -1
	shy	%r0, 0
	shy	%r0, 1
	shy	%r0, 524287
	shy	%r0, 0(%r1)
	shy	%r0, 0(%r15)
	shy	%r0, 524287(%r1,%r15)
	shy	%r0, 524287(%r15,%r1)
	shy	%r15, 0

#CHECK: sie	0                       # encoding: [0xb2,0x14,0x00,0x00]
#CHECK: sie	0(%r1)                  # encoding: [0xb2,0x14,0x10,0x00]
#CHECK: sie	0(%r15)                 # encoding: [0xb2,0x14,0xf0,0x00]
#CHECK: sie	4095                    # encoding: [0xb2,0x14,0x0f,0xff]
#CHECK: sie	4095(%r1)               # encoding: [0xb2,0x14,0x1f,0xff]
#CHECK: sie	4095(%r15)              # encoding: [0xb2,0x14,0xff,0xff]

	sie	0
	sie	0(%r1)
	sie	0(%r15)
	sie	4095
	sie	4095(%r1)
	sie	4095(%r15)

#CHECK: siga	0                       # encoding: [0xb2,0x74,0x00,0x00]
#CHECK: siga	0(%r1)                  # encoding: [0xb2,0x74,0x10,0x00]
#CHECK: siga	0(%r15)                 # encoding: [0xb2,0x74,0xf0,0x00]
#CHECK: siga	4095                    # encoding: [0xb2,0x74,0x0f,0xff]
#CHECK: siga	4095(%r1)               # encoding: [0xb2,0x74,0x1f,0xff]
#CHECK: siga	4095(%r15)              # encoding: [0xb2,0x74,0xff,0xff]

	siga	0
	siga	0(%r1)
	siga	0(%r15)
	siga	4095
	siga	4095(%r1)
	siga	4095(%r15)

#CHECK: sigp	%r0, %r0, 0             # encoding: [0xae,0x00,0x00,0x00]
#CHECK: sigp	%r0, %r15, 0            # encoding: [0xae,0x0f,0x00,0x00]
#CHECK: sigp	%r14, %r15, 0           # encoding: [0xae,0xef,0x00,0x00]
#CHECK: sigp	%r15, %r15, 0           # encoding: [0xae,0xff,0x00,0x00]
#CHECK: sigp	%r0, %r0, 4095          # encoding: [0xae,0x00,0x0f,0xff]
#CHECK: sigp	%r0, %r0, 1             # encoding: [0xae,0x00,0x00,0x01]
#CHECK: sigp	%r0, %r0, 0(%r1)        # encoding: [0xae,0x00,0x10,0x00]
#CHECK: sigp	%r0, %r0, 0(%r15)       # encoding: [0xae,0x00,0xf0,0x00]
#CHECK: sigp	%r0, %r0, 4095(%r1)     # encoding: [0xae,0x00,0x1f,0xff]
#CHECK: sigp	%r0, %r0, 4095(%r15)    # encoding: [0xae,0x00,0xff,0xff]

	sigp	%r0,%r0,0
	sigp	%r0,%r15,0
	sigp	%r14,%r15,0
	sigp	%r15,%r15,0
	sigp	%r0,%r0,4095
	sigp	%r0,%r0,1
	sigp	%r0,%r0,0(%r1)
	sigp	%r0,%r0,0(%r15)
	sigp	%r0,%r0,4095(%r1)
	sigp	%r0,%r0,4095(%r15)

#CHECK: sl	%r0, 0                  # encoding: [0x5f,0x00,0x00,0x00]
#CHECK: sl	%r0, 4095               # encoding: [0x5f,0x00,0x0f,0xff]
#CHECK: sl	%r0, 0(%r1)             # encoding: [0x5f,0x00,0x10,0x00]
#CHECK: sl	%r0, 0(%r15)            # encoding: [0x5f,0x00,0xf0,0x00]
#CHECK: sl	%r0, 4095(%r1,%r15)     # encoding: [0x5f,0x01,0xff,0xff]
#CHECK: sl	%r0, 4095(%r15,%r1)     # encoding: [0x5f,0x0f,0x1f,0xff]
#CHECK: sl	%r15, 0                 # encoding: [0x5f,0xf0,0x00,0x00]

	sl	%r0, 0
	sl	%r0, 4095
	sl	%r0, 0(%r1)
	sl	%r0, 0(%r15)
	sl	%r0, 4095(%r1,%r15)
	sl	%r0, 4095(%r15,%r1)
	sl	%r15, 0

#CHECK: sla	%r0, 0                  # encoding: [0x8b,0x00,0x00,0x00]
#CHECK: sla	%r7, 0                  # encoding: [0x8b,0x70,0x00,0x00]
#CHECK: sla	%r15, 0                 # encoding: [0x8b,0xf0,0x00,0x00]
#CHECK: sla	%r0, 4095               # encoding: [0x8b,0x00,0x0f,0xff]
#CHECK: sla	%r0, 0(%r1)             # encoding: [0x8b,0x00,0x10,0x00]
#CHECK: sla	%r0, 0(%r15)            # encoding: [0x8b,0x00,0xf0,0x00]
#CHECK: sla	%r0, 4095(%r1)          # encoding: [0x8b,0x00,0x1f,0xff]
#CHECK: sla	%r0, 4095(%r15)         # encoding: [0x8b,0x00,0xff,0xff]

	sla	%r0,0
	sla	%r7,0
	sla	%r15,0
	sla	%r0,4095
	sla	%r0,0(%r1)
	sla	%r0,0(%r15)
	sla	%r0,4095(%r1)
	sla	%r0,4095(%r15)

#CHECK: slag	%r0, %r0, 0             # encoding: [0xeb,0x00,0x00,0x00,0x00,0x0b]
#CHECK: slag	%r15, %r1, 0            # encoding: [0xeb,0xf1,0x00,0x00,0x00,0x0b]
#CHECK: slag	%r1, %r15, 0            # encoding: [0xeb,0x1f,0x00,0x00,0x00,0x0b]
#CHECK: slag	%r15, %r15, 0           # encoding: [0xeb,0xff,0x00,0x00,0x00,0x0b]
#CHECK: slag	%r0, %r0, -524288       # encoding: [0xeb,0x00,0x00,0x00,0x80,0x0b]
#CHECK: slag	%r0, %r0, -1            # encoding: [0xeb,0x00,0x0f,0xff,0xff,0x0b]
#CHECK: slag	%r0, %r0, 1             # encoding: [0xeb,0x00,0x00,0x01,0x00,0x0b]
#CHECK: slag	%r0, %r0, 524287        # encoding: [0xeb,0x00,0x0f,0xff,0x7f,0x0b]
#CHECK: slag	%r0, %r0, 0(%r1)        # encoding: [0xeb,0x00,0x10,0x00,0x00,0x0b]
#CHECK: slag	%r0, %r0, 0(%r15)       # encoding: [0xeb,0x00,0xf0,0x00,0x00,0x0b]
#CHECK: slag	%r0, %r0, 524287(%r1)   # encoding: [0xeb,0x00,0x1f,0xff,0x7f,0x0b]
#CHECK: slag	%r0, %r0, 524287(%r15)  # encoding: [0xeb,0x00,0xff,0xff,0x7f,0x0b]

	slag	%r0,%r0,0
	slag	%r15,%r1,0
	slag	%r1,%r15,0
	slag	%r15,%r15,0
	slag	%r0,%r0,-524288
	slag	%r0,%r0,-1
	slag	%r0,%r0,1
	slag	%r0,%r0,524287
	slag	%r0,%r0,0(%r1)
	slag	%r0,%r0,0(%r15)
	slag	%r0,%r0,524287(%r1)
	slag	%r0,%r0,524287(%r15)

#CHECK: slb	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x99]
#CHECK: slb	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x99]
#CHECK: slb	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x99]
#CHECK: slb	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x99]
#CHECK: slb	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x99]
#CHECK: slb	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x99]
#CHECK: slb	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x99]
#CHECK: slb	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x99]
#CHECK: slb	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x99]
#CHECK: slb	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x99]

	slb	%r0, -524288
	slb	%r0, -1
	slb	%r0, 0
	slb	%r0, 1
	slb	%r0, 524287
	slb	%r0, 0(%r1)
	slb	%r0, 0(%r15)
	slb	%r0, 524287(%r1,%r15)
	slb	%r0, 524287(%r15,%r1)
	slb	%r15, 0

#CHECK: slbg	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x89]
#CHECK: slbg	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x89]
#CHECK: slbg	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x89]
#CHECK: slbg	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x89]
#CHECK: slbg	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x89]
#CHECK: slbg	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x89]
#CHECK: slbg	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x89]
#CHECK: slbg	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x89]
#CHECK: slbg	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x89]
#CHECK: slbg	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x89]

	slbg	%r0, -524288
	slbg	%r0, -1
	slbg	%r0, 0
	slbg	%r0, 1
	slbg	%r0, 524287
	slbg	%r0, 0(%r1)
	slbg	%r0, 0(%r15)
	slbg	%r0, 524287(%r1,%r15)
	slbg	%r0, 524287(%r15,%r1)
	slbg	%r15, 0

#CHECK: slbgr	%r0, %r0                # encoding: [0xb9,0x89,0x00,0x00]
#CHECK: slbgr	%r0, %r15               # encoding: [0xb9,0x89,0x00,0x0f]
#CHECK: slbgr	%r15, %r0               # encoding: [0xb9,0x89,0x00,0xf0]
#CHECK: slbgr	%r7, %r8                # encoding: [0xb9,0x89,0x00,0x78]

	slbgr	%r0,%r0
	slbgr	%r0,%r15
	slbgr	%r15,%r0
	slbgr	%r7,%r8

#CHECK: slbr	%r0, %r0                # encoding: [0xb9,0x99,0x00,0x00]
#CHECK: slbr	%r0, %r15               # encoding: [0xb9,0x99,0x00,0x0f]
#CHECK: slbr	%r15, %r0               # encoding: [0xb9,0x99,0x00,0xf0]
#CHECK: slbr	%r7, %r8                # encoding: [0xb9,0x99,0x00,0x78]

	slbr	%r0,%r0
	slbr	%r0,%r15
	slbr	%r15,%r0
	slbr	%r7,%r8

#CHECK: slda	%r0, 0                  # encoding: [0x8f,0x00,0x00,0x00]
#CHECK: slda	%r6, 0                  # encoding: [0x8f,0x60,0x00,0x00]
#CHECK: slda	%r14, 0                 # encoding: [0x8f,0xe0,0x00,0x00]
#CHECK: slda	%r0, 4095               # encoding: [0x8f,0x00,0x0f,0xff]
#CHECK: slda	%r0, 0(%r1)             # encoding: [0x8f,0x00,0x10,0x00]
#CHECK: slda	%r0, 0(%r15)            # encoding: [0x8f,0x00,0xf0,0x00]
#CHECK: slda	%r0, 4095(%r1)          # encoding: [0x8f,0x00,0x1f,0xff]
#CHECK: slda	%r0, 4095(%r15)         # encoding: [0x8f,0x00,0xff,0xff]

	slda	%r0,0
	slda	%r6,0
	slda	%r14,0
	slda	%r0,4095
	slda	%r0,0(%r1)
	slda	%r0,0(%r15)
	slda	%r0,4095(%r1)
	slda	%r0,4095(%r15)

#CHECK: sldl	%r0, 0                  # encoding: [0x8d,0x00,0x00,0x00]
#CHECK: sldl	%r6, 0                  # encoding: [0x8d,0x60,0x00,0x00]
#CHECK: sldl	%r14, 0                 # encoding: [0x8d,0xe0,0x00,0x00]
#CHECK: sldl	%r0, 4095               # encoding: [0x8d,0x00,0x0f,0xff]
#CHECK: sldl	%r0, 0(%r1)             # encoding: [0x8d,0x00,0x10,0x00]
#CHECK: sldl	%r0, 0(%r15)            # encoding: [0x8d,0x00,0xf0,0x00]
#CHECK: sldl	%r0, 4095(%r1)          # encoding: [0x8d,0x00,0x1f,0xff]
#CHECK: sldl	%r0, 4095(%r15)         # encoding: [0x8d,0x00,0xff,0xff]

	sldl	%r0,0
	sldl	%r6,0
	sldl	%r14,0
	sldl	%r0,4095
	sldl	%r0,0(%r1)
	sldl	%r0,0(%r15)
	sldl	%r0,4095(%r1)
	sldl	%r0,4095(%r15)

#CHECK: sldt	%f0, %f0, 0             # encoding: [0xed,0x00,0x00,0x00,0x00,0x40]
#CHECK: sldt	%f0, %f0, 4095          # encoding: [0xed,0x00,0x0f,0xff,0x00,0x40]
#CHECK: sldt	%f0, %f0, 0(%r1)        # encoding: [0xed,0x00,0x10,0x00,0x00,0x40]
#CHECK: sldt	%f0, %f0, 0(%r15)       # encoding: [0xed,0x00,0xf0,0x00,0x00,0x40]
#CHECK: sldt	%f0, %f0, 4095(%r1,%r15) # encoding: [0xed,0x01,0xff,0xff,0x00,0x40]
#CHECK: sldt	%f0, %f0, 4095(%r15,%r1) # encoding: [0xed,0x0f,0x1f,0xff,0x00,0x40]
#CHECK: sldt	%f0, %f15, 0            # encoding: [0xed,0xf0,0x00,0x00,0x00,0x40]
#CHECK: sldt	%f15, %f0, 0            # encoding: [0xed,0x00,0x00,0x00,0xf0,0x40]
#CHECK: sldt	%f15, %f15, 0           # encoding: [0xed,0xf0,0x00,0x00,0xf0,0x40]

	sldt	%f0, %f0, 0
	sldt	%f0, %f0, 4095
	sldt	%f0, %f0, 0(%r1)
	sldt	%f0, %f0, 0(%r15)
	sldt	%f0, %f0, 4095(%r1,%r15)
	sldt	%f0, %f0, 4095(%r15,%r1)
	sldt	%f0, %f15, 0
	sldt	%f15, %f0, 0
	sldt	%f15, %f15, 0

#CHECK: slfi	%r0, 0                  # encoding: [0xc2,0x05,0x00,0x00,0x00,0x00]
#CHECK: slfi	%r0, 4294967295         # encoding: [0xc2,0x05,0xff,0xff,0xff,0xff]
#CHECK: slfi	%r15, 0                 # encoding: [0xc2,0xf5,0x00,0x00,0x00,0x00]

	slfi	%r0, 0
	slfi	%r0, (1 << 32) - 1
	slfi	%r15, 0

#CHECK: slg	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x0b]
#CHECK: slg	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x0b]
#CHECK: slg	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x0b]
#CHECK: slg	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x0b]
#CHECK: slg	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x0b]
#CHECK: slg	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x0b]
#CHECK: slg	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x0b]
#CHECK: slg	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x0b]
#CHECK: slg	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x0b]
#CHECK: slg	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x0b]

	slg	%r0, -524288
	slg	%r0, -1
	slg	%r0, 0
	slg	%r0, 1
	slg	%r0, 524287
	slg	%r0, 0(%r1)
	slg	%r0, 0(%r15)
	slg	%r0, 524287(%r1,%r15)
	slg	%r0, 524287(%r15,%r1)
	slg	%r15, 0

#CHECK: slgf	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x1b]
#CHECK: slgf	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x1b]
#CHECK: slgf	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x1b]
#CHECK: slgf	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x1b]
#CHECK: slgf	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x1b]
#CHECK: slgf	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x1b]
#CHECK: slgf	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x1b]
#CHECK: slgf	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x1b]
#CHECK: slgf	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x1b]
#CHECK: slgf	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x1b]

	slgf	%r0, -524288
	slgf	%r0, -1
	slgf	%r0, 0
	slgf	%r0, 1
	slgf	%r0, 524287
	slgf	%r0, 0(%r1)
	slgf	%r0, 0(%r15)
	slgf	%r0, 524287(%r1,%r15)
	slgf	%r0, 524287(%r15,%r1)
	slgf	%r15, 0

#CHECK: slgfi	%r0, 0                  # encoding: [0xc2,0x04,0x00,0x00,0x00,0x00]
#CHECK: slgfi	%r0, 4294967295         # encoding: [0xc2,0x04,0xff,0xff,0xff,0xff]
#CHECK: slgfi	%r15, 0                 # encoding: [0xc2,0xf4,0x00,0x00,0x00,0x00]

	slgfi	%r0, 0
	slgfi	%r0, (1 << 32) - 1
	slgfi	%r15, 0

#CHECK: slgfr	%r0, %r0                # encoding: [0xb9,0x1b,0x00,0x00]
#CHECK: slgfr	%r0, %r15               # encoding: [0xb9,0x1b,0x00,0x0f]
#CHECK: slgfr	%r15, %r0               # encoding: [0xb9,0x1b,0x00,0xf0]
#CHECK: slgfr	%r7, %r8                # encoding: [0xb9,0x1b,0x00,0x78]

	slgfr	%r0,%r0
	slgfr	%r0,%r15
	slgfr	%r15,%r0
	slgfr	%r7,%r8

#CHECK: slgr	%r0, %r0                # encoding: [0xb9,0x0b,0x00,0x00]
#CHECK: slgr	%r0, %r15               # encoding: [0xb9,0x0b,0x00,0x0f]
#CHECK: slgr	%r15, %r0               # encoding: [0xb9,0x0b,0x00,0xf0]
#CHECK: slgr	%r7, %r8                # encoding: [0xb9,0x0b,0x00,0x78]

	slgr	%r0,%r0
	slgr	%r0,%r15
	slgr	%r15,%r0
	slgr	%r7,%r8

#CHECK: sll	%r0, 0                  # encoding: [0x89,0x00,0x00,0x00]
#CHECK: sll	%r7, 0                  # encoding: [0x89,0x70,0x00,0x00]
#CHECK: sll	%r15, 0                 # encoding: [0x89,0xf0,0x00,0x00]
#CHECK: sll	%r0, 4095               # encoding: [0x89,0x00,0x0f,0xff]
#CHECK: sll	%r0, 0(%r1)             # encoding: [0x89,0x00,0x10,0x00]
#CHECK: sll	%r0, 0(%r15)            # encoding: [0x89,0x00,0xf0,0x00]
#CHECK: sll	%r0, 4095(%r1)          # encoding: [0x89,0x00,0x1f,0xff]
#CHECK: sll	%r0, 4095(%r15)         # encoding: [0x89,0x00,0xff,0xff]

	sll	%r0,0
	sll	%r7,0
	sll	%r15,0
	sll	%r0,4095
	sll	%r0,0(%r1)
	sll	%r0,0(%r15)
	sll	%r0,4095(%r1)
	sll	%r0,4095(%r15)

#CHECK: sllg	%r0, %r0, 0             # encoding: [0xeb,0x00,0x00,0x00,0x00,0x0d]
#CHECK: sllg	%r15, %r1, 0            # encoding: [0xeb,0xf1,0x00,0x00,0x00,0x0d]
#CHECK: sllg	%r1, %r15, 0            # encoding: [0xeb,0x1f,0x00,0x00,0x00,0x0d]
#CHECK: sllg	%r15, %r15, 0           # encoding: [0xeb,0xff,0x00,0x00,0x00,0x0d]
#CHECK: sllg	%r0, %r0, -524288       # encoding: [0xeb,0x00,0x00,0x00,0x80,0x0d]
#CHECK: sllg	%r0, %r0, -1            # encoding: [0xeb,0x00,0x0f,0xff,0xff,0x0d]
#CHECK: sllg	%r0, %r0, 1             # encoding: [0xeb,0x00,0x00,0x01,0x00,0x0d]
#CHECK: sllg	%r0, %r0, 524287        # encoding: [0xeb,0x00,0x0f,0xff,0x7f,0x0d]
#CHECK: sllg	%r0, %r0, 0(%r1)        # encoding: [0xeb,0x00,0x10,0x00,0x00,0x0d]
#CHECK: sllg	%r0, %r0, 0(%r15)       # encoding: [0xeb,0x00,0xf0,0x00,0x00,0x0d]
#CHECK: sllg	%r0, %r0, 524287(%r1)   # encoding: [0xeb,0x00,0x1f,0xff,0x7f,0x0d]
#CHECK: sllg	%r0, %r0, 524287(%r15)  # encoding: [0xeb,0x00,0xff,0xff,0x7f,0x0d]

	sllg	%r0,%r0,0
	sllg	%r15,%r1,0
	sllg	%r1,%r15,0
	sllg	%r15,%r15,0
	sllg	%r0,%r0,-524288
	sllg	%r0,%r0,-1
	sllg	%r0,%r0,1
	sllg	%r0,%r0,524287
	sllg	%r0,%r0,0(%r1)
	sllg	%r0,%r0,0(%r15)
	sllg	%r0,%r0,524287(%r1)
	sllg	%r0,%r0,524287(%r15)

#CHECK: slr	%r0, %r0                # encoding: [0x1f,0x00]
#CHECK: slr	%r0, %r15               # encoding: [0x1f,0x0f]
#CHECK: slr	%r15, %r0               # encoding: [0x1f,0xf0]
#CHECK: slr	%r7, %r8                # encoding: [0x1f,0x78]

	slr	%r0,%r0
	slr	%r0,%r15
	slr	%r15,%r0
	slr	%r7,%r8

#CHECK: slxt	%f0, %f0, 0             # encoding: [0xed,0x00,0x00,0x00,0x00,0x48]
#CHECK: slxt	%f0, %f0, 4095          # encoding: [0xed,0x00,0x0f,0xff,0x00,0x48]
#CHECK: slxt	%f0, %f0, 0(%r1)        # encoding: [0xed,0x00,0x10,0x00,0x00,0x48]
#CHECK: slxt	%f0, %f0, 0(%r15)       # encoding: [0xed,0x00,0xf0,0x00,0x00,0x48]
#CHECK: slxt	%f0, %f0, 4095(%r1,%r15) # encoding: [0xed,0x01,0xff,0xff,0x00,0x48]
#CHECK: slxt	%f0, %f0, 4095(%r15,%r1) # encoding: [0xed,0x0f,0x1f,0xff,0x00,0x48]
#CHECK: slxt	%f0, %f13, 0            # encoding: [0xed,0xd0,0x00,0x00,0x00,0x48]
#CHECK: slxt	%f13, %f0, 0            # encoding: [0xed,0x00,0x00,0x00,0xd0,0x48]
#CHECK: slxt	%f13, %f13, 0           # encoding: [0xed,0xd0,0x00,0x00,0xd0,0x48]

	slxt	%f0, %f0, 0
	slxt	%f0, %f0, 4095
	slxt	%f0, %f0, 0(%r1)
	slxt	%f0, %f0, 0(%r15)
	slxt	%f0, %f0, 4095(%r1,%r15)
	slxt	%f0, %f0, 4095(%r15,%r1)
	slxt	%f0, %f13, 0
	slxt	%f13, %f0, 0
	slxt	%f13, %f13, 0

#CHECK: sly	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x5f]
#CHECK: sly	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x5f]
#CHECK: sly	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x5f]
#CHECK: sly	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x5f]
#CHECK: sly	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x5f]
#CHECK: sly	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x5f]
#CHECK: sly	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x5f]
#CHECK: sly	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x5f]
#CHECK: sly	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x5f]
#CHECK: sly	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x5f]

	sly	%r0, -524288
	sly	%r0, -1
	sly	%r0, 0
	sly	%r0, 1
	sly	%r0, 524287
	sly	%r0, 0(%r1)
	sly	%r0, 0(%r15)
	sly	%r0, 524287(%r1,%r15)
	sly	%r0, 524287(%r15,%r1)
	sly	%r15, 0

#CHECK: sp	0(1), 0(1)              # encoding: [0xfb,0x00,0x00,0x00,0x00,0x00]
#CHECK: sp	0(1), 0(1,%r1)          # encoding: [0xfb,0x00,0x00,0x00,0x10,0x00]
#CHECK: sp	0(1), 0(1,%r15)         # encoding: [0xfb,0x00,0x00,0x00,0xf0,0x00]
#CHECK: sp	0(1), 4095(1)           # encoding: [0xfb,0x00,0x00,0x00,0x0f,0xff]
#CHECK: sp	0(1), 4095(1,%r1)       # encoding: [0xfb,0x00,0x00,0x00,0x1f,0xff]
#CHECK: sp	0(1), 4095(1,%r15)      # encoding: [0xfb,0x00,0x00,0x00,0xff,0xff]
#CHECK: sp	0(1,%r1), 0(1)          # encoding: [0xfb,0x00,0x10,0x00,0x00,0x00]
#CHECK: sp	0(1,%r15), 0(1)         # encoding: [0xfb,0x00,0xf0,0x00,0x00,0x00]
#CHECK: sp	4095(1,%r1), 0(1)       # encoding: [0xfb,0x00,0x1f,0xff,0x00,0x00]
#CHECK: sp	4095(1,%r15), 0(1)      # encoding: [0xfb,0x00,0xff,0xff,0x00,0x00]
#CHECK: sp	0(16,%r1), 0(1)         # encoding: [0xfb,0xf0,0x10,0x00,0x00,0x00]
#CHECK: sp	0(16,%r15), 0(1)        # encoding: [0xfb,0xf0,0xf0,0x00,0x00,0x00]
#CHECK: sp	0(1), 0(16,%r1)         # encoding: [0xfb,0x0f,0x00,0x00,0x10,0x00]
#CHECK: sp	0(1), 0(16,%r15)        # encoding: [0xfb,0x0f,0x00,0x00,0xf0,0x00]

	sp	0(1), 0(1)
	sp	0(1), 0(1,%r1)
	sp	0(1), 0(1,%r15)
	sp	0(1), 4095(1)
	sp	0(1), 4095(1,%r1)
	sp	0(1), 4095(1,%r15)
	sp	0(1,%r1), 0(1)
	sp	0(1,%r15), 0(1)
	sp	4095(1,%r1), 0(1)
	sp	4095(1,%r15), 0(1)
	sp	0(16,%r1), 0(1)
	sp	0(16,%r15), 0(1)
	sp	0(1), 0(16,%r1)
	sp	0(1), 0(16,%r15)

#CHECK: spctr	%r0, %r0                # encoding: [0xb2,0xe1,0x00,0x00]
#CHECK: spctr	%r0, %r15               # encoding: [0xb2,0xe1,0x00,0x0f]
#CHECK: spctr	%r15, %r0               # encoding: [0xb2,0xe1,0x00,0xf0]
#CHECK: spctr	%r7, %r8                # encoding: [0xb2,0xe1,0x00,0x78]

	spctr	%r0,%r0
	spctr	%r0,%r15
	spctr	%r15,%r0
	spctr	%r7,%r8

#CHECK: spka	0                       # encoding: [0xb2,0x0a,0x00,0x00]
#CHECK: spka	0(%r1)                  # encoding: [0xb2,0x0a,0x10,0x00]
#CHECK: spka	0(%r15)                 # encoding: [0xb2,0x0a,0xf0,0x00]
#CHECK: spka	4095                    # encoding: [0xb2,0x0a,0x0f,0xff]
#CHECK: spka	4095(%r1)               # encoding: [0xb2,0x0a,0x1f,0xff]
#CHECK: spka	4095(%r15)              # encoding: [0xb2,0x0a,0xff,0xff]

	spka	0
	spka	0(%r1)
	spka	0(%r15)
	spka	4095
	spka	4095(%r1)
	spka	4095(%r15)

#CHECK: spm	%r0                     # encoding: [0x04,0x00]
#CHECK: spm	%r1                     # encoding: [0x04,0x10]
#CHECK: spm	%r15                    # encoding: [0x04,0xf0]

	spm	%r0
	spm	%r1
	spm	%r15

#CHECK: spt	0                  	# encoding: [0xb2,0x08,0x00,0x00]
#CHECK: spt	0(%r1)             	# encoding: [0xb2,0x08,0x10,0x00]
#CHECK: spt	0(%r15)            	# encoding: [0xb2,0x08,0xf0,0x00]
#CHECK: spt	4095                 	# encoding: [0xb2,0x08,0x0f,0xff]
#CHECK: spt	4095(%r1)             	# encoding: [0xb2,0x08,0x1f,0xff]
#CHECK: spt	4095(%r15)             	# encoding: [0xb2,0x08,0xff,0xff]

	spt	0
	spt	0(%r1)
	spt	0(%r15)
	spt	4095
	spt	4095(%r1)
	spt	4095(%r15)

#CHECK: spx	0                       # encoding: [0xb2,0x10,0x00,0x00]
#CHECK: spx	0(%r1)                  # encoding: [0xb2,0x10,0x10,0x00]
#CHECK: spx	0(%r15)                 # encoding: [0xb2,0x10,0xf0,0x00]
#CHECK: spx	4095                    # encoding: [0xb2,0x10,0x0f,0xff]
#CHECK: spx	4095(%r1)               # encoding: [0xb2,0x10,0x1f,0xff]
#CHECK: spx	4095(%r15)              # encoding: [0xb2,0x10,0xff,0xff]

	spx	0
	spx	0(%r1)
	spx	0(%r15)
	spx	4095
	spx	4095(%r1)
	spx	4095(%r15)

#CHECK: sqd	%f0, 0                  # encoding: [0xed,0x00,0x00,0x00,0x00,0x35]
#CHECK: sqd	%f0, 4095               # encoding: [0xed,0x00,0x0f,0xff,0x00,0x35]
#CHECK: sqd	%f0, 0(%r1)             # encoding: [0xed,0x00,0x10,0x00,0x00,0x35]
#CHECK: sqd	%f0, 0(%r15)            # encoding: [0xed,0x00,0xf0,0x00,0x00,0x35]
#CHECK: sqd	%f0, 4095(%r1,%r15)     # encoding: [0xed,0x01,0xff,0xff,0x00,0x35]
#CHECK: sqd	%f0, 4095(%r15,%r1)     # encoding: [0xed,0x0f,0x1f,0xff,0x00,0x35]
#CHECK: sqd	%f15, 0                 # encoding: [0xed,0xf0,0x00,0x00,0x00,0x35]

	sqd	%f0, 0
	sqd	%f0, 4095
	sqd	%f0, 0(%r1)
	sqd	%f0, 0(%r15)
	sqd	%f0, 4095(%r1,%r15)
	sqd	%f0, 4095(%r15,%r1)
	sqd	%f15, 0

#CHECK: sqdb	%f0, 0                  # encoding: [0xed,0x00,0x00,0x00,0x00,0x15]
#CHECK: sqdb	%f0, 4095               # encoding: [0xed,0x00,0x0f,0xff,0x00,0x15]
#CHECK: sqdb	%f0, 0(%r1)             # encoding: [0xed,0x00,0x10,0x00,0x00,0x15]
#CHECK: sqdb	%f0, 0(%r15)            # encoding: [0xed,0x00,0xf0,0x00,0x00,0x15]
#CHECK: sqdb	%f0, 4095(%r1,%r15)     # encoding: [0xed,0x01,0xff,0xff,0x00,0x15]
#CHECK: sqdb	%f0, 4095(%r15,%r1)     # encoding: [0xed,0x0f,0x1f,0xff,0x00,0x15]
#CHECK: sqdb	%f15, 0                 # encoding: [0xed,0xf0,0x00,0x00,0x00,0x15]

	sqdb	%f0, 0
	sqdb	%f0, 4095
	sqdb	%f0, 0(%r1)
	sqdb	%f0, 0(%r15)
	sqdb	%f0, 4095(%r1,%r15)
	sqdb	%f0, 4095(%r15,%r1)
	sqdb	%f15, 0

#CHECK: sqdbr	%f0, %f0                # encoding: [0xb3,0x15,0x00,0x00]
#CHECK: sqdbr	%f0, %f15               # encoding: [0xb3,0x15,0x00,0x0f]
#CHECK: sqdbr	%f7, %f8                # encoding: [0xb3,0x15,0x00,0x78]
#CHECK: sqdbr	%f15, %f0               # encoding: [0xb3,0x15,0x00,0xf0]

	sqdbr	%f0, %f0
	sqdbr	%f0, %f15
	sqdbr	%f7, %f8
	sqdbr	%f15, %f0

#CHECK: sqdr	%f0, %f0                # encoding: [0xb2,0x44,0x00,0x00]
#CHECK: sqdr	%f0, %f15               # encoding: [0xb2,0x44,0x00,0x0f]
#CHECK: sqdr	%f7, %f8                # encoding: [0xb2,0x44,0x00,0x78]
#CHECK: sqdr	%f15, %f0               # encoding: [0xb2,0x44,0x00,0xf0]

	sqdr	%f0, %f0
	sqdr	%f0, %f15
	sqdr	%f7, %f8
	sqdr	%f15, %f0

#CHECK: sqe	%f0, 0                  # encoding: [0xed,0x00,0x00,0x00,0x00,0x34]
#CHECK: sqe	%f0, 4095               # encoding: [0xed,0x00,0x0f,0xff,0x00,0x34]
#CHECK: sqe	%f0, 0(%r1)             # encoding: [0xed,0x00,0x10,0x00,0x00,0x34]
#CHECK: sqe	%f0, 0(%r15)            # encoding: [0xed,0x00,0xf0,0x00,0x00,0x34]
#CHECK: sqe	%f0, 4095(%r1,%r15)     # encoding: [0xed,0x01,0xff,0xff,0x00,0x34]
#CHECK: sqe	%f0, 4095(%r15,%r1)     # encoding: [0xed,0x0f,0x1f,0xff,0x00,0x34]
#CHECK: sqe	%f15, 0                 # encoding: [0xed,0xf0,0x00,0x00,0x00,0x34]

	sqe	%f0, 0
	sqe	%f0, 4095
	sqe	%f0, 0(%r1)
	sqe	%f0, 0(%r15)
	sqe	%f0, 4095(%r1,%r15)
	sqe	%f0, 4095(%r15,%r1)
	sqe	%f15, 0

#CHECK: sqeb	%f0, 0                  # encoding: [0xed,0x00,0x00,0x00,0x00,0x14]
#CHECK: sqeb	%f0, 4095               # encoding: [0xed,0x00,0x0f,0xff,0x00,0x14]
#CHECK: sqeb	%f0, 0(%r1)             # encoding: [0xed,0x00,0x10,0x00,0x00,0x14]
#CHECK: sqeb	%f0, 0(%r15)            # encoding: [0xed,0x00,0xf0,0x00,0x00,0x14]
#CHECK: sqeb	%f0, 4095(%r1,%r15)     # encoding: [0xed,0x01,0xff,0xff,0x00,0x14]
#CHECK: sqeb	%f0, 4095(%r15,%r1)     # encoding: [0xed,0x0f,0x1f,0xff,0x00,0x14]
#CHECK: sqeb	%f15, 0                 # encoding: [0xed,0xf0,0x00,0x00,0x00,0x14]

	sqeb	%f0, 0
	sqeb	%f0, 4095
	sqeb	%f0, 0(%r1)
	sqeb	%f0, 0(%r15)
	sqeb	%f0, 4095(%r1,%r15)
	sqeb	%f0, 4095(%r15,%r1)
	sqeb	%f15, 0

#CHECK: sqebr	%f0, %f0                # encoding: [0xb3,0x14,0x00,0x00]
#CHECK: sqebr	%f0, %f15               # encoding: [0xb3,0x14,0x00,0x0f]
#CHECK: sqebr	%f7, %f8                # encoding: [0xb3,0x14,0x00,0x78]
#CHECK: sqebr	%f15, %f0               # encoding: [0xb3,0x14,0x00,0xf0]

	sqebr	%f0, %f0
	sqebr	%f0, %f15
	sqebr	%f7, %f8
	sqebr	%f15, %f0

#CHECK: sqer	%f0, %f0                # encoding: [0xb2,0x45,0x00,0x00]
#CHECK: sqer	%f0, %f15               # encoding: [0xb2,0x45,0x00,0x0f]
#CHECK: sqer	%f7, %f8                # encoding: [0xb2,0x45,0x00,0x78]
#CHECK: sqer	%f15, %f0               # encoding: [0xb2,0x45,0x00,0xf0]

	sqer	%f0, %f0
	sqer	%f0, %f15
	sqer	%f7, %f8
	sqer	%f15, %f0

#CHECK: sqxbr	%f0, %f0                # encoding: [0xb3,0x16,0x00,0x00]
#CHECK: sqxbr	%f0, %f13               # encoding: [0xb3,0x16,0x00,0x0d]
#CHECK: sqxbr	%f8, %f8                # encoding: [0xb3,0x16,0x00,0x88]
#CHECK: sqxbr	%f13, %f0               # encoding: [0xb3,0x16,0x00,0xd0]

	sqxbr	%f0, %f0
	sqxbr	%f0, %f13
	sqxbr	%f8, %f8
	sqxbr	%f13, %f0

#CHECK: sqxr	%f0, %f0                # encoding: [0xb3,0x36,0x00,0x00]
#CHECK: sqxr	%f0, %f13               # encoding: [0xb3,0x36,0x00,0x0d]
#CHECK: sqxr	%f8, %f8                # encoding: [0xb3,0x36,0x00,0x88]
#CHECK: sqxr	%f13, %f0               # encoding: [0xb3,0x36,0x00,0xd0]

	sqxr	%f0, %f0
	sqxr	%f0, %f13
	sqxr	%f8, %f8
	sqxr	%f13, %f0

#CHECK: sr	%r0, %r0                # encoding: [0x1b,0x00]
#CHECK: sr	%r0, %r15               # encoding: [0x1b,0x0f]
#CHECK: sr	%r15, %r0               # encoding: [0x1b,0xf0]
#CHECK: sr	%r7, %r8                # encoding: [0x1b,0x78]

	sr	%r0,%r0
	sr	%r0,%r15
	sr	%r15,%r0
	sr	%r7,%r8

#CHECK: sra	%r0, 0                  # encoding: [0x8a,0x00,0x00,0x00]
#CHECK: sra	%r7, 0                  # encoding: [0x8a,0x70,0x00,0x00]
#CHECK: sra	%r15, 0                 # encoding: [0x8a,0xf0,0x00,0x00]
#CHECK: sra	%r0, 4095               # encoding: [0x8a,0x00,0x0f,0xff]
#CHECK: sra	%r0, 0(%r1)             # encoding: [0x8a,0x00,0x10,0x00]
#CHECK: sra	%r0, 0(%r15)            # encoding: [0x8a,0x00,0xf0,0x00]
#CHECK: sra	%r0, 4095(%r1)          # encoding: [0x8a,0x00,0x1f,0xff]
#CHECK: sra	%r0, 4095(%r15)         # encoding: [0x8a,0x00,0xff,0xff]

	sra	%r0,0
	sra	%r7,0
	sra	%r15,0
	sra	%r0,4095
	sra	%r0,0(%r1)
	sra	%r0,0(%r15)
	sra	%r0,4095(%r1)
	sra	%r0,4095(%r15)

#CHECK: srag	%r0, %r0, 0             # encoding: [0xeb,0x00,0x00,0x00,0x00,0x0a]
#CHECK: srag	%r15, %r1, 0            # encoding: [0xeb,0xf1,0x00,0x00,0x00,0x0a]
#CHECK: srag	%r1, %r15, 0            # encoding: [0xeb,0x1f,0x00,0x00,0x00,0x0a]
#CHECK: srag	%r15, %r15, 0           # encoding: [0xeb,0xff,0x00,0x00,0x00,0x0a]
#CHECK: srag	%r0, %r0, -524288       # encoding: [0xeb,0x00,0x00,0x00,0x80,0x0a]
#CHECK: srag	%r0, %r0, -1            # encoding: [0xeb,0x00,0x0f,0xff,0xff,0x0a]
#CHECK: srag	%r0, %r0, 1             # encoding: [0xeb,0x00,0x00,0x01,0x00,0x0a]
#CHECK: srag	%r0, %r0, 524287        # encoding: [0xeb,0x00,0x0f,0xff,0x7f,0x0a]
#CHECK: srag	%r0, %r0, 0(%r1)        # encoding: [0xeb,0x00,0x10,0x00,0x00,0x0a]
#CHECK: srag	%r0, %r0, 0(%r15)       # encoding: [0xeb,0x00,0xf0,0x00,0x00,0x0a]
#CHECK: srag	%r0, %r0, 524287(%r1)   # encoding: [0xeb,0x00,0x1f,0xff,0x7f,0x0a]
#CHECK: srag	%r0, %r0, 524287(%r15)  # encoding: [0xeb,0x00,0xff,0xff,0x7f,0x0a]

	srag	%r0,%r0,0
	srag	%r15,%r1,0
	srag	%r1,%r15,0
	srag	%r15,%r15,0
	srag	%r0,%r0,-524288
	srag	%r0,%r0,-1
	srag	%r0,%r0,1
	srag	%r0,%r0,524287
	srag	%r0,%r0,0(%r1)
	srag	%r0,%r0,0(%r15)
	srag	%r0,%r0,524287(%r1)
	srag	%r0,%r0,524287(%r15)

#CHECK: srda	%r0, 0                  # encoding: [0x8e,0x00,0x00,0x00]
#CHECK: srda	%r6, 0                  # encoding: [0x8e,0x60,0x00,0x00]
#CHECK: srda	%r14, 0                 # encoding: [0x8e,0xe0,0x00,0x00]
#CHECK: srda	%r0, 4095               # encoding: [0x8e,0x00,0x0f,0xff]
#CHECK: srda	%r0, 0(%r1)             # encoding: [0x8e,0x00,0x10,0x00]
#CHECK: srda	%r0, 0(%r15)            # encoding: [0x8e,0x00,0xf0,0x00]
#CHECK: srda	%r0, 4095(%r1)          # encoding: [0x8e,0x00,0x1f,0xff]
#CHECK: srda	%r0, 4095(%r15)         # encoding: [0x8e,0x00,0xff,0xff]

	srda	%r0,0
	srda	%r6,0
	srda	%r14,0
	srda	%r0,4095
	srda	%r0,0(%r1)
	srda	%r0,0(%r15)
	srda	%r0,4095(%r1)
	srda	%r0,4095(%r15)

#CHECK: srdl	%r0, 0                  # encoding: [0x8c,0x00,0x00,0x00]
#CHECK: srdl	%r6, 0                  # encoding: [0x8c,0x60,0x00,0x00]
#CHECK: srdl	%r14, 0                 # encoding: [0x8c,0xe0,0x00,0x00]
#CHECK: srdl	%r0, 4095               # encoding: [0x8c,0x00,0x0f,0xff]
#CHECK: srdl	%r0, 0(%r1)             # encoding: [0x8c,0x00,0x10,0x00]
#CHECK: srdl	%r0, 0(%r15)            # encoding: [0x8c,0x00,0xf0,0x00]
#CHECK: srdl	%r0, 4095(%r1)          # encoding: [0x8c,0x00,0x1f,0xff]
#CHECK: srdl	%r0, 4095(%r15)         # encoding: [0x8c,0x00,0xff,0xff]

	srdl	%r0,0
	srdl	%r6,0
	srdl	%r14,0
	srdl	%r0,4095
	srdl	%r0,0(%r1)
	srdl	%r0,0(%r15)
	srdl	%r0,4095(%r1)
	srdl	%r0,4095(%r15)

#CHECK: srdt	%f0, %f0, 0             # encoding: [0xed,0x00,0x00,0x00,0x00,0x41]
#CHECK: srdt	%f0, %f0, 4095          # encoding: [0xed,0x00,0x0f,0xff,0x00,0x41]
#CHECK: srdt	%f0, %f0, 0(%r1)        # encoding: [0xed,0x00,0x10,0x00,0x00,0x41]
#CHECK: srdt	%f0, %f0, 0(%r15)       # encoding: [0xed,0x00,0xf0,0x00,0x00,0x41]
#CHECK: srdt	%f0, %f0, 4095(%r1,%r15) # encoding: [0xed,0x01,0xff,0xff,0x00,0x41]
#CHECK: srdt	%f0, %f0, 4095(%r15,%r1) # encoding: [0xed,0x0f,0x1f,0xff,0x00,0x41]
#CHECK: srdt	%f0, %f15, 0            # encoding: [0xed,0xf0,0x00,0x00,0x00,0x41]
#CHECK: srdt	%f15, %f0, 0            # encoding: [0xed,0x00,0x00,0x00,0xf0,0x41]
#CHECK: srdt	%f15, %f15, 0           # encoding: [0xed,0xf0,0x00,0x00,0xf0,0x41]

	srdt	%f0, %f0, 0
	srdt	%f0, %f0, 4095
	srdt	%f0, %f0, 0(%r1)
	srdt	%f0, %f0, 0(%r15)
	srdt	%f0, %f0, 4095(%r1,%r15)
	srdt	%f0, %f0, 4095(%r15,%r1)
	srdt	%f0, %f15, 0
	srdt	%f15, %f0, 0
	srdt	%f15, %f15, 0

#CHECK: srl	%r0, 0                  # encoding: [0x88,0x00,0x00,0x00]
#CHECK: srl	%r7, 0                  # encoding: [0x88,0x70,0x00,0x00]
#CHECK: srl	%r15, 0                 # encoding: [0x88,0xf0,0x00,0x00]
#CHECK: srl	%r0, 4095               # encoding: [0x88,0x00,0x0f,0xff]
#CHECK: srl	%r0, 0(%r1)             # encoding: [0x88,0x00,0x10,0x00]
#CHECK: srl	%r0, 0(%r15)            # encoding: [0x88,0x00,0xf0,0x00]
#CHECK: srl	%r0, 4095(%r1)          # encoding: [0x88,0x00,0x1f,0xff]
#CHECK: srl	%r0, 4095(%r15)         # encoding: [0x88,0x00,0xff,0xff]

	srl	%r0,0
	srl	%r7,0
	srl	%r15,0
	srl	%r0,4095
	srl	%r0,0(%r1)
	srl	%r0,0(%r15)
	srl	%r0,4095(%r1)
	srl	%r0,4095(%r15)

#CHECK: srlg	%r0, %r0, 0             # encoding: [0xeb,0x00,0x00,0x00,0x00,0x0c]
#CHECK: srlg	%r15, %r1, 0            # encoding: [0xeb,0xf1,0x00,0x00,0x00,0x0c]
#CHECK: srlg	%r1, %r15, 0            # encoding: [0xeb,0x1f,0x00,0x00,0x00,0x0c]
#CHECK: srlg	%r15, %r15, 0           # encoding: [0xeb,0xff,0x00,0x00,0x00,0x0c]
#CHECK: srlg	%r0, %r0, -524288       # encoding: [0xeb,0x00,0x00,0x00,0x80,0x0c]
#CHECK: srlg	%r0, %r0, -1            # encoding: [0xeb,0x00,0x0f,0xff,0xff,0x0c]
#CHECK: srlg	%r0, %r0, 1             # encoding: [0xeb,0x00,0x00,0x01,0x00,0x0c]
#CHECK: srlg	%r0, %r0, 524287        # encoding: [0xeb,0x00,0x0f,0xff,0x7f,0x0c]
#CHECK: srlg	%r0, %r0, 0(%r1)        # encoding: [0xeb,0x00,0x10,0x00,0x00,0x0c]
#CHECK: srlg	%r0, %r0, 0(%r15)       # encoding: [0xeb,0x00,0xf0,0x00,0x00,0x0c]
#CHECK: srlg	%r0, %r0, 524287(%r1)   # encoding: [0xeb,0x00,0x1f,0xff,0x7f,0x0c]
#CHECK: srlg	%r0, %r0, 524287(%r15)  # encoding: [0xeb,0x00,0xff,0xff,0x7f,0x0c]

	srlg	%r0,%r0,0
	srlg	%r15,%r1,0
	srlg	%r1,%r15,0
	srlg	%r15,%r15,0
	srlg	%r0,%r0,-524288
	srlg	%r0,%r0,-1
	srlg	%r0,%r0,1
	srlg	%r0,%r0,524287
	srlg	%r0,%r0,0(%r1)
	srlg	%r0,%r0,0(%r15)
	srlg	%r0,%r0,524287(%r1)
	srlg	%r0,%r0,524287(%r15)

#CHECK: srnm	0                       # encoding: [0xb2,0x99,0x00,0x00]
#CHECK: srnm	0(%r1)                  # encoding: [0xb2,0x99,0x10,0x00]
#CHECK: srnm	0(%r15)                 # encoding: [0xb2,0x99,0xf0,0x00]
#CHECK: srnm	4095                    # encoding: [0xb2,0x99,0x0f,0xff]
#CHECK: srnm	4095(%r1)               # encoding: [0xb2,0x99,0x1f,0xff]
#CHECK: srnm	4095(%r15)              # encoding: [0xb2,0x99,0xff,0xff]

	srnm	0
	srnm	0(%r1)
	srnm	0(%r15)
	srnm	4095
	srnm	4095(%r1)
	srnm	4095(%r15)

#CHECK: srnmt	0                       # encoding: [0xb2,0xb9,0x00,0x00]
#CHECK: srnmt	0(%r1)                  # encoding: [0xb2,0xb9,0x10,0x00]
#CHECK: srnmt	0(%r15)                 # encoding: [0xb2,0xb9,0xf0,0x00]
#CHECK: srnmt	4095                    # encoding: [0xb2,0xb9,0x0f,0xff]
#CHECK: srnmt	4095(%r1)               # encoding: [0xb2,0xb9,0x1f,0xff]
#CHECK: srnmt	4095(%r15)              # encoding: [0xb2,0xb9,0xff,0xff]

	srnmt	0
	srnmt	0(%r1)
	srnmt	0(%r15)
	srnmt	4095
	srnmt	4095(%r1)
	srnmt	4095(%r15)

#CHECK: srp	0(1), 0, 0              # encoding: [0xf0,0x00,0x00,0x00,0x00,0x00]
#CHECK: srp	0(1), 0, 15             # encoding: [0xf0,0x0f,0x00,0x00,0x00,0x00]
#CHECK: srp	0(1), 0(%r1), 0         # encoding: [0xf0,0x00,0x00,0x00,0x10,0x00]
#CHECK: srp	0(1), 0(%r15), 0        # encoding: [0xf0,0x00,0x00,0x00,0xf0,0x00]
#CHECK: srp	0(1), 4095, 0           # encoding: [0xf0,0x00,0x00,0x00,0x0f,0xff]
#CHECK: srp	0(1), 4095(%r1), 0      # encoding: [0xf0,0x00,0x00,0x00,0x1f,0xff]
#CHECK: srp	0(1), 4095(%r15), 0     # encoding: [0xf0,0x00,0x00,0x00,0xff,0xff]
#CHECK: srp	0(1,%r1), 0, 0          # encoding: [0xf0,0x00,0x10,0x00,0x00,0x00]
#CHECK: srp	0(1,%r15), 0, 0         # encoding: [0xf0,0x00,0xf0,0x00,0x00,0x00]
#CHECK: srp	4095(1,%r1), 0, 0       # encoding: [0xf0,0x00,0x1f,0xff,0x00,0x00]
#CHECK: srp	4095(1,%r15), 0, 0      # encoding: [0xf0,0x00,0xff,0xff,0x00,0x00]
#CHECK: srp	0(16,%r1), 0, 0         # encoding: [0xf0,0xf0,0x10,0x00,0x00,0x00]
#CHECK: srp	0(16,%r15), 0, 0        # encoding: [0xf0,0xf0,0xf0,0x00,0x00,0x00]

	srp	0(1), 0, 0
	srp	0(1), 0, 15
	srp	0(1), 0(%r1), 0
	srp	0(1), 0(%r15), 0
	srp	0(1), 4095, 0
	srp	0(1), 4095(%r1), 0
	srp	0(1), 4095(%r15), 0
	srp	0(1,%r1), 0, 0
	srp	0(1,%r15), 0, 0
	srp	4095(1,%r1), 0, 0
	srp	4095(1,%r15), 0, 0
	srp	0(16,%r1), 0, 0
	srp	0(16,%r15), 0, 0

#CHECK: srst	%r0, %r0                # encoding: [0xb2,0x5e,0x00,0x00]
#CHECK: srst	%r0, %r15               # encoding: [0xb2,0x5e,0x00,0x0f]
#CHECK: srst	%r15, %r0               # encoding: [0xb2,0x5e,0x00,0xf0]
#CHECK: srst	%r7, %r8                # encoding: [0xb2,0x5e,0x00,0x78]

	srst	%r0,%r0
	srst	%r0,%r15
	srst	%r15,%r0
	srst	%r7,%r8

#CHECK: srstu	%r0, %r0                # encoding: [0xb9,0xbe,0x00,0x00]
#CHECK: srstu	%r0, %r15               # encoding: [0xb9,0xbe,0x00,0x0f]
#CHECK: srstu	%r15, %r0               # encoding: [0xb9,0xbe,0x00,0xf0]
#CHECK: srstu	%r7, %r8                # encoding: [0xb9,0xbe,0x00,0x78]

	srstu	%r0,%r0
	srstu	%r0,%r15
	srstu	%r15,%r0
	srstu	%r7,%r8

#CHECK: srxt	%f0, %f0, 0             # encoding: [0xed,0x00,0x00,0x00,0x00,0x49]
#CHECK: srxt	%f0, %f0, 4095          # encoding: [0xed,0x00,0x0f,0xff,0x00,0x49]
#CHECK: srxt	%f0, %f0, 0(%r1)        # encoding: [0xed,0x00,0x10,0x00,0x00,0x49]
#CHECK: srxt	%f0, %f0, 0(%r15)       # encoding: [0xed,0x00,0xf0,0x00,0x00,0x49]
#CHECK: srxt	%f0, %f0, 4095(%r1,%r15) # encoding: [0xed,0x01,0xff,0xff,0x00,0x49]
#CHECK: srxt	%f0, %f0, 4095(%r15,%r1) # encoding: [0xed,0x0f,0x1f,0xff,0x00,0x49]
#CHECK: srxt	%f0, %f13, 0            # encoding: [0xed,0xd0,0x00,0x00,0x00,0x49]
#CHECK: srxt	%f13, %f0, 0            # encoding: [0xed,0x00,0x00,0x00,0xd0,0x49]
#CHECK: srxt	%f13, %f13, 0           # encoding: [0xed,0xd0,0x00,0x00,0xd0,0x49]

	srxt	%f0, %f0, 0
	srxt	%f0, %f0, 4095
	srxt	%f0, %f0, 0(%r1)
	srxt	%f0, %f0, 0(%r15)
	srxt	%f0, %f0, 4095(%r1,%r15)
	srxt	%f0, %f0, 4095(%r15,%r1)
	srxt	%f0, %f13, 0
	srxt	%f13, %f0, 0
	srxt	%f13, %f13, 0

#CHECK: ssar	%r0                     # encoding: [0xb2,0x25,0x00,0x00]
#CHECK: ssar	%r1                     # encoding: [0xb2,0x25,0x00,0x10]
#CHECK: ssar	%r15                    # encoding: [0xb2,0x25,0x00,0xf0]

	ssar	%r0
	ssar	%r1
	ssar	%r15

#CHECK: ssair	%r0                     # encoding: [0xb9,0x9f,0x00,0x00]
#CHECK: ssair	%r1                     # encoding: [0xb9,0x9f,0x00,0x10]
#CHECK: ssair	%r15                    # encoding: [0xb9,0x9f,0x00,0xf0]

	ssair	%r0
	ssair	%r1
	ssair	%r15

#CHECK: ssch	0                       # encoding: [0xb2,0x33,0x00,0x00]
#CHECK: ssch	0(%r1)                  # encoding: [0xb2,0x33,0x10,0x00]
#CHECK: ssch	0(%r15)                 # encoding: [0xb2,0x33,0xf0,0x00]
#CHECK: ssch	4095                    # encoding: [0xb2,0x33,0x0f,0xff]
#CHECK: ssch	4095(%r1)               # encoding: [0xb2,0x33,0x1f,0xff]
#CHECK: ssch	4095(%r15)              # encoding: [0xb2,0x33,0xff,0xff]

	ssch	0
	ssch	0(%r1)
	ssch	0(%r15)
	ssch	4095
	ssch	4095(%r1)
	ssch	4095(%r15)

#CHECK: sske	%r0, %r0                # encoding: [0xb2,0x2b,0x00,0x00]
#CHECK: sske	%r0, %r15               # encoding: [0xb2,0x2b,0x00,0x0f]
#CHECK: sske	%r15, %r0               # encoding: [0xb2,0x2b,0x00,0xf0]
#CHECK: sske	%r0, %r0, 15            # encoding: [0xb2,0x2b,0xf0,0x00]
#CHECK: sske	%r4, %r6, 7             # encoding: [0xb2,0x2b,0x70,0x46]

	sske	%r0, %r0
	sske	%r0, %r15
	sske	%r15, %r0
	sske	%r0, %r0, 15
	sske	%r4, %r6, 7

#CHECK: ssm	0                       # encoding: [0x80,0x00,0x00,0x00]
#CHECK: ssm	0(%r1)                  # encoding: [0x80,0x00,0x10,0x00]
#CHECK: ssm	0(%r15)                 # encoding: [0x80,0x00,0xf0,0x00]
#CHECK: ssm	4095                    # encoding: [0x80,0x00,0x0f,0xff]
#CHECK: ssm	4095(%r1)               # encoding: [0x80,0x00,0x1f,0xff]
#CHECK: ssm	4095(%r15)              # encoding: [0x80,0x00,0xff,0xff]

	ssm	0
	ssm	0(%r1)
	ssm	0(%r15)
	ssm	4095
	ssm	4095(%r1)
	ssm	4095(%r15)

#CHECK: st	%r0, 0                  # encoding: [0x50,0x00,0x00,0x00]
#CHECK: st	%r0, 4095               # encoding: [0x50,0x00,0x0f,0xff]
#CHECK: st	%r0, 0(%r1)             # encoding: [0x50,0x00,0x10,0x00]
#CHECK: st	%r0, 0(%r15)            # encoding: [0x50,0x00,0xf0,0x00]
#CHECK: st	%r0, 4095(%r1,%r15)     # encoding: [0x50,0x01,0xff,0xff]
#CHECK: st	%r0, 4095(%r15,%r1)     # encoding: [0x50,0x0f,0x1f,0xff]
#CHECK: st	%r15, 0                 # encoding: [0x50,0xf0,0x00,0x00]

	st	%r0, 0
	st	%r0, 4095
	st	%r0, 0(%r1)
	st	%r0, 0(%r15)
	st	%r0, 4095(%r1,%r15)
	st	%r0, 4095(%r15,%r1)
	st	%r15, 0

#CHECK: stam	%a0, %a0, 0             # encoding: [0x9b,0x00,0x00,0x00]
#CHECK: stam	%a0, %a15, 0            # encoding: [0x9b,0x0f,0x00,0x00]
#CHECK: stam	%a14, %a15, 0           # encoding: [0x9b,0xef,0x00,0x00]
#CHECK: stam	%a15, %a15, 0           # encoding: [0x9b,0xff,0x00,0x00]
#CHECK: stam	%a0, %a0, 4095          # encoding: [0x9b,0x00,0x0f,0xff]
#CHECK: stam	%a0, %a0, 1             # encoding: [0x9b,0x00,0x00,0x01]
#CHECK: stam	%a0, %a0, 0(%r1)        # encoding: [0x9b,0x00,0x10,0x00]
#CHECK: stam	%a0, %a0, 0(%r15)       # encoding: [0x9b,0x00,0xf0,0x00]
#CHECK: stam	%a0, %a0, 4095(%r1)     # encoding: [0x9b,0x00,0x1f,0xff]
#CHECK: stam	%a0, %a0, 4095(%r15)    # encoding: [0x9b,0x00,0xff,0xff]

	stam	%a0,%a0,0
	stam	%a0,%a15,0
	stam	%a14,%a15,0
	stam	%a15,%a15,0
	stam	%a0,%a0,4095
	stam	%a0,%a0,1
	stam	%a0,%a0,0(%r1)
	stam	%a0,%a0,0(%r15)
	stam	%a0,%a0,4095(%r1)
	stam	%a0,%a0,4095(%r15)

#CHECK: stamy	%a0, %a0, 0             # encoding: [0xeb,0x00,0x00,0x00,0x00,0x9b]
#CHECK: stamy	%a0, %a15, 0            # encoding: [0xeb,0x0f,0x00,0x00,0x00,0x9b]
#CHECK: stamy	%a14, %a15, 0           # encoding: [0xeb,0xef,0x00,0x00,0x00,0x9b]
#CHECK: stamy	%a15, %a15, 0           # encoding: [0xeb,0xff,0x00,0x00,0x00,0x9b]
#CHECK: stamy	%a0, %a0, -524288       # encoding: [0xeb,0x00,0x00,0x00,0x80,0x9b]
#CHECK: stamy	%a0, %a0, -1            # encoding: [0xeb,0x00,0x0f,0xff,0xff,0x9b]
#CHECK: stamy	%a0, %a0, 0             # encoding: [0xeb,0x00,0x00,0x00,0x00,0x9b]
#CHECK: stamy	%a0, %a0, 1             # encoding: [0xeb,0x00,0x00,0x01,0x00,0x9b]
#CHECK: stamy	%a0, %a0, 524287        # encoding: [0xeb,0x00,0x0f,0xff,0x7f,0x9b]
#CHECK: stamy	%a0, %a0, 0(%r1)        # encoding: [0xeb,0x00,0x10,0x00,0x00,0x9b]
#CHECK: stamy	%a0, %a0, 0(%r15)       # encoding: [0xeb,0x00,0xf0,0x00,0x00,0x9b]
#CHECK: stamy	%a0, %a0, 524287(%r1)   # encoding: [0xeb,0x00,0x1f,0xff,0x7f,0x9b]
#CHECK: stamy	%a0, %a0, 524287(%r15)  # encoding: [0xeb,0x00,0xff,0xff,0x7f,0x9b]

	stamy	%a0,%a0,0
	stamy	%a0,%a15,0
	stamy	%a14,%a15,0
	stamy	%a15,%a15,0
	stamy	%a0,%a0,-524288
	stamy	%a0,%a0,-1
	stamy	%a0,%a0,0
	stamy	%a0,%a0,1
	stamy	%a0,%a0,524287
	stamy	%a0,%a0,0(%r1)
	stamy	%a0,%a0,0(%r15)
	stamy	%a0,%a0,524287(%r1)
	stamy	%a0,%a0,524287(%r15)

#CHECK: stap	0                  	# encoding: [0xb2,0x12,0x00,0x00]
#CHECK: stap	0(%r1)             	# encoding: [0xb2,0x12,0x10,0x00]
#CHECK: stap	0(%r15)            	# encoding: [0xb2,0x12,0xf0,0x00]
#CHECK: stap	4095                 	# encoding: [0xb2,0x12,0x0f,0xff]
#CHECK: stap	4095(%r1)             	# encoding: [0xb2,0x12,0x1f,0xff]
#CHECK: stap	4095(%r15)             	# encoding: [0xb2,0x12,0xff,0xff]

	stap	0
	stap	0(%r1)
	stap	0(%r15)
	stap	4095
	stap	4095(%r1)
	stap	4095(%r15)

#CHECK: stc	%r0, 0                  # encoding: [0x42,0x00,0x00,0x00]
#CHECK: stc	%r0, 4095               # encoding: [0x42,0x00,0x0f,0xff]
#CHECK: stc	%r0, 0(%r1)             # encoding: [0x42,0x00,0x10,0x00]
#CHECK: stc	%r0, 0(%r15)            # encoding: [0x42,0x00,0xf0,0x00]
#CHECK: stc	%r0, 4095(%r1,%r15)     # encoding: [0x42,0x01,0xff,0xff]
#CHECK: stc	%r0, 4095(%r15,%r1)     # encoding: [0x42,0x0f,0x1f,0xff]
#CHECK: stc	%r15, 0                 # encoding: [0x42,0xf0,0x00,0x00]

	stc	%r0, 0
	stc	%r0, 4095
	stc	%r0, 0(%r1)
	stc	%r0, 0(%r15)
	stc	%r0, 4095(%r1,%r15)
	stc	%r0, 4095(%r15,%r1)
	stc	%r15, 0

#CHECK: stck	0                  	# encoding: [0xb2,0x05,0x00,0x00]
#CHECK: stck	0(%r1)             	# encoding: [0xb2,0x05,0x10,0x00]
#CHECK: stck	0(%r15)            	# encoding: [0xb2,0x05,0xf0,0x00]
#CHECK: stck	4095                 	# encoding: [0xb2,0x05,0x0f,0xff]
#CHECK: stck	4095(%r1)             	# encoding: [0xb2,0x05,0x1f,0xff]
#CHECK: stck	4095(%r15)             	# encoding: [0xb2,0x05,0xff,0xff]

	stck	0
	stck	0(%r1)
	stck	0(%r15)
	stck	4095
	stck	4095(%r1)
	stck	4095(%r15)

#CHECK: stckc	0                  	# encoding: [0xb2,0x07,0x00,0x00]
#CHECK: stckc	0(%r1)             	# encoding: [0xb2,0x07,0x10,0x00]
#CHECK: stckc	0(%r15)            	# encoding: [0xb2,0x07,0xf0,0x00]
#CHECK: stckc	4095                 	# encoding: [0xb2,0x07,0x0f,0xff]
#CHECK: stckc	4095(%r1)             	# encoding: [0xb2,0x07,0x1f,0xff]
#CHECK: stckc	4095(%r15)             	# encoding: [0xb2,0x07,0xff,0xff]

	stckc	0
	stckc	0(%r1)
	stckc	0(%r15)
	stckc	4095
	stckc	4095(%r1)
	stckc	4095(%r15)

#CHECK: stcke	0                  	# encoding: [0xb2,0x78,0x00,0x00]
#CHECK: stcke	0(%r1)             	# encoding: [0xb2,0x78,0x10,0x00]
#CHECK: stcke	0(%r15)            	# encoding: [0xb2,0x78,0xf0,0x00]
#CHECK: stcke	4095                 	# encoding: [0xb2,0x78,0x0f,0xff]
#CHECK: stcke	4095(%r1)             	# encoding: [0xb2,0x78,0x1f,0xff]
#CHECK: stcke	4095(%r15)             	# encoding: [0xb2,0x78,0xff,0xff]

	stcke	0
	stcke	0(%r1)
	stcke	0(%r15)
	stcke	4095
	stcke	4095(%r1)
	stcke	4095(%r15)

#CHECK: stckf	0                  	# encoding: [0xb2,0x7c,0x00,0x00]
#CHECK: stckf	0(%r1)             	# encoding: [0xb2,0x7c,0x10,0x00]
#CHECK: stckf	0(%r15)            	# encoding: [0xb2,0x7c,0xf0,0x00]
#CHECK: stckf	4095                 	# encoding: [0xb2,0x7c,0x0f,0xff]
#CHECK: stckf	4095(%r1)             	# encoding: [0xb2,0x7c,0x1f,0xff]
#CHECK: stckf	4095(%r15)             	# encoding: [0xb2,0x7c,0xff,0xff]

	stckf	0
	stckf	0(%r1)
	stckf	0(%r15)
	stckf	4095
	stckf	4095(%r1)
	stckf	4095(%r15)

#CHECK: stcm	%r0, 0, 0               # encoding: [0xbe,0x00,0x00,0x00]
#CHECK: stcm	%r0, 15, 4095           # encoding: [0xbe,0x0f,0x0f,0xff]
#CHECK: stcm	%r0, 0, 0(%r1)          # encoding: [0xbe,0x00,0x10,0x00]
#CHECK: stcm	%r0, 0, 0(%r15)         # encoding: [0xbe,0x00,0xf0,0x00]
#CHECK: stcm	%r15, 15, 4095(%r1)     # encoding: [0xbe,0xff,0x1f,0xff]
#CHECK: stcm	%r0, 0, 4095(%r15)      # encoding: [0xbe,0x00,0xff,0xff]
#CHECK: stcm	%r15, 0, 0              # encoding: [0xbe,0xf0,0x00,0x00]

	stcm	%r0, 0, 0
	stcm	%r0, 15, 4095
	stcm	%r0, 0, 0(%r1)
	stcm	%r0, 0, 0(%r15)
	stcm	%r15, 15, 4095(%r1)
	stcm	%r0, 0, 4095(%r15)
	stcm	%r15, 0, 0

#CHECK: stcmh	%r0, 0, -524288            # encoding: [0xeb,0x00,0x00,0x00,0x80,0x2c]
#CHECK: stcmh	%r0, 0, -1                 # encoding: [0xeb,0x00,0x0f,0xff,0xff,0x2c]
#CHECK: stcmh	%r0, 15, 0                 # encoding: [0xeb,0x0f,0x00,0x00,0x00,0x2c]
#CHECK: stcmh	%r0, 15, 1                 # encoding: [0xeb,0x0f,0x00,0x01,0x00,0x2c]
#CHECK: stcmh	%r0, 8, 524287             # encoding: [0xeb,0x08,0x0f,0xff,0x7f,0x2c]
#CHECK: stcmh	%r0, 8, 0(%r1)             # encoding: [0xeb,0x08,0x10,0x00,0x00,0x2c]
#CHECK: stcmh	%r0, 4, 0(%r15)            # encoding: [0xeb,0x04,0xf0,0x00,0x00,0x2c]
#CHECK: stcmh	%r0, 4, 524287(%r15)       # encoding: [0xeb,0x04,0xff,0xff,0x7f,0x2c]
#CHECK: stcmh	%r0, 0, 524287(%r1)        # encoding: [0xeb,0x00,0x1f,0xff,0x7f,0x2c]
#CHECK: stcmh	%r15, 0, 0                 # encoding: [0xeb,0xf0,0x00,0x00,0x00,0x2c]

	stcmh	%r0, 0, -524288
	stcmh	%r0, 0, -1
	stcmh	%r0, 15, 0
	stcmh	%r0, 15, 1
	stcmh	%r0, 8, 524287
	stcmh	%r0, 8, 0(%r1)
	stcmh	%r0, 4, 0(%r15)
	stcmh	%r0, 4, 524287(%r15)
	stcmh	%r0, 0, 524287(%r1)
	stcmh	%r15, 0, 0

#CHECK: stcmy	%r0, 0, -524288            # encoding: [0xeb,0x00,0x00,0x00,0x80,0x2d]
#CHECK: stcmy	%r0, 0, -1                 # encoding: [0xeb,0x00,0x0f,0xff,0xff,0x2d]
#CHECK: stcmy	%r0, 15, 0                 # encoding: [0xeb,0x0f,0x00,0x00,0x00,0x2d]
#CHECK: stcmy	%r0, 15, 1                 # encoding: [0xeb,0x0f,0x00,0x01,0x00,0x2d]
#CHECK: stcmy	%r0, 8, 524287             # encoding: [0xeb,0x08,0x0f,0xff,0x7f,0x2d]
#CHECK: stcmy	%r0, 8, 0(%r1)             # encoding: [0xeb,0x08,0x10,0x00,0x00,0x2d]
#CHECK: stcmy	%r0, 4, 0(%r15)            # encoding: [0xeb,0x04,0xf0,0x00,0x00,0x2d]
#CHECK: stcmy	%r0, 4, 524287(%r15)       # encoding: [0xeb,0x04,0xff,0xff,0x7f,0x2d]
#CHECK: stcmy	%r0, 0, 524287(%r1)        # encoding: [0xeb,0x00,0x1f,0xff,0x7f,0x2d]
#CHECK: stcmy	%r15, 0, 0                 # encoding: [0xeb,0xf0,0x00,0x00,0x00,0x2d]

	stcmy	%r0, 0, -524288
	stcmy	%r0, 0, -1
	stcmy	%r0, 15, 0
	stcmy	%r0, 15, 1
	stcmy	%r0, 8, 524287
	stcmy	%r0, 8, 0(%r1)
	stcmy	%r0, 4, 0(%r15)
	stcmy	%r0, 4, 524287(%r15)
	stcmy	%r0, 0, 524287(%r1)
	stcmy	%r15, 0, 0

#CHECK: stcps	0                       # encoding: [0xb2,0x3a,0x00,0x00]
#CHECK: stcps	0(%r1)                  # encoding: [0xb2,0x3a,0x10,0x00]
#CHECK: stcps	0(%r15)                 # encoding: [0xb2,0x3a,0xf0,0x00]
#CHECK: stcps	4095                    # encoding: [0xb2,0x3a,0x0f,0xff]
#CHECK: stcps	4095(%r1)               # encoding: [0xb2,0x3a,0x1f,0xff]
#CHECK: stcps	4095(%r15)              # encoding: [0xb2,0x3a,0xff,0xff]

	stcps	0
	stcps	0(%r1)
	stcps	0(%r15)
	stcps	4095
	stcps	4095(%r1)
	stcps	4095(%r15)

#CHECK: stcrw	0                       # encoding: [0xb2,0x39,0x00,0x00]
#CHECK: stcrw	0(%r1)                  # encoding: [0xb2,0x39,0x10,0x00]
#CHECK: stcrw	0(%r15)                 # encoding: [0xb2,0x39,0xf0,0x00]
#CHECK: stcrw	4095                    # encoding: [0xb2,0x39,0x0f,0xff]
#CHECK: stcrw	4095(%r1)               # encoding: [0xb2,0x39,0x1f,0xff]
#CHECK: stcrw	4095(%r15)              # encoding: [0xb2,0x39,0xff,0xff]

	stcrw	0
	stcrw	0(%r1)
	stcrw	0(%r15)
	stcrw	4095
	stcrw	4095(%r1)
	stcrw	4095(%r15)

#CHECK: stctg	%c0, %c0, 0             # encoding: [0xeb,0x00,0x00,0x00,0x00,0x25]
#CHECK: stctg	%c0, %c15, 0            # encoding: [0xeb,0x0f,0x00,0x00,0x00,0x25]
#CHECK: stctg	%c14, %c15, 0           # encoding: [0xeb,0xef,0x00,0x00,0x00,0x25]
#CHECK: stctg	%c15, %c15, 0           # encoding: [0xeb,0xff,0x00,0x00,0x00,0x25]
#CHECK: stctg	%c0, %c0, -524288       # encoding: [0xeb,0x00,0x00,0x00,0x80,0x25]
#CHECK: stctg	%c0, %c0, -1            # encoding: [0xeb,0x00,0x0f,0xff,0xff,0x25]
#CHECK: stctg	%c0, %c0, 0             # encoding: [0xeb,0x00,0x00,0x00,0x00,0x25]
#CHECK: stctg	%c0, %c0, 1             # encoding: [0xeb,0x00,0x00,0x01,0x00,0x25]
#CHECK: stctg	%c0, %c0, 524287        # encoding: [0xeb,0x00,0x0f,0xff,0x7f,0x25]
#CHECK: stctg	%c0, %c0, 0(%r1)        # encoding: [0xeb,0x00,0x10,0x00,0x00,0x25]
#CHECK: stctg	%c0, %c0, 0(%r15)       # encoding: [0xeb,0x00,0xf0,0x00,0x00,0x25]
#CHECK: stctg	%c0, %c0, 524287(%r1)   # encoding: [0xeb,0x00,0x1f,0xff,0x7f,0x25]
#CHECK: stctg	%c0, %c0, 524287(%r15)  # encoding: [0xeb,0x00,0xff,0xff,0x7f,0x25]

	stctg	%c0,%c0,0
	stctg	%c0,%c15,0
	stctg	%c14,%c15,0
	stctg	%c15,%c15,0
	stctg	%c0,%c0,-524288
	stctg	%c0,%c0,-1
	stctg	%c0,%c0,0
	stctg	%c0,%c0,1
	stctg	%c0,%c0,524287
	stctg	%c0,%c0,0(%r1)
	stctg	%c0,%c0,0(%r15)
	stctg	%c0,%c0,524287(%r1)
	stctg	%c0,%c0,524287(%r15)

#CHECK: stctl	%c0, %c0, 0             # encoding: [0xb6,0x00,0x00,0x00]
#CHECK: stctl	%c0, %c15, 0            # encoding: [0xb6,0x0f,0x00,0x00]
#CHECK: stctl	%c14, %c15, 0           # encoding: [0xb6,0xef,0x00,0x00]
#CHECK: stctl	%c15, %c15, 0           # encoding: [0xb6,0xff,0x00,0x00]
#CHECK: stctl	%c0, %c0, 4095          # encoding: [0xb6,0x00,0x0f,0xff]
#CHECK: stctl	%c0, %c0, 1             # encoding: [0xb6,0x00,0x00,0x01]
#CHECK: stctl	%c0, %c0, 0(%r1)        # encoding: [0xb6,0x00,0x10,0x00]
#CHECK: stctl	%c0, %c0, 0(%r15)       # encoding: [0xb6,0x00,0xf0,0x00]
#CHECK: stctl	%c0, %c0, 4095(%r1)     # encoding: [0xb6,0x00,0x1f,0xff]
#CHECK: stctl	%c0, %c0, 4095(%r15)    # encoding: [0xb6,0x00,0xff,0xff]

	stctl	%c0,%c0,0
	stctl	%c0,%c15,0
	stctl	%c14,%c15,0
	stctl	%c15,%c15,0
	stctl	%c0,%c0,4095
	stctl	%c0,%c0,1
	stctl	%c0,%c0,0(%r1)
	stctl	%c0,%c0,0(%r15)
	stctl	%c0,%c0,4095(%r1)
	stctl	%c0,%c0,4095(%r15)

#CHECK: stcy	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x72]
#CHECK: stcy	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x72]
#CHECK: stcy	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x72]
#CHECK: stcy	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x72]
#CHECK: stcy	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x72]
#CHECK: stcy	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x72]
#CHECK: stcy	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x72]
#CHECK: stcy	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x72]
#CHECK: stcy	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x72]
#CHECK: stcy	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x72]

	stcy	%r0, -524288
	stcy	%r0, -1
	stcy	%r0, 0
	stcy	%r0, 1
	stcy	%r0, 524287
	stcy	%r0, 0(%r1)
	stcy	%r0, 0(%r15)
	stcy	%r0, 524287(%r1,%r15)
	stcy	%r0, 524287(%r15,%r1)
	stcy	%r15, 0

#CHECK: std	%f0, 0                  # encoding: [0x60,0x00,0x00,0x00]
#CHECK: std	%f0, 4095               # encoding: [0x60,0x00,0x0f,0xff]
#CHECK: std	%f0, 0(%r1)             # encoding: [0x60,0x00,0x10,0x00]
#CHECK: std	%f0, 0(%r15)            # encoding: [0x60,0x00,0xf0,0x00]
#CHECK: std	%f0, 4095(%r1,%r15)     # encoding: [0x60,0x01,0xff,0xff]
#CHECK: std	%f0, 4095(%r15,%r1)     # encoding: [0x60,0x0f,0x1f,0xff]
#CHECK: std	%f15, 0                 # encoding: [0x60,0xf0,0x00,0x00]

	std	%f0, 0
	std	%f0, 4095
	std	%f0, 0(%r1)
	std	%f0, 0(%r15)
	std	%f0, 4095(%r1,%r15)
	std	%f0, 4095(%r15,%r1)
	std	%f15, 0

#CHECK: stdy	%f0, -524288            # encoding: [0xed,0x00,0x00,0x00,0x80,0x67]
#CHECK: stdy	%f0, -1                 # encoding: [0xed,0x00,0x0f,0xff,0xff,0x67]
#CHECK: stdy	%f0, 0                  # encoding: [0xed,0x00,0x00,0x00,0x00,0x67]
#CHECK: stdy	%f0, 1                  # encoding: [0xed,0x00,0x00,0x01,0x00,0x67]
#CHECK: stdy	%f0, 524287             # encoding: [0xed,0x00,0x0f,0xff,0x7f,0x67]
#CHECK: stdy	%f0, 0(%r1)             # encoding: [0xed,0x00,0x10,0x00,0x00,0x67]
#CHECK: stdy	%f0, 0(%r15)            # encoding: [0xed,0x00,0xf0,0x00,0x00,0x67]
#CHECK: stdy	%f0, 524287(%r1,%r15)   # encoding: [0xed,0x01,0xff,0xff,0x7f,0x67]
#CHECK: stdy	%f0, 524287(%r15,%r1)   # encoding: [0xed,0x0f,0x1f,0xff,0x7f,0x67]
#CHECK: stdy	%f15, 0                 # encoding: [0xed,0xf0,0x00,0x00,0x00,0x67]

	stdy	%f0, -524288
	stdy	%f0, -1
	stdy	%f0, 0
	stdy	%f0, 1
	stdy	%f0, 524287
	stdy	%f0, 0(%r1)
	stdy	%f0, 0(%r15)
	stdy	%f0, 524287(%r1,%r15)
	stdy	%f0, 524287(%r15,%r1)
	stdy	%f15, 0

#CHECK: ste	%f0, 0                  # encoding: [0x70,0x00,0x00,0x00]
#CHECK: ste	%f0, 4095               # encoding: [0x70,0x00,0x0f,0xff]
#CHECK: ste	%f0, 0(%r1)             # encoding: [0x70,0x00,0x10,0x00]
#CHECK: ste	%f0, 0(%r15)            # encoding: [0x70,0x00,0xf0,0x00]
#CHECK: ste	%f0, 4095(%r1,%r15)     # encoding: [0x70,0x01,0xff,0xff]
#CHECK: ste	%f0, 4095(%r15,%r1)     # encoding: [0x70,0x0f,0x1f,0xff]
#CHECK: ste	%f15, 0                 # encoding: [0x70,0xf0,0x00,0x00]

	ste	%f0, 0
	ste	%f0, 4095
	ste	%f0, 0(%r1)
	ste	%f0, 0(%r15)
	ste	%f0, 4095(%r1,%r15)
	ste	%f0, 4095(%r15,%r1)
	ste	%f15, 0

#CHECK: stey	%f0, -524288            # encoding: [0xed,0x00,0x00,0x00,0x80,0x66]
#CHECK: stey	%f0, -1                 # encoding: [0xed,0x00,0x0f,0xff,0xff,0x66]
#CHECK: stey	%f0, 0                  # encoding: [0xed,0x00,0x00,0x00,0x00,0x66]
#CHECK: stey	%f0, 1                  # encoding: [0xed,0x00,0x00,0x01,0x00,0x66]
#CHECK: stey	%f0, 524287             # encoding: [0xed,0x00,0x0f,0xff,0x7f,0x66]
#CHECK: stey	%f0, 0(%r1)             # encoding: [0xed,0x00,0x10,0x00,0x00,0x66]
#CHECK: stey	%f0, 0(%r15)            # encoding: [0xed,0x00,0xf0,0x00,0x00,0x66]
#CHECK: stey	%f0, 524287(%r1,%r15)   # encoding: [0xed,0x01,0xff,0xff,0x7f,0x66]
#CHECK: stey	%f0, 524287(%r15,%r1)   # encoding: [0xed,0x0f,0x1f,0xff,0x7f,0x66]
#CHECK: stey	%f15, 0                 # encoding: [0xed,0xf0,0x00,0x00,0x00,0x66]

	stey	%f0, -524288
	stey	%f0, -1
	stey	%f0, 0
	stey	%f0, 1
	stey	%f0, 524287
	stey	%f0, 0(%r1)
	stey	%f0, 0(%r15)
	stey	%f0, 524287(%r1,%r15)
	stey	%f0, 524287(%r15,%r1)
	stey	%f15, 0

#CHECK: stfl	0                  	# encoding: [0xb2,0xb1,0x00,0x00]
#CHECK: stfl	0(%r1)             	# encoding: [0xb2,0xb1,0x10,0x00]
#CHECK: stfl	0(%r15)            	# encoding: [0xb2,0xb1,0xf0,0x00]
#CHECK: stfl	4095                 	# encoding: [0xb2,0xb1,0x0f,0xff]
#CHECK: stfl	4095(%r1)             	# encoding: [0xb2,0xb1,0x1f,0xff]
#CHECK: stfl	4095(%r15)             	# encoding: [0xb2,0xb1,0xff,0xff]

	stfl	0
	stfl	0(%r1)
	stfl	0(%r15)
	stfl	4095
	stfl	4095(%r1)
	stfl	4095(%r15)

#CHECK: stfle	0                  	# encoding: [0xb2,0xb0,0x00,0x00]
#CHECK: stfle	0(%r1)             	# encoding: [0xb2,0xb0,0x10,0x00]
#CHECK: stfle	0(%r15)            	# encoding: [0xb2,0xb0,0xf0,0x00]
#CHECK: stfle	4095                 	# encoding: [0xb2,0xb0,0x0f,0xff]
#CHECK: stfle	4095(%r1)             	# encoding: [0xb2,0xb0,0x1f,0xff]
#CHECK: stfle	4095(%r15)             	# encoding: [0xb2,0xb0,0xff,0xff]

	stfle	0
	stfle	0(%r1)
	stfle	0(%r15)
	stfle	4095
	stfle	4095(%r1)
	stfle	4095(%r15)

#CHECK: stfpc	0                       # encoding: [0xb2,0x9c,0x00,0x00]
#CHECK: stfpc	0(%r1)                  # encoding: [0xb2,0x9c,0x10,0x00]
#CHECK: stfpc	0(%r15)                 # encoding: [0xb2,0x9c,0xf0,0x00]
#CHECK: stfpc	4095                    # encoding: [0xb2,0x9c,0x0f,0xff]
#CHECK: stfpc	4095(%r1)               # encoding: [0xb2,0x9c,0x1f,0xff]
#CHECK: stfpc	4095(%r15)              # encoding: [0xb2,0x9c,0xff,0xff]

	stfpc	0
	stfpc	0(%r1)
	stfpc	0(%r15)
	stfpc	4095
	stfpc	4095(%r1)
	stfpc	4095(%r15)

#CHECK: stg	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x24]
#CHECK: stg	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x24]
#CHECK: stg	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x24]
#CHECK: stg	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x24]
#CHECK: stg	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x24]
#CHECK: stg	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x24]
#CHECK: stg	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x24]
#CHECK: stg	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x24]
#CHECK: stg	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x24]
#CHECK: stg	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x24]

	stg	%r0, -524288
	stg	%r0, -1
	stg	%r0, 0
	stg	%r0, 1
	stg	%r0, 524287
	stg	%r0, 0(%r1)
	stg	%r0, 0(%r15)
	stg	%r0, 524287(%r1,%r15)
	stg	%r0, 524287(%r15,%r1)
	stg	%r15, 0

#CHECK: stgrl	%r0, .[[LAB:L.*]]-4294967296 # encoding: [0xc4,0x0b,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-4294967296)+2, kind: FK_390_PC32DBL
	stgrl	%r0, -0x100000000
#CHECK: stgrl	%r0, .[[LAB:L.*]]-2	# encoding: [0xc4,0x0b,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC32DBL
	stgrl	%r0, -2
#CHECK: stgrl	%r0, .[[LAB:L.*]]	# encoding: [0xc4,0x0b,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC32DBL
	stgrl	%r0, 0
#CHECK: stgrl	%r0, .[[LAB:L.*]]+4294967294 # encoding: [0xc4,0x0b,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+4294967294)+2, kind: FK_390_PC32DBL
	stgrl	%r0, 0xfffffffe

#CHECK: stgrl	%r0, foo                # encoding: [0xc4,0x0b,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: stgrl	%r15, foo               # encoding: [0xc4,0xfb,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL

	stgrl	%r0,foo
	stgrl	%r15,foo

#CHECK: stgrl	%r3, bar+100            # encoding: [0xc4,0x3b,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: stgrl	%r4, bar+100            # encoding: [0xc4,0x4b,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL

	stgrl	%r3,bar+100
	stgrl	%r4,bar+100

#CHECK: stgrl	%r7, frob@PLT           # encoding: [0xc4,0x7b,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL
#CHECK: stgrl	%r8, frob@PLT           # encoding: [0xc4,0x8b,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL

	stgrl	%r7,frob@PLT
	stgrl	%r8,frob@PLT

#CHECK: sth	%r0, 0                  # encoding: [0x40,0x00,0x00,0x00]
#CHECK: sth	%r0, 4095               # encoding: [0x40,0x00,0x0f,0xff]
#CHECK: sth	%r0, 0(%r1)             # encoding: [0x40,0x00,0x10,0x00]
#CHECK: sth	%r0, 0(%r15)            # encoding: [0x40,0x00,0xf0,0x00]
#CHECK: sth	%r0, 4095(%r1,%r15)     # encoding: [0x40,0x01,0xff,0xff]
#CHECK: sth	%r0, 4095(%r15,%r1)     # encoding: [0x40,0x0f,0x1f,0xff]
#CHECK: sth	%r15, 0                 # encoding: [0x40,0xf0,0x00,0x00]

	sth	%r0, 0
	sth	%r0, 4095
	sth	%r0, 0(%r1)
	sth	%r0, 0(%r15)
	sth	%r0, 4095(%r1,%r15)
	sth	%r0, 4095(%r15,%r1)
	sth	%r15, 0

#CHECK: sthrl	%r0, .[[LAB:L.*]]-4294967296 # encoding: [0xc4,0x07,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-4294967296)+2, kind: FK_390_PC32DBL
	sthrl	%r0, -0x100000000
#CHECK: sthrl	%r0, .[[LAB:L.*]]-2	# encoding: [0xc4,0x07,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC32DBL
	sthrl	%r0, -2
#CHECK: sthrl	%r0, .[[LAB:L.*]]	# encoding: [0xc4,0x07,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC32DBL
	sthrl	%r0, 0
#CHECK: sthrl	%r0, .[[LAB:L.*]]+4294967294 # encoding: [0xc4,0x07,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+4294967294)+2, kind: FK_390_PC32DBL
	sthrl	%r0, 0xfffffffe

#CHECK: sthrl	%r0, foo                # encoding: [0xc4,0x07,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: sthrl	%r15, foo               # encoding: [0xc4,0xf7,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL

	sthrl	%r0,foo
	sthrl	%r15,foo

#CHECK: sthrl	%r3, bar+100            # encoding: [0xc4,0x37,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: sthrl	%r4, bar+100            # encoding: [0xc4,0x47,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL

	sthrl	%r3,bar+100
	sthrl	%r4,bar+100

#CHECK: sthrl	%r7, frob@PLT           # encoding: [0xc4,0x77,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL
#CHECK: sthrl	%r8, frob@PLT           # encoding: [0xc4,0x87,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL

	sthrl	%r7,frob@PLT
	sthrl	%r8,frob@PLT

#CHECK: sthy	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x70]
#CHECK: sthy	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x70]
#CHECK: sthy	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x70]
#CHECK: sthy	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x70]
#CHECK: sthy	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x70]
#CHECK: sthy	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x70]
#CHECK: sthy	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x70]
#CHECK: sthy	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x70]
#CHECK: sthy	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x70]
#CHECK: sthy	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x70]

	sthy	%r0, -524288
	sthy	%r0, -1
	sthy	%r0, 0
	sthy	%r0, 1
	sthy	%r0, 524287
	sthy	%r0, 0(%r1)
	sthy	%r0, 0(%r15)
	sthy	%r0, 524287(%r1,%r15)
	sthy	%r0, 524287(%r15,%r1)
	sthy	%r15, 0

#CHECK: stidp	0                  	# encoding: [0xb2,0x02,0x00,0x00]
#CHECK: stidp	0(%r1)             	# encoding: [0xb2,0x02,0x10,0x00]
#CHECK: stidp	0(%r15)            	# encoding: [0xb2,0x02,0xf0,0x00]
#CHECK: stidp	4095                 	# encoding: [0xb2,0x02,0x0f,0xff]
#CHECK: stidp	4095(%r1)             	# encoding: [0xb2,0x02,0x1f,0xff]
#CHECK: stidp	4095(%r15)             	# encoding: [0xb2,0x02,0xff,0xff]

	stidp	0
	stidp	0(%r1)
	stidp	0(%r15)
	stidp	4095
	stidp	4095(%r1)
	stidp	4095(%r15)

#CHECK: stm	%r0, %r0, 0             # encoding: [0x90,0x00,0x00,0x00]
#CHECK: stm	%r0, %r15, 0            # encoding: [0x90,0x0f,0x00,0x00]
#CHECK: stm	%r14, %r15, 0           # encoding: [0x90,0xef,0x00,0x00]
#CHECK: stm	%r15, %r15, 0           # encoding: [0x90,0xff,0x00,0x00]
#CHECK: stm	%r0, %r0, 4095          # encoding: [0x90,0x00,0x0f,0xff]
#CHECK: stm	%r0, %r0, 1             # encoding: [0x90,0x00,0x00,0x01]
#CHECK: stm	%r0, %r0, 0(%r1)        # encoding: [0x90,0x00,0x10,0x00]
#CHECK: stm	%r0, %r0, 0(%r15)       # encoding: [0x90,0x00,0xf0,0x00]
#CHECK: stm	%r0, %r0, 4095(%r1)     # encoding: [0x90,0x00,0x1f,0xff]
#CHECK: stm	%r0, %r0, 4095(%r15)    # encoding: [0x90,0x00,0xff,0xff]

	stm	%r0,%r0,0
	stm	%r0,%r15,0
	stm	%r14,%r15,0
	stm	%r15,%r15,0
	stm	%r0,%r0,4095
	stm	%r0,%r0,1
	stm	%r0,%r0,0(%r1)
	stm	%r0,%r0,0(%r15)
	stm	%r0,%r0,4095(%r1)
	stm	%r0,%r0,4095(%r15)

#CHECK: stmg	%r0, %r0, 0             # encoding: [0xeb,0x00,0x00,0x00,0x00,0x24]
#CHECK: stmg	%r0, %r15, 0            # encoding: [0xeb,0x0f,0x00,0x00,0x00,0x24]
#CHECK: stmg	%r14, %r15, 0           # encoding: [0xeb,0xef,0x00,0x00,0x00,0x24]
#CHECK: stmg	%r15, %r15, 0           # encoding: [0xeb,0xff,0x00,0x00,0x00,0x24]
#CHECK: stmg	%r0, %r0, -524288       # encoding: [0xeb,0x00,0x00,0x00,0x80,0x24]
#CHECK: stmg	%r0, %r0, -1            # encoding: [0xeb,0x00,0x0f,0xff,0xff,0x24]
#CHECK: stmg	%r0, %r0, 0             # encoding: [0xeb,0x00,0x00,0x00,0x00,0x24]
#CHECK: stmg	%r0, %r0, 1             # encoding: [0xeb,0x00,0x00,0x01,0x00,0x24]
#CHECK: stmg	%r0, %r0, 524287        # encoding: [0xeb,0x00,0x0f,0xff,0x7f,0x24]
#CHECK: stmg	%r0, %r0, 0(%r1)        # encoding: [0xeb,0x00,0x10,0x00,0x00,0x24]
#CHECK: stmg	%r0, %r0, 0(%r15)       # encoding: [0xeb,0x00,0xf0,0x00,0x00,0x24]
#CHECK: stmg	%r0, %r0, 524287(%r1)   # encoding: [0xeb,0x00,0x1f,0xff,0x7f,0x24]
#CHECK: stmg	%r0, %r0, 524287(%r15)  # encoding: [0xeb,0x00,0xff,0xff,0x7f,0x24]

	stmg	%r0,%r0,0
	stmg	%r0,%r15,0
	stmg	%r14,%r15,0
	stmg	%r15,%r15,0
	stmg	%r0,%r0,-524288
	stmg	%r0,%r0,-1
	stmg	%r0,%r0,0
	stmg	%r0,%r0,1
	stmg	%r0,%r0,524287
	stmg	%r0,%r0,0(%r1)
	stmg	%r0,%r0,0(%r15)
	stmg	%r0,%r0,524287(%r1)
	stmg	%r0,%r0,524287(%r15)

#CHECK: stmh	%r0, %r0, 0             # encoding: [0xeb,0x00,0x00,0x00,0x00,0x26]
#CHECK: stmh	%r0, %r15, 0            # encoding: [0xeb,0x0f,0x00,0x00,0x00,0x26]
#CHECK: stmh	%r14, %r15, 0           # encoding: [0xeb,0xef,0x00,0x00,0x00,0x26]
#CHECK: stmh	%r15, %r15, 0           # encoding: [0xeb,0xff,0x00,0x00,0x00,0x26]
#CHECK: stmh	%r0, %r0, -524288       # encoding: [0xeb,0x00,0x00,0x00,0x80,0x26]
#CHECK: stmh	%r0, %r0, -1            # encoding: [0xeb,0x00,0x0f,0xff,0xff,0x26]
#CHECK: stmh	%r0, %r0, 0             # encoding: [0xeb,0x00,0x00,0x00,0x00,0x26]
#CHECK: stmh	%r0, %r0, 1             # encoding: [0xeb,0x00,0x00,0x01,0x00,0x26]
#CHECK: stmh	%r0, %r0, 524287        # encoding: [0xeb,0x00,0x0f,0xff,0x7f,0x26]
#CHECK: stmh	%r0, %r0, 0(%r1)        # encoding: [0xeb,0x00,0x10,0x00,0x00,0x26]
#CHECK: stmh	%r0, %r0, 0(%r15)       # encoding: [0xeb,0x00,0xf0,0x00,0x00,0x26]
#CHECK: stmh	%r0, %r0, 524287(%r1)   # encoding: [0xeb,0x00,0x1f,0xff,0x7f,0x26]
#CHECK: stmh	%r0, %r0, 524287(%r15)  # encoding: [0xeb,0x00,0xff,0xff,0x7f,0x26]

	stmh	%r0,%r0,0
	stmh	%r0,%r15,0
	stmh	%r14,%r15,0
	stmh	%r15,%r15,0
	stmh	%r0,%r0,-524288
	stmh	%r0,%r0,-1
	stmh	%r0,%r0,0
	stmh	%r0,%r0,1
	stmh	%r0,%r0,524287
	stmh	%r0,%r0,0(%r1)
	stmh	%r0,%r0,0(%r15)
	stmh	%r0,%r0,524287(%r1)
	stmh	%r0,%r0,524287(%r15)

#CHECK: stmy	%r0, %r0, 0             # encoding: [0xeb,0x00,0x00,0x00,0x00,0x90]
#CHECK: stmy	%r0, %r15, 0            # encoding: [0xeb,0x0f,0x00,0x00,0x00,0x90]
#CHECK: stmy	%r14, %r15, 0           # encoding: [0xeb,0xef,0x00,0x00,0x00,0x90]
#CHECK: stmy	%r15, %r15, 0           # encoding: [0xeb,0xff,0x00,0x00,0x00,0x90]
#CHECK: stmy	%r0, %r0, -524288       # encoding: [0xeb,0x00,0x00,0x00,0x80,0x90]
#CHECK: stmy	%r0, %r0, -1            # encoding: [0xeb,0x00,0x0f,0xff,0xff,0x90]
#CHECK: stmy	%r0, %r0, 0             # encoding: [0xeb,0x00,0x00,0x00,0x00,0x90]
#CHECK: stmy	%r0, %r0, 1             # encoding: [0xeb,0x00,0x00,0x01,0x00,0x90]
#CHECK: stmy	%r0, %r0, 524287        # encoding: [0xeb,0x00,0x0f,0xff,0x7f,0x90]
#CHECK: stmy	%r0, %r0, 0(%r1)        # encoding: [0xeb,0x00,0x10,0x00,0x00,0x90]
#CHECK: stmy	%r0, %r0, 0(%r15)       # encoding: [0xeb,0x00,0xf0,0x00,0x00,0x90]
#CHECK: stmy	%r0, %r0, 524287(%r1)   # encoding: [0xeb,0x00,0x1f,0xff,0x7f,0x90]
#CHECK: stmy	%r0, %r0, 524287(%r15)  # encoding: [0xeb,0x00,0xff,0xff,0x7f,0x90]

	stmy	%r0,%r0,0
	stmy	%r0,%r15,0
	stmy	%r14,%r15,0
	stmy	%r15,%r15,0
	stmy	%r0,%r0,-524288
	stmy	%r0,%r0,-1
	stmy	%r0,%r0,0
	stmy	%r0,%r0,1
	stmy	%r0,%r0,524287
	stmy	%r0,%r0,0(%r1)
	stmy	%r0,%r0,0(%r15)
	stmy	%r0,%r0,524287(%r1)
	stmy	%r0,%r0,524287(%r15)

#CHECK: stnsm	0, 0                    # encoding: [0xac,0x00,0x00,0x00]
#CHECK: stnsm	4095, 0                 # encoding: [0xac,0x00,0x0f,0xff]
#CHECK: stnsm	0, 255                  # encoding: [0xac,0xff,0x00,0x00]
#CHECK: stnsm	0(%r1), 42              # encoding: [0xac,0x2a,0x10,0x00]
#CHECK: stnsm	0(%r15), 42             # encoding: [0xac,0x2a,0xf0,0x00]
#CHECK: stnsm	4095(%r1), 42           # encoding: [0xac,0x2a,0x1f,0xff]
#CHECK: stnsm	4095(%r15), 42          # encoding: [0xac,0x2a,0xff,0xff]

	stnsm	0, 0
	stnsm	4095, 0
	stnsm	0, 255
	stnsm	0(%r1), 42
	stnsm	0(%r15), 42
	stnsm	4095(%r1), 42
	stnsm	4095(%r15), 42

#CHECK: stosm	0, 0                    # encoding: [0xad,0x00,0x00,0x00]
#CHECK: stosm	4095, 0                 # encoding: [0xad,0x00,0x0f,0xff]
#CHECK: stosm	0, 255                  # encoding: [0xad,0xff,0x00,0x00]
#CHECK: stosm	0(%r1), 42              # encoding: [0xad,0x2a,0x10,0x00]
#CHECK: stosm	0(%r15), 42             # encoding: [0xad,0x2a,0xf0,0x00]
#CHECK: stosm	4095(%r1), 42           # encoding: [0xad,0x2a,0x1f,0xff]
#CHECK: stosm	4095(%r15), 42          # encoding: [0xad,0x2a,0xff,0xff]

	stosm	0, 0
	stosm	4095, 0
	stosm	0, 255
	stosm	0(%r1), 42
	stosm	0(%r15), 42
	stosm	4095(%r1), 42
	stosm	4095(%r15), 42

#CHECK: stpt	0                  	# encoding: [0xb2,0x09,0x00,0x00]
#CHECK: stpt	0(%r1)             	# encoding: [0xb2,0x09,0x10,0x00]
#CHECK: stpt	0(%r15)            	# encoding: [0xb2,0x09,0xf0,0x00]
#CHECK: stpt	4095                 	# encoding: [0xb2,0x09,0x0f,0xff]
#CHECK: stpt	4095(%r1)             	# encoding: [0xb2,0x09,0x1f,0xff]
#CHECK: stpt	4095(%r15)             	# encoding: [0xb2,0x09,0xff,0xff]

	stpt	0
	stpt	0(%r1)
	stpt	0(%r15)
	stpt	4095
	stpt	4095(%r1)
	stpt	4095(%r15)

#CHECK: stpq	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x8e]
#CHECK: stpq	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x8e]
#CHECK: stpq	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x8e]
#CHECK: stpq	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x8e]
#CHECK: stpq	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x8e]
#CHECK: stpq	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x8e]
#CHECK: stpq	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x8e]
#CHECK: stpq	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x8e]
#CHECK: stpq	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x8e]
#CHECK: stpq	%r14, 0                 # encoding: [0xe3,0xe0,0x00,0x00,0x00,0x8e]

	stpq	%r0, -524288
	stpq	%r0, -1
	stpq	%r0, 0
	stpq	%r0, 1
	stpq	%r0, 524287
	stpq	%r0, 0(%r1)
	stpq	%r0, 0(%r15)
	stpq	%r0, 524287(%r1,%r15)
	stpq	%r0, 524287(%r15,%r1)
	stpq	%r14, 0

#CHECK: stpx	0                       # encoding: [0xb2,0x11,0x00,0x00]
#CHECK: stpx	0(%r1)                  # encoding: [0xb2,0x11,0x10,0x00]
#CHECK: stpx	0(%r15)                 # encoding: [0xb2,0x11,0xf0,0x00]
#CHECK: stpx	4095                    # encoding: [0xb2,0x11,0x0f,0xff]
#CHECK: stpx	4095(%r1)               # encoding: [0xb2,0x11,0x1f,0xff]
#CHECK: stpx	4095(%r15)              # encoding: [0xb2,0x11,0xff,0xff]

	stpx	0
	stpx	0(%r1)
	stpx	0(%r15)
	stpx	4095
	stpx	4095(%r1)
	stpx	4095(%r15)

#CHECK: strag	0, 0                    # encoding: [0xe5,0x02,0x00,0x00,0x00,0x00]
#CHECK: strag	0(%r1), 0(%r2)          # encoding: [0xe5,0x02,0x10,0x00,0x20,0x00]
#CHECK: strag	160(%r1), 320(%r15)     # encoding: [0xe5,0x02,0x10,0xa0,0xf1,0x40]
#CHECK: strag	0(%r1), 4095            # encoding: [0xe5,0x02,0x10,0x00,0x0f,0xff]
#CHECK: strag	0(%r1), 4095(%r2)       # encoding: [0xe5,0x02,0x10,0x00,0x2f,0xff]
#CHECK: strag	0(%r1), 4095(%r15)      # encoding: [0xe5,0x02,0x10,0x00,0xff,0xff]
#CHECK: strag	0(%r1), 0               # encoding: [0xe5,0x02,0x10,0x00,0x00,0x00]
#CHECK: strag	0(%r15), 0              # encoding: [0xe5,0x02,0xf0,0x00,0x00,0x00]
#CHECK: strag	4095(%r1), 0            # encoding: [0xe5,0x02,0x1f,0xff,0x00,0x00]
#CHECK: strag	4095(%r15), 0           # encoding: [0xe5,0x02,0xff,0xff,0x00,0x00]

	strag	0, 0
	strag	0(%r1), 0(%r2)
	strag	160(%r1), 320(%r15)
	strag	0(%r1), 4095
	strag	0(%r1), 4095(%r2)
	strag	0(%r1), 4095(%r15)
	strag	0(%r1), 0
	strag	0(%r15), 0
	strag	4095(%r1), 0
	strag	4095(%r15), 0

#CHECK: strl	%r0, .[[LAB:L.*]]-4294967296 # encoding: [0xc4,0x0f,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-4294967296)+2, kind: FK_390_PC32DBL
	strl	%r0, -0x100000000
#CHECK: strl	%r0, .[[LAB:L.*]]-2	# encoding: [0xc4,0x0f,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]-2)+2, kind: FK_390_PC32DBL
	strl	%r0, -2
#CHECK: strl	%r0, .[[LAB:L.*]]	# encoding: [0xc4,0x0f,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: .[[LAB]]+2, kind: FK_390_PC32DBL
	strl	%r0, 0
#CHECK: strl	%r0, .[[LAB:L.*]]+4294967294 # encoding: [0xc4,0x0f,A,A,A,A]
#CHECK:  fixup A - offset: 2, value: (.[[LAB]]+4294967294)+2, kind: FK_390_PC32DBL
	strl	%r0, 0xfffffffe

#CHECK: strl	%r0, foo                # encoding: [0xc4,0x0f,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL
#CHECK: strl	%r15, foo               # encoding: [0xc4,0xff,A,A,A,A]
# fixup A - offset: 2, value: foo+2, kind: FK_390_PC32DBL

	strl	%r0,foo
	strl	%r15,foo

#CHECK: strl	%r3, bar+100            # encoding: [0xc4,0x3f,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL
#CHECK: strl	%r4, bar+100            # encoding: [0xc4,0x4f,A,A,A,A]
# fixup A - offset: 2, value: (bar+100)+2, kind: FK_390_PC32DBL

	strl	%r3,bar+100
	strl	%r4,bar+100

#CHECK: strl	%r7, frob@PLT           # encoding: [0xc4,0x7f,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL
#CHECK: strl	%r8, frob@PLT           # encoding: [0xc4,0x8f,A,A,A,A]
# fixup A - offset: 2, value: frob@PLT+2, kind: FK_390_PC32DBL

	strl	%r7,frob@PLT
	strl	%r8,frob@PLT

#CHECK: strv	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x3e]
#CHECK: strv	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x3e]
#CHECK: strv	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x3e]
#CHECK: strv	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x3e]
#CHECK: strv	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x3e]
#CHECK: strv	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x3e]
#CHECK: strv	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x3e]
#CHECK: strv	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x3e]
#CHECK: strv	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x3e]
#CHECK: strv	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x3e]

	strv	%r0,-524288
	strv	%r0,-1
	strv	%r0,0
	strv	%r0,1
	strv	%r0,524287
	strv	%r0,0(%r1)
	strv	%r0,0(%r15)
	strv	%r0,524287(%r1,%r15)
	strv	%r0,524287(%r15,%r1)
	strv	%r15,0

#CHECK: strvg	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x2f]
#CHECK: strvg	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x2f]
#CHECK: strvg	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x2f]
#CHECK: strvg	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x2f]
#CHECK: strvg	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x2f]
#CHECK: strvg	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x2f]
#CHECK: strvg	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x2f]
#CHECK: strvg	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x2f]
#CHECK: strvg	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x2f]
#CHECK: strvg	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x2f]

	strvg	%r0,-524288
	strvg	%r0,-1
	strvg	%r0,0
	strvg	%r0,1
	strvg	%r0,524287
	strvg	%r0,0(%r1)
	strvg	%r0,0(%r15)
	strvg	%r0,524287(%r1,%r15)
	strvg	%r0,524287(%r15,%r1)
	strvg	%r15,0

#CHECK: strvh	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x3f]
#CHECK: strvh	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x3f]
#CHECK: strvh	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x3f]
#CHECK: strvh	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x3f]
#CHECK: strvh	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x3f]
#CHECK: strvh	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x3f]
#CHECK: strvh	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x3f]
#CHECK: strvh	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x3f]
#CHECK: strvh	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x3f]
#CHECK: strvh	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x3f]

	strvh	%r0,-524288
	strvh	%r0,-1
	strvh	%r0,0
	strvh	%r0,1
	strvh	%r0,524287
	strvh	%r0,0(%r1)
	strvh	%r0,0(%r15)
	strvh	%r0,524287(%r1,%r15)
	strvh	%r0,524287(%r15,%r1)
	strvh	%r15,0

#CHECK: stsch	0                       # encoding: [0xb2,0x34,0x00,0x00]
#CHECK: stsch	0(%r1)                  # encoding: [0xb2,0x34,0x10,0x00]
#CHECK: stsch	0(%r15)                 # encoding: [0xb2,0x34,0xf0,0x00]
#CHECK: stsch	4095                    # encoding: [0xb2,0x34,0x0f,0xff]
#CHECK: stsch	4095(%r1)               # encoding: [0xb2,0x34,0x1f,0xff]
#CHECK: stsch	4095(%r15)              # encoding: [0xb2,0x34,0xff,0xff]

	stsch	0
	stsch	0(%r1)
	stsch	0(%r15)
	stsch	4095
	stsch	4095(%r1)
	stsch	4095(%r15)

#CHECK: stsi	0                  	# encoding: [0xb2,0x7d,0x00,0x00]
#CHECK: stsi	0(%r1)             	# encoding: [0xb2,0x7d,0x10,0x00]
#CHECK: stsi	0(%r15)            	# encoding: [0xb2,0x7d,0xf0,0x00]
#CHECK: stsi	4095                 	# encoding: [0xb2,0x7d,0x0f,0xff]
#CHECK: stsi	4095(%r1)             	# encoding: [0xb2,0x7d,0x1f,0xff]
#CHECK: stsi	4095(%r15)             	# encoding: [0xb2,0x7d,0xff,0xff]

	stsi	0
	stsi	0(%r1)
	stsi	0(%r15)
	stsi	4095
	stsi	4095(%r1)
	stsi	4095(%r15)

#CHECK: stura	%r0, %r0                # encoding: [0xb2,0x46,0x00,0x00]
#CHECK: stura	%r0, %r15               # encoding: [0xb2,0x46,0x00,0x0f]
#CHECK: stura	%r15, %r0               # encoding: [0xb2,0x46,0x00,0xf0]
#CHECK: stura	%r7, %r8                # encoding: [0xb2,0x46,0x00,0x78]

	stura	%r0,%r0
	stura	%r0,%r15
	stura	%r15,%r0
	stura	%r7,%r8

#CHECK: sturg	%r0, %r0                # encoding: [0xb9,0x25,0x00,0x00]
#CHECK: sturg	%r0, %r15               # encoding: [0xb9,0x25,0x00,0x0f]
#CHECK: sturg	%r15, %r0               # encoding: [0xb9,0x25,0x00,0xf0]
#CHECK: sturg	%r7, %r8                # encoding: [0xb9,0x25,0x00,0x78]

	sturg	%r0,%r0
	sturg	%r0,%r15
	sturg	%r15,%r0
	sturg	%r7,%r8

#CHECK: sty	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x50]
#CHECK: sty	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x50]
#CHECK: sty	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x50]
#CHECK: sty	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x50]
#CHECK: sty	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x50]
#CHECK: sty	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x50]
#CHECK: sty	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x50]
#CHECK: sty	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x50]
#CHECK: sty	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x50]
#CHECK: sty	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x50]

	sty	%r0, -524288
	sty	%r0, -1
	sty	%r0, 0
	sty	%r0, 1
	sty	%r0, 524287
	sty	%r0, 0(%r1)
	sty	%r0, 0(%r15)
	sty	%r0, 524287(%r1,%r15)
	sty	%r0, 524287(%r15,%r1)
	sty	%r15, 0

#CHECK: su	%f0, 0                  # encoding: [0x7f,0x00,0x00,0x00]
#CHECK: su	%f0, 4095               # encoding: [0x7f,0x00,0x0f,0xff]
#CHECK: su	%f0, 0(%r1)             # encoding: [0x7f,0x00,0x10,0x00]
#CHECK: su	%f0, 0(%r15)            # encoding: [0x7f,0x00,0xf0,0x00]
#CHECK: su	%f0, 4095(%r1,%r15)     # encoding: [0x7f,0x01,0xff,0xff]
#CHECK: su	%f0, 4095(%r15,%r1)     # encoding: [0x7f,0x0f,0x1f,0xff]
#CHECK: su	%f15, 0                 # encoding: [0x7f,0xf0,0x00,0x00]

	su	%f0, 0
	su	%f0, 4095
	su	%f0, 0(%r1)
	su	%f0, 0(%r15)
	su	%f0, 4095(%r1,%r15)
	su	%f0, 4095(%r15,%r1)
	su	%f15, 0

#CHECK: sur	%f0, %f0                # encoding: [0x3f,0x00]
#CHECK: sur	%f0, %f15               # encoding: [0x3f,0x0f]
#CHECK: sur	%f7, %f8                # encoding: [0x3f,0x78]
#CHECK: sur	%f15, %f0               # encoding: [0x3f,0xf0]

	sur	%f0, %f0
	sur	%f0, %f15
	sur	%f7, %f8
	sur	%f15, %f0

#CHECK: svc	0			# encoding: [0x0a,0x00]
#CHECK: svc	3			# encoding: [0x0a,0x03]
#CHECK: svc	128			# encoding: [0x0a,0x80]
#CHECK: svc	255			# encoding: [0x0a,0xff]

	svc	0
	svc	3
	svc	128
	svc	0xff

#CHECK: sw	%f0, 0                  # encoding: [0x6f,0x00,0x00,0x00]
#CHECK: sw	%f0, 4095               # encoding: [0x6f,0x00,0x0f,0xff]
#CHECK: sw	%f0, 0(%r1)             # encoding: [0x6f,0x00,0x10,0x00]
#CHECK: sw	%f0, 0(%r15)            # encoding: [0x6f,0x00,0xf0,0x00]
#CHECK: sw	%f0, 4095(%r1,%r15)     # encoding: [0x6f,0x01,0xff,0xff]
#CHECK: sw	%f0, 4095(%r15,%r1)     # encoding: [0x6f,0x0f,0x1f,0xff]
#CHECK: sw	%f15, 0                 # encoding: [0x6f,0xf0,0x00,0x00]

	sw	%f0, 0
	sw	%f0, 4095
	sw	%f0, 0(%r1)
	sw	%f0, 0(%r15)
	sw	%f0, 4095(%r1,%r15)
	sw	%f0, 4095(%r15,%r1)
	sw	%f15, 0

#CHECK: swr	%f0, %f0                # encoding: [0x2f,0x00]
#CHECK: swr	%f0, %f15               # encoding: [0x2f,0x0f]
#CHECK: swr	%f7, %f8                # encoding: [0x2f,0x78]
#CHECK: swr	%f15, %f0               # encoding: [0x2f,0xf0]

	swr	%f0, %f0
	swr	%f0, %f15
	swr	%f7, %f8
	swr	%f15, %f0

#CHECK: sxbr	%f0, %f0                # encoding: [0xb3,0x4b,0x00,0x00]
#CHECK: sxbr	%f0, %f13               # encoding: [0xb3,0x4b,0x00,0x0d]
#CHECK: sxbr	%f8, %f8                # encoding: [0xb3,0x4b,0x00,0x88]
#CHECK: sxbr	%f13, %f0               # encoding: [0xb3,0x4b,0x00,0xd0]

	sxbr	%f0, %f0
	sxbr	%f0, %f13
	sxbr	%f8, %f8
	sxbr	%f13, %f0

#CHECK: sxr	%f0, %f0                # encoding: [0x37,0x00]
#CHECK: sxr	%f0, %f13               # encoding: [0x37,0x0d]
#CHECK: sxr	%f8, %f8                # encoding: [0x37,0x88]
#CHECK: sxr	%f13, %f0               # encoding: [0x37,0xd0]

	sxr	%f0, %f0
	sxr	%f0, %f13
	sxr	%f8, %f8
	sxr	%f13, %f0

#CHECK: sxtr	%f0, %f0, %f0           # encoding: [0xb3,0xdb,0x00,0x00]
#CHECK: sxtr	%f0, %f0, %f13          # encoding: [0xb3,0xdb,0xd0,0x00]
#CHECK: sxtr	%f0, %f13, %f0          # encoding: [0xb3,0xdb,0x00,0x0d]
#CHECK: sxtr	%f13, %f0, %f0          # encoding: [0xb3,0xdb,0x00,0xd0]
#CHECK: sxtr	%f8, %f8, %f8           # encoding: [0xb3,0xdb,0x80,0x88]

	sxtr	%f0, %f0, %f0
	sxtr	%f0, %f0, %f13
	sxtr	%f0, %f13, %f0
	sxtr	%f13, %f0, %f0
	sxtr	%f8, %f8, %f8

#CHECK: sy	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x5b]
#CHECK: sy	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x5b]
#CHECK: sy	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x5b]
#CHECK: sy	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x5b]
#CHECK: sy	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x5b]
#CHECK: sy	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x5b]
#CHECK: sy	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x5b]
#CHECK: sy	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x5b]
#CHECK: sy	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x5b]
#CHECK: sy	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x5b]

	sy	%r0, -524288
	sy	%r0, -1
	sy	%r0, 0
	sy	%r0, 1
	sy	%r0, 524287
	sy	%r0, 0(%r1)
	sy	%r0, 0(%r15)
	sy	%r0, 524287(%r1,%r15)
	sy	%r0, 524287(%r15,%r1)
	sy	%r15, 0

#CHECK: tam                             # encoding: [0x01,0x0b]

	tam

#CHECK: tar	%a0, %r0                # encoding: [0xb2,0x4c,0x00,0x00]
#CHECK: tar	%a0, %r15               # encoding: [0xb2,0x4c,0x00,0x0f]
#CHECK: tar	%a15, %r0               # encoding: [0xb2,0x4c,0x00,0xf0]
#CHECK: tar	%a7, %r8                # encoding: [0xb2,0x4c,0x00,0x78]

	tar	%a0,%r0
	tar	%a0,%r15
	tar	%a15,%r0
	tar	%a7,%r8

#CHECK: tb	%r0, %r0                # encoding: [0xb2,0x2c,0x00,0x00]
#CHECK: tb	%r0, %r15               # encoding: [0xb2,0x2c,0x00,0x0f]
#CHECK: tb	%r15, %r0               # encoding: [0xb2,0x2c,0x00,0xf0]
#CHECK: tb	%r7, %r8                # encoding: [0xb2,0x2c,0x00,0x78]
#CHECK: tb	%r15, %r15              # encoding: [0xb2,0x2c,0x00,0xff]

	tb	%r0,%r0
	tb	%r0,%r15
	tb	%r15,%r0
	tb	%r7,%r8
	tb	%r15,%r15

#CHECK: tbdr	%f0, 0, %f0             # encoding: [0xb3,0x51,0x00,0x00]
#CHECK: tbdr	%f0, 0, %f15            # encoding: [0xb3,0x51,0x00,0x0f]
#CHECK: tbdr	%f0, 15, %f0            # encoding: [0xb3,0x51,0xf0,0x00]
#CHECK: tbdr	%f4, 5, %f6             # encoding: [0xb3,0x51,0x50,0x46]
#CHECK: tbdr	%f15, 0, %f0            # encoding: [0xb3,0x51,0x00,0xf0]

	tbdr	%f0, 0, %f0
	tbdr	%f0, 0, %f15
	tbdr	%f0, 15, %f0
	tbdr	%f4, 5, %f6
	tbdr	%f15, 0, %f0

#CHECK: tbedr	%f0, 0, %f0             # encoding: [0xb3,0x50,0x00,0x00]
#CHECK: tbedr	%f0, 0, %f15            # encoding: [0xb3,0x50,0x00,0x0f]
#CHECK: tbedr	%f0, 15, %f0            # encoding: [0xb3,0x50,0xf0,0x00]
#CHECK: tbedr	%f4, 5, %f6             # encoding: [0xb3,0x50,0x50,0x46]
#CHECK: tbedr	%f15, 0, %f0            # encoding: [0xb3,0x50,0x00,0xf0]

	tbedr	%f0, 0, %f0
	tbedr	%f0, 0, %f15
	tbedr	%f0, 15, %f0
	tbedr	%f4, 5, %f6
	tbedr	%f15, 0, %f0

#CHECK: tcdb	%f0, 0                  # encoding: [0xed,0x00,0x00,0x00,0x00,0x11]
#CHECK: tcdb	%f0, 4095               # encoding: [0xed,0x00,0x0f,0xff,0x00,0x11]
#CHECK: tcdb	%f0, 0(%r1)             # encoding: [0xed,0x00,0x10,0x00,0x00,0x11]
#CHECK: tcdb	%f0, 0(%r15)            # encoding: [0xed,0x00,0xf0,0x00,0x00,0x11]
#CHECK: tcdb	%f0, 4095(%r1,%r15)     # encoding: [0xed,0x01,0xff,0xff,0x00,0x11]
#CHECK: tcdb	%f0, 4095(%r15,%r1)     # encoding: [0xed,0x0f,0x1f,0xff,0x00,0x11]
#CHECK: tcdb	%f15, 0                 # encoding: [0xed,0xf0,0x00,0x00,0x00,0x11]

	tcdb	%f0, 0
	tcdb	%f0, 4095
	tcdb	%f0, 0(%r1)
	tcdb	%f0, 0(%r15)
	tcdb	%f0, 4095(%r1,%r15)
	tcdb	%f0, 4095(%r15,%r1)
	tcdb	%f15, 0

#CHECK: tceb	%f0, 0                  # encoding: [0xed,0x00,0x00,0x00,0x00,0x10]
#CHECK: tceb	%f0, 4095               # encoding: [0xed,0x00,0x0f,0xff,0x00,0x10]
#CHECK: tceb	%f0, 0(%r1)             # encoding: [0xed,0x00,0x10,0x00,0x00,0x10]
#CHECK: tceb	%f0, 0(%r15)            # encoding: [0xed,0x00,0xf0,0x00,0x00,0x10]
#CHECK: tceb	%f0, 4095(%r1,%r15)     # encoding: [0xed,0x01,0xff,0xff,0x00,0x10]
#CHECK: tceb	%f0, 4095(%r15,%r1)     # encoding: [0xed,0x0f,0x1f,0xff,0x00,0x10]
#CHECK: tceb	%f15, 0                 # encoding: [0xed,0xf0,0x00,0x00,0x00,0x10]

	tceb	%f0, 0
	tceb	%f0, 4095
	tceb	%f0, 0(%r1)
	tceb	%f0, 0(%r15)
	tceb	%f0, 4095(%r1,%r15)
	tceb	%f0, 4095(%r15,%r1)
	tceb	%f15, 0

#CHECK: tcxb	%f0, 0                  # encoding: [0xed,0x00,0x00,0x00,0x00,0x12]
#CHECK: tcxb	%f0, 4095               # encoding: [0xed,0x00,0x0f,0xff,0x00,0x12]
#CHECK: tcxb	%f0, 0(%r1)             # encoding: [0xed,0x00,0x10,0x00,0x00,0x12]
#CHECK: tcxb	%f0, 0(%r15)            # encoding: [0xed,0x00,0xf0,0x00,0x00,0x12]
#CHECK: tcxb	%f0, 4095(%r1,%r15)     # encoding: [0xed,0x01,0xff,0xff,0x00,0x12]
#CHECK: tcxb	%f0, 4095(%r15,%r1)     # encoding: [0xed,0x0f,0x1f,0xff,0x00,0x12]
#CHECK: tcxb	%f13, 0                 # encoding: [0xed,0xd0,0x00,0x00,0x00,0x12]

	tcxb	%f0, 0
	tcxb	%f0, 4095
	tcxb	%f0, 0(%r1)
	tcxb	%f0, 0(%r15)
	tcxb	%f0, 4095(%r1,%r15)
	tcxb	%f0, 4095(%r15,%r1)
	tcxb	%f13, 0

#CHECK: tdcdt	%f0, 0                  # encoding: [0xed,0x00,0x00,0x00,0x00,0x54]
#CHECK: tdcdt	%f0, 4095               # encoding: [0xed,0x00,0x0f,0xff,0x00,0x54]
#CHECK: tdcdt	%f0, 0(%r1)             # encoding: [0xed,0x00,0x10,0x00,0x00,0x54]
#CHECK: tdcdt	%f0, 0(%r15)            # encoding: [0xed,0x00,0xf0,0x00,0x00,0x54]
#CHECK: tdcdt	%f0, 4095(%r1,%r15)     # encoding: [0xed,0x01,0xff,0xff,0x00,0x54]
#CHECK: tdcdt	%f0, 4095(%r15,%r1)     # encoding: [0xed,0x0f,0x1f,0xff,0x00,0x54]
#CHECK: tdcdt	%f15, 0                 # encoding: [0xed,0xf0,0x00,0x00,0x00,0x54]

	tdcdt	%f0, 0
	tdcdt	%f0, 4095
	tdcdt	%f0, 0(%r1)
	tdcdt	%f0, 0(%r15)
	tdcdt	%f0, 4095(%r1,%r15)
	tdcdt	%f0, 4095(%r15,%r1)
	tdcdt	%f15, 0

#CHECK: tdcet	%f0, 0                  # encoding: [0xed,0x00,0x00,0x00,0x00,0x50]
#CHECK: tdcet	%f0, 4095               # encoding: [0xed,0x00,0x0f,0xff,0x00,0x50]
#CHECK: tdcet	%f0, 0(%r1)             # encoding: [0xed,0x00,0x10,0x00,0x00,0x50]
#CHECK: tdcet	%f0, 0(%r15)            # encoding: [0xed,0x00,0xf0,0x00,0x00,0x50]
#CHECK: tdcet	%f0, 4095(%r1,%r15)     # encoding: [0xed,0x01,0xff,0xff,0x00,0x50]
#CHECK: tdcet	%f0, 4095(%r15,%r1)     # encoding: [0xed,0x0f,0x1f,0xff,0x00,0x50]
#CHECK: tdcet	%f15, 0                 # encoding: [0xed,0xf0,0x00,0x00,0x00,0x50]

	tdcet	%f0, 0
	tdcet	%f0, 4095
	tdcet	%f0, 0(%r1)
	tdcet	%f0, 0(%r15)
	tdcet	%f0, 4095(%r1,%r15)
	tdcet	%f0, 4095(%r15,%r1)
	tdcet	%f15, 0

#CHECK: tdcxt	%f0, 0                  # encoding: [0xed,0x00,0x00,0x00,0x00,0x58]
#CHECK: tdcxt	%f0, 4095               # encoding: [0xed,0x00,0x0f,0xff,0x00,0x58]
#CHECK: tdcxt	%f0, 0(%r1)             # encoding: [0xed,0x00,0x10,0x00,0x00,0x58]
#CHECK: tdcxt	%f0, 0(%r15)            # encoding: [0xed,0x00,0xf0,0x00,0x00,0x58]
#CHECK: tdcxt	%f0, 4095(%r1,%r15)     # encoding: [0xed,0x01,0xff,0xff,0x00,0x58]
#CHECK: tdcxt	%f0, 4095(%r15,%r1)     # encoding: [0xed,0x0f,0x1f,0xff,0x00,0x58]
#CHECK: tdcxt	%f13, 0                 # encoding: [0xed,0xd0,0x00,0x00,0x00,0x58]

	tdcxt	%f0, 0
	tdcxt	%f0, 4095
	tdcxt	%f0, 0(%r1)
	tdcxt	%f0, 0(%r15)
	tdcxt	%f0, 4095(%r1,%r15)
	tdcxt	%f0, 4095(%r15,%r1)
	tdcxt	%f13, 0

#CHECK: tdgdt	%f0, 0                  # encoding: [0xed,0x00,0x00,0x00,0x00,0x55]
#CHECK: tdgdt	%f0, 4095               # encoding: [0xed,0x00,0x0f,0xff,0x00,0x55]
#CHECK: tdgdt	%f0, 0(%r1)             # encoding: [0xed,0x00,0x10,0x00,0x00,0x55]
#CHECK: tdgdt	%f0, 0(%r15)            # encoding: [0xed,0x00,0xf0,0x00,0x00,0x55]
#CHECK: tdgdt	%f0, 4095(%r1,%r15)     # encoding: [0xed,0x01,0xff,0xff,0x00,0x55]
#CHECK: tdgdt	%f0, 4095(%r15,%r1)     # encoding: [0xed,0x0f,0x1f,0xff,0x00,0x55]
#CHECK: tdgdt	%f15, 0                 # encoding: [0xed,0xf0,0x00,0x00,0x00,0x55]

	tdgdt	%f0, 0
	tdgdt	%f0, 4095
	tdgdt	%f0, 0(%r1)
	tdgdt	%f0, 0(%r15)
	tdgdt	%f0, 4095(%r1,%r15)
	tdgdt	%f0, 4095(%r15,%r1)
	tdgdt	%f15, 0

#CHECK: tdget	%f0, 0                  # encoding: [0xed,0x00,0x00,0x00,0x00,0x51]
#CHECK: tdget	%f0, 4095               # encoding: [0xed,0x00,0x0f,0xff,0x00,0x51]
#CHECK: tdget	%f0, 0(%r1)             # encoding: [0xed,0x00,0x10,0x00,0x00,0x51]
#CHECK: tdget	%f0, 0(%r15)            # encoding: [0xed,0x00,0xf0,0x00,0x00,0x51]
#CHECK: tdget	%f0, 4095(%r1,%r15)     # encoding: [0xed,0x01,0xff,0xff,0x00,0x51]
#CHECK: tdget	%f0, 4095(%r15,%r1)     # encoding: [0xed,0x0f,0x1f,0xff,0x00,0x51]
#CHECK: tdget	%f15, 0                 # encoding: [0xed,0xf0,0x00,0x00,0x00,0x51]

	tdget	%f0, 0
	tdget	%f0, 4095
	tdget	%f0, 0(%r1)
	tdget	%f0, 0(%r15)
	tdget	%f0, 4095(%r1,%r15)
	tdget	%f0, 4095(%r15,%r1)
	tdget	%f15, 0

#CHECK: tdgxt	%f0, 0                  # encoding: [0xed,0x00,0x00,0x00,0x00,0x59]
#CHECK: tdgxt	%f0, 4095               # encoding: [0xed,0x00,0x0f,0xff,0x00,0x59]
#CHECK: tdgxt	%f0, 0(%r1)             # encoding: [0xed,0x00,0x10,0x00,0x00,0x59]
#CHECK: tdgxt	%f0, 0(%r15)            # encoding: [0xed,0x00,0xf0,0x00,0x00,0x59]
#CHECK: tdgxt	%f0, 4095(%r1,%r15)     # encoding: [0xed,0x01,0xff,0xff,0x00,0x59]
#CHECK: tdgxt	%f0, 4095(%r15,%r1)     # encoding: [0xed,0x0f,0x1f,0xff,0x00,0x59]
#CHECK: tdgxt	%f13, 0                 # encoding: [0xed,0xd0,0x00,0x00,0x00,0x59]

	tdgxt	%f0, 0
	tdgxt	%f0, 4095
	tdgxt	%f0, 0(%r1)
	tdgxt	%f0, 0(%r15)
	tdgxt	%f0, 4095(%r1,%r15)
	tdgxt	%f0, 4095(%r15,%r1)
	tdgxt	%f13, 0

#CHECK: thder	%f0, %f9                # encoding: [0xb3,0x58,0x00,0x09]
#CHECK: thder	%f0, %f15               # encoding: [0xb3,0x58,0x00,0x0f]
#CHECK: thder	%f15, %f0               # encoding: [0xb3,0x58,0x00,0xf0]
#CHECK: thder	%f15, %f9               # encoding: [0xb3,0x58,0x00,0xf9]

	thder	%f0,%f9
	thder	%f0,%f15
	thder	%f15,%f0
	thder	%f15,%f9

#CHECK: thdr	%f0, %f9                # encoding: [0xb3,0x59,0x00,0x09]
#CHECK: thdr	%f0, %f15               # encoding: [0xb3,0x59,0x00,0x0f]
#CHECK: thdr	%f15, %f0               # encoding: [0xb3,0x59,0x00,0xf0]
#CHECK: thdr	%f15, %f9               # encoding: [0xb3,0x59,0x00,0xf9]

	thdr	%f0,%f9
	thdr	%f0,%f15
	thdr	%f15,%f0
	thdr	%f15,%f9

#CHECK: tm	0, 0                    # encoding: [0x91,0x00,0x00,0x00]
#CHECK: tm	4095, 0                 # encoding: [0x91,0x00,0x0f,0xff]
#CHECK: tm	0, 255                  # encoding: [0x91,0xff,0x00,0x00]
#CHECK: tm	0(%r1), 42              # encoding: [0x91,0x2a,0x10,0x00]
#CHECK: tm	0(%r15), 42             # encoding: [0x91,0x2a,0xf0,0x00]
#CHECK: tm	4095(%r1), 42           # encoding: [0x91,0x2a,0x1f,0xff]
#CHECK: tm	4095(%r15), 42          # encoding: [0x91,0x2a,0xff,0xff]

	tm	0, 0
	tm	4095, 0
	tm	0, 255
	tm	0(%r1), 42
	tm	0(%r15), 42
	tm	4095(%r1), 42
	tm	4095(%r15), 42

#CHECK: tmhh	%r0, 0                  # encoding: [0xa7,0x02,0x00,0x00]
#CHECK: tmhh	%r0, 32768              # encoding: [0xa7,0x02,0x80,0x00]
#CHECK: tmhh	%r0, 65535              # encoding: [0xa7,0x02,0xff,0xff]
#CHECK: tmhh	%r15, 0                 # encoding: [0xa7,0xf2,0x00,0x00]

	tmhh	%r0, 0
	tmhh	%r0, 0x8000
	tmhh	%r0, 0xffff
	tmhh	%r15, 0

#CHECK: tmhl	%r0, 0                  # encoding: [0xa7,0x03,0x00,0x00]
#CHECK: tmhl	%r0, 32768              # encoding: [0xa7,0x03,0x80,0x00]
#CHECK: tmhl	%r0, 65535              # encoding: [0xa7,0x03,0xff,0xff]
#CHECK: tmhl	%r15, 0                 # encoding: [0xa7,0xf3,0x00,0x00]

	tmhl	%r0, 0
	tmhl	%r0, 0x8000
	tmhl	%r0, 0xffff
	tmhl	%r15, 0

#CHECK: tmlh	%r0, 0                  # encoding: [0xa7,0x00,0x00,0x00]
#CHECK: tmlh	%r0, 32768              # encoding: [0xa7,0x00,0x80,0x00]
#CHECK: tmlh	%r0, 65535              # encoding: [0xa7,0x00,0xff,0xff]
#CHECK: tmlh	%r15, 0                 # encoding: [0xa7,0xf0,0x00,0x00]

	tmh	%r0, 0
	tmh	%r0, 0x8000
	tmh	%r0, 0xffff
	tmh	%r15, 0

#CHECK: tmlh	%r0, 0                  # encoding: [0xa7,0x00,0x00,0x00]
#CHECK: tmlh	%r0, 32768              # encoding: [0xa7,0x00,0x80,0x00]
#CHECK: tmlh	%r0, 65535              # encoding: [0xa7,0x00,0xff,0xff]
#CHECK: tmlh	%r15, 0                 # encoding: [0xa7,0xf0,0x00,0x00]

	tmlh	%r0, 0
	tmlh	%r0, 0x8000
	tmlh	%r0, 0xffff
	tmlh	%r15, 0

#CHECK: tmll	%r0, 0                  # encoding: [0xa7,0x01,0x00,0x00]
#CHECK: tmll	%r0, 32768              # encoding: [0xa7,0x01,0x80,0x00]
#CHECK: tmll	%r0, 65535              # encoding: [0xa7,0x01,0xff,0xff]
#CHECK: tmll	%r15, 0                 # encoding: [0xa7,0xf1,0x00,0x00]

	tml	%r0, 0
	tml	%r0, 0x8000
	tml	%r0, 0xffff
	tml	%r15, 0

#CHECK: tmll	%r0, 0                  # encoding: [0xa7,0x01,0x00,0x00]
#CHECK: tmll	%r0, 32768              # encoding: [0xa7,0x01,0x80,0x00]
#CHECK: tmll	%r0, 65535              # encoding: [0xa7,0x01,0xff,0xff]
#CHECK: tmll	%r15, 0                 # encoding: [0xa7,0xf1,0x00,0x00]

	tmll	%r0, 0
	tmll	%r0, 0x8000
	tmll	%r0, 0xffff
	tmll	%r15, 0

#CHECK: tmy	-524288, 0              # encoding: [0xeb,0x00,0x00,0x00,0x80,0x51]
#CHECK: tmy	-1, 0                   # encoding: [0xeb,0x00,0x0f,0xff,0xff,0x51]
#CHECK: tmy	0, 0                    # encoding: [0xeb,0x00,0x00,0x00,0x00,0x51]
#CHECK: tmy	1, 0                    # encoding: [0xeb,0x00,0x00,0x01,0x00,0x51]
#CHECK: tmy	524287, 0               # encoding: [0xeb,0x00,0x0f,0xff,0x7f,0x51]
#CHECK: tmy	0, 255                  # encoding: [0xeb,0xff,0x00,0x00,0x00,0x51]
#CHECK: tmy	0(%r1), 42              # encoding: [0xeb,0x2a,0x10,0x00,0x00,0x51]
#CHECK: tmy	0(%r15), 42             # encoding: [0xeb,0x2a,0xf0,0x00,0x00,0x51]
#CHECK: tmy	524287(%r1), 42         # encoding: [0xeb,0x2a,0x1f,0xff,0x7f,0x51]
#CHECK: tmy	524287(%r15), 42        # encoding: [0xeb,0x2a,0xff,0xff,0x7f,0x51]

	tmy	-524288, 0
	tmy	-1, 0
	tmy	0, 0
	tmy	1, 0
	tmy	524287, 0
	tmy	0, 255
	tmy	0(%r1), 42
	tmy	0(%r15), 42
	tmy	524287(%r1), 42
	tmy	524287(%r15), 42

#CHECK: tp	0(1)                    # encoding: [0xeb,0x00,0x00,0x00,0x00,0xc0]
#CHECK: tp	0(1,%r1)                # encoding: [0xeb,0x00,0x10,0x00,0x00,0xc0]
#CHECK: tp	0(1,%r15)               # encoding: [0xeb,0x00,0xf0,0x00,0x00,0xc0]
#CHECK: tp	4095(1,%r1)             # encoding: [0xeb,0x00,0x1f,0xff,0x00,0xc0]
#CHECK: tp	4095(1,%r15)            # encoding: [0xeb,0x00,0xff,0xff,0x00,0xc0]
#CHECK: tp	0(16,%r1)               # encoding: [0xeb,0xf0,0x10,0x00,0x00,0xc0]
#CHECK: tp	0(16,%r15)              # encoding: [0xeb,0xf0,0xf0,0x00,0x00,0xc0]

	tp	0(1)
	tp	0(1,%r1)
	tp	0(1,%r15)
	tp	4095(1,%r1)
	tp	4095(1,%r15)
	tp	0(16,%r1)
	tp	0(16,%r15)

#CHECK: tpi	0                       # encoding: [0xb2,0x36,0x00,0x00]
#CHECK: tpi	0(%r1)                  # encoding: [0xb2,0x36,0x10,0x00]
#CHECK: tpi	0(%r15)                 # encoding: [0xb2,0x36,0xf0,0x00]
#CHECK: tpi	4095                    # encoding: [0xb2,0x36,0x0f,0xff]
#CHECK: tpi	4095(%r1)               # encoding: [0xb2,0x36,0x1f,0xff]
#CHECK: tpi	4095(%r15)              # encoding: [0xb2,0x36,0xff,0xff]

	tpi	0
	tpi	0(%r1)
	tpi	0(%r15)
	tpi	4095
	tpi	4095(%r1)
	tpi	4095(%r15)

#CHECK: tprot	0, 0                    # encoding: [0xe5,0x01,0x00,0x00,0x00,0x00]
#CHECK: tprot	0(%r1), 0(%r2)          # encoding: [0xe5,0x01,0x10,0x00,0x20,0x00]
#CHECK: tprot	160(%r1), 320(%r15)     # encoding: [0xe5,0x01,0x10,0xa0,0xf1,0x40]
#CHECK: tprot	0(%r1), 4095            # encoding: [0xe5,0x01,0x10,0x00,0x0f,0xff]
#CHECK: tprot	0(%r1), 4095(%r2)       # encoding: [0xe5,0x01,0x10,0x00,0x2f,0xff]
#CHECK: tprot	0(%r1), 4095(%r15)      # encoding: [0xe5,0x01,0x10,0x00,0xff,0xff]
#CHECK: tprot	0(%r1), 0               # encoding: [0xe5,0x01,0x10,0x00,0x00,0x00]
#CHECK: tprot	0(%r15), 0              # encoding: [0xe5,0x01,0xf0,0x00,0x00,0x00]
#CHECK: tprot	4095(%r1), 0            # encoding: [0xe5,0x01,0x1f,0xff,0x00,0x00]
#CHECK: tprot	4095(%r15), 0           # encoding: [0xe5,0x01,0xff,0xff,0x00,0x00]

	tprot	0, 0
	tprot	0(%r1), 0(%r2)
	tprot	160(%r1), 320(%r15)
	tprot	0(%r1), 4095
	tprot	0(%r1), 4095(%r2)
	tprot	0(%r1), 4095(%r15)
	tprot	0(%r1), 0
	tprot	0(%r15), 0
	tprot	4095(%r1), 0
	tprot	4095(%r15), 0

#CHECK: tr	0(1), 0                 # encoding: [0xdc,0x00,0x00,0x00,0x00,0x00]
#CHECK: tr	0(1), 0(%r1)            # encoding: [0xdc,0x00,0x00,0x00,0x10,0x00]
#CHECK: tr	0(1), 0(%r15)           # encoding: [0xdc,0x00,0x00,0x00,0xf0,0x00]
#CHECK: tr	0(1), 4095              # encoding: [0xdc,0x00,0x00,0x00,0x0f,0xff]
#CHECK: tr	0(1), 4095(%r1)         # encoding: [0xdc,0x00,0x00,0x00,0x1f,0xff]
#CHECK: tr	0(1), 4095(%r15)        # encoding: [0xdc,0x00,0x00,0x00,0xff,0xff]
#CHECK: tr	0(1,%r1), 0             # encoding: [0xdc,0x00,0x10,0x00,0x00,0x00]
#CHECK: tr	0(1,%r15), 0            # encoding: [0xdc,0x00,0xf0,0x00,0x00,0x00]
#CHECK: tr	4095(1,%r1), 0          # encoding: [0xdc,0x00,0x1f,0xff,0x00,0x00]
#CHECK: tr	4095(1,%r15), 0         # encoding: [0xdc,0x00,0xff,0xff,0x00,0x00]
#CHECK: tr	0(256,%r1), 0           # encoding: [0xdc,0xff,0x10,0x00,0x00,0x00]
#CHECK: tr	0(256,%r15), 0          # encoding: [0xdc,0xff,0xf0,0x00,0x00,0x00]

	tr	0(1), 0
	tr	0(1), 0(%r1)
	tr	0(1), 0(%r15)
	tr	0(1), 4095
	tr	0(1), 4095(%r1)
	tr	0(1), 4095(%r15)
	tr	0(1,%r1), 0
	tr	0(1,%r15), 0
	tr	4095(1,%r1), 0
	tr	4095(1,%r15), 0
	tr	0(256,%r1), 0
	tr	0(256,%r15), 0

#CHECK: trace	%r0, %r0, 0             # encoding: [0x99,0x00,0x00,0x00]
#CHECK: trace	%r0, %r15, 0            # encoding: [0x99,0x0f,0x00,0x00]
#CHECK: trace	%r14, %r15, 0           # encoding: [0x99,0xef,0x00,0x00]
#CHECK: trace	%r15, %r15, 0           # encoding: [0x99,0xff,0x00,0x00]
#CHECK: trace	%r0, %r0, 4095          # encoding: [0x99,0x00,0x0f,0xff]
#CHECK: trace	%r0, %r0, 1             # encoding: [0x99,0x00,0x00,0x01]
#CHECK: trace	%r0, %r0, 0(%r1)        # encoding: [0x99,0x00,0x10,0x00]
#CHECK: trace	%r0, %r0, 0(%r15)       # encoding: [0x99,0x00,0xf0,0x00]
#CHECK: trace	%r0, %r0, 4095(%r1)     # encoding: [0x99,0x00,0x1f,0xff]
#CHECK: trace	%r0, %r0, 4095(%r15)    # encoding: [0x99,0x00,0xff,0xff]

	trace	%r0,%r0,0
	trace	%r0,%r15,0
	trace	%r14,%r15,0
	trace	%r15,%r15,0
	trace	%r0,%r0,4095
	trace	%r0,%r0,1
	trace	%r0,%r0,0(%r1)
	trace	%r0,%r0,0(%r15)
	trace	%r0,%r0,4095(%r1)
	trace	%r0,%r0,4095(%r15)

#CHECK: tracg	%r0, %r0, 0             # encoding: [0xeb,0x00,0x00,0x00,0x00,0x0f]
#CHECK: tracg	%r0, %r15, 0            # encoding: [0xeb,0x0f,0x00,0x00,0x00,0x0f]
#CHECK: tracg	%r14, %r15, 0           # encoding: [0xeb,0xef,0x00,0x00,0x00,0x0f]
#CHECK: tracg	%r15, %r15, 0           # encoding: [0xeb,0xff,0x00,0x00,0x00,0x0f]
#CHECK: tracg	%r0, %r0, -524288       # encoding: [0xeb,0x00,0x00,0x00,0x80,0x0f]
#CHECK: tracg	%r0, %r0, -1            # encoding: [0xeb,0x00,0x0f,0xff,0xff,0x0f]
#CHECK: tracg	%r0, %r0, 0             # encoding: [0xeb,0x00,0x00,0x00,0x00,0x0f]
#CHECK: tracg	%r0, %r0, 1             # encoding: [0xeb,0x00,0x00,0x01,0x00,0x0f]
#CHECK: tracg	%r0, %r0, 524287        # encoding: [0xeb,0x00,0x0f,0xff,0x7f,0x0f]
#CHECK: tracg	%r0, %r0, 0(%r1)        # encoding: [0xeb,0x00,0x10,0x00,0x00,0x0f]
#CHECK: tracg	%r0, %r0, 0(%r15)       # encoding: [0xeb,0x00,0xf0,0x00,0x00,0x0f]
#CHECK: tracg	%r0, %r0, 524287(%r1)   # encoding: [0xeb,0x00,0x1f,0xff,0x7f,0x0f]
#CHECK: tracg	%r0, %r0, 524287(%r15)  # encoding: [0xeb,0x00,0xff,0xff,0x7f,0x0f]

	tracg	%r0,%r0,0
	tracg	%r0,%r15,0
	tracg	%r14,%r15,0
	tracg	%r15,%r15,0
	tracg	%r0,%r0,-524288
	tracg	%r0,%r0,-1
	tracg	%r0,%r0,0
	tracg	%r0,%r0,1
	tracg	%r0,%r0,524287
	tracg	%r0,%r0,0(%r1)
	tracg	%r0,%r0,0(%r15)
	tracg	%r0,%r0,524287(%r1)
	tracg	%r0,%r0,524287(%r15)

#CHECK: trap2                           # encoding: [0x01,0xff]

        trap2

#CHECK: trap4	0                       # encoding: [0xb2,0xff,0x00,0x00]
#CHECK: trap4	0(%r1)                  # encoding: [0xb2,0xff,0x10,0x00]
#CHECK: trap4	0(%r15)                 # encoding: [0xb2,0xff,0xf0,0x00]
#CHECK: trap4	4095                    # encoding: [0xb2,0xff,0x0f,0xff]
#CHECK: trap4	4095(%r1)               # encoding: [0xb2,0xff,0x1f,0xff]
#CHECK: trap4	4095(%r15)              # encoding: [0xb2,0xff,0xff,0xff]

	trap4	0
	trap4	0(%r1)
	trap4	0(%r15)
	trap4	4095
	trap4	4095(%r1)
	trap4	4095(%r15)

#CHECK: tre	%r0, %r0                # encoding: [0xb2,0xa5,0x00,0x00]
#CHECK: tre	%r0, %r15               # encoding: [0xb2,0xa5,0x00,0x0f]
#CHECK: tre	%r14, %r0               # encoding: [0xb2,0xa5,0x00,0xe0]
#CHECK: tre	%r6, %r8                # encoding: [0xb2,0xa5,0x00,0x68]

	tre	%r0, %r0
	tre	%r0, %r15
	tre	%r14, %r0
	tre	%r6, %r8

#CHECK: troo	%r0, %r0                # encoding: [0xb9,0x93,0x00,0x00]
#CHECK: troo	%r0, %r15               # encoding: [0xb9,0x93,0x00,0x0f]
#CHECK: troo	%r14, %r0               # encoding: [0xb9,0x93,0x00,0xe0]
#CHECK: troo	%r6, %r8                # encoding: [0xb9,0x93,0x00,0x68]
#CHECK: troo	%r4, %r13, 0            # encoding: [0xb9,0x93,0x00,0x4d]
#CHECK: troo	%r4, %r13, 15           # encoding: [0xb9,0x93,0xf0,0x4d]

	troo	%r0, %r0
	troo	%r0, %r15
	troo	%r14, %r0
	troo	%r6, %r8
	troo	%r4, %r13, 0
	troo	%r4, %r13, 15

#CHECK: trot	%r0, %r0                # encoding: [0xb9,0x92,0x00,0x00]
#CHECK: trot	%r0, %r15               # encoding: [0xb9,0x92,0x00,0x0f]
#CHECK: trot	%r14, %r0               # encoding: [0xb9,0x92,0x00,0xe0]
#CHECK: trot	%r6, %r8                # encoding: [0xb9,0x92,0x00,0x68]
#CHECK: trot	%r4, %r13, 0            # encoding: [0xb9,0x92,0x00,0x4d]
#CHECK: trot	%r4, %r13, 15           # encoding: [0xb9,0x92,0xf0,0x4d]

	trot	%r0, %r0
	trot	%r0, %r15
	trot	%r14, %r0
	trot	%r6, %r8
	trot	%r4, %r13, 0
	trot	%r4, %r13, 15

#CHECK: trt	0(1), 0                 # encoding: [0xdd,0x00,0x00,0x00,0x00,0x00]
#CHECK: trt	0(1), 0(%r1)            # encoding: [0xdd,0x00,0x00,0x00,0x10,0x00]
#CHECK: trt	0(1), 0(%r15)           # encoding: [0xdd,0x00,0x00,0x00,0xf0,0x00]
#CHECK: trt	0(1), 4095              # encoding: [0xdd,0x00,0x00,0x00,0x0f,0xff]
#CHECK: trt	0(1), 4095(%r1)         # encoding: [0xdd,0x00,0x00,0x00,0x1f,0xff]
#CHECK: trt	0(1), 4095(%r15)        # encoding: [0xdd,0x00,0x00,0x00,0xff,0xff]
#CHECK: trt	0(1,%r1), 0             # encoding: [0xdd,0x00,0x10,0x00,0x00,0x00]
#CHECK: trt	0(1,%r15), 0            # encoding: [0xdd,0x00,0xf0,0x00,0x00,0x00]
#CHECK: trt	4095(1,%r1), 0          # encoding: [0xdd,0x00,0x1f,0xff,0x00,0x00]
#CHECK: trt	4095(1,%r15), 0         # encoding: [0xdd,0x00,0xff,0xff,0x00,0x00]
#CHECK: trt	0(256,%r1), 0           # encoding: [0xdd,0xff,0x10,0x00,0x00,0x00]
#CHECK: trt	0(256,%r15), 0          # encoding: [0xdd,0xff,0xf0,0x00,0x00,0x00]

	trt	0(1), 0
	trt	0(1), 0(%r1)
	trt	0(1), 0(%r15)
	trt	0(1), 4095
	trt	0(1), 4095(%r1)
	trt	0(1), 4095(%r15)
	trt	0(1,%r1), 0
	trt	0(1,%r15), 0
	trt	4095(1,%r1), 0
	trt	4095(1,%r15), 0
	trt	0(256,%r1), 0
	trt	0(256,%r15), 0

#CHECK: trte	%r0, %r0                # encoding: [0xb9,0xbf,0x00,0x00]
#CHECK: trte	%r0, %r15               # encoding: [0xb9,0xbf,0x00,0x0f]
#CHECK: trte	%r14, %r0               # encoding: [0xb9,0xbf,0x00,0xe0]
#CHECK: trte	%r6, %r8                # encoding: [0xb9,0xbf,0x00,0x68]
#CHECK: trte	%r4, %r13, 0            # encoding: [0xb9,0xbf,0x00,0x4d]
#CHECK: trte	%r4, %r13, 15           # encoding: [0xb9,0xbf,0xf0,0x4d]

	trte	%r0, %r0
	trte	%r0, %r15
	trte	%r14, %r0
	trte	%r6, %r8
	trte	%r4, %r13, 0
	trte	%r4, %r13, 15

#CHECK: trto	%r0, %r0                # encoding: [0xb9,0x91,0x00,0x00]
#CHECK: trto	%r0, %r15               # encoding: [0xb9,0x91,0x00,0x0f]
#CHECK: trto	%r14, %r0               # encoding: [0xb9,0x91,0x00,0xe0]
#CHECK: trto	%r6, %r8                # encoding: [0xb9,0x91,0x00,0x68]
#CHECK: trto	%r4, %r13, 0            # encoding: [0xb9,0x91,0x00,0x4d]
#CHECK: trto	%r4, %r13, 15           # encoding: [0xb9,0x91,0xf0,0x4d]

	trto	%r0, %r0
	trto	%r0, %r15
	trto	%r14, %r0
	trto	%r6, %r8
	trto	%r4, %r13, 0
	trto	%r4, %r13, 15

#CHECK: trtr	0(1), 0                 # encoding: [0xd0,0x00,0x00,0x00,0x00,0x00]
#CHECK: trtr	0(1), 0(%r1)            # encoding: [0xd0,0x00,0x00,0x00,0x10,0x00]
#CHECK: trtr	0(1), 0(%r15)           # encoding: [0xd0,0x00,0x00,0x00,0xf0,0x00]
#CHECK: trtr	0(1), 4095              # encoding: [0xd0,0x00,0x00,0x00,0x0f,0xff]
#CHECK: trtr	0(1), 4095(%r1)         # encoding: [0xd0,0x00,0x00,0x00,0x1f,0xff]
#CHECK: trtr	0(1), 4095(%r15)        # encoding: [0xd0,0x00,0x00,0x00,0xff,0xff]
#CHECK: trtr	0(1,%r1), 0             # encoding: [0xd0,0x00,0x10,0x00,0x00,0x00]
#CHECK: trtr	0(1,%r15), 0            # encoding: [0xd0,0x00,0xf0,0x00,0x00,0x00]
#CHECK: trtr	4095(1,%r1), 0          # encoding: [0xd0,0x00,0x1f,0xff,0x00,0x00]
#CHECK: trtr	4095(1,%r15), 0         # encoding: [0xd0,0x00,0xff,0xff,0x00,0x00]
#CHECK: trtr	0(256,%r1), 0           # encoding: [0xd0,0xff,0x10,0x00,0x00,0x00]
#CHECK: trtr	0(256,%r15), 0          # encoding: [0xd0,0xff,0xf0,0x00,0x00,0x00]

	trtr	0(1), 0
	trtr	0(1), 0(%r1)
	trtr	0(1), 0(%r15)
	trtr	0(1), 4095
	trtr	0(1), 4095(%r1)
	trtr	0(1), 4095(%r15)
	trtr	0(1,%r1), 0
	trtr	0(1,%r15), 0
	trtr	4095(1,%r1), 0
	trtr	4095(1,%r15), 0
	trtr	0(256,%r1), 0
	trtr	0(256,%r15), 0

#CHECK: trtre	%r0, %r0                # encoding: [0xb9,0xbd,0x00,0x00]
#CHECK: trtre	%r0, %r15               # encoding: [0xb9,0xbd,0x00,0x0f]
#CHECK: trtre	%r14, %r0               # encoding: [0xb9,0xbd,0x00,0xe0]
#CHECK: trtre	%r6, %r8                # encoding: [0xb9,0xbd,0x00,0x68]
#CHECK: trtre	%r4, %r13, 0            # encoding: [0xb9,0xbd,0x00,0x4d]
#CHECK: trtre	%r4, %r13, 15           # encoding: [0xb9,0xbd,0xf0,0x4d]

	trtre	%r0, %r0
	trtre	%r0, %r15
	trtre	%r14, %r0
	trtre	%r6, %r8
	trtre	%r4, %r13, 0
	trtre	%r4, %r13, 15

#CHECK: trtt	%r0, %r0                # encoding: [0xb9,0x90,0x00,0x00]
#CHECK: trtt	%r0, %r15               # encoding: [0xb9,0x90,0x00,0x0f]
#CHECK: trtt	%r14, %r0               # encoding: [0xb9,0x90,0x00,0xe0]
#CHECK: trtt	%r6, %r8                # encoding: [0xb9,0x90,0x00,0x68]
#CHECK: trtt	%r4, %r13, 0            # encoding: [0xb9,0x90,0x00,0x4d]
#CHECK: trtt	%r4, %r13, 15           # encoding: [0xb9,0x90,0xf0,0x4d]

	trtt	%r0, %r0
	trtt	%r0, %r15
	trtt	%r14, %r0
	trtt	%r6, %r8
	trtt	%r4, %r13, 0
	trtt	%r4, %r13, 15

#CHECK: ts	0                  	# encoding: [0x93,0x00,0x00,0x00]
#CHECK: ts	0(%r1)             	# encoding: [0x93,0x00,0x10,0x00]
#CHECK: ts	0(%r15)            	# encoding: [0x93,0x00,0xf0,0x00]
#CHECK: ts	4095                 	# encoding: [0x93,0x00,0x0f,0xff]
#CHECK: ts	4095(%r1)             	# encoding: [0x93,0x00,0x1f,0xff]
#CHECK: ts	4095(%r15)             	# encoding: [0x93,0x00,0xff,0xff]

	ts	0
	ts	0(%r1)
	ts	0(%r15)
	ts	4095
	ts	4095(%r1)
	ts	4095(%r15)

#CHECK: tsch	0                       # encoding: [0xb2,0x35,0x00,0x00]
#CHECK: tsch	0(%r1)                  # encoding: [0xb2,0x35,0x10,0x00]
#CHECK: tsch	0(%r15)                 # encoding: [0xb2,0x35,0xf0,0x00]
#CHECK: tsch	4095                    # encoding: [0xb2,0x35,0x0f,0xff]
#CHECK: tsch	4095(%r1)               # encoding: [0xb2,0x35,0x1f,0xff]
#CHECK: tsch	4095(%r15)              # encoding: [0xb2,0x35,0xff,0xff]

	tsch	0
	tsch	0(%r1)
	tsch	0(%r15)
	tsch	4095
	tsch	4095(%r1)
	tsch	4095(%r15)

#CHECK: unpk	0(1), 0(1)              # encoding: [0xf3,0x00,0x00,0x00,0x00,0x00]
#CHECK: unpk	0(1), 0(1,%r1)          # encoding: [0xf3,0x00,0x00,0x00,0x10,0x00]
#CHECK: unpk	0(1), 0(1,%r15)         # encoding: [0xf3,0x00,0x00,0x00,0xf0,0x00]
#CHECK: unpk	0(1), 4095(1)           # encoding: [0xf3,0x00,0x00,0x00,0x0f,0xff]
#CHECK: unpk	0(1), 4095(1,%r1)       # encoding: [0xf3,0x00,0x00,0x00,0x1f,0xff]
#CHECK: unpk	0(1), 4095(1,%r15)      # encoding: [0xf3,0x00,0x00,0x00,0xff,0xff]
#CHECK: unpk	0(1,%r1), 0(1)          # encoding: [0xf3,0x00,0x10,0x00,0x00,0x00]
#CHECK: unpk	0(1,%r15), 0(1)         # encoding: [0xf3,0x00,0xf0,0x00,0x00,0x00]
#CHECK: unpk	4095(1,%r1), 0(1)       # encoding: [0xf3,0x00,0x1f,0xff,0x00,0x00]
#CHECK: unpk	4095(1,%r15), 0(1)      # encoding: [0xf3,0x00,0xff,0xff,0x00,0x00]
#CHECK: unpk	0(16,%r1), 0(1)         # encoding: [0xf3,0xf0,0x10,0x00,0x00,0x00]
#CHECK: unpk	0(16,%r15), 0(1)        # encoding: [0xf3,0xf0,0xf0,0x00,0x00,0x00]
#CHECK: unpk	0(1), 0(16,%r1)         # encoding: [0xf3,0x0f,0x00,0x00,0x10,0x00]
#CHECK: unpk	0(1), 0(16,%r15)        # encoding: [0xf3,0x0f,0x00,0x00,0xf0,0x00]

	unpk	0(1), 0(1)
	unpk	0(1), 0(1,%r1)
	unpk	0(1), 0(1,%r15)
	unpk	0(1), 4095(1)
	unpk	0(1), 4095(1,%r1)
	unpk	0(1), 4095(1,%r15)
	unpk	0(1,%r1), 0(1)
	unpk	0(1,%r15), 0(1)
	unpk	4095(1,%r1), 0(1)
	unpk	4095(1,%r15), 0(1)
	unpk	0(16,%r1), 0(1)
	unpk	0(16,%r15), 0(1)
	unpk	0(1), 0(16,%r1)
	unpk	0(1), 0(16,%r15)

#CHECK: unpka	0(1), 0                 # encoding: [0xea,0x00,0x00,0x00,0x00,0x00]
#CHECK: unpka	0(1), 0(%r1)            # encoding: [0xea,0x00,0x00,0x00,0x10,0x00]
#CHECK: unpka	0(1), 0(%r15)           # encoding: [0xea,0x00,0x00,0x00,0xf0,0x00]
#CHECK: unpka	0(1), 4095              # encoding: [0xea,0x00,0x00,0x00,0x0f,0xff]
#CHECK: unpka	0(1), 4095(%r1)         # encoding: [0xea,0x00,0x00,0x00,0x1f,0xff]
#CHECK: unpka	0(1), 4095(%r15)        # encoding: [0xea,0x00,0x00,0x00,0xff,0xff]
#CHECK: unpka	0(1,%r1), 0             # encoding: [0xea,0x00,0x10,0x00,0x00,0x00]
#CHECK: unpka	0(1,%r15), 0            # encoding: [0xea,0x00,0xf0,0x00,0x00,0x00]
#CHECK: unpka	4095(1,%r1), 0          # encoding: [0xea,0x00,0x1f,0xff,0x00,0x00]
#CHECK: unpka	4095(1,%r15), 0         # encoding: [0xea,0x00,0xff,0xff,0x00,0x00]
#CHECK: unpka	0(256,%r1), 0           # encoding: [0xea,0xff,0x10,0x00,0x00,0x00]
#CHECK: unpka	0(256,%r15), 0          # encoding: [0xea,0xff,0xf0,0x00,0x00,0x00]

	unpka	0(1), 0
	unpka	0(1), 0(%r1)
	unpka	0(1), 0(%r15)
	unpka	0(1), 4095
	unpka	0(1), 4095(%r1)
	unpka	0(1), 4095(%r15)
	unpka	0(1,%r1), 0
	unpka	0(1,%r15), 0
	unpka	4095(1,%r1), 0
	unpka	4095(1,%r15), 0
	unpka	0(256,%r1), 0
	unpka	0(256,%r15), 0

#CHECK: unpku	0(1), 0                 # encoding: [0xe2,0x00,0x00,0x00,0x00,0x00]
#CHECK: unpku	0(1), 0(%r1)            # encoding: [0xe2,0x00,0x00,0x00,0x10,0x00]
#CHECK: unpku	0(1), 0(%r15)           # encoding: [0xe2,0x00,0x00,0x00,0xf0,0x00]
#CHECK: unpku	0(1), 4095              # encoding: [0xe2,0x00,0x00,0x00,0x0f,0xff]
#CHECK: unpku	0(1), 4095(%r1)         # encoding: [0xe2,0x00,0x00,0x00,0x1f,0xff]
#CHECK: unpku	0(1), 4095(%r15)        # encoding: [0xe2,0x00,0x00,0x00,0xff,0xff]
#CHECK: unpku	0(1,%r1), 0             # encoding: [0xe2,0x00,0x10,0x00,0x00,0x00]
#CHECK: unpku	0(1,%r15), 0            # encoding: [0xe2,0x00,0xf0,0x00,0x00,0x00]
#CHECK: unpku	4095(1,%r1), 0          # encoding: [0xe2,0x00,0x1f,0xff,0x00,0x00]
#CHECK: unpku	4095(1,%r15), 0         # encoding: [0xe2,0x00,0xff,0xff,0x00,0x00]
#CHECK: unpku	0(256,%r1), 0           # encoding: [0xe2,0xff,0x10,0x00,0x00,0x00]
#CHECK: unpku	0(256,%r15), 0          # encoding: [0xe2,0xff,0xf0,0x00,0x00,0x00]

	unpku	0(1), 0
	unpku	0(1), 0(%r1)
	unpku	0(1), 0(%r15)
	unpku	0(1), 4095
	unpku	0(1), 4095(%r1)
	unpku	0(1), 4095(%r15)
	unpku	0(1,%r1), 0
	unpku	0(1,%r15), 0
	unpku	4095(1,%r1), 0
	unpku	4095(1,%r15), 0
	unpku	0(256,%r1), 0
	unpku	0(256,%r15), 0

#CHECK: upt                             # encoding: [0x01,0x02]

	upt

#CHECK: x	%r0, 0                  # encoding: [0x57,0x00,0x00,0x00]
#CHECK: x	%r0, 4095               # encoding: [0x57,0x00,0x0f,0xff]
#CHECK: x	%r0, 0(%r1)             # encoding: [0x57,0x00,0x10,0x00]
#CHECK: x	%r0, 0(%r15)            # encoding: [0x57,0x00,0xf0,0x00]
#CHECK: x	%r0, 4095(%r1,%r15)     # encoding: [0x57,0x01,0xff,0xff]
#CHECK: x	%r0, 4095(%r15,%r1)     # encoding: [0x57,0x0f,0x1f,0xff]
#CHECK: x	%r15, 0                 # encoding: [0x57,0xf0,0x00,0x00]

	x	%r0, 0
	x	%r0, 4095
	x	%r0, 0(%r1)
	x	%r0, 0(%r15)
	x	%r0, 4095(%r1,%r15)
	x	%r0, 4095(%r15,%r1)
	x	%r15, 0

#CHECK: xc	0(1), 0                 # encoding: [0xd7,0x00,0x00,0x00,0x00,0x00]
#CHECK: xc	0(1), 0(%r1)            # encoding: [0xd7,0x00,0x00,0x00,0x10,0x00]
#CHECK: xc	0(1), 0(%r15)           # encoding: [0xd7,0x00,0x00,0x00,0xf0,0x00]
#CHECK: xc	0(1), 4095              # encoding: [0xd7,0x00,0x00,0x00,0x0f,0xff]
#CHECK: xc	0(1), 4095(%r1)         # encoding: [0xd7,0x00,0x00,0x00,0x1f,0xff]
#CHECK: xc	0(1), 4095(%r15)        # encoding: [0xd7,0x00,0x00,0x00,0xff,0xff]
#CHECK: xc	0(1,%r1), 0             # encoding: [0xd7,0x00,0x10,0x00,0x00,0x00]
#CHECK: xc	0(1,%r15), 0            # encoding: [0xd7,0x00,0xf0,0x00,0x00,0x00]
#CHECK: xc	4095(1,%r1), 0          # encoding: [0xd7,0x00,0x1f,0xff,0x00,0x00]
#CHECK: xc	4095(1,%r15), 0         # encoding: [0xd7,0x00,0xff,0xff,0x00,0x00]
#CHECK: xc	0(256,%r1), 0           # encoding: [0xd7,0xff,0x10,0x00,0x00,0x00]
#CHECK: xc	0(256,%r15), 0          # encoding: [0xd7,0xff,0xf0,0x00,0x00,0x00]

	xc	0(1), 0
	xc	0(1), 0(%r1)
	xc	0(1), 0(%r15)
	xc	0(1), 4095
	xc	0(1), 4095(%r1)
	xc	0(1), 4095(%r15)
	xc	0(1,%r1), 0
	xc	0(1,%r15), 0
	xc	4095(1,%r1), 0
	xc	4095(1,%r15), 0
	xc	0(256,%r1), 0
	xc	0(256,%r15), 0

#CHECK: xg	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x82]
#CHECK: xg	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x82]
#CHECK: xg	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x82]
#CHECK: xg	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x82]
#CHECK: xg	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x82]
#CHECK: xg	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x82]
#CHECK: xg	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x82]
#CHECK: xg	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x82]
#CHECK: xg	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x82]
#CHECK: xg	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x82]

	xg	%r0, -524288
	xg	%r0, -1
	xg	%r0, 0
	xg	%r0, 1
	xg	%r0, 524287
	xg	%r0, 0(%r1)
	xg	%r0, 0(%r15)
	xg	%r0, 524287(%r1,%r15)
	xg	%r0, 524287(%r15,%r1)
	xg	%r15, 0

#CHECK: xgr	%r0, %r0                # encoding: [0xb9,0x82,0x00,0x00]
#CHECK: xgr	%r0, %r15               # encoding: [0xb9,0x82,0x00,0x0f]
#CHECK: xgr	%r15, %r0               # encoding: [0xb9,0x82,0x00,0xf0]
#CHECK: xgr	%r7, %r8                # encoding: [0xb9,0x82,0x00,0x78]

	xgr	%r0,%r0
	xgr	%r0,%r15
	xgr	%r15,%r0
	xgr	%r7,%r8

#CHECK: xi	0, 0                    # encoding: [0x97,0x00,0x00,0x00]
#CHECK: xi	4095, 0                 # encoding: [0x97,0x00,0x0f,0xff]
#CHECK: xi	0, 255                  # encoding: [0x97,0xff,0x00,0x00]
#CHECK: xi	0(%r1), 42              # encoding: [0x97,0x2a,0x10,0x00]
#CHECK: xi	0(%r15), 42             # encoding: [0x97,0x2a,0xf0,0x00]
#CHECK: xi	4095(%r1), 42           # encoding: [0x97,0x2a,0x1f,0xff]
#CHECK: xi	4095(%r15), 42          # encoding: [0x97,0x2a,0xff,0xff]

	xi	0, 0
	xi	4095, 0
	xi	0, 255
	xi	0(%r1), 42
	xi	0(%r15), 42
	xi	4095(%r1), 42
	xi	4095(%r15), 42

#CHECK: xihf	%r0, 0                  # encoding: [0xc0,0x06,0x00,0x00,0x00,0x00]
#CHECK: xihf	%r0, 4294967295         # encoding: [0xc0,0x06,0xff,0xff,0xff,0xff]
#CHECK: xihf	%r15, 0                 # encoding: [0xc0,0xf6,0x00,0x00,0x00,0x00]

	xihf	%r0, 0
	xihf	%r0, 0xffffffff
	xihf	%r15, 0

#CHECK: xilf	%r0, 0                  # encoding: [0xc0,0x07,0x00,0x00,0x00,0x00]
#CHECK: xilf	%r0, 4294967295         # encoding: [0xc0,0x07,0xff,0xff,0xff,0xff]
#CHECK: xilf	%r15, 0                 # encoding: [0xc0,0xf7,0x00,0x00,0x00,0x00]

	xilf	%r0, 0
	xilf	%r0, 0xffffffff
	xilf	%r15, 0

#CHECK: xiy	-524288, 0              # encoding: [0xeb,0x00,0x00,0x00,0x80,0x57]
#CHECK: xiy	-1, 0                   # encoding: [0xeb,0x00,0x0f,0xff,0xff,0x57]
#CHECK: xiy	0, 0                    # encoding: [0xeb,0x00,0x00,0x00,0x00,0x57]
#CHECK: xiy	1, 0                    # encoding: [0xeb,0x00,0x00,0x01,0x00,0x57]
#CHECK: xiy	524287, 0               # encoding: [0xeb,0x00,0x0f,0xff,0x7f,0x57]
#CHECK: xiy	0, 255                  # encoding: [0xeb,0xff,0x00,0x00,0x00,0x57]
#CHECK: xiy	0(%r1), 42              # encoding: [0xeb,0x2a,0x10,0x00,0x00,0x57]
#CHECK: xiy	0(%r15), 42             # encoding: [0xeb,0x2a,0xf0,0x00,0x00,0x57]
#CHECK: xiy	524287(%r1), 42         # encoding: [0xeb,0x2a,0x1f,0xff,0x7f,0x57]
#CHECK: xiy	524287(%r15), 42        # encoding: [0xeb,0x2a,0xff,0xff,0x7f,0x57]

	xiy	-524288, 0
	xiy	-1, 0
	xiy	0, 0
	xiy	1, 0
	xiy	524287, 0
	xiy	0, 255
	xiy	0(%r1), 42
	xiy	0(%r15), 42
	xiy	524287(%r1), 42
	xiy	524287(%r15), 42

#CHECK: xr	%r0, %r0                # encoding: [0x17,0x00]
#CHECK: xr	%r0, %r15               # encoding: [0x17,0x0f]
#CHECK: xr	%r15, %r0               # encoding: [0x17,0xf0]
#CHECK: xr	%r7, %r8                # encoding: [0x17,0x78]

	xr	%r0,%r0
	xr	%r0,%r15
	xr	%r15,%r0
	xr	%r7,%r8

#CHECK: xsch                            # encoding: [0xb2,0x76,0x00,0x00]

	xsch

#CHECK: xy	%r0, -524288            # encoding: [0xe3,0x00,0x00,0x00,0x80,0x57]
#CHECK: xy	%r0, -1                 # encoding: [0xe3,0x00,0x0f,0xff,0xff,0x57]
#CHECK: xy	%r0, 0                  # encoding: [0xe3,0x00,0x00,0x00,0x00,0x57]
#CHECK: xy	%r0, 1                  # encoding: [0xe3,0x00,0x00,0x01,0x00,0x57]
#CHECK: xy	%r0, 524287             # encoding: [0xe3,0x00,0x0f,0xff,0x7f,0x57]
#CHECK: xy	%r0, 0(%r1)             # encoding: [0xe3,0x00,0x10,0x00,0x00,0x57]
#CHECK: xy	%r0, 0(%r15)            # encoding: [0xe3,0x00,0xf0,0x00,0x00,0x57]
#CHECK: xy	%r0, 524287(%r1,%r15)   # encoding: [0xe3,0x01,0xff,0xff,0x7f,0x57]
#CHECK: xy	%r0, 524287(%r15,%r1)   # encoding: [0xe3,0x0f,0x1f,0xff,0x7f,0x57]
#CHECK: xy	%r15, 0                 # encoding: [0xe3,0xf0,0x00,0x00,0x00,0x57]

	xy	%r0, -524288
	xy	%r0, -1
	xy	%r0, 0
	xy	%r0, 1
	xy	%r0, 524287
	xy	%r0, 0(%r1)
	xy	%r0, 0(%r15)
	xy	%r0, 524287(%r1,%r15)
	xy	%r0, 524287(%r15,%r1)
	xy	%r15, 0

#CHECK: zap	0(1), 0(1)              # encoding: [0xf8,0x00,0x00,0x00,0x00,0x00]
#CHECK: zap	0(1), 0(1,%r1)          # encoding: [0xf8,0x00,0x00,0x00,0x10,0x00]
#CHECK: zap	0(1), 0(1,%r15)         # encoding: [0xf8,0x00,0x00,0x00,0xf0,0x00]
#CHECK: zap	0(1), 4095(1)           # encoding: [0xf8,0x00,0x00,0x00,0x0f,0xff]
#CHECK: zap	0(1), 4095(1,%r1)       # encoding: [0xf8,0x00,0x00,0x00,0x1f,0xff]
#CHECK: zap	0(1), 4095(1,%r15)      # encoding: [0xf8,0x00,0x00,0x00,0xff,0xff]
#CHECK: zap	0(1,%r1), 0(1)          # encoding: [0xf8,0x00,0x10,0x00,0x00,0x00]
#CHECK: zap	0(1,%r15), 0(1)         # encoding: [0xf8,0x00,0xf0,0x00,0x00,0x00]
#CHECK: zap	4095(1,%r1), 0(1)       # encoding: [0xf8,0x00,0x1f,0xff,0x00,0x00]
#CHECK: zap	4095(1,%r15), 0(1)      # encoding: [0xf8,0x00,0xff,0xff,0x00,0x00]
#CHECK: zap	0(16,%r1), 0(1)         # encoding: [0xf8,0xf0,0x10,0x00,0x00,0x00]
#CHECK: zap	0(16,%r15), 0(1)        # encoding: [0xf8,0xf0,0xf0,0x00,0x00,0x00]
#CHECK: zap	0(1), 0(16,%r1)         # encoding: [0xf8,0x0f,0x00,0x00,0x10,0x00]
#CHECK: zap	0(1), 0(16,%r15)        # encoding: [0xf8,0x0f,0x00,0x00,0xf0,0x00]

	zap	0(1), 0(1)
	zap	0(1), 0(1,%r1)
	zap	0(1), 0(1,%r15)
	zap	0(1), 4095(1)
	zap	0(1), 4095(1,%r1)
	zap	0(1), 4095(1,%r15)
	zap	0(1,%r1), 0(1)
	zap	0(1,%r15), 0(1)
	zap	4095(1,%r1), 0(1)
	zap	4095(1,%r15), 0(1)
	zap	0(16,%r1), 0(1)
	zap	0(16,%r15), 0(1)
	zap	0(1), 0(16,%r1)
	zap	0(1), 0(16,%r15)
