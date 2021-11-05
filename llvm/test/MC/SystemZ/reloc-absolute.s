# RUN: llvm-mc -triple s390x-unknown-unknown -mcpu=z13 --show-encoding %s | FileCheck %s

# RUN: llvm-mc -triple s390x-unknown-unknown -filetype=obj -mcpu=z13 %s | \
# RUN: llvm-objdump -d - --mcpu=z13 | FileCheck %s -check-prefix=CHECK-REL

# Test relocations that can be lowered by the integrated assembler.

	.text

## BD12
# CHECK: vl %v0, b-a                            # encoding: [0xe7,0x00,0b0000AAAA,A,0x00,0x06]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: b-a, kind: FK_390_12
# CHECK-REL:                                    e7 00 00 04 00 06    	vl	%v0, 4
        .align 16
        vl %v0, b-a

# CHECK: vl %v0, b-a(%r1)                       # encoding: [0xe7,0x00,0b0001AAAA,A,0x00,0x06]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: b-a, kind: FK_390_12
# CHECK-REL:                                    e7 00 10 04 00 06    	vl	%v0, 4(%r1)
        .align 16
        vl %v0, b-a(%r1)

# CHECK: .insn vrx,253987186016262,%v0,b-a(%r1),3  # encoding: [0xe7,0x00,0b0001AAAA,A,0x30,0x06]
# CHECK-NEXT:                                      # fixup A - offset: 2, value: b-a, kind: FK_390_12
# CHECK-REL:                                       e7 00 10 04 30 06   	vl	%v0, 4(%r1), 3
        .align 16
        .insn vrx,0xe70000000006,%v0,b-a(%r1),3	   # vl

## BD20
# CHECK: lmg %r6, %r15, b-a                     # encoding: [0xeb,0x6f,0b0000AAAA,A,A,0x04]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: b-a, kind: FK_390_20
# CHECK-REL:                                    eb 6f 00 04 00 04    	lmg	%r6, %r15, 4
	.align 16
        lmg %r6, %r15, b-a

# CHECK: lmg %r6, %r15, b-a(%r1)                # encoding: [0xeb,0x6f,0b0001AAAA,A,A,0x04]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: b-a, kind: FK_390_20
# CHECK-REL:                                    eb 6f 10 04 00 04    	lmg	%r6, %r15, 4(%r1)
	.align 16
        lmg %r6, %r15, b-a(%r1)

# CHECK: .insn siy,258385232527441,b-a(%r15),240  # encoding: [0xeb,0xf0,0b1111AAAA,A,A,0x51]
# CHECK-NEXT:                                     # fixup A - offset: 2, value: b-a, kind: FK_390_20
# CHECK-REL:                                      eb f0 f0 04 00 51    	tmy	4(%r15), 240
	.align 16
        .insn siy,0xeb0000000051,b-a(%r15),240	  # tmy

## BDX12
# CHECK: la %r14, b-a                           # encoding: [0x41,0xe0,0b0000AAAA,A]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: b-a, kind: FK_390_12
# CHECK-REL:                                    41 e0 00 04  	la	%r14, 4
        .align 16
        la %r14, b-a

# CHECK: la %r14, b-a(%r1)                      # encoding: [0x41,0xe0,0b0001AAAA,A]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: b-a, kind: FK_390_12
# CHECK-REL:                                    41 e0 10 04  	la	%r14, 4(%r1)
        .align 16
        la %r14, b-a(%r1)

# CHECK: la %r14, b-a(%r1,%r2)                  # encoding: [0x41,0xe1,0b0010AAAA,A]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: b-a, kind: FK_390_12
# CHECK-REL:                                    41 e1 20 04  	la	%r14, 4(%r1,%r2)
        .align 16
        la %r14, b-a(%r1, %r2)

# CHECK: .insn vrx,253987186016262,%v2,b-a(%r2,%r3),3  # encoding: [0xe7,0x22,0b0011AAAA,A,0x30,0x06]
# CHECK-NEXT:	                                       # fixup A - offset: 2, value: b-a, kind: FK_390_12
# CHECK-REL:                                           e7 22 30 04 30 06    	vl	%v2, 4(%r2,%r3), 3
        .align 16
        .insn vrx,0xe70000000006,%v2,b-a(%r2, %r3),3   # vl

##BDX20
# CHECK: lg %r14, b-a                           # encoding: [0xe3,0xe0,0b0000AAAA,A,A,0x04]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: b-a, kind: FK_390_20
# CHECK-REL:                                    e3 e0 00 04 00 04    	lg	%r14, 4
	.align 16
	lg %r14, b-a

# CHECK: lg %r14, b-a(%r1)                      # encoding: [0xe3,0xe0,0b0001AAAA,A,A,0x04]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: b-a, kind: FK_390_20
# CHECK-REL:                                    e3 e0 10 04 00 04    	lg	%r14, 4(%r1)
	.align 16
	lg %r14, b-a(%r1)

# CHECK: lg %r14, b-a(%r1,%r2)                  # encoding: [0xe3,0xe1,0b0010AAAA,A,A,0x04]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: b-a, kind: FK_390_20
# CHECK-REL:                                    e3 e1 20 04 00 04    	lg	%r14, 4(%r1,%r2)
	.align 16
	lg %r14, b-a(%r1, %r2)

# CHECK:  .insn rxy,260584255783013,%f1,b-a(%r2,%r15)  # encoding: [0xed,0x12,0b1111AAAA,A,A,0x65]
# CHECK-NEXT:                                          # fixup A - offset: 2, value: b-a, kind: FK_390_20
# CHECK-REL:                                           ed 12 f0 04 00 65   	ldy	%f1, 4(%r2,%r15)
	.align 16
	.insn rxy,0xed0000000065,%f1,b-a(%r2,%r15)     # ldy

##BD12L4
# CHECK: tp b-a(16)                             # encoding: [0xeb,0xf0,0b0000AAAA,A,0x00,0xc0]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: b-a, kind: FK_390_12
# CHECK-REL:                                    eb f0 00 04 00 c0    	tp	4(16)
	.align 16
        tp b-a(16)

# CHECK: tp b-a(16,%r1)                         # encoding: [0xeb,0xf0,0b0001AAAA,A,0x00,0xc0]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: b-a, kind: FK_390_12
# CHECK-REL:                                    eb f0 10 04 00 c0    	tp	4(16,%r1)
	.align 16
        tp b-a(16, %r1)

##BD12L8
#SSa
# CHECK: mvc c-b(1,%r1), b-a(%r1)               # encoding: [0xd2,0x00,0b0001AAAA,A,0b0001BBBB,B]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: c-b, kind: FK_390_12
# CHECK-NEXT:                                   # fixup B - offset: 4, value: b-a, kind: FK_390_12
# CHECK-REL:                                    d2 00 10 08 10 04    	mvc	8(1,%r1), 4(%r1)
        .align 16
        mvc c-b(1,%r1), b-a(%r1)

#SSb
# CHECK: mvo c-b(16,%r1), b-a(1,%r2)            # encoding: [0xf1,0xf0,0b0001AAAA,A,0b0010BBBB,B]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: c-b, kind: FK_390_12
# CHECK-NEXT:                                   # fixup B - offset: 4, value: b-a, kind: FK_390_12
# CHECK-REL:                                    f1 f0 10 08 20 04    	mvo	8(16,%r1), 4(1,%r2)
        .align 16
        mvo c-b(16,%r1), b-a(1,%r2)

#SSc
# CHECK: srp b-a(1,%r1), b-a(%r15), 0           # encoding: [0xf0,0x00,0b0001AAAA,A,0b1111BBBB,B]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: b-a, kind: FK_390_12
# CHECK-NEXT:                                   # fixup B - offset: 4, value: b-a, kind: FK_390_12
# CHECK-REL:                                    f0 00 10 04 f0 04    	srp	4(1,%r1), 4(%r15), 0
        .align 16
        srp b-a(1,%r1), b-a(%r15), 0

##BDR12
#SSd
# CHECK: mvck c-b(%r2,%r1), b-a, %r3            # encoding: [0xd9,0x23,0b0001AAAA,A,0b0000BBBB,B]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: c-b, kind: FK_390_12
# CHECK-NEXT:                                   # fixup B - offset: 4, value: b-a, kind: FK_390_12
# CHECK-REL:                                    d9 23 10 08 00 04    	mvck	8(%r2,%r1), 4, %r3
        .align 16
	mvck c-b(%r2,%r1), b-a, %r3

# CHECK: .insn ss,238594023227392,c-b(%r2,%r1),b-a,%r3  # encoding: [0xd9,0x23,0b0001AAAA,A,0b0000BBBB,B]
# CHECK-NEXT:                                           # fixup A - offset: 2, value: c-b, kind: FK_390_12
# CHECK-NEXT:                                           # fixup B - offset: 4, value: b-a, kind: FK_390_12
# CHECK-REL:                                            d9 23 10 08 00 04    	mvck	8(%r2,%r1), 4, %r3
        .align 16
        .insn ss,0xd90000000000,c-b(%r2,%r1),b-a,%r3	# mvck

#SSe
# CHECK: lmd %r2, %r4, b-a(%r1), c-b(%r1)       # encoding: [0xef,0x24,0b0001AAAA,A,0b0001BBBB,B]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: b-a, kind: FK_390_12
# CHECK-NEXT:                                   # fixup B - offset: 4, value: c-b, kind: FK_390_12
# CHECK-REL:                                    ef 24 10 04 10 08    	lmd	%r2, %r4, 4(%r1), 8(%r1)
        .align 16
        lmd %r2, %r4, b-a(%r1), c-b(%r1)

#SSf
# CHECK: pka c-b(%r15), b-a(256,%r15)           # encoding: [0xe9,0xff,0b1111AAAA,A,0b1111BBBB,B]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: c-b, kind: FK_390_12
# CHECK-NEXT:                                   # fixup B - offset: 4, value: b-a, kind: FK_390_12
# CHECK-REL:                                    e9 ff f0 08 f0 04    	pka	8(%r15), 4(256,%r15)
        .align 16
	pka     c-b(%r15), b-a(256,%r15)

#SSE
# CHECK: strag c-b(%r1), b-a(%r15)              # encoding: [0xe5,0x02,0b0001AAAA,A,0b1111BBBB,B]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: c-b, kind: FK_390_12
# CHECK-NEXT:                                   # fixup B - offset: 4, value: b-a, kind: FK_390_12
# CHECK-REL:                                    e5 02 10 08 f0 04    	strag	8(%r1), 4(%r15)
        .align 16
        strag c-b(%r1), b-a(%r15)

# CHECK: .insn sse,251796752695296,c-b(%r1),b-a(%r15)  # encoding: [0xe5,0x02,0b0001AAAA,A,0b1111BBBB,B]
# CHECK-NEXT:                                          # fixup A - offset: 2, value: c-b, kind: FK_390_12
# CHECK-NEXT:                                          # fixup B - offset: 4, value: b-a, kind: FK_390_12
# CHECK-REL:                                           e5 02 10 08 f0 04    	strag	8(%r1), 4(%r15)
	.align 16
	.insn sse,0xe50200000000,c-b(%r1),b-a(%r15)    # strag

#SSF
# CHECK: ectg b-a, b-a(%r15), %r2               # encoding: [0xc8,0x21,0b0000AAAA,A,0b1111BBBB,B]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: b-a, kind: FK_390_12
# CHECK-NEXT:                                   # fixup B - offset: 4, value: b-a, kind: FK_390_12
# CHECK-REL:                                    c8 21 00 04 f0 04    	ectg	4, 4(%r15), %r2
        .align 16
        ectg b-a, b-a(%r15), %r2

# CHECK: .insn ssf,219906620522496,b-a,b-a(%r15),%r2   # encoding: [0xc8,0x21,0b0000AAAA,A,0b1111BBBB,B]
# CHECK-NEXT:                                          # fixup A - offset: 2, value: b-a, kind: FK_390_12
# CHECK-NEXT:                                          # fixup B - offset: 4, value: b-a, kind: FK_390_12
# CHECK-REL:                                           c8 21 00 04 f0 04    	ectg	4, 4(%r15), %r2
        .align 16
        .insn ssf,0xc80100000000,b-a,b-a(%r15),%r2     # ectg

##BDV12
# CHECK: vgeg %v0, b-a(%v0,%r1), 0              # encoding: [0xe7,0x00,0b0001AAAA,A,0x00,0x12]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: b-a, kind: FK_390_12
# CHECK-REL:                                    e7 00 10 04 00 12    	vgeg	%v0, 4(%v0,%r1), 0
        .align 16
        vgeg %v0, b-a(%v0,%r1), 0

	.type	a,@object
	.local	a
	.comm	a,4,4
	.type	b,@object
	.local	b
	.comm	b,8,4
	.type	c,@object
	.local	c
	.comm	c,4,4
