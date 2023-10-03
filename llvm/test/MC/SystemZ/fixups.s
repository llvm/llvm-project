
# RUN: llvm-mc -triple s390x-unknown-unknown -mcpu=z13 --show-encoding %s | FileCheck %s

# RUN: llvm-mc -triple s390x-unknown-unknown -mcpu=z13 -filetype=obj %s | \
# RUN: llvm-readobj -r - | FileCheck %s -check-prefix=CHECK-REL

# RUN: llvm-mc -triple s390x-unknown-unknown -mcpu=z13 -filetype=obj %s | \
# RUN: llvm-objdump -d - | FileCheck %s -check-prefix=CHECK-DIS

# CHECK: larl %r14, target                      # encoding: [0xc0,0xe0,A,A,A,A]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: target+2, kind: FK_390_PC32DBL
# CHECK-REL:                                    0x{{[0-9A-F]*2}} R_390_PC32DBL target 0x2
	.align 16
	larl %r14, target

# CHECK: larl %r14, target@GOT                  # encoding: [0xc0,0xe0,A,A,A,A]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: target@GOT+2, kind: FK_390_PC32DBL
# CHECK-REL:                                    0x{{[0-9A-F]*2}} R_390_GOTENT target 0x2
	.align 16
	larl %r14, target@got

# CHECK: larl %r14, target@INDNTPOFF            # encoding: [0xc0,0xe0,A,A,A,A]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: target@INDNTPOFF+2, kind: FK_390_PC32DBL
# CHECK-REL:                                    0x{{[0-9A-F]*2}} R_390_TLS_IEENT target 0x2
	.align 16
	larl %r14, target@indntpoff

# CHECK: brasl %r14, target                     # encoding: [0xc0,0xe5,A,A,A,A]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: target+2, kind: FK_390_PC32DBL
# CHECK-REL:                                    0x{{[0-9A-F]*2}} R_390_PC32DBL target 0x2
	.align 16
	brasl %r14, target

# CHECK: brasl %r14, target@PLT                 # encoding: [0xc0,0xe5,A,A,A,A]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: target@PLT+2, kind: FK_390_PC32DBL
# CHECK-REL:                                    0x{{[0-9A-F]*2}} R_390_PLT32DBL target 0x2
	.align 16
	brasl %r14, target@plt

# CHECK: brasl %r14, target@PLT:tls_gdcall:sym  # encoding: [0xc0,0xe5,A,A,A,A]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: target@PLT+2, kind: FK_390_PC32DBL
# CHECK-NEXT:                                   # fixup B - offset: 0, value: sym@TLSGD, kind: FK_390_TLS_CALL
# CHECK-REL:                                    0x{{[0-9A-F]*2}} R_390_PLT32DBL target 0x2
# CHECK-REL:                                    0x{{[0-9A-F]*0}} R_390_TLS_GDCALL sym 0x0
	.align 16
	brasl %r14, target@plt:tls_gdcall:sym

# CHECK: brasl %r14, target@PLT:tls_ldcall:sym  # encoding: [0xc0,0xe5,A,A,A,A]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: target@PLT+2, kind: FK_390_PC32DBL
# CHECK-NEXT:                                   # fixup B - offset: 0, value: sym@TLSLDM, kind: FK_390_TLS_CALL
# CHECK-REL:                                    0x{{[0-9A-F]*2}} R_390_PLT32DBL target 0x2
# CHECK-REL:                                    0x{{[0-9A-F]*0}} R_390_TLS_LDCALL sym 0x0
	.align 16
	brasl %r14, target@plt:tls_ldcall:sym

# CHECK: bras %r14, target                      # encoding: [0xa7,0xe5,A,A]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: target+2, kind: FK_390_PC16DBL
# CHECK-REL:                                    0x{{[0-9A-F]*2}} R_390_PC16DBL target 0x2
	.align 16
	bras %r14, target

# CHECK: bras %r14, target@PLT                  # encoding: [0xa7,0xe5,A,A]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: target@PLT+2, kind: FK_390_PC16DBL
# CHECK-REL:                                    0x{{[0-9A-F]*2}} R_390_PLT16DBL target 0x2
	.align 16
	bras %r14, target@plt

# CHECK: bras %r14, target@PLT:tls_gdcall:sym   # encoding: [0xa7,0xe5,A,A]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: target@PLT+2, kind: FK_390_PC16DBL
# CHECK-NEXT:                                   # fixup B - offset: 0, value: sym@TLSGD, kind: FK_390_TLS_CALL
# CHECK-REL:                                    0x{{[0-9A-F]*2}} R_390_PLT16DBL target 0x2
# CHECK-REL:                                    0x{{[0-9A-F]*0}} R_390_TLS_GDCALL sym 0x0
	.align 16
	bras %r14, target@plt:tls_gdcall:sym

# CHECK: bras %r14, target@PLT:tls_ldcall:sym   # encoding: [0xa7,0xe5,A,A]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: target@PLT+2, kind: FK_390_PC16DBL
# CHECK-NEXT:                                   # fixup B - offset: 0, value: sym@TLSLDM, kind: FK_390_TLS_CALL
# CHECK-REL:                                    0x{{[0-9A-F]*2}} R_390_PLT16DBL target 0x2
# CHECK-REL:                                    0x{{[0-9A-F]*0}} R_390_TLS_LDCALL sym 0x0
	.align 16
	bras %r14, target@plt:tls_ldcall:sym


# Symbolic displacements

## BD12
# CHECK: vl %v0, src                            # encoding: [0xe7,0x00,0b0000AAAA,A,0x00,0x06]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: src, kind: FK_390_U12Imm
# CHECK-REL:                                    0x{{[0-9A-F]*2}} R_390_12 src 0x0
        .align 16
        vl %v0, src

# CHECK: vl %v0, src(%r1)                       # encoding: [0xe7,0x00,0b0001AAAA,A,0x00,0x06]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: src, kind: FK_390_U12Imm
# CHECK-REL:                                    0x{{[0-9A-F]*2}} R_390_12 src 0x0
        .align 16
        vl %v0, src(%r1)

# CHECK: .insn vrx,253987186016262,%v0,src(%r1),3  # encoding: [0xe7,0x00,0b0001AAAA,A,0x30,0x06]
# CHECK-NEXT:                                      # fixup A - offset: 2, value: src, kind: FK_390_U12Imm
# CHECK-REL:                                       0x{{[0-9A-F]*2}} R_390_12 src 0x0
        .align 16
        .insn vrx,0xe70000000006,%v0,src(%r1),3	   # vl

## BD20
# CHECK: lmg %r6, %r15, src                     # encoding: [0xeb,0x6f,0b0000AAAA,A,A,0x04]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: src, kind: FK_390_S20Imm
# CHECK-REL:                                    0x{{[0-9A-F]*2}} R_390_20 src 0x0
	.align 16
        lmg %r6, %r15, src

# CHECK: lmg %r6, %r15, src(%r1)                # encoding: [0xeb,0x6f,0b0001AAAA,A,A,0x04]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: src, kind: FK_390_S20Imm
# CHECK-REL:                                    0x{{[0-9A-F]*2}} R_390_20 src 0x0
	.align 16
        lmg %r6, %r15, src(%r1)

# CHECK: .insn siy,258385232527441,src(%r15),240  # encoding: [0xeb,0xf0,0b1111AAAA,A,A,0x51]
# CHECK-NEXT:                                     # fixup A - offset: 2, value: src, kind: FK_390_S20Imm
# CHECK-REL:                                      0x{{[0-9A-F]*2}} R_390_20 src 0x0
	.align 16
        .insn siy,0xeb0000000051,src(%r15),240	  # tmy

## BDX12
# CHECK: la %r14, src                           # encoding: [0x41,0xe0,0b0000AAAA,A]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: src, kind: FK_390_U12Imm
# CHECK-REL:                                    0x{{[0-9A-F]*2}} R_390_12 src 0x0
        .align 16
        la %r14, src

# CHECK: la %r14, src(%r1)                      # encoding: [0x41,0xe0,0b0001AAAA,A]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: src, kind: FK_390_U12Imm
# CHECK-REL:                                    0x{{[0-9A-F]*2}} R_390_12 src 0x0
        .align 16
        la %r14, src(%r1)

# CHECK: la %r14, src(%r1,%r2)                  # encoding: [0x41,0xe1,0b0010AAAA,A]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: src, kind: FK_390_U12Imm
# CHECK-REL:                                    0x{{[0-9A-F]*2}} R_390_12 src 0x0
        .align 16
        la %r14, src(%r1, %r2)

# CHECK: .insn vrx,253987186016262,%v2,src(%r2,%r3),3  # encoding: [0xe7,0x22,0b0011AAAA,A,0x30,0x06]
# CHECK-NEXT:	                                       # fixup A - offset: 2, value: src, kind: FK_390_U12Imm
# CHECK-REL:                                           0x{{[0-9A-F]*2}} R_390_12 src 0x0
        .align 16
        .insn vrx,0xe70000000006,%v2,src(%r2, %r3),3   # vl

##BDX20
# CHECK: lg %r14, src                           # encoding: [0xe3,0xe0,0b0000AAAA,A,A,0x04]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: src, kind: FK_390_S20Imm
# CHECK-REL:                                    0x{{[0-9A-F]*2}} R_390_20 src 0x0
	.align 16
	lg %r14, src

# CHECK: lg %r14, src(%r1)                      # encoding: [0xe3,0xe0,0b0001AAAA,A,A,0x04]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: src, kind: FK_390_S20Imm
# CHECK-REL:                                    0x{{[0-9A-F]*2}} R_390_20 src 0x0
	.align 16
	lg %r14, src(%r1)

# CHECK: lg %r14, src(%r1,%r2)                  # encoding: [0xe3,0xe1,0b0010AAAA,A,A,0x04]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: src, kind: FK_390_S20Imm
# CHECK-REL:                                    0x{{[0-9A-F]*2}} R_390_20 src 0x0
	.align 16
	lg %r14, src(%r1, %r2)

# CHECK:  .insn rxy,260584255783013,%f1,src(%r2,%r15)  # encoding: [0xed,0x12,0b1111AAAA,A,A,0x65]
# CHECK-NEXT:                                          # fixup A - offset: 2, value: src, kind: FK_390_S20Imm
# CHECK-REL:                                           0x{{[0-9A-F]*2}} R_390_20 src 0x0
	.align 16
	.insn rxy,0xed0000000065,%f1,src(%r2,%r15)     # ldy

##BD12L4
# CHECK: tp src(16)                             # encoding: [0xeb,0xf0,0b0000AAAA,A,0x00,0xc0]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: src, kind: FK_390_U12Imm
# CHECK-REL:                                    0x{{[0-9A-F]*2}} R_390_12 src 0x0
	.align 16
        tp src(16)

# CHECK: tp src(16,%r1)                         # encoding: [0xeb,0xf0,0b0001AAAA,A,0x00,0xc0]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: src, kind: FK_390_U12Imm
# CHECK-REL:                                    0x{{[0-9A-F]*2}} R_390_12 src 0x0
	.align 16
        tp src(16, %r1)

##BD12L8
#SSa
# CHECK: mvc dst(1,%r1), src(%r1)               # encoding: [0xd2,0x00,0b0001AAAA,A,0b0001BBBB,B]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: dst, kind: FK_390_U12Imm
# CHECK-NEXT:                                   # fixup B - offset: 4, value: src, kind: FK_390_U12Imm
# CHECK-REL:                                    0x{{[0-9A-F]*2}} R_390_12 dst 0x0
# CHECK-REL:                                    0x{{[0-9A-F]*4}} R_390_12 src 0x0
        .align 16
        mvc dst(1,%r1), src(%r1)

#SSb
# CHECK: mvo src(16,%r1), src(1,%r2)            # encoding: [0xf1,0xf0,0b0001AAAA,A,0b0010BBBB,B]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: src, kind: FK_390_U12Imm
# CHECK-NEXT:                                   # fixup B - offset: 4, value: src, kind: FK_390_U12Imm
# CHECK-REL:                                    0x{{[0-9A-F]*2}} R_390_12 src 0x0
# CHECK-REL:                                    0x{{[0-9A-F]*4}} R_390_12 src 0x0
        .align 16
        mvo src(16,%r1), src(1,%r2)

#SSc
# CHECK: srp src(1,%r1), src(%r15), 0           # encoding: [0xf0,0x00,0b0001AAAA,A,0b1111BBBB,B]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: src, kind: FK_390_U12Imm
# CHECK-NEXT:                                   # fixup B - offset: 4, value: src, kind: FK_390_U12Imm
# CHECK-REL:                                    0x{{[0-9A-F]*2}} R_390_12 src 0x0
# CHECK-REL:                                    0x{{[0-9A-F]*4}} R_390_12 src 0x0
        .align 16
        srp src(1,%r1), src(%r15), 0

##BDR12
#SSd
# CHECK: mvck dst(%r2,%r1), src, %r3            # encoding: [0xd9,0x23,0b0001AAAA,A,0b0000BBBB,B]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: dst, kind: FK_390_U12Imm
# CHECK-NEXT:                                   # fixup B - offset: 4, value: src, kind: FK_390_U12Imm
# CHECK-REL:                                    0x{{[0-9A-F]*2}} R_390_12 dst 0x0
# CHECK-REL:                                    0x{{[0-9A-F]*4}} R_390_12 src 0x0
        .align 16
	mvck dst(%r2,%r1), src, %r3

# CHECK: .insn ss,238594023227392,dst(%r2,%r1),src,%r3  # encoding: [0xd9,0x23,0b0001AAAA,A,0b0000BBBB,B]
# CHECK-NEXT:                                           # fixup A - offset: 2, value: dst, kind: FK_390_U12Imm
# CHECK-NEXT:                                           # fixup B - offset: 4, value: src, kind: FK_390_U12Imm
# CHECK-REL:                                            0x{{[0-9A-F]*2}} R_390_12 dst 0x0
# CHECK-REL:                                            0x{{[0-9A-F]*4}} R_390_12 src 0x0
        .align 16
        .insn ss,0xd90000000000,dst(%r2,%r1),src,%r3	# mvck

#SSe
# CHECK: lmd %r2, %r4, src1(%r1), src2(%r1)     # encoding: [0xef,0x24,0b0001AAAA,A,0b0001BBBB,B]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: src1, kind: FK_390_U12Imm
# CHECK-NEXT:                                   # fixup B - offset: 4, value: src2, kind: FK_390_U12Imm
# CHECK-REL:                                    0x{{[0-9A-F]*2}} R_390_12 src1 0x0
# CHECK-REL:                                    0x{{[0-9A-F]*4}} R_390_12 src2 0x0
        .align 16
        lmd %r2, %r4, src1(%r1), src2(%r1)

#SSf
# CHECK: pka dst(%r15), src(256,%r15)           # encoding: [0xe9,0xff,0b1111AAAA,A,0b1111BBBB,B]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: dst, kind: FK_390_U12Imm
# CHECK-NEXT:                                   # fixup B - offset: 4, value: src, kind: FK_390_U12Imm
# CHECK-REL:                                    0x{{[0-9A-F]*2}} R_390_12 dst 0x0
# CHECK-REL:                                    0x{{[0-9A-F]*4}} R_390_12 src 0x0
        .align 16
	pka     dst(%r15), src(256,%r15)

#SSE
# CHECK: strag dst(%r1), src(%r15)              # encoding: [0xe5,0x02,0b0001AAAA,A,0b1111BBBB,B]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: dst, kind: FK_390_U12Imm
# CHECK-NEXT:                                   # fixup B - offset: 4, value: src, kind: FK_390_U12Imm
# CHECK-REL:                                    0x{{[0-9A-F]*2}} R_390_12 dst 0x0
# CHECK-REL:                                    0x{{[0-9A-F]*4}} R_390_12 src 0x0
        .align 16
        strag dst(%r1), src(%r15)

# CHECK: .insn sse,251796752695296,dst(%r1),src(%r15)  # encoding: [0xe5,0x02,0b0001AAAA,A,0b1111BBBB,B]
# CHECK-NEXT:                                          # fixup A - offset: 2, value: dst, kind: FK_390_U12Imm
# CHECK-NEXT:                                          # fixup B - offset: 4, value: src, kind: FK_390_U12Imm
# CHECK-REL:                                           0x{{[0-9A-F]*2}} R_390_12 dst 0x0
# CHECK-REL:                                           0x{{[0-9A-F]*4}} R_390_12 src 0x0
	.align 16
	.insn sse,0xe50200000000,dst(%r1),src(%r15)    # strag

#SSF
# CHECK: ectg src, src(%r15), %r2               # encoding: [0xc8,0x21,0b0000AAAA,A,0b1111BBBB,B]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: src, kind: FK_390_U12Imm
# CHECK-NEXT:                                   # fixup B - offset: 4, value: src, kind: FK_390_U12Imm
# CHECK-REL:                                    0x{{[0-9A-F]*2}} R_390_12 src 0x0
# CHECK-REL:                                    0x{{[0-9A-F]*4}} R_390_12 src 0x0
        .align 16
        ectg src, src(%r15), %r2

# CHECK: .insn ssf,219906620522496,src,src(%r15),%r2   # encoding: [0xc8,0x21,0b0000AAAA,A,0b1111BBBB,B]
# CHECK-NEXT:                                          # fixup A - offset: 2, value: src, kind: FK_390_U12Imm
# CHECK-NEXT:                                          # fixup B - offset: 4, value: src, kind: FK_390_U12Imm
# CHECK-REL:                                           0x{{[0-9A-F]*2}} R_390_12 src 0x0
# CHECK-REL:                                           0x{{[0-9A-F]*4}} R_390_12 src 0x0
        .align 16
        .insn ssf,0xc80100000000,src,src(%r15),%r2     # ectg

##BDV12
# CHECK: vgeg %v0, src(%v0,%r1), 0              # encoding: [0xe7,0x00,0b0001AAAA,A,0x00,0x12]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: src, kind: FK_390_U12Imm
# CHECK-REL:                                    0x{{[0-9A-F]*2}} R_390_12 src 0x0
        .align 16
        vgeg %v0, src(%v0,%r1), 0

## Fixup for second operand only
# CHECK:  mvc     32(8,%r0), src                # encoding: [0xd2,0x07,0x00,0x20,0b0000AAAA,A]
# CHECK-NEXT:                                   # fixup A - offset: 4, value: src, kind: FK_390_U12Imm
        .align 16
        mvc     32(8,%r0),src

##U8
# CHECK: cli 0(%r1), src                        # encoding: [0x95,A,0x10,0x00]
# CHECK-NEXT:                                   # fixup A - offset: 1, value: src, kind: FK_390_U8Imm
# CHECK-REL:                                    0x{{[0-9A-F]+}} R_390_8 src 0x0
       .align 16
       cli 0(%r1),src

# CHECK: [[L:\..+]]:
# CHECK-NEXT: cli 0(%r1), local_u8-[[L]]        # encoding: [0x95,A,0x10,0x00]
# CHECK-NEXT:                                   # fixup A - offset: 1, value: local_u8-[[L]], kind: FK_390_U8Imm
# CHECK-DIS:                                    95 04 10 00   cli     0(%r1), 4
       .align 16
       cli 0(%r1),local_u8-.
local_u8:

##S8
# CHECK: asi 0(%r1), src                        # encoding: [0xeb,A,0x10,0x00,0x00,0x6a]
# CHECK-NEXT:                                   # fixup A - offset: 1, value: src, kind: FK_390_S8Imm
# CHECK-REL:                                    0x{{[0-9A-F]+}} R_390_8 src 0x0
       .align 16
       asi 0(%r1),src

# CHECK: [[L:\..+]]:
# CHECK-NEXT: asi 0(%r1), local_s8-[[L]]        # encoding: [0xeb,A,0x10,0x00,0x00,0x6a]
# CHECK-NEXT:                                   # fixup A - offset: 1, value: local_s8-[[L]], kind: FK_390_S8Imm
# CHECK-DIS:                                    eb 06 10 00 00 6a     asi     0(%r1), 6
       .align 16
       asi 0(%r1),local_s8-.
local_s8:

##U16
# CHECK: oill %r1, src                          # encoding: [0xa5,0x1b,A,A]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: src, kind: FK_390_U16Imm
# CHECK-REL:                                    0x{{[0-9A-F]+}} R_390_16 src 0x0
        .align 16
        oill %r1,src

# CHECK: [[L:\..+]]:
# CHECK-NEXT: oill %r1, local_u16-[[L]]         # encoding: [0xa5,0x1b,A,A]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: local_u16-[[L]], kind: FK_390_U16Imm
# CHECK-DIS:                                    a5 1b 00 04   oill    %r1, 4
        .align 16
        oill %r1,local_u16-.
local_u16:

# CHECK: [[L:\..+]]:
# CHECK-NEXT: oill %r1, src-[[L]]               # encoding: [0xa5,0x1b,A,A]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: src-[[L]], kind: FK_390_U16Imm
# CHECK-REL:                                    0x{{[0-9A-F]+}} R_390_PC16 src 0x2
        .align 16
        oill %r1,src-.

##S16
# CHECK: lghi %r1, src                          # encoding: [0xa7,0x19,A,A]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: src, kind: FK_390_S16Imm
# CHECK-REL:                                    0x{{[0-9A-F]+}} R_390_16 src 0x0
        .align 16
        lghi %r1,src

# CHECK: [[L:\..+]]:
# CHECK-NEXT: lghi %r1, local_s16-[[L]]         # encoding: [0xa7,0x19,A,A]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: local_s16-[[L]], kind: FK_390_S16Imm
# CHECK-DIS:                                    a7 19 00 04   lghi    %r1, 4
        .align 16
        lghi %r1,local_s16-.
local_s16:

# CHECK: [[L:\..+]]:
# CHECK-NEXT: lghi %r1, src-[[L]]               # encoding: [0xa7,0x19,A,A]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: src-[[L]], kind: FK_390_S16Imm
# CHECK-REL:                                    0x{{[0-9A-F]+}} R_390_PC16 src 0x2
        .align 16
        lghi %r1,src-.

##U32
# CHECK: clfi %r1, src                          # encoding: [0xc2,0x1f,A,A,A,A]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: src, kind: FK_390_U32Imm
# CHECK-REL:                                    0x{{[0-9A-F]+}} R_390_32 src 0x0
        .align 16
        clfi %r1,src

# CHECK: [[L:\..+]]:
# CHECK-NEXT: clfi %r1, local_u32-[[L]]         # encoding: [0xc2,0x1f,A,A,A,A]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: local_u32-[[L]], kind: FK_390_U32Imm
# CHECK-DIS:                                    c2 1f 00 00 00 06     clfi    %r1, 6
        .align 16
        clfi %r1,local_u32-.
local_u32:

# CHECK: [[L:\..+]]:
# CHECK: clfi %r1, src-[[L]]                    # encoding: [0xc2,0x1f,A,A,A,A]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: src-[[L]], kind: FK_390_U32Imm
# CHECK-REL:                                    0x{{[0-9A-F]+}} R_390_PC32 src 0x2
        .align 16
        clfi %r1,src-.

##S32
# CHECK: lgfi %r1, src                          # encoding: [0xc0,0x11,A,A,A,A]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: src, kind: FK_390_S32Imm
# CHECK-REL:                                    0x{{[0-9A-F]+}} R_390_32 src 0x0
        .align 16
        lgfi %r1,src

# CHECK: [[L:\..+]]:
# CHECK: lgfi %r1, local_s32-[[L]]              # encoding: [0xc0,0x11,A,A,A,A]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: local_s32-[[L]], kind: FK_390_S32Imm
# CHECK-DIS:                                    c0 11 00 00 00 06     lgfi    %r1, 6
        .align 16
        lgfi %r1,local_s32-.
local_s32:

# CHECK: [[L:\..+]]:
# CHECK: lgfi %r1, src-[[L]]                    # encoding: [0xc0,0x11,A,A,A,A]
# CHECK-NEXT:                                   # fixup A - offset: 2, value: src-[[L]], kind: FK_390_S32Imm
# CHECK-REL:                                    0x{{[0-9A-F]+}} R_390_PC32 src 0x2
        .align 16
        lgfi %r1,src-.

# Data relocs
# llvm-mc does not show any "encoding" string for data, so we just check the relocs

# CHECK-REL: .rela.data
	.data

# CHECK-REL: 0x{{[0-9A-F]*0}} R_390_TLS_LE64 target 0x0
	.align 16
	.quad target@ntpoff

# CHECK-REL: 0x{{[0-9A-F]*0}} R_390_TLS_LDO64 target 0x0
	.align 16
	.quad target@dtpoff

# CHECK-REL: 0x{{[0-9A-F]*0}} R_390_TLS_LDM64 target 0x0
	.align 16
	.quad target@tlsldm

# CHECK-REL: 0x{{[0-9A-F]*0}} R_390_TLS_GD64 target 0x0
	.align 16
	.quad target@tlsgd

# CHECK-REL: 0x{{[0-9A-F]*0}} R_390_TLS_LE32 target 0x0
	.align 16
	.long target@ntpoff

# CHECK-REL: 0x{{[0-9A-F]*0}} R_390_TLS_LDO32 target 0x0
	.align 16
	.long target@dtpoff

# CHECK-REL: 0x{{[0-9A-F]*0}} R_390_TLS_LDM32 target 0x0
	.align 16
	.long target@tlsldm

# CHECK-REL: 0x{{[0-9A-F]*0}} R_390_TLS_GD32 target 0x0
	.align 16
	.long target@tlsgd

