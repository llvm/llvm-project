! RUN: llvm-mc -triple=sparcv9 -mattr=+osa2011 -filetype=obj %s | llvm-objdump --mattr=+osa2011 --no-print-imm-hex -d - | FileCheck %s --check-prefix=BIN

!! SPARCv9/SPARC64 BPr branches have different offset encoding from the others,
!! make sure that our offset bits don't trample on other fields.
!! This is particularly important with backwards branches.

! BIN:  0: 02 c8 40 01  	brz %g1, 0x4
! BIN:  4: 04 c8 40 01  	brlez %g1, 0x8
! BIN:  8: 06 c8 40 01  	brlz %g1, 0xc
! BIN:  c: 0a c8 40 01  	brnz %g1, 0x10
! BIN: 10: 0c c8 40 01  	brgz %g1, 0x14
! BIN: 14: 0e c8 40 01  	brgez %g1, 0x18
brz   %g1, .+4
brlez %g1, .+4
brlz  %g1, .+4
brnz  %g1, .+4
brgz  %g1, .+4
brgez %g1, .+4

! BIN: 18: 02 f8 7f ff  	brz %g1, 0x14
! BIN: 1c: 04 f8 7f ff  	brlez %g1, 0x18
! BIN: 20: 06 f8 7f ff  	brlz %g1, 0x1c
! BIN: 24: 0a f8 7f ff  	brnz %g1, 0x20
! BIN: 28: 0c f8 7f ff  	brgz %g1, 0x24
! BIN: 2c: 0e f8 7f ff  	brgez %g1, 0x28
brz   %g1, .-4
brlez %g1, .-4
brlz  %g1, .-4
brnz  %g1, .-4
brgz  %g1, .-4
brgez %g1, .-4

!! Similarly, OSA2011 CBCond branches have different offset encoding,
!! make sure that our offset bits don't trample on other fields.
!! This is particularly important with backwards branches.

!BIN: 30: 32 c2 00 29  	cwbne	%o0, %o1, 0x34
!BIN: 34: 12 c2 00 29  	cwbe	%o0, %o1, 0x38
!BIN: 38: 34 c2 00 29  	cwbg	%o0, %o1, 0x3c
!BIN: 3c: 14 c2 00 29  	cwble	%o0, %o1, 0x40
!BIN: 40: 36 c2 00 29  	cwbge	%o0, %o1, 0x44
!BIN: 44: 16 c2 00 29  	cwbl	%o0, %o1, 0x48
!BIN: 48: 38 c2 00 29  	cwbgu	%o0, %o1, 0x4c
!BIN: 4c: 18 c2 00 29  	cwbleu	%o0, %o1, 0x50
!BIN: 50: 3a c2 00 29  	cwbcc	%o0, %o1, 0x54
!BIN: 54: 1a c2 00 29  	cwbcs	%o0, %o1, 0x58
!BIN: 58: 3c c2 00 29  	cwbpos	%o0, %o1, 0x5c
!BIN: 5c: 1c c2 00 29  	cwbneg	%o0, %o1, 0x60
!BIN: 60: 3e c2 00 29  	cwbvc	%o0, %o1, 0x64
!BIN: 64: 1e c2 00 29  	cwbvs	%o0, %o1, 0x68
cwbne  %o0, %o1, .+4
cwbe   %o0, %o1, .+4
cwbg   %o0, %o1, .+4
cwble  %o0, %o1, .+4
cwbge  %o0, %o1, .+4
cwbl   %o0, %o1, .+4
cwbgu  %o0, %o1, .+4
cwbleu %o0, %o1, .+4
cwbcc  %o0, %o1, .+4
cwbcs  %o0, %o1, .+4
cwbpos %o0, %o1, .+4
cwbneg %o0, %o1, .+4
cwbvc  %o0, %o1, .+4
cwbvs  %o0, %o1, .+4

!BIN: 68: 32 da 1f e9  	cwbne	%o0, %o1, 0x64
!BIN: 6c: 12 da 1f e9  	cwbe	%o0, %o1, 0x68
!BIN: 70: 34 da 1f e9  	cwbg	%o0, %o1, 0x6c
!BIN: 74: 14 da 1f e9  	cwble	%o0, %o1, 0x70
!BIN: 78: 36 da 1f e9  	cwbge	%o0, %o1, 0x74
!BIN: 7c: 16 da 1f e9  	cwbl	%o0, %o1, 0x78
!BIN: 80: 38 da 1f e9  	cwbgu	%o0, %o1, 0x7c
!BIN: 84: 18 da 1f e9  	cwbleu	%o0, %o1, 0x80
!BIN: 88: 3a da 1f e9  	cwbcc	%o0, %o1, 0x84
!BIN: 8c: 1a da 1f e9  	cwbcs	%o0, %o1, 0x88
!BIN: 90: 3c da 1f e9  	cwbpos	%o0, %o1, 0x8c
!BIN: 94: 1c da 1f e9  	cwbneg	%o0, %o1, 0x90
!BIN: 98: 3e da 1f e9  	cwbvc	%o0, %o1, 0x94
!BIN: 9c: 1e da 1f e9  	cwbvs	%o0, %o1, 0x98
cwbne  %o0, %o1, .-4
cwbe   %o0, %o1, .-4
cwbg   %o0, %o1, .-4
cwble  %o0, %o1, .-4
cwbge  %o0, %o1, .-4
cwbl   %o0, %o1, .-4
cwbgu  %o0, %o1, .-4
cwbleu %o0, %o1, .-4
cwbcc  %o0, %o1, .-4
cwbcs  %o0, %o1, .-4
cwbpos %o0, %o1, .-4
cwbneg %o0, %o1, .-4
cwbvc  %o0, %o1, .-4
cwbvs  %o0, %o1, .-4

!BIN: a0: 32 c2 20 21  	cwbne	%o0, 1, 0xa4
!BIN: a4: 12 c2 20 21  	cwbe	%o0, 1, 0xa8
!BIN: a8: 34 c2 20 21  	cwbg	%o0, 1, 0xac
!BIN: ac: 14 c2 20 21  	cwble	%o0, 1, 0xb0
!BIN: b0: 36 c2 20 21  	cwbge	%o0, 1, 0xb4
!BIN: b4: 16 c2 20 21  	cwbl	%o0, 1, 0xb8
!BIN: b8: 38 c2 20 21  	cwbgu	%o0, 1, 0xbc
!BIN: bc: 18 c2 20 21  	cwbleu	%o0, 1, 0xc0
!BIN: c0: 3a c2 20 21  	cwbcc	%o0, 1, 0xc4
!BIN: c4: 1a c2 20 21  	cwbcs	%o0, 1, 0xc8
!BIN: c8: 3c c2 20 21  	cwbpos	%o0, 1, 0xcc
!BIN: cc: 1c c2 20 21  	cwbneg	%o0, 1, 0xd0
!BIN: d0: 3e c2 20 21  	cwbvc	%o0, 1, 0xd4
!BIN: d4: 1e c2 20 21  	cwbvs	%o0, 1, 0xd8
cwbne  %o0, 1, .+4
cwbe   %o0, 1, .+4
cwbg   %o0, 1, .+4
cwble  %o0, 1, .+4
cwbge  %o0, 1, .+4
cwbl   %o0, 1, .+4
cwbgu  %o0, 1, .+4
cwbleu %o0, 1, .+4
cwbcc  %o0, 1, .+4
cwbcs  %o0, 1, .+4
cwbpos %o0, 1, .+4
cwbneg %o0, 1, .+4
cwbvc  %o0, 1, .+4
cwbvs  %o0, 1, .+4

!BIN:  d8: 32 da 3f e1  	cwbne	%o0, 1, 0xd4
!BIN:  dc: 12 da 3f e1  	cwbe	%o0, 1, 0xd8
!BIN:  e0: 34 da 3f e1  	cwbg	%o0, 1, 0xdc
!BIN:  e4: 14 da 3f e1  	cwble	%o0, 1, 0xe0
!BIN:  e8: 36 da 3f e1  	cwbge	%o0, 1, 0xe4
!BIN:  ec: 16 da 3f e1  	cwbl	%o0, 1, 0xe8
!BIN:  f0: 38 da 3f e1  	cwbgu	%o0, 1, 0xec
!BIN:  f4: 18 da 3f e1  	cwbleu	%o0, 1, 0xf0
!BIN:  f8: 3a da 3f e1  	cwbcc	%o0, 1, 0xf4
!BIN:  fc: 1a da 3f e1  	cwbcs	%o0, 1, 0xf8
!BIN: 100: 3c da 3f e1  	cwbpos	%o0, 1, 0xfc
!BIN: 104: 1c da 3f e1  	cwbneg	%o0, 1, 0x100
!BIN: 108: 3e da 3f e1  	cwbvc	%o0, 1, 0x104
!BIN: 10c: 1e da 3f e1  	cwbvs	%o0, 1, 0x108
cwbne  %o0, 1, .-4
cwbe   %o0, 1, .-4
cwbg   %o0, 1, .-4
cwble  %o0, 1, .-4
cwbge  %o0, 1, .-4
cwbl   %o0, 1, .-4
cwbgu  %o0, 1, .-4
cwbleu %o0, 1, .-4
cwbcc  %o0, 1, .-4
cwbcs  %o0, 1, .-4
cwbpos %o0, 1, .-4
cwbneg %o0, 1, .-4
cwbvc  %o0, 1, .-4
cwbvs  %o0, 1, .-4

!BIN: 110: 32 e2 00 29  	cxbne	%o0, %o1, 0x114
!BIN: 114: 12 e2 00 29  	cxbe	%o0, %o1, 0x118
!BIN: 118: 34 e2 00 29  	cxbg	%o0, %o1, 0x11c
!BIN: 11c: 14 e2 00 29  	cxble	%o0, %o1, 0x120
!BIN: 120: 36 e2 00 29  	cxbge	%o0, %o1, 0x124
!BIN: 124: 16 e2 00 29  	cxbl	%o0, %o1, 0x128
!BIN: 128: 38 e2 00 29  	cxbgu	%o0, %o1, 0x12c
!BIN: 12c: 18 e2 00 29  	cxbleu	%o0, %o1, 0x130
!BIN: 130: 3a e2 00 29  	cxbcc	%o0, %o1, 0x134
!BIN: 134: 1a e2 00 29  	cxbcs	%o0, %o1, 0x138
!BIN: 138: 3c e2 00 29  	cxbpos	%o0, %o1, 0x13c
!BIN: 13c: 1c e2 00 29  	cxbneg	%o0, %o1, 0x140
!BIN: 140: 3e e2 00 29  	cxbvc	%o0, %o1, 0x144
!BIN: 144: 1e e2 00 29  	cxbvs	%o0, %o1, 0x148
cxbne  %o0, %o1, .+4
cxbe   %o0, %o1, .+4
cxbg   %o0, %o1, .+4
cxble  %o0, %o1, .+4
cxbge  %o0, %o1, .+4
cxbl   %o0, %o1, .+4
cxbgu  %o0, %o1, .+4
cxbleu %o0, %o1, .+4
cxbcc  %o0, %o1, .+4
cxbcs  %o0, %o1, .+4
cxbpos %o0, %o1, .+4
cxbneg %o0, %o1, .+4
cxbvc  %o0, %o1, .+4
cxbvs  %o0, %o1, .+4

!BIN: 148: 32 fa 1f e9  	cxbne	%o0, %o1, 0x144
!BIN: 14c: 12 fa 1f e9  	cxbe	%o0, %o1, 0x148
!BIN: 150: 34 fa 1f e9  	cxbg	%o0, %o1, 0x14c
!BIN: 154: 14 fa 1f e9  	cxble	%o0, %o1, 0x150
!BIN: 158: 36 fa 1f e9  	cxbge	%o0, %o1, 0x154
!BIN: 15c: 16 fa 1f e9  	cxbl	%o0, %o1, 0x158
!BIN: 160: 38 fa 1f e9  	cxbgu	%o0, %o1, 0x15c
!BIN: 164: 18 fa 1f e9  	cxbleu	%o0, %o1, 0x160
!BIN: 168: 3a fa 1f e9  	cxbcc	%o0, %o1, 0x164
!BIN: 16c: 1a fa 1f e9  	cxbcs	%o0, %o1, 0x168
!BIN: 170: 3c fa 1f e9  	cxbpos	%o0, %o1, 0x16c
!BIN: 174: 1c fa 1f e9  	cxbneg	%o0, %o1, 0x170
!BIN: 178: 3e fa 1f e9  	cxbvc	%o0, %o1, 0x174
!BIN: 17c: 1e fa 1f e9  	cxbvs	%o0, %o1, 0x178
cxbne  %o0, %o1, .-4
cxbe   %o0, %o1, .-4
cxbg   %o0, %o1, .-4
cxble  %o0, %o1, .-4
cxbge  %o0, %o1, .-4
cxbl   %o0, %o1, .-4
cxbgu  %o0, %o1, .-4
cxbleu %o0, %o1, .-4
cxbcc  %o0, %o1, .-4
cxbcs  %o0, %o1, .-4
cxbpos %o0, %o1, .-4
cxbneg %o0, %o1, .-4
cxbvc  %o0, %o1, .-4
cxbvs  %o0, %o1, .-4

!BIN: 180: 32 e2 20 21  	cxbne	%o0, 1, 0x184
!BIN: 184: 12 e2 20 21  	cxbe	%o0, 1, 0x188
!BIN: 188: 34 e2 20 21  	cxbg	%o0, 1, 0x18c
!BIN: 18c: 14 e2 20 21  	cxble	%o0, 1, 0x190
!BIN: 190: 36 e2 20 21  	cxbge	%o0, 1, 0x194
!BIN: 194: 16 e2 20 21  	cxbl	%o0, 1, 0x198
!BIN: 198: 38 e2 20 21  	cxbgu	%o0, 1, 0x19c
!BIN: 19c: 18 e2 20 21  	cxbleu	%o0, 1, 0x1a0
!BIN: 1a0: 3a e2 20 21  	cxbcc	%o0, 1, 0x1a4
!BIN: 1a4: 1a e2 20 21  	cxbcs	%o0, 1, 0x1a8
!BIN: 1a8: 3c e2 20 21  	cxbpos	%o0, 1, 0x1ac
!BIN: 1ac: 1c e2 20 21  	cxbneg	%o0, 1, 0x1b0
!BIN: 1b0: 3e e2 20 21  	cxbvc	%o0, 1, 0x1b4
!BIN: 1b4: 1e e2 20 21  	cxbvs	%o0, 1, 0x1b8
cxbne  %o0, 1, .+4
cxbe   %o0, 1, .+4
cxbg   %o0, 1, .+4
cxble  %o0, 1, .+4
cxbge  %o0, 1, .+4
cxbl   %o0, 1, .+4
cxbgu  %o0, 1, .+4
cxbleu %o0, 1, .+4
cxbcc  %o0, 1, .+4
cxbcs  %o0, 1, .+4
cxbpos %o0, 1, .+4
cxbneg %o0, 1, .+4
cxbvc  %o0, 1, .+4
cxbvs  %o0, 1, .+4

!BIN: 1b8: 32 fa 3f e1  	cxbne	%o0, 1, 0x1b4
!BIN: 1bc: 12 fa 3f e1  	cxbe	%o0, 1, 0x1b8
!BIN: 1c0: 34 fa 3f e1  	cxbg	%o0, 1, 0x1bc
!BIN: 1c4: 14 fa 3f e1  	cxble	%o0, 1, 0x1c0
!BIN: 1c8: 36 fa 3f e1  	cxbge	%o0, 1, 0x1c4
!BIN: 1cc: 16 fa 3f e1  	cxbl	%o0, 1, 0x1c8
!BIN: 1d0: 38 fa 3f e1  	cxbgu	%o0, 1, 0x1cc
!BIN: 1d4: 18 fa 3f e1  	cxbleu	%o0, 1, 0x1d0
!BIN: 1d8: 3a fa 3f e1  	cxbcc	%o0, 1, 0x1d4
!BIN: 1dc: 1a fa 3f e1  	cxbcs	%o0, 1, 0x1d8
!BIN: 1e0: 3c fa 3f e1  	cxbpos	%o0, 1, 0x1dc
!BIN: 1e4: 1c fa 3f e1  	cxbneg	%o0, 1, 0x1e0
!BIN: 1e8: 3e fa 3f e1  	cxbvc	%o0, 1, 0x1e4
!BIN: 1ec: 1e fa 3f e1  	cxbvs	%o0, 1, 0x1e8
cxbne  %o0, 1, .-4
cxbe   %o0, 1, .-4
cxbg   %o0, 1, .-4
cxble  %o0, 1, .-4
cxbge  %o0, 1, .-4
cxbl   %o0, 1, .-4
cxbgu  %o0, 1, .-4
cxbleu %o0, 1, .-4
cxbcc  %o0, 1, .-4
cxbcs  %o0, 1, .-4
cxbpos %o0, 1, .-4
cxbneg %o0, 1, .-4
cxbvc  %o0, 1, .-4
cxbvs  %o0, 1, .-4
