! RUN: llvm-mc %s -triple=sparc   -mcpu=v9 -mattr=+hard-quad-float -show-encoding | FileCheck %s --check-prefixes=SPARC32
! RUN: llvm-mc %s -triple=sparcv9 -mcpu=v9 -mattr=+hard-quad-float -show-encoding | FileCheck %s --check-prefixes=SPARC64

!! Solaris Natural Instructions.

! SPARC32: .word    305419896
! SPARC64: .xword   305419896
.nword 0x12345678

! SPARC32: ld [%o0+8], %g1                         ! encoding: [0xc2,0x02,0x20,0x08]
! SPARC64: ldx [%o0+8], %g1                        ! encoding: [0xc2,0x5a,0x20,0x08]
ldn [%o0 + 8], %g1
! SPARC32: st %g1, [%o0+8]                         ! encoding: [0xc2,0x22,0x20,0x08]
! SPARC64: stx %g1, [%o0+8]                        ! encoding: [0xc2,0x72,0x20,0x08]
stn %g1, [%o0 + 8]
! SPARC32: lda [%o0] #ASI_AIUP, %g1                ! encoding: [0xc2,0x82,0x02,0x00]
! SPARC64: ldxa [%o0] #ASI_AIUP, %g1               ! encoding: [0xc2,0xda,0x02,0x00]
ldna [%o0] 0x10, %g1
! SPARC32: sta %g1, [%o0] #ASI_AIUP                ! encoding: [0xc2,0xa2,0x02,0x00]
! SPARC64: stxa %g1, [%o0] #ASI_AIUP               ! encoding: [0xc2,0xf2,0x02,0x00]
stna %g1, [%o0] 0x10
! SPARC32: cas  [%o0], %g0, %g1                    ! encoding: [0xc3,0xe2,0x10,0x00]
! SPARC64: casx [%o0], %g0, %g1                    ! encoding: [0xc3,0xf2,0x10,0x00]
casn [%o0], %g0, %g1
! SPARC32: sll %g0, %g1, %g2                       ! encoding: [0x85,0x28,0x00,0x01]
! SPARC64: sllx %g0, %g1, %g2                      ! encoding: [0x85,0x28,0x10,0x01]
slln %g0, %g1, %g2
! SPARC32: srl %g0, %g1, %g2                       ! encoding: [0x85,0x30,0x00,0x01]
! SPARC64: srlx %g0, %g1, %g2                      ! encoding: [0x85,0x30,0x10,0x01]
srln %g0, %g1, %g2
! SPARC32: sra %g0, %g1, %g2                       ! encoding: [0x85,0x38,0x00,0x01]
! SPARC64: srax %g0, %g1, %g2                      ! encoding: [0x85,0x38,0x10,0x01]
sran %g0, %g1, %g2
! SPARC32: st %g0, [%o0+8]                         ! encoding: [0xc0,0x22,0x20,0x08]
! SPARC64: stx %g0, [%o0+8]                        ! encoding: [0xc0,0x72,0x20,0x08]
clrn [%o0 + 8]

! SPARC32: ba %icc, .Ltmp0                         ! encoding: [0x10,0b01001AAA,A,A]
! SPARC32:                                         !   fixup A - offset: 0, value: .Ltmp0, relocation type: 41
! SPARC64: ba %xcc, .Ltmp0                         ! encoding: [0x10,0b01101AAA,A,A]
! SPARC64:                                         !   fixup A - offset: 0, value: .Ltmp0, relocation type: 41
ba %ncc, .
! SPARC32: ta %icc, 109                            ! encoding: [0x91,0xd0,0x20,0x6d]
! SPARC64: ta %xcc, 109                            ! encoding: [0x91,0xd0,0x30,0x6d]
ta %ncc, 0x6d
! SPARC32: move %icc, %g1, %g2                     ! encoding: [0x85,0x64,0x40,0x01]
! SPARC64: move %xcc, %g1, %g2                     ! encoding: [0x85,0x64,0x50,0x01]
move %ncc, %g1, %g2
! SPARC32: fmovse %icc, %f0, %f1                   ! encoding: [0x83,0xa8,0x60,0x20]
! SPARC64: fmovse %xcc, %f0, %f1                   ! encoding: [0x83,0xa8,0x70,0x20]
fmovse %ncc, %f0, %f1
! SPARC32: fmovde %icc, %f0, %f2                   ! encoding: [0x85,0xa8,0x60,0x40]
! SPARC64: fmovde %xcc, %f0, %f2                   ! encoding: [0x85,0xa8,0x70,0x40]
fmovde %ncc, %f0, %f2
! SPARC32: fmovqe %icc, %f0, %f4                   ! encoding: [0x89,0xa8,0x60,0x60]
! SPARC64: fmovqe %xcc, %f0, %f4                   ! encoding: [0x89,0xa8,0x70,0x60]
fmovqe %ncc, %f0, %f4
