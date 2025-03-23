! RUN: not llvm-mc %s -triple=sparcv9 -show-encoding 2>&1 | FileCheck %s --check-prefixes=NO-VIS3
! RUN: llvm-mc %s -triple=sparcv9 -mattr=+vis3 -show-encoding | FileCheck %s --check-prefixes=VIS3 --implicit-check-not=error:

!! VIS 3 instructions.

! NO-VIS3: error: instruction requires a CPU feature not currently enabled
! VIS3: addxc %o0, %o1, %o2                     ! encoding: [0x95,0xb2,0x02,0x29]
addxc %o0, %o1, %o2
! NO-VIS3: error: instruction requires a CPU feature not currently enabled
! VIS3: addxccc %o0, %o1, %o2                   ! encoding: [0x95,0xb2,0x02,0x69]
addxccc %o0, %o1, %o2

! NO-VIS3: error: instruction requires a CPU feature not currently enabled
! VIS3: cmask8 %o0                              ! encoding: [0x81,0xb0,0x03,0x68]
cmask8 %o0
! NO-VIS3: error: instruction requires a CPU feature not currently enabled
! VIS3: cmask16 %o0                             ! encoding: [0x81,0xb0,0x03,0xa8]
cmask16 %o0
! NO-VIS3: error: instruction requires a CPU feature not currently enabled
! VIS3: cmask32 %o0                             ! encoding: [0x81,0xb0,0x03,0xe8]
cmask32 %o0

! NO-VIS3: error: instruction requires a CPU feature not currently enabled
! VIS3: fchksm16 %f0, %f2, %f4                  ! encoding: [0x89,0xb0,0x08,0x82]
fchksm16 %f0, %f2, %f4
! NO-VIS3: error: instruction requires a CPU feature not currently enabled
! VIS3: fmean16 %f0, %f2, %f4                   ! encoding: [0x89,0xb0,0x08,0x02]
fmean16 %f0, %f2, %f4

! NO-VIS3: error: instruction requires a CPU feature not currently enabled
! VIS3: fhadds %f1, %f3, %f5                    ! encoding: [0x8b,0xa0,0x4c,0x23]
fhadds %f1, %f3, %f5
! NO-VIS3: error: instruction requires a CPU feature not currently enabled
! VIS3: fhaddd %f0, %f2, %f4                    ! encoding: [0x89,0xa0,0x0c,0x42]
fhaddd %f0, %f2, %f4
! NO-VIS3: error: instruction requires a CPU feature not currently enabled
! VIS3: fhsubs %f1, %f3, %f5                    ! encoding: [0x8b,0xa0,0x4c,0xa3]
fhsubs %f1, %f3, %f5
! NO-VIS3: error: instruction requires a CPU feature not currently enabled
! VIS3: fhsubd %f0, %f2, %f4                    ! encoding: [0x89,0xa0,0x0c,0xc2]
fhsubd %f0, %f2, %f4
! NO-VIS3: error: instruction requires a CPU feature not currently enabled
! VIS3: flcmps %fcc0, %f3, %f5                  ! encoding: [0x81,0xb0,0xea,0x25]
flcmps %fcc0, %f3, %f5
! NO-VIS3: error: instruction requires a CPU feature not currently enabled
! VIS3: flcmpd %fcc0, %f2, %f4                  ! encoding: [0x81,0xb0,0xaa,0x44]
flcmpd %fcc0, %f2, %f4

! NO-VIS3: error: instruction requires a CPU feature not currently enabled
! VIS3: fnadds %f1, %f3, %f5                    ! encoding: [0x8b,0xa0,0x4a,0x23]
fnadds %f1, %f3, %f5
! NO-VIS3: error: instruction requires a CPU feature not currently enabled
! VIS3: fnaddd %f0, %f2, %f4                    ! encoding: [0x89,0xa0,0x0a,0x42]
fnaddd %f0, %f2, %f4
! NO-VIS3: error: instruction requires a CPU feature not currently enabled
! VIS3: fnhadds %f1, %f3, %f5                   ! encoding: [0x8b,0xa0,0x4e,0x23]
fnhadds %f1, %f3, %f5
! NO-VIS3: error: instruction requires a CPU feature not currently enabled
! VIS3: fnhaddd %f0, %f2, %f4                   ! encoding: [0x89,0xa0,0x0e,0x42]
fnhaddd %f0, %f2, %f4

! NO-VIS3: error: instruction requires a CPU feature not currently enabled
! VIS3: fnmuls %f1, %f3, %f5                    ! encoding: [0x8b,0xa0,0x4b,0x23]
fnmuls %f1, %f3, %f5
! NO-VIS3: error: instruction requires a CPU feature not currently enabled
! VIS3: fnmuld %f0, %f2, %f4                    ! encoding: [0x89,0xa0,0x0b,0x42]
fnmuld %f0, %f2, %f4
! NO-VIS3: error: instruction requires a CPU feature not currently enabled
! VIS3: fnsmuld %f1, %f3, %f4                   ! encoding: [0x89,0xa0,0x4f,0x23]
fnsmuld %f1, %f3, %f4

! NO-VIS3: error: instruction requires a CPU feature not currently enabled
! VIS3: fpadd64 %f0, %f2, %f4                   ! encoding: [0x89,0xb0,0x08,0x42]
fpadd64 %f0, %f2, %f4

! NO-VIS3: error: instruction requires a CPU feature not currently enabled
! VIS3: fsll16 %f0, %f2, %f4                    ! encoding: [0x89,0xb0,0x04,0x22]
fsll16 %f0, %f2, %f4
! NO-VIS3: error: instruction requires a CPU feature not currently enabled
! VIS3: fsrl16 %f0, %f2, %f4                    ! encoding: [0x89,0xb0,0x04,0x62]
fsrl16 %f0, %f2, %f4
! NO-VIS3: error: instruction requires a CPU feature not currently enabled
! VIS3: fsll32 %f0, %f2, %f4                    ! encoding: [0x89,0xb0,0x04,0xa2]
fsll32 %f0, %f2, %f4
! NO-VIS3: error: instruction requires a CPU feature not currently enabled
! VIS3: fsrl32 %f0, %f2, %f4                    ! encoding: [0x89,0xb0,0x04,0xe2]
fsrl32 %f0, %f2, %f4
! NO-VIS3: error: instruction requires a CPU feature not currently enabled
! VIS3: fslas16 %f0, %f2, %f4                   ! encoding: [0x89,0xb0,0x05,0x22]
fslas16 %f0, %f2, %f4
! NO-VIS3: error: instruction requires a CPU feature not currently enabled
! VIS3: fsra16 %f0, %f2, %f4                    ! encoding: [0x89,0xb0,0x05,0x62]
fsra16 %f0, %f2, %f4
! NO-VIS3: error: instruction requires a CPU feature not currently enabled
! VIS3: fslas32 %f0, %f2, %f4                   ! encoding: [0x89,0xb0,0x05,0xa2]
fslas32 %f0, %f2, %f4
! NO-VIS3: error: instruction requires a CPU feature not currently enabled
! VIS3: fsra32 %f0, %f2, %f4                    ! encoding: [0x89,0xb0,0x05,0xe2]
fsra32 %f0, %f2, %f4

! NO-VIS3: error: instruction requires a CPU feature not currently enabled
! VIS3: lzcnt %o0, %o1                          ! encoding: [0x93,0xb0,0x02,0xe8]
lzcnt %o0, %o1

! NO-VIS3: error: instruction requires a CPU feature not currently enabled
! VIS3: movstosw %f0, %o0                       ! encoding: [0x91,0xb0,0x22,0x60]
movstosw %f0, %o0
! NO-VIS3: error: instruction requires a CPU feature not currently enabled
! VIS3: movstouw %f0, %o0                       ! encoding: [0x91,0xb0,0x22,0x20]
movstouw %f0, %o0
! NO-VIS3: error: instruction requires a CPU feature not currently enabled
! VIS3: movdtox %f0, %o0                        ! encoding: [0x91,0xb0,0x22,0x00]
movdtox %f0, %o0
! NO-VIS3: error: instruction requires a CPU feature not currently enabled
! VIS3: movwtos %o0, %f0                        ! encoding: [0x81,0xb0,0x23,0x28]
movwtos %o0, %f0
! NO-VIS3: error: instruction requires a CPU feature not currently enabled
! VIS3: movxtod %o0, %f0                        ! encoding: [0x81,0xb0,0x23,0x08]
movxtod %o0, %f0

! NO-VIS3: error: instruction requires a CPU feature not currently enabled
! VIS3: pdistn %f0, %f2, %o0                    ! encoding: [0x91,0xb0,0x07,0xe2]
pdistn %f0, %f2, %o0

! NO-VIS3: error: instruction requires a CPU feature not currently enabled
! VIS3: umulxhi %o0, %o1, %o2                   ! encoding: [0x95,0xb2,0x02,0xc9]
umulxhi %o0, %o1, %o2
! NO-VIS3: error: instruction requires a CPU feature not currently enabled
! VIS3: xmulx %o0, %o1, %o2                     ! encoding: [0x95,0xb2,0x22,0xa9]
xmulx %o0, %o1, %o2
! NO-VIS3: error: instruction requires a CPU feature not currently enabled
! VIS3: xmulxhi %o0, %o1, %o2                   ! encoding: [0x95,0xb2,0x22,0xc9]
xmulxhi %o0, %o1, %o2
