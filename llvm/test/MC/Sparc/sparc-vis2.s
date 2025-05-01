! RUN: not llvm-mc %s -triple=sparcv9 -show-encoding 2>&1 | FileCheck %s --check-prefixes=NO-VIS2
! RUN: llvm-mc %s -triple=sparcv9 -mattr=+vis2 -show-encoding | FileCheck %s --check-prefixes=VIS2 --implicit-check-not=error:

!! VIS 2 instructions.

! NO-VIS2: error: instruction requires a CPU feature not currently enabled
! VIS2: bmask %o0, %o1, %o2                     ! encoding: [0x95,0xb2,0x03,0x29]
bmask %o0, %o1, %o2
! NO-VIS2: error: instruction requires a CPU feature not currently enabled
! VIS2: bshuffle %f0, %f2, %f4                  ! encoding: [0x89,0xb0,0x09,0x82]
bshuffle %f0, %f2, %f4

! NO-VIS2: error: instruction requires a CPU feature not currently enabled
! VIS2: siam 0                                  ! encoding: [0x81,0xb0,0x10,0x20]
siam 0
! NO-VIS2: error: instruction requires a CPU feature not currently enabled
! VIS2: siam 1                                  ! encoding: [0x81,0xb0,0x10,0x21]
siam 1
! NO-VIS2: error: instruction requires a CPU feature not currently enabled
! VIS2: siam 2                                  ! encoding: [0x81,0xb0,0x10,0x22]
siam 2
! NO-VIS2: error: instruction requires a CPU feature not currently enabled
! VIS2: siam 3                                  ! encoding: [0x81,0xb0,0x10,0x23]
siam 3
! NO-VIS2: error: instruction requires a CPU feature not currently enabled
! VIS2: siam 4                                  ! encoding: [0x81,0xb0,0x10,0x24]
siam 4
! NO-VIS2: error: instruction requires a CPU feature not currently enabled
! VIS2: siam 5                                  ! encoding: [0x81,0xb0,0x10,0x25]
siam 5
! NO-VIS2: error: instruction requires a CPU feature not currently enabled
! VIS2: siam 6                                  ! encoding: [0x81,0xb0,0x10,0x26]
siam 6
! NO-VIS2: error: instruction requires a CPU feature not currently enabled
! VIS2: siam 7                                  ! encoding: [0x81,0xb0,0x10,0x27]
siam 7

! NO-VIS2: error: instruction requires a CPU feature not currently enabled
! VIS2: edge8n %o0, %o1, %o2                    ! encoding: [0x95,0xb2,0x00,0x29]
edge8n %o0, %o1, %o2
! NO-VIS2: error: instruction requires a CPU feature not currently enabled
! VIS2: edge8ln %o0, %o1, %o2                   ! encoding: [0x95,0xb2,0x00,0x69]
edge8ln %o0, %o1, %o2
! NO-VIS2: error: instruction requires a CPU feature not currently enabled
! VIS2: edge16n %o0, %o1, %o2                   ! encoding: [0x95,0xb2,0x00,0xa9]
edge16n %o0, %o1, %o2
! NO-VIS2: error: instruction requires a CPU feature not currently enabled
! VIS2: edge16ln %o0, %o1, %o2                  ! encoding: [0x95,0xb2,0x00,0xe9]
edge16ln %o0, %o1, %o2
! NO-VIS2: error: instruction requires a CPU feature not currently enabled
! VIS2: edge32n %o0, %o1, %o2                   ! encoding: [0x95,0xb2,0x01,0x29]
edge32n %o0, %o1, %o2
! NO-VIS2: error: instruction requires a CPU feature not currently enabled
! VIS2: edge32ln %o0, %o1, %o2                  ! encoding: [0x95,0xb2,0x01,0x69]
edge32ln %o0, %o1, %o2
