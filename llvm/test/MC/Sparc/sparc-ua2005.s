! RUN: not llvm-mc %s -triple=sparcv9 -show-encoding 2>&1 | FileCheck %s --check-prefixes=NO-UA2005 --implicit-check-not=error:
! RUN: llvm-mc %s -triple=sparcv9 -mattr=+ua2005 -show-encoding | FileCheck %s --check-prefixes=UA2005

!! UA 2005 instructions.

! NO-UA2005: error: instruction requires a CPU feature not currently enabled
! UA2005: allclean                   ! encoding: [0x85,0x88,0x00,0x00]
allclean
! NO-UA2005: error: instruction requires a CPU feature not currently enabled
! UA2005: invalw                     ! encoding: [0x8b,0x88,0x00,0x00]
invalw
! NO-UA2005: error: instruction requires a CPU feature not currently enabled
! UA2005: otherw                     ! encoding: [0x87,0x88,0x00,0x00]
otherw
! NO-UA2005: error: instruction requires a CPU feature not currently enabled
! UA2005: normalw                    ! encoding: [0x89,0x88,0x00,0x00]
normalw
