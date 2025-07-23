! RUN: not llvm-mc %s -triple=sparcv9 -show-encoding 2>&1 | FileCheck %s --check-prefixes=NO-UA2007 --implicit-check-not=error:
! RUN: llvm-mc %s -triple=sparcv9 -mattr=+ua2007 -show-encoding | FileCheck %s --check-prefixes=UA2007

!! UA 2007 instructions.

! NO-UA2007: error: instruction requires a CPU feature not currently enabled
! UA2007: fmadds %f1, %f3, %f5, %f7               ! encoding: [0x8f,0xb8,0x4a,0x23]
fmadds %f1, %f3, %f5, %f7
! NO-UA2007: error: instruction requires a CPU feature not currently enabled
! UA2007: fmaddd %f0, %f2, %f4, %f6               ! encoding: [0x8d,0xb8,0x08,0x42]
fmaddd %f0, %f2, %f4, %f6
! NO-UA2007: error: instruction requires a CPU feature not currently enabled
! UA2007: fmsubs %f1, %f3, %f5, %f7               ! encoding: [0x8f,0xb8,0x4a,0xa3]
fmsubs %f1, %f3, %f5, %f7
! NO-UA2007: error: instruction requires a CPU feature not currently enabled
! UA2007: fmsubd %f0, %f2, %f4, %f6               ! encoding: [0x8d,0xb8,0x08,0xc2]
fmsubd %f0, %f2, %f4, %f6

! NO-UA2007: error: instruction requires a CPU feature not currently enabled
! UA2007: fnmadds %f1, %f3, %f5, %f7              ! encoding: [0x8f,0xb8,0x4b,0xa3]
fnmadds %f1, %f3, %f5, %f7
! NO-UA2007: error: instruction requires a CPU feature not currently enabled
! UA2007: fnmaddd %f0, %f2, %f4, %f6              ! encoding: [0x8d,0xb8,0x09,0xc2]
fnmaddd %f0, %f2, %f4, %f6
! NO-UA2007: error: instruction requires a CPU feature not currently enabled
! UA2007: fnmsubs %f1, %f3, %f5, %f7              ! encoding: [0x8f,0xb8,0x4b,0x23]
fnmsubs %f1, %f3, %f5, %f7
! NO-UA2007: error: instruction requires a CPU feature not currently enabled
! UA2007: fnmsubd %f0, %f2, %f4, %f6              ! encoding: [0x8d,0xb8,0x09,0x42]
fnmsubd %f0, %f2, %f4, %f6
