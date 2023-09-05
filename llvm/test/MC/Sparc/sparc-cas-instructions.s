! RUN: not llvm-mc %s -arch=sparc -show-encoding 2>&1 | FileCheck %s --check-prefix=V8
! RUN: not llvm-mc %s -arch=sparc -mattr=+hasleoncasa -show-encoding 2>&1 | FileCheck %s --check-prefix=LEON
! RUN: llvm-mc %s -arch=sparcv9 -show-encoding | FileCheck %s --check-prefix=V9

! V8: error: instruction requires a CPU feature not currently enabled
! V9: cas [%i0], %l6, %o2   ! encoding: [0xd5,0xe6,0x10,0x16]
! LEON: error: instruction requires a CPU feature not currently enabled
cas [%i0], %l6, %o2

! V8: error: instruction requires a CPU feature not currently enabled
! V9: casl [%i0], %l6, %o2   ! encoding: [0xd5,0xe6,0x11,0x16]
! LEON: error: instruction requires a CPU feature not currently enabled
casl [%i0], %l6, %o2

! V8: error: instruction requires a CPU feature not currently enabled
! V9: casx [%i0], %l6, %o2  ! encoding: [0xd5,0xf6,0x10,0x16]
! LEON: error: instruction requires a CPU feature not currently enabled
casx [%i0], %l6, %o2

! V8: error: instruction requires a CPU feature not currently enabled
! V9: casxl [%i0], %l6, %o2  ! encoding: [0xd5,0xf6,0x11,0x16]
! LEON: error: instruction requires a CPU feature not currently enabled
casxl [%i0], %l6, %o2

! V8: error: malformed ASI tag, must be a constant integer expression
! V9: casxa [%i0] %asi, %l6, %o2   ! encoding: [0xd5,0xf6,0x20,0x16]
! LEON: error: malformed ASI tag, must be a constant integer expression
casxa [%i0] %asi, %l6, %o2

! V8: error: instruction requires a CPU feature not currently enabled
! V9: casxa [%i0] 128, %l6, %o2   ! encoding: [0xd5,0xf6,0x10,0x16]
! LEON: error: instruction requires a CPU feature not currently enabled
casxa [%i0] 0x80, %l6, %o2

! V8: error: instruction requires a CPU feature not currently enabled
! V9: casxa [%i0] 128, %l6, %o2   ! encoding: [0xd5,0xf6,0x10,0x16]
! LEON: error: instruction requires a CPU feature not currently enabled
casxa [%i0] (0x40+0x40), %l6, %o2

! V8: error: malformed ASI tag, must be a constant integer expression
! V9: casa [%i0] %asi, %l6, %o2   ! encoding: [0xd5,0xe6,0x20,0x16]
! LEON: error: malformed ASI tag, must be a constant integer expression
casa [%i0] %asi, %l6, %o2

! V8: error: instruction requires a CPU feature not currently enabled
! V9: casa [%i0] 128, %l6, %o2   ! encoding: [0xd5,0xe6,0x10,0x16]
! LEON: casa [%i0] 128, %l6, %o2   ! encoding: [0xd5,0xe6,0x10,0x16]
casa [%i0] 0x80, %l6, %o2

! V8: error: instruction requires a CPU feature not currently enabled
! V9: casa [%i0] 128, %l6, %o2   ! encoding: [0xd5,0xe6,0x10,0x16]
! LEON: casa [%i0] 128, %l6, %o2   ! encoding: [0xd5,0xe6,0x10,0x16]
casa [%i0] (0x40+0x40), %l6, %o2
