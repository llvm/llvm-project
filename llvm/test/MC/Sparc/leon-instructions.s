! RUN: llvm-mc %s -arch=sparc -mcpu=leon3 -show-encoding | FileCheck %s -check-prefix=CHECK_NO_CASA
! RUN: llvm-mc %s -arch=sparc -mcpu=ut699 -show-encoding | FileCheck %s -check-prefix=CHECK_NO_CASA
! RUN: llvm-mc %s -arch=sparc -mcpu=gr712rc -show-encoding | FileCheck %s
! RUN: llvm-mc %s -arch=sparc -mcpu=leon4 -show-encoding | FileCheck %s
! RUN: llvm-mc %s -arch=sparc -mcpu=gr740 -show-encoding | FileCheck %s


        ! CHECK: casa [%i0] 10, %l6, %o2    ! encoding: [0xd5,0xe6,0x01,0x56]
        casa [%i0] 10, %l6, %o2

        ! CHECK: casa [%i0] 5, %l6, %o2    ! encoding: [0xd5,0xe6,0x00,0xb6]
        casa [%i0] 5, %l6, %o2

        ! CHECK: umac %i0, %l6, %o2    ! encoding: [0x95,0xf6,0x00,0x16]
        ! CHECK_NO_CASA: umac %i0, %l6, %o2    ! encoding: [0x95,0xf6,0x00,0x16]
        umac %i0, %l6, %o2

        ! CHECK: smac %i0, %l6, %o2    ! encoding: [0x95,0xfe,0x00,0x16]
        ! CHECK_NO_CASA: smac %i0, %l6, %o2    ! encoding: [0x95,0xfe,0x00,0x16]
        smac %i0, %l6, %o2
