! RUN: llvm-mc %s -arch=sparcv9 -show-encoding | FileCheck %s
! RUN: llvm-mc %s -arch=sparc -show-encoding | FileCheck %s

        ! CHECK: stbar                 ! encoding: [0x81,0x43,0xc0,0x00]
        stbar

        ! CHECK: swap [%i0+%l6], %o2   ! encoding: [0xd4,0x7e,0x00,0x16]
        swap [%i0+%l6], %o2

        ! CHECK: swap [%i0+32], %o2    ! encoding: [0xd4,0x7e,0x20,0x20]
        swap [%i0+32], %o2

        ! CHECK: swapa [%i0+%l6] 131, %o2   ! encoding: [0xd4,0xfe,0x10,0x76]
        swapa [%i0+%l6] 131, %o2

        ! CHECK: swapa [%i0+%l6] 131, %o2   ! encoding: [0xd4,0xfe,0x10,0x76]
        swapa [%i0+%l6] (130+1), %o2

        ! CHECK: ldstub [%i0+40], %g1 ! encoding: [0xc2,0x6e,0x20,0x28]
        ldstub [%i0+40], %g1

        ! CHECK: ldstub [%i0+%i2], %g1 ! encoding: [0xc2,0x6e,0x00,0x1a]
        ldstub [%i0+%i2], %g1

        ! CHECK: ldstuba [%i0+%i2] 131, %g1 ! encoding: [0xc2,0xee,0x10,0x7a]
        ldstuba [%i0+%i2] 131, %g1

        ! CHECK: ldstuba [%i0+%i2] 131, %g1 ! encoding: [0xc2,0xee,0x10,0x7a]
        ldstuba [%i0+%i2] (130+1), %g1
