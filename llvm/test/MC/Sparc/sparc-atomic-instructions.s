! RUN: llvm-mc %s -arch=sparcv9 -show-encoding | FileCheck %s --check-prefix=V9
! RUN: llvm-mc %s -arch=sparc   -show-encoding | FileCheck %s --check-prefix=V8

        ! V8: stbar                 ! encoding: [0x81,0x43,0xc0,0x00]
        ! V9: stbar                 ! encoding: [0x81,0x43,0xc0,0x00]
        stbar

        ! V8: swap [%i0+%l6], %o2   ! encoding: [0xd4,0x7e,0x00,0x16]
        ! V9: swap [%i0+%l6], %o2   ! encoding: [0xd4,0x7e,0x00,0x16]
        swap [%i0+%l6], %o2

        ! V8: swap [%i0+32], %o2    ! encoding: [0xd4,0x7e,0x20,0x20]
        ! V9: swap [%i0+32], %o2    ! encoding: [0xd4,0x7e,0x20,0x20]
        swap [%i0+32], %o2

        ! V8: swapa [%i0+%l6] 131, %o2   ! encoding: [0xd4,0xfe,0x10,0x76]
        ! V9: swapa [%i0+%l6] #ASI_SNF, %o2   ! encoding: [0xd4,0xfe,0x10,0x76]
        swapa [%i0+%l6] 131, %o2

        ! V8: swapa [%i0+%l6] 131, %o2   ! encoding: [0xd4,0xfe,0x10,0x76]
        ! V9: swapa [%i0+%l6] #ASI_SNF, %o2   ! encoding: [0xd4,0xfe,0x10,0x76]
        swapa [%i0+%l6] (130+1), %o2

        ! V8: ldstub [%i0+40], %g1 ! encoding: [0xc2,0x6e,0x20,0x28]
        ! V9: ldstub [%i0+40], %g1 ! encoding: [0xc2,0x6e,0x20,0x28]
        ldstub [%i0+40], %g1

        ! V8: ldstub [%i0+%i2], %g1 ! encoding: [0xc2,0x6e,0x00,0x1a]
        ! V9: ldstub [%i0+%i2], %g1 ! encoding: [0xc2,0x6e,0x00,0x1a]
        ldstub [%i0+%i2], %g1

        ! V8: ldstuba [%i0+%i2] 131, %g1 ! encoding: [0xc2,0xee,0x10,0x7a]
        ! V9: ldstuba [%i0+%i2] #ASI_SNF, %g1 ! encoding: [0xc2,0xee,0x10,0x7a]
        ldstuba [%i0+%i2] 131, %g1

        ! V8: ldstuba [%i0+%i2] 131, %g1 ! encoding: [0xc2,0xee,0x10,0x7a]
        ! V9: ldstuba [%i0+%i2] #ASI_SNF, %g1 ! encoding: [0xc2,0xee,0x10,0x7a]
        ldstuba [%i0+%i2] (130+1), %g1
