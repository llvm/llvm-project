! RUN: llvm-mc %s -triple=sparc64-unknown-linux-gnu -show-encoding | FileCheck %s

        ! CHECK: sllx %g1, %i2, %i0  ! encoding: [0xb1,0x28,0x50,0x1a]
        sllx %g1, %i2, %i0

        ! CHECK: sllx %g1, 63, %i0   ! encoding: [0xb1,0x28,0x70,0x3f]
        sllx %g1, 63, %i0

        ! CHECK: srlx %g1, %i2, %i0  ! encoding: [0xb1,0x30,0x50,0x1a]
        srlx %g1, %i2, %i0

        ! CHECK: srlx %g1, 63, %i0   ! encoding: [0xb1,0x30,0x70,0x3f]
        srlx %g1, 63, %i0

        ! CHECK: srax %g1, %i2, %i0  ! encoding: [0xb1,0x38,0x50,0x1a]
        srax %g1, %i2, %i0

        ! CHECK: srax %g1, 63, %i0   ! encoding: [0xb1,0x38,0x70,0x3f]
        srax %g1, 63, %i0

        ! CHECK: mulx %g1, %i2, %i0  ! encoding: [0xb0,0x48,0x40,0x1a]
        mulx %g1, %i2, %i0

        ! CHECK: mulx %g1, 63, %i0   ! encoding: [0xb0,0x48,0x60,0x3f]
        mulx %g1, 63, %i0

        ! CHECK: sdivx %g1, %i2, %i0 ! encoding: [0xb1,0x68,0x40,0x1a]
        sdivx %g1, %i2, %i0

        ! CHECK: sdivx %g1, 63, %i0  ! encoding: [0xb1,0x68,0x60,0x3f]
        sdivx %g1, 63, %i0

        ! CHECK: udivx %g1, %i2, %i0 ! encoding: [0xb0,0x68,0x40,0x1a]
        udivx %g1, %i2, %i0

        ! CHECK: udivx %g1, 63, %i0  ! encoding: [0xb0,0x68,0x60,0x3f]
        udivx %g1, 63, %i0

