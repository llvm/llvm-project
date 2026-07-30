! RUN: llvm-mc %s -triple=sparc -show-encoding | FileCheck %s

        ! CHECK: fcmps %f0, %f4           ! encoding: [0x81,0xa8,0x0a,0x24]
        ! CHECK: fcmpd %f0, %f4           ! encoding: [0x81,0xa8,0x0a,0x44]
        ! CHECK: fcmpq %f0, %f4           ! encoding: [0x81,0xa8,0x0a,0x64]
        fcmps %f0, %f4
        fcmpd %f0, %f4
        fcmpq %f0, %f4

        ! CHECK: fcmpes %f0, %f4          ! encoding: [0x81,0xa8,0x0a,0xa4]
        ! CHECK: fcmped %f0, %f4          ! encoding: [0x81,0xa8,0x0a,0xc4]
        ! CHECK: fcmpeq %f0, %f4          ! encoding: [0x81,0xa8,0x0a,0xe4]
        fcmpes %f0, %f4
        fcmped %f0, %f4
        fcmpeq %f0, %f4
