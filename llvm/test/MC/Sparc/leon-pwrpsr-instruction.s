! RUN: llvm-mc %s -arch=sparc -mcpu=gr740 -show-encoding | FileCheck %s

    ! CHECK: pwr %g0, 0, %psr                ! encoding: [0x83,0x88,0x20,0x00]
    pwr 0, %psr

    ! CHECK: pwr %g0, %l7, %psr              ! encoding: [0x83,0x88,0x00,0x17]
    pwr %l7, %psr

    ! CHECK: pwr %g0, 32, %psr              ! encoding: [0x83,0x88,0x20,0x20]
    pwr 32, %psr
