! RUN: llvm-mc %s -triple=sparc -mcpu=leon3 -show-encoding | FileCheck %s
! RUN: llvm-mc %s -triple=sparc -mcpu=ut699 -show-encoding | FileCheck %s
! RUN: llvm-mc %s -triple=sparc -mcpu=gr712rc -show-encoding | FileCheck %s
! RUN: llvm-mc %s -triple=sparc -mcpu=leon4 -show-encoding | FileCheck %s
! RUN: llvm-mc %s -triple=sparc -mcpu=gr740 -show-encoding | FileCheck %s

        ! CHECK: umac %i0, %l6, %o2    ! encoding: [0x95,0xf6,0x00,0x16]
        umac %i0, %l6, %o2

        ! CHECK: smac %i0, %l6, %o2    ! encoding: [0x95,0xfe,0x00,0x16]
        smac %i0, %l6, %o2
