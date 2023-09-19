! RUN: llvm-mc %s -triple=sparc   -show-encoding | FileCheck %s
! RUN: llvm-mc %s -triple=sparcv9 -show-encoding | FileCheck %s

        ! CHECK: add %g0, %g0, %g0    ! encoding: [0x80,0x00,0x00,0x00]
        add %g0, %g0, %g0
        ! CHECK: add %g1, %g2, %g3    ! encoding: [0x86,0x00,0x40,0x02]
        add %g1, %g2, %g3
        ! CHECK: add %o0, %o1, %l0    ! encoding: [0xa0,0x02,0x00,0x09]
        add %r8, %r9, %l0
        ! CHECK: add %o0, 10,  %l0    ! encoding: [0xa0,0x02,0x20,0x0a]
        add %o0, 10, %l0

        ! CHECK: addcc %g1, %g2, %g3  ! encoding: [0x86,0x80,0x40,0x02]
        addcc %g1, %g2, %g3

        ! CHECK: addxcc %g1, %g2, %g3 ! encoding: [0x86,0xc0,0x40,0x02]
        addxcc %g1, %g2, %g3

        ! CHECK: udiv %g1, %g2, %g3   ! encoding: [0x86,0x70,0x40,0x02]
        udiv %g1, %g2, %g3

        ! CHECK: sdiv %g1, %g2, %g3   ! encoding: [0x86,0x78,0x40,0x02]
        sdiv %g1, %g2, %g3

        ! CHECK: and %g1, %g2, %g3    ! encoding: [0x86,0x08,0x40,0x02]
        and %g1, %g2, %g3
        ! CHECK: andn %g1, %g2, %g3   ! encoding: [0x86,0x28,0x40,0x02]
        andn %g1, %g2, %g3
        ! CHECK: or %g1, %g2, %g3     ! encoding: [0x86,0x10,0x40,0x02]
        or  %g1, %g2, %g3
        ! CHECK: orn %g1, %g2, %g3    ! encoding: [0x86,0x30,0x40,0x02]
        orn %g1, %g2, %g3
        ! CHECK: xor %g1, %g2, %g3    ! encoding: [0x86,0x18,0x40,0x02]
        xor %g1, %g2, %g3
        ! CHECK: xnor %g1, %g2, %g3   ! encoding: [0x86,0x38,0x40,0x02]
        xnor %g1, %g2, %g3

        ! CHECK: umul %g1, %g2, %g3   ! encoding: [0x86,0x50,0x40,0x02]
        umul %g1, %g2, %g3

        ! CHECK: smul %g1, %g2, %g3   ! encoding: [0x86,0x58,0x40,0x02]
        smul %g1, %g2, %g3

        ! CHECK: nop                  ! encoding: [0x01,0x00,0x00,0x00]
        nop

        ! CHECK: sethi 10, %l0        ! encoding: [0x21,0x00,0x00,0x0a]
        sethi 10, %l0

        ! CHECK: sll %g1, %g2, %g3    ! encoding: [0x87,0x28,0x40,0x02]
        sll %g1, %g2, %g3
        ! CHECK: sll %g1, 31, %g3     ! encoding: [0x87,0x28,0x60,0x1f]
        sll %g1, 31, %g3

        ! CHECK: srl %g1, %g2, %g3    ! encoding: [0x87,0x30,0x40,0x02]
        srl %g1, %g2, %g3
        ! CHECK: srl %g1, 31, %g3     ! encoding: [0x87,0x30,0x60,0x1f]
        srl %g1, 31, %g3

        ! CHECK: sra %g1, %g2, %g3    ! encoding: [0x87,0x38,0x40,0x02]
        sra %g1, %g2, %g3
        ! CHECK: sra %g1, 31, %g3     ! encoding: [0x87,0x38,0x60,0x1f]
        sra %g1, 31, %g3

        ! CHECK: sub %g1, %g2, %g3    ! encoding: [0x86,0x20,0x40,0x02]
        sub %g1, %g2, %g3
        ! CHECK: subcc %g1, %g2, %g3  ! encoding: [0x86,0xa0,0x40,0x02]
        subcc %g1, %g2, %g3

        ! CHECK: subxcc %g1, %g2, %g3 ! encoding: [0x86,0xe0,0x40,0x02]
        subxcc %g1, %g2, %g3

        ! CHECK: mov %g1, %g3     ! encoding: [0x86,0x10,0x00,0x01]
        mov %g1, %g3

        ! CHECK: mov 255, %g3     ! encoding: [0x86,0x10,0x20,0xff]
        mov 0xff, %g3

        ! CHECK: restore %g0, %g0, %g1           ! encoding: [0x83,0xe8,0x00,0x00]
        restore %g0, %g0, %g1

        ! CHECK: addx %g2, %g1, %g3              ! encoding: [0x86,0x40,0x80,0x01]
        addx %g2, %g1, %g3

        ! CHECK:  subx %g2, %g1, %g3              ! encoding: [0x86,0x60,0x80,0x01]
        subx %g2, %g1, %g3

        ! CHECK:  umulcc %g2, %g1, %g3            ! encoding: [0x86,0xd0,0x80,0x01]
        umulcc %g2, %g1, %g3

        ! CHECK:  smulcc %g2, %g1, %g3            ! encoding: [0x86,0xd8,0x80,0x01]
        smulcc %g2, %g1, %g3

        ! CHECK:  mulscc %g2, %g1, %g3            ! encoding: [0x87,0x20,0x80,0x01]
        mulscc %g2, %g1, %g3

        ! CHECK:  mulscc %g2, 254, %g3            ! encoding: [0x87,0x20,0xa0,0xfe]
        mulscc %g2, 254, %g3

        ! CHECK:  udivcc %g2, %g1, %g3            ! encoding: [0x86,0xf0,0x80,0x01]
        udivcc %g2, %g1, %g3

        ! CHECK:  sdivcc %g2, %g1, %g3            ! encoding: [0x86,0xf8,0x80,0x01]
        sdivcc %g2, %g1, %g3

        ! CHECK:  andcc %g2, %g1, %g3             ! encoding: [0x86,0x88,0x80,0x01]
        andcc %g2, %g1, %g3

        ! CHECK:  andncc %g2, %g1, %g3            ! encoding: [0x86,0xa8,0x80,0x01]
        andncc %g2, %g1, %g3

        ! CHECK:  orcc %g2, %g1, %g3              ! encoding: [0x86,0x90,0x80,0x01]
        orcc %g2, %g1, %g3

        ! CHECK:  orncc %g2, %g1, %g3             ! encoding: [0x86,0xb0,0x80,0x01]
        orncc %g2, %g1, %g3

        ! CHECK:  xorcc %g2, %g1, %g3             ! encoding: [0x86,0x98,0x80,0x01]
        xorcc %g2, %g1, %g3

        ! CHECK:  xnorcc %g2, %g1, %g3            ! encoding: [0x86,0xb8,0x80,0x01]
        xnorcc %g2, %g1, %g3

        ! CHECK:  taddcc %g2, %g1, %g3            ! encoding: [0x87,0x00,0x80,0x01]
        taddcc %g2, %g1, %g3

        ! CHECK:  tsubcc %g2, %g1, %g3            ! encoding: [0x87,0x08,0x80,0x01]
        tsubcc %g2, %g1, %g3

        ! CHECK:  taddcctv %g2, %g1, %g3          ! encoding: [0x87,0x10,0x80,0x01]
        taddcctv %g2, %g1, %g3

        ! CHECK:  tsubcctv %g2, %g1, %g3          ! encoding: [0x87,0x18,0x80,0x01]
        tsubcctv %g2, %g1, %g3
