! RUN: llvm-mc %s -triple=sparc   -show-encoding | FileCheck %s
! RUN: llvm-mc %s -triple=sparcv9 -show-encoding | FileCheck %s

! Section A.3 Synthetic Instructions
        ! CHECK: cmp %g1, %g2                     ! encoding: [0x80,0xa0,0x40,0x02]
        cmp %g1, %g2
        ! CHECK: cmp %g1, 5                       ! encoding: [0x80,0xa0,0x60,0x05]
        cmp %g1, 5

        ! jmp and call are tested in sparc-ctrl-instructions.

        ! CHECK: tst %g1                          ! encoding: [0x80,0x90,0x40,0x00]
        tst %g1

        ! CHECK: ret                              ! encoding: [0x81,0xc7,0xe0,0x08]
        ret
        ! CHECK: retl                             ! encoding: [0x81,0xc3,0xe0,0x08]
        retl

        ! CHECK: restore                          ! encoding: [0x81,0xe8,0x00,0x00]
        restore
        ! CHECK: save                             ! encoding: [0x81,0xe0,0x00,0x00]
        save

        ! CHECK: sethi %hi(40000), %g1            ! encoding: [0x03,0b00AAAAAA,A,A]
        ! CHECK:                                  !   fixup A - offset: 0, value: %hi(40000), kind: fixup_sparc_hi22
        ! CHECK: or %g1, %lo(40000), %g1          ! encoding: [0x82,0x10,0b011000AA,A]
        ! CHECK:                                  !   fixup A - offset: 0, value: %lo(40000), kind: fixup_sparc_lo10
        set 40000, %g1
        ! CHECK: mov 1, %g1 ! encoding: [0x82,0x10,0x20,0x01]
        set 1, %g1
        ! CHECK: sethi %hi(32768), %g1            ! encoding: [0x03,0b00AAAAAA,A,A]
        ! CHECK:                                  !   fixup A - offset: 0, value: %hi(32768), kind: fixup_sparc_hi22
        set 32768, %g1

        ! Expect a 'sethi' without an 'or'.
        ! CHECK: sethi %hi(268431360), %o1       ! encoding: [0x13,0b00AAAAAA,A,A]
        ! CHECK:                                 !   fixup A - offset: 0, value: %hi(268431360), kind: fixup_sparc_hi22
        set 0x0ffff000, %o1

        ! CHECK: sethi %hi(268433408), %o1       ! encoding: [0x13,0b00AAAAAA,A,A]
        ! CHECK:                                 !   fixup A - offset: 0, value: %hi(268433408), kind: fixup_sparc_hi22
        set 0x0ffff800, %o1

        ! This is the boundary case that uses the lowest of the 22 bits in sethi.
        ! CHECK: sethi %hi(268434432), %o1       ! encoding: [0x13,0b00AAAAAA,A,A]
        ! CHECK:                                 !   fixup A - offset: 0, value: %hi(268434432), kind: fixup_sparc_hi22
        set 0x0ffffc00, %o1

        ! Now the synthetic instruction becomes two instructions.
        ! CHECK: sethi %hi(2147483647), %o1      ! encoding: [0x13,0b00AAAAAA,A,A]
        ! CHECK:                                 !   fixup A - offset: 0, value: %hi(2147483647), kind: fixup_sparc_hi22
        ! CHECK: or %o1, %lo(2147483647), %o1    ! encoding: [0x92,0x12,0b011000AA,A]
        ! CHECK:                                 !   fixup A - offset: 0, value: %lo(2147483647), kind: fixup_sparc_lo10
        set 2147483647, %o1

        ! CHECK: xnor %g1, %g0, %g2               ! encoding: [0x84,0x38,0x40,0x00]
        not %g1, %g2
        ! CHECK: xnor %g1, %g0, %g1               ! encoding: [0x82,0x38,0x40,0x00]
        not %g1

        ! CHECK: sub %g0, %g1, %g2                ! encoding: [0x84,0x20,0x00,0x01]
        neg %g1, %g2
        ! CHECK: sub %g0, %g1, %g1                ! encoding: [0x82,0x20,0x00,0x01]
        neg %g1

        ! CHECK: add %g1, 1, %g1                  ! encoding: [0x82,0x00,0x60,0x01]
        inc %g1
        ! CHECK: add %g1, 55, %g1                 ! encoding: [0x82,0x00,0x60,0x37]
        inc 55, %g1
        ! CHECK: addcc %g1, 1, %g1                ! encoding: [0x82,0x80,0x60,0x01]
        inccc %g1
        ! CHECK: addcc %g1, 55, %g1               ! encoding: [0x82,0x80,0x60,0x37]
        inccc 55, %g1

        ! CHECK: sub %g1, 1, %g1                  ! encoding: [0x82,0x20,0x60,0x01]
        dec %g1
        ! CHECK: sub %g1, 55, %g1                 ! encoding: [0x82,0x20,0x60,0x37]
        dec 55, %g1
        ! CHECK: subcc %g1, 1, %g1                ! encoding: [0x82,0xa0,0x60,0x01]
        deccc %g1
        ! CHECK: subcc %g1, 55, %g1               ! encoding: [0x82,0xa0,0x60,0x37]
        deccc 55, %g1

        ! CHECK: andcc %g2, %g1, %g0              ! encoding: [0x80,0x88,0x80,0x01]
        btst %g1, %g2
        ! CHECK: andcc %g2, 4, %g0                ! encoding: [0x80,0x88,0xa0,0x04]
        btst 4, %g2
        ! CHECK: or %g2, %g1, %g2                 ! encoding: [0x84,0x10,0x80,0x01]
        bset %g1, %g2
        ! CHECK: or %g2, 4, %g2                   ! encoding: [0x84,0x10,0xa0,0x04]
        bset 4, %g2
        ! CHECK: andn %g2, %g1, %g2               ! encoding: [0x84,0x28,0x80,0x01]
        bclr %g1, %g2
        ! CHECK: andn %g2, 4, %g2                 ! encoding: [0x84,0x28,0xa0,0x04]
        bclr 4, %g2
        ! CHECK: xor %g2, %g1, %g2                ! encoding: [0x84,0x18,0x80,0x01]
        btog %g1, %g2
        ! CHECK: xor %g2, 4, %g2                  ! encoding: [0x84,0x18,0xa0,0x04]
        btog 4, %g2

        ! CHECK: mov %g0, %g1                     ! encoding: [0x82,0x10,0x00,0x00]
        clr %g1
        ! CHECK: stb %g0, [%g1+%g2]               ! encoding: [0xc0,0x28,0x40,0x02]
        clrb [%g1+%g2]
        ! CHECK: sth %g0, [%g1+%g2]               ! encoding: [0xc0,0x30,0x40,0x02]
        clrh [%g1+%g2]
        ! CHECK: st %g0, [%g1+%g2]                ! encoding: [0xc0,0x20,0x40,0x02]
        clr [%g1+%g2]

        ! mov reg_or_imm,reg tested in sparc-alu-instructions.s

        ! CHECK: rd %y, %i0                       ! encoding: [0xb1,0x40,0x00,0x00]
        mov %y, %i0
        ! CHECK: rd %asr1, %i0                    ! encoding: [0xb1,0x40,0x40,0x00]
        mov %asr1, %i0
        ! CHECK: rd %psr, %i0                     ! encoding: [0xb1,0x48,0x00,0x00]
        mov %psr, %i0
        ! CHECK: rd %wim, %i0                     ! encoding: [0xb1,0x50,0x00,0x00]
        mov %wim, %i0
        ! CHECK: rd %tbr, %i0                     ! encoding: [0xb1,0x58,0x00,0x00]
        mov %tbr, %i0

        ! CHECK: wr %g0, %i0, %y                  ! encoding: [0x81,0x80,0x00,0x18]
        mov %i0, %y
        ! CHECK: wr %g0, 5, %y                    ! encoding: [0x81,0x80,0x20,0x05]
        mov 5, %y
        ! CHECK: wr %g0, %i0, %asr15              ! encoding: [0x9f,0x80,0x00,0x18]
        mov %i0, %asr15
        ! CHECK: wr %g0, 5, %asr15                ! encoding: [0x9f,0x80,0x20,0x05]
        mov 5, %asr15
        ! CHECK: wr %g0, %i0, %psr                ! encoding: [0x81,0x88,0x00,0x18]
        mov %i0, %psr
        ! CHECK: wr %g0, 5, %psr                  ! encoding: [0x81,0x88,0x20,0x05]
        mov 5, %psr
        ! CHECK: wr %g0, %i0, %wim                ! encoding: [0x81,0x90,0x00,0x18]
        mov %i0, %wim
        ! CHECK: wr %g0, 5, %wim                  ! encoding: [0x81,0x90,0x20,0x05]
        mov 5, %wim
        ! CHECK: wr %g0, %i0, %tbr                ! encoding: [0x81,0x98,0x00,0x18]
        mov %i0, %tbr
        ! CHECK: wr %g0, 5, %tbr                  ! encoding: [0x81,0x98,0x20,0x05]
        mov 5, %tbr

! Other aliases
        ! CHECK: wr %g0, %i0, %y                  ! encoding: [0x81,0x80,0x00,0x18]
        wr %i0, %y
        ! CHECK: wr %g0, 5, %y                    ! encoding: [0x81,0x80,0x20,0x05]
        wr 5, %y
        ! CHECK: wr %g0, %i0, %asr15              ! encoding: [0x9f,0x80,0x00,0x18]
        wr %i0, %asr15
        ! CHECK: wr %g0, 5, %asr15                ! encoding: [0x9f,0x80,0x20,0x05]
        wr 5, %asr15
        ! CHECK: wr %g0, %i0, %psr                ! encoding: [0x81,0x88,0x00,0x18]
        wr %i0, %psr
        ! CHECK: wr %g0, 5, %psr                  ! encoding: [0x81,0x88,0x20,0x05]
        wr 5, %psr
        ! CHECK: wr %g0, %i0, %wim                ! encoding: [0x81,0x90,0x00,0x18]
        wr %i0, %wim
        ! CHECK: wr %g0, 5, %wim                  ! encoding: [0x81,0x90,0x20,0x05]
        wr 5, %wim
        ! CHECK: wr %g0, %i0, %tbr                ! encoding: [0x81,0x98,0x00,0x18]
        wr %i0, %tbr
        ! CHECK: wr %g0, 5, %tbr                  ! encoding: [0x81,0x98,0x20,0x05]
        wr 5, %tbr
        ! CHECK: or %g2, 4, %g2                   ! encoding: [0x84,0x10,0xa0,0x04]
        or 4, %g2, %g2

! The following tests exercise 'set' in such a way that its output differs
! depending on whether targeting V8 or V9.
!
! RUN: llvm-mc %s -triple=sparc   -show-encoding | FileCheck %s --check-prefix=V8
! RUN: llvm-mc %s -triple=sparcv9 -show-encoding | FileCheck %s --check-prefix=V9

        ! V8: mov        -1, %o1              ! encoding: [0x92,0x10,0x3f,0xff]
        ! V9: sethi %hi(-1), %o1              ! encoding: [0x13,0b00AAAAAA,A,A]
        ! V9:                                 !   fixup A - offset: 0, value: %hi(-1), kind: fixup_sparc_hi22
        ! V9: or %o1, %lo(-1), %o1            ! encoding: [0x92,0x12,0b011000AA,A]
        ! V9:                                 !   fixup A - offset: 0, value: %lo(-1), kind: fixup_sparc_lo10
        set 0xffffffff, %o1

        ! V8: mov        -2, %o1              ! encoding: [0x92,0x10,0x3f,0xfe]
        ! V9: sethi %hi(-2), %o1              ! encoding: [0x13,0b00AAAAAA,A,A]
        ! V9:                                 !   fixup A - offset: 0, value: %hi(-2), kind: fixup_sparc_hi22
        ! V9: or %o1, %lo(-2), %o1            ! encoding: [0x92,0x12,0b011000AA,A]
        ! V9:                                 !   fixup A - offset: 0, value: %lo(-2), kind: fixup_sparc_lo10
        set 0xfffffffe, %o1

        ! V8: mov        -16, %o1             ! encoding: [0x92,0x10,0x3f,0xf0]
        ! V9: sethi %hi(-16), %o1             ! encoding: [0x13,0b00AAAAAA,A,A]
        ! V9:                                 !   fixup A - offset: 0, value: %hi(-16), kind: fixup_sparc_hi22
        ! V9: or %o1, %lo(-16), %o1           ! encoding: [0x92,0x12,0b011000AA,A]
        ! V9:                                 !   fixup A - offset: 0, value: %lo(-16), kind: fixup_sparc_lo10
        set 0xfffffff0, %o1

        ! V8: mov        -256, %o1            ! encoding: [0x92,0x10,0x3f,0x00]
        ! V9: sethi %hi(-256), %o1            ! encoding: [0x13,0b00AAAAAA,A,A]
        ! V9:                                 !   fixup A - offset: 0, value: %hi(-256), kind: fixup_sparc_hi22
        ! V9: or %o1, %lo(-256), %o1          ! encoding: [0x92,0x12,0b011000AA,A]
        ! V9:                                 !   fixup A - offset: 0, value: %lo(-256), kind: fixup_sparc_lo10
        set 0xffffff00, %o1

        ! V8: mov        -4096, %o1           ! encoding: [0x92,0x10,0x30,0x00]
        ! V9: sethi %hi(-4096), %o1           ! encoding: [0x13,0b00AAAAAA,A,A]
        ! V9:                                 !   fixup A - offset: 0, value: %hi(-4096), kind: fixup_sparc_hi22
        set 0xfffff000, %o1

        ! These results are the same for V8 and V9, so this test could have
        ! been with the others that weren't segregated by architecture,
        ! but logically it belongs here as a boundary case.
        ! V8: sethi %hi(-8192), %o1           ! encoding: [0x13,0b00AAAAAA,A,A]
        ! V8:                                 !   fixup A - offset: 0, value: %hi(-8192), kind: fixup_sparc_hi22
        ! V9: sethi %hi(-8192), %o1           ! encoding: [0x13,0b00AAAAAA,A,A]
        ! V9:                                 !   fixup A - offset: 0, value: %hi(-8192), kind: fixup_sparc_hi22
        set 0xffffe000, %o1
