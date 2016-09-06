! RUN: llvm-mc %s -arch=sparc   -show-encoding | FileCheck %s
! RUN: llvm-mc %s -arch=sparcv9 -show-encoding | FileCheck %s

        ! CHECK: call foo     ! encoding: [0b01AAAAAA,A,A,A]
        ! CHECK:              !   fixup A - offset: 0, value: foo, kind: fixup_sparc_call30
        call foo

        ! CHECK: call %g1+%i2 ! encoding: [0x9f,0xc0,0x40,0x1a]
        call %g1 + %i2

        ! CHECK: call %o1+8   ! encoding: [0x9f,0xc2,0x60,0x08]
        call %o1 + 8

        ! CHECK: call %g1     ! encoding: [0x9f,0xc0,0x40,0x00]
        call %g1

        ! CHECK: call %g1+%lo(sym)   ! encoding: [0x9f,0xc0,0b011000AA,A]
        ! CHECK-NEXT:                ! fixup A - offset: 0, value: %lo(sym), kind: fixup_sparc_lo10
        call %g1+%lo(sym)

        ! CHECK: jmp %g1+%i2  ! encoding: [0x81,0xc0,0x40,0x1a]
        jmp %g1 + %i2

        ! CHECK: jmp %o1+8    ! encoding: [0x81,0xc2,0x60,0x08]
        jmp %o1 + 8

        ! CHECK: jmp %g1      ! encoding: [0x81,0xc0,0x40,0x00]
        jmp %g1

        ! CHECK: jmp %g1+%lo(sym)   ! encoding: [0x81,0xc0,0b011000AA,A]
        ! CHECK-NEXT:                ! fixup A - offset: 0, value: %lo(sym), kind: fixup_sparc_lo10
        jmp %g1+%lo(sym)

        ! CHECK: jmpl %g1+%i2, %g2  ! encoding: [0x85,0xc0,0x40,0x1a]
        jmpl %g1 + %i2, %g2

        ! CHECK: jmpl %o1+8, %g2    ! encoding: [0x85,0xc2,0x60,0x08]
        jmpl %o1 + 8, %g2

        ! CHECK: jmpl %g1, %g2      ! encoding: [0x85,0xc0,0x40,0x00]
        jmpl %g1, %g2

        ! CHECK: jmpl %g1+%lo(sym), %g2   ! encoding: [0x85,0xc0,0b011000AA,A]
        ! CHECK-NEXT:                     ! fixup A - offset: 0, value: %lo(sym), kind: fixup_sparc_lo10
        jmpl %g1+%lo(sym), %g2

        ! CHECK: ba .BB0      ! encoding: [0x10,0b10AAAAAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        ba .BB0

        ! CHECK: bne .BB0     ! encoding: [0x12,0b10AAAAAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        bne .BB0

        ! CHECK: bne .BB0     ! encoding: [0x12,0b10AAAAAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        bnz .BB0

        ! CHECK: be .BB0      ! encoding: [0x02,0b10AAAAAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        be .BB0

        ! CHECK: be .BB0      ! encoding: [0x02,0b10AAAAAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        bz .BB0

        ! CHECK: be .BB0      ! encoding: [0x02,0b10AAAAAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        beq .BB0

        ! CHECK: bg .BB0      ! encoding: [0x14,0b10AAAAAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        bg .BB0

        ! CHECK: ble .BB0      ! encoding: [0x04,0b10AAAAAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        ble .BB0

        ! CHECK: bge .BB0      ! encoding: [0x16,0b10AAAAAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        bge .BB0

        ! CHECK: bl .BB0      ! encoding: [0x06,0b10AAAAAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        bl .BB0

        ! CHECK: bgu .BB0      ! encoding: [0x18,0b10AAAAAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        bgu .BB0

        ! CHECK: bleu .BB0      ! encoding: [0x08,0b10AAAAAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        bleu .BB0

        ! CHECK: bcc .BB0      ! encoding: [0x1a,0b10AAAAAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        bcc .BB0

        ! CHECK: bcc .BB0      ! encoding: [0x1a,0b10AAAAAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        bgeu .BB0

        ! CHECK: bcs .BB0      ! encoding: [0x0a,0b10AAAAAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        bcs .BB0

        ! CHECK: bcs .BB0      ! encoding: [0x0a,0b10AAAAAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        blu .BB0

        ! CHECK: bpos .BB0      ! encoding: [0x1c,0b10AAAAAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        bpos .BB0

        ! CHECK: bneg .BB0      ! encoding: [0x0c,0b10AAAAAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        bneg .BB0

        ! CHECK: bvc .BB0      ! encoding: [0x1e,0b10AAAAAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        bvc .BB0

        ! CHECK: bvs .BB0      ! encoding: [0x0e,0b10AAAAAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        bvs .BB0

        ! CHECK:             fba .BB0                        ! encoding: [0x11,0b10AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        fba .BB0

        ! CHECK:             fba .BB0                        ! encoding: [0x11,0b10AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        fb .BB0

        ! CHECK:             fbn .BB0                        ! encoding: [0x01,0b10AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        fbn .BB0

        ! CHECK:             fbu .BB0                        ! encoding: [0x0f,0b10AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        fbu .BB0

        ! CHECK:             fbg .BB0                        ! encoding: [0x0d,0b10AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        fbg .BB0

        ! CHECK:             fbug .BB0                       ! encoding: [0x0b,0b10AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        fbug .BB0

        ! CHECK:             fbl .BB0                        ! encoding: [0x09,0b10AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        fbl .BB0

        ! CHECK:             fbul .BB0                       ! encoding: [0x07,0b10AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        fbul .BB0

        ! CHECK:             fblg .BB0                       ! encoding: [0x05,0b10AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        fblg .BB0

        ! CHECK:             fbne .BB0                       ! encoding: [0x03,0b10AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        fbne .BB0

        ! CHECK:             fbne .BB0                       ! encoding: [0x03,0b10AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        fbnz .BB0

        ! CHECK:             fbe .BB0                        ! encoding: [0x13,0b10AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        fbe .BB0

        ! CHECK:             fbe .BB0                        ! encoding: [0x13,0b10AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        fbz .BB0

        ! CHECK:             fbue .BB0                       ! encoding: [0x15,0b10AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        fbue .BB0

        ! CHECK:             fbge .BB0                       ! encoding: [0x17,0b10AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        fbge .BB0

        ! CHECK:             fbuge .BB0                      ! encoding: [0x19,0b10AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        fbuge .BB0

        ! CHECK:             fble .BB0                       ! encoding: [0x1b,0b10AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        fble .BB0

        ! CHECK:             fbule .BB0                      ! encoding: [0x1d,0b10AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        fbule .BB0

        ! CHECK:             fbo .BB0                        ! encoding: [0x1f,0b10AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        fbo .BB0
        
        ! CHECK:             cba .BB0                        ! encoding: [0x11,0b11AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        cb .BB0

        ! CHECK:             cba .BB0                        ! encoding: [0x11,0b11AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        cba .BB0

        ! CHECK:             cbn .BB0                        ! encoding: [0x01,0b11AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        cbn .BB0

        ! CHECK:             cb3 .BB0                        ! encoding: [0x0f,0b11AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        cb3 .BB0

        ! CHECK:             cb2 .BB0                        ! encoding: [0x0d,0b11AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        cb2 .BB0

        ! CHECK:             cb23 .BB0                       ! encoding: [0x0b,0b11AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        cb23 .BB0

        ! CHECK:             cb1 .BB0                        ! encoding: [0x09,0b11AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        cb1 .BB0

        ! CHECK:             cb13 .BB0                       ! encoding: [0x07,0b11AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        cb13 .BB0

        ! CHECK:             cb12 .BB0                       ! encoding: [0x05,0b11AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        cb12 .BB0

        ! CHECK:             cb123 .BB0                      ! encoding: [0x03,0b11AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        cb123 .BB0

        ! CHECK:             cb0 .BB0                        ! encoding: [0x13,0b11AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        cb0 .BB0

        ! CHECK:             cb03 .BB0                       ! encoding: [0x15,0b11AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        cb03 .BB0

        ! CHECK:             cb02 .BB0                       ! encoding: [0x17,0b11AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        cb02 .BB0

        ! CHECK:             cb023 .BB0                      ! encoding: [0x19,0b11AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        cb023 .BB0

        ! CHECK:             cb01 .BB0                       ! encoding: [0x1b,0b11AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        cb01 .BB0

        ! CHECK:             cb013 .BB0                      ! encoding: [0x1d,0b11AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        cb013 .BB0

        ! CHECK:             cb012 .BB0                      ! encoding: [0x1f,0b11AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        cb012 .BB0

        ! CHECK: ba,a .BB0    ! encoding: [0x30,0b10AAAAAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        ba,a .BB0

        ! CHECK: bne,a .BB0   ! encoding: [0x32,0b10AAAAAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        bne,a .BB0

        ! CHECK: be,a .BB0    ! encoding: [0x22,0b10AAAAAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        be,a .BB0

        ! CHECK: bg,a .BB0    ! encoding: [0x34,0b10AAAAAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        bg,a .BB0

        ! CHECK: ble,a .BB0   ! encoding: [0x24,0b10AAAAAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        ble,a .BB0

        ! CHECK: bge,a .BB0   ! encoding: [0x36,0b10AAAAAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        bge,a .BB0

        ! CHECK: bl,a .BB0    ! encoding: [0x26,0b10AAAAAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        bl,a .BB0

        ! CHECK: bgu,a .BB0   ! encoding: [0x38,0b10AAAAAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        bgu,a .BB0

        ! CHECK: bleu,a .BB0  ! encoding: [0x28,0b10AAAAAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        bleu,a .BB0

        ! CHECK: bcc,a .BB0   ! encoding: [0x3a,0b10AAAAAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        bcc,a .BB0

        ! CHECK: bcs,a .BB0   ! encoding: [0x2a,0b10AAAAAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        bcs,a .BB0

        ! CHECK: bpos,a .BB0  ! encoding: [0x3c,0b10AAAAAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        bpos,a .BB0

        ! CHECK: bneg,a .BB0  ! encoding: [0x2c,0b10AAAAAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        bneg,a .BB0

        ! CHECK: bvc,a .BB0   ! encoding: [0x3e,0b10AAAAAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        bvc,a .BB0

        ! CHECK: bvs,a .BB0   ! encoding: [0x2e,0b10AAAAAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        bvs,a .BB0

        ! CHECK:             fbu,a .BB0                      ! encoding: [0x2f,0b10AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        fbu,a .BB0

        ! CHECK:             fbg,a .BB0                      ! encoding: [0x2d,0b10AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        fbg,a .BB0
        ! CHECK:             fbug,a .BB0                     ! encoding: [0x2b,0b10AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        fbug,a .BB0

        ! CHECK:             fbl,a .BB0                      ! encoding: [0x29,0b10AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        fbl,a .BB0

        ! CHECK:             fbul,a .BB0                     ! encoding: [0x27,0b10AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        fbul,a .BB0

        ! CHECK:             fblg,a .BB0                     ! encoding: [0x25,0b10AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        fblg,a .BB0

        ! CHECK:             fbne,a .BB0                     ! encoding: [0x23,0b10AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        fbne,a .BB0

        ! CHECK:             fbe,a .BB0                      ! encoding: [0x33,0b10AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        fbe,a .BB0

        ! CHECK:             fbue,a .BB0                     ! encoding: [0x35,0b10AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        fbue,a .BB0

        ! CHECK:             fbge,a .BB0                     ! encoding: [0x37,0b10AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        fbge,a .BB0

        ! CHECK:             fbuge,a .BB0                    ! encoding: [0x39,0b10AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        fbuge,a .BB0

        ! CHECK:             fble,a .BB0                     ! encoding: [0x3b,0b10AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        fble,a .BB0

        ! CHECK:             fbule,a .BB0                    ! encoding: [0x3d,0b10AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        fbule,a .BB0

        ! CHECK:             fbo,a .BB0                      ! encoding: [0x3f,0b10AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        fbo,a .BB0

        ! CHECK:  rett %i7+8   ! encoding: [0x81,0xcf,0xe0,0x08]
        rett %i7 + 8

        ! CHECK:             cb3,a .BB0                      ! encoding: [0x2f,0b11AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        cb3,a .BB0

        ! CHECK:             cb2,a .BB0                      ! encoding: [0x2d,0b11AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        cb2,a .BB0

        ! CHECK:             cb23,a .BB0                     ! encoding: [0x2b,0b11AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        cb23,a .BB0

        ! CHECK:             cb1,a .BB0                      ! encoding: [0x29,0b11AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        cb1,a .BB0

        ! CHECK:             cb13,a .BB0                     ! encoding: [0x27,0b11AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        cb13,a .BB0

        ! CHECK:             cb12,a .BB0                     ! encoding: [0x25,0b11AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        cb12,a .BB0

        ! CHECK:             cb123,a .BB0                    ! encoding: [0x23,0b11AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        cb123,a .BB0

        ! CHECK:             cb0,a .BB0                      ! encoding: [0x33,0b11AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        cb0,a .BB0

        ! CHECK:             cb03,a .BB0                     ! encoding: [0x35,0b11AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        cb03,a .BB0

        ! CHECK:             cb02,a .BB0                     ! encoding: [0x37,0b11AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        cb02,a .BB0

        ! CHECK:             cb023,a .BB0                    ! encoding: [0x39,0b11AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        cb023,a .BB0

        ! CHECK:             cb01,a .BB0                     ! encoding: [0x3b,0b11AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        cb01,a .BB0

        ! CHECK:             cb013,a .BB0                    ! encoding: [0x3d,0b11AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        cb013,a .BB0

        ! CHECK:             cb012,a .BB0                    ! encoding: [0x3f,0b11AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        cb012,a .BB0

        ! CHECK:             cb3,a .BB0                      ! encoding: [0x2f,0b11AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        cb3,a .BB0

        ! CHECK:             cb2,a .BB0                      ! encoding: [0x2d,0b11AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        cb2,a .BB0

        ! CHECK:             cb23,a .BB0                     ! encoding: [0x2b,0b11AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        cb23,a .BB0

        ! CHECK:             cb1,a .BB0                      ! encoding: [0x29,0b11AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        cb1,a .BB0

        ! CHECK:             cb13,a .BB0                     ! encoding: [0x27,0b11AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        cb13,a .BB0

        ! CHECK:             cb12,a .BB0                     ! encoding: [0x25,0b11AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        cb12,a .BB0

        ! CHECK:             cb123,a .BB0                    ! encoding: [0x23,0b11AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        cb123,a .BB0

        ! CHECK:             cb0,a .BB0                      ! encoding: [0x33,0b11AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        cb0,a .BB0

        ! CHECK:             cb03,a .BB0                     ! encoding: [0x35,0b11AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        cb03,a .BB0

        ! CHECK:             cb02,a .BB0                     ! encoding: [0x37,0b11AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        cb02,a .BB0

        ! CHECK:             cb023,a .BB0                    ! encoding: [0x39,0b11AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        cb023,a .BB0

        ! CHECK:             cb01,a .BB0                     ! encoding: [0x3b,0b11AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        cb01,a .BB0

        ! CHECK:             cb013,a .BB0                    ! encoding: [0x3d,0b11AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        cb013,a .BB0

        ! CHECK:             cb012,a .BB0                    ! encoding: [0x3f,0b11AAAAAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br22
        cb012,a .BB0

        ! CHECK:  rett %i7+8                                 ! encoding: [0x81,0xcf,0xe0,0x08]
        rett %i7 + 8

