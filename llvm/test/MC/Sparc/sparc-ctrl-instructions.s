! RUN: llvm-mc %s -triple=sparc   -show-encoding | FileCheck %s
! RUN: llvm-mc %s -triple=sparcv9 -show-encoding | FileCheck %s

        ! CHECK: call foo     ! encoding: [0b01AAAAAA,A,A,A]
        call foo

        ! CHECK: call foo, 0  ! encoding: [0b01AAAAAA,A,A,A]
        call foo, 0

        ! CHECK: call %g1+%i2 ! encoding: [0x9f,0xc0,0x40,0x1a]
        call %g1 + %i2

        ! CHECK: call %g1+%i2, 1 ! encoding: [0x9f,0xc0,0x40,0x1a]
        call %g1 + %i2, 1

        ! CHECK: call %o1+8   ! encoding: [0x9f,0xc2,0x60,0x08]
        call %o1 + 8

        ! CHECK: call %o1+8, 2   ! encoding: [0x9f,0xc2,0x60,0x08]
        call %o1 + 8, 2

        ! CHECK: call %g1     ! encoding: [0x9f,0xc0,0x40,0x00]
        call %g1

        ! CHECK: call %g1, 3     ! encoding: [0x9f,0xc0,0x40,0x00]
        call %g1, 3

        ! CHECK: call %g1+%lo(sym)   ! encoding: [0x9f,0xc0,0b011000AA,A]
        call %g1+%lo(sym)

        ! CHECK: call %g1+%lo(sym), 4   ! encoding: [0x9f,0xc0,0b011000AA,A]
        call %g1+%lo(sym), 4

        ! CHECK-LABEL: .Ltmp0:
        ! CHECK: call .Ltmp0-4 ! encoding: [0b01AAAAAA,A,A,A]
        call . - 4

        ! CHECK-LABEL: .Ltmp1:
        ! CHECK: call .Ltmp1-4, 5 ! encoding: [0b01AAAAAA,A,A,A]
        call . - 4, 5

        ! CHECK: jmp %g1+%i2  ! encoding: [0x81,0xc0,0x40,0x1a]
        jmp %g1 + %i2

        ! CHECK: jmp %o1+8    ! encoding: [0x81,0xc2,0x60,0x08]
        jmp %o1 + 8

        ! CHECK: jmp %g1      ! encoding: [0x81,0xc0,0x40,0x00]
        jmp %g1

        ! CHECK: jmp %g1+%lo(sym)   ! encoding: [0x81,0xc0,0b011000AA,A]
        jmp %g1+%lo(sym)

        ! CHECK: jmp sym ! encoding: [0x81,0xc0,0b001AAAAA,A]
        jmp sym

        ! CHECK: jmpl %g1+%i2, %g2  ! encoding: [0x85,0xc0,0x40,0x1a]
        jmpl %g1 + %i2, %g2

        ! CHECK: jmpl %o1+8, %g2    ! encoding: [0x85,0xc2,0x60,0x08]
        jmpl %o1 + 8, %g2

        ! CHECK: jmpl %g1, %g2      ! encoding: [0x85,0xc0,0x40,0x00]
        jmpl %g1, %g2

        ! CHECK: jmpl %g1+%lo(sym), %g2   ! encoding: [0x85,0xc0,0b011000AA,A]
        jmpl %g1+%lo(sym), %g2

        ! CHECK: ba .BB0      ! encoding: [0x10,0b10AAAAAA,A,A]
        ba .BB0

        ! CHECK: bne .BB0     ! encoding: [0x12,0b10AAAAAA,A,A]
        bne .BB0

        ! CHECK: bne .BB0     ! encoding: [0x12,0b10AAAAAA,A,A]
        bnz .BB0

        ! CHECK: be .BB0      ! encoding: [0x02,0b10AAAAAA,A,A]
        be .BB0

        ! CHECK: be .BB0      ! encoding: [0x02,0b10AAAAAA,A,A]
        bz .BB0

        ! CHECK: be .BB0      ! encoding: [0x02,0b10AAAAAA,A,A]
        beq .BB0

        ! CHECK: bg .BB0      ! encoding: [0x14,0b10AAAAAA,A,A]
        bg .BB0

        ! CHECK: bg .BB0      ! encoding: [0x14,0b10AAAAAA,A,A]
        bgt .BB0

        ! CHECK: ble .BB0      ! encoding: [0x04,0b10AAAAAA,A,A]
        ble .BB0

        ! CHECK: bge .BB0      ! encoding: [0x16,0b10AAAAAA,A,A]
        bge .BB0

        ! CHECK: bl .BB0      ! encoding: [0x06,0b10AAAAAA,A,A]
        bl .BB0

        ! CHECK: bl .BB0      ! encoding: [0x06,0b10AAAAAA,A,A]
        blt .BB0

        ! CHECK: bgu .BB0      ! encoding: [0x18,0b10AAAAAA,A,A]
        bgu .BB0

        ! CHECK: bleu .BB0      ! encoding: [0x08,0b10AAAAAA,A,A]
        bleu .BB0

        ! CHECK: bcc .BB0      ! encoding: [0x1a,0b10AAAAAA,A,A]
        bcc .BB0

        ! CHECK: bcc .BB0      ! encoding: [0x1a,0b10AAAAAA,A,A]
        bgeu .BB0

        ! CHECK: bcs .BB0      ! encoding: [0x0a,0b10AAAAAA,A,A]
        bcs .BB0

        ! CHECK: bcs .BB0      ! encoding: [0x0a,0b10AAAAAA,A,A]
        blu .BB0

        ! CHECK: bpos .BB0      ! encoding: [0x1c,0b10AAAAAA,A,A]
        bpos .BB0

        ! CHECK: bneg .BB0      ! encoding: [0x0c,0b10AAAAAA,A,A]
        bneg .BB0

        ! CHECK: bvc .BB0      ! encoding: [0x1e,0b10AAAAAA,A,A]
        bvc .BB0

        ! CHECK: bvs .BB0      ! encoding: [0x0e,0b10AAAAAA,A,A]
        bvs .BB0

        ! CHECK:             fba .BB0                        ! encoding: [0x11,0b10AAAAAA,A,A]
        fba .BB0

        ! CHECK:             fba .BB0                        ! encoding: [0x11,0b10AAAAAA,A,A]
        fb .BB0

        ! CHECK:             fbn .BB0                        ! encoding: [0x01,0b10AAAAAA,A,A]
        fbn .BB0

        ! CHECK:             fbu .BB0                        ! encoding: [0x0f,0b10AAAAAA,A,A]
        fbu .BB0

        ! CHECK:             fbg .BB0                        ! encoding: [0x0d,0b10AAAAAA,A,A]
        fbg .BB0

        ! CHECK:             fbug .BB0                       ! encoding: [0x0b,0b10AAAAAA,A,A]
        fbug .BB0

        ! CHECK:             fbl .BB0                        ! encoding: [0x09,0b10AAAAAA,A,A]
        fbl .BB0

        ! CHECK:             fbul .BB0                       ! encoding: [0x07,0b10AAAAAA,A,A]
        fbul .BB0

        ! CHECK:             fblg .BB0                       ! encoding: [0x05,0b10AAAAAA,A,A]
        fblg .BB0

        ! CHECK:             fbne .BB0                       ! encoding: [0x03,0b10AAAAAA,A,A]
        fbne .BB0

        ! CHECK:             fbne .BB0                       ! encoding: [0x03,0b10AAAAAA,A,A]
        fbnz .BB0

        ! CHECK:             fbe .BB0                        ! encoding: [0x13,0b10AAAAAA,A,A]
        fbe .BB0

        ! CHECK:             fbe .BB0                        ! encoding: [0x13,0b10AAAAAA,A,A]
        fbz .BB0

        ! CHECK:             fbue .BB0                       ! encoding: [0x15,0b10AAAAAA,A,A]
        fbue .BB0

        ! CHECK:             fbge .BB0                       ! encoding: [0x17,0b10AAAAAA,A,A]
        fbge .BB0

        ! CHECK:             fbuge .BB0                      ! encoding: [0x19,0b10AAAAAA,A,A]
        fbuge .BB0

        ! CHECK:             fble .BB0                       ! encoding: [0x1b,0b10AAAAAA,A,A]
        fble .BB0

        ! CHECK:             fbule .BB0                      ! encoding: [0x1d,0b10AAAAAA,A,A]
        fbule .BB0

        ! CHECK:             fbo .BB0                        ! encoding: [0x1f,0b10AAAAAA,A,A]
        fbo .BB0
        
        ! CHECK:             cba .BB0                        ! encoding: [0x11,0b11AAAAAA,A,A]
        cb .BB0

        ! CHECK:             cba .BB0                        ! encoding: [0x11,0b11AAAAAA,A,A]
        cba .BB0

        ! CHECK:             cbn .BB0                        ! encoding: [0x01,0b11AAAAAA,A,A]
        cbn .BB0

        ! CHECK:             cb3 .BB0                        ! encoding: [0x0f,0b11AAAAAA,A,A]
        cb3 .BB0

        ! CHECK:             cb2 .BB0                        ! encoding: [0x0d,0b11AAAAAA,A,A]
        cb2 .BB0

        ! CHECK:             cb23 .BB0                       ! encoding: [0x0b,0b11AAAAAA,A,A]
        cb23 .BB0

        ! CHECK:             cb1 .BB0                        ! encoding: [0x09,0b11AAAAAA,A,A]
        cb1 .BB0

        ! CHECK:             cb13 .BB0                       ! encoding: [0x07,0b11AAAAAA,A,A]
        cb13 .BB0

        ! CHECK:             cb12 .BB0                       ! encoding: [0x05,0b11AAAAAA,A,A]
        cb12 .BB0

        ! CHECK:             cb123 .BB0                      ! encoding: [0x03,0b11AAAAAA,A,A]
        cb123 .BB0

        ! CHECK:             cb0 .BB0                        ! encoding: [0x13,0b11AAAAAA,A,A]
        cb0 .BB0

        ! CHECK:             cb03 .BB0                       ! encoding: [0x15,0b11AAAAAA,A,A]
        cb03 .BB0

        ! CHECK:             cb02 .BB0                       ! encoding: [0x17,0b11AAAAAA,A,A]
        cb02 .BB0

        ! CHECK:             cb023 .BB0                      ! encoding: [0x19,0b11AAAAAA,A,A]
        cb023 .BB0

        ! CHECK:             cb01 .BB0                       ! encoding: [0x1b,0b11AAAAAA,A,A]
        cb01 .BB0

        ! CHECK:             cb013 .BB0                      ! encoding: [0x1d,0b11AAAAAA,A,A]
        cb013 .BB0

        ! CHECK:             cb012 .BB0                      ! encoding: [0x1f,0b11AAAAAA,A,A]
        cb012 .BB0

        ! CHECK: ba,a .BB0    ! encoding: [0x30,0b10AAAAAA,A,A]
        ba,a .BB0

        ! CHECK: bne,a .BB0   ! encoding: [0x32,0b10AAAAAA,A,A]
        bne,a .BB0

        ! CHECK: be,a .BB0    ! encoding: [0x22,0b10AAAAAA,A,A]
        be,a .BB0

        ! CHECK: bg,a .BB0    ! encoding: [0x34,0b10AAAAAA,A,A]
        bg,a .BB0

        ! CHECK: ble,a .BB0   ! encoding: [0x24,0b10AAAAAA,A,A]
        ble,a .BB0

        ! CHECK: bge,a .BB0   ! encoding: [0x36,0b10AAAAAA,A,A]
        bge,a .BB0

        ! CHECK: bl,a .BB0    ! encoding: [0x26,0b10AAAAAA,A,A]
        bl,a .BB0

        ! CHECK: bgu,a .BB0   ! encoding: [0x38,0b10AAAAAA,A,A]
        bgu,a .BB0

        ! CHECK: bleu,a .BB0  ! encoding: [0x28,0b10AAAAAA,A,A]
        bleu,a .BB0

        ! CHECK: bcc,a .BB0   ! encoding: [0x3a,0b10AAAAAA,A,A]
        bcc,a .BB0

        ! CHECK: bcs,a .BB0   ! encoding: [0x2a,0b10AAAAAA,A,A]
        bcs,a .BB0

        ! CHECK: bpos,a .BB0  ! encoding: [0x3c,0b10AAAAAA,A,A]
        bpos,a .BB0

        ! CHECK: bneg,a .BB0  ! encoding: [0x2c,0b10AAAAAA,A,A]
        bneg,a .BB0

        ! CHECK: bvc,a .BB0   ! encoding: [0x3e,0b10AAAAAA,A,A]
        bvc,a .BB0

        ! CHECK: bvs,a .BB0   ! encoding: [0x2e,0b10AAAAAA,A,A]
        bvs,a .BB0

        ! CHECK:             fbu,a .BB0                      ! encoding: [0x2f,0b10AAAAAA,A,A]
        fbu,a .BB0

        ! CHECK:             fbg,a .BB0                      ! encoding: [0x2d,0b10AAAAAA,A,A]
        fbg,a .BB0
        ! CHECK:             fbug,a .BB0                     ! encoding: [0x2b,0b10AAAAAA,A,A]
        fbug,a .BB0

        ! CHECK:             fbl,a .BB0                      ! encoding: [0x29,0b10AAAAAA,A,A]
        fbl,a .BB0

        ! CHECK:             fbul,a .BB0                     ! encoding: [0x27,0b10AAAAAA,A,A]
        fbul,a .BB0

        ! CHECK:             fblg,a .BB0                     ! encoding: [0x25,0b10AAAAAA,A,A]
        fblg,a .BB0

        ! CHECK:             fbne,a .BB0                     ! encoding: [0x23,0b10AAAAAA,A,A]
        fbne,a .BB0

        ! CHECK:             fbe,a .BB0                      ! encoding: [0x33,0b10AAAAAA,A,A]
        fbe,a .BB0

        ! CHECK:             fbue,a .BB0                     ! encoding: [0x35,0b10AAAAAA,A,A]
        fbue,a .BB0

        ! CHECK:             fbge,a .BB0                     ! encoding: [0x37,0b10AAAAAA,A,A]
        fbge,a .BB0

        ! CHECK:             fbuge,a .BB0                    ! encoding: [0x39,0b10AAAAAA,A,A]
        fbuge,a .BB0

        ! CHECK:             fble,a .BB0                     ! encoding: [0x3b,0b10AAAAAA,A,A]
        fble,a .BB0

        ! CHECK:             fbule,a .BB0                    ! encoding: [0x3d,0b10AAAAAA,A,A]
        fbule,a .BB0

        ! CHECK:             fbo,a .BB0                      ! encoding: [0x3f,0b10AAAAAA,A,A]
        fbo,a .BB0

        ! CHECK:  rett %i7+8   ! encoding: [0x81,0xcf,0xe0,0x08]
        rett %i7 + 8

        ! CHECK:             cb3,a .BB0                      ! encoding: [0x2f,0b11AAAAAA,A,A]
        cb3,a .BB0

        ! CHECK:             cb2,a .BB0                      ! encoding: [0x2d,0b11AAAAAA,A,A]
        cb2,a .BB0

        ! CHECK:             cb23,a .BB0                     ! encoding: [0x2b,0b11AAAAAA,A,A]
        cb23,a .BB0

        ! CHECK:             cb1,a .BB0                      ! encoding: [0x29,0b11AAAAAA,A,A]
        cb1,a .BB0

        ! CHECK:             cb13,a .BB0                     ! encoding: [0x27,0b11AAAAAA,A,A]
        cb13,a .BB0

        ! CHECK:             cb12,a .BB0                     ! encoding: [0x25,0b11AAAAAA,A,A]
        cb12,a .BB0

        ! CHECK:             cb123,a .BB0                    ! encoding: [0x23,0b11AAAAAA,A,A]
        cb123,a .BB0

        ! CHECK:             cb0,a .BB0                      ! encoding: [0x33,0b11AAAAAA,A,A]
        cb0,a .BB0

        ! CHECK:             cb03,a .BB0                     ! encoding: [0x35,0b11AAAAAA,A,A]
        cb03,a .BB0

        ! CHECK:             cb02,a .BB0                     ! encoding: [0x37,0b11AAAAAA,A,A]
        cb02,a .BB0

        ! CHECK:             cb023,a .BB0                    ! encoding: [0x39,0b11AAAAAA,A,A]
        cb023,a .BB0

        ! CHECK:             cb01,a .BB0                     ! encoding: [0x3b,0b11AAAAAA,A,A]
        cb01,a .BB0

        ! CHECK:             cb013,a .BB0                    ! encoding: [0x3d,0b11AAAAAA,A,A]
        cb013,a .BB0

        ! CHECK:             cb012,a .BB0                    ! encoding: [0x3f,0b11AAAAAA,A,A]
        cb012,a .BB0

        ! CHECK:             cb3,a .BB0                      ! encoding: [0x2f,0b11AAAAAA,A,A]
        cb3,a .BB0

        ! CHECK:             cb2,a .BB0                      ! encoding: [0x2d,0b11AAAAAA,A,A]
        cb2,a .BB0

        ! CHECK:             cb23,a .BB0                     ! encoding: [0x2b,0b11AAAAAA,A,A]
        cb23,a .BB0

        ! CHECK:             cb1,a .BB0                      ! encoding: [0x29,0b11AAAAAA,A,A]
        cb1,a .BB0

        ! CHECK:             cb13,a .BB0                     ! encoding: [0x27,0b11AAAAAA,A,A]
        cb13,a .BB0

        ! CHECK:             cb12,a .BB0                     ! encoding: [0x25,0b11AAAAAA,A,A]
        cb12,a .BB0

        ! CHECK:             cb123,a .BB0                    ! encoding: [0x23,0b11AAAAAA,A,A]
        cb123,a .BB0

        ! CHECK:             cb0,a .BB0                      ! encoding: [0x33,0b11AAAAAA,A,A]
        cb0,a .BB0

        ! CHECK:             cb03,a .BB0                     ! encoding: [0x35,0b11AAAAAA,A,A]
        cb03,a .BB0

        ! CHECK:             cb02,a .BB0                     ! encoding: [0x37,0b11AAAAAA,A,A]
        cb02,a .BB0

        ! CHECK:             cb023,a .BB0                    ! encoding: [0x39,0b11AAAAAA,A,A]
        cb023,a .BB0

        ! CHECK:             cb01,a .BB0                     ! encoding: [0x3b,0b11AAAAAA,A,A]
        cb01,a .BB0

        ! CHECK:             cb013,a .BB0                    ! encoding: [0x3d,0b11AAAAAA,A,A]
        cb013,a .BB0

        ! CHECK:             cb012,a .BB0                    ! encoding: [0x3f,0b11AAAAAA,A,A]
        cb012,a .BB0

        ! CHECK:  rett %i7+8                                 ! encoding: [0x81,0xcf,0xe0,0x08]
        rett %i7 + 8

