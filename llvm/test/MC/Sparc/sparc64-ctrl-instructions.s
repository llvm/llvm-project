! RUN: llvm-mc %s -triple=sparc64-unknown-linux-gnu -show-encoding | FileCheck %s


        ! CHECK: bne %xcc, .BB0     ! encoding: [0x12,0b01101AAA,A,A]
        bne %xcc, .BB0

        ! CHECK: be %xcc, .BB0      ! encoding: [0x02,0b01101AAA,A,A]
        be %xcc, .BB0

        ! CHECK: bg %xcc, .BB0      ! encoding: [0x14,0b01101AAA,A,A]
        bg %xcc, .BB0

        ! CHECK: ble %xcc, .BB0      ! encoding: [0x04,0b01101AAA,A,A]
        ble %xcc, .BB0

        ! CHECK: bge %xcc, .BB0      ! encoding: [0x16,0b01101AAA,A,A]
        bge %xcc, .BB0

        ! CHECK: bl %xcc, .BB0      ! encoding: [0x06,0b01101AAA,A,A]
        bl %xcc, .BB0

        ! CHECK: bgu %xcc, .BB0      ! encoding: [0x18,0b01101AAA,A,A]
        bgu %xcc, .BB0

        ! CHECK: bleu %xcc, .BB0      ! encoding: [0x08,0b01101AAA,A,A]
        bleu %xcc, .BB0

        ! CHECK: bcc %xcc, .BB0      ! encoding: [0x1a,0b01101AAA,A,A]
        bcc %xcc, .BB0

        ! CHECK: bcs %xcc, .BB0      ! encoding: [0x0a,0b01101AAA,A,A]
        bcs %xcc, .BB0

        ! CHECK: bpos %xcc, .BB0      ! encoding: [0x1c,0b01101AAA,A,A]
        bpos %xcc, .BB0

        ! CHECK: bneg %xcc, .BB0      ! encoding: [0x0c,0b01101AAA,A,A]
        bneg %xcc, .BB0

        ! CHECK: bvc %xcc, .BB0      ! encoding: [0x1e,0b01101AAA,A,A]
        bvc %xcc, .BB0

        ! CHECK: bvs %xcc, .BB0      ! encoding: [0x0e,0b01101AAA,A,A]
        bvs %xcc, .BB0


        ! CHECK: movne %icc, %g1, %g2            ! encoding: [0x85,0x66,0x40,0x01]
        ! CHECK: move %icc, %g1, %g2             ! encoding: [0x85,0x64,0x40,0x01]
        ! CHECK: movg %icc, %g1, %g2             ! encoding: [0x85,0x66,0x80,0x01]
        ! CHECK: movle %icc, %g1, %g2            ! encoding: [0x85,0x64,0x80,0x01]
        ! CHECK: movge %icc, %g1, %g2            ! encoding: [0x85,0x66,0xc0,0x01]
        ! CHECK: movl %icc, %g1, %g2             ! encoding: [0x85,0x64,0xc0,0x01]
        ! CHECK: movgu %icc, %g1, %g2            ! encoding: [0x85,0x67,0x00,0x01]
        ! CHECK: movleu %icc, %g1, %g2           ! encoding: [0x85,0x65,0x00,0x01]
        ! CHECK: movcc %icc, %g1, %g2            ! encoding: [0x85,0x67,0x40,0x01]
        ! CHECK: movcs %icc, %g1, %g2            ! encoding: [0x85,0x65,0x40,0x01]
        ! CHECK: movpos %icc, %g1, %g2           ! encoding: [0x85,0x67,0x80,0x01]
        ! CHECK: movneg %icc, %g1, %g2           ! encoding: [0x85,0x65,0x80,0x01]
        ! CHECK: movvc %icc, %g1, %g2            ! encoding: [0x85,0x67,0xc0,0x01]
        ! CHECK: movvs %icc, %g1, %g2            ! encoding: [0x85,0x65,0xc0,0x01]
        movne  %icc, %g1, %g2
        move   %icc, %g1, %g2
        movg   %icc, %g1, %g2
        movle  %icc, %g1, %g2
        movge  %icc, %g1, %g2
        movl   %icc, %g1, %g2
        movgu  %icc, %g1, %g2
        movleu %icc, %g1, %g2
        movcc  %icc, %g1, %g2
        movcs  %icc, %g1, %g2
        movpos %icc, %g1, %g2
        movneg %icc, %g1, %g2
        movvc  %icc, %g1, %g2
        movvs  %icc, %g1, %g2

        ! CHECK: movne %xcc, %g1, %g2            ! encoding: [0x85,0x66,0x50,0x01]
        ! CHECK: move %xcc, %g1, %g2             ! encoding: [0x85,0x64,0x50,0x01]
        ! CHECK: movg %xcc, %g1, %g2             ! encoding: [0x85,0x66,0x90,0x01]
        ! CHECK: movle %xcc, %g1, %g2            ! encoding: [0x85,0x64,0x90,0x01]
        ! CHECK: movge %xcc, %g1, %g2            ! encoding: [0x85,0x66,0xd0,0x01]
        ! CHECK: movl %xcc, %g1, %g2             ! encoding: [0x85,0x64,0xd0,0x01]
        ! CHECK: movgu %xcc, %g1, %g2            ! encoding: [0x85,0x67,0x10,0x01]
        ! CHECK: movleu %xcc, %g1, %g2           ! encoding: [0x85,0x65,0x10,0x01]
        ! CHECK: movcc %xcc, %g1, %g2            ! encoding: [0x85,0x67,0x50,0x01]
        ! CHECK: movcs %xcc, %g1, %g2            ! encoding: [0x85,0x65,0x50,0x01]
        ! CHECK: movpos %xcc, %g1, %g2           ! encoding: [0x85,0x67,0x90,0x01]
        ! CHECK: movneg %xcc, %g1, %g2           ! encoding: [0x85,0x65,0x90,0x01]
        ! CHECK: movvc %xcc, %g1, %g2            ! encoding: [0x85,0x67,0xd0,0x01]
        ! CHECK: movvs %xcc, %g1, %g2            ! encoding: [0x85,0x65,0xd0,0x01]
        movne  %xcc, %g1, %g2
        move   %xcc, %g1, %g2
        movg   %xcc, %g1, %g2
        movle  %xcc, %g1, %g2
        movge  %xcc, %g1, %g2
        movl   %xcc, %g1, %g2
        movgu  %xcc, %g1, %g2
        movleu %xcc, %g1, %g2
        movcc  %xcc, %g1, %g2
        movcs  %xcc, %g1, %g2
        movpos %xcc, %g1, %g2
        movneg %xcc, %g1, %g2
        movvc  %xcc, %g1, %g2
        movvs  %xcc, %g1, %g2

        ! CHECK: movu %fcc0, %g1, %g2            ! encoding: [0x85,0x61,0xc0,0x01]
        ! CHECK: movg %fcc0, %g1, %g2            ! encoding: [0x85,0x61,0x80,0x01]
        ! CHECK: movug %fcc0, %g1, %g2           ! encoding: [0x85,0x61,0x40,0x01]
        ! CHECK: movl %fcc0, %g1, %g2            ! encoding: [0x85,0x61,0x00,0x01]
        ! CHECK: movul %fcc0, %g1, %g2           ! encoding: [0x85,0x60,0xc0,0x01]
        ! CHECK: movlg %fcc0, %g1, %g2           ! encoding: [0x85,0x60,0x80,0x01]
        ! CHECK: movne %fcc0, %g1, %g2           ! encoding: [0x85,0x60,0x40,0x01]
        ! CHECK: move %fcc0, %g1, %g2            ! encoding: [0x85,0x62,0x40,0x01]
        ! CHECK: movue %fcc0, %g1, %g2           ! encoding: [0x85,0x62,0x80,0x01]
        ! CHECK: movge %fcc0, %g1, %g2           ! encoding: [0x85,0x62,0xc0,0x01]
        ! CHECK: movuge %fcc0, %g1, %g2          ! encoding: [0x85,0x63,0x00,0x01]
        ! CHECK: movle %fcc0, %g1, %g2           ! encoding: [0x85,0x63,0x40,0x01]
        ! CHECK: movule %fcc0, %g1, %g2          ! encoding: [0x85,0x63,0x80,0x01]
        ! CHECK: movo %fcc0, %g1, %g2            ! encoding: [0x85,0x63,0xc0,0x01]
        movu   %fcc0, %g1, %g2
        movg   %fcc0, %g1, %g2
        movug  %fcc0, %g1, %g2
        movl   %fcc0, %g1, %g2
        movul  %fcc0, %g1, %g2
        movlg  %fcc0, %g1, %g2
        movne  %fcc0, %g1, %g2
        move   %fcc0, %g1, %g2
        movue  %fcc0, %g1, %g2
        movge  %fcc0, %g1, %g2
        movuge %fcc0, %g1, %g2
        movle  %fcc0, %g1, %g2
        movule %fcc0, %g1, %g2
        movo   %fcc0, %g1, %g2


        ! CHECK: fmovsne %icc, %f1, %f2          ! encoding: [0x85,0xaa,0x60,0x21]
        ! CHECK: fmovse %icc, %f1, %f2           ! encoding: [0x85,0xa8,0x60,0x21]
        ! CHECK: fmovsg %icc, %f1, %f2           ! encoding: [0x85,0xaa,0xa0,0x21]
        ! CHECK: fmovsle %icc, %f1, %f2          ! encoding: [0x85,0xa8,0xa0,0x21]
        ! CHECK: fmovsge %icc, %f1, %f2          ! encoding: [0x85,0xaa,0xe0,0x21]
        ! CHECK: fmovsl %icc, %f1, %f2           ! encoding: [0x85,0xa8,0xe0,0x21]
        ! CHECK: fmovsgu %icc, %f1, %f2          ! encoding: [0x85,0xab,0x20,0x21]
        ! CHECK: fmovsleu %icc, %f1, %f2         ! encoding: [0x85,0xa9,0x20,0x21]
        ! CHECK: fmovscc %icc, %f1, %f2          ! encoding: [0x85,0xab,0x60,0x21]
        ! CHECK: fmovscs %icc, %f1, %f2          ! encoding: [0x85,0xa9,0x60,0x21]
        ! CHECK: fmovspos %icc, %f1, %f2         ! encoding: [0x85,0xab,0xa0,0x21]
        ! CHECK: fmovsneg %icc, %f1, %f2         ! encoding: [0x85,0xa9,0xa0,0x21]
        ! CHECK: fmovsvc %icc, %f1, %f2          ! encoding: [0x85,0xab,0xe0,0x21]
        ! CHECK: fmovsvs %icc, %f1, %f2          ! encoding: [0x85,0xa9,0xe0,0x21]
        fmovsne  %icc, %f1, %f2
        fmovse   %icc, %f1, %f2
        fmovsg   %icc, %f1, %f2
        fmovsle  %icc, %f1, %f2
        fmovsge  %icc, %f1, %f2
        fmovsl   %icc, %f1, %f2
        fmovsgu  %icc, %f1, %f2
        fmovsleu %icc, %f1, %f2
        fmovscc  %icc, %f1, %f2
        fmovscs  %icc, %f1, %f2
        fmovspos %icc, %f1, %f2
        fmovsneg %icc, %f1, %f2
        fmovsvc  %icc, %f1, %f2
        fmovsvs  %icc, %f1, %f2

        ! CHECK: fmovsne %xcc, %f1, %f2          ! encoding: [0x85,0xaa,0x70,0x21]
        ! CHECK: fmovse %xcc, %f1, %f2           ! encoding: [0x85,0xa8,0x70,0x21]
        ! CHECK: fmovsg %xcc, %f1, %f2           ! encoding: [0x85,0xaa,0xb0,0x21]
        ! CHECK: fmovsle %xcc, %f1, %f2          ! encoding: [0x85,0xa8,0xb0,0x21]
        ! CHECK: fmovsge %xcc, %f1, %f2          ! encoding: [0x85,0xaa,0xf0,0x21]
        ! CHECK: fmovsl %xcc, %f1, %f2           ! encoding: [0x85,0xa8,0xf0,0x21]
        ! CHECK: fmovsgu %xcc, %f1, %f2          ! encoding: [0x85,0xab,0x30,0x21]
        ! CHECK: fmovsleu %xcc, %f1, %f2         ! encoding: [0x85,0xa9,0x30,0x21]
        ! CHECK: fmovscc %xcc, %f1, %f2          ! encoding: [0x85,0xab,0x70,0x21]
        ! CHECK: fmovscs %xcc, %f1, %f2          ! encoding: [0x85,0xa9,0x70,0x21]
        ! CHECK: fmovspos %xcc, %f1, %f2         ! encoding: [0x85,0xab,0xb0,0x21]
        ! CHECK: fmovsneg %xcc, %f1, %f2         ! encoding: [0x85,0xa9,0xb0,0x21]
        ! CHECK: fmovsvc %xcc, %f1, %f2          ! encoding: [0x85,0xab,0xf0,0x21]
        ! CHECK: fmovsvs %xcc, %f1, %f2          ! encoding: [0x85,0xa9,0xf0,0x21]
        fmovsne  %xcc, %f1, %f2
        fmovse   %xcc, %f1, %f2
        fmovsg   %xcc, %f1, %f2
        fmovsle  %xcc, %f1, %f2
        fmovsge  %xcc, %f1, %f2
        fmovsl   %xcc, %f1, %f2
        fmovsgu  %xcc, %f1, %f2
        fmovsleu %xcc, %f1, %f2
        fmovscc  %xcc, %f1, %f2
        fmovscs  %xcc, %f1, %f2
        fmovspos %xcc, %f1, %f2
        fmovsneg %xcc, %f1, %f2
        fmovsvc  %xcc, %f1, %f2
        fmovsvs  %xcc, %f1, %f2

        ! CHECK: fmovsu %fcc0, %f1, %f2          ! encoding: [0x85,0xa9,0xc0,0x21]
        ! CHECK: fmovsg %fcc0, %f1, %f2          ! encoding: [0x85,0xa9,0x80,0x21]
        ! CHECK: fmovsug %fcc0, %f1, %f2         ! encoding: [0x85,0xa9,0x40,0x21]
        ! CHECK: fmovsl %fcc0, %f1, %f2          ! encoding: [0x85,0xa9,0x00,0x21]
        ! CHECK: fmovsul %fcc0, %f1, %f2         ! encoding: [0x85,0xa8,0xc0,0x21]
        ! CHECK: fmovslg %fcc0, %f1, %f2         ! encoding: [0x85,0xa8,0x80,0x21]
        ! CHECK: fmovsne %fcc0, %f1, %f2         ! encoding: [0x85,0xa8,0x40,0x21]
        ! CHECK: fmovse %fcc0, %f1, %f2          ! encoding: [0x85,0xaa,0x40,0x21]
        ! CHECK: fmovsue %fcc0, %f1, %f2         ! encoding: [0x85,0xaa,0x80,0x21]
        ! CHECK: fmovsge %fcc0, %f1, %f2         ! encoding: [0x85,0xaa,0xc0,0x21]
        ! CHECK: fmovsuge %fcc0, %f1, %f2        ! encoding: [0x85,0xab,0x00,0x21]
        ! CHECK: fmovsle %fcc0, %f1, %f2         ! encoding: [0x85,0xab,0x40,0x21]
        ! CHECK: fmovsule %fcc0, %f1, %f2        ! encoding: [0x85,0xab,0x80,0x21]
        ! CHECK: fmovso %fcc0, %f1, %f2          ! encoding: [0x85,0xab,0xc0,0x21]
        fmovsu   %fcc0, %f1, %f2
        fmovsg   %fcc0, %f1, %f2
        fmovsug  %fcc0, %f1, %f2
        fmovsl   %fcc0, %f1, %f2
        fmovsul  %fcc0, %f1, %f2
        fmovslg  %fcc0, %f1, %f2
        fmovsne  %fcc0, %f1, %f2
        fmovse   %fcc0, %f1, %f2
        fmovsue  %fcc0, %f1, %f2
        fmovsge  %fcc0, %f1, %f2
        fmovsuge %fcc0, %f1, %f2
        fmovsle  %fcc0, %f1, %f2
        fmovsule %fcc0, %f1, %f2
        fmovso   %fcc0, %f1, %f2

        ! CHECK: bne,a %icc, .BB0     ! encoding: [0x32,0b01001AAA,A,A]
        bne,a %icc, .BB0

        ! CHECK: be,a %icc, .BB0      ! encoding: [0x22,0b01001AAA,A,A]
        be,a %icc, .BB0

        ! CHECK: bg,a %icc, .BB0      ! encoding: [0x34,0b01001AAA,A,A]
        bg,a %icc, .BB0

        ! CHECK: ble,a %icc, .BB0      ! encoding: [0x24,0b01001AAA,A,A]
        ble,a %icc, .BB0

        ! CHECK: bge,a %icc, .BB0      ! encoding: [0x36,0b01001AAA,A,A]
        bge,a %icc, .BB0

        ! CHECK: bl,a %icc, .BB0      ! encoding: [0x26,0b01001AAA,A,A]
        bl,a %icc, .BB0

        ! CHECK: bgu,a %icc, .BB0      ! encoding: [0x38,0b01001AAA,A,A]
        bgu,a %icc, .BB0

        ! CHECK: bleu,a %icc, .BB0      ! encoding: [0x28,0b01001AAA,A,A]
        bleu,a %icc, .BB0

        ! CHECK: bcc,a %icc, .BB0      ! encoding: [0x3a,0b01001AAA,A,A]
        bcc,a %icc, .BB0

        ! CHECK: bcs,a %icc, .BB0      ! encoding: [0x2a,0b01001AAA,A,A]
        bcs,a %icc, .BB0

        ! CHECK: bpos,a %icc, .BB0      ! encoding: [0x3c,0b01001AAA,A,A]
        bpos,a %icc, .BB0

        ! CHECK: bneg,a %icc, .BB0      ! encoding: [0x2c,0b01001AAA,A,A]
        bneg,a %icc, .BB0

        ! CHECK: bvc,a %icc, .BB0      ! encoding: [0x3e,0b01001AAA,A,A]
        bvc,a %icc, .BB0

        ! CHECK: bvs,a %icc, .BB0      ! encoding: [0x2e,0b01001AAA,A,A]
        bvs,a %icc, .BB0

        ! CHECK: bne,pn %icc, .BB0     ! encoding: [0x12,0b01000AAA,A,A]
        bne,pn %icc, .BB0

        ! CHECK: be,pn %icc, .BB0      ! encoding: [0x02,0b01000AAA,A,A]
        be,pn %icc, .BB0

        ! CHECK: bg,pn %icc, .BB0      ! encoding: [0x14,0b01000AAA,A,A]
        bg,pn %icc, .BB0

        ! CHECK: ble,pn %icc, .BB0      ! encoding: [0x04,0b01000AAA,A,A]
        ble,pn %icc, .BB0

        ! CHECK: bge,pn %icc, .BB0      ! encoding: [0x16,0b01000AAA,A,A]
        bge,pn %icc, .BB0

        ! CHECK: bl,pn %icc, .BB0      ! encoding: [0x06,0b01000AAA,A,A]
        bl,pn %icc, .BB0

        ! CHECK: bgu,pn %icc, .BB0      ! encoding: [0x18,0b01000AAA,A,A]
        bgu,pn %icc, .BB0

        ! CHECK: bleu,pn %icc, .BB0      ! encoding: [0x08,0b01000AAA,A,A]
        bleu,pn %icc, .BB0

        ! CHECK: bcc,pn %icc, .BB0      ! encoding: [0x1a,0b01000AAA,A,A]
        bcc,pn %icc, .BB0

        ! CHECK: bcs,pn %icc, .BB0      ! encoding: [0x0a,0b01000AAA,A,A]
        bcs,pn %icc, .BB0

        ! CHECK: bpos,pn %icc, .BB0      ! encoding: [0x1c,0b01000AAA,A,A]
        bpos,pn %icc, .BB0

        ! CHECK: bneg,pn %icc, .BB0      ! encoding: [0x0c,0b01000AAA,A,A]
        bneg,pn %icc, .BB0

        ! CHECK: bvc,pn %icc, .BB0      ! encoding: [0x1e,0b01000AAA,A,A]
        bvc,pn %icc, .BB0

        ! CHECK: bvs,pn %icc, .BB0      ! encoding: [0x0e,0b01000AAA,A,A]
        bvs,pn %icc, .BB0

        ! CHECK: bne,a,pn %icc, .BB0     ! encoding: [0x32,0b01000AAA,A,A]
        bne,a,pn %icc, .BB0

        ! CHECK: be,a,pn %icc, .BB0      ! encoding: [0x22,0b01000AAA,A,A]
        be,a,pn %icc, .BB0

        ! CHECK: bg,a,pn %icc, .BB0      ! encoding: [0x34,0b01000AAA,A,A]
        bg,a,pn %icc, .BB0

        ! CHECK: ble,a,pn %icc, .BB0      ! encoding: [0x24,0b01000AAA,A,A]
        ble,a,pn %icc, .BB0

        ! CHECK: bge,a,pn %icc, .BB0      ! encoding: [0x36,0b01000AAA,A,A]
        bge,a,pn %icc, .BB0

        ! CHECK: bl,a,pn %icc, .BB0      ! encoding: [0x26,0b01000AAA,A,A]
        bl,a,pn %icc, .BB0

        ! CHECK: bgu,a,pn %icc, .BB0      ! encoding: [0x38,0b01000AAA,A,A]
        bgu,a,pn %icc, .BB0

        ! CHECK: bleu,a,pn %icc, .BB0      ! encoding: [0x28,0b01000AAA,A,A]
        bleu,a,pn %icc, .BB0

        ! CHECK: bcc,a,pn %icc, .BB0      ! encoding: [0x3a,0b01000AAA,A,A]
        bcc,a,pn %icc, .BB0

        ! CHECK: bcs,a,pn %icc, .BB0      ! encoding: [0x2a,0b01000AAA,A,A]
        bcs,a,pn %icc, .BB0

        ! CHECK: bpos,a,pn %icc, .BB0      ! encoding: [0x3c,0b01000AAA,A,A]
        bpos,a,pn %icc, .BB0

        ! CHECK: bneg,a,pn %icc, .BB0      ! encoding: [0x2c,0b01000AAA,A,A]
        bneg,a,pn %icc, .BB0

        ! CHECK: bvc,a,pn %icc, .BB0      ! encoding: [0x3e,0b01000AAA,A,A]
        bvc,a,pn %icc, .BB0

        ! CHECK: bvs,a,pn %icc, .BB0      ! encoding: [0x2e,0b01000AAA,A,A]
        bvs,a,pn %icc, .BB0

        ! CHECK: bne %icc, .BB0     ! encoding: [0x12,0b01001AAA,A,A]
        bne,pt %icc, .BB0

        ! CHECK: be %icc, .BB0      ! encoding: [0x02,0b01001AAA,A,A]
        be,pt %icc, .BB0

        ! CHECK: bg %icc, .BB0      ! encoding: [0x14,0b01001AAA,A,A]
        bg,pt %icc, .BB0

        ! CHECK: ble %icc, .BB0      ! encoding: [0x04,0b01001AAA,A,A]
        ble,pt %icc, .BB0

        ! CHECK: bge %icc, .BB0      ! encoding: [0x16,0b01001AAA,A,A]
        bge,pt %icc, .BB0

        ! CHECK: bl %icc, .BB0      ! encoding: [0x06,0b01001AAA,A,A]
        bl,pt %icc, .BB0

        ! CHECK: bgu %icc, .BB0      ! encoding: [0x18,0b01001AAA,A,A]
        bgu,pt %icc, .BB0

        ! CHECK: bleu %icc, .BB0      ! encoding: [0x08,0b01001AAA,A,A]
        bleu,pt %icc, .BB0

        ! CHECK: bcc %icc, .BB0      ! encoding: [0x1a,0b01001AAA,A,A]
        bcc,pt %icc, .BB0

        ! CHECK: bcs %icc, .BB0      ! encoding: [0x0a,0b01001AAA,A,A]
        bcs,pt %icc, .BB0

        ! CHECK: bpos %icc, .BB0      ! encoding: [0x1c,0b01001AAA,A,A]
        bpos,pt %icc, .BB0

        ! CHECK: bneg %icc, .BB0      ! encoding: [0x0c,0b01001AAA,A,A]
        bneg,pt %icc, .BB0

        ! CHECK: bvc %icc, .BB0      ! encoding: [0x1e,0b01001AAA,A,A]
        bvc,pt %icc, .BB0

        ! CHECK: bvs %icc, .BB0      ! encoding: [0x0e,0b01001AAA,A,A]
        bvs,pt %icc, .BB0

        ! CHECK: bne,a %icc, .BB0     ! encoding: [0x32,0b01001AAA,A,A]
        bne,a,pt %icc, .BB0

        ! CHECK: be,a %icc, .BB0      ! encoding: [0x22,0b01001AAA,A,A]
        be,a,pt %icc, .BB0

        ! CHECK: bg,a %icc, .BB0      ! encoding: [0x34,0b01001AAA,A,A]
        bg,a,pt %icc, .BB0

        ! CHECK: ble,a %icc, .BB0      ! encoding: [0x24,0b01001AAA,A,A]
        ble,a,pt %icc, .BB0

        ! CHECK: bge,a %icc, .BB0      ! encoding: [0x36,0b01001AAA,A,A]
        bge,a,pt %icc, .BB0

        ! CHECK: bl,a %icc, .BB0      ! encoding: [0x26,0b01001AAA,A,A]
        bl,a,pt %icc, .BB0

        ! CHECK: bgu,a %icc, .BB0      ! encoding: [0x38,0b01001AAA,A,A]
        bgu,a,pt %icc, .BB0

        ! CHECK: bleu,a %icc, .BB0      ! encoding: [0x28,0b01001AAA,A,A]
        bleu,a,pt %icc, .BB0

        ! CHECK: bcc,a %icc, .BB0      ! encoding: [0x3a,0b01001AAA,A,A]
        bcc,a,pt %icc, .BB0

        ! CHECK: bcs,a %icc, .BB0      ! encoding: [0x2a,0b01001AAA,A,A]
        bcs,a,pt %icc, .BB0

        ! CHECK: bpos,a %icc, .BB0      ! encoding: [0x3c,0b01001AAA,A,A]
        bpos,a,pt %icc, .BB0


        ! CHECK: bne,a %xcc, .BB0     ! encoding: [0x32,0b01101AAA,A,A]
        bne,a %xcc, .BB0

        ! CHECK: be,a %xcc, .BB0      ! encoding: [0x22,0b01101AAA,A,A]
        be,a %xcc, .BB0

        ! CHECK: bg,a %xcc, .BB0      ! encoding: [0x34,0b01101AAA,A,A]
        bg,a %xcc, .BB0

        ! CHECK: ble,a %xcc, .BB0      ! encoding: [0x24,0b01101AAA,A,A]
        ble,a %xcc, .BB0

        ! CHECK: bge,a %xcc, .BB0      ! encoding: [0x36,0b01101AAA,A,A]
        bge,a %xcc, .BB0

        ! CHECK: bl,a %xcc, .BB0      ! encoding: [0x26,0b01101AAA,A,A]
        bl,a %xcc, .BB0

        ! CHECK: bgu,a %xcc, .BB0      ! encoding: [0x38,0b01101AAA,A,A]
        bgu,a %xcc, .BB0

        ! CHECK: bleu,a %xcc, .BB0      ! encoding: [0x28,0b01101AAA,A,A]
        bleu,a %xcc, .BB0

        ! CHECK: bcc,a %xcc, .BB0      ! encoding: [0x3a,0b01101AAA,A,A]
        bcc,a %xcc, .BB0

        ! CHECK: bcs,a %xcc, .BB0      ! encoding: [0x2a,0b01101AAA,A,A]
        bcs,a %xcc, .BB0

        ! CHECK: bpos,a %xcc, .BB0      ! encoding: [0x3c,0b01101AAA,A,A]
        bpos,a %xcc, .BB0

        ! CHECK: bneg,a %xcc, .BB0      ! encoding: [0x2c,0b01101AAA,A,A]
        bneg,a %xcc, .BB0

        ! CHECK: bvc,a %xcc, .BB0      ! encoding: [0x3e,0b01101AAA,A,A]
        bvc,a %xcc, .BB0

        ! CHECK: bvs,a %xcc, .BB0      ! encoding: [0x2e,0b01101AAA,A,A]
        bvs,a %xcc, .BB0

        ! CHECK: bne,pn %xcc, .BB0     ! encoding: [0x12,0b01100AAA,A,A]
        bne,pn %xcc, .BB0

        ! CHECK: be,pn %xcc, .BB0      ! encoding: [0x02,0b01100AAA,A,A]
        be,pn %xcc, .BB0

        ! CHECK: bg,pn %xcc, .BB0      ! encoding: [0x14,0b01100AAA,A,A]
        bg,pn %xcc, .BB0

        ! CHECK: ble,pn %xcc, .BB0      ! encoding: [0x04,0b01100AAA,A,A]
        ble,pn %xcc, .BB0

        ! CHECK: bge,pn %xcc, .BB0      ! encoding: [0x16,0b01100AAA,A,A]
        bge,pn %xcc, .BB0

        ! CHECK: bl,pn %xcc, .BB0      ! encoding: [0x06,0b01100AAA,A,A]
        bl,pn %xcc, .BB0

        ! CHECK: bgu,pn %xcc, .BB0      ! encoding: [0x18,0b01100AAA,A,A]
        bgu,pn %xcc, .BB0

        ! CHECK: bleu,pn %xcc, .BB0      ! encoding: [0x08,0b01100AAA,A,A]
        bleu,pn %xcc, .BB0

        ! CHECK: bcc,pn %xcc, .BB0      ! encoding: [0x1a,0b01100AAA,A,A]
        bcc,pn %xcc, .BB0

        ! CHECK: bcs,pn %xcc, .BB0      ! encoding: [0x0a,0b01100AAA,A,A]
        bcs,pn %xcc, .BB0

        ! CHECK: bpos,pn %xcc, .BB0      ! encoding: [0x1c,0b01100AAA,A,A]
        bpos,pn %xcc, .BB0

        ! CHECK: bneg,pn %xcc, .BB0      ! encoding: [0x0c,0b01100AAA,A,A]
        bneg,pn %xcc, .BB0

        ! CHECK: bvc,pn %xcc, .BB0      ! encoding: [0x1e,0b01100AAA,A,A]
        bvc,pn %xcc, .BB0

        ! CHECK: bvs,pn %xcc, .BB0      ! encoding: [0x0e,0b01100AAA,A,A]
        bvs,pn %xcc, .BB0

        ! CHECK: bne,a,pn %xcc, .BB0     ! encoding: [0x32,0b01100AAA,A,A]
        bne,a,pn %xcc, .BB0

        ! CHECK: be,a,pn %xcc, .BB0      ! encoding: [0x22,0b01100AAA,A,A]
        be,a,pn %xcc, .BB0

        ! CHECK: bg,a,pn %xcc, .BB0      ! encoding: [0x34,0b01100AAA,A,A]
        bg,a,pn %xcc, .BB0

        ! CHECK: ble,a,pn %xcc, .BB0      ! encoding: [0x24,0b01100AAA,A,A]
        ble,a,pn %xcc, .BB0

        ! CHECK: bge,a,pn %xcc, .BB0      ! encoding: [0x36,0b01100AAA,A,A]
        bge,a,pn %xcc, .BB0

        ! CHECK: bl,a,pn %xcc, .BB0      ! encoding: [0x26,0b01100AAA,A,A]
        bl,a,pn %xcc, .BB0

        ! CHECK: bgu,a,pn %xcc, .BB0      ! encoding: [0x38,0b01100AAA,A,A]
        bgu,a,pn %xcc, .BB0

        ! CHECK: bleu,a,pn %xcc, .BB0      ! encoding: [0x28,0b01100AAA,A,A]
        bleu,a,pn %xcc, .BB0

        ! CHECK: bcc,a,pn %xcc, .BB0      ! encoding: [0x3a,0b01100AAA,A,A]
        bcc,a,pn %xcc, .BB0

        ! CHECK: bcs,a,pn %xcc, .BB0      ! encoding: [0x2a,0b01100AAA,A,A]
        bcs,a,pn %xcc, .BB0

        ! CHECK: bpos,a,pn %xcc, .BB0      ! encoding: [0x3c,0b01100AAA,A,A]
        bpos,a,pn %xcc, .BB0

        ! CHECK: bneg,a,pn %xcc, .BB0      ! encoding: [0x2c,0b01100AAA,A,A]
        bneg,a,pn %xcc, .BB0

        ! CHECK: bvc,a,pn %xcc, .BB0      ! encoding: [0x3e,0b01100AAA,A,A]
        bvc,a,pn %xcc, .BB0

        ! CHECK: bvs,a,pn %xcc, .BB0      ! encoding: [0x2e,0b01100AAA,A,A]
        bvs,a,pn %xcc, .BB0

        ! CHECK: bne %xcc, .BB0     ! encoding: [0x12,0b01101AAA,A,A]
        bne,pt %xcc, .BB0

        ! CHECK: be %xcc, .BB0      ! encoding: [0x02,0b01101AAA,A,A]
        be,pt %xcc, .BB0

        ! CHECK: bg %xcc, .BB0      ! encoding: [0x14,0b01101AAA,A,A]
        bg,pt %xcc, .BB0

        ! CHECK: ble %xcc, .BB0      ! encoding: [0x04,0b01101AAA,A,A]
        ble,pt %xcc, .BB0

        ! CHECK: bge %xcc, .BB0      ! encoding: [0x16,0b01101AAA,A,A]
        bge,pt %xcc, .BB0

        ! CHECK: bl %xcc, .BB0      ! encoding: [0x06,0b01101AAA,A,A]
        bl,pt %xcc, .BB0

        ! CHECK: bgu %xcc, .BB0      ! encoding: [0x18,0b01101AAA,A,A]
        bgu,pt %xcc, .BB0

        ! CHECK: bleu %xcc, .BB0      ! encoding: [0x08,0b01101AAA,A,A]
        bleu,pt %xcc, .BB0

        ! CHECK: bcc %xcc, .BB0      ! encoding: [0x1a,0b01101AAA,A,A]
        bcc,pt %xcc, .BB0

        ! CHECK: bcs %xcc, .BB0      ! encoding: [0x0a,0b01101AAA,A,A]
        bcs,pt %xcc, .BB0

        ! CHECK: bpos %xcc, .BB0      ! encoding: [0x1c,0b01101AAA,A,A]
        bpos,pt %xcc, .BB0

        ! CHECK: bneg %xcc, .BB0      ! encoding: [0x0c,0b01101AAA,A,A]
        bneg,pt %xcc, .BB0

        ! CHECK: bvc %xcc, .BB0      ! encoding: [0x1e,0b01101AAA,A,A]
        bvc,pt %xcc, .BB0

        ! CHECK: bvs %xcc, .BB0      ! encoding: [0x0e,0b01101AAA,A,A]
        bvs,pt %xcc, .BB0

        ! CHECK: bne,a %xcc, .BB0     ! encoding: [0x32,0b01101AAA,A,A]
        bne,a,pt %xcc, .BB0

        ! CHECK: be,a %xcc, .BB0      ! encoding: [0x22,0b01101AAA,A,A]
        be,a,pt %xcc, .BB0

        ! CHECK: bg,a %xcc, .BB0      ! encoding: [0x34,0b01101AAA,A,A]
        bg,a,pt %xcc, .BB0

        ! CHECK: ble,a %xcc, .BB0      ! encoding: [0x24,0b01101AAA,A,A]
        ble,a,pt %xcc, .BB0

        ! CHECK: bge,a %xcc, .BB0      ! encoding: [0x36,0b01101AAA,A,A]
        bge,a,pt %xcc, .BB0

        ! CHECK: bl,a %xcc, .BB0      ! encoding: [0x26,0b01101AAA,A,A]
        bl,a,pt %xcc, .BB0

        ! CHECK: bgu,a %xcc, .BB0      ! encoding: [0x38,0b01101AAA,A,A]
        bgu,a,pt %xcc, .BB0

        ! CHECK: bleu,a %xcc, .BB0      ! encoding: [0x28,0b01101AAA,A,A]
        bleu,a,pt %xcc, .BB0

        ! CHECK: bcc,a %xcc, .BB0      ! encoding: [0x3a,0b01101AAA,A,A]
        bcc,a,pt %xcc, .BB0

        ! CHECK: bcs,a %xcc, .BB0      ! encoding: [0x2a,0b01101AAA,A,A]
        bcs,a,pt %xcc, .BB0

        ! CHECK: bpos,a %xcc, .BB0      ! encoding: [0x3c,0b01101AAA,A,A]
        bpos,a,pt %xcc, .BB0

        ! CHECK:             fba %fcc0, .BB0                        ! encoding: [0x11,0b01001AAA,A,A]
        fba %fcc0, .BB0

        ! CHECK:             fba %fcc0, .BB0                        ! encoding: [0x11,0b01001AAA,A,A]
        fb %fcc0, .BB0

        ! CHECK:             fbn %fcc0, .BB0                        ! encoding: [0x01,0b01001AAA,A,A]
        fbn %fcc0, .BB0

        ! CHECK:             fbu %fcc0, .BB0                      ! encoding: [0x0f,0b01001AAA,A,A]
        fbu %fcc0, .BB0

        ! CHECK:             fbg %fcc0, .BB0                      ! encoding: [0x0d,0b01001AAA,A,A]
        fbg %fcc0, .BB0
        ! CHECK:             fbug %fcc0, .BB0                     ! encoding: [0x0b,0b01001AAA,A,A]
        fbug %fcc0, .BB0

        ! CHECK:             fbl %fcc0, .BB0                      ! encoding: [0x09,0b01001AAA,A,A]
        fbl %fcc0, .BB0

        ! CHECK:             fbul %fcc0, .BB0                     ! encoding: [0x07,0b01001AAA,A,A]
        fbul %fcc0, .BB0

        ! CHECK:             fblg %fcc0, .BB0                     ! encoding: [0x05,0b01001AAA,A,A]
        fblg %fcc0, .BB0

        ! CHECK:             fbne %fcc0, .BB0                     ! encoding: [0x03,0b01001AAA,A,A]
        fbne %fcc0, .BB0

        ! CHECK:             fbe %fcc0, .BB0                      ! encoding: [0x13,0b01001AAA,A,A]
        fbe %fcc0, .BB0

        ! CHECK:             fbue %fcc0, .BB0                     ! encoding: [0x15,0b01001AAA,A,A]
        fbue %fcc0, .BB0

        ! CHECK:             fbge %fcc0, .BB0                     ! encoding: [0x17,0b01001AAA,A,A]
        fbge %fcc0, .BB0

        ! CHECK:             fbuge %fcc0, .BB0                    ! encoding: [0x19,0b01001AAA,A,A]
        fbuge %fcc0, .BB0

        ! CHECK:             fble %fcc0, .BB0                     ! encoding: [0x1b,0b01001AAA,A,A]
        fble %fcc0, .BB0

        ! CHECK:             fbule %fcc0, .BB0                    ! encoding: [0x1d,0b01001AAA,A,A]
        fbule %fcc0, .BB0

        ! CHECK:             fbo %fcc0, .BB0                      ! encoding: [0x1f,0b01001AAA,A,A]
        fbo %fcc0, .BB0

        ! CHECK:             fbu %fcc0, .BB0                      ! encoding: [0x0f,0b01001AAA,A,A]
        fbu,pt %fcc0, .BB0

        ! CHECK:             fbg %fcc0, .BB0                      ! encoding: [0x0d,0b01001AAA,A,A]
        fbg,pt %fcc0, .BB0
        ! CHECK:             fbug %fcc0, .BB0                     ! encoding: [0x0b,0b01001AAA,A,A]
        fbug,pt %fcc0, .BB0

        ! CHECK:             fbl %fcc0, .BB0                      ! encoding: [0x09,0b01001AAA,A,A]
        fbl,pt %fcc0, .BB0

        ! CHECK:             fbul %fcc0, .BB0                     ! encoding: [0x07,0b01001AAA,A,A]
        fbul,pt %fcc0, .BB0

        ! CHECK:             fblg %fcc0, .BB0                     ! encoding: [0x05,0b01001AAA,A,A]
        fblg,pt %fcc0, .BB0

        ! CHECK:             fbne %fcc0, .BB0                     ! encoding: [0x03,0b01001AAA,A,A]
        fbne,pt %fcc0, .BB0

        ! CHECK:             fbe %fcc0, .BB0                      ! encoding: [0x13,0b01001AAA,A,A]
        fbe,pt %fcc0, .BB0

        ! CHECK:             fbue %fcc0, .BB0                     ! encoding: [0x15,0b01001AAA,A,A]
        fbue,pt %fcc0, .BB0

        ! CHECK:             fbge %fcc0, .BB0                     ! encoding: [0x17,0b01001AAA,A,A]
        fbge,pt %fcc0, .BB0

        ! CHECK:             fbuge %fcc0, .BB0                    ! encoding: [0x19,0b01001AAA,A,A]
        fbuge,pt %fcc0, .BB0

        ! CHECK:             fble %fcc0, .BB0                     ! encoding: [0x1b,0b01001AAA,A,A]
        fble,pt %fcc0, .BB0

        ! CHECK:             fbule %fcc0, .BB0                    ! encoding: [0x1d,0b01001AAA,A,A]
        fbule,pt %fcc0, .BB0

        ! CHECK:             fbo %fcc0, .BB0                      ! encoding: [0x1f,0b01001AAA,A,A]
        fbo,pt %fcc0, .BB0


        ! CHECK:             fbo,a %fcc0, .BB0                      ! encoding: [0x3f,0b01001AAA,A,A]
        fbo,a %fcc0, .BB0

        ! CHECK:             fbu,a %fcc0, .BB0                      ! encoding: [0x2f,0b01001AAA,A,A]
        fbu,a %fcc0, .BB0

        ! CHECK:             fbg,a %fcc0, .BB0                      ! encoding: [0x2d,0b01001AAA,A,A]
        fbg,a %fcc0, .BB0
        ! CHECK:             fbug,a %fcc0, .BB0                     ! encoding: [0x2b,0b01001AAA,A,A]
        fbug,a %fcc0, .BB0

        ! CHECK:             fbl,a %fcc0, .BB0                      ! encoding: [0x29,0b01001AAA,A,A]
        fbl,a %fcc0, .BB0

        ! CHECK:             fbul,a %fcc0, .BB0                     ! encoding: [0x27,0b01001AAA,A,A]
        fbul,a %fcc0, .BB0

        ! CHECK:             fblg,a %fcc0, .BB0                     ! encoding: [0x25,0b01001AAA,A,A]
        fblg,a %fcc0, .BB0

        ! CHECK:             fbne,a %fcc0, .BB0                     ! encoding: [0x23,0b01001AAA,A,A]
        fbne,a %fcc0, .BB0

        ! CHECK:             fbe,a %fcc0, .BB0                      ! encoding: [0x33,0b01001AAA,A,A]
        fbe,a %fcc0, .BB0

        ! CHECK:             fbue,a %fcc0, .BB0                     ! encoding: [0x35,0b01001AAA,A,A]
        fbue,a %fcc0, .BB0

        ! CHECK:             fbge,a %fcc0, .BB0                     ! encoding: [0x37,0b01001AAA,A,A]
        fbge,a %fcc0, .BB0

        ! CHECK:             fbuge,a %fcc0, .BB0                    ! encoding: [0x39,0b01001AAA,A,A]
        fbuge,a %fcc0, .BB0

        ! CHECK:             fble,a %fcc0, .BB0                     ! encoding: [0x3b,0b01001AAA,A,A]
        fble,a %fcc0, .BB0

        ! CHECK:             fbule,a %fcc0, .BB0                    ! encoding: [0x3d,0b01001AAA,A,A]
        fbule,a %fcc0, .BB0

        ! CHECK:             fbo,a %fcc0, .BB0                      ! encoding: [0x3f,0b01001AAA,A,A]
        fbo,a %fcc0, .BB0

                ! CHECK:             fbo,a %fcc0, .BB0                      ! encoding: [0x3f,0b01001AAA,A,A]
        fbo,a %fcc0, .BB0

        ! CHECK:             fbu,a %fcc0, .BB0                      ! encoding: [0x2f,0b01001AAA,A,A]
        fbu,a,pt %fcc0, .BB0

        ! CHECK:             fbg,a %fcc0, .BB0                      ! encoding: [0x2d,0b01001AAA,A,A]
        fbg,a,pt %fcc0, .BB0

        ! CHECK:             fbug,a %fcc0, .BB0                     ! encoding: [0x2b,0b01001AAA,A,A]
        fbug,a,pt %fcc0, .BB0

        ! CHECK:             fbl,a %fcc0, .BB0                      ! encoding: [0x29,0b01001AAA,A,A]
        fbl,a,pt %fcc0, .BB0

        ! CHECK:             fbul,a %fcc0, .BB0                     ! encoding: [0x27,0b01001AAA,A,A]
        fbul,a,pt %fcc0, .BB0

        ! CHECK:             fblg,a %fcc0, .BB0                     ! encoding: [0x25,0b01001AAA,A,A]
        fblg,a,pt %fcc0, .BB0

        ! CHECK:             fbne,a %fcc0, .BB0                     ! encoding: [0x23,0b01001AAA,A,A]
        fbne,a,pt %fcc0, .BB0

        ! CHECK:             fbe,a %fcc0, .BB0                      ! encoding: [0x33,0b01001AAA,A,A]
        fbe,a,pt %fcc0, .BB0

        ! CHECK:             fbue,a %fcc0, .BB0                     ! encoding: [0x35,0b01001AAA,A,A]
        fbue,a,pt %fcc0, .BB0

        ! CHECK:             fbge,a %fcc0, .BB0                     ! encoding: [0x37,0b01001AAA,A,A]
        fbge,a,pt %fcc0, .BB0

        ! CHECK:             fbuge,a %fcc0, .BB0                    ! encoding: [0x39,0b01001AAA,A,A]
        fbuge,a,pt %fcc0, .BB0

        ! CHECK:             fble,a %fcc0, .BB0                     ! encoding: [0x3b,0b01001AAA,A,A]
        fble,a,pt %fcc0, .BB0

        ! CHECK:             fbule,a %fcc0, .BB0                    ! encoding: [0x3d,0b01001AAA,A,A]
        fbule,a,pt %fcc0, .BB0

        ! CHECK:             fbo,a %fcc0, .BB0                      ! encoding: [0x3f,0b01001AAA,A,A]
        fbo,a,pt %fcc0, .BB0

        ! CHECK:             fbu,pn %fcc0, .BB0                 ! encoding: [0x0f,0b01000AAA,A,A]
        fbu,pn %fcc0, .BB0

        ! CHECK:             fbg,pn %fcc0, .BB0                      ! encoding: [0x0d,0b01000AAA,A,A]
        fbg,pn %fcc0, .BB0
        ! CHECK:             fbug,pn %fcc0, .BB0                     ! encoding: [0x0b,0b01000AAA,A,A]
        fbug,pn %fcc0, .BB0

        ! CHECK:             fbl,pn %fcc0, .BB0                      ! encoding: [0x09,0b01000AAA,A,A]
        fbl,pn %fcc0, .BB0

        ! CHECK:             fbul,pn %fcc0, .BB0                     ! encoding: [0x07,0b01000AAA,A,A]
        fbul,pn %fcc0, .BB0

        ! CHECK:             fblg,pn %fcc0, .BB0                     ! encoding: [0x05,0b01000AAA,A,A]
        fblg,pn %fcc0, .BB0

        ! CHECK:             fbne,pn %fcc0, .BB0                     ! encoding: [0x03,0b01000AAA,A,A]
        fbne,pn %fcc0, .BB0

        ! CHECK:             fbe,pn %fcc0, .BB0                      ! encoding: [0x13,0b01000AAA,A,A]
        fbe,pn %fcc0, .BB0

        ! CHECK:             fbue,pn %fcc0, .BB0                     ! encoding: [0x15,0b01000AAA,A,A]
        fbue,pn %fcc0, .BB0

        ! CHECK:             fbge,pn %fcc0, .BB0                     ! encoding: [0x17,0b01000AAA,A,A]
        fbge,pn %fcc0, .BB0

        ! CHECK:             fbuge,pn %fcc0, .BB0                    ! encoding: [0x19,0b01000AAA,A,A]
        fbuge,pn %fcc0, .BB0

        ! CHECK:             fble,pn %fcc0, .BB0                     ! encoding: [0x1b,0b01000AAA,A,A]
        fble,pn %fcc0, .BB0

        ! CHECK:             fbule,pn %fcc0, .BB0                    ! encoding: [0x1d,0b01000AAA,A,A]
        fbule,pn %fcc0, .BB0

        ! CHECK:             fbo,pn %fcc0, .BB0                      ! encoding: [0x1f,0b01000AAA,A,A]
        fbo,pn %fcc0, .BB0

                ! CHECK:             fbu,a,pn %fcc0, .BB0                      ! encoding: [0x2f,0b01000AAA,A,A]
        fbu,a,pn %fcc0, .BB0

        ! CHECK:             fbg,a,pn %fcc0, .BB0                      ! encoding: [0x2d,0b01000AAA,A,A]
        fbg,a,pn %fcc0, .BB0

        ! CHECK:             fbug,a,pn %fcc0, .BB0                     ! encoding: [0x2b,0b01000AAA,A,A]
        fbug,a,pn %fcc0, .BB0

        ! CHECK:             fbl,a,pn %fcc0, .BB0                      ! encoding: [0x29,0b01000AAA,A,A]
        fbl,a,pn %fcc0, .BB0

        ! CHECK:             fbul,a,pn %fcc0, .BB0                     ! encoding: [0x27,0b01000AAA,A,A]
        fbul,a,pn %fcc0, .BB0

        ! CHECK:             fblg,a,pn %fcc0, .BB0                     ! encoding: [0x25,0b01000AAA,A,A]
        fblg,a,pn %fcc0, .BB0

        ! CHECK:             fbne,a,pn %fcc0, .BB0                     ! encoding: [0x23,0b01000AAA,A,A]
        fbne,a,pn %fcc0, .BB0

        ! CHECK:             fbe,a,pn %fcc0, .BB0                      ! encoding: [0x33,0b01000AAA,A,A]
        fbe,a,pn %fcc0, .BB0

        ! CHECK:             fbue,a,pn %fcc0, .BB0                     ! encoding: [0x35,0b01000AAA,A,A]
        fbue,a,pn %fcc0, .BB0

        ! CHECK:             fbge,a,pn %fcc0, .BB0                     ! encoding: [0x37,0b01000AAA,A,A]
        fbge,a,pn %fcc0, .BB0

        ! CHECK:             fbuge,a,pn %fcc0, .BB0                    ! encoding: [0x39,0b01000AAA,A,A]
        fbuge,a,pn %fcc0, .BB0

        ! CHECK:             fble,a,pn %fcc0, .BB0                     ! encoding: [0x3b,0b01000AAA,A,A]
        fble,a,pn %fcc0, .BB0

        ! CHECK:             fbule,a,pn %fcc0, .BB0                    ! encoding: [0x3d,0b01000AAA,A,A]
        fbule,a,pn %fcc0, .BB0

        ! CHECK:             fbo,a,pn %fcc0, .BB0                      ! encoding: [0x3f,0b01000AAA,A,A]
        fbo,a,pn %fcc0, .BB0

        ! CHECK: movu %fcc1, %g1, %g2            ! encoding: [0x85,0x61,0xc8,0x01]
        movu %fcc1, %g1, %g2

        ! CHECK: fmovsg %fcc2, %f1, %f2          ! encoding: [0x85,0xa9,0x90,0x21]
        fmovsg %fcc2, %f1, %f2

        ! CHECK:             fbug %fcc3, .BB0                ! encoding: [0x0b,0b01111AAA,A,A]
        fbug %fcc3, .BB0

        ! CHECK:             fbu %fcc3, .BB0                 ! encoding: [0x0f,0b01111AAA,A,A]
        fbu,pt %fcc3, .BB0

        ! CHECK:             fbl,a %fcc3, .BB0               ! encoding: [0x29,0b01111AAA,A,A]
        fbl,a %fcc3, .BB0

        ! CHECK:             fbue,pn %fcc3, .BB0             ! encoding: [0x15,0b01110AAA,A,A]
        fbue,pn %fcc3, .BB0

        ! CHECK:             fbne,a,pn %fcc3, .BB0           ! encoding: [0x23,0b01110AAA,A,A]
        fbne,a,pn %fcc3, .BB0


        ! CHECK:                brz %g1, .BB0                   ! encoding: [0x02'A',0xc8'A',0x40'A',A]
        ! CHECK:                brlez %g1, .BB0                 ! encoding: [0x04'A',0xc8'A',0x40'A',A]
        ! CHECK:                brlz %g1, .BB0                  ! encoding: [0x06'A',0xc8'A',0x40'A',A]
        ! CHECK:                brnz %g1, .BB0                  ! encoding: [0x0a'A',0xc8'A',0x40'A',A]
        ! CHECK:                brgz %g1, .BB0                  ! encoding: [0x0c'A',0xc8'A',0x40'A',A]
        ! CHECK:                brgez %g1, .BB0                 ! encoding: [0x0e'A',0xc8'A',0x40'A',A]

        brz   %g1, .BB0
        brlez %g1, .BB0
        brlz  %g1, .BB0
        brnz  %g1, .BB0
        brgz  %g1, .BB0
        brgez %g1, .BB0

        ! CHECK: brz %g1, .BB0                   ! encoding: [0x02'A',0xc8'A',0x40'A',A]
        brz,pt   %g1, .BB0

        ! CHECK: brz,a %g1, .BB0                 ! encoding: [0x22'A',0xc8'A',0x40'A',A]
        brz,a   %g1, .BB0

        ! CHECK: brz,a %g1, .BB0                 ! encoding: [0x22'A',0xc8'A',0x40'A',A]
        brz,a,pt   %g1, .BB0

        ! CHECK:  brz,pn %g1, .BB0               ! encoding: [0x02'A',0xc0'A',0x40'A',A]
        brz,pn   %g1, .BB0

        ! CHECK:  brz,a,pn %g1, .BB0              ! encoding: [0x22'A',0xc0'A',0x40'A',A]
        brz,a,pn   %g1, .BB0

        ! CHECK: movrz   %g1, %g2, %g3 ! encoding: [0x87,0x78,0x44,0x02]
        ! CHECK: movrz   %g1, %g2, %g3 ! encoding: [0x87,0x78,0x44,0x02]
        ! CHECK: movrlez %g1, %g2, %g3 ! encoding: [0x87,0x78,0x48,0x02]
        ! CHECK: movrlz  %g1, %g2, %g3 ! encoding: [0x87,0x78,0x4c,0x02]
        ! CHECK: movrnz  %g1, %g2, %g3 ! encoding: [0x87,0x78,0x54,0x02]
        ! CHECK: movrnz  %g1, %g2, %g3 ! encoding: [0x87,0x78,0x54,0x02]
        ! CHECK: movrgz  %g1, %g2, %g3 ! encoding: [0x87,0x78,0x58,0x02]
        ! CHECK: movrgez %g1, %g2, %g3 ! encoding: [0x87,0x78,0x5c,0x02]
        movrz   %g1, %g2, %g3
        movre   %g1, %g2, %g3
        movrlez %g1, %g2, %g3
        movrlz  %g1, %g2, %g3
        movrnz  %g1, %g2, %g3
        movrne  %g1, %g2, %g3
        movrgz  %g1, %g2, %g3
        movrgez %g1, %g2, %g3

        ! CHECK: movrz   %g1, 2, %g3 ! encoding: [0x87,0x78,0x64,0x02]
        ! CHECK: movrz   %g1, 2, %g3 ! encoding: [0x87,0x78,0x64,0x02]
        ! CHECK: movrlez %g1, 2, %g3 ! encoding: [0x87,0x78,0x68,0x02]
        ! CHECK: movrlz  %g1, 2, %g3 ! encoding: [0x87,0x78,0x6c,0x02]
        ! CHECK: movrnz  %g1, 2, %g3 ! encoding: [0x87,0x78,0x74,0x02]
        ! CHECK: movrnz  %g1, 2, %g3 ! encoding: [0x87,0x78,0x74,0x02]
        ! CHECK: movrgz  %g1, 2, %g3 ! encoding: [0x87,0x78,0x78,0x02]
        ! CHECK: movrgez %g1, 2, %g3 ! encoding: [0x87,0x78,0x7c,0x02]
        movrz   %g1, 2, %g3
        movre   %g1, 2, %g3
        movrlez %g1, 2, %g3
        movrlz  %g1, 2, %g3
        movrnz  %g1, 2, %g3
        movrne  %g1, 2, %g3
        movrgz  %g1, 2, %g3
        movrgez %g1, 2, %g3

        ! CHECK: fmovrsz %g1, %f2, %f3         ! encoding: [0x87,0xa8,0x44,0xa2]
        ! CHECK: fmovrslez %g1, %f2, %f3       ! encoding: [0x87,0xa8,0x48,0xa2]
        ! CHECK: fmovrslz %g1, %f2, %f3        ! encoding: [0x87,0xa8,0x4c,0xa2]
        ! CHECK: fmovrsnz %g1, %f2, %f3        ! encoding: [0x87,0xa8,0x54,0xa2]
        ! CHECK: fmovrsgz %g1, %f2, %f3        ! encoding: [0x87,0xa8,0x58,0xa2]
        ! CHECK: fmovrsgez %g1, %f2, %f3       ! encoding: [0x87,0xa8,0x5c,0xa2]
        fmovrsz   %g1, %f2, %f3
        fmovrslez %g1, %f2, %f3
        fmovrslz  %g1, %f2, %f3
        fmovrsnz  %g1, %f2, %f3
        fmovrsgz  %g1, %f2, %f3
        fmovrsgez %g1, %f2, %f3

        ! CHECK: fmovrdz %g1, %f2, %f4         ! encoding: [0x89,0xa8,0x44,0xc2]
        ! CHECK: fmovrdlez %g1, %f2, %f4       ! encoding: [0x89,0xa8,0x48,0xc2]
        ! CHECK: fmovrdlz %g1, %f2, %f4        ! encoding: [0x89,0xa8,0x4c,0xc2]
        ! CHECK: fmovrdnz %g1, %f2, %f4        ! encoding: [0x89,0xa8,0x54,0xc2]
        ! CHECK: fmovrdgz %g1, %f2, %f4        ! encoding: [0x89,0xa8,0x58,0xc2]
        ! CHECK: fmovrdgez %g1, %f2, %f4       ! encoding: [0x89,0xa8,0x5c,0xc2]
        fmovrdz   %g1, %f2, %f4
        fmovrdlez %g1, %f2, %f4
        fmovrdlz  %g1, %f2, %f4
        fmovrdnz  %g1, %f2, %f4
        fmovrdgz  %g1, %f2, %f4
        fmovrdgez %g1, %f2, %f4

        ! CHECK: fmovrqz %g1, %f4, %f8         ! encoding: [0x91,0xa8,0x44,0xe4]
        ! CHECK: fmovrqlez %g1, %f4, %f8       ! encoding: [0x91,0xa8,0x48,0xe4]
        ! CHECK: fmovrqlz %g1, %f4, %f8        ! encoding: [0x91,0xa8,0x4c,0xe4]
        ! CHECK: fmovrqnz %g1, %f4, %f8        ! encoding: [0x91,0xa8,0x54,0xe4]
        ! CHECK: fmovrqgz %g1, %f4, %f8        ! encoding: [0x91,0xa8,0x58,0xe4]
        ! CHECK: fmovrqgez %g1, %f4, %f8       ! encoding: [0x91,0xa8,0x5c,0xe4]
        fmovrqz   %g1, %f4, %f8
        fmovrqlez %g1, %f4, %f8
        fmovrqlz  %g1, %f4, %f8
        fmovrqnz  %g1, %f4, %f8
        fmovrqgz  %g1, %f4, %f8
        fmovrqgez %g1, %f4, %f8

        ! CHECK:  rett %i7+8   ! encoding: [0x81,0xcf,0xe0,0x08]
        return %i7 + 8

