! RUN: llvm-mc %s -triple=sparc64-unknown-linux-gnu -show-encoding | FileCheck %s


        ! CHECK: bne %xcc, .BB0     ! encoding: [0x12,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bne %xcc, .BB0

        ! CHECK: be %xcc, .BB0      ! encoding: [0x02,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        be %xcc, .BB0

        ! CHECK: bg %xcc, .BB0      ! encoding: [0x14,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bg %xcc, .BB0

        ! CHECK: ble %xcc, .BB0      ! encoding: [0x04,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        ble %xcc, .BB0

        ! CHECK: bge %xcc, .BB0      ! encoding: [0x16,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bge %xcc, .BB0

        ! CHECK: bl %xcc, .BB0      ! encoding: [0x06,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bl %xcc, .BB0

        ! CHECK: bgu %xcc, .BB0      ! encoding: [0x18,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bgu %xcc, .BB0

        ! CHECK: bleu %xcc, .BB0      ! encoding: [0x08,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bleu %xcc, .BB0

        ! CHECK: bcc %xcc, .BB0      ! encoding: [0x1a,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bcc %xcc, .BB0

        ! CHECK: bcs %xcc, .BB0      ! encoding: [0x0a,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bcs %xcc, .BB0

        ! CHECK: bpos %xcc, .BB0      ! encoding: [0x1c,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bpos %xcc, .BB0

        ! CHECK: bneg %xcc, .BB0      ! encoding: [0x0c,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bneg %xcc, .BB0

        ! CHECK: bvc %xcc, .BB0      ! encoding: [0x1e,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bvc %xcc, .BB0

        ! CHECK: bvs %xcc, .BB0      ! encoding: [0x0e,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
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
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bne,a %icc, .BB0

        ! CHECK: be,a %icc, .BB0      ! encoding: [0x22,0b01001AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        be,a %icc, .BB0

        ! CHECK: bg,a %icc, .BB0      ! encoding: [0x34,0b01001AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bg,a %icc, .BB0

        ! CHECK: ble,a %icc, .BB0      ! encoding: [0x24,0b01001AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        ble,a %icc, .BB0

        ! CHECK: bge,a %icc, .BB0      ! encoding: [0x36,0b01001AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bge,a %icc, .BB0

        ! CHECK: bl,a %icc, .BB0      ! encoding: [0x26,0b01001AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bl,a %icc, .BB0

        ! CHECK: bgu,a %icc, .BB0      ! encoding: [0x38,0b01001AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bgu,a %icc, .BB0

        ! CHECK: bleu,a %icc, .BB0      ! encoding: [0x28,0b01001AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bleu,a %icc, .BB0

        ! CHECK: bcc,a %icc, .BB0      ! encoding: [0x3a,0b01001AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bcc,a %icc, .BB0

        ! CHECK: bcs,a %icc, .BB0      ! encoding: [0x2a,0b01001AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bcs,a %icc, .BB0

        ! CHECK: bpos,a %icc, .BB0      ! encoding: [0x3c,0b01001AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bpos,a %icc, .BB0

        ! CHECK: bneg,a %icc, .BB0      ! encoding: [0x2c,0b01001AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bneg,a %icc, .BB0

        ! CHECK: bvc,a %icc, .BB0      ! encoding: [0x3e,0b01001AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bvc,a %icc, .BB0

        ! CHECK: bvs,a %icc, .BB0      ! encoding: [0x2e,0b01001AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bvs,a %icc, .BB0

        ! CHECK: bne,pn %icc, .BB0     ! encoding: [0x12,0b01000AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bne,pn %icc, .BB0

        ! CHECK: be,pn %icc, .BB0      ! encoding: [0x02,0b01000AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        be,pn %icc, .BB0

        ! CHECK: bg,pn %icc, .BB0      ! encoding: [0x14,0b01000AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bg,pn %icc, .BB0

        ! CHECK: ble,pn %icc, .BB0      ! encoding: [0x04,0b01000AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        ble,pn %icc, .BB0

        ! CHECK: bge,pn %icc, .BB0      ! encoding: [0x16,0b01000AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bge,pn %icc, .BB0

        ! CHECK: bl,pn %icc, .BB0      ! encoding: [0x06,0b01000AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bl,pn %icc, .BB0

        ! CHECK: bgu,pn %icc, .BB0      ! encoding: [0x18,0b01000AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bgu,pn %icc, .BB0

        ! CHECK: bleu,pn %icc, .BB0      ! encoding: [0x08,0b01000AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bleu,pn %icc, .BB0

        ! CHECK: bcc,pn %icc, .BB0      ! encoding: [0x1a,0b01000AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bcc,pn %icc, .BB0

        ! CHECK: bcs,pn %icc, .BB0      ! encoding: [0x0a,0b01000AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bcs,pn %icc, .BB0

        ! CHECK: bpos,pn %icc, .BB0      ! encoding: [0x1c,0b01000AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bpos,pn %icc, .BB0

        ! CHECK: bneg,pn %icc, .BB0      ! encoding: [0x0c,0b01000AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bneg,pn %icc, .BB0

        ! CHECK: bvc,pn %icc, .BB0      ! encoding: [0x1e,0b01000AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bvc,pn %icc, .BB0

        ! CHECK: bvs,pn %icc, .BB0      ! encoding: [0x0e,0b01000AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bvs,pn %icc, .BB0

        ! CHECK: bne,a,pn %icc, .BB0     ! encoding: [0x32,0b01000AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bne,a,pn %icc, .BB0

        ! CHECK: be,a,pn %icc, .BB0      ! encoding: [0x22,0b01000AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        be,a,pn %icc, .BB0

        ! CHECK: bg,a,pn %icc, .BB0      ! encoding: [0x34,0b01000AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bg,a,pn %icc, .BB0

        ! CHECK: ble,a,pn %icc, .BB0      ! encoding: [0x24,0b01000AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        ble,a,pn %icc, .BB0

        ! CHECK: bge,a,pn %icc, .BB0      ! encoding: [0x36,0b01000AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bge,a,pn %icc, .BB0

        ! CHECK: bl,a,pn %icc, .BB0      ! encoding: [0x26,0b01000AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bl,a,pn %icc, .BB0

        ! CHECK: bgu,a,pn %icc, .BB0      ! encoding: [0x38,0b01000AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bgu,a,pn %icc, .BB0

        ! CHECK: bleu,a,pn %icc, .BB0      ! encoding: [0x28,0b01000AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bleu,a,pn %icc, .BB0

        ! CHECK: bcc,a,pn %icc, .BB0      ! encoding: [0x3a,0b01000AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bcc,a,pn %icc, .BB0

        ! CHECK: bcs,a,pn %icc, .BB0      ! encoding: [0x2a,0b01000AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bcs,a,pn %icc, .BB0

        ! CHECK: bpos,a,pn %icc, .BB0      ! encoding: [0x3c,0b01000AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bpos,a,pn %icc, .BB0

        ! CHECK: bneg,a,pn %icc, .BB0      ! encoding: [0x2c,0b01000AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bneg,a,pn %icc, .BB0

        ! CHECK: bvc,a,pn %icc, .BB0      ! encoding: [0x3e,0b01000AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bvc,a,pn %icc, .BB0

        ! CHECK: bvs,a,pn %icc, .BB0      ! encoding: [0x2e,0b01000AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bvs,a,pn %icc, .BB0

        ! CHECK: bne %icc, .BB0     ! encoding: [0x12,0b01001AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bne,pt %icc, .BB0

        ! CHECK: be %icc, .BB0      ! encoding: [0x02,0b01001AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        be,pt %icc, .BB0

        ! CHECK: bg %icc, .BB0      ! encoding: [0x14,0b01001AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bg,pt %icc, .BB0

        ! CHECK: ble %icc, .BB0      ! encoding: [0x04,0b01001AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        ble,pt %icc, .BB0

        ! CHECK: bge %icc, .BB0      ! encoding: [0x16,0b01001AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bge,pt %icc, .BB0

        ! CHECK: bl %icc, .BB0      ! encoding: [0x06,0b01001AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bl,pt %icc, .BB0

        ! CHECK: bgu %icc, .BB0      ! encoding: [0x18,0b01001AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bgu,pt %icc, .BB0

        ! CHECK: bleu %icc, .BB0      ! encoding: [0x08,0b01001AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bleu,pt %icc, .BB0

        ! CHECK: bcc %icc, .BB0      ! encoding: [0x1a,0b01001AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bcc,pt %icc, .BB0

        ! CHECK: bcs %icc, .BB0      ! encoding: [0x0a,0b01001AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bcs,pt %icc, .BB0

        ! CHECK: bpos %icc, .BB0      ! encoding: [0x1c,0b01001AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bpos,pt %icc, .BB0

        ! CHECK: bneg %icc, .BB0      ! encoding: [0x0c,0b01001AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bneg,pt %icc, .BB0

        ! CHECK: bvc %icc, .BB0      ! encoding: [0x1e,0b01001AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bvc,pt %icc, .BB0

        ! CHECK: bvs %icc, .BB0      ! encoding: [0x0e,0b01001AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bvs,pt %icc, .BB0

        ! CHECK: bne,a %icc, .BB0     ! encoding: [0x32,0b01001AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bne,a,pt %icc, .BB0

        ! CHECK: be,a %icc, .BB0      ! encoding: [0x22,0b01001AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        be,a,pt %icc, .BB0

        ! CHECK: bg,a %icc, .BB0      ! encoding: [0x34,0b01001AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bg,a,pt %icc, .BB0

        ! CHECK: ble,a %icc, .BB0      ! encoding: [0x24,0b01001AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        ble,a,pt %icc, .BB0

        ! CHECK: bge,a %icc, .BB0      ! encoding: [0x36,0b01001AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bge,a,pt %icc, .BB0

        ! CHECK: bl,a %icc, .BB0      ! encoding: [0x26,0b01001AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bl,a,pt %icc, .BB0

        ! CHECK: bgu,a %icc, .BB0      ! encoding: [0x38,0b01001AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bgu,a,pt %icc, .BB0

        ! CHECK: bleu,a %icc, .BB0      ! encoding: [0x28,0b01001AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bleu,a,pt %icc, .BB0

        ! CHECK: bcc,a %icc, .BB0      ! encoding: [0x3a,0b01001AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bcc,a,pt %icc, .BB0

        ! CHECK: bcs,a %icc, .BB0      ! encoding: [0x2a,0b01001AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bcs,a,pt %icc, .BB0

        ! CHECK: bpos,a %icc, .BB0      ! encoding: [0x3c,0b01001AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bpos,a,pt %icc, .BB0


        ! CHECK: bne,a %xcc, .BB0     ! encoding: [0x32,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bne,a %xcc, .BB0

        ! CHECK: be,a %xcc, .BB0      ! encoding: [0x22,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        be,a %xcc, .BB0

        ! CHECK: bg,a %xcc, .BB0      ! encoding: [0x34,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bg,a %xcc, .BB0

        ! CHECK: ble,a %xcc, .BB0      ! encoding: [0x24,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        ble,a %xcc, .BB0

        ! CHECK: bge,a %xcc, .BB0      ! encoding: [0x36,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bge,a %xcc, .BB0

        ! CHECK: bl,a %xcc, .BB0      ! encoding: [0x26,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bl,a %xcc, .BB0

        ! CHECK: bgu,a %xcc, .BB0      ! encoding: [0x38,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bgu,a %xcc, .BB0

        ! CHECK: bleu,a %xcc, .BB0      ! encoding: [0x28,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bleu,a %xcc, .BB0

        ! CHECK: bcc,a %xcc, .BB0      ! encoding: [0x3a,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bcc,a %xcc, .BB0

        ! CHECK: bcs,a %xcc, .BB0      ! encoding: [0x2a,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bcs,a %xcc, .BB0

        ! CHECK: bpos,a %xcc, .BB0      ! encoding: [0x3c,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bpos,a %xcc, .BB0

        ! CHECK: bneg,a %xcc, .BB0      ! encoding: [0x2c,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bneg,a %xcc, .BB0

        ! CHECK: bvc,a %xcc, .BB0      ! encoding: [0x3e,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bvc,a %xcc, .BB0

        ! CHECK: bvs,a %xcc, .BB0      ! encoding: [0x2e,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bvs,a %xcc, .BB0

        ! CHECK: bne,pn %xcc, .BB0     ! encoding: [0x12,0b01100AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bne,pn %xcc, .BB0

        ! CHECK: be,pn %xcc, .BB0      ! encoding: [0x02,0b01100AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        be,pn %xcc, .BB0

        ! CHECK: bg,pn %xcc, .BB0      ! encoding: [0x14,0b01100AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bg,pn %xcc, .BB0

        ! CHECK: ble,pn %xcc, .BB0      ! encoding: [0x04,0b01100AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        ble,pn %xcc, .BB0

        ! CHECK: bge,pn %xcc, .BB0      ! encoding: [0x16,0b01100AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bge,pn %xcc, .BB0

        ! CHECK: bl,pn %xcc, .BB0      ! encoding: [0x06,0b01100AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bl,pn %xcc, .BB0

        ! CHECK: bgu,pn %xcc, .BB0      ! encoding: [0x18,0b01100AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bgu,pn %xcc, .BB0

        ! CHECK: bleu,pn %xcc, .BB0      ! encoding: [0x08,0b01100AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bleu,pn %xcc, .BB0

        ! CHECK: bcc,pn %xcc, .BB0      ! encoding: [0x1a,0b01100AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bcc,pn %xcc, .BB0

        ! CHECK: bcs,pn %xcc, .BB0      ! encoding: [0x0a,0b01100AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bcs,pn %xcc, .BB0

        ! CHECK: bpos,pn %xcc, .BB0      ! encoding: [0x1c,0b01100AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bpos,pn %xcc, .BB0

        ! CHECK: bneg,pn %xcc, .BB0      ! encoding: [0x0c,0b01100AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bneg,pn %xcc, .BB0

        ! CHECK: bvc,pn %xcc, .BB0      ! encoding: [0x1e,0b01100AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bvc,pn %xcc, .BB0

        ! CHECK: bvs,pn %xcc, .BB0      ! encoding: [0x0e,0b01100AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bvs,pn %xcc, .BB0

        ! CHECK: bne,a,pn %xcc, .BB0     ! encoding: [0x32,0b01100AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bne,a,pn %xcc, .BB0

        ! CHECK: be,a,pn %xcc, .BB0      ! encoding: [0x22,0b01100AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        be,a,pn %xcc, .BB0

        ! CHECK: bg,a,pn %xcc, .BB0      ! encoding: [0x34,0b01100AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bg,a,pn %xcc, .BB0

        ! CHECK: ble,a,pn %xcc, .BB0      ! encoding: [0x24,0b01100AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        ble,a,pn %xcc, .BB0

        ! CHECK: bge,a,pn %xcc, .BB0      ! encoding: [0x36,0b01100AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bge,a,pn %xcc, .BB0

        ! CHECK: bl,a,pn %xcc, .BB0      ! encoding: [0x26,0b01100AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bl,a,pn %xcc, .BB0

        ! CHECK: bgu,a,pn %xcc, .BB0      ! encoding: [0x38,0b01100AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bgu,a,pn %xcc, .BB0

        ! CHECK: bleu,a,pn %xcc, .BB0      ! encoding: [0x28,0b01100AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bleu,a,pn %xcc, .BB0

        ! CHECK: bcc,a,pn %xcc, .BB0      ! encoding: [0x3a,0b01100AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bcc,a,pn %xcc, .BB0

        ! CHECK: bcs,a,pn %xcc, .BB0      ! encoding: [0x2a,0b01100AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bcs,a,pn %xcc, .BB0

        ! CHECK: bpos,a,pn %xcc, .BB0      ! encoding: [0x3c,0b01100AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bpos,a,pn %xcc, .BB0

        ! CHECK: bneg,a,pn %xcc, .BB0      ! encoding: [0x2c,0b01100AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bneg,a,pn %xcc, .BB0

        ! CHECK: bvc,a,pn %xcc, .BB0      ! encoding: [0x3e,0b01100AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bvc,a,pn %xcc, .BB0

        ! CHECK: bvs,a,pn %xcc, .BB0      ! encoding: [0x2e,0b01100AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bvs,a,pn %xcc, .BB0

        ! CHECK: bne %xcc, .BB0     ! encoding: [0x12,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bne,pt %xcc, .BB0

        ! CHECK: be %xcc, .BB0      ! encoding: [0x02,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        be,pt %xcc, .BB0

        ! CHECK: bg %xcc, .BB0      ! encoding: [0x14,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bg,pt %xcc, .BB0

        ! CHECK: ble %xcc, .BB0      ! encoding: [0x04,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        ble,pt %xcc, .BB0

        ! CHECK: bge %xcc, .BB0      ! encoding: [0x16,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bge,pt %xcc, .BB0

        ! CHECK: bl %xcc, .BB0      ! encoding: [0x06,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bl,pt %xcc, .BB0

        ! CHECK: bgu %xcc, .BB0      ! encoding: [0x18,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bgu,pt %xcc, .BB0

        ! CHECK: bleu %xcc, .BB0      ! encoding: [0x08,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bleu,pt %xcc, .BB0

        ! CHECK: bcc %xcc, .BB0      ! encoding: [0x1a,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bcc,pt %xcc, .BB0

        ! CHECK: bcs %xcc, .BB0      ! encoding: [0x0a,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bcs,pt %xcc, .BB0

        ! CHECK: bpos %xcc, .BB0      ! encoding: [0x1c,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bpos,pt %xcc, .BB0

        ! CHECK: bneg %xcc, .BB0      ! encoding: [0x0c,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bneg,pt %xcc, .BB0

        ! CHECK: bvc %xcc, .BB0      ! encoding: [0x1e,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bvc,pt %xcc, .BB0

        ! CHECK: bvs %xcc, .BB0      ! encoding: [0x0e,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bvs,pt %xcc, .BB0

        ! CHECK: bne,a %xcc, .BB0     ! encoding: [0x32,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bne,a,pt %xcc, .BB0

        ! CHECK: be,a %xcc, .BB0      ! encoding: [0x22,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        be,a,pt %xcc, .BB0

        ! CHECK: bg,a %xcc, .BB0      ! encoding: [0x34,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bg,a,pt %xcc, .BB0

        ! CHECK: ble,a %xcc, .BB0      ! encoding: [0x24,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        ble,a,pt %xcc, .BB0

        ! CHECK: bge,a %xcc, .BB0      ! encoding: [0x36,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bge,a,pt %xcc, .BB0

        ! CHECK: bl,a %xcc, .BB0      ! encoding: [0x26,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bl,a,pt %xcc, .BB0

        ! CHECK: bgu,a %xcc, .BB0      ! encoding: [0x38,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bgu,a,pt %xcc, .BB0

        ! CHECK: bleu,a %xcc, .BB0      ! encoding: [0x28,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bleu,a,pt %xcc, .BB0

        ! CHECK: bcc,a %xcc, .BB0      ! encoding: [0x3a,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bcc,a,pt %xcc, .BB0

        ! CHECK: bcs,a %xcc, .BB0      ! encoding: [0x2a,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bcs,a,pt %xcc, .BB0

        ! CHECK: bpos,a %xcc, .BB0      ! encoding: [0x3c,0b01101AAA,A,A]
        ! CHECK-NEXT:         ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        bpos,a,pt %xcc, .BB0

        ! CHECK:             fba %fcc0, .BB0                        ! encoding: [0x11,0b01001AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fba %fcc0, .BB0

        ! CHECK:             fba %fcc0, .BB0                        ! encoding: [0x11,0b01001AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fb %fcc0, .BB0

        ! CHECK:             fbn %fcc0, .BB0                        ! encoding: [0x01,0b01001AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbn %fcc0, .BB0

        ! CHECK:             fbu %fcc0, .BB0                      ! encoding: [0x0f,0b01001AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbu %fcc0, .BB0

        ! CHECK:             fbg %fcc0, .BB0                      ! encoding: [0x0d,0b01001AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbg %fcc0, .BB0
        ! CHECK:             fbug %fcc0, .BB0                     ! encoding: [0x0b,0b01001AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbug %fcc0, .BB0

        ! CHECK:             fbl %fcc0, .BB0                      ! encoding: [0x09,0b01001AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbl %fcc0, .BB0

        ! CHECK:             fbul %fcc0, .BB0                     ! encoding: [0x07,0b01001AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbul %fcc0, .BB0

        ! CHECK:             fblg %fcc0, .BB0                     ! encoding: [0x05,0b01001AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fblg %fcc0, .BB0

        ! CHECK:             fbne %fcc0, .BB0                     ! encoding: [0x03,0b01001AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbne %fcc0, .BB0

        ! CHECK:             fbe %fcc0, .BB0                      ! encoding: [0x13,0b01001AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbe %fcc0, .BB0

        ! CHECK:             fbue %fcc0, .BB0                     ! encoding: [0x15,0b01001AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbue %fcc0, .BB0

        ! CHECK:             fbge %fcc0, .BB0                     ! encoding: [0x17,0b01001AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbge %fcc0, .BB0

        ! CHECK:             fbuge %fcc0, .BB0                    ! encoding: [0x19,0b01001AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbuge %fcc0, .BB0

        ! CHECK:             fble %fcc0, .BB0                     ! encoding: [0x1b,0b01001AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fble %fcc0, .BB0

        ! CHECK:             fbule %fcc0, .BB0                    ! encoding: [0x1d,0b01001AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbule %fcc0, .BB0

        ! CHECK:             fbo %fcc0, .BB0                      ! encoding: [0x1f,0b01001AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbo %fcc0, .BB0

        ! CHECK:             fbu %fcc0, .BB0                      ! encoding: [0x0f,0b01001AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbu,pt %fcc0, .BB0

        ! CHECK:             fbg %fcc0, .BB0                      ! encoding: [0x0d,0b01001AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbg,pt %fcc0, .BB0
        ! CHECK:             fbug %fcc0, .BB0                     ! encoding: [0x0b,0b01001AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbug,pt %fcc0, .BB0

        ! CHECK:             fbl %fcc0, .BB0                      ! encoding: [0x09,0b01001AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbl,pt %fcc0, .BB0

        ! CHECK:             fbul %fcc0, .BB0                     ! encoding: [0x07,0b01001AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbul,pt %fcc0, .BB0

        ! CHECK:             fblg %fcc0, .BB0                     ! encoding: [0x05,0b01001AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fblg,pt %fcc0, .BB0

        ! CHECK:             fbne %fcc0, .BB0                     ! encoding: [0x03,0b01001AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbne,pt %fcc0, .BB0

        ! CHECK:             fbe %fcc0, .BB0                      ! encoding: [0x13,0b01001AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbe,pt %fcc0, .BB0

        ! CHECK:             fbue %fcc0, .BB0                     ! encoding: [0x15,0b01001AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbue,pt %fcc0, .BB0

        ! CHECK:             fbge %fcc0, .BB0                     ! encoding: [0x17,0b01001AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbge,pt %fcc0, .BB0

        ! CHECK:             fbuge %fcc0, .BB0                    ! encoding: [0x19,0b01001AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbuge,pt %fcc0, .BB0

        ! CHECK:             fble %fcc0, .BB0                     ! encoding: [0x1b,0b01001AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fble,pt %fcc0, .BB0

        ! CHECK:             fbule %fcc0, .BB0                    ! encoding: [0x1d,0b01001AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbule,pt %fcc0, .BB0

        ! CHECK:             fbo %fcc0, .BB0                      ! encoding: [0x1f,0b01001AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbo,pt %fcc0, .BB0


        ! CHECK:             fbo,a %fcc0, .BB0                      ! encoding: [0x3f,0b01001AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbo,a %fcc0, .BB0

        ! CHECK:             fbu,a %fcc0, .BB0                      ! encoding: [0x2f,0b01001AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbu,a %fcc0, .BB0

        ! CHECK:             fbg,a %fcc0, .BB0                      ! encoding: [0x2d,0b01001AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbg,a %fcc0, .BB0
        ! CHECK:             fbug,a %fcc0, .BB0                     ! encoding: [0x2b,0b01001AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbug,a %fcc0, .BB0

        ! CHECK:             fbl,a %fcc0, .BB0                      ! encoding: [0x29,0b01001AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbl,a %fcc0, .BB0

        ! CHECK:             fbul,a %fcc0, .BB0                     ! encoding: [0x27,0b01001AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbul,a %fcc0, .BB0

        ! CHECK:             fblg,a %fcc0, .BB0                     ! encoding: [0x25,0b01001AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fblg,a %fcc0, .BB0

        ! CHECK:             fbne,a %fcc0, .BB0                     ! encoding: [0x23,0b01001AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbne,a %fcc0, .BB0

        ! CHECK:             fbe,a %fcc0, .BB0                      ! encoding: [0x33,0b01001AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbe,a %fcc0, .BB0

        ! CHECK:             fbue,a %fcc0, .BB0                     ! encoding: [0x35,0b01001AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbue,a %fcc0, .BB0

        ! CHECK:             fbge,a %fcc0, .BB0                     ! encoding: [0x37,0b01001AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbge,a %fcc0, .BB0

        ! CHECK:             fbuge,a %fcc0, .BB0                    ! encoding: [0x39,0b01001AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbuge,a %fcc0, .BB0

        ! CHECK:             fble,a %fcc0, .BB0                     ! encoding: [0x3b,0b01001AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fble,a %fcc0, .BB0

        ! CHECK:             fbule,a %fcc0, .BB0                    ! encoding: [0x3d,0b01001AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbule,a %fcc0, .BB0

        ! CHECK:             fbo,a %fcc0, .BB0                      ! encoding: [0x3f,0b01001AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbo,a %fcc0, .BB0

                ! CHECK:             fbo,a %fcc0, .BB0                      ! encoding: [0x3f,0b01001AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbo,a %fcc0, .BB0

        ! CHECK:             fbu,a %fcc0, .BB0                      ! encoding: [0x2f,0b01001AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbu,a,pt %fcc0, .BB0

        ! CHECK:             fbg,a %fcc0, .BB0                      ! encoding: [0x2d,0b01001AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbg,a,pt %fcc0, .BB0

        ! CHECK:             fbug,a %fcc0, .BB0                     ! encoding: [0x2b,0b01001AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbug,a,pt %fcc0, .BB0

        ! CHECK:             fbl,a %fcc0, .BB0                      ! encoding: [0x29,0b01001AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbl,a,pt %fcc0, .BB0

        ! CHECK:             fbul,a %fcc0, .BB0                     ! encoding: [0x27,0b01001AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbul,a,pt %fcc0, .BB0

        ! CHECK:             fblg,a %fcc0, .BB0                     ! encoding: [0x25,0b01001AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fblg,a,pt %fcc0, .BB0

        ! CHECK:             fbne,a %fcc0, .BB0                     ! encoding: [0x23,0b01001AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbne,a,pt %fcc0, .BB0

        ! CHECK:             fbe,a %fcc0, .BB0                      ! encoding: [0x33,0b01001AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbe,a,pt %fcc0, .BB0

        ! CHECK:             fbue,a %fcc0, .BB0                     ! encoding: [0x35,0b01001AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbue,a,pt %fcc0, .BB0

        ! CHECK:             fbge,a %fcc0, .BB0                     ! encoding: [0x37,0b01001AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbge,a,pt %fcc0, .BB0

        ! CHECK:             fbuge,a %fcc0, .BB0                    ! encoding: [0x39,0b01001AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbuge,a,pt %fcc0, .BB0

        ! CHECK:             fble,a %fcc0, .BB0                     ! encoding: [0x3b,0b01001AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fble,a,pt %fcc0, .BB0

        ! CHECK:             fbule,a %fcc0, .BB0                    ! encoding: [0x3d,0b01001AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbule,a,pt %fcc0, .BB0

        ! CHECK:             fbo,a %fcc0, .BB0                      ! encoding: [0x3f,0b01001AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbo,a,pt %fcc0, .BB0

        ! CHECK:             fbu,pn %fcc0, .BB0                 ! encoding: [0x0f,0b01000AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbu,pn %fcc0, .BB0

        ! CHECK:             fbg,pn %fcc0, .BB0                      ! encoding: [0x0d,0b01000AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbg,pn %fcc0, .BB0
        ! CHECK:             fbug,pn %fcc0, .BB0                     ! encoding: [0x0b,0b01000AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbug,pn %fcc0, .BB0

        ! CHECK:             fbl,pn %fcc0, .BB0                      ! encoding: [0x09,0b01000AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbl,pn %fcc0, .BB0

        ! CHECK:             fbul,pn %fcc0, .BB0                     ! encoding: [0x07,0b01000AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbul,pn %fcc0, .BB0

        ! CHECK:             fblg,pn %fcc0, .BB0                     ! encoding: [0x05,0b01000AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fblg,pn %fcc0, .BB0

        ! CHECK:             fbne,pn %fcc0, .BB0                     ! encoding: [0x03,0b01000AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbne,pn %fcc0, .BB0

        ! CHECK:             fbe,pn %fcc0, .BB0                      ! encoding: [0x13,0b01000AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbe,pn %fcc0, .BB0

        ! CHECK:             fbue,pn %fcc0, .BB0                     ! encoding: [0x15,0b01000AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbue,pn %fcc0, .BB0

        ! CHECK:             fbge,pn %fcc0, .BB0                     ! encoding: [0x17,0b01000AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbge,pn %fcc0, .BB0

        ! CHECK:             fbuge,pn %fcc0, .BB0                    ! encoding: [0x19,0b01000AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbuge,pn %fcc0, .BB0

        ! CHECK:             fble,pn %fcc0, .BB0                     ! encoding: [0x1b,0b01000AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fble,pn %fcc0, .BB0

        ! CHECK:             fbule,pn %fcc0, .BB0                    ! encoding: [0x1d,0b01000AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbule,pn %fcc0, .BB0

        ! CHECK:             fbo,pn %fcc0, .BB0                      ! encoding: [0x1f,0b01000AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbo,pn %fcc0, .BB0

                ! CHECK:             fbu,a,pn %fcc0, .BB0                      ! encoding: [0x2f,0b01000AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbu,a,pn %fcc0, .BB0

        ! CHECK:             fbg,a,pn %fcc0, .BB0                      ! encoding: [0x2d,0b01000AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbg,a,pn %fcc0, .BB0

        ! CHECK:             fbug,a,pn %fcc0, .BB0                     ! encoding: [0x2b,0b01000AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbug,a,pn %fcc0, .BB0

        ! CHECK:             fbl,a,pn %fcc0, .BB0                      ! encoding: [0x29,0b01000AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbl,a,pn %fcc0, .BB0

        ! CHECK:             fbul,a,pn %fcc0, .BB0                     ! encoding: [0x27,0b01000AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbul,a,pn %fcc0, .BB0

        ! CHECK:             fblg,a,pn %fcc0, .BB0                     ! encoding: [0x25,0b01000AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fblg,a,pn %fcc0, .BB0

        ! CHECK:             fbne,a,pn %fcc0, .BB0                     ! encoding: [0x23,0b01000AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbne,a,pn %fcc0, .BB0

        ! CHECK:             fbe,a,pn %fcc0, .BB0                      ! encoding: [0x33,0b01000AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbe,a,pn %fcc0, .BB0

        ! CHECK:             fbue,a,pn %fcc0, .BB0                     ! encoding: [0x35,0b01000AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbue,a,pn %fcc0, .BB0

        ! CHECK:             fbge,a,pn %fcc0, .BB0                     ! encoding: [0x37,0b01000AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbge,a,pn %fcc0, .BB0

        ! CHECK:             fbuge,a,pn %fcc0, .BB0                    ! encoding: [0x39,0b01000AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbuge,a,pn %fcc0, .BB0

        ! CHECK:             fble,a,pn %fcc0, .BB0                     ! encoding: [0x3b,0b01000AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fble,a,pn %fcc0, .BB0

        ! CHECK:             fbule,a,pn %fcc0, .BB0                    ! encoding: [0x3d,0b01000AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbule,a,pn %fcc0, .BB0

        ! CHECK:             fbo,a,pn %fcc0, .BB0                      ! encoding: [0x3f,0b01000AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbo,a,pn %fcc0, .BB0

        ! CHECK: movu %fcc1, %g1, %g2            ! encoding: [0x85,0x61,0xc8,0x01]
        movu %fcc1, %g1, %g2

        ! CHECK: fmovsg %fcc2, %f1, %f2          ! encoding: [0x85,0xa9,0x90,0x21]
        fmovsg %fcc2, %f1, %f2

        ! CHECK:             fbug %fcc3, .BB0                ! encoding: [0x0b,0b01111AAA,A,A]
        ! CHECK-NEXT:                                        ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbug %fcc3, .BB0

        ! CHECK:             fbu %fcc3, .BB0                 ! encoding: [0x0f,0b01111AAA,A,A]
        ! CHECK-NEXT:                                        ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbu,pt %fcc3, .BB0

        ! CHECK:             fbl,a %fcc3, .BB0               ! encoding: [0x29,0b01111AAA,A,A]
        ! CHECK-NEXT:                                        ! fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbl,a %fcc3, .BB0

        ! CHECK:             fbue,pn %fcc3, .BB0             ! encoding: [0x15,0b01110AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbue,pn %fcc3, .BB0

        ! CHECK:             fbne,a,pn %fcc3, .BB0           ! encoding: [0x23,0b01110AAA,A,A]
        ! CHECK-NEXT:                                        !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br19
        fbne,a,pn %fcc3, .BB0


        ! CHECK:                brz %g1, .BB0                   ! encoding: [0x02,0b11AA1000,0b01BBBBBB,B]
        ! CHECK-NEXT:                                           !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br16_2
        ! CHECK-NEXT:                                           !   fixup B - offset: 0, value: .BB0, kind: fixup_sparc_br16_14
        ! CHECK:                brlez %g1, .BB0                 ! encoding: [0x04,0b11AA1000,0b01BBBBBB,B]
        ! CHECK-NEXT:                                           !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br16_2
        ! CHECK-NEXT:                                           !   fixup B - offset: 0, value: .BB0, kind: fixup_sparc_br16_14
        ! CHECK:                brlz %g1, .BB0                  ! encoding: [0x06,0b11AA1000,0b01BBBBBB,B]
        ! CHECK-NEXT:                                           !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br16_2
        ! CHECK-NEXT:                                           !   fixup B - offset: 0, value: .BB0, kind: fixup_sparc_br16_14
        ! CHECK:                brnz %g1, .BB0                  ! encoding: [0x0a,0b11AA1000,0b01BBBBBB,B]
        ! CHECK-NEXT:                                           !  fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br16_2
        ! CHECK-NEXT:                                           !   fixup B - offset: 0, value: .BB0, kind: fixup_sparc_br16_14
        ! CHECK:                brgz %g1, .BB0                  ! encoding: [0x0c,0b11AA1000,0b01BBBBBB,B]
        ! CHECK-NEXT:                                           !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br16_2
        ! CHECK-NEXT:                                           !   fixup B - offset: 0, value: .BB0, kind: fixup_sparc_br16_14
        ! CHECK:                brgez %g1, .BB0                 ! encoding: [0x0e,0b11AA1000,0b01BBBBBB,B]
        ! CHECK-NEXT:                                           !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br16_2
        ! CHECK-NEXT:                                           !   fixup B - offset: 0, value: .BB0, kind: fixup_sparc_br16_14

        brz   %g1, .BB0
        brlez %g1, .BB0
        brlz  %g1, .BB0
        brnz  %g1, .BB0
        brgz  %g1, .BB0
        brgez %g1, .BB0

        ! CHECK: brz %g1, .BB0                   ! encoding: [0x02,0b11AA1000,0b01BBBBBB,B]
        ! CHECK-NEXT:                            !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br16_2
        ! CHECK-NEXT:                            !   fixup B - offset: 0, value: .BB0, kind: fixup_sparc_br16_14
        brz,pt   %g1, .BB0

        ! CHECK: brz,a %g1, .BB0                 ! encoding: [0x22,0b11AA1000,0b01BBBBBB,B]
        ! CHECK-NEXT:                            !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br16_2
        ! CHECK-NEXT:                            !   fixup B - offset: 0, value: .BB0, kind: fixup_sparc_br16_14
        brz,a   %g1, .BB0

        ! CHECK: brz,a %g1, .BB0                 ! encoding: [0x22,0b11AA1000,0b01BBBBBB,B]
        ! CHECK-NEXT:                            !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br16_2
        ! CHECK-NEXT:                            !   fixup B - offset: 0, value: .BB0, kind: fixup_sparc_br16_14
        brz,a,pt   %g1, .BB0

        ! CHECK:  brz,pn %g1, .BB0               ! encoding: [0x02,0b11AA0000,0b01BBBBBB,B]
        ! CHECK-NEXT:                            !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br16_2
        ! CHECK-NEXT:                            !   fixup B - offset: 0, value: .BB0, kind: fixup_sparc_br16_14
        brz,pn   %g1, .BB0

        ! CHECK:  brz,a,pn %g1, .BB0              ! encoding: [0x22,0b11AA0000,0b01BBBBBB,B]
        ! CHECK-NEXT:                             !   fixup A - offset: 0, value: .BB0, kind: fixup_sparc_br16_2
        ! CHECK-NEXT:                             !   fixup B - offset: 0, value: .BB0, kind: fixup_sparc_br16_14
        brz,a,pn   %g1, .BB0

        ! CHECK: movrz   %g1, %g2, %g3 ! encoding: [0x87,0x78,0x44,0x02]
        ! CHECK: movrlez %g1, %g2, %g3 ! encoding: [0x87,0x78,0x48,0x02]
        ! CHECK: movrlz  %g1, %g2, %g3 ! encoding: [0x87,0x78,0x4c,0x02]
        ! CHECK: movrnz  %g1, %g2, %g3 ! encoding: [0x87,0x78,0x54,0x02]
        ! CHECK: movrgz  %g1, %g2, %g3 ! encoding: [0x87,0x78,0x58,0x02]
        ! CHECK: movrgez %g1, %g2, %g3 ! encoding: [0x87,0x78,0x5c,0x02]
        movrz   %g1, %g2, %g3
        movrlez %g1, %g2, %g3
        movrlz  %g1, %g2, %g3
        movrnz  %g1, %g2, %g3
        movrgz  %g1, %g2, %g3
        movrgez %g1, %g2, %g3

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

        ! CHECK:  rett %i7+8   ! encoding: [0x81,0xcf,0xe0,0x08]
        return %i7 + 8

        ! CHECK: ta %icc, %g0 + 5               ! encoding: [0x91,0xd0,0x20,0x05]
        ta 5

        ! CHECK: te %xcc, %g0 + 3               ! encoding: [0x83,0xd0,0x30,0x03]
        te %xcc, 3

