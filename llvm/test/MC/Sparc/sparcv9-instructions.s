! RUN: not llvm-mc %s -arch=sparc   -show-encoding 2>&1 | FileCheck %s --check-prefix=V8
! RUN: llvm-mc %s -arch=sparcv9 -show-encoding | FileCheck %s --check-prefix=V9

        ! V8:      error: invalid instruction mnemonic
        ! V8-NEXT: addc %g2, %g1, %g3
        ! V9:      addx %g2, %g1, %g3              ! encoding: [0x86,0x40,0x80,0x01]
        addc %g2, %g1, %g3

        ! V8:      error: invalid instruction mnemonic
        ! V8-NEXT: addccc %g1, %g2, %g3
        ! V9:      addxcc %g1, %g2, %g3            ! encoding: [0x86,0xc0,0x40,0x02]
        addccc %g1, %g2, %g3

        ! V8:      error: invalid instruction mnemonic
        ! V8-NEXT: subc %g2, %g1, %g3
        ! V9:      subx %g2, %g1, %g3          ! encoding: [0x86,0x60,0x80,0x01]
        subc %g2, %g1, %g3

        ! V8:      error: invalid instruction mnemonic
        ! V8-NEXT: subccc %g1, %g2, %g3
        ! V9:      subxcc %g1, %g2, %g3         ! encoding: [0x86,0xe0,0x40,0x02]
        subccc %g1, %g2, %g3

        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: popc %g1, %g2
        ! V9:      popc %g1, %g2                ! encoding: [0x85,0x70,0x00,0x01]
        popc %g1, %g2


        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: signx %g1, %g2
        ! V9: sra %g1, %g0, %g2               ! encoding: [0x85,0x38,0x40,0x00]
        signx %g1, %g2
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: signx %g1
        ! V9: sra %g1, %g0, %g1               ! encoding: [0x83,0x38,0x40,0x00]
        signx %g1

        ! V8:      error: invalid instruction mnemonic
        ! V8-NEXT: lduw [%i0 + %l6], %o2
        ! V9: ld [%i0+%l6], %o2    ! encoding: [0xd4,0x06,0x00,0x16]
        lduw [%i0 + %l6], %o2
        ! V8:      error: invalid instruction mnemonic
        ! V8-NEXT: lduw [%i0 + 32], %o2
        ! V9: ld [%i0+32], %o2     ! encoding: [0xd4,0x06,0x20,0x20]
        lduw [%i0 + 32], %o2
        ! V8:      error: invalid instruction mnemonic
        ! V8-NEXT: lduw [%g1], %o2
        ! V9: ld [%g1], %o2        ! encoding: [0xd4,0x00,0x40,0x00]
        lduw [%g1], %o2
        ! V8:      error: invalid instruction mnemonic
        ! V8-NEXT: lduwa [%i0 + %l6] 131, %o2
        ! V9: lda [%i0+%l6] #ASI_SNF, %o2 ! encoding: [0xd4,0x86,0x10,0x76]
        lduwa [%i0 + %l6] 131, %o2
        ! V8:      error: invalid instruction mnemonic
        ! V8-NEXT: lduwa [%i0 + %l6] (130+1), %o2
        ! V9: lda [%i0+%l6] #ASI_SNF, %o2 ! encoding: [0xd4,0x86,0x10,0x76]
        lduwa [%i0 + %l6] (130+1), %o2

        ! V9: ldsw [%i0+%l6], %o2    ! encoding: [0xd4,0x46,0x00,0x16]
        ldsw [%i0 + %l6], %o2
        ! V9: ldsw [%i0+32], %o2     ! encoding: [0xd4,0x46,0x20,0x20]
        ldsw [%i0 + 32], %o2
        ! V9: ldsw [%g1], %o2        ! encoding: [0xd4,0x40,0x40,0x00]
        ldsw [%g1], %o2
        ! V9: ldswa [%i0+%l6] #ASI_SNF, %o2 ! encoding: [0xd4,0xc6,0x10,0x76]
        ldswa [%i0 + %l6] 131, %o2
        ! V9: ldswa [%i0+%l6] #ASI_SNF, %o2 ! encoding: [0xd4,0xc6,0x10,0x76]
        ldswa [%i0 + %l6] (130+1), %o2

        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: lda [%l0] 0xf0, %f29
        ! V9: lda [%l0] 240, %f29             ! encoding: [0xfb,0x84,0x1e,0x00]
        lda [%l0] 0xf0, %f29
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: lda [%l0] (0xef+0x01), %f29
        ! V9: lda [%l0] 240, %f29             ! encoding: [0xfb,0x84,0x1e,0x00]
        lda [%l0] (0xef+0x01), %f29

        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: ldda [%l0] 0xf0, %f48
        ! V9: ldda [%l0] 240, %f48            ! encoding: [0xe3,0x9c,0x1e,0x00]
        ldda [%l0] 0xf0, %f48
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: ldda [%l0] (0xef+0x01), %f48
        ! V9: ldda [%l0] 240, %f48            ! encoding: [0xe3,0x9c,0x1e,0x00]
        ldda [%l0] (0xef+0x01), %f48

        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: ldqa [%l0] 0xf0, %f48
        ! V9: ldqa [%l0] 240, %f48            ! encoding: [0xe3,0x94,0x1e,0x00]
        ldqa [%l0] 0xf0, %f48
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: ldqa [%l0] (0xef+0x01), %f48
        ! V9: ldqa [%l0] 240, %f48            ! encoding: [0xe3,0x94,0x1e,0x00]
        ldqa [%l0] (0xef+0x01), %f48
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: ldq [%l0], %f48
        ! V9: ldq [%l0], %f48                 ! encoding: [0xe3,0x14,0x00,0x00]
        ldq [%l0], %f48


        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: sta %f29, [%l0] 0xf0
        ! V9: sta %f29, [%l0] 240             ! encoding: [0xfb,0xa4,0x1e,0x00]
        sta %f29, [%l0] 0xf0
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: sta %f29, [%l0] (0xef+0x01)
        ! V9: sta %f29, [%l0] 240             ! encoding: [0xfb,0xa4,0x1e,0x00]
        sta %f29, [%l0] (0xef+0x01)

        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: stda %f48, [%l0] 0xf0
        ! V9: stda %f48, [%l0] 240            ! encoding: [0xe3,0xbc,0x1e,0x00]
        stda %f48, [%l0] 0xf0
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: stda %f48, [%l0] (0xef+0x01)
        ! V9: stda %f48, [%l0] 240            ! encoding: [0xe3,0xbc,0x1e,0x00]
        stda %f48, [%l0] (0xef+0x01)

        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: stqa %f48, [%l0] 0xf0
        ! V9: stqa %f48, [%l0] 240            ! encoding: [0xe3,0xb4,0x1e,0x00]
        stqa %f48, [%l0] 0xf0
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: stqa %f48, [%l0] (0xef+0x01)
        ! V9: stqa %f48, [%l0] 240            ! encoding: [0xe3,0xb4,0x1e,0x00]
        stqa %f48, [%l0] (0xef+0x01)
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: stq %f48, [%l0]
        ! V9: stq %f48, [%l0]                 ! encoding: [0xe3,0x34,0x00,0x00]
        stq %f48, [%l0]

        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: ldx [%g2 + 20],%fsr
        ! V9: ldx [%g2+20], %fsr    ! encoding: [0xc3,0x08,0xa0,0x14]
        ldx [%g2 + 20],%fsr

        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: ldx [%g2 + %i5],%fsr
        ! V9: ldx [%g2+%i5], %fsr   ! encoding: [0xc3,0x08,0x80,0x1d]
        ldx [%g2 + %i5],%fsr

        ! V9: ldxa [%g2+%i5] #ASI_SNF, %g0   ! encoding: [0xc0,0xd8,0x90,0x7d]
        ldxa [%g2 + %i5] 131, %g0
        ! V9: ldxa [%g2+%i5] #ASI_SNF, %g0   ! encoding: [0xc0,0xd8,0x90,0x7d]
        ldxa [%g2 + %i5] (130+1), %g0

        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: stx %fsr,[%g2 + 20]
        ! V9: stx %fsr, [%g2+20]    ! encoding: [0xc3,0x28,0xa0,0x14]
        stx %fsr,[%g2 + 20]

        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: stx %fsr,[%g2 + %i5]
        ! V9: stx %fsr, [%g2+%i5]   ! encoding: [0xc3,0x28,0x80,0x1d]
        stx %fsr,[%g2 + %i5]

        ! V9: stxa %g0, [%g2+%i5] #ASI_SNF   ! encoding: [0xc0,0xf0,0x90,0x7d]
        stxa %g0, [%g2 + %i5] 131
        ! V9: stxa %g0, [%g2+%i5] #ASI_SNF   ! encoding: [0xc0,0xf0,0x90,0x7d]
        stxa %g0, [%g2 + %i5] (130+1)

        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: wrpr %g6,%i6,%tpc
        ! V9: wrpr %g6, %fp, %tpc        ! encoding: [0x81,0x91,0x80,0x1e]
        wrpr %g6,%i6,%tpc
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: wrpr %g6,%i6,%tnpc
        ! V9: wrpr %g6, %fp, %tnpc       ! encoding: [0x83,0x91,0x80,0x1e]
        wrpr %g6,%i6,%tnpc
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: wrpr %g6,%i6,%tstate
        ! V9: wrpr %g6, %fp, %tstate     ! encoding: [0x85,0x91,0x80,0x1e]
        wrpr %g6,%i6,%tstate
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: wrpr %g6,%i6,%tt
        ! V9: wrpr %g6, %fp, %tt         ! encoding: [0x87,0x91,0x80,0x1e]
        wrpr %g6,%i6,%tt
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: wrpr %g6,%i6,%tick
        ! V9: wrpr %g6, %fp, %tick       ! encoding: [0x89,0x91,0x80,0x1e]
        wrpr %g6,%i6,%tick
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: wrpr %g6,%i6,%tba
        ! V9: wrpr %g6, %fp, %tba        ! encoding: [0x8b,0x91,0x80,0x1e]
        wrpr %g6,%i6,%tba
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: wrpr %g6,%i6,%pstate
        ! V9: wrpr %g6, %fp, %pstate     ! encoding: [0x8d,0x91,0x80,0x1e]
        wrpr %g6,%i6,%pstate
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: wrpr %g6,%i6,%tl
        ! V9: wrpr %g6, %fp, %tl         ! encoding: [0x8f,0x91,0x80,0x1e]
        wrpr %g6,%i6,%tl
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: wrpr %g6,%i6,%pil
        ! V9: wrpr %g6, %fp, %pil        ! encoding: [0x91,0x91,0x80,0x1e]
        wrpr %g6,%i6,%pil
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: wrpr %g6,%i6,%cwp
        ! V9: wrpr %g6, %fp, %cwp        ! encoding: [0x93,0x91,0x80,0x1e]
        wrpr %g6,%i6,%cwp
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: wrpr %g6,%i6,%cansave
        ! V9: wrpr %g6, %fp, %cansave    ! encoding: [0x95,0x91,0x80,0x1e]
        wrpr %g6,%i6,%cansave
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: wrpr %g6,%i6,%canrestore
        ! V9: wrpr %g6, %fp, %canrestore ! encoding: [0x97,0x91,0x80,0x1e]
        wrpr %g6,%i6,%canrestore
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: wrpr %g6,%i6,%cleanwin
        ! V9: wrpr %g6, %fp, %cleanwin   ! encoding: [0x99,0x91,0x80,0x1e]
        wrpr %g6,%i6,%cleanwin
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: wrpr %g6,%i6,%otherwin
        ! V9: wrpr %g6, %fp, %otherwin   ! encoding: [0x9b,0x91,0x80,0x1e]
        wrpr %g6,%i6,%otherwin
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: wrpr %g6,%i6,%wstate
        ! V9: wrpr %g6, %fp, %wstate     ! encoding: [0x9d,0x91,0x80,0x1e]
        wrpr %g6,%i6,%wstate

        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: wrpr %g6,255,%tpc
        ! V9: wrpr %g6, 255, %tpc        ! encoding: [0x81,0x91,0xa0,0xff]
        wrpr %g6,255,%tpc
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: wrpr %g6,255,%tnpc
        ! V9: wrpr %g6, 255, %tnpc       ! encoding: [0x83,0x91,0xa0,0xff]
        wrpr %g6,255,%tnpc
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: wrpr %g6,255,%tstate
        ! V9: wrpr %g6, 255, %tstate     ! encoding: [0x85,0x91,0xa0,0xff]
        wrpr %g6,255,%tstate
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: wrpr %g6,255,%tt
        ! V9: wrpr %g6, 255, %tt         ! encoding: [0x87,0x91,0xa0,0xff]
        wrpr %g6,255,%tt
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: wrpr %g6,255,%tick
        ! V9: wrpr %g6, 255, %tick       ! encoding: [0x89,0x91,0xa0,0xff]
        wrpr %g6,255,%tick
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: wrpr %g6,255,%tba
        ! V9: wrpr %g6, 255, %tba        ! encoding: [0x8b,0x91,0xa0,0xff]
        wrpr %g6,255,%tba
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: wrpr %g6,255,%pstate
        ! V9: wrpr %g6, 255, %pstate     ! encoding: [0x8d,0x91,0xa0,0xff]
        wrpr %g6,255,%pstate
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: wrpr %g6,255,%tl
        ! V9: wrpr %g6, 255, %tl         ! encoding: [0x8f,0x91,0xa0,0xff]
        wrpr %g6,255,%tl
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: wrpr %g6,255,%pil
        ! V9: wrpr %g6, 255, %pil        ! encoding: [0x91,0x91,0xa0,0xff]
        wrpr %g6,255,%pil
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: wrpr %g6,255,%cwp
        ! V9: wrpr %g6, 255, %cwp        ! encoding: [0x93,0x91,0xa0,0xff]
        wrpr %g6,255,%cwp
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: wrpr %g6,255,%cansave
        ! V9: wrpr %g6, 255, %cansave    ! encoding: [0x95,0x91,0xa0,0xff]
        wrpr %g6,255,%cansave
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: wrpr %g6,255,%canrestore
        ! V9: wrpr %g6, 255, %canrestore ! encoding: [0x97,0x91,0xa0,0xff]
        wrpr %g6,255,%canrestore
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: wrpr %g6,255,%cleanwin
        ! V9: wrpr %g6, 255, %cleanwin   ! encoding: [0x99,0x91,0xa0,0xff]
        wrpr %g6,255,%cleanwin
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: wrpr %g6,255,%otherwin
        ! V9: wrpr %g6, 255, %otherwin   ! encoding: [0x9b,0x91,0xa0,0xff]
        wrpr %g6,255,%otherwin
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: wrpr %g6,255,%wstate
        ! V9: wrpr %g6, 255, %wstate     ! encoding: [0x9d,0x91,0xa0,0xff]
        wrpr %g6,255,%wstate

        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: wrpr %i6,%tpc
        ! V9: wrpr %fp, %g0, %tpc        ! encoding: [0x81,0x97,0x80,0x00]
        wrpr %i6,%tpc
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: wrpr %i6,%tnpc
        ! V9: wrpr %fp, %g0, %tnpc       ! encoding: [0x83,0x97,0x80,0x00]
        wrpr %i6,%tnpc
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: wrpr %i6,%tstate
        ! V9: wrpr %fp, %g0, %tstate     ! encoding: [0x85,0x97,0x80,0x00]
        wrpr %i6,%tstate
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: wrpr %i6,%tt
        ! V9: wrpr %fp, %g0, %tt         ! encoding: [0x87,0x97,0x80,0x00]
        wrpr %i6,%tt
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: wrpr %i6,%tick
        ! V9: wrpr %fp, %g0, %tick       ! encoding: [0x89,0x97,0x80,0x00]
        wrpr %i6,%tick
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: wrpr %i6,%tba
        ! V9: wrpr %fp, %g0, %tba        ! encoding: [0x8b,0x97,0x80,0x00]
        wrpr %i6,%tba
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: wrpr %i6,%pstate
        ! V9: wrpr %fp, %g0, %pstate     ! encoding: [0x8d,0x97,0x80,0x00]
        wrpr %i6,%pstate
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: wrpr %i6,%tl
        ! V9: wrpr %fp, %g0, %tl         ! encoding: [0x8f,0x97,0x80,0x00]
        wrpr %i6,%tl
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: wrpr %i6,%pil
        ! V9: wrpr %fp, %g0, %pil        ! encoding: [0x91,0x97,0x80,0x00]
        wrpr %i6,%pil
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: wrpr %i6,%cwp
        ! V9: wrpr %fp, %g0, %cwp        ! encoding: [0x93,0x97,0x80,0x00]
        wrpr %i6,%cwp
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: wrpr %i6,%cansave
        ! V9: wrpr %fp, %g0, %cansave    ! encoding: [0x95,0x97,0x80,0x00]
        wrpr %i6,%cansave
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: wrpr %i6,%canrestore
        ! V9: wrpr %fp, %g0, %canrestore ! encoding: [0x97,0x97,0x80,0x00]
        wrpr %i6,%canrestore
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: wrpr %i6,%cleanwin
        ! V9: wrpr %fp, %g0, %cleanwin   ! encoding: [0x99,0x97,0x80,0x00]
        wrpr %i6,%cleanwin
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: wrpr %i6,%otherwin
        ! V9: wrpr %fp, %g0, %otherwin   ! encoding: [0x9b,0x97,0x80,0x00]
        wrpr %i6,%otherwin
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: wrpr %i6,%wstate
        ! V9: wrpr %fp, %g0, %wstate     ! encoding: [0x9d,0x97,0x80,0x00]
        wrpr %i6,%wstate

        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: wrpr 255,%tpc
        ! V9: wrpr %g0, 255, %tpc        ! encoding: [0x81,0x90,0x20,0xff]
        wrpr 255,%tpc
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: wrpr 255,%tnpc
        ! V9: wrpr %g0, 255, %tnpc       ! encoding: [0x83,0x90,0x20,0xff]
        wrpr 255,%tnpc
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: wrpr 255,%tstate
        ! V9: wrpr %g0, 255, %tstate     ! encoding: [0x85,0x90,0x20,0xff]
        wrpr 255,%tstate
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: wrpr 255,%tt
        ! V9: wrpr %g0, 255, %tt         ! encoding: [0x87,0x90,0x20,0xff]
        wrpr 255,%tt
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: wrpr 255,%tick
        ! V9: wrpr %g0, 255, %tick       ! encoding: [0x89,0x90,0x20,0xff]
        wrpr 255,%tick
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: wrpr 255,%tba
        ! V9: wrpr %g0, 255, %tba        ! encoding: [0x8b,0x90,0x20,0xff]
        wrpr 255,%tba
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: wrpr 255,%pstate
        ! V9: wrpr %g0, 255, %pstate     ! encoding: [0x8d,0x90,0x20,0xff]
        wrpr 255,%pstate
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: wrpr 255,%tl
        ! V9: wrpr %g0, 255, %tl         ! encoding: [0x8f,0x90,0x20,0xff]
        wrpr 255,%tl
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: wrpr 255,%pil
        ! V9: wrpr %g0, 255, %pil        ! encoding: [0x91,0x90,0x20,0xff]
        wrpr 255,%pil
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: wrpr 255,%cwp
        ! V9: wrpr %g0, 255, %cwp        ! encoding: [0x93,0x90,0x20,0xff]
        wrpr 255,%cwp
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: wrpr 255,%cansave
        ! V9: wrpr %g0, 255, %cansave    ! encoding: [0x95,0x90,0x20,0xff]
        wrpr 255,%cansave
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: wrpr 255,%canrestore
        ! V9: wrpr %g0, 255, %canrestore ! encoding: [0x97,0x90,0x20,0xff]
        wrpr 255,%canrestore
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: wrpr 255,%cleanwin
        ! V9: wrpr %g0, 255, %cleanwin   ! encoding: [0x99,0x90,0x20,0xff]
        wrpr 255,%cleanwin
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: wrpr 255,%otherwin
        ! V9: wrpr %g0, 255, %otherwin   ! encoding: [0x9b,0x90,0x20,0xff]
        wrpr 255,%otherwin
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: wrpr 255,%wstate
        ! V9: wrpr %g0, 255, %wstate     ! encoding: [0x9d,0x90,0x20,0xff]
        wrpr 255,%wstate

        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: rdpr %tpc,%i5
        ! V9: rdpr %tpc, %i5            ! encoding: [0xbb,0x50,0x00,0x00]
        rdpr %tpc,%i5
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: rdpr %tnpc,%i5
        ! V9: rdpr %tnpc, %i5           ! encoding: [0xbb,0x50,0x40,0x00]
        rdpr %tnpc,%i5
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: rdpr %tstate,%i5
        ! V9: rdpr %tstate, %i5         ! encoding: [0xbb,0x50,0x80,0x00]
        rdpr %tstate,%i5
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: rdpr %tt,%i5
        ! V9: rdpr %tt, %i5             ! encoding: [0xbb,0x50,0xc0,0x00]
        rdpr %tt,%i5
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: rdpr %tick,%i5
        ! V9: rdpr %tick, %i5           ! encoding: [0xbb,0x51,0x00,0x00]
        rdpr %tick,%i5
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: rdpr %tba,%i5
        ! V9: rdpr %tba, %i5            ! encoding: [0xbb,0x51,0x40,0x00]
        rdpr %tba,%i5
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: rdpr %pstate,%i5
        ! V9: rdpr %pstate, %i5         ! encoding: [0xbb,0x51,0x80,0x00]
        rdpr %pstate,%i5
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: rdpr %tl,%i5
        ! V9: rdpr %tl, %i5             ! encoding: [0xbb,0x51,0xc0,0x00]
        rdpr %tl,%i5
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: rdpr %pil,%i5
        ! V9: rdpr %pil, %i5            ! encoding: [0xbb,0x52,0x00,0x00]
        rdpr %pil,%i5
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: rdpr %cwp,%i5
        ! V9: rdpr %cwp, %i5            ! encoding: [0xbb,0x52,0x40,0x00]
        rdpr %cwp,%i5
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: rdpr %cansave,%i5
        ! V9: rdpr %cansave, %i5        ! encoding: [0xbb,0x52,0x80,0x00]
        rdpr %cansave,%i5
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: rdpr %canrestore,%i5
        ! V9: rdpr %canrestore, %i5     ! encoding: [0xbb,0x52,0xc0,0x00]
        rdpr %canrestore,%i5
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: rdpr %cleanwin,%i5
        ! V9: rdpr %cleanwin, %i5       ! encoding: [0xbb,0x53,0x00,0x00]
        rdpr %cleanwin,%i5
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: rdpr %otherwin,%i5
        ! V9: rdpr %otherwin, %i5       ! encoding: [0xbb,0x53,0x40,0x00]
        rdpr %otherwin,%i5
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: rdpr %wstate,%i5
        ! V9: rdpr %wstate, %i5         ! encoding: [0xbb,0x53,0x80,0x00]
        rdpr %wstate,%i5

        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: rdpr %fq,%i5
        ! V9: rdpr %fq, %i5             ! encoding: [0xbb,0x53,0xc0,0x00]
        rdpr %fq,%i5
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: rdpr %ver,%i5
        ! V9: rdpr %ver, %i5            ! encoding: [0xbb,0x57,0xc0,0x00]
        rdpr %ver,%i5

        ! V9: rd %pc, %o7               ! encoding: [0x9f,0x41,0x40,0x00]
        rd %pc, %o7
        ! V9: rd %asi, %g1              ! encoding: [0x83,0x40,0xc0,0x00]
        rd %asi, %g1
        ! V9: rd %ccr, %g1              ! encoding: [0x83,0x40,0x80,0x00]
        rd %ccr, %g1
        ! V9: rd %tick, %i5             ! encoding: [0xbb,0x41,0x00,0x00]
        rd %tick,%i5

        ! V9: wr %i0, %i1, %asi         ! encoding: [0x87,0x86,0x00,0x19]
        wr %i0, %i1, %asi
        ! V9: wr %i0, 1, %asi           ! encoding: [0x87,0x86,0x20,0x01]
        wr %i0, 1, %asi
        ! V9: wr %i0, %i1, %ccr         ! encoding: [0x85,0x86,0x00,0x19]
        wr %i0, %i1, %ccr
        ! V9: wr %i0, 1, %ccr           ! encoding: [0x85,0x86,0x20,0x01]
        wr %i0, 1, %ccr

        ! V9: st %o1, [%o0]             ! encoding: [0xd2,0x22,0x00,0x00]
        stw %o1, [%o0]
        ! V9: st %o1, [%o0]             ! encoding: [0xd2,0x22,0x00,0x00]
        stuw %o1, [%o0]
        ! V9: st %o1, [%o0]             ! encoding: [0xd2,0x22,0x00,0x00]
        stsw %o1, [%o0]

        ! V9: sta %o2, [%i0+%l6] #ASI_SNF ! encoding: [0xd4,0xa6,0x10,0x76]
        stwa %o2, [%i0 + %l6] 131
        ! V9: sta %o2, [%i0+%l6] #ASI_SNF ! encoding: [0xd4,0xa6,0x10,0x76]
        stuwa %o2, [%i0 + %l6] 131
        ! V9: sta %o2, [%i0+%l6] #ASI_SNF ! encoding: [0xd4,0xa6,0x10,0x76]
        stswa %o2, [%i0 + %l6] 131

        !! SPARCv9 provides a new variant of ASI-tagged memory accesses.
        ! V9: ldxa [%g2] %asi, %g0    ! encoding: [0xc0,0xd8,0xa0,0x00]
        ldxa [%g2] %asi, %g0
        ! V9: stxa %g0, [%g2] %asi    ! encoding: [0xc0,0xf0,0xa0,0x00]
        stxa %g0, [%g2] %asi
        ! V9: ldxa [%g2+5] %asi, %g0    ! encoding: [0xc0,0xd8,0xa0,0x05]
        ldxa [%g2 + 5] %asi, %g0
        ! V9: stxa %g0, [%g2+5] %asi    ! encoding: [0xc0,0xf0,0xa0,0x05]
        stxa %g0, [%g2 + 5] %asi

        !! Also make sure named ASI tags are parsed properly.
        ! V9: ldxa [%g2+%i5] #ASI_SNF, %g0   ! encoding: [0xc0,0xd8,0x90,0x7d]
        ldxa [%g2 + %i5] #ASI_SNF, %g0
        ! V9: stxa %g0, [%g2+%i5] #ASI_SNF   ! encoding: [0xc0,0xf0,0x90,0x7d]
        stxa %g0, [%g2 + %i5] #ASI_SNF

        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: prefetch  [ %i1 + 0xf80 ], 1
        ! V9: prefetch  [%i1+3968], 1  ! encoding: [0xc3,0x6e,0x6f,0x80]
        prefetch  [ %i1 + 0xf80 ], 1

        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: prefetch  [ %i1 + %i2 ], 1
        ! V9: prefetch  [%i1+%i2], 1  ! encoding: [0xc3,0x6e,0x40,0x1a]
        prefetch  [ %i1 + %i2 ], 1

        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: done
        ! V9: done      ! encoding: [0x81,0xf0,0x00,0x00]
        done
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: retry
        ! V9: retry     ! encoding: [0x83,0xf0,0x00,0x00]
        retry
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: saved
        ! V9: saved     ! encoding: [0x81,0x88,0x00,0x00]
        saved
        ! V8:      error: instruction requires a CPU feature not currently enabled
        ! V8-NEXT: restored
        ! V9: restored  ! encoding: [0x83,0x88,0x00,0x00]
        restored
