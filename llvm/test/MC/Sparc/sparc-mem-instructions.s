! RUN: llvm-mc %s -arch=sparc   -show-encoding | FileCheck %s --check-prefix=V8
! RUN: llvm-mc %s -arch=sparcv9 -show-encoding | FileCheck %s --check-prefix=V9

        ! V8: ldsb [%i0+%l6], %o2  ! encoding: [0xd4,0x4e,0x00,0x16]
        ! V9: ldsb [%i0+%l6], %o2  ! encoding: [0xd4,0x4e,0x00,0x16]
        ldsb [%i0 + %l6], %o2
        ! V8: ldsb [%i0+32], %o2   ! encoding: [0xd4,0x4e,0x20,0x20]
        ! V9: ldsb [%i0+32], %o2   ! encoding: [0xd4,0x4e,0x20,0x20]
        ldsb [%i0 + 32], %o2
        ! V8: ldsb [%g1], %o4      ! encoding: [0xd8,0x48,0x40,0x00]
        ! V9: ldsb [%g1], %o4      ! encoding: [0xd8,0x48,0x40,0x00]
        ldsb [%g1], %o4
        ! V8: ldsba [%i0+%l6] 131, %o2  ! encoding: [0xd4,0xce,0x10,0x76]
        ! V9: ldsba [%i0+%l6] #ASI_SNF, %o2  ! encoding: [0xd4,0xce,0x10,0x76]
        ldsba [%i0 + %l6] 131, %o2
        ! V8: ldsba [%i0+%l6] 131, %o2  ! encoding: [0xd4,0xce,0x10,0x76]
        ! V9: ldsba [%i0+%l6] #ASI_SNF, %o2  ! encoding: [0xd4,0xce,0x10,0x76]
        ldsba [%i0 + %l6] (130+1), %o2

        ! V8: ldsh [%i0+%l6], %o2  ! encoding: [0xd4,0x56,0x00,0x16]
        ! V9: ldsh [%i0+%l6], %o2  ! encoding: [0xd4,0x56,0x00,0x16]
        ldsh [%i0 + %l6], %o2
        ! V8: ldsh [%i0+32], %o2   ! encoding: [0xd4,0x56,0x20,0x20]
        ! V9: ldsh [%i0+32], %o2   ! encoding: [0xd4,0x56,0x20,0x20]
        ldsh [%i0 + 32], %o2
        ! V8: ldsh [%g1], %o4      ! encoding: [0xd8,0x50,0x40,0x00]
        ! V9: ldsh [%g1], %o4      ! encoding: [0xd8,0x50,0x40,0x00]
        ldsh [%g1], %o4
        ! V8: ldsha [%i0+%l6] 131, %o2 ! encoding: [0xd4,0xd6,0x10,0x76]
        ! V9: ldsha [%i0+%l6] #ASI_SNF, %o2 ! encoding: [0xd4,0xd6,0x10,0x76]
        ldsha [%i0 + %l6] 131, %o2
        ! V8: ldsha [%i0+%l6] 131, %o2 ! encoding: [0xd4,0xd6,0x10,0x76]
        ! V9: ldsha [%i0+%l6] #ASI_SNF, %o2 ! encoding: [0xd4,0xd6,0x10,0x76]
        ldsha [%i0 + %l6] (130+1), %o2

        ! V8: ldub [%i0+%l6], %o2  ! encoding: [0xd4,0x0e,0x00,0x16]
        ! V9: ldub [%i0+%l6], %o2  ! encoding: [0xd4,0x0e,0x00,0x16]
        ldub [%i0 + %l6], %o2
        ! V8: ldub [%i0+32], %o2   ! encoding: [0xd4,0x0e,0x20,0x20]
        ! V9: ldub [%i0+32], %o2   ! encoding: [0xd4,0x0e,0x20,0x20]
        ldub [%i0 + 32], %o2
        ! V8: ldub [%g1], %o2      ! encoding: [0xd4,0x08,0x40,0x00]
        ! V9: ldub [%g1], %o2      ! encoding: [0xd4,0x08,0x40,0x00]
        ldub [%g1], %o2
        ! V8: lduba [%i0+%l6] 131, %o2 ! encoding: [0xd4,0x8e,0x10,0x76]
        ! V9: lduba [%i0+%l6] #ASI_SNF, %o2 ! encoding: [0xd4,0x8e,0x10,0x76]
        lduba [%i0 + %l6] 131, %o2
        ! V8: lduba [%i0+%l6] 131, %o2 ! encoding: [0xd4,0x8e,0x10,0x76]
        ! V9: lduba [%i0+%l6] #ASI_SNF, %o2 ! encoding: [0xd4,0x8e,0x10,0x76]
        lduba [%i0 + %l6] (130+1), %o2

        ! V8: lduh [%i0+%l6], %o2  ! encoding: [0xd4,0x16,0x00,0x16]
        ! V9: lduh [%i0+%l6], %o2  ! encoding: [0xd4,0x16,0x00,0x16]
        lduh [%i0 + %l6], %o2
        ! V8: lduh [%i0+32], %o2   ! encoding: [0xd4,0x16,0x20,0x20]
        ! V9: lduh [%i0+32], %o2   ! encoding: [0xd4,0x16,0x20,0x20]
        lduh [%i0 + 32], %o2
        ! V8: lduh [%g1], %o2      ! encoding: [0xd4,0x10,0x40,0x00]
        ! V9: lduh [%g1], %o2      ! encoding: [0xd4,0x10,0x40,0x00]
        lduh [%g1], %o2
        ! V8: lduha [%i0+%l6] 131, %o2 ! encoding: [0xd4,0x96,0x10,0x76]
        ! V9: lduha [%i0+%l6] #ASI_SNF, %o2 ! encoding: [0xd4,0x96,0x10,0x76]
        lduha [%i0 + %l6] 131, %o2
        ! V8: lduha [%i0+%l6] 131, %o2 ! encoding: [0xd4,0x96,0x10,0x76]
        ! V9: lduha [%i0+%l6] #ASI_SNF, %o2 ! encoding: [0xd4,0x96,0x10,0x76]
        lduha [%i0 + %l6] (130+1), %o2

        ! V8: ld [%i0+%l6], %o2    ! encoding: [0xd4,0x06,0x00,0x16]
        ! V9: ld [%i0+%l6], %o2    ! encoding: [0xd4,0x06,0x00,0x16]
        ld [%i0 + %l6], %o2
        ! V8: ld [%i0+32], %o2     ! encoding: [0xd4,0x06,0x20,0x20]
        ! V9: ld [%i0+32], %o2     ! encoding: [0xd4,0x06,0x20,0x20]
        ld [%i0 + 32], %o2
        ! V8: ld [%g1], %o2        ! encoding: [0xd4,0x00,0x40,0x00]
        ! V9: ld [%g1], %o2        ! encoding: [0xd4,0x00,0x40,0x00]
        ld [%g1], %o2
        ! V8: lda [%i0+%l6] 131, %o2 ! encoding: [0xd4,0x86,0x10,0x76]
        ! V9: lda [%i0+%l6] #ASI_SNF, %o2 ! encoding: [0xd4,0x86,0x10,0x76]
        lda [%i0 + %l6] 131, %o2
        ! V8: lda [%i0+%l6] 131, %o2 ! encoding: [0xd4,0x86,0x10,0x76]
        ! V9: lda [%i0+%l6] #ASI_SNF, %o2 ! encoding: [0xd4,0x86,0x10,0x76]
        lda [%i0 + %l6] (130+1), %o2

        ! V8: ldd [%i0+%l6], %o2    ! encoding: [0xd4,0x1e,0x00,0x16]
        ! V9: ldd [%i0+%l6], %o2    ! encoding: [0xd4,0x1e,0x00,0x16]
        ldd [%i0 + %l6], %o2
        ! V8: ldd [%i0+32], %o2     ! encoding: [0xd4,0x1e,0x20,0x20]
        ! V9: ldd [%i0+32], %o2     ! encoding: [0xd4,0x1e,0x20,0x20]
        ldd [%i0 + 32], %o2
        ! V8: ldd [%g1], %o2        ! encoding: [0xd4,0x18,0x40,0x00]
        ! V9: ldd [%g1], %o2        ! encoding: [0xd4,0x18,0x40,0x00]
        ldd [%g1], %o2
        ! V8: ldda [%i0+%l6] 131, %o2 ! encoding: [0xd4,0x9e,0x10,0x76]
        ! V9: ldda [%i0+%l6] #ASI_SNF, %o2 ! encoding: [0xd4,0x9e,0x10,0x76]
        ldda [%i0 + %l6] 131, %o2
        ! V8: ldda [%i0+%l6] 131, %o2 ! encoding: [0xd4,0x9e,0x10,0x76]
        ! V9: ldda [%i0+%l6] #ASI_SNF, %o2 ! encoding: [0xd4,0x9e,0x10,0x76]
        ldda [%i0 + %l6] (130+1), %o2

        ! V8: stb %o2, [%i0+%l6]   ! encoding: [0xd4,0x2e,0x00,0x16]
        ! V9: stb %o2, [%i0+%l6]   ! encoding: [0xd4,0x2e,0x00,0x16]
        stb %o2, [%i0 + %l6]
        ! V8: stb %o2, [%i0+32]    ! encoding: [0xd4,0x2e,0x20,0x20]
        ! V9: stb %o2, [%i0+32]    ! encoding: [0xd4,0x2e,0x20,0x20]
        stb %o2, [%i0 + 32]
        ! V8: stb %o2, [%g1]       ! encoding: [0xd4,0x28,0x40,0x00]
        ! V9: stb %o2, [%g1]       ! encoding: [0xd4,0x28,0x40,0x00]
        stb %o2, [%g1]
        ! V8: stb %o2, [%g1]       ! encoding: [0xd4,0x28,0x40,0x00]
        ! V9: stb %o2, [%g1]       ! encoding: [0xd4,0x28,0x40,0x00]
        stub %o2, [%g1]
        ! V8: stb %o2, [%g1]       ! encoding: [0xd4,0x28,0x40,0x00]
        ! V9: stb %o2, [%g1]       ! encoding: [0xd4,0x28,0x40,0x00]
        stsb %o2, [%g1]
        ! V8: stba %o2, [%i0+%l6] 131 ! encoding: [0xd4,0xae,0x10,0x76]
        ! V9: stba %o2, [%i0+%l6] #ASI_SNF ! encoding: [0xd4,0xae,0x10,0x76]
        stba %o2, [%i0 + %l6] 131
        ! V8: stba %o2, [%i0+%l6] 131 ! encoding: [0xd4,0xae,0x10,0x76]
        ! V9: stba %o2, [%i0+%l6] #ASI_SNF ! encoding: [0xd4,0xae,0x10,0x76]
        stba %o2, [%i0 + %l6] (130+1)
        ! V8: stba %o2, [%i0+%l6] 131 ! encoding: [0xd4,0xae,0x10,0x76]
        ! V9: stba %o2, [%i0+%l6] #ASI_SNF ! encoding: [0xd4,0xae,0x10,0x76]
        stuba %o2, [%i0 + %l6] 131
        ! V8: stba %o2, [%i0+%l6] 131 ! encoding: [0xd4,0xae,0x10,0x76]
        ! V9: stba %o2, [%i0+%l6] #ASI_SNF ! encoding: [0xd4,0xae,0x10,0x76]
        stuba %o2, [%i0 + %l6] (130+1)
        ! V8: stba %o2, [%i0+%l6] 131 ! encoding: [0xd4,0xae,0x10,0x76]
        ! V9: stba %o2, [%i0+%l6] #ASI_SNF ! encoding: [0xd4,0xae,0x10,0x76]
        stsba %o2, [%i0 + %l6] 131
        ! V8: stba %o2, [%i0+%l6] 131 ! encoding: [0xd4,0xae,0x10,0x76]
        ! V9: stba %o2, [%i0+%l6] #ASI_SNF ! encoding: [0xd4,0xae,0x10,0x76]
        stsba %o2, [%i0 + %l6] (130+1)

        ! V8: sth %o2, [%i0+%l6]   ! encoding: [0xd4,0x36,0x00,0x16]
        ! V9: sth %o2, [%i0+%l6]   ! encoding: [0xd4,0x36,0x00,0x16]
        sth %o2, [%i0 + %l6]
        ! V8: sth %o2, [%i0+32]    ! encoding: [0xd4,0x36,0x20,0x20]
        ! V9: sth %o2, [%i0+32]    ! encoding: [0xd4,0x36,0x20,0x20]
        sth %o2, [%i0 + 32]
        ! V8: sth %o2, [%g1]       ! encoding: [0xd4,0x30,0x40,0x00]
        ! V9: sth %o2, [%g1]       ! encoding: [0xd4,0x30,0x40,0x00]
        sth %o2, [%g1]
        ! V8: sth %o2, [%g1]       ! encoding: [0xd4,0x30,0x40,0x00]
        ! V9: sth %o2, [%g1]       ! encoding: [0xd4,0x30,0x40,0x00]
        stuh %o2, [%g1]
        ! V8: sth %o2, [%g1]       ! encoding: [0xd4,0x30,0x40,0x00]
        ! V9: sth %o2, [%g1]       ! encoding: [0xd4,0x30,0x40,0x00]
        stsh %o2, [%g1]
        ! V8: stha %o2, [%i0+%l6] 131 ! encoding: [0xd4,0xb6,0x10,0x76]
        ! V9: stha %o2, [%i0+%l6] #ASI_SNF ! encoding: [0xd4,0xb6,0x10,0x76]
        stha %o2, [%i0 + %l6] 131
        ! V8: stha %o2, [%i0+%l6] 131 ! encoding: [0xd4,0xb6,0x10,0x76]
        ! V9: stha %o2, [%i0+%l6] #ASI_SNF ! encoding: [0xd4,0xb6,0x10,0x76]
        stha %o2, [%i0 + %l6] (130+1)
        ! V8: stha %o2, [%i0+%l6] 131 ! encoding: [0xd4,0xb6,0x10,0x76]
        ! V9: stha %o2, [%i0+%l6] #ASI_SNF ! encoding: [0xd4,0xb6,0x10,0x76]
        stuha %o2, [%i0 + %l6] 131
        ! V8: stha %o2, [%i0+%l6] 131 ! encoding: [0xd4,0xb6,0x10,0x76]
        ! V9: stha %o2, [%i0+%l6] #ASI_SNF ! encoding: [0xd4,0xb6,0x10,0x76]
        stuha %o2, [%i0 + %l6] (130+1)
        ! V8: stha %o2, [%i0+%l6] 131 ! encoding: [0xd4,0xb6,0x10,0x76]
        ! V9: stha %o2, [%i0+%l6] #ASI_SNF ! encoding: [0xd4,0xb6,0x10,0x76]
        stsha %o2, [%i0 + %l6] 131
        ! V8: stha %o2, [%i0+%l6] 131 ! encoding: [0xd4,0xb6,0x10,0x76]
        ! V9: stha %o2, [%i0+%l6] #ASI_SNF ! encoding: [0xd4,0xb6,0x10,0x76]
        stsha %o2, [%i0 + %l6] (130+1)

        ! V8: st %o2, [%i0+%l6]    ! encoding: [0xd4,0x26,0x00,0x16]
        ! V9: st %o2, [%i0+%l6]    ! encoding: [0xd4,0x26,0x00,0x16]
        st %o2, [%i0 + %l6]
        ! V8: st %o2, [%i0+32]     ! encoding: [0xd4,0x26,0x20,0x20]
        ! V9: st %o2, [%i0+32]     ! encoding: [0xd4,0x26,0x20,0x20]
        st %o2, [%i0 + 32]
        ! V8: st %o2, [%g1]        ! encoding: [0xd4,0x20,0x40,0x00]
        ! V9: st %o2, [%g1]        ! encoding: [0xd4,0x20,0x40,0x00]
        st %o2, [%g1]
        ! V8: sta %o2, [%i0+%l6] 131 ! encoding: [0xd4,0xa6,0x10,0x76]
        ! V9: sta %o2, [%i0+%l6] #ASI_SNF ! encoding: [0xd4,0xa6,0x10,0x76]
        sta %o2, [%i0 + %l6] 131
        ! V8: sta %o2, [%i0+%l6] 131 ! encoding: [0xd4,0xa6,0x10,0x76]
        ! V9: sta %o2, [%i0+%l6] #ASI_SNF ! encoding: [0xd4,0xa6,0x10,0x76]
        sta %o2, [%i0 + %l6] (130+1)

        ! V8: std %o2, [%i0+%l6]    ! encoding: [0xd4,0x3e,0x00,0x16]
        ! V9: std %o2, [%i0+%l6]    ! encoding: [0xd4,0x3e,0x00,0x16]
        std %o2, [%i0 + %l6]
        ! V8: std %o2, [%i0+32]     ! encoding: [0xd4,0x3e,0x20,0x20]
        ! V9: std %o2, [%i0+32]     ! encoding: [0xd4,0x3e,0x20,0x20]
        std %o2, [%i0 + 32]
        ! V8: std %o2, [%g1]        ! encoding: [0xd4,0x38,0x40,0x00]
        ! V9: std %o2, [%g1]        ! encoding: [0xd4,0x38,0x40,0x00]
        std %o2, [%g1]
        ! V8: stda %o2, [%i0+%l6] 131 ! encoding: [0xd4,0xbe,0x10,0x76]
        ! V9: stda %o2, [%i0+%l6] #ASI_SNF ! encoding: [0xd4,0xbe,0x10,0x76]
        stda %o2, [%i0 + %l6] 131
        ! V8: stda %o2, [%i0+%l6] 131 ! encoding: [0xd4,0xbe,0x10,0x76]
        ! V9: stda %o2, [%i0+%l6] #ASI_SNF ! encoding: [0xd4,0xbe,0x10,0x76]
        stda %o2, [%i0 + %l6] (130+1)

        ! V8:  flush %g1+%g2         ! encoding: [0x81,0xd8,0x40,0x02]
        ! V9:  flush %g1+%g2         ! encoding: [0x81,0xd8,0x40,0x02]
        flush %g1 + %g2
        ! V8:  flush %g1+8           ! encoding: [0x81,0xd8,0x60,0x08]
        ! V9:  flush %g1+8           ! encoding: [0x81,0xd8,0x60,0x08]
        flush %g1 + 8
        ! V8:  flush %g1             ! encoding: [0x81,0xd8,0x40,0x00]
        ! V9:  flush %g1             ! encoding: [0x81,0xd8,0x40,0x00]
        flush %g1
        ! Not specified in manual, but accepted by gas.
        ! V8:  flush %g0             ! encoding: [0x81,0xd8,0x00,0x00]
        ! V9:  flush %g0             ! encoding: [0x81,0xd8,0x00,0x00]
        flush
        ! V8:  flush %g0             ! encoding: [0x81,0xd8,0x00,0x00]
        ! V9:  flush %g0             ! encoding: [0x81,0xd8,0x00,0x00]
        iflush
