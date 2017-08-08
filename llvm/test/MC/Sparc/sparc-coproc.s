! RUN: llvm-mc %s -arch=sparc -show-encoding | FileCheck %s

        ! CHECK: ld [%i1], %c4        ! encoding: [0xc9,0x86,0x40,0x00]
        ! CHECK: ld [%i1+-15], %c4    ! encoding: [0xc9,0x86,0x7f,0xf1]
        ! CHECK: ld [%i1+%o3], %c4    ! encoding: [0xc9,0x86,0x40,0x0b]
        ! CHECK: ld [%i7], %c4        ! encoding: [0xc9,0x87,0xc0,0x00]
        ! CHECK: ld [%i1], %c19       ! encoding: [0xe7,0x86,0x40,0x00]
        ld [%i1], %c4
        ld [%i1 - 15], %c4
        ld [%i1 +%o3], %c4
        ld [%i7], %c4
        ld [%i1], %c19


        ! CHECK: ldd [%i1], %c4       ! encoding: [0xc9,0x9e,0x40,0x00]
        ! CHECK: ldd [%i7], %c4       ! encoding: [0xc9,0x9f,0xc0,0x00]
        ! CHECK: ldd [%i7+200], %c4   ! encoding: [0xc9,0x9f,0xe0,0xc8]
        ! CHECK: ldd [%i7+%o3], %c4   ! encoding: [0xc9,0x9f,0xc0,0x0b]
        ! CHECK: ldd [%i1], %c30      ! encoding: [0xfd,0x9e,0x40,0x00]
        ldd [%i1], %c4
        ldd [%i7], %c4
        ldd [%i7 + 200], %c4
        ldd [%i7+%o3], %c4
        ldd [%i1], %c30


        ! CHECK: st %c4, [%i1]        ! encoding: [0xc9,0xa6,0x40,0x00]
        ! CHECK: st %c4, [%i7]        ! encoding: [0xc9,0xa7,0xc0,0x00]
        ! CHECK: st %c4, [%i7+48]     ! encoding: [0xc9,0xa7,0xe0,0x30]
        ! CHECK: st %c4, [%i4+%o2]    ! encoding: [0xc9,0xa7,0x00,0x0a]
        ! CHECK: st %c19, [%i1]       ! encoding: [0xe7,0xa6,0x40,0x00]
        st %c4, [%i1]
        st %c4, [%i7]
        st %c4, [%i7+48]
        st %c4, [%i4+%o2]
        st %c19, [%i1]


        ! CHECK: std %c4, [%i1]       ! encoding: [0xc9,0xbe,0x40,0x00]
        ! CHECK: std %c4, [%i7]       ! encoding: [0xc9,0xbf,0xc0,0x00]
        ! CHECK: std %c4, [%i2+-240]  ! encoding: [0xc9,0xbe,0xbf,0x10]
        ! CHECK: std %c4, [%i1+%o5]   ! encoding: [0xc9,0xbe,0x40,0x0d]
        ! CHECK: std %c30, [%i1]      ! encoding: [0xfd,0xbe,0x40,0x00]
        std %c4, [%i1]
        std %c4, [%i7]
        std %c4, [%i2-240]
        std %c4, [%i1+%o5]
        std %c30, [%i1]


        ! CHECK: ld [%i5], %csr       ! encoding: [0xc1,0x8f,0x40,0x00]
        ! CHECK: ld [%l2+3], %csr     ! encoding: [0xc1,0x8c,0xa0,0x03]
        ! CHECK: ld [%l4+%l5], %csr   ! encoding: [0xc1,0x8d,0x00,0x15]
        ld [%i5], %csr
        ld [%l2+3], %csr
        ld [%l4+%l5], %csr


        ! CHECK: st %csr, [%i2]       ! encoding: [0xc1,0xae,0x80,0x00]
        ! CHECK: st %csr, [%i2+31]    ! encoding: [0xc1,0xae,0xa0,0x1f]
        ! CHECK: st %csr, [%i2+%o2]   ! encoding: [0xc1,0xae,0x80,0x0a]
        st %csr, [%i2]
        st %csr, [%i2+31]
        st %csr, [%i2+%o2]

        ! CHECK: std %cq, [%o3]       ! encoding: [0xc1,0xb2,0xc0,0x00]
        ! CHECK: std %cq, [%o3+-93]   ! encoding: [0xc1,0xb2,0xff,0xa3]
        ! CHECK: std %cq, [%o3+%l5]   ! encoding: [0xc1,0xb2,0xc0,0x15]
        std %cq, [%o3]
        std %cq, [%o3-93]
        std %cq, [%o3+%l5]
        