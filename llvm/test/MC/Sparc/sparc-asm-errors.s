! RUN: not llvm-mc %s -triple=sparc   -show-encoding 2>&1 | FileCheck %s --check-prefix=CHECK --check-prefix=V8 \
! RUN:   --implicit-check-not=error:
! RUN: not llvm-mc %s -triple=sparcv9 -show-encoding 2>&1 | FileCheck %s --check-prefix=CHECK --check-prefix=V9 \
! RUN:   --implicit-check-not=error:

! Test the lower and upper bounds of 'set'
        ! CHECK: error: set: argument must be between
        set -2147483649, %o1
        ! CHECK: error: set: argument must be between
        set 4294967296, %o1

        ! V8: error: instruction requires a CPU feature not currently enabled
        ! V9: error: unknown membar tag
        membar #BadTag

        ! V8: error: instruction requires a CPU feature not currently enabled
        ! V9: error: invalid membar mask number
        membar -127

        ! V8: error: instruction requires a CPU feature not currently enabled
        ! V9: error: invalid register name
        membar %foo

! Test the boundary checks on the shift amount
        ! CHECK: error: immediate shift value out of range
        sll %g1, 32, %g2
        ! V8: error: instruction requires a CPU feature not currently enabled
        ! V9: error: immediate shift value out of range
        sllx %g1, 64, %g2

! %r0 - %r31 only; don't truncate the register number to two digits.
        ! CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register name
        mov %r32, %g1
        ! CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register name
        mov %r100, %g1
        ! CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register name
        mov %r145, %g1
        ! CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register name
        mov %r310, %g1

! An unknown name after '%' is reported as a register name, or as a
! relocation specifier when it is followed by '('.
        ! CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register name
        mov %foo, %g1
        ! CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register name
        mov %1, %g1
        ! CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register name
        ldd [%g1], %f33
        ! CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register name
        ld [%g1 + %foo], %g1
        ! CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: invalid relocation specifier
        sethi %foo(sym), %g1
        ! CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: unexpected relocation specifier
        sethi %tgd_add(sym), %g1

! Register names are case-sensitive, as in GNU as.
        ! CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register name
        mov %G1, %g2
        ! CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register name
        mov %SP, %g1
        ! CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register name
        mov %R14, %g1
        ! CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register name
        fmovs %F0, %f1
        ! CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register name
        rd %Y, %g1
        ! CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register name
        mov %O6, %g1
        ! CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register name
        mov %I6, %g1

        ! CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: expected ']'
        ld [%g1 + 4, %g2

        ! V8: :[[@LINE+2]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        ! V9: :[[@LINE+1]]:{{[0-9]+}}: error: expected register name
        cas [4], %g2, %g3
        ! V8: :[[@LINE+2]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled
        ! V9: :[[@LINE+1]]:{{[0-9]+}}: error: expected register name
        cas [%foo], %g2, %g3

        ! CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: expected branch modifier
        ba,1 label
        ! CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: expected branch modifier
        bne,a,1 label
        ! Only branches take modifiers after the mnemonic.
        ! CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: unexpected token
        mov , %g1

        ! CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: expected register or immediate
        ld [], %g2
        ! CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: expected register or immediate
        ld [%g1 + ], %g2
        ! V8: :[[@LINE+2]]:{{[0-9]+}}: error: malformed ASI tag, must be a constant integer expression
        ! V9: :[[@LINE+1]]:{{[0-9]+}}: error: malformed ASI tag, must be %asi, a constant integer expression, or a named tag
        ld [%g1] ), %g2
