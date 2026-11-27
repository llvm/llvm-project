! RUN: not llvm-mc %s -triple=sparc   -show-encoding 2>&1 | FileCheck %s --check-prefix=CHECK --check-prefix=V8
! RUN: not llvm-mc %s -triple=sparcv9 -show-encoding 2>&1 | FileCheck %s --check-prefix=CHECK --check-prefix=V9

! Test the lower and upper bounds of 'set'
        ! CHECK: argument must be between
        set -2147483649, %o1
        ! CHECK: argument must be between
        set 4294967296, %o1

        ! V8: instruction requires a CPU feature not currently enabled
        ! V9: unknown membar tag
        membar #BadTag

        ! V8: instruction requires a CPU feature not currently enabled
        ! V9: invalid membar mask number
        membar -127

! Test the boundary checks on the shift amount
        ! V8: immediate shift value out of range
        sll %g1, 32, %g2
        ! V9: immediate shift value out of range
        slx %g1, 64, %g2

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
