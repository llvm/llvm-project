! RUN: llvm-mc %s -triple=sparc -mattr=+osa2011 | FileCheck %s
! RUN: llvm-mc %s -triple=sparc -mattr=+osa2011 -filetype=obj | llvm-objdump -r -d - | FileCheck %s --check-prefix=OBJDUMP

        ! CHECK: mov 1033, %o1
        mov      (0x400|9), %o1
        ! CHECK: mov 60, %o2
        mov      ((12+3)<<2), %o2

        ! CHECK:   ba      symStart+4
        ! OBJDUMP: ba    0xc
symStart:
        b        symStart + 4

        ! CHECK:   mov     symEnd-symStart, %g1
        ! OBJDUMP: mov	   0x20, %g1
        mov      symEnd - symStart, %g1

        ! CHECK:   sethi %hi(sym+10), %g2
        ! OBJDUMP: R_SPARC_HI22	sym+0xa
        sethi    %hi(sym + 10), %g2

        ! CHECK:   call foo+40
        ! OBJDUMP: R_SPARC_WDISP30 foo+0x28
        call     foo + 40

        ! CHECK:   add %g1, val+100, %g1
        ! OBJDUMP: R_SPARC_13 val+0x64
        add      %g1, val + 100, %g1

        ! CHECK:   add %g1, 100+val, %g2
        ! OBJDUMP: R_SPARC_13	val+0x64
        add      %g1, 100 + val, %g2

        ! CHECK:   cwbe %g0, val+2, symStart
        ! OBJDUMP: R_SPARC_5	val+0x2
        cwbe %g0, val + 2, symStart

        ! CHECK:   cxbe %g0, val+2, symStart
        ! OBJDUMP: R_SPARC_5	val+0x2
        cxbe %g0, val + 2, symStart
symEnd:

! "." is exactly like a temporary symbol equated to the current line.
! RUN: llvm-mc %s -triple=sparc -mattr=+osa2011 | FileCheck %s --check-prefix=DOTEXPR

        ! DOTEXPR: .Ltmp0
        ! DOTEXPR-NEXT: ba .Ltmp0+8
        b . + 8
