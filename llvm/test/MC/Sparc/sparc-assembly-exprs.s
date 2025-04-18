! RUN: llvm-mc %s -triple=sparc   -show-encoding | FileCheck %s
! RUN: llvm-mc %s -triple=sparc -filetype=obj | llvm-objdump -r -d - | FileCheck %s --check-prefix=OBJDUMP

        ! CHECK: mov 1033, %o1  ! encoding: [0x92,0x10,0x24,0x09]
        mov      (0x400|9), %o1
        ! CHECK: mov 60, %o2    ! encoding: [0x94,0x10,0x20,0x3c]
        mov      ((12+3)<<2), %o2

        ! CHECK:   ba      symStart+4           ! encoding: [0x10,0b10AAAAAA,A,A]
        ! OBJDUMP: ba    1
symStart:
        b        symStart + 4

        ! CHECK:   mov     symEnd-symStart, %g1 ! encoding: [0x82,0x10,0b001AAAAA,A]
        ! OBJDUMP: mov	   24, %g1
        mov      symEnd - symStart, %g1

        ! CHECK:   sethi %hi(sym+10), %g2       ! encoding: [0x05,0b00AAAAAA,A,A]
        ! OBJDUMP: R_SPARC_HI22	sym+0xa
        sethi    %hi(sym + 10), %g2

        ! CHECK:   call foo+40                  ! encoding: [0b01AAAAAA,A,A,A]
        ! OBJDUMP: R_SPARC_WDISP30 foo+0x28
        call     foo + 40

        ! CHECK:   add %g1, val+100, %g1        ! encoding: [0x82,0x00,0b011AAAAA,A]
        ! OBJDUMP: R_SPARC_13 val+0x64
        add      %g1, val + 100, %g1

        ! CHECK:   add %g1, 100+val, %g2        ! encoding: [0x84,0x00,0b011AAAAA,A]
        ! OBJDUMP: R_SPARC_13	val+0x64
        add      %g1, 100 + val, %g2
symEnd:

! "." is exactly like a temporary symbol equated to the current line.
! RUN: llvm-mc %s -triple=sparc | FileCheck %s --check-prefix=DOTEXPR

        ! DOTEXPR: .Ltmp0
        ! DOTEXPR-NEXT: ba .Ltmp0+8
        b . + 8
