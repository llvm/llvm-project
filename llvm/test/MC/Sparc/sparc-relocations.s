! RUN: llvm-mc %s -arch=sparcv9 -show-encoding | FileCheck %s
! RUN: llvm-mc %s -arch=sparcv9 -filetype=obj | llvm-readobj -r | FileCheck %s --check-prefix=CHECK-OBJ

        ! CHECK-OBJ: Format: ELF64-sparc
        ! CHECK-OBJ: Relocations [
        ! CHECK-OBJ: 0x{{[0-9,A-F]+}} R_SPARC_WDISP30 foo
        ! CHECK-OBJ: 0x{{[0-9,A-F]+}} R_SPARC_LO10 sym
        ! CHECK-OBJ: 0x{{[0-9,A-F]+}} R_SPARC_HI22 sym
        ! CHECK-OBJ: 0x{{[0-9,A-F]+}} R_SPARC_H44 sym
        ! CHECK-OBJ: 0x{{[0-9,A-F]+}} R_SPARC_M44 sym
        ! CHECK-OBJ: 0x{{[0-9,A-F]+}} R_SPARC_L44 sym
        ! CHECK-OBJ: 0x{{[0-9,A-F]+}} R_SPARC_HH22 sym
        ! CHECK-OBJ: 0x{{[0-9,A-F]+}} R_SPARC_HM10 sym
        ! CHECK-OBJ: 0x{{[0-9,A-F]+}} R_SPARC_13 sym
        ! CHECK-ELF: ]

        ! CHECK: call foo     ! encoding: [0b01AAAAAA,A,A,A]
        ! CHECK:              !   fixup A - offset: 0, value: foo, kind: fixup_sparc_call30
        call foo

        ! CHECK: or %g1, %lo(sym), %g3 ! encoding: [0x86,0x10,0b011000AA,A]
        ! CHECK-NEXT:                  !   fixup A - offset: 0, value: %lo(sym), kind: fixup_sparc_lo10
        or %g1, %lo(sym), %g3

        ! CHECK: sethi %hi(sym), %l0  ! encoding: [0x21,0b00AAAAAA,A,A]
        ! CHECK-NEXT:                 !   fixup A - offset: 0, value: %hi(sym), kind: fixup_sparc_hi22
        sethi %hi(sym), %l0

        ! CHECK: sethi %h44(sym), %l0  ! encoding: [0x21,0b00AAAAAA,A,A]
        ! CHECK-NEXT:                  !   fixup A - offset: 0, value: %h44(sym), kind: fixup_sparc_h44
        sethi %h44(sym), %l0

        ! CHECK: or %g1, %m44(sym), %g3 ! encoding: [0x86,0x10,0b011000AA,A]
        ! CHECK-NEXT:                   !   fixup A - offset: 0, value: %m44(sym), kind: fixup_sparc_m44
        or %g1, %m44(sym), %g3

        ! CHECK: or %g1, %l44(sym), %g3 ! encoding: [0x86,0x10,0b0110AAAA,A]
        ! CHECK-NEXT:                   !   fixup A - offset: 0, value: %l44(sym), kind: fixup_sparc_l44
        or %g1, %l44(sym), %g3

        ! CHECK: sethi %hh(sym), %l0  ! encoding: [0x21,0b00AAAAAA,A,A]
        ! CHECK-NEXT:                 !   fixup A - offset: 0, value: %hh(sym), kind: fixup_sparc_hh
        sethi %hh(sym), %l0

        ! CHECK: or %g1, %hm(sym), %g3 ! encoding: [0x86,0x10,0b011000AA,A]
        ! CHECK-NEXT:                  !   fixup A - offset: 0, value: %hm(sym), kind: fixup_sparc_hm
        or %g1, %hm(sym), %g3

        ! CHECK: or %g1, sym, %g3 ! encoding: [0x86,0x10,0b011AAAAA,A]
        ! CHECK-NEXT:                  !   fixup A - offset: 0, value: sym, kind: fixup_sparc_13
        or %g1, sym, %g3

        ! This test needs to placed last in the file
        ! CHECK: .half	a-.Ltmp0
        .half a - .
        .byte a - .
a:
