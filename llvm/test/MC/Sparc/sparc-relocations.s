! RUN: llvm-mc %s -triple=sparcv9 -show-encoding | FileCheck %s
! RUN: llvm-mc %s -triple=sparcv9 -filetype=obj | llvm-readobj -r - | FileCheck %s --check-prefix=CHECK-OBJ

        ! CHECK-OBJ: Format: elf64-sparc
        ! CHECK-OBJ: .rela.text {
        ! CHECK-OBJ-NEXT: 0x{{[0-9,A-F]+}} R_SPARC_WDISP30 foo
        ! CHECK-OBJ-NEXT: 0x{{[0-9,A-F]+}} R_SPARC_LO10 sym
        ! CHECK-OBJ-NEXT: 0x{{[0-9,A-F]+}} R_SPARC_HI22 sym
        ! CHECK-OBJ-NEXT: 0x{{[0-9,A-F]+}} R_SPARC_H44 sym
        ! CHECK-OBJ-NEXT: 0x{{[0-9,A-F]+}} R_SPARC_M44 sym
        ! CHECK-OBJ-NEXT: 0x{{[0-9,A-F]+}} R_SPARC_L44 sym
        ! CHECK-OBJ-NEXT: 0x{{[0-9,A-F]+}} R_SPARC_HH22 sym
        ! CHECK-OBJ-NEXT: 0x{{[0-9,A-F]+}} R_SPARC_HH22 sym
        ! CHECK-OBJ-NEXT: 0x{{[0-9,A-F]+}} R_SPARC_HM10 sym
        ! CHECK-OBJ-NEXT: 0x{{[0-9,A-F]+}} R_SPARC_HM10 sym
        ! CHECK-OBJ-NEXT: 0x{{[0-9,A-F]+}} R_SPARC_LM22 sym
        ! CHECK-OBJ-NEXT: 0x{{[0-9,A-F]+}} R_SPARC_13 sym
        ! CHECK-OBJ-NEXT: 0x{{[0-9,A-F]+}} R_SPARC_13 sym
        ! CHECK-OBJ-NEXT: 0x{{[0-9,A-F]+}} R_SPARC_HIX22 sym
        ! CHECK-OBJ-NEXT: 0x{{[0-9,A-F]+}} R_SPARC_LOX10 sym
        ! CHECK-OBJ-NEXT: 0x{{[0-9,A-F]+}} R_SPARC_GOTDATA_HIX22 sym
        ! CHECK-OBJ-NEXT: 0x{{[0-9,A-F]+}} R_SPARC_GOTDATA_LOX10 sym
        ! CHECK-OBJ-NEXT: 0x{{[0-9,A-F]+}} R_SPARC_GOTDATA_OP sym
        ! CHECK-OBJ-NEXT: }

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

        ! CHECK: sethi %hh(sym), %l0  ! encoding: [0x21,0b00AAAAAA,A,A]
        ! CHECK-NEXT:                 !   fixup A - offset: 0, value: %hh(sym), kind: fixup_sparc_hh
        sethi %uhi(sym), %l0

        ! CHECK: or %g1, %hm(sym), %g3 ! encoding: [0x86,0x10,0b011000AA,A]
        ! CHECK-NEXT:                  !   fixup A - offset: 0, value: %hm(sym), kind: fixup_sparc_hm
        or %g1, %hm(sym), %g3

        ! CHECK: or %g1, %hm(sym), %g3 ! encoding: [0x86,0x10,0b011000AA,A]
        ! CHECK-NEXT:                  !   fixup A - offset: 0, value: %hm(sym), kind: fixup_sparc_hm
        or %g1, %ulo(sym), %g3

        ! CHECK: sethi %lm(sym), %l0  ! encoding: [0x21,0b00AAAAAA,A,A]
        ! CHECK-NEXT:                 !   fixup A - offset: 0, value: %lm(sym), kind: fixup_sparc_lm
        sethi %lm(sym), %l0

        ! CHECK: or %g1, sym, %g3 ! encoding: [0x86,0x10,0b011AAAAA,A]
        ! CHECK-NEXT:                  !   fixup A - offset: 0, value: sym, kind: fixup_sparc_13
        or %g1, sym, %g3

        ! CHECK: or %g1, sym+4, %g3 ! encoding: [0x86,0x10,0b011AAAAA,A]
        ! CHECK-NEXT:                  ! fixup A - offset: 0, value: sym+4, kind: fixup_sparc_13
        or %g1, (sym+4), %g3

        ! CHECK: sethi %hix(sym), %g1 ! encoding: [0x03,0b00AAAAAA,A,A]
        ! CHECK-NEXT:                 ! fixup A - offset: 0, value: %hix(sym), kind: fixup_sparc_hix22
        sethi %hix(sym), %g1

        ! CHECK: xor %g1, %lox(sym), %g1 ! encoding: [0x82,0x18,0b011AAAAA,A]
        ! CHECK-NEXT:                    ! fixup A - offset: 0, value: %lox(sym), kind: fixup_sparc_lox10
        xor %g1, %lox(sym), %g1

        ! CHECK: sethi %gdop_hix22(sym), %l1 ! encoding: [0x23,0x00,0x00,0x00]
        ! CHECK-NEXT:                        ! fixup A - offset: 0, value: %gdop_hix22(sym), kind: fixup_sparc_gotdata_hix22
        sethi %gdop_hix22(sym), %l1

        ! CHECK: or %l1, %gdop_lox10(sym), %l1 ! encoding: [0xa2,0x14,0x60,0x00]
        ! CHECK-NEXT:                          ! fixup A - offset: 0, value: %gdop_lox10(sym), kind: fixup_sparc_gotdata_lox10
        or %l1, %gdop_lox10(sym), %l1

        ! CHECK: ldx [%l7+%l1], %l2, %gdop(sym) ! encoding: [0xe4,0x5d,0xc0,0x11]
        ! CHECK-NEXT:                           ! fixup A - offset: 0, value: %gdop(sym), kind: fixup_sparc_gotdata_op
        ldx [%l7 + %l1], %l2, %gdop(sym)

        ! This test needs to placed last in the file
        ! CHECK: .half	a-.Ltmp0
        .half a - .
        .byte a - .
a:
