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

        call a
        ! CHECK: call foo     ! encoding: [0b01AAAAAA,A,A,A]
        call foo

        ! CHECK: or %g1, %lo(sym), %g3 ! encoding: [0x86,0x10,0b011000AA,A]
        or %g1, %lo(sym), %g3

        ! CHECK: sethi %hi(sym), %l0  ! encoding: [0x21,0b00AAAAAA,A,A]
        sethi %hi(sym), %l0

        ! CHECK: sethi %h44(sym), %l0  ! encoding: [0x21,0b00AAAAAA,A,A]
        sethi %h44(sym), %l0

        ! CHECK: or %g1, %m44(sym), %g3 ! encoding: [0x86,0x10,0b011000AA,A]
        or %g1, %m44(sym), %g3

        ! CHECK: or %g1, %l44(sym), %g3 ! encoding: [0x86,0x10,0b0110AAAA,A]
        or %g1, %l44(sym), %g3

        ! CHECK: sethi %hh(sym), %l0  ! encoding: [0x21,0b00AAAAAA,A,A]
        sethi %hh(sym), %l0

        ! CHECK: sethi %hh(sym), %l0  ! encoding: [0x21,0b00AAAAAA,A,A]
        sethi %uhi(sym), %l0

        ! CHECK: or %g1, %hm(sym), %g3 ! encoding: [0x86,0x10,0b011000AA,A]
        or %g1, %hm(sym), %g3

        ! CHECK: or %g1, %hm(sym), %g3 ! encoding: [0x86,0x10,0b011000AA,A]
        or %g1, %ulo(sym), %g3

        ! CHECK: sethi %lm(sym), %l0  ! encoding: [0x21,0b00AAAAAA,A,A]
        sethi %lm(sym), %l0

        ! CHECK: or %g1, sym, %g3 ! encoding: [0x86,0x10,0b011AAAAA,A]
        or %g1, sym, %g3

        ! CHECK: or %g1, sym+4, %g3 ! encoding: [0x86,0x10,0b011AAAAA,A]
        or %g1, (sym+4), %g3

        ! CHECK: sethi %hix(sym), %g1 ! encoding: [0x03,0b00AAAAAA,A,A]
        sethi %hix(sym), %g1

        ! CHECK: xor %g1, %lox(sym), %g1 ! encoding: [0x82,0x18,0b011AAAAA,A]
        xor %g1, %lox(sym), %g1

        ! CHECK: sethi %gdop_hix22(sym), %l1 ! encoding: [0x23,0x00,0x00,0x00]
        sethi %gdop_hix22(sym), %l1

        ! CHECK: or %l1, %gdop_lox10(sym), %l1 ! encoding: [0xa2,0x14,0x60,0x00]
        or %l1, %gdop_lox10(sym), %l1

        ! CHECK: ldx [%l7+%l1], %l2, %gdop(sym) ! encoding: [0xe4,0x5d,0xc0,0x11]
        ldx [%l7 + %l1], %l2, %gdop(sym)

        ! This test needs to placed last in the file
        ! CHECK: .half	a-.Ltmp0
        .half a - .
        .byte a - .
a:
