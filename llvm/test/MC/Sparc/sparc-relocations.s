! RUN: llvm-mc %s -triple=sparcv9 | FileCheck %s
! RUN: llvm-mc %s -triple=sparcv9 -filetype=obj | llvm-readobj -r - | FileCheck %s --check-prefix=CHECK-OBJ

        ! CHECK-OBJ: Format: elf64-sparc
        ! CHECK-OBJ: .rela.text {
        ! CHECK-OBJ-NEXT: 0x{{[0-9,A-F]+}} R_SPARC_WDISP30 foo
        ! CHECK-OBJ-NEXT: 0x{{[0-9,A-F]+}} R_SPARC_13 sym
        ! CHECK-OBJ-NEXT: 0x{{[0-9,A-F]+}} R_SPARC_13 sym
        ! CHECK-OBJ-NEXT: }

        call a
        ! CHECK: call foo
        call foo

        ! CHECK: or %g1, sym, %g3
        or %g1, sym, %g3

        ! CHECK: or %g1, sym+4, %g3
        or %g1, (sym+4), %g3

        ! This test needs to placed last in the file
        ! CHECK: .half	a-.Ltmp0
        .half a - .
        .byte a - .
a:
