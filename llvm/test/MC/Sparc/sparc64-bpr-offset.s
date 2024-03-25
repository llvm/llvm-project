! RUN: llvm-mc -triple=sparcv9 -filetype=obj %s | llvm-objdump -d - | FileCheck %s --check-prefix=BIN

        !! SPARCv9/SPARC64 BPr branches have different offset encoding from the others,
        !! make sure that our offset bits don't trample on other fields.
        !! This is particularly important with backwards branches.

        ! BIN:  0: 02 c8 40 01  	brz %g1, 1
        ! BIN:  4: 04 c8 40 01  	brlez %g1, 1
        ! BIN:  8: 06 c8 40 01  	brlz %g1, 1
        ! BIN:  c: 0a c8 40 01  	brnz %g1, 1
        ! BIN: 10: 0c c8 40 01  	brgz %g1, 1
        ! BIN: 14: 0e c8 40 01  	brgez %g1, 1
        brz   %g1, .+4
        brlez %g1, .+4
        brlz  %g1, .+4
        brnz  %g1, .+4
        brgz  %g1, .+4
        brgez %g1, .+4

        ! BIN: 18: 02 f8 7f ff  	brz %g1, 65535
        ! BIN: 1c: 04 f8 7f ff  	brlez %g1, 65535
        ! BIN: 20: 06 f8 7f ff  	brlz %g1, 65535
        ! BIN: 24: 0a f8 7f ff  	brnz %g1, 65535
        ! BIN: 28: 0c f8 7f ff  	brgz %g1, 65535
        ! BIN: 2c: 0e f8 7f ff  	brgez %g1, 65535
        brz   %g1, .-4
        brlez %g1, .-4
        brlz  %g1, .-4
        brnz  %g1, .-4
        brgz  %g1, .-4
        brgez %g1, .-4
