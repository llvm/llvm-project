! RUN: llvm-mc -triple=sparcv9 -mattr=+osa2011 -filetype=obj %s | llvm-objdump --mattr=+osa2011 -d - | FileCheck %s --check-prefix=BIN

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

        !! Similarly, OSA2011 CBCond branches have different offset encoding,
        !! make sure that our offset bits don't trample on other fields.
        !! This is particularly important with backwards branches.

        !BIN: 30: 32 c2 00 29  	cwbne	%o0, %o1, 1
        !BIN: 34: 12 c2 00 29  	cwbe	%o0, %o1, 1
        !BIN: 38: 34 c2 00 29  	cwbg	%o0, %o1, 1
        !BIN: 3c: 14 c2 00 29  	cwble	%o0, %o1, 1
        !BIN: 40: 36 c2 00 29  	cwbge	%o0, %o1, 1
        !BIN: 44: 16 c2 00 29  	cwbl	%o0, %o1, 1
        !BIN: 48: 38 c2 00 29  	cwbgu	%o0, %o1, 1
        !BIN: 4c: 18 c2 00 29  	cwbleu	%o0, %o1, 1
        !BIN: 50: 3a c2 00 29  	cwbcc	%o0, %o1, 1
        !BIN: 54: 1a c2 00 29  	cwbcs	%o0, %o1, 1
        !BIN: 58: 3c c2 00 29  	cwbpos	%o0, %o1, 1
        !BIN: 5c: 1c c2 00 29  	cwbneg	%o0, %o1, 1
        !BIN: 60: 3e c2 00 29  	cwbvc	%o0, %o1, 1
        !BIN: 64: 1e c2 00 29  	cwbvs	%o0, %o1, 1
        cwbne  %o0, %o1, .+4
        cwbe   %o0, %o1, .+4
        cwbg   %o0, %o1, .+4
        cwble  %o0, %o1, .+4
        cwbge  %o0, %o1, .+4
        cwbl   %o0, %o1, .+4
        cwbgu  %o0, %o1, .+4
        cwbleu %o0, %o1, .+4
        cwbcc  %o0, %o1, .+4
        cwbcs  %o0, %o1, .+4
        cwbpos %o0, %o1, .+4
        cwbneg %o0, %o1, .+4
        cwbvc  %o0, %o1, .+4
        cwbvs  %o0, %o1, .+4

        !BIN: 68: 32 da 1f e9  	cwbne	%o0, %o1, 1023
        !BIN: 6c: 12 da 1f e9  	cwbe	%o0, %o1, 1023
        !BIN: 70: 34 da 1f e9  	cwbg	%o0, %o1, 1023
        !BIN: 74: 14 da 1f e9  	cwble	%o0, %o1, 1023
        !BIN: 78: 36 da 1f e9  	cwbge	%o0, %o1, 1023
        !BIN: 7c: 16 da 1f e9  	cwbl	%o0, %o1, 1023
        !BIN: 80: 38 da 1f e9  	cwbgu	%o0, %o1, 1023
        !BIN: 84: 18 da 1f e9  	cwbleu	%o0, %o1, 1023
        !BIN: 88: 3a da 1f e9  	cwbcc	%o0, %o1, 1023
        !BIN: 8c: 1a da 1f e9  	cwbcs	%o0, %o1, 1023
        !BIN: 90: 3c da 1f e9  	cwbpos	%o0, %o1, 1023
        !BIN: 94: 1c da 1f e9  	cwbneg	%o0, %o1, 1023
        !BIN: 98: 3e da 1f e9  	cwbvc	%o0, %o1, 1023
        !BIN: 9c: 1e da 1f e9  	cwbvs	%o0, %o1, 1023
        cwbne  %o0, %o1, .-4
        cwbe   %o0, %o1, .-4
        cwbg   %o0, %o1, .-4
        cwble  %o0, %o1, .-4
        cwbge  %o0, %o1, .-4
        cwbl   %o0, %o1, .-4
        cwbgu  %o0, %o1, .-4
        cwbleu %o0, %o1, .-4
        cwbcc  %o0, %o1, .-4
        cwbcs  %o0, %o1, .-4
        cwbpos %o0, %o1, .-4
        cwbneg %o0, %o1, .-4
        cwbvc  %o0, %o1, .-4
        cwbvs  %o0, %o1, .-4

        !BIN: a0: 32 c2 20 21  	cwbne	%o0, 1, 1
        !BIN: a4: 12 c2 20 21  	cwbe	%o0, 1, 1
        !BIN: a8: 34 c2 20 21  	cwbg	%o0, 1, 1
        !BIN: ac: 14 c2 20 21  	cwble	%o0, 1, 1
        !BIN: b0: 36 c2 20 21  	cwbge	%o0, 1, 1
        !BIN: b4: 16 c2 20 21  	cwbl	%o0, 1, 1
        !BIN: b8: 38 c2 20 21  	cwbgu	%o0, 1, 1
        !BIN: bc: 18 c2 20 21  	cwbleu	%o0, 1, 1
        !BIN: c0: 3a c2 20 21  	cwbcc	%o0, 1, 1
        !BIN: c4: 1a c2 20 21  	cwbcs	%o0, 1, 1
        !BIN: c8: 3c c2 20 21  	cwbpos	%o0, 1, 1
        !BIN: cc: 1c c2 20 21  	cwbneg	%o0, 1, 1
        !BIN: d0: 3e c2 20 21  	cwbvc	%o0, 1, 1
        !BIN: d4: 1e c2 20 21  	cwbvs	%o0, 1, 1
        cwbne  %o0, 1, .+4
        cwbe   %o0, 1, .+4
        cwbg   %o0, 1, .+4
        cwble  %o0, 1, .+4
        cwbge  %o0, 1, .+4
        cwbl   %o0, 1, .+4
        cwbgu  %o0, 1, .+4
        cwbleu %o0, 1, .+4
        cwbcc  %o0, 1, .+4
        cwbcs  %o0, 1, .+4
        cwbpos %o0, 1, .+4
        cwbneg %o0, 1, .+4
        cwbvc  %o0, 1, .+4
        cwbvs  %o0, 1, .+4

        !BIN:  d8: 32 da 3f e1  	cwbne	%o0, 1, 1023
        !BIN:  dc: 12 da 3f e1  	cwbe	%o0, 1, 1023
        !BIN:  e0: 34 da 3f e1  	cwbg	%o0, 1, 1023
        !BIN:  e4: 14 da 3f e1  	cwble	%o0, 1, 1023
        !BIN:  e8: 36 da 3f e1  	cwbge	%o0, 1, 1023
        !BIN:  ec: 16 da 3f e1  	cwbl	%o0, 1, 1023
        !BIN:  f0: 38 da 3f e1  	cwbgu	%o0, 1, 1023
        !BIN:  f4: 18 da 3f e1  	cwbleu	%o0, 1, 1023
        !BIN:  f8: 3a da 3f e1  	cwbcc	%o0, 1, 1023
        !BIN:  fc: 1a da 3f e1  	cwbcs	%o0, 1, 1023
        !BIN: 100: 3c da 3f e1  	cwbpos	%o0, 1, 1023
        !BIN: 104: 1c da 3f e1  	cwbneg	%o0, 1, 1023
        !BIN: 108: 3e da 3f e1  	cwbvc	%o0, 1, 1023
        !BIN: 10c: 1e da 3f e1  	cwbvs	%o0, 1, 1023
        cwbne  %o0, 1, .-4
        cwbe   %o0, 1, .-4
        cwbg   %o0, 1, .-4
        cwble  %o0, 1, .-4
        cwbge  %o0, 1, .-4
        cwbl   %o0, 1, .-4
        cwbgu  %o0, 1, .-4
        cwbleu %o0, 1, .-4
        cwbcc  %o0, 1, .-4
        cwbcs  %o0, 1, .-4
        cwbpos %o0, 1, .-4
        cwbneg %o0, 1, .-4
        cwbvc  %o0, 1, .-4
        cwbvs  %o0, 1, .-4

        !BIN: 110: 32 e2 00 29  	cxbne	%o0, %o1, 1
        !BIN: 114: 12 e2 00 29  	cxbe	%o0, %o1, 1
        !BIN: 118: 34 e2 00 29  	cxbg	%o0, %o1, 1
        !BIN: 11c: 14 e2 00 29  	cxble	%o0, %o1, 1
        !BIN: 120: 36 e2 00 29  	cxbge	%o0, %o1, 1
        !BIN: 124: 16 e2 00 29  	cxbl	%o0, %o1, 1
        !BIN: 128: 38 e2 00 29  	cxbgu	%o0, %o1, 1
        !BIN: 12c: 18 e2 00 29  	cxbleu	%o0, %o1, 1
        !BIN: 130: 3a e2 00 29  	cxbcc	%o0, %o1, 1
        !BIN: 134: 1a e2 00 29  	cxbcs	%o0, %o1, 1
        !BIN: 138: 3c e2 00 29  	cxbpos	%o0, %o1, 1
        !BIN: 13c: 1c e2 00 29  	cxbneg	%o0, %o1, 1
        !BIN: 140: 3e e2 00 29  	cxbvc	%o0, %o1, 1
        !BIN: 144: 1e e2 00 29  	cxbvs	%o0, %o1, 1
        cxbne  %o0, %o1, .+4
        cxbe   %o0, %o1, .+4
        cxbg   %o0, %o1, .+4
        cxble  %o0, %o1, .+4
        cxbge  %o0, %o1, .+4
        cxbl   %o0, %o1, .+4
        cxbgu  %o0, %o1, .+4
        cxbleu %o0, %o1, .+4
        cxbcc  %o0, %o1, .+4
        cxbcs  %o0, %o1, .+4
        cxbpos %o0, %o1, .+4
        cxbneg %o0, %o1, .+4
        cxbvc  %o0, %o1, .+4
        cxbvs  %o0, %o1, .+4

        !BIN: 148: 32 fa 1f e9  	cxbne	%o0, %o1, 1023
        !BIN: 14c: 12 fa 1f e9  	cxbe	%o0, %o1, 1023
        !BIN: 150: 34 fa 1f e9  	cxbg	%o0, %o1, 1023
        !BIN: 154: 14 fa 1f e9  	cxble	%o0, %o1, 1023
        !BIN: 158: 36 fa 1f e9  	cxbge	%o0, %o1, 1023
        !BIN: 15c: 16 fa 1f e9  	cxbl	%o0, %o1, 1023
        !BIN: 160: 38 fa 1f e9  	cxbgu	%o0, %o1, 1023
        !BIN: 164: 18 fa 1f e9  	cxbleu	%o0, %o1, 1023
        !BIN: 168: 3a fa 1f e9  	cxbcc	%o0, %o1, 1023
        !BIN: 16c: 1a fa 1f e9  	cxbcs	%o0, %o1, 1023
        !BIN: 170: 3c fa 1f e9  	cxbpos	%o0, %o1, 1023
        !BIN: 174: 1c fa 1f e9  	cxbneg	%o0, %o1, 1023
        !BIN: 178: 3e fa 1f e9  	cxbvc	%o0, %o1, 1023
        !BIN: 17c: 1e fa 1f e9  	cxbvs	%o0, %o1, 1023
        cxbne  %o0, %o1, .-4
        cxbe   %o0, %o1, .-4
        cxbg   %o0, %o1, .-4
        cxble  %o0, %o1, .-4
        cxbge  %o0, %o1, .-4
        cxbl   %o0, %o1, .-4
        cxbgu  %o0, %o1, .-4
        cxbleu %o0, %o1, .-4
        cxbcc  %o0, %o1, .-4
        cxbcs  %o0, %o1, .-4
        cxbpos %o0, %o1, .-4
        cxbneg %o0, %o1, .-4
        cxbvc  %o0, %o1, .-4
        cxbvs  %o0, %o1, .-4

        !BIN: 180: 32 e2 20 21  	cxbne	%o0, 1, 1
        !BIN: 184: 12 e2 20 21  	cxbe	%o0, 1, 1
        !BIN: 188: 34 e2 20 21  	cxbg	%o0, 1, 1
        !BIN: 18c: 14 e2 20 21  	cxble	%o0, 1, 1
        !BIN: 190: 36 e2 20 21  	cxbge	%o0, 1, 1
        !BIN: 194: 16 e2 20 21  	cxbl	%o0, 1, 1
        !BIN: 198: 38 e2 20 21  	cxbgu	%o0, 1, 1
        !BIN: 19c: 18 e2 20 21  	cxbleu	%o0, 1, 1
        !BIN: 1a0: 3a e2 20 21  	cxbcc	%o0, 1, 1
        !BIN: 1a4: 1a e2 20 21  	cxbcs	%o0, 1, 1
        !BIN: 1a8: 3c e2 20 21  	cxbpos	%o0, 1, 1
        !BIN: 1ac: 1c e2 20 21  	cxbneg	%o0, 1, 1
        !BIN: 1b0: 3e e2 20 21  	cxbvc	%o0, 1, 1
        !BIN: 1b4: 1e e2 20 21  	cxbvs	%o0, 1, 1
        cxbne  %o0, 1, .+4
        cxbe   %o0, 1, .+4
        cxbg   %o0, 1, .+4
        cxble  %o0, 1, .+4
        cxbge  %o0, 1, .+4
        cxbl   %o0, 1, .+4
        cxbgu  %o0, 1, .+4
        cxbleu %o0, 1, .+4
        cxbcc  %o0, 1, .+4
        cxbcs  %o0, 1, .+4
        cxbpos %o0, 1, .+4
        cxbneg %o0, 1, .+4
        cxbvc  %o0, 1, .+4
        cxbvs  %o0, 1, .+4

        !BIN: 1b8: 32 fa 3f e1  	cxbne	%o0, 1, 1023
        !BIN: 1bc: 12 fa 3f e1  	cxbe	%o0, 1, 1023
        !BIN: 1c0: 34 fa 3f e1  	cxbg	%o0, 1, 1023
        !BIN: 1c4: 14 fa 3f e1  	cxble	%o0, 1, 1023
        !BIN: 1c8: 36 fa 3f e1  	cxbge	%o0, 1, 1023
        !BIN: 1cc: 16 fa 3f e1  	cxbl	%o0, 1, 1023
        !BIN: 1d0: 38 fa 3f e1  	cxbgu	%o0, 1, 1023
        !BIN: 1d4: 18 fa 3f e1  	cxbleu	%o0, 1, 1023
        !BIN: 1d8: 3a fa 3f e1  	cxbcc	%o0, 1, 1023
        !BIN: 1dc: 1a fa 3f e1  	cxbcs	%o0, 1, 1023
        !BIN: 1e0: 3c fa 3f e1  	cxbpos	%o0, 1, 1023
        !BIN: 1e4: 1c fa 3f e1  	cxbneg	%o0, 1, 1023
        !BIN: 1e8: 3e fa 3f e1  	cxbvc	%o0, 1, 1023
        !BIN: 1ec: 1e fa 3f e1  	cxbvs	%o0, 1, 1023
        cxbne  %o0, 1, .-4
        cxbe   %o0, 1, .-4
        cxbg   %o0, 1, .-4
        cxble  %o0, 1, .-4
        cxbge  %o0, 1, .-4
        cxbl   %o0, 1, .-4
        cxbgu  %o0, 1, .-4
        cxbleu %o0, 1, .-4
        cxbcc  %o0, 1, .-4
        cxbcs  %o0, 1, .-4
        cxbpos %o0, 1, .-4
        cxbneg %o0, 1, .-4
        cxbvc  %o0, 1, .-4
        cxbvs  %o0, 1, .-4
