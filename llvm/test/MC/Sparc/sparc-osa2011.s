! RUN: not llvm-mc %s -triple=sparcv9 -show-encoding 2>&1 | FileCheck %s --check-prefixes=NO-OSA2011 --implicit-check-not=error:
! RUN: llvm-mc %s -triple=sparcv9 -mattr=+osa2011 -show-encoding | FileCheck %s --check-prefixes=OSA2011

!! OSA 2011 instructions.

! NO-OSA2011: error: instruction requires a CPU feature not currently enabled
! OSA2011: cwbne	%o0, %o1, .BB0                  ! encoding: [0x32'A',0xc2'A',A,0x09'A']
cwbne %o0, %o1, .BB0
! NO-OSA2011: error: instruction requires a CPU feature not currently enabled
! OSA2011: cwbe	%o0, %o1, .BB0                  ! encoding: [0x12'A',0xc2'A',A,0x09'A']
cwbe %o0, %o1, .BB0
! NO-OSA2011: error: instruction requires a CPU feature not currently enabled
! OSA2011: cwbg	%o0, %o1, .BB0                  ! encoding: [0x34'A',0xc2'A',A,0x09'A']
cwbg %o0, %o1, .BB0
! NO-OSA2011: error: instruction requires a CPU feature not currently enabled
! OSA2011: cwble	%o0, %o1, .BB0                  ! encoding: [0x14'A',0xc2'A',A,0x09'A']
cwble %o0, %o1, .BB0
! NO-OSA2011: error: instruction requires a CPU feature not currently enabled
! OSA2011: cwbge	%o0, %o1, .BB0                  ! encoding: [0x36'A',0xc2'A',A,0x09'A']
cwbge %o0, %o1, .BB0
! NO-OSA2011: error: instruction requires a CPU feature not currently enabled
! OSA2011: cwbl	%o0, %o1, .BB0                  ! encoding: [0x16'A',0xc2'A',A,0x09'A']
cwbl %o0, %o1, .BB0
! NO-OSA2011: error: instruction requires a CPU feature not currently enabled
! OSA2011: cwbgu	%o0, %o1, .BB0                  ! encoding: [0x38'A',0xc2'A',A,0x09'A']
cwbgu %o0, %o1, .BB0
! NO-OSA2011: error: instruction requires a CPU feature not currently enabled
! OSA2011: cwbleu	%o0, %o1, .BB0                  ! encoding: [0x18'A',0xc2'A',A,0x09'A']
cwbleu %o0, %o1, .BB0
! NO-OSA2011: error: instruction requires a CPU feature not currently enabled
! OSA2011: cwbcc	%o0, %o1, .BB0                  ! encoding: [0x3a'A',0xc2'A',A,0x09'A']
cwbcc %o0, %o1, .BB0
! NO-OSA2011: error: instruction requires a CPU feature not currently enabled
! OSA2011: cwbcs	%o0, %o1, .BB0                  ! encoding: [0x1a'A',0xc2'A',A,0x09'A']
cwbcs %o0, %o1, .BB0
! NO-OSA2011: error: instruction requires a CPU feature not currently enabled
! OSA2011: cwbpos	%o0, %o1, .BB0                  ! encoding: [0x3c'A',0xc2'A',A,0x09'A']
cwbpos %o0, %o1, .BB0
! NO-OSA2011: error: instruction requires a CPU feature not currently enabled
! OSA2011: cwbneg	%o0, %o1, .BB0                  ! encoding: [0x1c'A',0xc2'A',A,0x09'A']
cwbneg %o0, %o1, .BB0
! NO-OSA2011: error: instruction requires a CPU feature not currently enabled
! OSA2011: cwbvc	%o0, %o1, .BB0                  ! encoding: [0x3e'A',0xc2'A',A,0x09'A']
cwbvc %o0, %o1, .BB0
! NO-OSA2011: error: instruction requires a CPU feature not currently enabled
! OSA2011: cwbvs	%o0, %o1, .BB0                  ! encoding: [0x1e'A',0xc2'A',A,0x09'A']
cwbvs %o0, %o1, .BB0

! NO-OSA2011: error: instruction requires a CPU feature not currently enabled
! OSA2011: cxbne	%o0, %o1, .BB0                  ! encoding: [0x32'A',0xe2'A',A,0x09'A']
cxbne %o0, %o1, .BB0
! NO-OSA2011: error: instruction requires a CPU feature not currently enabled
! OSA2011: cxbe	%o0, %o1, .BB0                  ! encoding: [0x12'A',0xe2'A',A,0x09'A']
cxbe %o0, %o1, .BB0
! NO-OSA2011: error: instruction requires a CPU feature not currently enabled
! OSA2011: cxbg	%o0, %o1, .BB0                  ! encoding: [0x34'A',0xe2'A',A,0x09'A']
cxbg %o0, %o1, .BB0
! NO-OSA2011: error: instruction requires a CPU feature not currently enabled
! OSA2011: cxble	%o0, %o1, .BB0                  ! encoding: [0x14'A',0xe2'A',A,0x09'A']
cxble %o0, %o1, .BB0
! NO-OSA2011: error: instruction requires a CPU feature not currently enabled
! OSA2011: cxbge	%o0, %o1, .BB0                  ! encoding: [0x36'A',0xe2'A',A,0x09'A']
cxbge %o0, %o1, .BB0
! NO-OSA2011: error: instruction requires a CPU feature not currently enabled
! OSA2011: cxbl	%o0, %o1, .BB0                  ! encoding: [0x16'A',0xe2'A',A,0x09'A']
cxbl %o0, %o1, .BB0
! NO-OSA2011: error: instruction requires a CPU feature not currently enabled
! OSA2011: cxbgu	%o0, %o1, .BB0                  ! encoding: [0x38'A',0xe2'A',A,0x09'A']
cxbgu %o0, %o1, .BB0
! NO-OSA2011: error: instruction requires a CPU feature not currently enabled
! OSA2011: cxbleu	%o0, %o1, .BB0                  ! encoding: [0x18'A',0xe2'A',A,0x09'A']
cxbleu %o0, %o1, .BB0
! NO-OSA2011: error: instruction requires a CPU feature not currently enabled
! OSA2011: cxbcc	%o0, %o1, .BB0                  ! encoding: [0x3a'A',0xe2'A',A,0x09'A']
cxbcc %o0, %o1, .BB0
! NO-OSA2011: error: instruction requires a CPU feature not currently enabled
! OSA2011: cxbcs	%o0, %o1, .BB0                  ! encoding: [0x1a'A',0xe2'A',A,0x09'A']
cxbcs %o0, %o1, .BB0
! NO-OSA2011: error: instruction requires a CPU feature not currently enabled
! OSA2011: cxbpos	%o0, %o1, .BB0                  ! encoding: [0x3c'A',0xe2'A',A,0x09'A']
cxbpos %o0, %o1, .BB0
! NO-OSA2011: error: instruction requires a CPU feature not currently enabled
! OSA2011: cxbneg	%o0, %o1, .BB0                  ! encoding: [0x1c'A',0xe2'A',A,0x09'A']
cxbneg %o0, %o1, .BB0
! NO-OSA2011: error: instruction requires a CPU feature not currently enabled
! OSA2011: cxbvc	%o0, %o1, .BB0                  ! encoding: [0x3e'A',0xe2'A',A,0x09'A']
cxbvc %o0, %o1, .BB0
! NO-OSA2011: error: instruction requires a CPU feature not currently enabled
! OSA2011: cxbvs	%o0, %o1, .BB0                  ! encoding: [0x1e'A',0xe2'A',A,0x09'A']
cxbvs %o0, %o1, .BB0

! NO-OSA2011: error: instruction requires a CPU feature not currently enabled
! OSA2011: cwbne	%o0, 1, .BB0                    ! encoding: [0x32'A',0xc2'A',0x20'A',0x01'A']
cwbne %o0, 1, .BB0
! NO-OSA2011: error: instruction requires a CPU feature not currently enabled
! OSA2011: cwbe	%o0, 1, .BB0                    ! encoding: [0x12'A',0xc2'A',0x20'A',0x01'A']
cwbe %o0, 1, .BB0
! NO-OSA2011: error: instruction requires a CPU feature not currently enabled
! OSA2011: cwbg	%o0, 1, .BB0                    ! encoding: [0x34'A',0xc2'A',0x20'A',0x01'A']
cwbg %o0, 1, .BB0
! NO-OSA2011: error: instruction requires a CPU feature not currently enabled
! OSA2011: cwble	%o0, 1, .BB0                    ! encoding: [0x14'A',0xc2'A',0x20'A',0x01'A']
cwble %o0, 1, .BB0
! NO-OSA2011: error: instruction requires a CPU feature not currently enabled
! OSA2011: cwbge	%o0, 1, .BB0                    ! encoding: [0x36'A',0xc2'A',0x20'A',0x01'A']
cwbge %o0, 1, .BB0
! NO-OSA2011: error: instruction requires a CPU feature not currently enabled
! OSA2011: cwbl	%o0, 1, .BB0                    ! encoding: [0x16'A',0xc2'A',0x20'A',0x01'A']
cwbl %o0, 1, .BB0
! NO-OSA2011: error: instruction requires a CPU feature not currently enabled
! OSA2011: cwbgu	%o0, 1, .BB0                    ! encoding: [0x38'A',0xc2'A',0x20'A',0x01'A']
cwbgu %o0, 1, .BB0
! NO-OSA2011: error: instruction requires a CPU feature not currently enabled
! OSA2011: cwbleu	%o0, 1, .BB0                    ! encoding: [0x18'A',0xc2'A',0x20'A',0x01'A']
cwbleu %o0, 1, .BB0
! NO-OSA2011: error: instruction requires a CPU feature not currently enabled
! OSA2011: cwbcc	%o0, 1, .BB0                    ! encoding: [0x3a'A',0xc2'A',0x20'A',0x01'A']
cwbcc %o0, 1, .BB0
! NO-OSA2011: error: instruction requires a CPU feature not currently enabled
! OSA2011: cwbcs	%o0, 1, .BB0                    ! encoding: [0x1a'A',0xc2'A',0x20'A',0x01'A']
cwbcs %o0, 1, .BB0
! NO-OSA2011: error: instruction requires a CPU feature not currently enabled
! OSA2011: cwbpos	%o0, 1, .BB0                    ! encoding: [0x3c'A',0xc2'A',0x20'A',0x01'A']
cwbpos %o0, 1, .BB0
! NO-OSA2011: error: instruction requires a CPU feature not currently enabled
! OSA2011: cwbneg	%o0, 1, .BB0                    ! encoding: [0x1c'A',0xc2'A',0x20'A',0x01'A']
cwbneg %o0, 1, .BB0
! NO-OSA2011: error: instruction requires a CPU feature not currently enabled
! OSA2011: cwbvc	%o0, 1, .BB0                    ! encoding: [0x3e'A',0xc2'A',0x20'A',0x01'A']
cwbvc %o0, 1, .BB0
! NO-OSA2011: error: instruction requires a CPU feature not currently enabled
! OSA2011: cwbvs	%o0, 1, .BB0                    ! encoding: [0x1e'A',0xc2'A',0x20'A',0x01'A']
cwbvs %o0, 1, .BB0

! NO-OSA2011: error: instruction requires a CPU feature not currently enabled
! OSA2011: cxbne	%o0, 1, .BB0                    ! encoding: [0x32'A',0xe2'A',0x20'A',0x01'A']
cxbne %o0, 1, .BB0
! NO-OSA2011: error: instruction requires a CPU feature not currently enabled
! OSA2011: cxbe	%o0, 1, .BB0                    ! encoding: [0x12'A',0xe2'A',0x20'A',0x01'A']
cxbe %o0, 1, .BB0
! NO-OSA2011: error: instruction requires a CPU feature not currently enabled
! OSA2011: cxbg	%o0, 1, .BB0                    ! encoding: [0x34'A',0xe2'A',0x20'A',0x01'A']
cxbg %o0, 1, .BB0
! NO-OSA2011: error: instruction requires a CPU feature not currently enabled
! OSA2011: cxble	%o0, 1, .BB0                    ! encoding: [0x14'A',0xe2'A',0x20'A',0x01'A']
cxble %o0, 1, .BB0
! NO-OSA2011: error: instruction requires a CPU feature not currently enabled
! OSA2011: cxbge	%o0, 1, .BB0                    ! encoding: [0x36'A',0xe2'A',0x20'A',0x01'A']
cxbge %o0, 1, .BB0
! NO-OSA2011: error: instruction requires a CPU feature not currently enabled
! OSA2011: cxbl	%o0, 1, .BB0                    ! encoding: [0x16'A',0xe2'A',0x20'A',0x01'A']
cxbl %o0, 1, .BB0
! NO-OSA2011: error: instruction requires a CPU feature not currently enabled
! OSA2011: cxbgu	%o0, 1, .BB0                    ! encoding: [0x38'A',0xe2'A',0x20'A',0x01'A']
cxbgu %o0, 1, .BB0
! NO-OSA2011: error: instruction requires a CPU feature not currently enabled
! OSA2011: cxbleu	%o0, 1, .BB0                    ! encoding: [0x18'A',0xe2'A',0x20'A',0x01'A']
cxbleu %o0, 1, .BB0
! NO-OSA2011: error: instruction requires a CPU feature not currently enabled
! OSA2011: cxbcc	%o0, 1, .BB0                    ! encoding: [0x3a'A',0xe2'A',0x20'A',0x01'A']
cxbcc %o0, 1, .BB0
! NO-OSA2011: error: instruction requires a CPU feature not currently enabled
! OSA2011: cxbcs	%o0, 1, .BB0                    ! encoding: [0x1a'A',0xe2'A',0x20'A',0x01'A']
cxbcs %o0, 1, .BB0
! NO-OSA2011: error: instruction requires a CPU feature not currently enabled
! OSA2011: cxbpos	%o0, 1, .BB0                    ! encoding: [0x3c'A',0xe2'A',0x20'A',0x01'A']
cxbpos %o0, 1, .BB0
! NO-OSA2011: error: instruction requires a CPU feature not currently enabled
! OSA2011: cxbneg	%o0, 1, .BB0                    ! encoding: [0x1c'A',0xe2'A',0x20'A',0x01'A']
cxbneg %o0, 1, .BB0
! NO-OSA2011: error: instruction requires a CPU feature not currently enabled
! OSA2011: cxbvc	%o0, 1, .BB0                    ! encoding: [0x3e'A',0xe2'A',0x20'A',0x01'A']
cxbvc %o0, 1, .BB0
! NO-OSA2011: error: instruction requires a CPU feature not currently enabled
! OSA2011: cxbvs	%o0, 1, .BB0                    ! encoding: [0x1e'A',0xe2'A',0x20'A',0x01'A']
cxbvs %o0, 1, .BB0

! NO-OSA2011: error: instruction requires a CPU feature not currently enabled
! OSA2011: fpmaddx %f0, %f2, %f4, %f6              ! encoding: [0x8d,0xb8,0x08,0x02]
fpmaddx %f0, %f2, %f4, %f6
! NO-OSA2011: error: instruction requires a CPU feature not currently enabled
! OSA2011: fpmaddxhi %f0, %f2, %f4, %f6            ! encoding: [0x8d,0xb8,0x08,0x82]
fpmaddxhi %f0, %f2, %f4, %f6

! NO-OSA2011: error: instruction requires a CPU feature not currently enabled
! OSA2011: pause	%o5                             ! encoding: [0xb7,0x80,0x00,0x0d]
pause %o5
! NO-OSA2011: error: instruction requires a CPU feature not currently enabled
! OSA2011: pause	5                               ! encoding: [0xb7,0x80,0x20,0x05]
pause 5
