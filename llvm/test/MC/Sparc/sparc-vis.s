! RUN: not llvm-mc %s -triple=sparcv9 -show-encoding 2>&1 | FileCheck %s --check-prefixes=NO-VIS --implicit-check-not=error:
! RUN: llvm-mc %s -triple=sparcv9 -mattr=+vis -show-encoding | FileCheck %s --check-prefixes=VIS

!! VIS 1 instructions.

! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: fpadd16 %f0, %f2, %f4                   ! encoding: [0x89,0xb0,0x0a,0x02]
fpadd16 %f0, %f2, %f4
! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: fpadd16s %f1, %f3, %f5                  ! encoding: [0x8b,0xb0,0x4a,0x23]
fpadd16s %f1, %f3, %f5
! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: fpadd32 %f0, %f2, %f4                   ! encoding: [0x89,0xb0,0x0a,0x42]
fpadd32 %f0, %f2, %f4
! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: fpadd32s %f1, %f3, %f5                  ! encoding: [0x8b,0xb0,0x4a,0x63]
fpadd32s %f1, %f3, %f5
! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: fpsub16 %f0, %f2, %f4                   ! encoding: [0x89,0xb0,0x0a,0x82]
fpsub16 %f0, %f2, %f4
! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: fpsub16s %f1, %f3, %f5                  ! encoding: [0x8b,0xb0,0x4a,0xa3]
fpsub16s %f1, %f3, %f5
! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: fpsub32 %f0, %f2, %f4                   ! encoding: [0x89,0xb0,0x0a,0xc2]
fpsub32 %f0, %f2, %f4
! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: fpsub32s %f1, %f3, %f5                  ! encoding: [0x8b,0xb0,0x4a,0xe3]
fpsub32s %f1, %f3, %f5

! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: fpack16 %f0, %f2                        ! encoding: [0x85,0xb0,0x07,0x60]
fpack16 %f0, %f2
! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: fpack32 %f0, %f2, %f4                   ! encoding: [0x89,0xb0,0x07,0x42]
fpack32 %f0, %f2, %f4
! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: fpackfix %f0, %f3                       ! encoding: [0x87,0xb0,0x07,0xa0]
fpackfix %f0, %f3
! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: fexpand %f1, %f2                        ! encoding: [0x85,0xb0,0x09,0xa1]
fexpand %f1, %f2
! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: fpmerge %f1, %f3, %f4                   ! encoding: [0x89,0xb0,0x49,0x63]
fpmerge %f1, %f3, %f4

! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: fmul8x16 %f1, %f2, %f4                  ! encoding: [0x89,0xb0,0x46,0x22]
fmul8x16 %f1, %f2, %f4
! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: fmul8x16au %f1, %f3, %f4                ! encoding: [0x89,0xb0,0x46,0x63]
fmul8x16au %f1, %f3, %f4
! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: fmul8x16al %f1, %f3, %f4                ! encoding: [0x89,0xb0,0x46,0xa3]
fmul8x16al %f1, %f3, %f4
! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: fmul8sux16 %f0, %f2, %f4                ! encoding: [0x89,0xb0,0x06,0xc2]
fmul8sux16 %f0, %f2, %f4
! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: fmul8ulx16 %f0, %f2, %f4                ! encoding: [0x89,0xb0,0x06,0xe2]
fmul8ulx16 %f0, %f2, %f4
! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: fmuld8sux16 %f1, %f3, %f4               ! encoding: [0x89,0xb0,0x47,0x03]
fmuld8sux16 %f1, %f3, %f4
! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: fmuld8ulx16 %f1, %f3, %f4               ! encoding: [0x89,0xb0,0x47,0x23]
fmuld8ulx16 %f1, %f3, %f4

! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: alignaddr %o0, %o1, %o2                 ! encoding: [0x95,0xb2,0x03,0x09]
alignaddr %o0, %o1, %o2
! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: alignaddrl %o0, %o1, %o2                ! encoding: [0x95,0xb2,0x03,0x49]
alignaddrl %o0, %o1, %o2
! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: faligndata %f0, %f2, %f4                ! encoding: [0x89,0xb0,0x09,0x02]
faligndata %f0, %f2, %f4

! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: fzero %f0                               ! encoding: [0x81,0xb0,0x0c,0x00]
fzero %f0
! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: fzeros %f1                              ! encoding: [0x83,0xb0,0x0c,0x20]
fzeros %f1
! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: fone %f0                                ! encoding: [0x81,0xb0,0x0f,0xc0]
fone %f0
! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: fones %f1                               ! encoding: [0x83,0xb0,0x0f,0xe0]
fones %f1
! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: fsrc1 %f0, %f2                          ! encoding: [0x85,0xb0,0x0e,0x80]
fsrc1 %f0, %f2
! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: fsrc1s %f1, %f3                         ! encoding: [0x87,0xb0,0x4e,0xa0]
fsrc1s %f1, %f3
! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: fsrc2 %f0, %f2                          ! encoding: [0x85,0xb0,0x0f,0x00]
fsrc2 %f0, %f2
! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: fsrc2s %f1, %f3                         ! encoding: [0x87,0xb0,0x0f,0x21]
fsrc2s %f1, %f3
! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: fnot1 %f0, %f2                          ! encoding: [0x85,0xb0,0x0d,0x40]
fnot1 %f0, %f2
! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: fnot1s %f1, %f3                         ! encoding: [0x87,0xb0,0x4d,0x60]
fnot1s %f1, %f3
! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: fnot2 %f0, %f2                          ! encoding: [0x85,0xb0,0x0c,0xc0]
fnot2 %f0, %f2
! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: fnot2s %f1, %f3                         ! encoding: [0x87,0xb0,0x0c,0xe1]
fnot2s %f1, %f3
! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: for %f0, %f2, %f4                       ! encoding: [0x89,0xb0,0x0f,0x82]
for %f0, %f2, %f4
! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: fors %f1, %f3, %f5                      ! encoding: [0x8b,0xb0,0x4f,0xa3]
fors %f1, %f3, %f5
! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: fnor %f0, %f2, %f4                      ! encoding: [0x89,0xb0,0x0c,0x42]
fnor %f0, %f2, %f4
! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: fnors %f1, %f3, %f5                     ! encoding: [0x8b,0xb0,0x4c,0x63]
fnors %f1, %f3, %f5
! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: fand %f0, %f2, %f4                      ! encoding: [0x89,0xb0,0x0e,0x02]
fand %f0, %f2, %f4
! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: fands %f1, %f3, %f5                     ! encoding: [0x8b,0xb0,0x4e,0x23]
fands %f1, %f3, %f5
! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: fnand %f0, %f2, %f4                     ! encoding: [0x89,0xb0,0x0d,0xc2]
fnand %f0, %f2, %f4
! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: fnands %f1, %f3, %f5                    ! encoding: [0x8b,0xb0,0x4d,0xe3]
fnands %f1, %f3, %f5
! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: fxor %f0, %f2, %f4                      ! encoding: [0x89,0xb0,0x0d,0x82]
fxor %f0, %f2, %f4
! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: fxors %f1, %f3, %f5                     ! encoding: [0x8b,0xb0,0x4d,0xa3]
fxors %f1, %f3, %f5
! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: fxnor %f0, %f2, %f4                     ! encoding: [0x89,0xb0,0x0e,0x42]
fxnor %f0, %f2, %f4
! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: fxnors %f1, %f3, %f5                    ! encoding: [0x8b,0xb0,0x4e,0x63]
fxnors %f1, %f3, %f5

! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: fornot1 %f0, %f2, %f4                   ! encoding: [0x89,0xb0,0x0f,0x42]
fornot1 %f0, %f2, %f4
! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: fornot1s %f1, %f3, %f5                  ! encoding: [0x8b,0xb0,0x4f,0x63]
fornot1s %f1, %f3, %f5
! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: fornot2 %f0, %f2, %f4                   ! encoding: [0x89,0xb0,0x0e,0xc2]
fornot2 %f0, %f2, %f4
! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: fornot2s %f1, %f3, %f5                  ! encoding: [0x8b,0xb0,0x4e,0xe3]
fornot2s %f1, %f3, %f5
! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: fandnot1 %f0, %f2, %f4                  ! encoding: [0x89,0xb0,0x0d,0x02]
fandnot1 %f0, %f2, %f4
! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: fandnot1s %f1, %f3, %f5                 ! encoding: [0x8b,0xb0,0x4d,0x23]
fandnot1s %f1, %f3, %f5
! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: fandnot2 %f0, %f2, %f4                  ! encoding: [0x89,0xb0,0x0c,0x82]
fandnot2 %f0, %f2, %f4
! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: fandnot2s %f1, %f3, %f5                 ! encoding: [0x8b,0xb0,0x4c,0xa3]
fandnot2s %f1, %f3, %f5

! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: fcmpgt16 %f0, %f2, %o0                  ! encoding: [0x91,0xb0,0x05,0x02]
fcmpgt16 %f0, %f2, %o0
! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: fcmpgt32 %f0, %f2, %o0                  ! encoding: [0x91,0xb0,0x05,0x82]
fcmpgt32 %f0, %f2, %o0
! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: fcmple16 %f0, %f2, %o0                  ! encoding: [0x91,0xb0,0x04,0x02]
fcmple16 %f0, %f2, %o0
! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: fcmple32 %f0, %f2, %o0                  ! encoding: [0x91,0xb0,0x04,0x82]
fcmple32 %f0, %f2, %o0
! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: fcmpne16 %f0, %f2, %o0                  ! encoding: [0x91,0xb0,0x04,0x42]
fcmpne16 %f0, %f2, %o0
! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: fcmpne32 %f0, %f2, %o0                  ! encoding: [0x91,0xb0,0x04,0xc2]
fcmpne32 %f0, %f2, %o0
! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: fcmpeq16 %f0, %f2, %o0                  ! encoding: [0x91,0xb0,0x05,0x42]
fcmpeq16 %f0, %f2, %o0
! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: fcmpeq32 %f0, %f2, %o0                  ! encoding: [0x91,0xb0,0x05,0xc2]
fcmpeq32 %f0, %f2, %o0

! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: edge8 %o0, %o1, %o2                     ! encoding: [0x95,0xb2,0x00,0x09]
edge8 %o0, %o1, %o2
! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: edge8l %o0, %o1, %o2                    ! encoding: [0x95,0xb2,0x00,0x49]
edge8l %o0, %o1, %o2
! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: edge16 %o0, %o1, %o2                    ! encoding: [0x95,0xb2,0x00,0x89]
edge16 %o0, %o1, %o2
! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: edge16l %o0, %o1, %o2                   ! encoding: [0x95,0xb2,0x00,0xc9]
edge16l %o0, %o1, %o2
! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: edge32 %o0, %o1, %o2                    ! encoding: [0x95,0xb2,0x01,0x09]
edge32 %o0, %o1, %o2
! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: edge32l %o0, %o1, %o2                   ! encoding: [0x95,0xb2,0x01,0x49]
edge32l %o0, %o1, %o2

! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: pdist %f0, %f2, %f4                     ! encoding: [0x89,0xb0,0x07,0xc2]
pdist %f0, %f2, %f4

! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: array8 %o0, %o1, %o2                    ! encoding: [0x95,0xb2,0x02,0x09]
array8 %o0, %o1, %o2
! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: array16 %o0, %o1, %o2                   ! encoding: [0x95,0xb2,0x02,0x49]
array16 %o0, %o1, %o2
! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: array32 %o0, %o1, %o2                   ! encoding: [0x95,0xb2,0x02,0x89]
array32 %o0, %o1, %o2

! NO-VIS: error: instruction requires a CPU feature not currently enabled
! VIS: shutdown                                ! encoding: [0x81,0xb0,0x10,0x00]
shutdown
