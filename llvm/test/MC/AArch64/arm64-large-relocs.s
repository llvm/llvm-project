// RUN: llvm-mc -triple=arm64-linux-gnu -show-encoding -o - %s | FileCheck %s
// RUN: llvm-mc -triple=arm64-linux-gnu -show-encoding -filetype=obj -o - %s | llvm-objdump -r - | FileCheck --check-prefix=CHECK-OBJ %s

        movz x2, #:abs_g0:sym
        movk w3, #:abs_g0_nc:sym
        movz x2, #:prel_g0:sym
        movk w3, #:prel_g0_nc:sym
// CHECK: movz    x2, #:abs_g0:sym        // encoding: [0bAAA00010,A,0b100AAAAA,0xd2]
// CHECK-NEXT:                            //   fixup A - offset: 0, value: :abs_g0:sym, kind: fixup_aarch64_movw
// CHECK: movk     w3, #:abs_g0_nc:sym    // encoding: [0bAAA00011,A,0b100AAAAA,0x72]
// CHECK-NEXT:                            //   fixup A - offset: 0, value: :abs_g0_nc:sym, kind: fixup_aarch64_movw
// CHECK: movz    x2, #:prel_g0:sym       // encoding: [0bAAA00010,A,0b100AAAAA,0xd2]
// CHECK-NEXT:                            //   fixup A - offset: 0, value: :prel_g0:sym, kind: fixup_aarch64_movw
// CHECK: movk     w3, #:prel_g0_nc:sym   // encoding: [0bAAA00011,A,0b100AAAAA,0x72]
// CHECK-NEXT:                            //   fixup A - offset: 0, value: :prel_g0_nc:sym, kind: fixup_aarch64_movw

// CHECK-OBJ: 0 R_AARCH64_MOVW_UABS_G0 sym
// CHECK-OBJ: 4 R_AARCH64_MOVW_UABS_G0_NC sym
// CHECK-OBJ: 8 R_AARCH64_MOVW_PREL_G0 sym
// CHECK-OBJ: c R_AARCH64_MOVW_PREL_G0_NC sym

        movz x4, #:abs_g1:sym
        movk w5, #:abs_g1_nc:sym
        movz x4, #:prel_g1:sym
        movk w5, #:prel_g1_nc:sym
// CHECK: movz     x4, #:abs_g1:sym       // encoding: [0bAAA00100,A,0b101AAAAA,0xd2]
// CHECK-NEXT:                            //   fixup A - offset: 0, value: :abs_g1:sym, kind: fixup_aarch64_movw
// CHECK: movk     w5, #:abs_g1_nc:sym    // encoding: [0bAAA00101,A,0b101AAAAA,0x72]
// CHECK-NEXT:                            //   fixup A - offset: 0, value: :abs_g1_nc:sym, kind: fixup_aarch64_movw
// CHECK: movz     x4, #:prel_g1:sym      // encoding: [0bAAA00100,A,0b101AAAAA,0xd2]
// CHECK-NEXT:                            //   fixup A - offset: 0, value: :prel_g1:sym, kind: fixup_aarch64_movw
// CHECK: movk     w5, #:prel_g1_nc:sym   // encoding: [0bAAA00101,A,0b101AAAAA,0x72]
// CHECK-NEXT:                            //   fixup A - offset: 0, value: :prel_g1_nc:sym, kind: fixup_aarch64_movw

// CHECK-OBJ: 10 R_AARCH64_MOVW_UABS_G1 sym
// CHECK-OBJ: 14 R_AARCH64_MOVW_UABS_G1_NC sym
// CHECK-OBJ: 18 R_AARCH64_MOVW_PREL_G1 sym
// CHECK-OBJ: 1c R_AARCH64_MOVW_PREL_G1_NC sym

        movz x6, #:abs_g2:sym
        movk x7, #:abs_g2_nc:sym
        movz x6, #:prel_g2:sym
        movk x7, #:prel_g2_nc:sym
// CHECK: movz     x6, #:abs_g2:sym       // encoding: [0bAAA00110,A,0b110AAAAA,0xd2]
// CHECK-NEXT:                            //   fixup A - offset: 0, value: :abs_g2:sym, kind: fixup_aarch64_movw
// CHECK: movk     x7, #:abs_g2_nc:sym    // encoding: [0bAAA00111,A,0b110AAAAA,0xf2]
// CHECK-NEXT:                            //   fixup A - offset: 0, value: :abs_g2_nc:sym, kind: fixup_aarch64_movw
// CHECK: movz     x6, #:prel_g2:sym      // encoding: [0bAAA00110,A,0b110AAAAA,0xd2]
// CHECK-NEXT:                            //   fixup A - offset: 0, value: :prel_g2:sym, kind: fixup_aarch64_movw
// CHECK: movk     x7, #:prel_g2_nc:sym   // encoding: [0bAAA00111,A,0b110AAAAA,0xf2]
// CHECK-NEXT:                            //   fixup A - offset: 0, value: :prel_g2_nc:sym, kind: fixup_aarch64_movw

// CHECK-OBJ: 20 R_AARCH64_MOVW_UABS_G2 sym
// CHECK-OBJ: 24 R_AARCH64_MOVW_UABS_G2_NC sym
// CHECK-OBJ: 28 R_AARCH64_MOVW_PREL_G2 sym
// CHECK-OBJ: 2c R_AARCH64_MOVW_PREL_G2_NC sym

        movz x8, #:abs_g3:sym
        movz x8, #:prel_g3:sym
// CHECK: movz     x8, #:abs_g3:sym       // encoding: [0bAAA01000,A,0b111AAAAA,0xd2]
// CHECK-NEXT:                            //   fixup A - offset: 0, value: :abs_g3:sym, kind: fixup_aarch64_movw
// CHECK: movz     x8, #:prel_g3:sym      // encoding: [0bAAA01000,A,0b111AAAAA,0xd2]
// CHECK-NEXT:                            //   fixup A - offset: 0, value: :prel_g3:sym, kind: fixup_aarch64_movw

// CHECK-OBJ: 30 R_AARCH64_MOVW_UABS_G3 sym
// CHECK-OBJ: 34 R_AARCH64_MOVW_PREL_G3 sym
