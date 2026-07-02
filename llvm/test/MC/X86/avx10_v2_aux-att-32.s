// RUN: llvm-mc -triple i386 --show-encoding -mattr=+avx10-v2-aux,+avx512vl %s | FileCheck %s

//
// Group A: PS->8bit truncating conversions
//

// vcvtps2bf8

// CHECK: vcvtps2bf8 %zmm1, %xmm0
// CHECK: encoding: [0x62,0xf5,0x7e,0x48,0x39,0xc1]
          vcvtps2bf8 %zmm1, %xmm0

// CHECK: vcvtps2bf8 %ymm1, %xmm0
// CHECK: encoding: [0x62,0xf5,0x7e,0x28,0x39,0xc1]
          vcvtps2bf8 %ymm1, %xmm0

// CHECK: vcvtps2bf8 %xmm1, %xmm0
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x39,0xc1]
          vcvtps2bf8 %xmm1, %xmm0

// CHECK: vcvtps2bf8 (%edi), %xmm0
// CHECK: encoding: [0x62,0xf5,0x7e,0x48,0x39,0x07]
          vcvtps2bf8 (%edi), %xmm0

// CHECK: vcvtps2bf8 %zmm1, %xmm0 {%k1}
// CHECK: encoding: [0x62,0xf5,0x7e,0x49,0x39,0xc1]
          vcvtps2bf8 %zmm1, %xmm0 {%k1}

// CHECK: vcvtps2bf8 %zmm1, %xmm0 {%k1} {z}
// CHECK: encoding: [0x62,0xf5,0x7e,0xc9,0x39,0xc1]
          vcvtps2bf8 %zmm1, %xmm0 {%k1} {z}

// vcvtps2bf8s

// CHECK: vcvtps2bf8s %zmm1, %xmm0
// CHECK: encoding: [0x62,0xf5,0x7e,0x48,0x3b,0xc1]
          vcvtps2bf8s %zmm1, %xmm0

// CHECK: vcvtps2bf8s %ymm1, %xmm0
// CHECK: encoding: [0x62,0xf5,0x7e,0x28,0x3b,0xc1]
          vcvtps2bf8s %ymm1, %xmm0

// CHECK: vcvtps2bf8s %xmm1, %xmm0
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x3b,0xc1]
          vcvtps2bf8s %xmm1, %xmm0

// CHECK: vcvtps2bf8s %zmm1, %xmm0 {%k1}
// CHECK: encoding: [0x62,0xf5,0x7e,0x49,0x3b,0xc1]
          vcvtps2bf8s %zmm1, %xmm0 {%k1}

// CHECK: vcvtps2bf8s %zmm1, %xmm0 {%k1} {z}
// CHECK: encoding: [0x62,0xf5,0x7e,0xc9,0x3b,0xc1]
          vcvtps2bf8s %zmm1, %xmm0 {%k1} {z}

// vcvtps2hf8

// CHECK: vcvtps2hf8 %zmm1, %xmm0
// CHECK: encoding: [0x62,0xf5,0x7e,0x48,0x38,0xc1]
          vcvtps2hf8 %zmm1, %xmm0

// CHECK: vcvtps2hf8 %ymm1, %xmm0
// CHECK: encoding: [0x62,0xf5,0x7e,0x28,0x38,0xc1]
          vcvtps2hf8 %ymm1, %xmm0

// CHECK: vcvtps2hf8 %xmm1, %xmm0
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x38,0xc1]
          vcvtps2hf8 %xmm1, %xmm0

// CHECK: vcvtps2hf8 %zmm1, %xmm0 {%k1}
// CHECK: encoding: [0x62,0xf5,0x7e,0x49,0x38,0xc1]
          vcvtps2hf8 %zmm1, %xmm0 {%k1}

// CHECK: vcvtps2hf8 %zmm1, %xmm0 {%k1} {z}
// CHECK: encoding: [0x62,0xf5,0x7e,0xc9,0x38,0xc1]
          vcvtps2hf8 %zmm1, %xmm0 {%k1} {z}

// vcvtps2hf8s

// CHECK: vcvtps2hf8s %zmm1, %xmm0
// CHECK: encoding: [0x62,0xf5,0x7e,0x48,0x3a,0xc1]
          vcvtps2hf8s %zmm1, %xmm0

// CHECK: vcvtps2hf8s %ymm1, %xmm0
// CHECK: encoding: [0x62,0xf5,0x7e,0x28,0x3a,0xc1]
          vcvtps2hf8s %ymm1, %xmm0

// CHECK: vcvtps2hf8s %xmm1, %xmm0
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x3a,0xc1]
          vcvtps2hf8s %xmm1, %xmm0

// CHECK: vcvtps2hf8s %zmm1, %xmm0 {%k1}
// CHECK: encoding: [0x62,0xf5,0x7e,0x49,0x3a,0xc1]
          vcvtps2hf8s %zmm1, %xmm0 {%k1}

// CHECK: vcvtps2hf8s %zmm1, %xmm0 {%k1} {z}
// CHECK: encoding: [0x62,0xf5,0x7e,0xc9,0x3a,0xc1]
          vcvtps2hf8s %zmm1, %xmm0 {%k1} {z}

// vcvtrops2hf8

// CHECK: vcvtrops2hf8 %zmm1, %xmm0
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x38,0xc1]
          vcvtrops2hf8 %zmm1, %xmm0

// CHECK: vcvtrops2hf8 %ymm1, %xmm0
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x38,0xc1]
          vcvtrops2hf8 %ymm1, %xmm0

// CHECK: vcvtrops2hf8 %xmm1, %xmm0
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x38,0xc1]
          vcvtrops2hf8 %xmm1, %xmm0

// CHECK: vcvtrops2hf8 %zmm1, %xmm0 {%k1}
// CHECK: encoding: [0x62,0xf5,0x7d,0x49,0x38,0xc1]
          vcvtrops2hf8 %zmm1, %xmm0 {%k1}

// CHECK: vcvtrops2hf8 %zmm1, %xmm0 {%k1} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0xc9,0x38,0xc1]
          vcvtrops2hf8 %zmm1, %xmm0 {%k1} {z}

// vcvtrops2hf8s

// CHECK: vcvtrops2hf8s %zmm1, %xmm0
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x3a,0xc1]
          vcvtrops2hf8s %zmm1, %xmm0

// CHECK: vcvtrops2hf8s %ymm1, %xmm0
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x3a,0xc1]
          vcvtrops2hf8s %ymm1, %xmm0

// CHECK: vcvtrops2hf8s %xmm1, %xmm0
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x3a,0xc1]
          vcvtrops2hf8s %xmm1, %xmm0

// CHECK: vcvtrops2hf8s %zmm1, %xmm0 {%k1}
// CHECK: encoding: [0x62,0xf5,0x7d,0x49,0x3a,0xc1]
          vcvtrops2hf8s %zmm1, %xmm0 {%k1}

// CHECK: vcvtrops2hf8s %zmm1, %xmm0 {%k1} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0xc9,0x3a,0xc1]
          vcvtrops2hf8s %zmm1, %xmm0 {%k1} {z}

//
// Group B: Bias PS->8bit conversions (3-operand)
//

// vcvtbiasps2bf8

// CHECK: vcvtbiasps2bf8 %zmm2, %zmm1, %xmm0
// CHECK: encoding: [0x62,0xf5,0x74,0x48,0x39,0xc2]
          vcvtbiasps2bf8 %zmm2, %zmm1, %xmm0

// CHECK: vcvtbiasps2bf8 %ymm2, %ymm1, %xmm0
// CHECK: encoding: [0x62,0xf5,0x74,0x28,0x39,0xc2]
          vcvtbiasps2bf8 %ymm2, %ymm1, %xmm0

// CHECK: vcvtbiasps2bf8 %xmm2, %xmm1, %xmm0
// CHECK: encoding: [0x62,0xf5,0x74,0x08,0x39,0xc2]
          vcvtbiasps2bf8 %xmm2, %xmm1, %xmm0

// CHECK: vcvtbiasps2bf8 %zmm2, %zmm1, %xmm0 {%k1}
// CHECK: encoding: [0x62,0xf5,0x74,0x49,0x39,0xc2]
          vcvtbiasps2bf8 %zmm2, %zmm1, %xmm0 {%k1}

// CHECK: vcvtbiasps2bf8 %zmm2, %zmm1, %xmm0 {%k1} {z}
// CHECK: encoding: [0x62,0xf5,0x74,0xc9,0x39,0xc2]
          vcvtbiasps2bf8 %zmm2, %zmm1, %xmm0 {%k1} {z}

// vcvtbiasps2bf8s

// CHECK: vcvtbiasps2bf8s %zmm2, %zmm1, %xmm0
// CHECK: encoding: [0x62,0xf5,0x74,0x48,0x3b,0xc2]
          vcvtbiasps2bf8s %zmm2, %zmm1, %xmm0

// CHECK: vcvtbiasps2bf8s %ymm2, %ymm1, %xmm0
// CHECK: encoding: [0x62,0xf5,0x74,0x28,0x3b,0xc2]
          vcvtbiasps2bf8s %ymm2, %ymm1, %xmm0

// CHECK: vcvtbiasps2bf8s %xmm2, %xmm1, %xmm0
// CHECK: encoding: [0x62,0xf5,0x74,0x08,0x3b,0xc2]
          vcvtbiasps2bf8s %xmm2, %xmm1, %xmm0

// CHECK: vcvtbiasps2bf8s %zmm2, %zmm1, %xmm0 {%k1}
// CHECK: encoding: [0x62,0xf5,0x74,0x49,0x3b,0xc2]
          vcvtbiasps2bf8s %zmm2, %zmm1, %xmm0 {%k1}

// CHECK: vcvtbiasps2bf8s %zmm2, %zmm1, %xmm0 {%k1} {z}
// CHECK: encoding: [0x62,0xf5,0x74,0xc9,0x3b,0xc2]
          vcvtbiasps2bf8s %zmm2, %zmm1, %xmm0 {%k1} {z}

// vcvtbiasps2hf8

// CHECK: vcvtbiasps2hf8 %zmm2, %zmm1, %xmm0
// CHECK: encoding: [0x62,0xf5,0x74,0x48,0x38,0xc2]
          vcvtbiasps2hf8 %zmm2, %zmm1, %xmm0

// CHECK: vcvtbiasps2hf8 %ymm2, %ymm1, %xmm0
// CHECK: encoding: [0x62,0xf5,0x74,0x28,0x38,0xc2]
          vcvtbiasps2hf8 %ymm2, %ymm1, %xmm0

// CHECK: vcvtbiasps2hf8 %xmm2, %xmm1, %xmm0
// CHECK: encoding: [0x62,0xf5,0x74,0x08,0x38,0xc2]
          vcvtbiasps2hf8 %xmm2, %xmm1, %xmm0

// CHECK: vcvtbiasps2hf8 %zmm2, %zmm1, %xmm0 {%k1}
// CHECK: encoding: [0x62,0xf5,0x74,0x49,0x38,0xc2]
          vcvtbiasps2hf8 %zmm2, %zmm1, %xmm0 {%k1}

// CHECK: vcvtbiasps2hf8 %zmm2, %zmm1, %xmm0 {%k1} {z}
// CHECK: encoding: [0x62,0xf5,0x74,0xc9,0x38,0xc2]
          vcvtbiasps2hf8 %zmm2, %zmm1, %xmm0 {%k1} {z}

// vcvtbiasps2hf8s

// CHECK: vcvtbiasps2hf8s %zmm2, %zmm1, %xmm0
// CHECK: encoding: [0x62,0xf5,0x74,0x48,0x3a,0xc2]
          vcvtbiasps2hf8s %zmm2, %zmm1, %xmm0

// CHECK: vcvtbiasps2hf8s %ymm2, %ymm1, %xmm0
// CHECK: encoding: [0x62,0xf5,0x74,0x28,0x3a,0xc2]
          vcvtbiasps2hf8s %ymm2, %ymm1, %xmm0

// CHECK: vcvtbiasps2hf8s %xmm2, %xmm1, %xmm0
// CHECK: encoding: [0x62,0xf5,0x74,0x08,0x3a,0xc2]
          vcvtbiasps2hf8s %xmm2, %xmm1, %xmm0

// CHECK: vcvtbiasps2hf8s %zmm2, %zmm1, %xmm0 {%k1}
// CHECK: encoding: [0x62,0xf5,0x74,0x49,0x3a,0xc2]
          vcvtbiasps2hf8s %zmm2, %zmm1, %xmm0 {%k1}

// CHECK: vcvtbiasps2hf8s %zmm2, %zmm1, %xmm0 {%k1} {z}
// CHECK: encoding: [0x62,0xf5,0x74,0xc9,0x3a,0xc2]
          vcvtbiasps2hf8s %zmm2, %zmm1, %xmm0 {%k1} {z}

//
// Group C: 8bit->PS expanding conversions
//

// vcvtbf82ps

// CHECK: vcvtbf82ps %xmm1, %zmm0
// CHECK: encoding: [0x62,0xf5,0xfc,0x48,0x36,0xc1]
          vcvtbf82ps %xmm1, %zmm0

// CHECK: vcvtbf82ps %xmm1, %ymm0
// CHECK: encoding: [0x62,0xf5,0xfc,0x28,0x36,0xc1]
          vcvtbf82ps %xmm1, %ymm0

// CHECK: vcvtbf82ps %xmm1, %xmm0
// CHECK: encoding: [0x62,0xf5,0xfc,0x08,0x36,0xc1]
          vcvtbf82ps %xmm1, %xmm0

// CHECK: vcvtbf82ps %xmm1, %zmm0 {%k1}
// CHECK: encoding: [0x62,0xf5,0xfc,0x49,0x36,0xc1]
          vcvtbf82ps %xmm1, %zmm0 {%k1}

// CHECK: vcvtbf82ps %xmm1, %zmm0 {%k1} {z}
// CHECK: encoding: [0x62,0xf5,0xfc,0xc9,0x36,0xc1]
          vcvtbf82ps %xmm1, %zmm0 {%k1} {z}

// vcvthf82ps

// CHECK: vcvthf82ps %xmm1, %zmm0
// CHECK: encoding: [0x62,0xf5,0x7c,0x48,0x36,0xc1]
          vcvthf82ps %xmm1, %zmm0

// CHECK: vcvthf82ps %xmm1, %ymm0
// CHECK: encoding: [0x62,0xf5,0x7c,0x28,0x36,0xc1]
          vcvthf82ps %xmm1, %ymm0

// CHECK: vcvthf82ps %xmm1, %xmm0
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x36,0xc1]
          vcvthf82ps %xmm1, %xmm0

// CHECK: vcvthf82ps %xmm1, %zmm0 {%k1}
// CHECK: encoding: [0x62,0xf5,0x7c,0x49,0x36,0xc1]
          vcvthf82ps %xmm1, %zmm0 {%k1}

// CHECK: vcvthf82ps %xmm1, %zmm0 {%k1} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0xc9,0x36,0xc1]
          vcvthf82ps %xmm1, %zmm0 {%k1} {z}

//
// Group D: BF8/HF8->BF4S truncations
//

// vcvtbf82bf4s

// CHECK: vcvtbf82bf4s %zmm1, %ymm0
// CHECK: encoding: [0x62,0xf5,0xfe,0x48,0x3d,0xc8]
          vcvtbf82bf4s %zmm1, %ymm0

// CHECK: vcvtbf82bf4s %ymm1, %xmm0
// CHECK: encoding: [0x62,0xf5,0xfe,0x28,0x3d,0xc8]
          vcvtbf82bf4s %ymm1, %xmm0

// CHECK: vcvtbf82bf4s %xmm1, %xmm0
// CHECK: encoding: [0x62,0xf5,0xfe,0x08,0x3d,0xc8]
          vcvtbf82bf4s %xmm1, %xmm0

// vcvthf82bf4s

// CHECK: vcvthf82bf4s %zmm1, %ymm0
// CHECK: encoding: [0x62,0xf5,0x7e,0x48,0x3d,0xc8]
          vcvthf82bf4s %zmm1, %ymm0

// CHECK: vcvthf82bf4s %ymm1, %xmm0
// CHECK: encoding: [0x62,0xf5,0x7e,0x28,0x3d,0xc8]
          vcvthf82bf4s %ymm1, %xmm0

// CHECK: vcvthf82bf4s %xmm1, %xmm0
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x3d,0xc8]
          vcvthf82bf4s %xmm1, %xmm0

//
// Group E: Same-size reg-only conversions (no masking)
//

// vcvtbf82bf6s

// CHECK: vcvtbf82bf6s %zmm1, %zmm0
// CHECK: encoding: [0x62,0xf5,0xfe,0x48,0x3e,0xc1]
          vcvtbf82bf6s %zmm1, %zmm0

// CHECK: vcvtbf82bf6s %ymm1, %ymm0
// CHECK: encoding: [0x62,0xf5,0xfe,0x28,0x3e,0xc1]
          vcvtbf82bf6s %ymm1, %ymm0

// CHECK: vcvtbf82bf6s %xmm1, %xmm0
// CHECK: encoding: [0x62,0xf5,0xfe,0x08,0x3e,0xc1]
          vcvtbf82bf6s %xmm1, %xmm0

// vcvthf82hf6s

// CHECK: vcvthf82hf6s %zmm1, %zmm0
// CHECK: encoding: [0x62,0xf5,0x7e,0x48,0x3c,0xc1]
          vcvthf82hf6s %zmm1, %zmm0

// CHECK: vcvthf82hf6s %ymm1, %ymm0
// CHECK: encoding: [0x62,0xf5,0x7e,0x28,0x3c,0xc1]
          vcvthf82hf6s %ymm1, %ymm0

// CHECK: vcvthf82hf6s %xmm1, %xmm0
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x3c,0xc1]
          vcvthf82hf6s %xmm1, %xmm0

//
// Group F: Expanding/same-size conversions with masking
//

// vcvtbf42hf8

// CHECK: vcvtbf42hf8 %ymm1, %zmm0
// CHECK: encoding: [0x62,0xf5,0x7c,0x48,0x37,0xc1]
          vcvtbf42hf8 %ymm1, %zmm0

// CHECK: vcvtbf42hf8 %xmm1, %ymm0
// CHECK: encoding: [0x62,0xf5,0x7c,0x28,0x37,0xc1]
          vcvtbf42hf8 %xmm1, %ymm0

// CHECK: vcvtbf42hf8 %xmm1, %xmm0
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x37,0xc1]
          vcvtbf42hf8 %xmm1, %xmm0

// CHECK: vcvtbf42hf8 %ymm1, %zmm0 {%k1}
// CHECK: encoding: [0x62,0xf5,0x7c,0x49,0x37,0xc1]
          vcvtbf42hf8 %ymm1, %zmm0 {%k1}

// CHECK: vcvtbf42hf8 %ymm1, %zmm0 {%k1} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0xc9,0x37,0xc1]
          vcvtbf42hf8 %ymm1, %zmm0 {%k1} {z}

// vcvtbf62hf8

// CHECK: vcvtbf62hf8 %zmm1, %zmm0
// CHECK: encoding: [0x62,0xf5,0xfd,0x48,0x37,0xc1]
          vcvtbf62hf8 %zmm1, %zmm0

// CHECK: vcvtbf62hf8 %ymm1, %ymm0
// CHECK: encoding: [0x62,0xf5,0xfd,0x28,0x37,0xc1]
          vcvtbf62hf8 %ymm1, %ymm0

// CHECK: vcvtbf62hf8 %xmm1, %xmm0
// CHECK: encoding: [0x62,0xf5,0xfd,0x08,0x37,0xc1]
          vcvtbf62hf8 %xmm1, %xmm0

// CHECK: vcvtbf62hf8 %zmm1, %zmm0 {%k1}
// CHECK: encoding: [0x62,0xf5,0xfd,0x49,0x37,0xc1]
          vcvtbf62hf8 %zmm1, %zmm0 {%k1}

// CHECK: vcvtbf62hf8 %zmm1, %zmm0 {%k1} {z}
// CHECK: encoding: [0x62,0xf5,0xfd,0xc9,0x37,0xc1]
          vcvtbf62hf8 %zmm1, %zmm0 {%k1} {z}

// vcvthf62hf8

// CHECK: vcvthf62hf8 %zmm1, %zmm0
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x37,0xc1]
          vcvthf62hf8 %zmm1, %zmm0

// CHECK: vcvthf62hf8 %ymm1, %ymm0
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x37,0xc1]
          vcvthf62hf8 %ymm1, %ymm0

// CHECK: vcvthf62hf8 %xmm1, %xmm0
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x37,0xc1]
          vcvthf62hf8 %xmm1, %xmm0

// CHECK: vcvthf62hf8 %zmm1, %zmm0 {%k1}
// CHECK: encoding: [0x62,0xf5,0x7d,0x49,0x37,0xc1]
          vcvthf62hf8 %zmm1, %zmm0 {%k1}

// CHECK: vcvthf62hf8 %zmm1, %zmm0 {%k1} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0xc9,0x37,0xc1]
          vcvthf62hf8 %zmm1, %zmm0 {%k1} {z}

//
// Group G: VPMOVSSDB - Integer DWord->Byte signed saturation
//

// CHECK: vpmovssdb %zmm1, %xmm0
// CHECK: encoding: [0x62,0xf2,0x7e,0x48,0x41,0xc8]
          vpmovssdb %zmm1, %xmm0

// CHECK: vpmovssdb %ymm1, %xmm0
// CHECK: encoding: [0x62,0xf2,0x7e,0x28,0x41,0xc8]
          vpmovssdb %ymm1, %xmm0

// CHECK: vpmovssdb %xmm1, %xmm0
// CHECK: encoding: [0x62,0xf2,0x7e,0x08,0x41,0xc8]
          vpmovssdb %xmm1, %xmm0

// CHECK: vpmovssdb %zmm1, %xmm0 {%k1}
// CHECK: encoding: [0x62,0xf2,0x7e,0x49,0x41,0xc8]
          vpmovssdb %zmm1, %xmm0 {%k1}

// CHECK: vpmovssdb %zmm1, %xmm0 {%k1} {z}
// CHECK: encoding: [0x62,0xf2,0x7e,0xc9,0x41,0xc8]
          vpmovssdb %zmm1, %xmm0 {%k1} {z}

//
// Group H: VUNPACKB - Byte unpack with immediate
//

// CHECK: vunpackb $1, %zmm1, %zmm0
// CHECK: encoding: [0x62,0xf3,0x7c,0x48,0x3d,0xc1,0x01]
          vunpackb $1, %zmm1, %zmm0

// CHECK: vunpackb $1, %ymm1, %ymm0
// CHECK: encoding: [0x62,0xf3,0x7c,0x28,0x3d,0xc1,0x01]
          vunpackb $1, %ymm1, %ymm0

// CHECK: vunpackb $1, %xmm1, %xmm0
// CHECK: encoding: [0x62,0xf3,0x7c,0x08,0x3d,0xc1,0x01]
          vunpackb $1, %xmm1, %xmm0

// CHECK: vunpackb $1, %zmm1, %zmm0 {%k1}
// CHECK: encoding: [0x62,0xf3,0x7c,0x49,0x3d,0xc1,0x01]
          vunpackb $1, %zmm1, %zmm0 {%k1}

// CHECK: vunpackb $1, %zmm1, %zmm0 {%k1} {z}
// CHECK: encoding: [0x62,0xf3,0x7c,0xc9,0x3d,0xc1,0x01]
          vunpackb $1, %zmm1, %zmm0 {%k1} {z}
