// RUN: llvm-mc -triple i386 -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding -mattr=+avx10-v2-aux,+avx512vl %s | FileCheck %s

//
// Group A: PS->8bit truncating conversions
//

// vcvtps2bf8

// CHECK: vcvtps2bf8 xmm0, zmm1
// CHECK: encoding: [0x62,0xf5,0x7e,0x48,0x39,0xc1]
          vcvtps2bf8 xmm0, zmm1

// CHECK: vcvtps2bf8 xmm0, ymm1
// CHECK: encoding: [0x62,0xf5,0x7e,0x28,0x39,0xc1]
          vcvtps2bf8 xmm0, ymm1

// CHECK: vcvtps2bf8 xmm0, xmm1
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x39,0xc1]
          vcvtps2bf8 xmm0, xmm1

// CHECK: vcvtps2bf8 xmm0 {k1}, zmm1
// CHECK: encoding: [0x62,0xf5,0x7e,0x49,0x39,0xc1]
          vcvtps2bf8 xmm0 {k1}, zmm1

// CHECK: vcvtps2bf8 xmm0 {k1} {z}, zmm1
// CHECK: encoding: [0x62,0xf5,0x7e,0xc9,0x39,0xc1]
          vcvtps2bf8 xmm0 {k1} {z}, zmm1

// vcvtps2bf8s

// CHECK: vcvtps2bf8s xmm0, zmm1
// CHECK: encoding: [0x62,0xf5,0x7e,0x48,0x3b,0xc1]
          vcvtps2bf8s xmm0, zmm1

// CHECK: vcvtps2bf8s xmm0, ymm1
// CHECK: encoding: [0x62,0xf5,0x7e,0x28,0x3b,0xc1]
          vcvtps2bf8s xmm0, ymm1

// CHECK: vcvtps2bf8s xmm0, xmm1
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x3b,0xc1]
          vcvtps2bf8s xmm0, xmm1

// vcvtps2hf8

// CHECK: vcvtps2hf8 xmm0, zmm1
// CHECK: encoding: [0x62,0xf5,0x7e,0x48,0x38,0xc1]
          vcvtps2hf8 xmm0, zmm1

// CHECK: vcvtps2hf8 xmm0, ymm1
// CHECK: encoding: [0x62,0xf5,0x7e,0x28,0x38,0xc1]
          vcvtps2hf8 xmm0, ymm1

// CHECK: vcvtps2hf8 xmm0, xmm1
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x38,0xc1]
          vcvtps2hf8 xmm0, xmm1

// vcvtps2hf8s

// CHECK: vcvtps2hf8s xmm0, zmm1
// CHECK: encoding: [0x62,0xf5,0x7e,0x48,0x3a,0xc1]
          vcvtps2hf8s xmm0, zmm1

// CHECK: vcvtps2hf8s xmm0, ymm1
// CHECK: encoding: [0x62,0xf5,0x7e,0x28,0x3a,0xc1]
          vcvtps2hf8s xmm0, ymm1

// CHECK: vcvtps2hf8s xmm0, xmm1
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x3a,0xc1]
          vcvtps2hf8s xmm0, xmm1

// vcvtrops2hf8

// CHECK: vcvtrops2hf8 xmm0, zmm1
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x38,0xc1]
          vcvtrops2hf8 xmm0, zmm1

// CHECK: vcvtrops2hf8 xmm0, ymm1
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x38,0xc1]
          vcvtrops2hf8 xmm0, ymm1

// CHECK: vcvtrops2hf8 xmm0, xmm1
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x38,0xc1]
          vcvtrops2hf8 xmm0, xmm1

// vcvtrops2hf8s

// CHECK: vcvtrops2hf8s xmm0, zmm1
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x3a,0xc1]
          vcvtrops2hf8s xmm0, zmm1

// CHECK: vcvtrops2hf8s xmm0, ymm1
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x3a,0xc1]
          vcvtrops2hf8s xmm0, ymm1

// CHECK: vcvtrops2hf8s xmm0, xmm1
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x3a,0xc1]
          vcvtrops2hf8s xmm0, xmm1

//
// Group B: Bias PS->8bit conversions (3-operand)
//

// vcvtbiasps2bf8

// CHECK: vcvtbiasps2bf8 xmm0, zmm1, zmm2
// CHECK: encoding: [0x62,0xf5,0x74,0x48,0x39,0xc2]
          vcvtbiasps2bf8 xmm0, zmm1, zmm2

// CHECK: vcvtbiasps2bf8 xmm0, ymm1, ymm2
// CHECK: encoding: [0x62,0xf5,0x74,0x28,0x39,0xc2]
          vcvtbiasps2bf8 xmm0, ymm1, ymm2

// CHECK: vcvtbiasps2bf8 xmm0, xmm1, xmm2
// CHECK: encoding: [0x62,0xf5,0x74,0x08,0x39,0xc2]
          vcvtbiasps2bf8 xmm0, xmm1, xmm2

// vcvtbiasps2bf8s

// CHECK: vcvtbiasps2bf8s xmm0, zmm1, zmm2
// CHECK: encoding: [0x62,0xf5,0x74,0x48,0x3b,0xc2]
          vcvtbiasps2bf8s xmm0, zmm1, zmm2

// CHECK: vcvtbiasps2bf8s xmm0, ymm1, ymm2
// CHECK: encoding: [0x62,0xf5,0x74,0x28,0x3b,0xc2]
          vcvtbiasps2bf8s xmm0, ymm1, ymm2

// CHECK: vcvtbiasps2bf8s xmm0, xmm1, xmm2
// CHECK: encoding: [0x62,0xf5,0x74,0x08,0x3b,0xc2]
          vcvtbiasps2bf8s xmm0, xmm1, xmm2

// vcvtbiasps2hf8

// CHECK: vcvtbiasps2hf8 xmm0, zmm1, zmm2
// CHECK: encoding: [0x62,0xf5,0x74,0x48,0x38,0xc2]
          vcvtbiasps2hf8 xmm0, zmm1, zmm2

// CHECK: vcvtbiasps2hf8 xmm0, ymm1, ymm2
// CHECK: encoding: [0x62,0xf5,0x74,0x28,0x38,0xc2]
          vcvtbiasps2hf8 xmm0, ymm1, ymm2

// CHECK: vcvtbiasps2hf8 xmm0, xmm1, xmm2
// CHECK: encoding: [0x62,0xf5,0x74,0x08,0x38,0xc2]
          vcvtbiasps2hf8 xmm0, xmm1, xmm2

// vcvtbiasps2hf8s

// CHECK: vcvtbiasps2hf8s xmm0, zmm1, zmm2
// CHECK: encoding: [0x62,0xf5,0x74,0x48,0x3a,0xc2]
          vcvtbiasps2hf8s xmm0, zmm1, zmm2

// CHECK: vcvtbiasps2hf8s xmm0, ymm1, ymm2
// CHECK: encoding: [0x62,0xf5,0x74,0x28,0x3a,0xc2]
          vcvtbiasps2hf8s xmm0, ymm1, ymm2

// CHECK: vcvtbiasps2hf8s xmm0, xmm1, xmm2
// CHECK: encoding: [0x62,0xf5,0x74,0x08,0x3a,0xc2]
          vcvtbiasps2hf8s xmm0, xmm1, xmm2

//
// Group C: 8bit->PS expanding conversions
//

// vcvtbf82ps

// CHECK: vcvtbf82ps zmm0, xmm1
// CHECK: encoding: [0x62,0xf5,0xfc,0x48,0x36,0xc1]
          vcvtbf82ps zmm0, xmm1

// CHECK: vcvtbf82ps ymm0, xmm1
// CHECK: encoding: [0x62,0xf5,0xfc,0x28,0x36,0xc1]
          vcvtbf82ps ymm0, xmm1

// CHECK: vcvtbf82ps xmm0, xmm1
// CHECK: encoding: [0x62,0xf5,0xfc,0x08,0x36,0xc1]
          vcvtbf82ps xmm0, xmm1

// vcvthf82ps

// CHECK: vcvthf82ps zmm0, xmm1
// CHECK: encoding: [0x62,0xf5,0x7c,0x48,0x36,0xc1]
          vcvthf82ps zmm0, xmm1

// CHECK: vcvthf82ps ymm0, xmm1
// CHECK: encoding: [0x62,0xf5,0x7c,0x28,0x36,0xc1]
          vcvthf82ps ymm0, xmm1

// CHECK: vcvthf82ps xmm0, xmm1
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x36,0xc1]
          vcvthf82ps xmm0, xmm1

//
// Group D: BF8/HF8->BF4S truncations
//

// vcvtbf82bf4s

// CHECK: vcvtbf82bf4s ymm0, zmm1
// CHECK: encoding: [0x62,0xf5,0xfe,0x48,0x3d,0xc8]
          vcvtbf82bf4s ymm0, zmm1

// CHECK: vcvtbf82bf4s xmm0, ymm1
// CHECK: encoding: [0x62,0xf5,0xfe,0x28,0x3d,0xc8]
          vcvtbf82bf4s xmm0, ymm1

// CHECK: vcvtbf82bf4s xmm0, xmm1
// CHECK: encoding: [0x62,0xf5,0xfe,0x08,0x3d,0xc8]
          vcvtbf82bf4s xmm0, xmm1

// vcvthf82bf4s

// CHECK: vcvthf82bf4s ymm0, zmm1
// CHECK: encoding: [0x62,0xf5,0x7e,0x48,0x3d,0xc8]
          vcvthf82bf4s ymm0, zmm1

// CHECK: vcvthf82bf4s xmm0, ymm1
// CHECK: encoding: [0x62,0xf5,0x7e,0x28,0x3d,0xc8]
          vcvthf82bf4s xmm0, ymm1

// CHECK: vcvthf82bf4s xmm0, xmm1
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x3d,0xc8]
          vcvthf82bf4s xmm0, xmm1

//
// Group E: Same-size reg-only conversions (no masking)
//

// vcvtbf82bf6s

// CHECK: vcvtbf82bf6s zmm0, zmm1
// CHECK: encoding: [0x62,0xf5,0xfe,0x48,0x3e,0xc1]
          vcvtbf82bf6s zmm0, zmm1

// CHECK: vcvtbf82bf6s ymm0, ymm1
// CHECK: encoding: [0x62,0xf5,0xfe,0x28,0x3e,0xc1]
          vcvtbf82bf6s ymm0, ymm1

// CHECK: vcvtbf82bf6s xmm0, xmm1
// CHECK: encoding: [0x62,0xf5,0xfe,0x08,0x3e,0xc1]
          vcvtbf82bf6s xmm0, xmm1

// vcvthf82hf6s

// CHECK: vcvthf82hf6s zmm0, zmm1
// CHECK: encoding: [0x62,0xf5,0x7e,0x48,0x3c,0xc1]
          vcvthf82hf6s zmm0, zmm1

// CHECK: vcvthf82hf6s ymm0, ymm1
// CHECK: encoding: [0x62,0xf5,0x7e,0x28,0x3c,0xc1]
          vcvthf82hf6s ymm0, ymm1

// CHECK: vcvthf82hf6s xmm0, xmm1
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x3c,0xc1]
          vcvthf82hf6s xmm0, xmm1

//
// Group F: Expanding/same-size conversions with masking
//

// vcvtbf42hf8

// CHECK: vcvtbf42hf8 zmm0, ymm1
// CHECK: encoding: [0x62,0xf5,0x7c,0x48,0x37,0xc1]
          vcvtbf42hf8 zmm0, ymm1

// CHECK: vcvtbf42hf8 ymm0, xmm1
// CHECK: encoding: [0x62,0xf5,0x7c,0x28,0x37,0xc1]
          vcvtbf42hf8 ymm0, xmm1

// CHECK: vcvtbf42hf8 xmm0, xmm1
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x37,0xc1]
          vcvtbf42hf8 xmm0, xmm1

// vcvtbf62hf8

// CHECK: vcvtbf62hf8 zmm0, zmm1
// CHECK: encoding: [0x62,0xf5,0xfd,0x48,0x37,0xc1]
          vcvtbf62hf8 zmm0, zmm1

// CHECK: vcvtbf62hf8 ymm0, ymm1
// CHECK: encoding: [0x62,0xf5,0xfd,0x28,0x37,0xc1]
          vcvtbf62hf8 ymm0, ymm1

// CHECK: vcvtbf62hf8 xmm0, xmm1
// CHECK: encoding: [0x62,0xf5,0xfd,0x08,0x37,0xc1]
          vcvtbf62hf8 xmm0, xmm1

// vcvthf62hf8

// CHECK: vcvthf62hf8 zmm0, zmm1
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x37,0xc1]
          vcvthf62hf8 zmm0, zmm1

// CHECK: vcvthf62hf8 ymm0, ymm1
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x37,0xc1]
          vcvthf62hf8 ymm0, ymm1

// CHECK: vcvthf62hf8 xmm0, xmm1
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x37,0xc1]
          vcvthf62hf8 xmm0, xmm1

//
// Group G: VPMOVSSDB - Integer DWord->Byte signed saturation
//

// CHECK: vpmovssdb xmm0, zmm1
// CHECK: encoding: [0x62,0xf2,0x7e,0x48,0x41,0xc8]
          vpmovssdb xmm0, zmm1

// CHECK: vpmovssdb xmm0, ymm1
// CHECK: encoding: [0x62,0xf2,0x7e,0x28,0x41,0xc8]
          vpmovssdb xmm0, ymm1

// CHECK: vpmovssdb xmm0, xmm1
// CHECK: encoding: [0x62,0xf2,0x7e,0x08,0x41,0xc8]
          vpmovssdb xmm0, xmm1

//
// Group H: VUNPACKB - Byte unpack with immediate
//

// CHECK: vunpackb zmm0, zmm1, 1
// CHECK: encoding: [0x62,0xf3,0x7c,0x48,0x3d,0xc1,0x01]
          vunpackb zmm0, zmm1, 1

// CHECK: vunpackb ymm0, ymm1, 1
// CHECK: encoding: [0x62,0xf3,0x7c,0x28,0x3d,0xc1,0x01]
          vunpackb ymm0, ymm1, 1

// CHECK: vunpackb xmm0, xmm1, 1
// CHECK: encoding: [0x62,0xf3,0x7c,0x08,0x3d,0xc1,0x01]
          vunpackb xmm0, xmm1, 1
