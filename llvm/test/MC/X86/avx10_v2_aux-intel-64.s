// RUN: llvm-mc -triple x86_64 -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding -mattr=+avx10v2aux,+avx512vl %s | FileCheck %s

// CHECK: vcvtps2bf8 xmm0, zmm1
// CHECK: encoding: [0x62,0xf5,0x7e,0x48,0x39,0xc1]
          vcvtps2bf8 xmm0, zmm1

// CHECK: vcvtps2bf8 xmm0, ymm1
// CHECK: encoding: [0x62,0xf5,0x7e,0x28,0x39,0xc1]
          vcvtps2bf8 xmm0, ymm1

// CHECK: vcvtps2bf8 xmm0, xmm1
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x39,0xc1]
          vcvtps2bf8 xmm0, xmm1

// CHECK: vcvtps2bf8 xmm0, zmmword ptr [rdi]
// CHECK: encoding: [0x62,0xf5,0x7e,0x48,0x39,0x07]
          vcvtps2bf8 xmm0, zmmword ptr [rdi]

// CHECK: vcvtps2bf8 xmm0, ymmword ptr [rdi]
// CHECK: encoding: [0x62,0xf5,0x7e,0x28,0x39,0x07]
          vcvtps2bf8 xmm0, ymmword ptr [rdi]

// CHECK: vcvtps2bf8 xmm0, xmmword ptr [rdi]
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x39,0x07]
          vcvtps2bf8 xmm0, xmmword ptr [rdi]

// CHECK: vcvtps2bf8 xmm0 {k1}, zmm1
// CHECK: encoding: [0x62,0xf5,0x7e,0x49,0x39,0xc1]
          vcvtps2bf8 xmm0 {k1}, zmm1

// CHECK: vcvtps2bf8 xmm0 {k1} {z}, zmm1
// CHECK: encoding: [0x62,0xf5,0x7e,0xc9,0x39,0xc1]
          vcvtps2bf8 xmm0 {k1} {z}, zmm1

// CHECK: vcvtps2bf8 xmm0, dword ptr [rdi]{1to16}
// CHECK: encoding: [0x62,0xf5,0x7e,0x58,0x39,0x07]
          vcvtps2bf8 xmm0, dword ptr [rdi]{1to16}

// CHECK: vcvtps2bf8 xmm0, dword ptr [rdi]{1to8}
// CHECK: encoding: [0x62,0xf5,0x7e,0x38,0x39,0x07]
          vcvtps2bf8 xmm0, dword ptr [rdi]{1to8}

// CHECK: vcvtps2bf8 xmm0, dword ptr [rdi]{1to4}
// CHECK: encoding: [0x62,0xf5,0x7e,0x18,0x39,0x07]
          vcvtps2bf8 xmm0, dword ptr [rdi]{1to4}

// CHECK: vcvtps2bf8s xmm0, zmm1
// CHECK: encoding: [0x62,0xf5,0x7e,0x48,0x3b,0xc1]
          vcvtps2bf8s xmm0, zmm1

// CHECK: vcvtps2bf8s xmm0, ymm1
// CHECK: encoding: [0x62,0xf5,0x7e,0x28,0x3b,0xc1]
          vcvtps2bf8s xmm0, ymm1

// CHECK: vcvtps2bf8s xmm0, xmm1
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x3b,0xc1]
          vcvtps2bf8s xmm0, xmm1

// CHECK: vcvtps2bf8s xmm0, zmmword ptr [rdi]
// CHECK: encoding: [0x62,0xf5,0x7e,0x48,0x3b,0x07]
          vcvtps2bf8s xmm0, zmmword ptr [rdi]

// CHECK: vcvtps2bf8s xmm0, ymmword ptr [rdi]
// CHECK: encoding: [0x62,0xf5,0x7e,0x28,0x3b,0x07]
          vcvtps2bf8s xmm0, ymmword ptr [rdi]

// CHECK: vcvtps2bf8s xmm0, xmmword ptr [rdi]
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x3b,0x07]
          vcvtps2bf8s xmm0, xmmword ptr [rdi]

// CHECK: vcvtps2bf8s xmm0 {k1}, zmm1
// CHECK: encoding: [0x62,0xf5,0x7e,0x49,0x3b,0xc1]
          vcvtps2bf8s xmm0 {k1}, zmm1

// CHECK: vcvtps2bf8s xmm0 {k1} {z}, zmm1
// CHECK: encoding: [0x62,0xf5,0x7e,0xc9,0x3b,0xc1]
          vcvtps2bf8s xmm0 {k1} {z}, zmm1

// CHECK: vcvtps2bf8s xmm0, dword ptr [rdi]{1to16}
// CHECK: encoding: [0x62,0xf5,0x7e,0x58,0x3b,0x07]
          vcvtps2bf8s xmm0, dword ptr [rdi]{1to16}

// CHECK: vcvtps2bf8s xmm0, dword ptr [rdi]{1to8}
// CHECK: encoding: [0x62,0xf5,0x7e,0x38,0x3b,0x07]
          vcvtps2bf8s xmm0, dword ptr [rdi]{1to8}

// CHECK: vcvtps2bf8s xmm0, dword ptr [rdi]{1to4}
// CHECK: encoding: [0x62,0xf5,0x7e,0x18,0x3b,0x07]
          vcvtps2bf8s xmm0, dword ptr [rdi]{1to4}

// CHECK: vcvtps2hf8 xmm0, zmm1
// CHECK: encoding: [0x62,0xf5,0x7e,0x48,0x38,0xc1]
          vcvtps2hf8 xmm0, zmm1

// CHECK: vcvtps2hf8 xmm0, ymm1
// CHECK: encoding: [0x62,0xf5,0x7e,0x28,0x38,0xc1]
          vcvtps2hf8 xmm0, ymm1

// CHECK: vcvtps2hf8 xmm0, xmm1
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x38,0xc1]
          vcvtps2hf8 xmm0, xmm1

// CHECK: vcvtps2hf8 xmm0, zmmword ptr [rdi]
// CHECK: encoding: [0x62,0xf5,0x7e,0x48,0x38,0x07]
          vcvtps2hf8 xmm0, zmmword ptr [rdi]

// CHECK: vcvtps2hf8 xmm0, ymmword ptr [rdi]
// CHECK: encoding: [0x62,0xf5,0x7e,0x28,0x38,0x07]
          vcvtps2hf8 xmm0, ymmword ptr [rdi]

// CHECK: vcvtps2hf8 xmm0, xmmword ptr [rdi]
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x38,0x07]
          vcvtps2hf8 xmm0, xmmword ptr [rdi]

// CHECK: vcvtps2hf8 xmm0 {k1}, zmm1
// CHECK: encoding: [0x62,0xf5,0x7e,0x49,0x38,0xc1]
          vcvtps2hf8 xmm0 {k1}, zmm1

// CHECK: vcvtps2hf8 xmm0 {k1} {z}, zmm1
// CHECK: encoding: [0x62,0xf5,0x7e,0xc9,0x38,0xc1]
          vcvtps2hf8 xmm0 {k1} {z}, zmm1

// CHECK: vcvtps2hf8 xmm0, dword ptr [rdi]{1to16}
// CHECK: encoding: [0x62,0xf5,0x7e,0x58,0x38,0x07]
          vcvtps2hf8 xmm0, dword ptr [rdi]{1to16}

// CHECK: vcvtps2hf8 xmm0, dword ptr [rdi]{1to8}
// CHECK: encoding: [0x62,0xf5,0x7e,0x38,0x38,0x07]
          vcvtps2hf8 xmm0, dword ptr [rdi]{1to8}

// CHECK: vcvtps2hf8 xmm0, dword ptr [rdi]{1to4}
// CHECK: encoding: [0x62,0xf5,0x7e,0x18,0x38,0x07]
          vcvtps2hf8 xmm0, dword ptr [rdi]{1to4}

// CHECK: vcvtps2hf8s xmm0, zmm1
// CHECK: encoding: [0x62,0xf5,0x7e,0x48,0x3a,0xc1]
          vcvtps2hf8s xmm0, zmm1

// CHECK: vcvtps2hf8s xmm0, ymm1
// CHECK: encoding: [0x62,0xf5,0x7e,0x28,0x3a,0xc1]
          vcvtps2hf8s xmm0, ymm1

// CHECK: vcvtps2hf8s xmm0, xmm1
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x3a,0xc1]
          vcvtps2hf8s xmm0, xmm1

// CHECK: vcvtps2hf8s xmm0, zmmword ptr [rdi]
// CHECK: encoding: [0x62,0xf5,0x7e,0x48,0x3a,0x07]
          vcvtps2hf8s xmm0, zmmword ptr [rdi]

// CHECK: vcvtps2hf8s xmm0, ymmword ptr [rdi]
// CHECK: encoding: [0x62,0xf5,0x7e,0x28,0x3a,0x07]
          vcvtps2hf8s xmm0, ymmword ptr [rdi]

// CHECK: vcvtps2hf8s xmm0, xmmword ptr [rdi]
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x3a,0x07]
          vcvtps2hf8s xmm0, xmmword ptr [rdi]

// CHECK: vcvtps2hf8s xmm0 {k1}, zmm1
// CHECK: encoding: [0x62,0xf5,0x7e,0x49,0x3a,0xc1]
          vcvtps2hf8s xmm0 {k1}, zmm1

// CHECK: vcvtps2hf8s xmm0 {k1} {z}, zmm1
// CHECK: encoding: [0x62,0xf5,0x7e,0xc9,0x3a,0xc1]
          vcvtps2hf8s xmm0 {k1} {z}, zmm1

// CHECK: vcvtps2hf8s xmm0, dword ptr [rdi]{1to16}
// CHECK: encoding: [0x62,0xf5,0x7e,0x58,0x3a,0x07]
          vcvtps2hf8s xmm0, dword ptr [rdi]{1to16}

// CHECK: vcvtps2hf8s xmm0, dword ptr [rdi]{1to8}
// CHECK: encoding: [0x62,0xf5,0x7e,0x38,0x3a,0x07]
          vcvtps2hf8s xmm0, dword ptr [rdi]{1to8}

// CHECK: vcvtps2hf8s xmm0, dword ptr [rdi]{1to4}
// CHECK: encoding: [0x62,0xf5,0x7e,0x18,0x3a,0x07]
          vcvtps2hf8s xmm0, dword ptr [rdi]{1to4}

// CHECK: vcvtrops2hf8 xmm0, zmm1
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x38,0xc1]
          vcvtrops2hf8 xmm0, zmm1

// CHECK: vcvtrops2hf8 xmm0, ymm1
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x38,0xc1]
          vcvtrops2hf8 xmm0, ymm1

// CHECK: vcvtrops2hf8 xmm0, xmm1
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x38,0xc1]
          vcvtrops2hf8 xmm0, xmm1

// CHECK: vcvtrops2hf8 xmm0, zmmword ptr [rdi]
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x38,0x07]
          vcvtrops2hf8 xmm0, zmmword ptr [rdi]

// CHECK: vcvtrops2hf8 xmm0, ymmword ptr [rdi]
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x38,0x07]
          vcvtrops2hf8 xmm0, ymmword ptr [rdi]

// CHECK: vcvtrops2hf8 xmm0, xmmword ptr [rdi]
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x38,0x07]
          vcvtrops2hf8 xmm0, xmmword ptr [rdi]

// CHECK: vcvtrops2hf8 xmm0 {k1}, zmm1
// CHECK: encoding: [0x62,0xf5,0x7d,0x49,0x38,0xc1]
          vcvtrops2hf8 xmm0 {k1}, zmm1

// CHECK: vcvtrops2hf8 xmm0 {k1} {z}, zmm1
// CHECK: encoding: [0x62,0xf5,0x7d,0xc9,0x38,0xc1]
          vcvtrops2hf8 xmm0 {k1} {z}, zmm1

// CHECK: vcvtrops2hf8 xmm0, dword ptr [rdi]{1to16}
// CHECK: encoding: [0x62,0xf5,0x7d,0x58,0x38,0x07]
          vcvtrops2hf8 xmm0, dword ptr [rdi]{1to16}

// CHECK: vcvtrops2hf8 xmm0, dword ptr [rdi]{1to8}
// CHECK: encoding: [0x62,0xf5,0x7d,0x38,0x38,0x07]
          vcvtrops2hf8 xmm0, dword ptr [rdi]{1to8}

// CHECK: vcvtrops2hf8 xmm0, dword ptr [rdi]{1to4}
// CHECK: encoding: [0x62,0xf5,0x7d,0x18,0x38,0x07]
          vcvtrops2hf8 xmm0, dword ptr [rdi]{1to4}

// CHECK: vcvtrops2hf8s xmm0, zmm1
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x3a,0xc1]
          vcvtrops2hf8s xmm0, zmm1

// CHECK: vcvtrops2hf8s xmm0, ymm1
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x3a,0xc1]
          vcvtrops2hf8s xmm0, ymm1

// CHECK: vcvtrops2hf8s xmm0, xmm1
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x3a,0xc1]
          vcvtrops2hf8s xmm0, xmm1

// CHECK: vcvtrops2hf8s xmm0, zmmword ptr [rdi]
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x3a,0x07]
          vcvtrops2hf8s xmm0, zmmword ptr [rdi]

// CHECK: vcvtrops2hf8s xmm0, ymmword ptr [rdi]
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x3a,0x07]
          vcvtrops2hf8s xmm0, ymmword ptr [rdi]

// CHECK: vcvtrops2hf8s xmm0, xmmword ptr [rdi]
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x3a,0x07]
          vcvtrops2hf8s xmm0, xmmword ptr [rdi]

// CHECK: vcvtrops2hf8s xmm0 {k1}, zmm1
// CHECK: encoding: [0x62,0xf5,0x7d,0x49,0x3a,0xc1]
          vcvtrops2hf8s xmm0 {k1}, zmm1

// CHECK: vcvtrops2hf8s xmm0 {k1} {z}, zmm1
// CHECK: encoding: [0x62,0xf5,0x7d,0xc9,0x3a,0xc1]
          vcvtrops2hf8s xmm0 {k1} {z}, zmm1

// CHECK: vcvtrops2hf8s xmm0, dword ptr [rdi]{1to16}
// CHECK: encoding: [0x62,0xf5,0x7d,0x58,0x3a,0x07]
          vcvtrops2hf8s xmm0, dword ptr [rdi]{1to16}

// CHECK: vcvtrops2hf8s xmm0, dword ptr [rdi]{1to8}
// CHECK: encoding: [0x62,0xf5,0x7d,0x38,0x3a,0x07]
          vcvtrops2hf8s xmm0, dword ptr [rdi]{1to8}

// CHECK: vcvtrops2hf8s xmm0, dword ptr [rdi]{1to4}
// CHECK: encoding: [0x62,0xf5,0x7d,0x18,0x3a,0x07]
          vcvtrops2hf8s xmm0, dword ptr [rdi]{1to4}

// CHECK: vcvtbiasps2bf8 xmm0, zmm1, zmm2
// CHECK: encoding: [0x62,0xf5,0x74,0x48,0x39,0xc2]
          vcvtbiasps2bf8 xmm0, zmm1, zmm2

// CHECK: vcvtbiasps2bf8 xmm0, ymm1, ymm2
// CHECK: encoding: [0x62,0xf5,0x74,0x28,0x39,0xc2]
          vcvtbiasps2bf8 xmm0, ymm1, ymm2

// CHECK: vcvtbiasps2bf8 xmm0, xmm1, xmm2
// CHECK: encoding: [0x62,0xf5,0x74,0x08,0x39,0xc2]
          vcvtbiasps2bf8 xmm0, xmm1, xmm2

// CHECK: vcvtbiasps2bf8 xmm0, zmm1, zmmword ptr [rdi]
// CHECK: encoding: [0x62,0xf5,0x74,0x48,0x39,0x07]
          vcvtbiasps2bf8 xmm0, zmm1, zmmword ptr [rdi]

// CHECK: vcvtbiasps2bf8 xmm0, ymm1, ymmword ptr [rdi]
// CHECK: encoding: [0x62,0xf5,0x74,0x28,0x39,0x07]
          vcvtbiasps2bf8 xmm0, ymm1, ymmword ptr [rdi]

// CHECK: vcvtbiasps2bf8 xmm0, xmm1, xmmword ptr [rdi]
// CHECK: encoding: [0x62,0xf5,0x74,0x08,0x39,0x07]
          vcvtbiasps2bf8 xmm0, xmm1, xmmword ptr [rdi]

// CHECK: vcvtbiasps2bf8 xmm0 {k1}, zmm1, zmm2
// CHECK: encoding: [0x62,0xf5,0x74,0x49,0x39,0xc2]
          vcvtbiasps2bf8 xmm0 {k1}, zmm1, zmm2

// CHECK: vcvtbiasps2bf8 xmm0 {k1} {z}, zmm1, zmm2
// CHECK: encoding: [0x62,0xf5,0x74,0xc9,0x39,0xc2]
          vcvtbiasps2bf8 xmm0 {k1} {z}, zmm1, zmm2

// CHECK: vcvtbiasps2bf8 xmm0, zmm1, dword ptr [rdi]{1to16}
// CHECK: encoding: [0x62,0xf5,0x74,0x58,0x39,0x07]
          vcvtbiasps2bf8 xmm0, zmm1, dword ptr [rdi]{1to16}

// CHECK: vcvtbiasps2bf8 xmm0, ymm1, dword ptr [rdi]{1to8}
// CHECK: encoding: [0x62,0xf5,0x74,0x38,0x39,0x07]
          vcvtbiasps2bf8 xmm0, ymm1, dword ptr [rdi]{1to8}

// CHECK: vcvtbiasps2bf8 xmm0, xmm1, dword ptr [rdi]{1to4}
// CHECK: encoding: [0x62,0xf5,0x74,0x18,0x39,0x07]
          vcvtbiasps2bf8 xmm0, xmm1, dword ptr [rdi]{1to4}

// CHECK: vcvtbiasps2bf8s xmm0, zmm1, zmm2
// CHECK: encoding: [0x62,0xf5,0x74,0x48,0x3b,0xc2]
          vcvtbiasps2bf8s xmm0, zmm1, zmm2

// CHECK: vcvtbiasps2bf8s xmm0, ymm1, ymm2
// CHECK: encoding: [0x62,0xf5,0x74,0x28,0x3b,0xc2]
          vcvtbiasps2bf8s xmm0, ymm1, ymm2

// CHECK: vcvtbiasps2bf8s xmm0, xmm1, xmm2
// CHECK: encoding: [0x62,0xf5,0x74,0x08,0x3b,0xc2]
          vcvtbiasps2bf8s xmm0, xmm1, xmm2

// CHECK: vcvtbiasps2bf8s xmm0, zmm1, zmmword ptr [rdi]
// CHECK: encoding: [0x62,0xf5,0x74,0x48,0x3b,0x07]
          vcvtbiasps2bf8s xmm0, zmm1, zmmword ptr [rdi]

// CHECK: vcvtbiasps2bf8s xmm0, ymm1, ymmword ptr [rdi]
// CHECK: encoding: [0x62,0xf5,0x74,0x28,0x3b,0x07]
          vcvtbiasps2bf8s xmm0, ymm1, ymmword ptr [rdi]

// CHECK: vcvtbiasps2bf8s xmm0, xmm1, xmmword ptr [rdi]
// CHECK: encoding: [0x62,0xf5,0x74,0x08,0x3b,0x07]
          vcvtbiasps2bf8s xmm0, xmm1, xmmword ptr [rdi]

// CHECK: vcvtbiasps2bf8s xmm0 {k1}, zmm1, zmm2
// CHECK: encoding: [0x62,0xf5,0x74,0x49,0x3b,0xc2]
          vcvtbiasps2bf8s xmm0 {k1}, zmm1, zmm2

// CHECK: vcvtbiasps2bf8s xmm0 {k1} {z}, zmm1, zmm2
// CHECK: encoding: [0x62,0xf5,0x74,0xc9,0x3b,0xc2]
          vcvtbiasps2bf8s xmm0 {k1} {z}, zmm1, zmm2

// CHECK: vcvtbiasps2bf8s xmm0, zmm1, dword ptr [rdi]{1to16}
// CHECK: encoding: [0x62,0xf5,0x74,0x58,0x3b,0x07]
          vcvtbiasps2bf8s xmm0, zmm1, dword ptr [rdi]{1to16}

// CHECK: vcvtbiasps2bf8s xmm0, ymm1, dword ptr [rdi]{1to8}
// CHECK: encoding: [0x62,0xf5,0x74,0x38,0x3b,0x07]
          vcvtbiasps2bf8s xmm0, ymm1, dword ptr [rdi]{1to8}

// CHECK: vcvtbiasps2bf8s xmm0, xmm1, dword ptr [rdi]{1to4}
// CHECK: encoding: [0x62,0xf5,0x74,0x18,0x3b,0x07]
          vcvtbiasps2bf8s xmm0, xmm1, dword ptr [rdi]{1to4}

// CHECK: vcvtbiasps2hf8 xmm0, zmm1, zmm2
// CHECK: encoding: [0x62,0xf5,0x74,0x48,0x38,0xc2]
          vcvtbiasps2hf8 xmm0, zmm1, zmm2

// CHECK: vcvtbiasps2hf8 xmm0, ymm1, ymm2
// CHECK: encoding: [0x62,0xf5,0x74,0x28,0x38,0xc2]
          vcvtbiasps2hf8 xmm0, ymm1, ymm2

// CHECK: vcvtbiasps2hf8 xmm0, xmm1, xmm2
// CHECK: encoding: [0x62,0xf5,0x74,0x08,0x38,0xc2]
          vcvtbiasps2hf8 xmm0, xmm1, xmm2

// CHECK: vcvtbiasps2hf8 xmm0, zmm1, zmmword ptr [rdi]
// CHECK: encoding: [0x62,0xf5,0x74,0x48,0x38,0x07]
          vcvtbiasps2hf8 xmm0, zmm1, zmmword ptr [rdi]

// CHECK: vcvtbiasps2hf8 xmm0, ymm1, ymmword ptr [rdi]
// CHECK: encoding: [0x62,0xf5,0x74,0x28,0x38,0x07]
          vcvtbiasps2hf8 xmm0, ymm1, ymmword ptr [rdi]

// CHECK: vcvtbiasps2hf8 xmm0, xmm1, xmmword ptr [rdi]
// CHECK: encoding: [0x62,0xf5,0x74,0x08,0x38,0x07]
          vcvtbiasps2hf8 xmm0, xmm1, xmmword ptr [rdi]

// CHECK: vcvtbiasps2hf8 xmm0 {k1}, zmm1, zmm2
// CHECK: encoding: [0x62,0xf5,0x74,0x49,0x38,0xc2]
          vcvtbiasps2hf8 xmm0 {k1}, zmm1, zmm2

// CHECK: vcvtbiasps2hf8 xmm0 {k1} {z}, zmm1, zmm2
// CHECK: encoding: [0x62,0xf5,0x74,0xc9,0x38,0xc2]
          vcvtbiasps2hf8 xmm0 {k1} {z}, zmm1, zmm2

// CHECK: vcvtbiasps2hf8 xmm0, zmm1, dword ptr [rdi]{1to16}
// CHECK: encoding: [0x62,0xf5,0x74,0x58,0x38,0x07]
          vcvtbiasps2hf8 xmm0, zmm1, dword ptr [rdi]{1to16}

// CHECK: vcvtbiasps2hf8 xmm0, ymm1, dword ptr [rdi]{1to8}
// CHECK: encoding: [0x62,0xf5,0x74,0x38,0x38,0x07]
          vcvtbiasps2hf8 xmm0, ymm1, dword ptr [rdi]{1to8}

// CHECK: vcvtbiasps2hf8 xmm0, xmm1, dword ptr [rdi]{1to4}
// CHECK: encoding: [0x62,0xf5,0x74,0x18,0x38,0x07]
          vcvtbiasps2hf8 xmm0, xmm1, dword ptr [rdi]{1to4}

// CHECK: vcvtbiasps2hf8s xmm0, zmm1, zmm2
// CHECK: encoding: [0x62,0xf5,0x74,0x48,0x3a,0xc2]
          vcvtbiasps2hf8s xmm0, zmm1, zmm2

// CHECK: vcvtbiasps2hf8s xmm0, ymm1, ymm2
// CHECK: encoding: [0x62,0xf5,0x74,0x28,0x3a,0xc2]
          vcvtbiasps2hf8s xmm0, ymm1, ymm2

// CHECK: vcvtbiasps2hf8s xmm0, xmm1, xmm2
// CHECK: encoding: [0x62,0xf5,0x74,0x08,0x3a,0xc2]
          vcvtbiasps2hf8s xmm0, xmm1, xmm2

// CHECK: vcvtbiasps2hf8s xmm0, zmm1, zmmword ptr [rdi]
// CHECK: encoding: [0x62,0xf5,0x74,0x48,0x3a,0x07]
          vcvtbiasps2hf8s xmm0, zmm1, zmmword ptr [rdi]

// CHECK: vcvtbiasps2hf8s xmm0, ymm1, ymmword ptr [rdi]
// CHECK: encoding: [0x62,0xf5,0x74,0x28,0x3a,0x07]
          vcvtbiasps2hf8s xmm0, ymm1, ymmword ptr [rdi]

// CHECK: vcvtbiasps2hf8s xmm0, xmm1, xmmword ptr [rdi]
// CHECK: encoding: [0x62,0xf5,0x74,0x08,0x3a,0x07]
          vcvtbiasps2hf8s xmm0, xmm1, xmmword ptr [rdi]

// CHECK: vcvtbiasps2hf8s xmm0 {k1}, zmm1, zmm2
// CHECK: encoding: [0x62,0xf5,0x74,0x49,0x3a,0xc2]
          vcvtbiasps2hf8s xmm0 {k1}, zmm1, zmm2

// CHECK: vcvtbiasps2hf8s xmm0 {k1} {z}, zmm1, zmm2
// CHECK: encoding: [0x62,0xf5,0x74,0xc9,0x3a,0xc2]
          vcvtbiasps2hf8s xmm0 {k1} {z}, zmm1, zmm2

// CHECK: vcvtbiasps2hf8s xmm0, zmm1, dword ptr [rdi]{1to16}
// CHECK: encoding: [0x62,0xf5,0x74,0x58,0x3a,0x07]
          vcvtbiasps2hf8s xmm0, zmm1, dword ptr [rdi]{1to16}

// CHECK: vcvtbiasps2hf8s xmm0, ymm1, dword ptr [rdi]{1to8}
// CHECK: encoding: [0x62,0xf5,0x74,0x38,0x3a,0x07]
          vcvtbiasps2hf8s xmm0, ymm1, dword ptr [rdi]{1to8}

// CHECK: vcvtbiasps2hf8s xmm0, xmm1, dword ptr [rdi]{1to4}
// CHECK: encoding: [0x62,0xf5,0x74,0x18,0x3a,0x07]
          vcvtbiasps2hf8s xmm0, xmm1, dword ptr [rdi]{1to4}

// CHECK: vcvtbf82ps zmm0, xmm1
// CHECK: encoding: [0x62,0xf5,0xfc,0x48,0x36,0xc1]
          vcvtbf82ps zmm0, xmm1

// CHECK: vcvtbf82ps ymm0, xmm1
// CHECK: encoding: [0x62,0xf5,0xfc,0x28,0x36,0xc1]
          vcvtbf82ps ymm0, xmm1

// CHECK: vcvtbf82ps xmm0, xmm1
// CHECK: encoding: [0x62,0xf5,0xfc,0x08,0x36,0xc1]
          vcvtbf82ps xmm0, xmm1

// CHECK: vcvtbf82ps zmm0, xmmword ptr [rdi]
// CHECK: encoding: [0x62,0xf5,0xfc,0x48,0x36,0x07]
          vcvtbf82ps zmm0, xmmword ptr [rdi]

// CHECK: vcvtbf82ps ymm0, qword ptr [rdi]
// CHECK: encoding: [0x62,0xf5,0xfc,0x28,0x36,0x07]
          vcvtbf82ps ymm0, qword ptr [rdi]

// CHECK: vcvtbf82ps xmm0, dword ptr [rdi]
// CHECK: encoding: [0x62,0xf5,0xfc,0x08,0x36,0x07]
          vcvtbf82ps xmm0, dword ptr [rdi]

// CHECK: vcvtbf82ps zmm0 {k1}, xmm1
// CHECK: encoding: [0x62,0xf5,0xfc,0x49,0x36,0xc1]
          vcvtbf82ps zmm0 {k1}, xmm1

// CHECK: vcvtbf82ps zmm0 {k1} {z}, xmm1
// CHECK: encoding: [0x62,0xf5,0xfc,0xc9,0x36,0xc1]
          vcvtbf82ps zmm0 {k1} {z}, xmm1

// CHECK: vcvthf82ps zmm0, xmm1
// CHECK: encoding: [0x62,0xf5,0x7c,0x48,0x36,0xc1]
          vcvthf82ps zmm0, xmm1

// CHECK: vcvthf82ps ymm0, xmm1
// CHECK: encoding: [0x62,0xf5,0x7c,0x28,0x36,0xc1]
          vcvthf82ps ymm0, xmm1

// CHECK: vcvthf82ps xmm0, xmm1
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x36,0xc1]
          vcvthf82ps xmm0, xmm1

// CHECK: vcvthf82ps zmm0, xmmword ptr [rdi]
// CHECK: encoding: [0x62,0xf5,0x7c,0x48,0x36,0x07]
          vcvthf82ps zmm0, xmmword ptr [rdi]

// CHECK: vcvthf82ps ymm0, qword ptr [rdi]
// CHECK: encoding: [0x62,0xf5,0x7c,0x28,0x36,0x07]
          vcvthf82ps ymm0, qword ptr [rdi]

// CHECK: vcvthf82ps xmm0, dword ptr [rdi]
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x36,0x07]
          vcvthf82ps xmm0, dword ptr [rdi]

// CHECK: vcvthf82ps zmm0 {k1}, xmm1
// CHECK: encoding: [0x62,0xf5,0x7c,0x49,0x36,0xc1]
          vcvthf82ps zmm0 {k1}, xmm1

// CHECK: vcvthf82ps zmm0 {k1} {z}, xmm1
// CHECK: encoding: [0x62,0xf5,0x7c,0xc9,0x36,0xc1]
          vcvthf82ps zmm0 {k1} {z}, xmm1

// CHECK: vcvtbf82bf4s ymm0, zmm1
// CHECK: encoding: [0x62,0xf5,0xfe,0x48,0x3d,0xc8]
          vcvtbf82bf4s ymm0, zmm1

// CHECK: vcvtbf82bf4s xmm0, ymm1
// CHECK: encoding: [0x62,0xf5,0xfe,0x28,0x3d,0xc8]
          vcvtbf82bf4s xmm0, ymm1

// CHECK: vcvtbf82bf4s xmm0, xmm1
// CHECK: encoding: [0x62,0xf5,0xfe,0x08,0x3d,0xc8]
          vcvtbf82bf4s xmm0, xmm1

// CHECK: vcvtbf82bf4s ymmword ptr [rdi], zmm1
// CHECK: encoding: [0x62,0xf5,0xfe,0x48,0x3d,0x0f]
          vcvtbf82bf4s ymmword ptr [rdi], zmm1

// CHECK: vcvtbf82bf4s xmmword ptr [rdi], ymm1
// CHECK: encoding: [0x62,0xf5,0xfe,0x28,0x3d,0x0f]
          vcvtbf82bf4s xmmword ptr [rdi], ymm1

// CHECK: vcvtbf82bf4s qword ptr [rdi], xmm1
// CHECK: encoding: [0x62,0xf5,0xfe,0x08,0x3d,0x0f]
          vcvtbf82bf4s qword ptr [rdi], xmm1

// CHECK: vcvthf82bf4s ymm0, zmm1
// CHECK: encoding: [0x62,0xf5,0x7e,0x48,0x3d,0xc8]
          vcvthf82bf4s ymm0, zmm1

// CHECK: vcvthf82bf4s xmm0, ymm1
// CHECK: encoding: [0x62,0xf5,0x7e,0x28,0x3d,0xc8]
          vcvthf82bf4s xmm0, ymm1

// CHECK: vcvthf82bf4s xmm0, xmm1
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x3d,0xc8]
          vcvthf82bf4s xmm0, xmm1

// CHECK: vcvthf82bf4s ymmword ptr [rdi], zmm1
// CHECK: encoding: [0x62,0xf5,0x7e,0x48,0x3d,0x0f]
          vcvthf82bf4s ymmword ptr [rdi], zmm1

// CHECK: vcvthf82bf4s xmmword ptr [rdi], ymm1
// CHECK: encoding: [0x62,0xf5,0x7e,0x28,0x3d,0x0f]
          vcvthf82bf4s xmmword ptr [rdi], ymm1

// CHECK: vcvthf82bf4s qword ptr [rdi], xmm1
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x3d,0x0f]
          vcvthf82bf4s qword ptr [rdi], xmm1

// CHECK: vcvtbf82bf6s zmm0, zmm1
// CHECK: encoding: [0x62,0xf5,0xfe,0x48,0x3e,0xc1]
          vcvtbf82bf6s zmm0, zmm1

// CHECK: vcvtbf82bf6s ymm0, ymm1
// CHECK: encoding: [0x62,0xf5,0xfe,0x28,0x3e,0xc1]
          vcvtbf82bf6s ymm0, ymm1

// CHECK: vcvtbf82bf6s xmm0, xmm1
// CHECK: encoding: [0x62,0xf5,0xfe,0x08,0x3e,0xc1]
          vcvtbf82bf6s xmm0, xmm1

// CHECK: vcvthf82hf6s zmm0, zmm1
// CHECK: encoding: [0x62,0xf5,0x7e,0x48,0x3c,0xc1]
          vcvthf82hf6s zmm0, zmm1

// CHECK: vcvthf82hf6s ymm0, ymm1
// CHECK: encoding: [0x62,0xf5,0x7e,0x28,0x3c,0xc1]
          vcvthf82hf6s ymm0, ymm1

// CHECK: vcvthf82hf6s xmm0, xmm1
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x3c,0xc1]
          vcvthf82hf6s xmm0, xmm1

// CHECK: vcvtbf42hf8 zmm0, ymm1
// CHECK: encoding: [0x62,0xf5,0x7c,0x48,0x37,0xc1]
          vcvtbf42hf8 zmm0, ymm1

// CHECK: vcvtbf42hf8 ymm0, xmm1
// CHECK: encoding: [0x62,0xf5,0x7c,0x28,0x37,0xc1]
          vcvtbf42hf8 ymm0, xmm1

// CHECK: vcvtbf42hf8 xmm0, xmm1
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x37,0xc1]
          vcvtbf42hf8 xmm0, xmm1

// CHECK: vcvtbf42hf8 zmm0, ymmword ptr [rdi]
// CHECK: encoding: [0x62,0xf5,0x7c,0x48,0x37,0x07]
          vcvtbf42hf8 zmm0, ymmword ptr [rdi]

// CHECK: vcvtbf42hf8 ymm0, xmmword ptr [rdi]
// CHECK: encoding: [0x62,0xf5,0x7c,0x28,0x37,0x07]
          vcvtbf42hf8 ymm0, xmmword ptr [rdi]

// CHECK: vcvtbf42hf8 xmm0, qword ptr [rdi]
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x37,0x07]
          vcvtbf42hf8 xmm0, qword ptr [rdi]

// CHECK: vcvtbf42hf8 zmm0 {k1}, ymm1
// CHECK: encoding: [0x62,0xf5,0x7c,0x49,0x37,0xc1]
          vcvtbf42hf8 zmm0 {k1}, ymm1

// CHECK: vcvtbf42hf8 zmm0 {k1} {z}, ymm1
// CHECK: encoding: [0x62,0xf5,0x7c,0xc9,0x37,0xc1]
          vcvtbf42hf8 zmm0 {k1} {z}, ymm1

// CHECK: vcvtbf62hf8 zmm0, zmm1
// CHECK: encoding: [0x62,0xf5,0xfd,0x48,0x37,0xc1]
          vcvtbf62hf8 zmm0, zmm1

// CHECK: vcvtbf62hf8 ymm0, ymm1
// CHECK: encoding: [0x62,0xf5,0xfd,0x28,0x37,0xc1]
          vcvtbf62hf8 ymm0, ymm1

// CHECK: vcvtbf62hf8 xmm0, xmm1
// CHECK: encoding: [0x62,0xf5,0xfd,0x08,0x37,0xc1]
          vcvtbf62hf8 xmm0, xmm1

// CHECK: vcvtbf62hf8 zmm0 {k1}, zmm1
// CHECK: encoding: [0x62,0xf5,0xfd,0x49,0x37,0xc1]
          vcvtbf62hf8 zmm0 {k1}, zmm1

// CHECK: vcvtbf62hf8 zmm0 {k1} {z}, zmm1
// CHECK: encoding: [0x62,0xf5,0xfd,0xc9,0x37,0xc1]
          vcvtbf62hf8 zmm0 {k1} {z}, zmm1

// CHECK: vcvthf62hf8 zmm0, zmm1
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x37,0xc1]
          vcvthf62hf8 zmm0, zmm1

// CHECK: vcvthf62hf8 ymm0, ymm1
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x37,0xc1]
          vcvthf62hf8 ymm0, ymm1

// CHECK: vcvthf62hf8 xmm0, xmm1
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x37,0xc1]
          vcvthf62hf8 xmm0, xmm1

// CHECK: vcvthf62hf8 zmm0 {k1}, zmm1
// CHECK: encoding: [0x62,0xf5,0x7d,0x49,0x37,0xc1]
          vcvthf62hf8 zmm0 {k1}, zmm1

// CHECK: vcvthf62hf8 zmm0 {k1} {z}, zmm1
// CHECK: encoding: [0x62,0xf5,0x7d,0xc9,0x37,0xc1]
          vcvthf62hf8 zmm0 {k1} {z}, zmm1

// CHECK: vpmovssdb xmm0, zmm1
// CHECK: encoding: [0x62,0xf2,0x7e,0x48,0x41,0xc8]
          vpmovssdb xmm0, zmm1

// CHECK: vpmovssdb xmm0, ymm1
// CHECK: encoding: [0x62,0xf2,0x7e,0x28,0x41,0xc8]
          vpmovssdb xmm0, ymm1

// CHECK: vpmovssdb xmm0, xmm1
// CHECK: encoding: [0x62,0xf2,0x7e,0x08,0x41,0xc8]
          vpmovssdb xmm0, xmm1

// CHECK: vpmovssdb xmmword ptr [rdi], zmm1
// CHECK: encoding: [0x62,0xf2,0x7e,0x48,0x41,0x0f]
          vpmovssdb xmmword ptr [rdi], zmm1

// CHECK: vpmovssdb qword ptr [rdi], ymm1
// CHECK: encoding: [0x62,0xf2,0x7e,0x28,0x41,0x0f]
          vpmovssdb qword ptr [rdi], ymm1

// CHECK: vpmovssdb dword ptr [rdi], xmm1
// CHECK: encoding: [0x62,0xf2,0x7e,0x08,0x41,0x0f]
          vpmovssdb dword ptr [rdi], xmm1

// CHECK: vpmovssdb xmm0 {k1}, zmm1
// CHECK: encoding: [0x62,0xf2,0x7e,0x49,0x41,0xc8]
          vpmovssdb xmm0 {k1}, zmm1

// CHECK: vpmovssdb xmm0 {k1} {z}, zmm1
// CHECK: encoding: [0x62,0xf2,0x7e,0xc9,0x41,0xc8]
          vpmovssdb xmm0 {k1} {z}, zmm1

// CHECK: vunpackb zmm0, zmm1, 1
// CHECK: encoding: [0x62,0xf3,0x7c,0x48,0x3d,0xc1,0x01]
          vunpackb zmm0, zmm1, 1

// CHECK: vunpackb ymm0, ymm1, 1
// CHECK: encoding: [0x62,0xf3,0x7c,0x28,0x3d,0xc1,0x01]
          vunpackb ymm0, ymm1, 1

// CHECK: vunpackb xmm0, xmm1, 1
// CHECK: encoding: [0x62,0xf3,0x7c,0x08,0x3d,0xc1,0x01]
          vunpackb xmm0, xmm1, 1

// CHECK: vunpackb zmm0, zmmword ptr [rdi], 1
// CHECK: encoding: [0x62,0xf3,0x7c,0x48,0x3d,0x07,0x01]
          vunpackb zmm0, zmmword ptr [rdi], 1

// CHECK: vunpackb ymm0, ymmword ptr [rdi], 1
// CHECK: encoding: [0x62,0xf3,0x7c,0x28,0x3d,0x07,0x01]
          vunpackb ymm0, ymmword ptr [rdi], 1

// CHECK: vunpackb xmm0, xmmword ptr [rdi], 1
// CHECK: encoding: [0x62,0xf3,0x7c,0x08,0x3d,0x07,0x01]
          vunpackb xmm0, xmmword ptr [rdi], 1

// CHECK: vunpackb zmm0 {k1}, zmm1, 1
// CHECK: encoding: [0x62,0xf3,0x7c,0x49,0x3d,0xc1,0x01]
          vunpackb zmm0 {k1}, zmm1, 1

// CHECK: vunpackb zmm0 {k1} {z}, zmm1, 1
// CHECK: encoding: [0x62,0xf3,0x7c,0xc9,0x3d,0xc1,0x01]
          vunpackb zmm0 {k1} {z}, zmm1, 1
