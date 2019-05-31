// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding %s | FileCheck %s

// v8i64 vectors
// CHECK: vp2intersectq        %zmm2, %zmm1, %k0
// CHECK: encoding: [0x62,0xf2,0xf7,0x48,0x68,0xc2]
vp2intersectq  %zmm2, %zmm1, %k0

// CHECK: vp2intersectq        (%rdi), %zmm1, %k0
// CHECK: encoding: [0x62,0xf2,0xf7,0x48,0x68,0x07]
vp2intersectq  (%rdi), %zmm1, %k0

// CHECK: vp2intersectq        (%rdi){1to8}, %zmm1, %k0
// CHECK: encoding: [0x62,0xf2,0xf7,0x58,0x68,0x07]
vp2intersectq  (%rdi){1to8}, %zmm1, %k0

// CHECK: vp2intersectq        %zmm2, %zmm1, %k0
// CHECK: encoding: [0x62,0xf2,0xf7,0x48,0x68,0xc2]
vp2intersectq  %zmm2, %zmm1, %k1

// CHECK: vp2intersectq        (%rdi), %zmm1, %k0
// CHECK: encoding: [0x62,0xf2,0xf7,0x48,0x68,0x07]
vp2intersectq  (%rdi), %zmm1, %k1

// CHECK: vp2intersectq        (%rdi){1to8}, %zmm1, %k0
// CHECK: encoding: [0x62,0xf2,0xf7,0x58,0x68,0x07]
vp2intersectq  (%rdi){1to8}, %zmm1, %k1

// CHECK: vp2intersectq        %zmm7, %zmm9, %k6
// CHECK: encoding: [0x62,0xf2,0xb7,0x48,0x68,0xf7]
vp2intersectq  %zmm7, %zmm9, %k6

// CHECK: vp2intersectq        (%rsi), %zmm9, %k6
// CHECK: encoding: [0x62,0xf2,0xb7,0x48,0x68,0x36]
vp2intersectq  (%rsi), %zmm9, %k6

// CHECK: vp2intersectq        (%rsi){1to8}, %zmm9, %k6
// CHECK: encoding: [0x62,0xf2,0xb7,0x58,0x68,0x36]
vp2intersectq  (%rsi){1to8}, %zmm9, %k6

// CHECK: vp2intersectq        %zmm7, %zmm9, %k6
// CHECK: encoding: [0x62,0xf2,0xb7,0x48,0x68,0xf7]
vp2intersectq  %zmm7, %zmm9, %k7

// CHECK: vp2intersectq        (%rsi), %zmm9, %k6
// CHECK: encoding: [0x62,0xf2,0xb7,0x48,0x68,0x36]
vp2intersectq  (%rsi), %zmm9, %k7

// CHECK: vp2intersectq        (%rsi){1to8}, %zmm9, %k6
// CHECK: encoding: [0x62,0xf2,0xb7,0x58,0x68,0x36]
vp2intersectq  (%rsi){1to8}, %zmm9, %k7

// v4i64 vectors
// CHECK: vp2intersectq        %ymm2, %ymm1, %k0
// CHECK: encoding: [0x62,0xf2,0xf7,0x28,0x68,0xc2]
vp2intersectq  %ymm2, %ymm1, %k0

// CHECK: vp2intersectq        (%rdi), %ymm1, %k0
// CHECK: encoding: [0x62,0xf2,0xf7,0x28,0x68,0x07]
vp2intersectq  (%rdi), %ymm1, %k0

// CHECK: vp2intersectq        (%rdi){1to4}, %ymm1, %k0
// CHECK: encoding: [0x62,0xf2,0xf7,0x38,0x68,0x07]
vp2intersectq  (%rdi){1to4}, %ymm1, %k0

// CHECK: vp2intersectq        %ymm2, %ymm1, %k0
// CHECK: encoding: [0x62,0xf2,0xf7,0x28,0x68,0xc2]
vp2intersectq  %ymm2, %ymm1, %k1

// CHECK: vp2intersectq        (%rdi), %ymm1, %k0
// CHECK: encoding: [0x62,0xf2,0xf7,0x28,0x68,0x07]
vp2intersectq  (%rdi), %ymm1, %k1

// CHECK: vp2intersectq        (%rdi){1to4}, %ymm1, %k0
// CHECK: encoding: [0x62,0xf2,0xf7,0x38,0x68,0x07]
vp2intersectq  (%rdi){1to4}, %ymm1, %k1

// CHECK: vp2intersectq        %ymm7, %ymm9, %k6
// CHECK: encoding: [0x62,0xf2,0xb7,0x28,0x68,0xf7]
vp2intersectq  %ymm7, %ymm9, %k6

// CHECK: vp2intersectq        (%rsi), %ymm9, %k6
// CHECK: encoding: [0x62,0xf2,0xb7,0x28,0x68,0x36]
vp2intersectq  (%rsi), %ymm9, %k6

// CHECK: vp2intersectq        (%rsi){1to4}, %ymm9, %k6
// CHECK: encoding: [0x62,0xf2,0xb7,0x38,0x68,0x36]
vp2intersectq  (%rsi){1to4}, %ymm9, %k6

// CHECK: vp2intersectq        %ymm7, %ymm9, %k6
// CHECK: encoding: [0x62,0xf2,0xb7,0x28,0x68,0xf7]
vp2intersectq  %ymm7, %ymm9, %k7

// CHECK: vp2intersectq        (%rsi), %ymm9, %k6
// CHECK: encoding: [0x62,0xf2,0xb7,0x28,0x68,0x36]
vp2intersectq  (%rsi), %ymm9, %k7

// v2i64 vectors
// CHECK: vp2intersectq        %xmm2, %xmm1, %k0
// CHECK: encoding: [0x62,0xf2,0xf7,0x08,0x68,0xc2]
vp2intersectq  %xmm2, %xmm1, %k0

// CHECK: vp2intersectq        (%rdi), %xmm1, %k0
// CHECK: encoding: [0x62,0xf2,0xf7,0x08,0x68,0x07]
vp2intersectq  (%rdi), %xmm1, %k0

// CHECK: vp2intersectq        (%rdi){1to2}, %xmm1, %k0
// CHECK: encoding: [0x62,0xf2,0xf7,0x18,0x68,0x07]
vp2intersectq  (%rdi){1to2}, %xmm1, %k0

// CHECK: vp2intersectq        %xmm2, %xmm1, %k0
// CHECK: encoding: [0x62,0xf2,0xf7,0x08,0x68,0xc2]
vp2intersectq  %xmm2, %xmm1, %k1

// CHECK: vp2intersectq        (%rdi), %xmm1, %k0
// CHECK: encoding: [0x62,0xf2,0xf7,0x08,0x68,0x07]
vp2intersectq  (%rdi), %xmm1, %k1

// CHECK: vp2intersectq        %xmm7, %xmm9, %k6
// CHECK: encoding: [0x62,0xf2,0xb7,0x08,0x68,0xf7]
vp2intersectq  %xmm7, %xmm9, %k6

// CHECK: vp2intersectq        (%rsi), %xmm9, %k6
// CHECK: encoding: [0x62,0xf2,0xb7,0x08,0x68,0x36]
vp2intersectq  (%rsi), %xmm9, %k6

// CHECK: vp2intersectq        %xmm7, %xmm9, %k6
// CHECK: encoding: [0x62,0xf2,0xb7,0x08,0x68,0xf7]
vp2intersectq  %xmm7, %xmm9, %k7

// CHECK: vp2intersectq        (%rsi), %xmm9, %k6
// CHECK: encoding: [0x62,0xf2,0xb7,0x08,0x68,0x36]
vp2intersectq  (%rsi), %xmm9, %k7

// v16i32 vectors
// CHECK: vp2intersectd        %zmm2, %zmm1, %k0
// CHECK: encoding: [0x62,0xf2,0x77,0x48,0x68,0xc2]
vp2intersectd  %zmm2, %zmm1, %k0

// CHECK: vp2intersectd        (%rdi), %zmm1, %k0
// CHECK: encoding: [0x62,0xf2,0x77,0x48,0x68,0x07]
vp2intersectd  (%rdi), %zmm1, %k0

// CHECK: vp2intersectd        %zmm2, %zmm1, %k0
// CHECK: encoding: [0x62,0xf2,0x77,0x48,0x68,0xc2]
vp2intersectd  %zmm2, %zmm1, %k1

// CHECK: vp2intersectd        (%rdi), %zmm1, %k0
// CHECK: encoding: [0x62,0xf2,0x77,0x48,0x68,0x07]
vp2intersectd  (%rdi), %zmm1, %k1

// CHECK: vp2intersectd        %zmm7, %zmm9, %k6
// CHECK: encoding: [0x62,0xf2,0x37,0x48,0x68,0xf7]
vp2intersectd  %zmm7, %zmm9, %k6

// CHECK: vp2intersectd        (%rsi), %zmm9, %k6
// CHECK: encoding: [0x62,0xf2,0x37,0x48,0x68,0x36]
vp2intersectd  (%rsi), %zmm9, %k6

// CHECK: vp2intersectd        %zmm7, %zmm9, %k6
// CHECK: encoding: [0x62,0xf2,0x37,0x48,0x68,0xf7]
vp2intersectd  %zmm7, %zmm9, %k7

// CHECK: vp2intersectd        (%rsi), %zmm9, %k6
// CHECK: encoding: [0x62,0xf2,0x37,0x48,0x68,0x36]
vp2intersectd  (%rsi), %zmm9, %k7

// v8i32 vectors
// CHECK: vp2intersectd        %ymm2, %ymm1, %k0
// CHECK: encoding: [0x62,0xf2,0x77,0x28,0x68,0xc2]
vp2intersectd  %ymm2, %ymm1, %k0

// CHECK: vp2intersectd        (%rdi), %ymm1, %k0
// CHECK: encoding: [0x62,0xf2,0x77,0x28,0x68,0x07]
vp2intersectd  (%rdi), %ymm1, %k0

// CHECK: vp2intersectd        %ymm2, %ymm1, %k0
// CHECK: encoding: [0x62,0xf2,0x77,0x28,0x68,0xc2]
vp2intersectd  %ymm2, %ymm1, %k1

// CHECK: vp2intersectd        (%rdi), %ymm1, %k0
// CHECK: encoding: [0x62,0xf2,0x77,0x28,0x68,0x07]
vp2intersectd  (%rdi), %ymm1, %k1

// CHECK: vp2intersectd        %ymm7, %ymm9, %k6
// CHECK: encoding: [0x62,0xf2,0x37,0x28,0x68,0xf7]
vp2intersectd  %ymm7, %ymm9, %k6

// CHECK: vp2intersectd        (%rsi), %ymm9, %k6
// CHECK: encoding: [0x62,0xf2,0x37,0x28,0x68,0x36]
vp2intersectd  (%rsi), %ymm9, %k6

// CHECK: vp2intersectd        %ymm7, %ymm9, %k6
// CHECK: encoding: [0x62,0xf2,0x37,0x28,0x68,0xf7]
vp2intersectd  %ymm7, %ymm9, %k7

// CHECK: vp2intersectd        (%rsi), %ymm9, %k6
// CHECK: encoding: [0x62,0xf2,0x37,0x28,0x68,0x36]
vp2intersectd  (%rsi), %ymm9, %k7

// v4i32 vectors
// CHECK: vp2intersectd        %xmm2, %xmm1, %k0
// CHECK: encoding: [0x62,0xf2,0x77,0x08,0x68,0xc2]
vp2intersectd  %xmm2, %xmm1, %k0

// CHECK: vp2intersectd        (%rdi), %xmm1, %k0
// CHECK: encoding: [0x62,0xf2,0x77,0x08,0x68,0x07]
vp2intersectd  (%rdi), %xmm1, %k0

// CHECK: vp2intersectd        %xmm2, %xmm1, %k0
// CHECK: encoding: [0x62,0xf2,0x77,0x08,0x68,0xc2]
vp2intersectd  %xmm2, %xmm1, %k1

// CHECK: vp2intersectd        (%rdi), %xmm1, %k0
// CHECK: encoding: [0x62,0xf2,0x77,0x08,0x68,0x07]
vp2intersectd  (%rdi), %xmm1, %k1

// CHECK: vp2intersectd        %xmm7, %xmm9, %k6
// CHECK: encoding: [0x62,0xf2,0x37,0x08,0x68,0xf7]
vp2intersectd  %xmm7, %xmm9, %k6

// CHECK: vp2intersectd        (%rsi), %xmm9, %k6
// CHECK: encoding: [0x62,0xf2,0x37,0x08,0x68,0x36]
vp2intersectd  (%rsi), %xmm9, %k6

// CHECK: vp2intersectd        %xmm7, %xmm9, %k6
// CHECK: encoding: [0x62,0xf2,0x37,0x08,0x68,0xf7]
vp2intersectd  %xmm7, %xmm9, %k7

// CHECK: vp2intersectd        (%rsi), %xmm9, %k6
// CHECK: encoding: [0x62,0xf2,0x37,0x08,0x68,0x36]
vp2intersectd  (%rsi), %xmm9, %k7
