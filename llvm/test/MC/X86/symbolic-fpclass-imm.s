// RUN: llvm-mc -triple x86_64-unknown-unknown -mattr=+avx512dq,+avx512vl %s | FileCheck %s
// RUN: llvm-mc -triple x86_64-unknown-unknown -mattr=+avx512dq,+avx512vl --show-encoding %s | FileCheck %s --check-prefix=ENC

// The FPCLASS category mask may be a symbol, whose value is only known at link
// time. The instruction comment printer used to assume it was always a literal
// and assert.

// CHECK: vfpclassps $f0, %zmm1, %k1{{$}}
// ENC: vfpclassps $f0, %zmm1, %k1
// ENC-SAME: encoding: [0x62,0xf3,0x7d,0x48,0x66,0xc9,A]
vfpclassps $f0, %zmm1, %k1

// A literal category is still described as before.

// CHECK: vfpclassps $3, %zmm1, %k1
// CHECK-SAME: k1 = isQuietNaN(zmm1) | isPositiveZero(zmm1)
vfpclassps $3, %zmm1, %k1

// CHECK: vfpclassps $0, %zmm1, %k1
// CHECK-SAME: k1 = false
vfpclassps $0, %zmm1, %k1
