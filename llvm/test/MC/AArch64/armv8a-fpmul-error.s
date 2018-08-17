// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+fp16fml,+neon < %s 2>&1 | FileCheck %s --check-prefix=CHECK

//------------------------------------------------------------------------------
// ARMV8.2-A Floating Point Multiplication
//------------------------------------------------------------------------------

fmlal  V0.2s, v1.2h, v2.h[8]
fmlsl  V0.2s, v1.2h, v2.h[8]
fmlal  V0.4s, v1.4h, v2.h[8]
fmlsl  V0.4s, v1.4h, v2.h[8]

fmlal2  V0.2s, v1.2h, v2.h[8]
fmlsl2  V0.2s, v1.2h, v2.h[8]
fmlal2  V0.4s, v1.4h, v2.h[8]
fmlsl2  V0.4s, v1.4h, v2.h[8]

fmlal  V0.2s, v1.2h, v2.h[-1]
fmlsl2  V0.2s, v1.2h, v2.h[-1]

//CHECK: error: vector lane must be an integer in range [0, 7].
//CHECK-NEXT: fmlal  V0.2s, v1.2h, v2.h[8]
//CHECK-NEXT:                          ^
//CHECK-NEXT: error: vector lane must be an integer in range [0, 7].
//CHECK-NEXT: fmlsl  V0.2s, v1.2h, v2.h[8]
//CHECK-NEXT:                          ^
//CHECK-NEXT: error: vector lane must be an integer in range [0, 7].
//CHECK-NEXT: fmlal  V0.4s, v1.4h, v2.h[8]
//CHECK-NEXT:                          ^
//CHECK-NEXT: error: vector lane must be an integer in range [0, 7].
//CHECK-NEXT: fmlsl  V0.4s, v1.4h, v2.h[8]
//CHECK-NEXT:                          ^

//CHECK-NEXT: error: vector lane must be an integer in range [0, 7].
//CHECK-NEXT: fmlal2  V0.2s, v1.2h, v2.h[8]
//CHECK-NEXT:                           ^
//CHECK-NEXT: error: vector lane must be an integer in range [0, 7].
//CHECK-NEXT: fmlsl2  V0.2s, v1.2h, v2.h[8]
//CHECK-NEXT:                           ^
//CHECK-NEXT: error: vector lane must be an integer in range [0, 7].
//CHECK-NEXT: fmlal2  V0.4s, v1.4h, v2.h[8]
//CHECK-NEXT:                           ^
//CHECK-NEXT: error: vector lane must be an integer in range [0, 7].
//CHECK-NEXT: fmlsl2  V0.4s, v1.4h, v2.h[8]
//CHECK-NEXT:                           ^

//CHECK-NEXT: error: vector lane must be an integer in range [0, 7].
//CHECK-NEXT: fmlal  V0.2s, v1.2h, v2.h[-1]
//CHECK-NEXT:                          ^
//CHECK-NEXT: error: vector lane must be an integer in range [0, 7].
//CHECK-NEXT: fmlsl2  V0.2s, v1.2h, v2.h[-1]
//CHECK-NEXT:                           ^
