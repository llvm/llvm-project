// RUN: not llvm-mc -triple aarch64 -mattr=+sm4,+sha3 -show-encoding < %s 2>&1 | FileCheck %s

  xar v26.2d, v21.2d, v27.2d, #-1
  xar v26.2d, v21.2d, v27.2d, #64
  sm3tt1a v20.4s, v23.4s, v21.s[4]
  sm3tt1b v20.4s, v23.4s, v21.s[4]
  sm3tt2a v20.4s, v23.4s, v21.s[4]
  sm3tt2b v20.4s, v23.4s, v21.s[4]
  sm3tt2b v20.4s, v23.4s, v21.s[-1]

// CHECK:      error: immediate must be an integer in range [0, 63].
// CHECK-NEXT: xar v26.2d, v21.2d, v27.2d, #-1
// CHECK-NEXT:                             ^
// CHECK-NEXT: error: immediate must be an integer in range [0, 63].
// CHECK-NEXT: xar v26.2d, v21.2d, v27.2d, #64
// CHECK-NEXT:                             ^
// CHECK-NEXT: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: sm3tt1a v20.4s, v23.4s, v21.s[4]
// CHECK-NEXT:                              ^
// CHECK-NEXT: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: sm3tt1b v20.4s, v23.4s, v21.s[4]
// CHECK-NEXT:                              ^
// CHECK-NEXT: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: sm3tt2a v20.4s, v23.4s, v21.s[4]
// CHECK-NEXT:                              ^
// CHECK-NEXT: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: sm3tt2b v20.4s, v23.4s, v21.s[4]
// CHECK-NEXT:                              ^
// CHECK-NEXT: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: sm3tt2b v20.4s, v23.4s, v21.s[-1]
// CHECK-NEXT:                              ^
