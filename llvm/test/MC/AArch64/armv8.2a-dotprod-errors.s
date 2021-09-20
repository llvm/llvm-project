// RUN: not llvm-mc -triple aarch64 -mattr=+dotprod -show-encoding < %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-ERROR < %t %s
// RUN: not llvm-mc -triple aarch64 -mattr=+v8r -show-encoding < %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-ERROR < %t %s

udot v0.2s, v1.8b, v2.4b[4]
sdot v0.2s, v1.8b, v2.4b[4]
udot v0.4s, v1.16b, v2.4b[4]
sdot v0.4s, v1.16b, v2.4b[4]

// CHECK-ERROR: vector lane must be an integer in range [0, 3]
// CHECK-ERROR: vector lane must be an integer in range [0, 3]
// CHECK-ERROR: vector lane must be an integer in range [0, 3]
// CHECK-ERROR: vector lane must be an integer in range [0, 3]
