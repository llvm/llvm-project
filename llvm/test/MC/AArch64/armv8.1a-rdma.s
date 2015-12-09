// RUN: not llvm-mc -triple aarch64-none-linux-gnu -mattr=+v8.1a -show-encoding < %s 2> %t | FileCheck %s
// RUN: FileCheck --check-prefix=CHECK-ERROR < %t %s
  .text

  //AdvSIMD RDMA vector
  sqrdmlah v0.4h, v1.4h, v2.4h
  sqrdmlsh v0.4h, v1.4h, v2.4h
  sqrdmlah v0.2s, v1.2s, v2.2s
  sqrdmlsh v0.2s, v1.2s, v2.2s
  sqrdmlah v0.4s, v1.4s, v2.4s
  sqrdmlsh v0.4s, v1.4s, v2.4s
  sqrdmlah v0.8h, v1.8h, v2.8h
  sqrdmlsh v0.8h, v1.8h, v2.8h
// CHECK: sqrdmlah  v0.4h, v1.4h, v2.4h // encoding: [0x20,0x84,0x42,0x2e]
// CHECK: sqrdmlsh  v0.4h, v1.4h, v2.4h // encoding: [0x20,0x8c,0x42,0x2e]
// CHECK: sqrdmlah  v0.2s, v1.2s, v2.2s // encoding: [0x20,0x84,0x82,0x2e]
// CHECK: sqrdmlsh  v0.2s, v1.2s, v2.2s // encoding: [0x20,0x8c,0x82,0x2e]
// CHECK: sqrdmlah  v0.4s, v1.4s, v2.4s // encoding: [0x20,0x84,0x82,0x6e]
// CHECK: sqrdmlsh  v0.4s, v1.4s, v2.4s // encoding: [0x20,0x8c,0x82,0x6e]
// CHECK: sqrdmlah  v0.8h, v1.8h, v2.8h // encoding: [0x20,0x84,0x42,0x6e]
// CHECK: sqrdmlsh  v0.8h, v1.8h, v2.8h // encoding: [0x20,0x8c,0x42,0x6e]

  sqrdmlah v0.2h, v1.2h, v2.2h
  sqrdmlsh v0.2h, v1.2h, v2.2h
  sqrdmlah v0.8s, v1.8s, v2.8s
  sqrdmlsh v0.8s, v1.8s, v2.8s
  sqrdmlah v0.2s, v1.4h, v2.8h
  sqrdmlsh v0.4s, v1.8h, v2.2s
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:   sqrdmlah v0.2h, v1.2h, v2.2h
// CHECK-ERROR:            ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:   sqrdmlsh v0.2h, v1.2h, v2.2h
// CHECK-ERROR:            ^
// CHECK-ERROR: error: invalid vector kind qualifier
// CHECK-ERROR:   sqrdmlah v0.8s, v1.8s, v2.8s
// CHECK-ERROR:            ^
// CHECK-ERROR: error: invalid vector kind qualifier
// CHECK-ERROR:   sqrdmlah v0.8s, v1.8s, v2.8s
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid vector kind qualifier
// CHECK-ERROR:   sqrdmlah v0.8s, v1.8s, v2.8s
// CHECK-ERROR:                          ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:   sqrdmlah v0.8s, v1.8s, v2.8s
// CHECK-ERROR:            ^
// CHECK-ERROR: error: invalid vector kind qualifier
// CHECK-ERROR:   sqrdmlsh v0.8s, v1.8s, v2.8s
// CHECK-ERROR:            ^
// CHECK-ERROR: error: invalid vector kind qualifier
// CHECK-ERROR:   sqrdmlsh v0.8s, v1.8s, v2.8s
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid vector kind qualifier
// CHECK-ERROR:   sqrdmlsh v0.8s, v1.8s, v2.8s
// CHECK-ERROR:                          ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:   sqrdmlsh v0.8s, v1.8s, v2.8s
// CHECK-ERROR:            ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:   sqrdmlah v0.2s, v1.4h, v2.8h
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:   sqrdmlsh v0.4s, v1.8h, v2.2s
// CHECK-ERROR:                   ^

  //AdvSIMD RDMA scalar
  sqrdmlah h0, h1, h2
  sqrdmlsh h0, h1, h2
  sqrdmlah s0, s1, s2
  sqrdmlsh s0, s1, s2
// CHECK: sqrdmlah h0, h1, h2  // encoding: [0x20,0x84,0x42,0x7e]
// CHECK: sqrdmlsh h0, h1, h2  // encoding: [0x20,0x8c,0x42,0x7e]
// CHECK: sqrdmlah s0, s1, s2  // encoding: [0x20,0x84,0x82,0x7e]
// CHECK: sqrdmlsh s0, s1, s2  // encoding: [0x20,0x8c,0x82,0x7e]

  //AdvSIMD RDMA vector by-element
  sqrdmlah v0.4h, v1.4h, v2.h[3]
  sqrdmlsh v0.4h, v1.4h, v2.h[3]
  sqrdmlah v0.2s, v1.2s, v2.s[1]
  sqrdmlsh v0.2s, v1.2s, v2.s[1]
  sqrdmlah v0.8h, v1.8h, v2.h[3]
  sqrdmlsh v0.8h, v1.8h, v2.h[3]
  sqrdmlah v0.4s, v1.4s, v2.s[3]
  sqrdmlsh v0.4s, v1.4s, v2.s[3]
// CHECK: sqrdmlah v0.4h, v1.4h, v2.h[3]  // encoding: [0x20,0xd0,0x72,0x2f]
// CHECK: sqrdmlsh v0.4h, v1.4h, v2.h[3]  // encoding: [0x20,0xf0,0x72,0x2f]
// CHECK: sqrdmlah v0.2s, v1.2s, v2.s[1]  // encoding: [0x20,0xd0,0xa2,0x2f]
// CHECK: sqrdmlsh v0.2s, v1.2s, v2.s[1]  // encoding: [0x20,0xf0,0xa2,0x2f]
// CHECK: sqrdmlah v0.8h, v1.8h, v2.h[3]  // encoding: [0x20,0xd0,0x72,0x6f]
// CHECK: sqrdmlsh v0.8h, v1.8h, v2.h[3]  // encoding: [0x20,0xf0,0x72,0x6f]
// CHECK: sqrdmlah v0.4s, v1.4s, v2.s[3]  // encoding: [0x20,0xd8,0xa2,0x6f]
// CHECK: sqrdmlsh v0.4s, v1.4s, v2.s[3]  // encoding: [0x20,0xf8,0xa2,0x6f]

  sqrdmlah v0.4s, v1.2s, v2.s[1]
  sqrdmlsh v0.2s, v1.2d, v2.s[1]
  sqrdmlah v0.8h, v1.8h, v2.s[3]
  sqrdmlsh v0.8h, v1.8h, v2.h[8]
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:   sqrdmlah v0.4s, v1.2s, v2.s[1]
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:   sqrdmlsh v0.2s, v1.2d, v2.s[1]
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:   sqrdmlah v0.8h, v1.8h, v2.s[3]
// CHECK-ERROR:                          ^
// CHECK-ERROR: error: vector lane must be an integer in range [0, 7].
// CHECK-ERROR:   sqrdmlsh v0.8h, v1.8h, v2.h[8]
// CHECK-ERROR:                              ^

  //AdvSIMD RDMA scalar by-element
  sqrdmlah h0, h1, v2.h[3]
  sqrdmlsh h0, h1, v2.h[3]
  sqrdmlah s0, s1, v2.s[3]
  sqrdmlsh s0, s1, v2.s[3]
// CHECK: sqrdmlah h0, h1, v2.h[3]  // encoding: [0x20,0xd0,0x72,0x7f]
// CHECK: sqrdmlsh h0, h1, v2.h[3]  // encoding: [0x20,0xf0,0x72,0x7f]
// CHECK: sqrdmlah s0, s1, v2.s[3]  // encoding: [0x20,0xd8,0xa2,0x7f]
// CHECK: sqrdmlsh s0, s1, v2.s[3]  // encoding: [0x20,0xf8,0xa2,0x7f]

  sqrdmlah b0, h1, v2.h[3]
  sqrdmlah s0, d1, v2.s[3]
  sqrdmlsh h0, h1, v2.s[3]
  sqrdmlsh s0, s1, v2.s[4]
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:   sqrdmlah b0, h1, v2.h[3]
// CHECK-ERROR:            ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:   sqrdmlah s0, d1, v2.s[3]
// CHECK-ERROR:                ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:   sqrdmlsh h0, h1, v2.s[3]
// CHECK-ERROR:                    ^
// CHECK-ERROR: error: vector lane must be an integer in range [0, 3].
// CHECK-ERROR:   sqrdmlsh s0, s1, v2.s[4]
// CHECK-ERROR:                        ^
