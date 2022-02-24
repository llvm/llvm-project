// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+streaming-sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

// ---------------------------------------------------------------------------//
// Test 64-bit form (x0) and its aliases
// ---------------------------------------------------------------------------//

sqdecb  x0
// CHECK-INST: sqdecb  x0
// CHECK-ENCODING: [0xe0,0xfb,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 fb 30 04 <unknown>

sqdecb  x0, all
// CHECK-INST: sqdecb  x0
// CHECK-ENCODING: [0xe0,0xfb,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 fb 30 04 <unknown>

sqdecb  x0, all, mul #1
// CHECK-INST: sqdecb  x0
// CHECK-ENCODING: [0xe0,0xfb,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 fb 30 04 <unknown>

sqdecb  x0, all, mul #16
// CHECK-INST: sqdecb  x0, all, mul #16
// CHECK-ENCODING: [0xe0,0xfb,0x3f,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 fb 3f 04 <unknown>


// ---------------------------------------------------------------------------//
// Test 32-bit form (x0, w0) and its aliases
// ---------------------------------------------------------------------------//

sqdecb  x0, w0
// CHECK-INST: sqdecb  x0, w0
// CHECK-ENCODING: [0xe0,0xfb,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 fb 20 04 <unknown>

sqdecb  x0, w0, all
// CHECK-INST: sqdecb  x0, w0
// CHECK-ENCODING: [0xe0,0xfb,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 fb 20 04 <unknown>

sqdecb  x0, w0, all, mul #1
// CHECK-INST: sqdecb  x0, w0
// CHECK-ENCODING: [0xe0,0xfb,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 fb 20 04 <unknown>

sqdecb  x0, w0, all, mul #16
// CHECK-INST: sqdecb  x0, w0, all, mul #16
// CHECK-ENCODING: [0xe0,0xfb,0x2f,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 fb 2f 04 <unknown>

sqdecb  x0, w0, pow2
// CHECK-INST: sqdecb  x0, w0, pow2
// CHECK-ENCODING: [0x00,0xf8,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 f8 20 04 <unknown>

sqdecb  x0, w0, pow2, mul #16
// CHECK-INST: sqdecb  x0, w0, pow2, mul #16
// CHECK-ENCODING: [0x00,0xf8,0x2f,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 f8 2f 04 <unknown>


// ---------------------------------------------------------------------------//
// Test all patterns for 64-bit form
// ---------------------------------------------------------------------------//

sqdecb  x0, pow2
// CHECK-INST: sqdecb  x0, pow2
// CHECK-ENCODING: [0x00,0xf8,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 f8 30 04 <unknown>

sqdecb  x0, vl1
// CHECK-INST: sqdecb  x0, vl1
// CHECK-ENCODING: [0x20,0xf8,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 f8 30 04 <unknown>

sqdecb  x0, vl2
// CHECK-INST: sqdecb  x0, vl2
// CHECK-ENCODING: [0x40,0xf8,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 40 f8 30 04 <unknown>

sqdecb  x0, vl3
// CHECK-INST: sqdecb  x0, vl3
// CHECK-ENCODING: [0x60,0xf8,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 60 f8 30 04 <unknown>

sqdecb  x0, vl4
// CHECK-INST: sqdecb  x0, vl4
// CHECK-ENCODING: [0x80,0xf8,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 80 f8 30 04 <unknown>

sqdecb  x0, vl5
// CHECK-INST: sqdecb  x0, vl5
// CHECK-ENCODING: [0xa0,0xf8,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a0 f8 30 04 <unknown>

sqdecb  x0, vl6
// CHECK-INST: sqdecb  x0, vl6
// CHECK-ENCODING: [0xc0,0xf8,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: c0 f8 30 04 <unknown>

sqdecb  x0, vl7
// CHECK-INST: sqdecb  x0, vl7
// CHECK-ENCODING: [0xe0,0xf8,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 f8 30 04 <unknown>

sqdecb  x0, vl8
// CHECK-INST: sqdecb  x0, vl8
// CHECK-ENCODING: [0x00,0xf9,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 f9 30 04 <unknown>

sqdecb  x0, vl16
// CHECK-INST: sqdecb  x0, vl16
// CHECK-ENCODING: [0x20,0xf9,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 f9 30 04 <unknown>

sqdecb  x0, vl32
// CHECK-INST: sqdecb  x0, vl32
// CHECK-ENCODING: [0x40,0xf9,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 40 f9 30 04 <unknown>

sqdecb  x0, vl64
// CHECK-INST: sqdecb  x0, vl64
// CHECK-ENCODING: [0x60,0xf9,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 60 f9 30 04 <unknown>

sqdecb  x0, vl128
// CHECK-INST: sqdecb  x0, vl128
// CHECK-ENCODING: [0x80,0xf9,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 80 f9 30 04 <unknown>

sqdecb  x0, vl256
// CHECK-INST: sqdecb  x0, vl256
// CHECK-ENCODING: [0xa0,0xf9,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a0 f9 30 04 <unknown>

sqdecb  x0, #14
// CHECK-INST: sqdecb  x0, #14
// CHECK-ENCODING: [0xc0,0xf9,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: c0 f9 30 04 <unknown>

sqdecb  x0, #15
// CHECK-INST: sqdecb  x0, #15
// CHECK-ENCODING: [0xe0,0xf9,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 f9 30 04 <unknown>

sqdecb  x0, #16
// CHECK-INST: sqdecb  x0, #16
// CHECK-ENCODING: [0x00,0xfa,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 fa 30 04 <unknown>

sqdecb  x0, #17
// CHECK-INST: sqdecb  x0, #17
// CHECK-ENCODING: [0x20,0xfa,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 fa 30 04 <unknown>

sqdecb  x0, #18
// CHECK-INST: sqdecb  x0, #18
// CHECK-ENCODING: [0x40,0xfa,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 40 fa 30 04 <unknown>

sqdecb  x0, #19
// CHECK-INST: sqdecb  x0, #19
// CHECK-ENCODING: [0x60,0xfa,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 60 fa 30 04 <unknown>

sqdecb  x0, #20
// CHECK-INST: sqdecb  x0, #20
// CHECK-ENCODING: [0x80,0xfa,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 80 fa 30 04 <unknown>

sqdecb  x0, #21
// CHECK-INST: sqdecb  x0, #21
// CHECK-ENCODING: [0xa0,0xfa,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a0 fa 30 04 <unknown>

sqdecb  x0, #22
// CHECK-INST: sqdecb  x0, #22
// CHECK-ENCODING: [0xc0,0xfa,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: c0 fa 30 04 <unknown>

sqdecb  x0, #23
// CHECK-INST: sqdecb  x0, #23
// CHECK-ENCODING: [0xe0,0xfa,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 fa 30 04 <unknown>

sqdecb  x0, #24
// CHECK-INST: sqdecb  x0, #24
// CHECK-ENCODING: [0x00,0xfb,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 fb 30 04 <unknown>

sqdecb  x0, #25
// CHECK-INST: sqdecb  x0, #25
// CHECK-ENCODING: [0x20,0xfb,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 fb 30 04 <unknown>

sqdecb  x0, #26
// CHECK-INST: sqdecb  x0, #26
// CHECK-ENCODING: [0x40,0xfb,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 40 fb 30 04 <unknown>

sqdecb  x0, #27
// CHECK-INST: sqdecb  x0, #27
// CHECK-ENCODING: [0x60,0xfb,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 60 fb 30 04 <unknown>

sqdecb  x0, #28
// CHECK-INST: sqdecb  x0, #28
// CHECK-ENCODING: [0x80,0xfb,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 80 fb 30 04 <unknown>
