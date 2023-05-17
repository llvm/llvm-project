// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump --no-print-imm-hex -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:   | llvm-objdump --no-print-imm-hex -d --mattr=-sve - | FileCheck %s --check-prefix=CHECK-UNKNOWN

// ---------------------------------------------------------------------------//
// Test 64-bit form (x0) and its aliases
// ---------------------------------------------------------------------------//

sqdecb  x0
// CHECK-INST: sqdecb  x0
// CHECK-ENCODING: [0xe0,0xfb,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430fbe0 <unknown>

sqdecb  x0, all
// CHECK-INST: sqdecb  x0
// CHECK-ENCODING: [0xe0,0xfb,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430fbe0 <unknown>

sqdecb  x0, all, mul #1
// CHECK-INST: sqdecb  x0
// CHECK-ENCODING: [0xe0,0xfb,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430fbe0 <unknown>

sqdecb  x0, all, mul #16
// CHECK-INST: sqdecb  x0, all, mul #16
// CHECK-ENCODING: [0xe0,0xfb,0x3f,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 043ffbe0 <unknown>


// ---------------------------------------------------------------------------//
// Test 32-bit form (x0, w0) and its aliases
// ---------------------------------------------------------------------------//

sqdecb  x0, w0
// CHECK-INST: sqdecb  x0, w0
// CHECK-ENCODING: [0xe0,0xfb,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420fbe0 <unknown>

sqdecb  x0, w0, all
// CHECK-INST: sqdecb  x0, w0
// CHECK-ENCODING: [0xe0,0xfb,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420fbe0 <unknown>

sqdecb  x0, w0, all, mul #1
// CHECK-INST: sqdecb  x0, w0
// CHECK-ENCODING: [0xe0,0xfb,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420fbe0 <unknown>

sqdecb  x0, w0, all, mul #16
// CHECK-INST: sqdecb  x0, w0, all, mul #16
// CHECK-ENCODING: [0xe0,0xfb,0x2f,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 042ffbe0 <unknown>

sqdecb  x0, w0, pow2
// CHECK-INST: sqdecb  x0, w0, pow2
// CHECK-ENCODING: [0x00,0xf8,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420f800 <unknown>

sqdecb  x0, w0, pow2, mul #16
// CHECK-INST: sqdecb  x0, w0, pow2, mul #16
// CHECK-ENCODING: [0x00,0xf8,0x2f,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 042ff800 <unknown>


// ---------------------------------------------------------------------------//
// Test all patterns for 64-bit form
// ---------------------------------------------------------------------------//

sqdecb  x0, pow2
// CHECK-INST: sqdecb  x0, pow2
// CHECK-ENCODING: [0x00,0xf8,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430f800 <unknown>

sqdecb  x0, vl1
// CHECK-INST: sqdecb  x0, vl1
// CHECK-ENCODING: [0x20,0xf8,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430f820 <unknown>

sqdecb  x0, vl2
// CHECK-INST: sqdecb  x0, vl2
// CHECK-ENCODING: [0x40,0xf8,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430f840 <unknown>

sqdecb  x0, vl3
// CHECK-INST: sqdecb  x0, vl3
// CHECK-ENCODING: [0x60,0xf8,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430f860 <unknown>

sqdecb  x0, vl4
// CHECK-INST: sqdecb  x0, vl4
// CHECK-ENCODING: [0x80,0xf8,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430f880 <unknown>

sqdecb  x0, vl5
// CHECK-INST: sqdecb  x0, vl5
// CHECK-ENCODING: [0xa0,0xf8,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430f8a0 <unknown>

sqdecb  x0, vl6
// CHECK-INST: sqdecb  x0, vl6
// CHECK-ENCODING: [0xc0,0xf8,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430f8c0 <unknown>

sqdecb  x0, vl7
// CHECK-INST: sqdecb  x0, vl7
// CHECK-ENCODING: [0xe0,0xf8,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430f8e0 <unknown>

sqdecb  x0, vl8
// CHECK-INST: sqdecb  x0, vl8
// CHECK-ENCODING: [0x00,0xf9,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430f900 <unknown>

sqdecb  x0, vl16
// CHECK-INST: sqdecb  x0, vl16
// CHECK-ENCODING: [0x20,0xf9,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430f920 <unknown>

sqdecb  x0, vl32
// CHECK-INST: sqdecb  x0, vl32
// CHECK-ENCODING: [0x40,0xf9,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430f940 <unknown>

sqdecb  x0, vl64
// CHECK-INST: sqdecb  x0, vl64
// CHECK-ENCODING: [0x60,0xf9,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430f960 <unknown>

sqdecb  x0, vl128
// CHECK-INST: sqdecb  x0, vl128
// CHECK-ENCODING: [0x80,0xf9,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430f980 <unknown>

sqdecb  x0, vl256
// CHECK-INST: sqdecb  x0, vl256
// CHECK-ENCODING: [0xa0,0xf9,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430f9a0 <unknown>

sqdecb  x0, #14
// CHECK-INST: sqdecb  x0, #14
// CHECK-ENCODING: [0xc0,0xf9,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430f9c0 <unknown>

sqdecb  x0, #15
// CHECK-INST: sqdecb  x0, #15
// CHECK-ENCODING: [0xe0,0xf9,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430f9e0 <unknown>

sqdecb  x0, #16
// CHECK-INST: sqdecb  x0, #16
// CHECK-ENCODING: [0x00,0xfa,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430fa00 <unknown>

sqdecb  x0, #17
// CHECK-INST: sqdecb  x0, #17
// CHECK-ENCODING: [0x20,0xfa,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430fa20 <unknown>

sqdecb  x0, #18
// CHECK-INST: sqdecb  x0, #18
// CHECK-ENCODING: [0x40,0xfa,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430fa40 <unknown>

sqdecb  x0, #19
// CHECK-INST: sqdecb  x0, #19
// CHECK-ENCODING: [0x60,0xfa,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430fa60 <unknown>

sqdecb  x0, #20
// CHECK-INST: sqdecb  x0, #20
// CHECK-ENCODING: [0x80,0xfa,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430fa80 <unknown>

sqdecb  x0, #21
// CHECK-INST: sqdecb  x0, #21
// CHECK-ENCODING: [0xa0,0xfa,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430faa0 <unknown>

sqdecb  x0, #22
// CHECK-INST: sqdecb  x0, #22
// CHECK-ENCODING: [0xc0,0xfa,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430fac0 <unknown>

sqdecb  x0, #23
// CHECK-INST: sqdecb  x0, #23
// CHECK-ENCODING: [0xe0,0xfa,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430fae0 <unknown>

sqdecb  x0, #24
// CHECK-INST: sqdecb  x0, #24
// CHECK-ENCODING: [0x00,0xfb,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430fb00 <unknown>

sqdecb  x0, #25
// CHECK-INST: sqdecb  x0, #25
// CHECK-ENCODING: [0x20,0xfb,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430fb20 <unknown>

sqdecb  x0, #26
// CHECK-INST: sqdecb  x0, #26
// CHECK-ENCODING: [0x40,0xfb,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430fb40 <unknown>

sqdecb  x0, #27
// CHECK-INST: sqdecb  x0, #27
// CHECK-ENCODING: [0x60,0xfb,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430fb60 <unknown>

sqdecb  x0, #28
// CHECK-INST: sqdecb  x0, #28
// CHECK-ENCODING: [0x80,0xfb,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430fb80 <unknown>
