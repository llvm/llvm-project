// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:   | llvm-objdump -d --mattr=-sve - | FileCheck %s --check-prefix=CHECK-UNKNOWN

// ---------------------------------------------------------------------------//
// Test 64-bit form (x0) and its aliases
// ---------------------------------------------------------------------------//

sqincb  x0
// CHECK-INST: sqincb  x0
// CHECK-ENCODING: [0xe0,0xf3,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430f3e0 <unknown>

sqincb  x0, all
// CHECK-INST: sqincb  x0
// CHECK-ENCODING: [0xe0,0xf3,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430f3e0 <unknown>

sqincb  x0, all, mul #1
// CHECK-INST: sqincb  x0
// CHECK-ENCODING: [0xe0,0xf3,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430f3e0 <unknown>

sqincb  x0, all, mul #16
// CHECK-INST: sqincb  x0, all, mul #16
// CHECK-ENCODING: [0xe0,0xf3,0x3f,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 043ff3e0 <unknown>


// ---------------------------------------------------------------------------//
// Test 32-bit form (x0, w0) and its aliases
// ---------------------------------------------------------------------------//

sqincb  x0, w0
// CHECK-INST: sqincb  x0, w0
// CHECK-ENCODING: [0xe0,0xf3,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420f3e0 <unknown>

sqincb  x0, w0, all
// CHECK-INST: sqincb  x0, w0
// CHECK-ENCODING: [0xe0,0xf3,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420f3e0 <unknown>

sqincb  x0, w0, all, mul #1
// CHECK-INST: sqincb  x0, w0
// CHECK-ENCODING: [0xe0,0xf3,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420f3e0 <unknown>

sqincb  x0, w0, all, mul #16
// CHECK-INST: sqincb  x0, w0, all, mul #16
// CHECK-ENCODING: [0xe0,0xf3,0x2f,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 042ff3e0 <unknown>

sqincb  x0, w0, pow2
// CHECK-INST: sqincb  x0, w0, pow2
// CHECK-ENCODING: [0x00,0xf0,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420f000 <unknown>

sqincb  x0, w0, pow2, mul #16
// CHECK-INST: sqincb  x0, w0, pow2, mul #16
// CHECK-ENCODING: [0x00,0xf0,0x2f,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 042ff000 <unknown>


// ---------------------------------------------------------------------------//
// Test all patterns for 64-bit form
// ---------------------------------------------------------------------------//

sqincb  x0, pow2
// CHECK-INST: sqincb  x0, pow2
// CHECK-ENCODING: [0x00,0xf0,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430f000 <unknown>

sqincb  x0, vl1
// CHECK-INST: sqincb  x0, vl1
// CHECK-ENCODING: [0x20,0xf0,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430f020 <unknown>

sqincb  x0, vl2
// CHECK-INST: sqincb  x0, vl2
// CHECK-ENCODING: [0x40,0xf0,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430f040 <unknown>

sqincb  x0, vl3
// CHECK-INST: sqincb  x0, vl3
// CHECK-ENCODING: [0x60,0xf0,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430f060 <unknown>

sqincb  x0, vl4
// CHECK-INST: sqincb  x0, vl4
// CHECK-ENCODING: [0x80,0xf0,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430f080 <unknown>

sqincb  x0, vl5
// CHECK-INST: sqincb  x0, vl5
// CHECK-ENCODING: [0xa0,0xf0,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430f0a0 <unknown>

sqincb  x0, vl6
// CHECK-INST: sqincb  x0, vl6
// CHECK-ENCODING: [0xc0,0xf0,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430f0c0 <unknown>

sqincb  x0, vl7
// CHECK-INST: sqincb  x0, vl7
// CHECK-ENCODING: [0xe0,0xf0,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430f0e0 <unknown>

sqincb  x0, vl8
// CHECK-INST: sqincb  x0, vl8
// CHECK-ENCODING: [0x00,0xf1,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430f100 <unknown>

sqincb  x0, vl16
// CHECK-INST: sqincb  x0, vl16
// CHECK-ENCODING: [0x20,0xf1,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430f120 <unknown>

sqincb  x0, vl32
// CHECK-INST: sqincb  x0, vl32
// CHECK-ENCODING: [0x40,0xf1,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430f140 <unknown>

sqincb  x0, vl64
// CHECK-INST: sqincb  x0, vl64
// CHECK-ENCODING: [0x60,0xf1,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430f160 <unknown>

sqincb  x0, vl128
// CHECK-INST: sqincb  x0, vl128
// CHECK-ENCODING: [0x80,0xf1,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430f180 <unknown>

sqincb  x0, vl256
// CHECK-INST: sqincb  x0, vl256
// CHECK-ENCODING: [0xa0,0xf1,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430f1a0 <unknown>

sqincb  x0, #14
// CHECK-INST: sqincb  x0, #14
// CHECK-ENCODING: [0xc0,0xf1,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430f1c0 <unknown>

sqincb  x0, #15
// CHECK-INST: sqincb  x0, #15
// CHECK-ENCODING: [0xe0,0xf1,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430f1e0 <unknown>

sqincb  x0, #16
// CHECK-INST: sqincb  x0, #16
// CHECK-ENCODING: [0x00,0xf2,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430f200 <unknown>

sqincb  x0, #17
// CHECK-INST: sqincb  x0, #17
// CHECK-ENCODING: [0x20,0xf2,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430f220 <unknown>

sqincb  x0, #18
// CHECK-INST: sqincb  x0, #18
// CHECK-ENCODING: [0x40,0xf2,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430f240 <unknown>

sqincb  x0, #19
// CHECK-INST: sqincb  x0, #19
// CHECK-ENCODING: [0x60,0xf2,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430f260 <unknown>

sqincb  x0, #20
// CHECK-INST: sqincb  x0, #20
// CHECK-ENCODING: [0x80,0xf2,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430f280 <unknown>

sqincb  x0, #21
// CHECK-INST: sqincb  x0, #21
// CHECK-ENCODING: [0xa0,0xf2,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430f2a0 <unknown>

sqincb  x0, #22
// CHECK-INST: sqincb  x0, #22
// CHECK-ENCODING: [0xc0,0xf2,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430f2c0 <unknown>

sqincb  x0, #23
// CHECK-INST: sqincb  x0, #23
// CHECK-ENCODING: [0xe0,0xf2,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430f2e0 <unknown>

sqincb  x0, #24
// CHECK-INST: sqincb  x0, #24
// CHECK-ENCODING: [0x00,0xf3,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430f300 <unknown>

sqincb  x0, #25
// CHECK-INST: sqincb  x0, #25
// CHECK-ENCODING: [0x20,0xf3,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430f320 <unknown>

sqincb  x0, #26
// CHECK-INST: sqincb  x0, #26
// CHECK-ENCODING: [0x40,0xf3,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430f340 <unknown>

sqincb  x0, #27
// CHECK-INST: sqincb  x0, #27
// CHECK-ENCODING: [0x60,0xf3,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430f360 <unknown>

sqincb  x0, #28
// CHECK-INST: sqincb  x0, #28
// CHECK-ENCODING: [0x80,0xf3,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430f380 <unknown>

