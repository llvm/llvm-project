// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve,+f64mm < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve,+f64mm < %s \
// RUN:        | llvm-objdump -d --mattr=+sve,+f64mm - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve,+f64mm < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

// --------------------------------------------------------------------------//
// FMMLA (SVE)

fmmla z0.d, z1.d, z2.d
// CHECK-INST: fmmla z0.d, z1.d, z2.d
// CHECK-ENCODING: [0x20,0xe4,0xe2,0x64]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: 20 e4 e2 64 <unknown>

// --------------------------------------------------------------------------//
// LD1RO (SVE, scalar plus immediate)

// With maximum immediate (224)

ld1rob { z0.b }, p1/z, [x2, #224]
// CHECK-INST: ld1rob { z0.b }, p1/z, [x2, #224]
// CHECK-ENCODING: [0x40,0x24,0x27,0xa4]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: 40 24 27 a4 <unknown>

ld1roh { z0.h }, p1/z, [x2, #224]
// CHECK-INST: ld1roh { z0.h }, p1/z, [x2, #224]
// CHECK-ENCODING: [0x40,0x24,0xa7,0xa4]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: 40 24 a7 a4 <unknown>

ld1row { z0.s }, p1/z, [x2, #224]
// CHECK-INST: ld1row { z0.s }, p1/z, [x2, #224]
// CHECK-ENCODING: [0x40,0x24,0x27,0xa5]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: 40 24 27 a5 <unknown>

ld1rod { z0.d }, p1/z, [x2, #224]
// CHECK-INST: ld1rod { z0.d }, p1/z, [x2, #224]
// CHECK-ENCODING: [0x40,0x24,0xa7,0xa5]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: 40 24 a7 a5 <unknown>

// With minimum immediate (-256)

ld1rob { z0.b }, p1/z, [x2, #-256]
// CHECK-INST: ld1rob { z0.b }, p1/z, [x2, #-256]
// CHECK-ENCODING: [0x40,0x24,0x28,0xa4]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: 40 24 28 a4 <unknown>

ld1roh { z0.h }, p1/z, [x2, #-256]
// CHECK-INST: ld1roh { z0.h }, p1/z, [x2, #-256]
// CHECK-ENCODING: [0x40,0x24,0xa8,0xa4]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: 40 24 a8 a4 <unknown>

ld1row { z0.s }, p1/z, [x2, #-256]
// CHECK-INST: ld1row { z0.s }, p1/z, [x2, #-256]
// CHECK-ENCODING: [0x40,0x24,0x28,0xa5]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: 40 24 28 a5 <unknown>

ld1rod { z0.d }, p1/z, [x2, #-256]
// CHECK-INST: ld1rod { z0.d }, p1/z, [x2, #-256]
// CHECK-ENCODING: [0x40,0x24,0xa8,0xa5]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: 40 24 a8 a5 <unknown>

// Aliases with a vector first operand, and omitted offset.

ld1rob { z0.b }, p1/z, [x2]
// CHECK-INST: ld1rob { z0.b }, p1/z, [x2]
// CHECK-ENCODING: [0x40,0x24,0x20,0xa4]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: 40 24 20 a4 <unknown>

ld1roh { z0.h }, p1/z, [x2]
// CHECK-INST: ld1roh { z0.h }, p1/z, [x2]
// CHECK-ENCODING: [0x40,0x24,0xa0,0xa4]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: 40 24 a0 a4 <unknown>

ld1row { z0.s }, p1/z, [x2]
// CHECK-INST: ld1row { z0.s }, p1/z, [x2]
// CHECK-ENCODING: [0x40,0x24,0x20,0xa5]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: 40 24 20 a5 <unknown>

ld1rod { z0.d }, p1/z, [x2]
// CHECK-INST: ld1rod { z0.d }, p1/z, [x2]
// CHECK-ENCODING: [0x40,0x24,0xa0,0xa5]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: 40 24 a0 a5 <unknown>

// Aliases with a plain (non-list) first operand, and omitted offset.

ld1rob z0.b, p1/z, [x2]
// CHECK-INST: ld1rob { z0.b }, p1/z, [x2]
// CHECK-ENCODING: [0x40,0x24,0x20,0xa4]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: 40 24 20 a4 <unknown>

ld1roh z0.h, p1/z, [x2]
// CHECK-INST: ld1roh { z0.h }, p1/z, [x2]
// CHECK-ENCODING: [0x40,0x24,0xa0,0xa4]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: 40 24 a0 a4 <unknown>

ld1row z0.s, p1/z, [x2]
// CHECK-INST: ld1row { z0.s }, p1/z, [x2]
// CHECK-ENCODING: [0x40,0x24,0x20,0xa5]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: 40 24 20 a5 <unknown>

ld1rod z0.d, p1/z, [x2]
// CHECK-INST: ld1rod { z0.d }, p1/z, [x2]
// CHECK-ENCODING: [0x40,0x24,0xa0,0xa5]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: 40 24 a0 a5 <unknown>

// Aliases with a plain (non-list) first operand, plus offset.

// With maximum immediate (224)

ld1rob z0.b, p1/z, [x2, #224]
// CHECK-INST: ld1rob { z0.b }, p1/z, [x2, #224]
// CHECK-ENCODING: [0x40,0x24,0x27,0xa4]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: 40 24 27 a4 <unknown>

ld1roh z0.h, p1/z, [x2, #224]
// CHECK-INST: ld1roh { z0.h }, p1/z, [x2, #224]
// CHECK-ENCODING: [0x40,0x24,0xa7,0xa4]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: 40 24 a7 a4 <unknown>

ld1row z0.s, p1/z, [x2, #224]
// CHECK-INST: ld1row { z0.s }, p1/z, [x2, #224]
// CHECK-ENCODING: [0x40,0x24,0x27,0xa5]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: 40 24 27 a5 <unknown>

ld1rod z0.d, p1/z, [x2, #224]
// CHECK-INST: ld1rod { z0.d }, p1/z, [x2, #224]
// CHECK-ENCODING: [0x40,0x24,0xa7,0xa5]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: 40 24 a7 a5 <unknown>

// With minimum immediate (-256)

ld1rob z0.b, p1/z, [x2, #-256]
// CHECK-INST: ld1rob { z0.b }, p1/z, [x2, #-256]
// CHECK-ENCODING: [0x40,0x24,0x28,0xa4]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: 40 24 28 a4 <unknown>

ld1roh z0.h, p1/z, [x2, #-256]
// CHECK-INST: ld1roh { z0.h }, p1/z, [x2, #-256]
// CHECK-ENCODING: [0x40,0x24,0xa8,0xa4]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: 40 24 a8 a4 <unknown>

ld1row z0.s, p1/z, [x2, #-256]
// CHECK-INST: ld1row { z0.s }, p1/z, [x2, #-256]
// CHECK-ENCODING: [0x40,0x24,0x28,0xa5]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: 40 24 28 a5 <unknown>

ld1rod z0.d, p1/z, [x2, #-256]
// CHECK-INST: ld1rod { z0.d }, p1/z, [x2, #-256]
// CHECK-ENCODING: [0x40,0x24,0xa8,0xa5]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: 40 24 a8 a5 <unknown>


// --------------------------------------------------------------------------//
// LD1RO (SVE, scalar plus scalar)

ld1rob { z0.b }, p1/z, [x2, x3, lsl #0]
// CHECK-INST: ld1rob { z0.b }, p1/z, [x2, x3]
// CHECK-ENCODING: [0x40,0x04,0x23,0xa4]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: 40 04 23 a4 <unknown>

ld1roh { z0.h }, p1/z, [x2, x3, lsl #1]
// CHECK-INST: ld1roh { z0.h }, p1/z, [x2, x3, lsl #1]
// CHECK-ENCODING: [0x40,0x04,0xa3,0xa4]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: 40 04 a3 a4 <unknown>

ld1row { z0.s }, p1/z, [x2, x3, lsl #2]
// CHECK-INST: ld1row { z0.s }, p1/z, [x2, x3, lsl #2]
// CHECK-ENCODING: [0x40,0x04,0x23,0xa5]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: 40 04 23 a5 <unknown>

ld1rod { z0.d }, p1/z, [x2, x3, lsl #3]
// CHECK-INST: ld1rod { z0.d }, p1/z, [x2, x3, lsl #3]
// CHECK-ENCODING: [0x40,0x04,0xa3,0xa5]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: 40 04 a3 a5 <unknown>

// Aliases with a plain (non-list) first operand, and omitted shift for the
// byte variant.

ld1rob z0.b, p1/z, [x2, x3]
// CHECK-INST: ld1rob { z0.b }, p1/z, [x2, x3]
// CHECK-ENCODING: [0x40,0x04,0x23,0xa4]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: 40 04 23 a4 <unknown>

ld1roh z0.h, p1/z, [x2, x3, lsl #1]
// CHECK-INST: ld1roh { z0.h }, p1/z, [x2, x3, lsl #1]
// CHECK-ENCODING: [0x40,0x04,0xa3,0xa4]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: 40 04 a3 a4 <unknown>

ld1row z0.s, p1/z, [x2, x3, lsl #2]
// CHECK-INST: ld1row { z0.s }, p1/z, [x2, x3, lsl #2]
// CHECK-ENCODING: [0x40,0x04,0x23,0xa5]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: 40 04 23 a5 <unknown>

ld1rod z0.d, p1/z, [x2, x3, lsl #3]
// CHECK-INST: ld1rod { z0.d }, p1/z, [x2, x3, lsl #3]
// CHECK-ENCODING: [0x40,0x04,0xa3,0xa5]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: 40 04 a3 a5 <unknown>


// --------------------------------------------------------------------------//
// ZIP1, ZIP2 (SVE, 128-bit element)

zip1 z0.q, z1.q, z2.q
// CHECK-INST: zip1 z0.q, z1.q, z2.q
// CHECK-ENCODING: [0x20,0x00,0xa2,0x05]
// CHECK-ERROR: instruction requires: f64mm streaming-sve
// CHECK-UNKNOWN: 20 00 a2 05 <unknown>

zip2 z0.q, z1.q, z2.q
// CHECK-INST: zip2 z0.q, z1.q, z2.q
// CHECK-ENCODING: [0x20,0x04,0xa2,0x05]
// CHECK-ERROR: instruction requires: f64mm streaming-sve
// CHECK-UNKNOWN: 20 04 a2 05 <unknown>


// --------------------------------------------------------------------------//
// UZP1, UZP2 (SVE, 128-bit element)

uzp1 z0.q, z1.q, z2.q
// CHECK-INST: uzp1 z0.q, z1.q, z2.q
// CHECK-ENCODING: [0x20,0x08,0xa2,0x05]
// CHECK-ERROR: instruction requires: f64mm streaming-sve
// CHECK-UNKNOWN: 20 08 a2 05 <unknown>

uzp2 z0.q, z1.q, z2.q
// CHECK-INST: uzp2 z0.q, z1.q, z2.q
// CHECK-ENCODING: [0x20,0x0c,0xa2,0x05]
// CHECK-ERROR: instruction requires: f64mm streaming-sve
// CHECK-UNKNOWN: 20 0c a2 05 <unknown>


// --------------------------------------------------------------------------//
// TRN1, TRN2 (SVE, 128-bit element)

trn1 z0.q, z1.q, z2.q
// CHECK-INST: trn1 z0.q, z1.q, z2.q
// CHECK-ENCODING: [0x20,0x18,0xa2,0x05]
// CHECK-ERROR: instruction requires: f64mm streaming-sve
// CHECK-UNKNOWN: 20 18 a2 05 <unknown>

trn2 z0.q, z1.q, z2.q
// CHECK-INST: trn2 z0.q, z1.q, z2.q
// CHECK-ENCODING: [0x20,0x1c,0xa2,0x05]
// CHECK-ERROR: instruction requires: f64mm streaming-sve
// CHECK-UNKNOWN: 20 1c a2 05 <unknown>
