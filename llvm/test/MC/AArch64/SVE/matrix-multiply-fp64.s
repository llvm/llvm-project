// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve,+f64mm < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve,+f64mm < %s \
// RUN:        | llvm-objdump --no-print-imm-hex -d --mattr=+sve,+f64mm - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve,+f64mm < %s \
// RUN:   | llvm-objdump --no-print-imm-hex -d --mattr=-sve - | FileCheck %s --check-prefix=CHECK-UNKNOWN

// --------------------------------------------------------------------------//
// FMMLA (SVE)

fmmla z0.d, z1.d, z2.d
// CHECK-INST: fmmla z0.d, z1.d, z2.d
// CHECK-ENCODING: [0x20,0xe4,0xe2,0x64]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: 64e2e420 <unknown>

// --------------------------------------------------------------------------//
// LD1RO (SVE, scalar plus immediate)

// With maximum immediate (224)

ld1rob { z0.b }, p1/z, [x2, #224]
// CHECK-INST: ld1rob { z0.b }, p1/z, [x2, #224]
// CHECK-ENCODING: [0x40,0x24,0x27,0xa4]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: a4272440 <unknown>

ld1roh { z0.h }, p1/z, [x2, #224]
// CHECK-INST: ld1roh { z0.h }, p1/z, [x2, #224]
// CHECK-ENCODING: [0x40,0x24,0xa7,0xa4]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: a4a72440 <unknown>

ld1row { z0.s }, p1/z, [x2, #224]
// CHECK-INST: ld1row { z0.s }, p1/z, [x2, #224]
// CHECK-ENCODING: [0x40,0x24,0x27,0xa5]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: a5272440 <unknown>

ld1rod { z0.d }, p1/z, [x2, #224]
// CHECK-INST: ld1rod { z0.d }, p1/z, [x2, #224]
// CHECK-ENCODING: [0x40,0x24,0xa7,0xa5]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: a5a72440 <unknown>

// With minimum immediate (-256)

ld1rob { z0.b }, p1/z, [x2, #-256]
// CHECK-INST: ld1rob { z0.b }, p1/z, [x2, #-256]
// CHECK-ENCODING: [0x40,0x24,0x28,0xa4]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: a4282440 <unknown>

ld1roh { z0.h }, p1/z, [x2, #-256]
// CHECK-INST: ld1roh { z0.h }, p1/z, [x2, #-256]
// CHECK-ENCODING: [0x40,0x24,0xa8,0xa4]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: a4a82440 <unknown>

ld1row { z0.s }, p1/z, [x2, #-256]
// CHECK-INST: ld1row { z0.s }, p1/z, [x2, #-256]
// CHECK-ENCODING: [0x40,0x24,0x28,0xa5]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: a5282440 <unknown>

ld1rod { z0.d }, p1/z, [x2, #-256]
// CHECK-INST: ld1rod { z0.d }, p1/z, [x2, #-256]
// CHECK-ENCODING: [0x40,0x24,0xa8,0xa5]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: a5a82440 <unknown>

// Aliases with a vector first operand, and omitted offset.

ld1rob { z0.b }, p1/z, [x2]
// CHECK-INST: ld1rob { z0.b }, p1/z, [x2]
// CHECK-ENCODING: [0x40,0x24,0x20,0xa4]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: a4202440 <unknown>

ld1roh { z0.h }, p1/z, [x2]
// CHECK-INST: ld1roh { z0.h }, p1/z, [x2]
// CHECK-ENCODING: [0x40,0x24,0xa0,0xa4]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: a4a02440 <unknown>

ld1row { z0.s }, p1/z, [x2]
// CHECK-INST: ld1row { z0.s }, p1/z, [x2]
// CHECK-ENCODING: [0x40,0x24,0x20,0xa5]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: a5202440 <unknown>

ld1rod { z0.d }, p1/z, [x2]
// CHECK-INST: ld1rod { z0.d }, p1/z, [x2]
// CHECK-ENCODING: [0x40,0x24,0xa0,0xa5]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: a5a02440 <unknown>

// Aliases with a plain (non-list) first operand, and omitted offset.

ld1rob z0.b, p1/z, [x2]
// CHECK-INST: ld1rob { z0.b }, p1/z, [x2]
// CHECK-ENCODING: [0x40,0x24,0x20,0xa4]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: a4202440 <unknown>

ld1roh z0.h, p1/z, [x2]
// CHECK-INST: ld1roh { z0.h }, p1/z, [x2]
// CHECK-ENCODING: [0x40,0x24,0xa0,0xa4]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: a4a02440 <unknown>

ld1row z0.s, p1/z, [x2]
// CHECK-INST: ld1row { z0.s }, p1/z, [x2]
// CHECK-ENCODING: [0x40,0x24,0x20,0xa5]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: a5202440 <unknown>

ld1rod z0.d, p1/z, [x2]
// CHECK-INST: ld1rod { z0.d }, p1/z, [x2]
// CHECK-ENCODING: [0x40,0x24,0xa0,0xa5]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: a5a02440 <unknown>

// Aliases with a plain (non-list) first operand, plus offset.

// With maximum immediate (224)

ld1rob z0.b, p1/z, [x2, #224]
// CHECK-INST: ld1rob { z0.b }, p1/z, [x2, #224]
// CHECK-ENCODING: [0x40,0x24,0x27,0xa4]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: a4272440 <unknown>

ld1roh z0.h, p1/z, [x2, #224]
// CHECK-INST: ld1roh { z0.h }, p1/z, [x2, #224]
// CHECK-ENCODING: [0x40,0x24,0xa7,0xa4]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: a4a72440 <unknown>

ld1row z0.s, p1/z, [x2, #224]
// CHECK-INST: ld1row { z0.s }, p1/z, [x2, #224]
// CHECK-ENCODING: [0x40,0x24,0x27,0xa5]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: a5272440 <unknown>

ld1rod z0.d, p1/z, [x2, #224]
// CHECK-INST: ld1rod { z0.d }, p1/z, [x2, #224]
// CHECK-ENCODING: [0x40,0x24,0xa7,0xa5]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: a5a72440 <unknown>

// With minimum immediate (-256)

ld1rob z0.b, p1/z, [x2, #-256]
// CHECK-INST: ld1rob { z0.b }, p1/z, [x2, #-256]
// CHECK-ENCODING: [0x40,0x24,0x28,0xa4]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: a4282440 <unknown>

ld1roh z0.h, p1/z, [x2, #-256]
// CHECK-INST: ld1roh { z0.h }, p1/z, [x2, #-256]
// CHECK-ENCODING: [0x40,0x24,0xa8,0xa4]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: a4a82440 <unknown>

ld1row z0.s, p1/z, [x2, #-256]
// CHECK-INST: ld1row { z0.s }, p1/z, [x2, #-256]
// CHECK-ENCODING: [0x40,0x24,0x28,0xa5]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: a5282440 <unknown>

ld1rod z0.d, p1/z, [x2, #-256]
// CHECK-INST: ld1rod { z0.d }, p1/z, [x2, #-256]
// CHECK-ENCODING: [0x40,0x24,0xa8,0xa5]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: a5a82440 <unknown>


// --------------------------------------------------------------------------//
// LD1RO (SVE, scalar plus scalar)

ld1rob { z0.b }, p1/z, [x2, x3, lsl #0]
// CHECK-INST: ld1rob { z0.b }, p1/z, [x2, x3]
// CHECK-ENCODING: [0x40,0x04,0x23,0xa4]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: a4230440 <unknown>

ld1roh { z0.h }, p1/z, [x2, x3, lsl #1]
// CHECK-INST: ld1roh { z0.h }, p1/z, [x2, x3, lsl #1]
// CHECK-ENCODING: [0x40,0x04,0xa3,0xa4]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: a4a30440 <unknown>

ld1row { z0.s }, p1/z, [x2, x3, lsl #2]
// CHECK-INST: ld1row { z0.s }, p1/z, [x2, x3, lsl #2]
// CHECK-ENCODING: [0x40,0x04,0x23,0xa5]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: a5230440 <unknown>

ld1rod { z0.d }, p1/z, [x2, x3, lsl #3]
// CHECK-INST: ld1rod { z0.d }, p1/z, [x2, x3, lsl #3]
// CHECK-ENCODING: [0x40,0x04,0xa3,0xa5]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: a5a30440 <unknown>

// Aliases with a plain (non-list) first operand, and omitted shift for the
// byte variant.

ld1rob z0.b, p1/z, [x2, x3]
// CHECK-INST: ld1rob { z0.b }, p1/z, [x2, x3]
// CHECK-ENCODING: [0x40,0x04,0x23,0xa4]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: a4230440 <unknown>

ld1roh z0.h, p1/z, [x2, x3, lsl #1]
// CHECK-INST: ld1roh { z0.h }, p1/z, [x2, x3, lsl #1]
// CHECK-ENCODING: [0x40,0x04,0xa3,0xa4]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: a4a30440 <unknown>

ld1row z0.s, p1/z, [x2, x3, lsl #2]
// CHECK-INST: ld1row { z0.s }, p1/z, [x2, x3, lsl #2]
// CHECK-ENCODING: [0x40,0x04,0x23,0xa5]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: a5230440 <unknown>

ld1rod z0.d, p1/z, [x2, x3, lsl #3]
// CHECK-INST: ld1rod { z0.d }, p1/z, [x2, x3, lsl #3]
// CHECK-ENCODING: [0x40,0x04,0xa3,0xa5]
// CHECK-ERROR: instruction requires: f64mm sve
// CHECK-UNKNOWN: a5a30440 <unknown>


// --------------------------------------------------------------------------//
// ZIP1, ZIP2 (SVE, 128-bit element)

zip1 z0.q, z1.q, z2.q
// CHECK-INST: zip1 z0.q, z1.q, z2.q
// CHECK-ENCODING: [0x20,0x00,0xa2,0x05]
// CHECK-ERROR: instruction requires: f64mm sve or sme
// CHECK-UNKNOWN: 05a20020 <unknown>

zip2 z0.q, z1.q, z2.q
// CHECK-INST: zip2 z0.q, z1.q, z2.q
// CHECK-ENCODING: [0x20,0x04,0xa2,0x05]
// CHECK-ERROR: instruction requires: f64mm sve or sme
// CHECK-UNKNOWN: 05a20420 <unknown>


// --------------------------------------------------------------------------//
// UZP1, UZP2 (SVE, 128-bit element)

uzp1 z0.q, z1.q, z2.q
// CHECK-INST: uzp1 z0.q, z1.q, z2.q
// CHECK-ENCODING: [0x20,0x08,0xa2,0x05]
// CHECK-ERROR: instruction requires: f64mm sve or sme
// CHECK-UNKNOWN: 05a20820 <unknown>

uzp2 z0.q, z1.q, z2.q
// CHECK-INST: uzp2 z0.q, z1.q, z2.q
// CHECK-ENCODING: [0x20,0x0c,0xa2,0x05]
// CHECK-ERROR: instruction requires: f64mm sve or sme
// CHECK-UNKNOWN: 05a20c20 <unknown>


// --------------------------------------------------------------------------//
// TRN1, TRN2 (SVE, 128-bit element)

trn1 z0.q, z1.q, z2.q
// CHECK-INST: trn1 z0.q, z1.q, z2.q
// CHECK-ENCODING: [0x20,0x18,0xa2,0x05]
// CHECK-ERROR: instruction requires: f64mm sve or sme
// CHECK-UNKNOWN: 05a21820 <unknown>

trn2 z0.q, z1.q, z2.q
// CHECK-INST: trn2 z0.q, z1.q, z2.q
// CHECK-ENCODING: [0x20,0x1c,0xa2,0x05]
// CHECK-ERROR: instruction requires: f64mm sve or sme
// CHECK-UNKNOWN: 05a21c20 <unknown>
