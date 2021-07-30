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
uqdecd  x0
// CHECK-INST: uqdecd  x0
// CHECK-ENCODING: [0xe0,0xff,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 ff f0 04 <unknown>

uqdecd  x0, all
// CHECK-INST: uqdecd  x0
// CHECK-ENCODING: [0xe0,0xff,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 ff f0 04 <unknown>

uqdecd  x0, all, mul #1
// CHECK-INST: uqdecd  x0
// CHECK-ENCODING: [0xe0,0xff,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 ff f0 04 <unknown>

uqdecd  x0, all, mul #16
// CHECK-INST: uqdecd  x0, all, mul #16
// CHECK-ENCODING: [0xe0,0xff,0xff,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 ff ff 04 <unknown>


// ---------------------------------------------------------------------------//
// Test 32-bit form (w0) and its aliases
// ---------------------------------------------------------------------------//

uqdecd  w0
// CHECK-INST: uqdecd  w0
// CHECK-ENCODING: [0xe0,0xff,0xe0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 ff e0 04 <unknown>

uqdecd  w0, all
// CHECK-INST: uqdecd  w0
// CHECK-ENCODING: [0xe0,0xff,0xe0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 ff e0 04 <unknown>

uqdecd  w0, all, mul #1
// CHECK-INST: uqdecd  w0
// CHECK-ENCODING: [0xe0,0xff,0xe0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 ff e0 04 <unknown>

uqdecd  w0, all, mul #16
// CHECK-INST: uqdecd  w0, all, mul #16
// CHECK-ENCODING: [0xe0,0xff,0xef,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 ff ef 04 <unknown>

uqdecd  w0, pow2
// CHECK-INST: uqdecd  w0, pow2
// CHECK-ENCODING: [0x00,0xfc,0xe0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 00 fc e0 04 <unknown>

uqdecd  w0, pow2, mul #16
// CHECK-INST: uqdecd  w0, pow2, mul #16
// CHECK-ENCODING: [0x00,0xfc,0xef,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 00 fc ef 04 <unknown>


// ---------------------------------------------------------------------------//
// Test vector form and aliases.
// ---------------------------------------------------------------------------//
uqdecd  z0.d
// CHECK-INST: uqdecd  z0.d
// CHECK-ENCODING: [0xe0,0xcf,0xe0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 cf e0 04 <unknown>

uqdecd  z0.d, all
// CHECK-INST: uqdecd  z0.d
// CHECK-ENCODING: [0xe0,0xcf,0xe0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 cf e0 04 <unknown>

uqdecd  z0.d, all, mul #1
// CHECK-INST: uqdecd  z0.d
// CHECK-ENCODING: [0xe0,0xcf,0xe0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 cf e0 04 <unknown>

uqdecd  z0.d, all, mul #16
// CHECK-INST: uqdecd  z0.d, all, mul #16
// CHECK-ENCODING: [0xe0,0xcf,0xef,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 cf ef 04 <unknown>

uqdecd  z0.d, pow2
// CHECK-INST: uqdecd  z0.d, pow2
// CHECK-ENCODING: [0x00,0xcc,0xe0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 00 cc e0 04 <unknown>

uqdecd  z0.d, pow2, mul #16
// CHECK-INST: uqdecd  z0.d, pow2, mul #16
// CHECK-ENCODING: [0x00,0xcc,0xef,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 00 cc ef 04 <unknown>


// ---------------------------------------------------------------------------//
// Test all patterns for 64-bit form
// ---------------------------------------------------------------------------//

uqdecd  x0, pow2
// CHECK-INST: uqdecd  x0, pow2
// CHECK-ENCODING: [0x00,0xfc,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 00 fc f0 04 <unknown>

uqdecd  x0, vl1
// CHECK-INST: uqdecd  x0, vl1
// CHECK-ENCODING: [0x20,0xfc,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 20 fc f0 04 <unknown>

uqdecd  x0, vl2
// CHECK-INST: uqdecd  x0, vl2
// CHECK-ENCODING: [0x40,0xfc,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 40 fc f0 04 <unknown>

uqdecd  x0, vl3
// CHECK-INST: uqdecd  x0, vl3
// CHECK-ENCODING: [0x60,0xfc,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 60 fc f0 04 <unknown>

uqdecd  x0, vl4
// CHECK-INST: uqdecd  x0, vl4
// CHECK-ENCODING: [0x80,0xfc,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 80 fc f0 04 <unknown>

uqdecd  x0, vl5
// CHECK-INST: uqdecd  x0, vl5
// CHECK-ENCODING: [0xa0,0xfc,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: a0 fc f0 04 <unknown>

uqdecd  x0, vl6
// CHECK-INST: uqdecd  x0, vl6
// CHECK-ENCODING: [0xc0,0xfc,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: c0 fc f0 04 <unknown>

uqdecd  x0, vl7
// CHECK-INST: uqdecd  x0, vl7
// CHECK-ENCODING: [0xe0,0xfc,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 fc f0 04 <unknown>

uqdecd  x0, vl8
// CHECK-INST: uqdecd  x0, vl8
// CHECK-ENCODING: [0x00,0xfd,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 00 fd f0 04 <unknown>

uqdecd  x0, vl16
// CHECK-INST: uqdecd  x0, vl16
// CHECK-ENCODING: [0x20,0xfd,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 20 fd f0 04 <unknown>

uqdecd  x0, vl32
// CHECK-INST: uqdecd  x0, vl32
// CHECK-ENCODING: [0x40,0xfd,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 40 fd f0 04 <unknown>

uqdecd  x0, vl64
// CHECK-INST: uqdecd  x0, vl64
// CHECK-ENCODING: [0x60,0xfd,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 60 fd f0 04 <unknown>

uqdecd  x0, vl128
// CHECK-INST: uqdecd  x0, vl128
// CHECK-ENCODING: [0x80,0xfd,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 80 fd f0 04 <unknown>

uqdecd  x0, vl256
// CHECK-INST: uqdecd  x0, vl256
// CHECK-ENCODING: [0xa0,0xfd,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: a0 fd f0 04 <unknown>

uqdecd  x0, #14
// CHECK-INST: uqdecd  x0, #14
// CHECK-ENCODING: [0xc0,0xfd,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: c0 fd f0 04 <unknown>

uqdecd  x0, #15
// CHECK-INST: uqdecd  x0, #15
// CHECK-ENCODING: [0xe0,0xfd,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 fd f0 04 <unknown>

uqdecd  x0, #16
// CHECK-INST: uqdecd  x0, #16
// CHECK-ENCODING: [0x00,0xfe,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 00 fe f0 04 <unknown>

uqdecd  x0, #17
// CHECK-INST: uqdecd  x0, #17
// CHECK-ENCODING: [0x20,0xfe,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 20 fe f0 04 <unknown>

uqdecd  x0, #18
// CHECK-INST: uqdecd  x0, #18
// CHECK-ENCODING: [0x40,0xfe,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 40 fe f0 04 <unknown>

uqdecd  x0, #19
// CHECK-INST: uqdecd  x0, #19
// CHECK-ENCODING: [0x60,0xfe,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 60 fe f0 04 <unknown>

uqdecd  x0, #20
// CHECK-INST: uqdecd  x0, #20
// CHECK-ENCODING: [0x80,0xfe,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 80 fe f0 04 <unknown>

uqdecd  x0, #21
// CHECK-INST: uqdecd  x0, #21
// CHECK-ENCODING: [0xa0,0xfe,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: a0 fe f0 04 <unknown>

uqdecd  x0, #22
// CHECK-INST: uqdecd  x0, #22
// CHECK-ENCODING: [0xc0,0xfe,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: c0 fe f0 04 <unknown>

uqdecd  x0, #23
// CHECK-INST: uqdecd  x0, #23
// CHECK-ENCODING: [0xe0,0xfe,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 fe f0 04 <unknown>

uqdecd  x0, #24
// CHECK-INST: uqdecd  x0, #24
// CHECK-ENCODING: [0x00,0xff,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 00 ff f0 04 <unknown>

uqdecd  x0, #25
// CHECK-INST: uqdecd  x0, #25
// CHECK-ENCODING: [0x20,0xff,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 20 ff f0 04 <unknown>

uqdecd  x0, #26
// CHECK-INST: uqdecd  x0, #26
// CHECK-ENCODING: [0x40,0xff,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 40 ff f0 04 <unknown>

uqdecd  x0, #27
// CHECK-INST: uqdecd  x0, #27
// CHECK-ENCODING: [0x60,0xff,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 60 ff f0 04 <unknown>

uqdecd  x0, #28
// CHECK-INST: uqdecd  x0, #28
// CHECK-ENCODING: [0x80,0xff,0xf0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 80 ff f0 04 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 bc 20 04 <unknown>

uqdecd  z0.d
// CHECK-INST: uqdecd	z0.d
// CHECK-ENCODING: [0xe0,0xcf,0xe0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 cf e0 04 <unknown>

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 bc 20 04 <unknown>

uqdecd  z0.d, pow2, mul #16
// CHECK-INST: uqdecd	z0.d, pow2, mul #16
// CHECK-ENCODING: [0x00,0xcc,0xef,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 00 cc ef 04 <unknown>

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e0 bc 20 04 <unknown>

uqdecd  z0.d, pow2
// CHECK-INST: uqdecd	z0.d, pow2
// CHECK-ENCODING: [0x00,0xcc,0xe0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 00 cc e0 04 <unknown>
