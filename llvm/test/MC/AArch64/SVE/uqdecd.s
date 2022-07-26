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
uqdecd  x0
// CHECK-INST: uqdecd  x0
// CHECK-ENCODING: [0xe0,0xff,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04f0ffe0 <unknown>

uqdecd  x0, all
// CHECK-INST: uqdecd  x0
// CHECK-ENCODING: [0xe0,0xff,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04f0ffe0 <unknown>

uqdecd  x0, all, mul #1
// CHECK-INST: uqdecd  x0
// CHECK-ENCODING: [0xe0,0xff,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04f0ffe0 <unknown>

uqdecd  x0, all, mul #16
// CHECK-INST: uqdecd  x0, all, mul #16
// CHECK-ENCODING: [0xe0,0xff,0xff,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04ffffe0 <unknown>


// ---------------------------------------------------------------------------//
// Test 32-bit form (w0) and its aliases
// ---------------------------------------------------------------------------//

uqdecd  w0
// CHECK-INST: uqdecd  w0
// CHECK-ENCODING: [0xe0,0xff,0xe0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04e0ffe0 <unknown>

uqdecd  w0, all
// CHECK-INST: uqdecd  w0
// CHECK-ENCODING: [0xe0,0xff,0xe0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04e0ffe0 <unknown>

uqdecd  w0, all, mul #1
// CHECK-INST: uqdecd  w0
// CHECK-ENCODING: [0xe0,0xff,0xe0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04e0ffe0 <unknown>

uqdecd  w0, all, mul #16
// CHECK-INST: uqdecd  w0, all, mul #16
// CHECK-ENCODING: [0xe0,0xff,0xef,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04efffe0 <unknown>

uqdecd  w0, pow2
// CHECK-INST: uqdecd  w0, pow2
// CHECK-ENCODING: [0x00,0xfc,0xe0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04e0fc00 <unknown>

uqdecd  w0, pow2, mul #16
// CHECK-INST: uqdecd  w0, pow2, mul #16
// CHECK-ENCODING: [0x00,0xfc,0xef,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04effc00 <unknown>


// ---------------------------------------------------------------------------//
// Test vector form and aliases.
// ---------------------------------------------------------------------------//
uqdecd  z0.d
// CHECK-INST: uqdecd  z0.d
// CHECK-ENCODING: [0xe0,0xcf,0xe0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04e0cfe0 <unknown>

uqdecd  z0.d, all
// CHECK-INST: uqdecd  z0.d
// CHECK-ENCODING: [0xe0,0xcf,0xe0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04e0cfe0 <unknown>

uqdecd  z0.d, all, mul #1
// CHECK-INST: uqdecd  z0.d
// CHECK-ENCODING: [0xe0,0xcf,0xe0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04e0cfe0 <unknown>

uqdecd  z0.d, all, mul #16
// CHECK-INST: uqdecd  z0.d, all, mul #16
// CHECK-ENCODING: [0xe0,0xcf,0xef,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04efcfe0 <unknown>

uqdecd  z0.d, pow2
// CHECK-INST: uqdecd  z0.d, pow2
// CHECK-ENCODING: [0x00,0xcc,0xe0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04e0cc00 <unknown>

uqdecd  z0.d, pow2, mul #16
// CHECK-INST: uqdecd  z0.d, pow2, mul #16
// CHECK-ENCODING: [0x00,0xcc,0xef,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04efcc00 <unknown>


// ---------------------------------------------------------------------------//
// Test all patterns for 64-bit form
// ---------------------------------------------------------------------------//

uqdecd  x0, pow2
// CHECK-INST: uqdecd  x0, pow2
// CHECK-ENCODING: [0x00,0xfc,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04f0fc00 <unknown>

uqdecd  x0, vl1
// CHECK-INST: uqdecd  x0, vl1
// CHECK-ENCODING: [0x20,0xfc,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04f0fc20 <unknown>

uqdecd  x0, vl2
// CHECK-INST: uqdecd  x0, vl2
// CHECK-ENCODING: [0x40,0xfc,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04f0fc40 <unknown>

uqdecd  x0, vl3
// CHECK-INST: uqdecd  x0, vl3
// CHECK-ENCODING: [0x60,0xfc,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04f0fc60 <unknown>

uqdecd  x0, vl4
// CHECK-INST: uqdecd  x0, vl4
// CHECK-ENCODING: [0x80,0xfc,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04f0fc80 <unknown>

uqdecd  x0, vl5
// CHECK-INST: uqdecd  x0, vl5
// CHECK-ENCODING: [0xa0,0xfc,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04f0fca0 <unknown>

uqdecd  x0, vl6
// CHECK-INST: uqdecd  x0, vl6
// CHECK-ENCODING: [0xc0,0xfc,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04f0fcc0 <unknown>

uqdecd  x0, vl7
// CHECK-INST: uqdecd  x0, vl7
// CHECK-ENCODING: [0xe0,0xfc,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04f0fce0 <unknown>

uqdecd  x0, vl8
// CHECK-INST: uqdecd  x0, vl8
// CHECK-ENCODING: [0x00,0xfd,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04f0fd00 <unknown>

uqdecd  x0, vl16
// CHECK-INST: uqdecd  x0, vl16
// CHECK-ENCODING: [0x20,0xfd,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04f0fd20 <unknown>

uqdecd  x0, vl32
// CHECK-INST: uqdecd  x0, vl32
// CHECK-ENCODING: [0x40,0xfd,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04f0fd40 <unknown>

uqdecd  x0, vl64
// CHECK-INST: uqdecd  x0, vl64
// CHECK-ENCODING: [0x60,0xfd,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04f0fd60 <unknown>

uqdecd  x0, vl128
// CHECK-INST: uqdecd  x0, vl128
// CHECK-ENCODING: [0x80,0xfd,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04f0fd80 <unknown>

uqdecd  x0, vl256
// CHECK-INST: uqdecd  x0, vl256
// CHECK-ENCODING: [0xa0,0xfd,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04f0fda0 <unknown>

uqdecd  x0, #14
// CHECK-INST: uqdecd  x0, #14
// CHECK-ENCODING: [0xc0,0xfd,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04f0fdc0 <unknown>

uqdecd  x0, #15
// CHECK-INST: uqdecd  x0, #15
// CHECK-ENCODING: [0xe0,0xfd,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04f0fde0 <unknown>

uqdecd  x0, #16
// CHECK-INST: uqdecd  x0, #16
// CHECK-ENCODING: [0x00,0xfe,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04f0fe00 <unknown>

uqdecd  x0, #17
// CHECK-INST: uqdecd  x0, #17
// CHECK-ENCODING: [0x20,0xfe,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04f0fe20 <unknown>

uqdecd  x0, #18
// CHECK-INST: uqdecd  x0, #18
// CHECK-ENCODING: [0x40,0xfe,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04f0fe40 <unknown>

uqdecd  x0, #19
// CHECK-INST: uqdecd  x0, #19
// CHECK-ENCODING: [0x60,0xfe,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04f0fe60 <unknown>

uqdecd  x0, #20
// CHECK-INST: uqdecd  x0, #20
// CHECK-ENCODING: [0x80,0xfe,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04f0fe80 <unknown>

uqdecd  x0, #21
// CHECK-INST: uqdecd  x0, #21
// CHECK-ENCODING: [0xa0,0xfe,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04f0fea0 <unknown>

uqdecd  x0, #22
// CHECK-INST: uqdecd  x0, #22
// CHECK-ENCODING: [0xc0,0xfe,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04f0fec0 <unknown>

uqdecd  x0, #23
// CHECK-INST: uqdecd  x0, #23
// CHECK-ENCODING: [0xe0,0xfe,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04f0fee0 <unknown>

uqdecd  x0, #24
// CHECK-INST: uqdecd  x0, #24
// CHECK-ENCODING: [0x00,0xff,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04f0ff00 <unknown>

uqdecd  x0, #25
// CHECK-INST: uqdecd  x0, #25
// CHECK-ENCODING: [0x20,0xff,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04f0ff20 <unknown>

uqdecd  x0, #26
// CHECK-INST: uqdecd  x0, #26
// CHECK-ENCODING: [0x40,0xff,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04f0ff40 <unknown>

uqdecd  x0, #27
// CHECK-INST: uqdecd  x0, #27
// CHECK-ENCODING: [0x60,0xff,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04f0ff60 <unknown>

uqdecd  x0, #28
// CHECK-INST: uqdecd  x0, #28
// CHECK-ENCODING: [0x80,0xff,0xf0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04f0ff80 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bce0 <unknown>

uqdecd  z0.d
// CHECK-INST: uqdecd	z0.d
// CHECK-ENCODING: [0xe0,0xcf,0xe0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04e0cfe0 <unknown>

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bce0 <unknown>

uqdecd  z0.d, pow2, mul #16
// CHECK-INST: uqdecd	z0.d, pow2, mul #16
// CHECK-ENCODING: [0x00,0xcc,0xef,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04efcc00 <unknown>

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bce0 <unknown>

uqdecd  z0.d, pow2
// CHECK-INST: uqdecd	z0.d, pow2
// CHECK-ENCODING: [0x00,0xcc,0xe0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04e0cc00 <unknown>
