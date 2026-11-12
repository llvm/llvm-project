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

sqdech  x0
// CHECK-INST: sqdech  x0
// CHECK-ENCODING: [0xe0,0xfb,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470fbe0 <unknown>

sqdech  x0, all
// CHECK-INST: sqdech  x0
// CHECK-ENCODING: [0xe0,0xfb,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470fbe0 <unknown>

sqdech  x0, all, mul #1
// CHECK-INST: sqdech  x0
// CHECK-ENCODING: [0xe0,0xfb,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470fbe0 <unknown>

sqdech  x0, all, mul #16
// CHECK-INST: sqdech  x0, all, mul #16
// CHECK-ENCODING: [0xe0,0xfb,0x7f,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 047ffbe0 <unknown>


// ---------------------------------------------------------------------------//
// Test 32-bit form (x0, w0) and its aliases
// ---------------------------------------------------------------------------//

sqdech  x0, w0
// CHECK-INST: sqdech  x0, w0
// CHECK-ENCODING: [0xe0,0xfb,0x60,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0460fbe0 <unknown>

sqdech  x0, w0, all
// CHECK-INST: sqdech  x0, w0
// CHECK-ENCODING: [0xe0,0xfb,0x60,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0460fbe0 <unknown>

sqdech  x0, w0, all, mul #1
// CHECK-INST: sqdech  x0, w0
// CHECK-ENCODING: [0xe0,0xfb,0x60,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0460fbe0 <unknown>

sqdech  x0, w0, all, mul #16
// CHECK-INST: sqdech  x0, w0, all, mul #16
// CHECK-ENCODING: [0xe0,0xfb,0x6f,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 046ffbe0 <unknown>

sqdech  x0, w0, pow2
// CHECK-INST: sqdech  x0, w0, pow2
// CHECK-ENCODING: [0x00,0xf8,0x60,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0460f800 <unknown>

sqdech  x0, w0, pow2, mul #16
// CHECK-INST: sqdech  x0, w0, pow2, mul #16
// CHECK-ENCODING: [0x00,0xf8,0x6f,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 046ff800 <unknown>


// ---------------------------------------------------------------------------//
// Test vector form and aliases.
// ---------------------------------------------------------------------------//
sqdech  z0.h
// CHECK-INST: sqdech  z0.h
// CHECK-ENCODING: [0xe0,0xcb,0x60,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0460cbe0 <unknown>

sqdech  z0.h, all
// CHECK-INST: sqdech  z0.h
// CHECK-ENCODING: [0xe0,0xcb,0x60,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0460cbe0 <unknown>

sqdech  z0.h, all, mul #1
// CHECK-INST: sqdech  z0.h
// CHECK-ENCODING: [0xe0,0xcb,0x60,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0460cbe0 <unknown>

sqdech  z0.h, all, mul #16
// CHECK-INST: sqdech  z0.h, all, mul #16
// CHECK-ENCODING: [0xe0,0xcb,0x6f,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 046fcbe0 <unknown>

sqdech  z0.h, pow2
// CHECK-INST: sqdech  z0.h, pow2
// CHECK-ENCODING: [0x00,0xc8,0x60,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0460c800 <unknown>

sqdech  z0.h, pow2, mul #16
// CHECK-INST: sqdech  z0.h, pow2, mul #16
// CHECK-ENCODING: [0x00,0xc8,0x6f,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 046fc800 <unknown>


// ---------------------------------------------------------------------------//
// Test all patterns for 64-bit form
// ---------------------------------------------------------------------------//

sqdech  x0, pow2
// CHECK-INST: sqdech  x0, pow2
// CHECK-ENCODING: [0x00,0xf8,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470f800 <unknown>

sqdech  x0, vl1
// CHECK-INST: sqdech  x0, vl1
// CHECK-ENCODING: [0x20,0xf8,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470f820 <unknown>

sqdech  x0, vl2
// CHECK-INST: sqdech  x0, vl2
// CHECK-ENCODING: [0x40,0xf8,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470f840 <unknown>

sqdech  x0, vl3
// CHECK-INST: sqdech  x0, vl3
// CHECK-ENCODING: [0x60,0xf8,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470f860 <unknown>

sqdech  x0, vl4
// CHECK-INST: sqdech  x0, vl4
// CHECK-ENCODING: [0x80,0xf8,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470f880 <unknown>

sqdech  x0, vl5
// CHECK-INST: sqdech  x0, vl5
// CHECK-ENCODING: [0xa0,0xf8,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470f8a0 <unknown>

sqdech  x0, vl6
// CHECK-INST: sqdech  x0, vl6
// CHECK-ENCODING: [0xc0,0xf8,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470f8c0 <unknown>

sqdech  x0, vl7
// CHECK-INST: sqdech  x0, vl7
// CHECK-ENCODING: [0xe0,0xf8,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470f8e0 <unknown>

sqdech  x0, vl8
// CHECK-INST: sqdech  x0, vl8
// CHECK-ENCODING: [0x00,0xf9,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470f900 <unknown>

sqdech  x0, vl16
// CHECK-INST: sqdech  x0, vl16
// CHECK-ENCODING: [0x20,0xf9,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470f920 <unknown>

sqdech  x0, vl32
// CHECK-INST: sqdech  x0, vl32
// CHECK-ENCODING: [0x40,0xf9,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470f940 <unknown>

sqdech  x0, vl64
// CHECK-INST: sqdech  x0, vl64
// CHECK-ENCODING: [0x60,0xf9,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470f960 <unknown>

sqdech  x0, vl128
// CHECK-INST: sqdech  x0, vl128
// CHECK-ENCODING: [0x80,0xf9,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470f980 <unknown>

sqdech  x0, vl256
// CHECK-INST: sqdech  x0, vl256
// CHECK-ENCODING: [0xa0,0xf9,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470f9a0 <unknown>

sqdech  x0, #14
// CHECK-INST: sqdech  x0, #14
// CHECK-ENCODING: [0xc0,0xf9,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470f9c0 <unknown>

sqdech  x0, #15
// CHECK-INST: sqdech  x0, #15
// CHECK-ENCODING: [0xe0,0xf9,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470f9e0 <unknown>

sqdech  x0, #16
// CHECK-INST: sqdech  x0, #16
// CHECK-ENCODING: [0x00,0xfa,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470fa00 <unknown>

sqdech  x0, #17
// CHECK-INST: sqdech  x0, #17
// CHECK-ENCODING: [0x20,0xfa,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470fa20 <unknown>

sqdech  x0, #18
// CHECK-INST: sqdech  x0, #18
// CHECK-ENCODING: [0x40,0xfa,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470fa40 <unknown>

sqdech  x0, #19
// CHECK-INST: sqdech  x0, #19
// CHECK-ENCODING: [0x60,0xfa,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470fa60 <unknown>

sqdech  x0, #20
// CHECK-INST: sqdech  x0, #20
// CHECK-ENCODING: [0x80,0xfa,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470fa80 <unknown>

sqdech  x0, #21
// CHECK-INST: sqdech  x0, #21
// CHECK-ENCODING: [0xa0,0xfa,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470faa0 <unknown>

sqdech  x0, #22
// CHECK-INST: sqdech  x0, #22
// CHECK-ENCODING: [0xc0,0xfa,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470fac0 <unknown>

sqdech  x0, #23
// CHECK-INST: sqdech  x0, #23
// CHECK-ENCODING: [0xe0,0xfa,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470fae0 <unknown>

sqdech  x0, #24
// CHECK-INST: sqdech  x0, #24
// CHECK-ENCODING: [0x00,0xfb,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470fb00 <unknown>

sqdech  x0, #25
// CHECK-INST: sqdech  x0, #25
// CHECK-ENCODING: [0x20,0xfb,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470fb20 <unknown>

sqdech  x0, #26
// CHECK-INST: sqdech  x0, #26
// CHECK-ENCODING: [0x40,0xfb,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470fb40 <unknown>

sqdech  x0, #27
// CHECK-INST: sqdech  x0, #27
// CHECK-ENCODING: [0x60,0xfb,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470fb60 <unknown>

sqdech  x0, #28
// CHECK-INST: sqdech  x0, #28
// CHECK-ENCODING: [0x80,0xfb,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0470fb80 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bce0 <unknown>

sqdech  z0.h
// CHECK-INST: sqdech	z0.h
// CHECK-ENCODING: [0xe0,0xcb,0x60,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0460cbe0 <unknown>

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bce0 <unknown>

sqdech  z0.h, pow2, mul #16
// CHECK-INST: sqdech	z0.h, pow2, mul #16
// CHECK-ENCODING: [0x00,0xc8,0x6f,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 046fc800 <unknown>

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bce0 <unknown>

sqdech  z0.h, pow2
// CHECK-INST: sqdech	z0.h, pow2
// CHECK-ENCODING: [0x00,0xc8,0x60,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0460c800 <unknown>
