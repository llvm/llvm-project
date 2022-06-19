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
uqdech  x0
// CHECK-INST: uqdech  x0
// CHECK-ENCODING: [0xe0,0xff,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 ff 70 04 <unknown>

uqdech  x0, all
// CHECK-INST: uqdech  x0
// CHECK-ENCODING: [0xe0,0xff,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 ff 70 04 <unknown>

uqdech  x0, all, mul #1
// CHECK-INST: uqdech  x0
// CHECK-ENCODING: [0xe0,0xff,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 ff 70 04 <unknown>

uqdech  x0, all, mul #16
// CHECK-INST: uqdech  x0, all, mul #16
// CHECK-ENCODING: [0xe0,0xff,0x7f,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 ff 7f 04 <unknown>


// ---------------------------------------------------------------------------//
// Test 32-bit form (w0) and its aliases
// ---------------------------------------------------------------------------//

uqdech  w0
// CHECK-INST: uqdech  w0
// CHECK-ENCODING: [0xe0,0xff,0x60,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 ff 60 04 <unknown>

uqdech  w0, all
// CHECK-INST: uqdech  w0
// CHECK-ENCODING: [0xe0,0xff,0x60,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 ff 60 04 <unknown>

uqdech  w0, all, mul #1
// CHECK-INST: uqdech  w0
// CHECK-ENCODING: [0xe0,0xff,0x60,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 ff 60 04 <unknown>

uqdech  w0, all, mul #16
// CHECK-INST: uqdech  w0, all, mul #16
// CHECK-ENCODING: [0xe0,0xff,0x6f,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 ff 6f 04 <unknown>

uqdech  w0, pow2
// CHECK-INST: uqdech  w0, pow2
// CHECK-ENCODING: [0x00,0xfc,0x60,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 fc 60 04 <unknown>

uqdech  w0, pow2, mul #16
// CHECK-INST: uqdech  w0, pow2, mul #16
// CHECK-ENCODING: [0x00,0xfc,0x6f,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 fc 6f 04 <unknown>


// ---------------------------------------------------------------------------//
// Test vector form and aliases.
// ---------------------------------------------------------------------------//
uqdech  z0.h
// CHECK-INST: uqdech  z0.h
// CHECK-ENCODING: [0xe0,0xcf,0x60,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 cf 60 04 <unknown>

uqdech  z0.h, all
// CHECK-INST: uqdech  z0.h
// CHECK-ENCODING: [0xe0,0xcf,0x60,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 cf 60 04 <unknown>

uqdech  z0.h, all, mul #1
// CHECK-INST: uqdech  z0.h
// CHECK-ENCODING: [0xe0,0xcf,0x60,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 cf 60 04 <unknown>

uqdech  z0.h, all, mul #16
// CHECK-INST: uqdech  z0.h, all, mul #16
// CHECK-ENCODING: [0xe0,0xcf,0x6f,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 cf 6f 04 <unknown>

uqdech  z0.h, pow2
// CHECK-INST: uqdech  z0.h, pow2
// CHECK-ENCODING: [0x00,0xcc,0x60,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 cc 60 04 <unknown>

uqdech  z0.h, pow2, mul #16
// CHECK-INST: uqdech  z0.h, pow2, mul #16
// CHECK-ENCODING: [0x00,0xcc,0x6f,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 cc 6f 04 <unknown>


// ---------------------------------------------------------------------------//
// Test all patterns for 64-bit form
// ---------------------------------------------------------------------------//

uqdech  x0, pow2
// CHECK-INST: uqdech  x0, pow2
// CHECK-ENCODING: [0x00,0xfc,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 fc 70 04 <unknown>

uqdech  x0, vl1
// CHECK-INST: uqdech  x0, vl1
// CHECK-ENCODING: [0x20,0xfc,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 fc 70 04 <unknown>

uqdech  x0, vl2
// CHECK-INST: uqdech  x0, vl2
// CHECK-ENCODING: [0x40,0xfc,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 40 fc 70 04 <unknown>

uqdech  x0, vl3
// CHECK-INST: uqdech  x0, vl3
// CHECK-ENCODING: [0x60,0xfc,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 60 fc 70 04 <unknown>

uqdech  x0, vl4
// CHECK-INST: uqdech  x0, vl4
// CHECK-ENCODING: [0x80,0xfc,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 80 fc 70 04 <unknown>

uqdech  x0, vl5
// CHECK-INST: uqdech  x0, vl5
// CHECK-ENCODING: [0xa0,0xfc,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a0 fc 70 04 <unknown>

uqdech  x0, vl6
// CHECK-INST: uqdech  x0, vl6
// CHECK-ENCODING: [0xc0,0xfc,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: c0 fc 70 04 <unknown>

uqdech  x0, vl7
// CHECK-INST: uqdech  x0, vl7
// CHECK-ENCODING: [0xe0,0xfc,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 fc 70 04 <unknown>

uqdech  x0, vl8
// CHECK-INST: uqdech  x0, vl8
// CHECK-ENCODING: [0x00,0xfd,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 fd 70 04 <unknown>

uqdech  x0, vl16
// CHECK-INST: uqdech  x0, vl16
// CHECK-ENCODING: [0x20,0xfd,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 fd 70 04 <unknown>

uqdech  x0, vl32
// CHECK-INST: uqdech  x0, vl32
// CHECK-ENCODING: [0x40,0xfd,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 40 fd 70 04 <unknown>

uqdech  x0, vl64
// CHECK-INST: uqdech  x0, vl64
// CHECK-ENCODING: [0x60,0xfd,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 60 fd 70 04 <unknown>

uqdech  x0, vl128
// CHECK-INST: uqdech  x0, vl128
// CHECK-ENCODING: [0x80,0xfd,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 80 fd 70 04 <unknown>

uqdech  x0, vl256
// CHECK-INST: uqdech  x0, vl256
// CHECK-ENCODING: [0xa0,0xfd,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a0 fd 70 04 <unknown>

uqdech  x0, #14
// CHECK-INST: uqdech  x0, #14
// CHECK-ENCODING: [0xc0,0xfd,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: c0 fd 70 04 <unknown>

uqdech  x0, #15
// CHECK-INST: uqdech  x0, #15
// CHECK-ENCODING: [0xe0,0xfd,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 fd 70 04 <unknown>

uqdech  x0, #16
// CHECK-INST: uqdech  x0, #16
// CHECK-ENCODING: [0x00,0xfe,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 fe 70 04 <unknown>

uqdech  x0, #17
// CHECK-INST: uqdech  x0, #17
// CHECK-ENCODING: [0x20,0xfe,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 fe 70 04 <unknown>

uqdech  x0, #18
// CHECK-INST: uqdech  x0, #18
// CHECK-ENCODING: [0x40,0xfe,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 40 fe 70 04 <unknown>

uqdech  x0, #19
// CHECK-INST: uqdech  x0, #19
// CHECK-ENCODING: [0x60,0xfe,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 60 fe 70 04 <unknown>

uqdech  x0, #20
// CHECK-INST: uqdech  x0, #20
// CHECK-ENCODING: [0x80,0xfe,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 80 fe 70 04 <unknown>

uqdech  x0, #21
// CHECK-INST: uqdech  x0, #21
// CHECK-ENCODING: [0xa0,0xfe,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: a0 fe 70 04 <unknown>

uqdech  x0, #22
// CHECK-INST: uqdech  x0, #22
// CHECK-ENCODING: [0xc0,0xfe,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: c0 fe 70 04 <unknown>

uqdech  x0, #23
// CHECK-INST: uqdech  x0, #23
// CHECK-ENCODING: [0xe0,0xfe,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 fe 70 04 <unknown>

uqdech  x0, #24
// CHECK-INST: uqdech  x0, #24
// CHECK-ENCODING: [0x00,0xff,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 ff 70 04 <unknown>

uqdech  x0, #25
// CHECK-INST: uqdech  x0, #25
// CHECK-ENCODING: [0x20,0xff,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 20 ff 70 04 <unknown>

uqdech  x0, #26
// CHECK-INST: uqdech  x0, #26
// CHECK-ENCODING: [0x40,0xff,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 40 ff 70 04 <unknown>

uqdech  x0, #27
// CHECK-INST: uqdech  x0, #27
// CHECK-ENCODING: [0x60,0xff,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 60 ff 70 04 <unknown>

uqdech  x0, #28
// CHECK-INST: uqdech  x0, #28
// CHECK-ENCODING: [0x80,0xff,0x70,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 80 ff 70 04 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 bc 20 04 <unknown>

uqdech  z0.h
// CHECK-INST: uqdech	z0.h
// CHECK-ENCODING: [0xe0,0xcf,0x60,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 cf 60 04 <unknown>

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 bc 20 04 <unknown>

uqdech  z0.h, pow2, mul #16
// CHECK-INST: uqdech	z0.h, pow2, mul #16
// CHECK-ENCODING: [0x00,0xcc,0x6f,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 cc 6f 04 <unknown>

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 bc 20 04 <unknown>

uqdech  z0.h, pow2
// CHECK-INST: uqdech	z0.h, pow2
// CHECK-ENCODING: [0x00,0xcc,0x60,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 cc 60 04 <unknown>
