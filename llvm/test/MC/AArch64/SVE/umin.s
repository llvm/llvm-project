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

umin    z0.b, z0.b, #0
// CHECK-INST: umin	z0.b, z0.b, #0
// CHECK-ENCODING: [0x00,0xc0,0x2b,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 c0 2b 25 <unknown>

umin    z31.b, z31.b, #255
// CHECK-INST: umin	z31.b, z31.b, #255
// CHECK-ENCODING: [0xff,0xdf,0x2b,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff df 2b 25 <unknown>

umin    z0.b, z0.b, #0
// CHECK-INST: umin	z0.b, z0.b, #0
// CHECK-ENCODING: [0x00,0xc0,0x2b,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 c0 2b 25 <unknown>

umin    z31.b, z31.b, #255
// CHECK-INST: umin	z31.b, z31.b, #255
// CHECK-ENCODING: [0xff,0xdf,0x2b,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff df 2b 25 <unknown>

umin    z0.b, z0.b, #0
// CHECK-INST: umin	z0.b, z0.b, #0
// CHECK-ENCODING: [0x00,0xc0,0x2b,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 c0 2b 25 <unknown>

umin    z31.b, z31.b, #255
// CHECK-INST: umin	z31.b, z31.b, #255
// CHECK-ENCODING: [0xff,0xdf,0x2b,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff df 2b 25 <unknown>

umin    z0.b, z0.b, #0
// CHECK-INST: umin	z0.b, z0.b, #0
// CHECK-ENCODING: [0x00,0xc0,0x2b,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 c0 2b 25 <unknown>

umin    z31.b, z31.b, #255
// CHECK-INST: umin	z31.b, z31.b, #255
// CHECK-ENCODING: [0xff,0xdf,0x2b,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff df 2b 25 <unknown>

umin    z31.b, p7/m, z31.b, z31.b
// CHECK-INST: umin	z31.b, p7/m, z31.b, z31.b
// CHECK-ENCODING: [0xff,0x1f,0x0b,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 1f 0b 04 <unknown>

umin    z31.h, p7/m, z31.h, z31.h
// CHECK-INST: umin	z31.h, p7/m, z31.h, z31.h
// CHECK-ENCODING: [0xff,0x1f,0x4b,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 1f 4b 04 <unknown>

umin    z31.s, p7/m, z31.s, z31.s
// CHECK-INST: umin	z31.s, p7/m, z31.s, z31.s
// CHECK-ENCODING: [0xff,0x1f,0x8b,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 1f 8b 04 <unknown>

umin    z31.d, p7/m, z31.d, z31.d
// CHECK-INST: umin	z31.d, p7/m, z31.d, z31.d
// CHECK-ENCODING: [0xff,0x1f,0xcb,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 1f cb 04 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z4.d, p7/z, z6.d
// CHECK-INST: movprfx	z4.d, p7/z, z6.d
// CHECK-ENCODING: [0xc4,0x3c,0xd0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: c4 3c d0 04 <unknown>

umin    z4.d, p7/m, z4.d, z31.d
// CHECK-INST: umin	z4.d, p7/m, z4.d, z31.d
// CHECK-ENCODING: [0xe4,0x1f,0xcb,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e4 1f cb 04 <unknown>

movprfx z4, z6
// CHECK-INST: movprfx	z4, z6
// CHECK-ENCODING: [0xc4,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: c4 bc 20 04 <unknown>

umin    z4.d, p7/m, z4.d, z31.d
// CHECK-INST: umin	z4.d, p7/m, z4.d, z31.d
// CHECK-ENCODING: [0xe4,0x1f,0xcb,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e4 1f cb 04 <unknown>

movprfx z31, z6
// CHECK-INST: movprfx	z31, z6
// CHECK-ENCODING: [0xdf,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: df bc 20 04 <unknown>

umin    z31.b, z31.b, #255
// CHECK-INST: umin	z31.b, z31.b, #255
// CHECK-ENCODING: [0xff,0xdf,0x2b,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff df 2b 25 <unknown>
