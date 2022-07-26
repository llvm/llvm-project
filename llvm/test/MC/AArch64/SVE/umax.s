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

umax    z0.b, z0.b, #0
// CHECK-INST: umax	z0.b, z0.b, #0
// CHECK-ENCODING: [0x00,0xc0,0x29,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2529c000 <unknown>

umax    z31.b, z31.b, #255
// CHECK-INST: umax	z31.b, z31.b, #255
// CHECK-ENCODING: [0xff,0xdf,0x29,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2529dfff <unknown>

umax    z0.b, z0.b, #0
// CHECK-INST: umax	z0.b, z0.b, #0
// CHECK-ENCODING: [0x00,0xc0,0x29,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2529c000 <unknown>

umax    z31.b, z31.b, #255
// CHECK-INST: umax	z31.b, z31.b, #255
// CHECK-ENCODING: [0xff,0xdf,0x29,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2529dfff <unknown>

umax    z0.b, z0.b, #0
// CHECK-INST: umax	z0.b, z0.b, #0
// CHECK-ENCODING: [0x00,0xc0,0x29,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2529c000 <unknown>

umax    z31.b, z31.b, #255
// CHECK-INST: umax	z31.b, z31.b, #255
// CHECK-ENCODING: [0xff,0xdf,0x29,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2529dfff <unknown>

umax    z0.b, z0.b, #0
// CHECK-INST: umax	z0.b, z0.b, #0
// CHECK-ENCODING: [0x00,0xc0,0x29,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2529c000 <unknown>

umax    z31.b, z31.b, #255
// CHECK-INST: umax	z31.b, z31.b, #255
// CHECK-ENCODING: [0xff,0xdf,0x29,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2529dfff <unknown>

umax    z31.b, p7/m, z31.b, z31.b
// CHECK-INST: umax    z31.b, p7/m, z31.b, z31.b
// CHECK-ENCODING: [0xff,0x1f,0x09,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04091fff <unknown>

umax    z31.h, p7/m, z31.h, z31.h
// CHECK-INST: umax    z31.h, p7/m, z31.h, z31.h
// CHECK-ENCODING: [0xff,0x1f,0x49,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04491fff <unknown>

umax    z31.s, p7/m, z31.s, z31.s
// CHECK-INST: umax    z31.s, p7/m, z31.s, z31.s
// CHECK-ENCODING: [0xff,0x1f,0x89,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04891fff <unknown>

umax    z31.d, p7/m, z31.d, z31.d
// CHECK-INST: umax    z31.d, p7/m, z31.d, z31.d
// CHECK-ENCODING: [0xff,0x1f,0xc9,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04c91fff <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z4.d, p7/z, z6.d
// CHECK-INST: movprfx	z4.d, p7/z, z6.d
// CHECK-ENCODING: [0xc4,0x3c,0xd0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04d03cc4 <unknown>

umax    z4.d, p7/m, z4.d, z31.d
// CHECK-INST: umax	z4.d, p7/m, z4.d, z31.d
// CHECK-ENCODING: [0xe4,0x1f,0xc9,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04c91fe4 <unknown>

movprfx z4, z6
// CHECK-INST: movprfx	z4, z6
// CHECK-ENCODING: [0xc4,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bcc4 <unknown>

umax    z4.d, p7/m, z4.d, z31.d
// CHECK-INST: umax	z4.d, p7/m, z4.d, z31.d
// CHECK-ENCODING: [0xe4,0x1f,0xc9,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04c91fe4 <unknown>

movprfx z31, z6
// CHECK-INST: movprfx	z31, z6
// CHECK-ENCODING: [0xdf,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bcdf <unknown>

umax    z31.b, z31.b, #255
// CHECK-INST: umax	z31.b, z31.b, #255
// CHECK-ENCODING: [0xff,0xdf,0x29,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2529dfff <unknown>
