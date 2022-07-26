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


subr    z0.b, p0/m, z0.b, z0.b
// CHECK-INST: subr z0.b, p0/m, z0.b, z0.b
// CHECK-ENCODING: [0x00,0x00,0x03,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04030000 <unknown>

subr    z0.h, p0/m, z0.h, z0.h
// CHECK-INST: subr z0.h, p0/m, z0.h, z0.h
// CHECK-ENCODING: [0x00,0x00,0x43,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04430000 <unknown>

subr    z0.s, p0/m, z0.s, z0.s
// CHECK-INST: subr z0.s, p0/m, z0.s, z0.s
// CHECK-ENCODING: [0x00,0x00,0x83,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04830000 <unknown>

subr    z0.d, p0/m, z0.d, z0.d
// CHECK-INST: subr z0.d, p0/m, z0.d, z0.d
// CHECK-ENCODING: [0x00,0x00,0xc3,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04c30000 <unknown>

subr    z0.b, z0.b, #0
// CHECK-INST: subr z0.b, z0.b, #0
// CHECK-ENCODING: [0x00,0xc0,0x23,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2523c000 <unknown>

subr    z31.b, z31.b, #255
// CHECK-INST: subr z31.b, z31.b, #255
// CHECK-ENCODING: [0xff,0xdf,0x23,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2523dfff <unknown>

subr    z0.h, z0.h, #0
// CHECK-INST: subr z0.h, z0.h, #0
// CHECK-ENCODING: [0x00,0xc0,0x63,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2563c000 <unknown>

subr    z0.h, z0.h, #0, lsl #8
// CHECK-INST: subr z0.h, z0.h, #0, lsl #8
// CHECK-ENCODING: [0x00,0xe0,0x63,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2563e000 <unknown>

subr    z31.h, z31.h, #255, lsl #8
// CHECK-INST: subr z31.h, z31.h, #65280
// CHECK-ENCODING: [0xff,0xff,0x63,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2563ffff <unknown>

subr    z31.h, z31.h, #65280
// CHECK-INST: subr z31.h, z31.h, #65280
// CHECK-ENCODING: [0xff,0xff,0x63,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2563ffff <unknown>

subr    z0.s, z0.s, #0
// CHECK-INST: subr z0.s, z0.s, #0
// CHECK-ENCODING: [0x00,0xc0,0xa3,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25a3c000 <unknown>

subr    z0.s, z0.s, #0, lsl #8
// CHECK-INST: subr z0.s, z0.s, #0, lsl #8
// CHECK-ENCODING: [0x00,0xe0,0xa3,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25a3e000 <unknown>

subr    z31.s, z31.s, #255, lsl #8
// CHECK-INST: subr z31.s, z31.s, #65280
// CHECK-ENCODING: [0xff,0xff,0xa3,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25a3ffff <unknown>

subr    z31.s, z31.s, #65280
// CHECK-INST: subr z31.s, z31.s, #65280
// CHECK-ENCODING: [0xff,0xff,0xa3,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25a3ffff <unknown>

subr    z0.d, z0.d, #0
// CHECK-INST: subr z0.d, z0.d, #0
// CHECK-ENCODING: [0x00,0xc0,0xe3,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25e3c000 <unknown>

subr    z0.d, z0.d, #0, lsl #8
// CHECK-INST: subr z0.d, z0.d, #0, lsl #8
// CHECK-ENCODING: [0x00,0xe0,0xe3,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25e3e000 <unknown>

subr    z31.d, z31.d, #255, lsl #8
// CHECK-INST: subr z31.d, z31.d, #65280
// CHECK-ENCODING: [0xff,0xff,0xe3,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25e3ffff <unknown>

subr    z31.d, z31.d, #65280
// CHECK-INST: subr z31.d, z31.d, #65280
// CHECK-ENCODING: [0xff,0xff,0xe3,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25e3ffff <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z5.d, p0/z, z7.d
// CHECK-INST: movprfx	z5.d, p0/z, z7.d
// CHECK-ENCODING: [0xe5,0x20,0xd0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04d020e5 <unknown>

subr    z5.d, p0/m, z5.d, z0.d
// CHECK-INST: subr	z5.d, p0/m, z5.d, z0.d
// CHECK-ENCODING: [0x05,0x00,0xc3,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04c30005 <unknown>

movprfx z5, z7
// CHECK-INST: movprfx	z5, z7
// CHECK-ENCODING: [0xe5,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bce5 <unknown>

subr    z5.d, p0/m, z5.d, z0.d
// CHECK-INST: subr	z5.d, p0/m, z5.d, z0.d
// CHECK-ENCODING: [0x05,0x00,0xc3,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04c30005 <unknown>

movprfx z31, z6
// CHECK-INST: movprfx	z31, z6
// CHECK-ENCODING: [0xdf,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bcdf <unknown>

subr    z31.d, z31.d, #65280
// CHECK-INST: subr	z31.d, z31.d, #65280
// CHECK-ENCODING: [0xff,0xff,0xe3,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25e3ffff <unknown>
