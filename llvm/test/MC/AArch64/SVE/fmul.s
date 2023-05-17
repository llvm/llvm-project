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

fmul    z0.h, p0/m, z0.h, #0.5000000000000
// CHECK-INST: fmul    z0.h, p0/m, z0.h, #0.5
// CHECK-ENCODING: [0x00,0x80,0x5a,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 655a8000 <unknown>

fmul    z0.h, p0/m, z0.h, #0.5
// CHECK-INST: fmul    z0.h, p0/m, z0.h, #0.5
// CHECK-ENCODING: [0x00,0x80,0x5a,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 655a8000 <unknown>

fmul    z0.s, p0/m, z0.s, #0.5
// CHECK-INST: fmul    z0.s, p0/m, z0.s, #0.5
// CHECK-ENCODING: [0x00,0x80,0x9a,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 659a8000 <unknown>

fmul    z0.d, p0/m, z0.d, #0.5
// CHECK-INST: fmul    z0.d, p0/m, z0.d, #0.5
// CHECK-ENCODING: [0x00,0x80,0xda,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 65da8000 <unknown>

fmul    z31.h, p7/m, z31.h, #2.0
// CHECK-INST: fmul    z31.h, p7/m, z31.h, #2.0
// CHECK-ENCODING: [0x3f,0x9c,0x5a,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 655a9c3f <unknown>

fmul    z31.s, p7/m, z31.s, #2.0
// CHECK-INST: fmul    z31.s, p7/m, z31.s, #2.0
// CHECK-ENCODING: [0x3f,0x9c,0x9a,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 659a9c3f <unknown>

fmul    z31.d, p7/m, z31.d, #2.0
// CHECK-INST: fmul    z31.d, p7/m, z31.d, #2.0
// CHECK-ENCODING: [0x3f,0x9c,0xda,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 65da9c3f <unknown>

fmul    z0.h, z0.h, z0.h[0]
// CHECK-INST: fmul    z0.h, z0.h, z0.h[0]
// CHECK-ENCODING: [0x00,0x20,0x20,0x64]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 64202000 <unknown>

fmul    z0.s, z0.s, z0.s[0]
// CHECK-INST: fmul    z0.s, z0.s, z0.s[0]
// CHECK-ENCODING: [0x00,0x20,0xa0,0x64]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 64a02000 <unknown>

fmul    z0.d, z0.d, z0.d[0]
// CHECK-INST: fmul    z0.d, z0.d, z0.d[0]
// CHECK-ENCODING: [0x00,0x20,0xe0,0x64]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 64e02000 <unknown>

fmul    z31.h, z31.h, z7.h[7]
// CHECK-INST: fmul    z31.h, z31.h, z7.h[7]
// CHECK-ENCODING: [0xff,0x23,0x7f,0x64]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 647f23ff <unknown>

fmul    z31.s, z31.s, z7.s[3]
// CHECK-INST: fmul    z31.s, z31.s, z7.s[3]
// CHECK-ENCODING: [0xff,0x23,0xbf,0x64]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 64bf23ff <unknown>

fmul    z31.d, z31.d, z15.d[1]
// CHECK-INST: fmul    z31.d, z31.d, z15.d[1]
// CHECK-ENCODING: [0xff,0x23,0xff,0x64]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 64ff23ff <unknown>

fmul    z0.h, p7/m, z0.h, z31.h
// CHECK-INST: fmul	z0.h, p7/m, z0.h, z31.h
// CHECK-ENCODING: [0xe0,0x9f,0x42,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 65429fe0 <unknown>

fmul    z0.s, p7/m, z0.s, z31.s
// CHECK-INST: fmul	z0.s, p7/m, z0.s, z31.s
// CHECK-ENCODING: [0xe0,0x9f,0x82,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 65829fe0 <unknown>

fmul    z0.d, p7/m, z0.d, z31.d
// CHECK-INST: fmul	z0.d, p7/m, z0.d, z31.d
// CHECK-ENCODING: [0xe0,0x9f,0xc2,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 65c29fe0 <unknown>

fmul z0.h, z1.h, z31.h
// CHECK-INST: fmul	z0.h, z1.h, z31.h
// CHECK-ENCODING: [0x20,0x08,0x5f,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 655f0820 <unknown>

fmul z0.s, z1.s, z31.s
// CHECK-INST: fmul	z0.s, z1.s, z31.s
// CHECK-ENCODING: [0x20,0x08,0x9f,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 659f0820 <unknown>

fmul z0.d, z1.d, z31.d
// CHECK-INST: fmul	z0.d, z1.d, z31.d
// CHECK-ENCODING: [0x20,0x08,0xdf,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 65df0820 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z31.d, p7/z, z6.d
// CHECK-INST: movprfx	z31.d, p7/z, z6.d
// CHECK-ENCODING: [0xdf,0x3c,0xd0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04d03cdf <unknown>

fmul    z31.d, p7/m, z31.d, #2.0
// CHECK-INST: fmul	z31.d, p7/m, z31.d, #2.0
// CHECK-ENCODING: [0x3f,0x9c,0xda,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 65da9c3f <unknown>

movprfx z31, z6
// CHECK-INST: movprfx	z31, z6
// CHECK-ENCODING: [0xdf,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bcdf <unknown>

fmul    z31.d, p7/m, z31.d, #2.0
// CHECK-INST: fmul	z31.d, p7/m, z31.d, #2.0
// CHECK-ENCODING: [0x3f,0x9c,0xda,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 65da9c3f <unknown>

movprfx z0.d, p7/z, z7.d
// CHECK-INST: movprfx	z0.d, p7/z, z7.d
// CHECK-ENCODING: [0xe0,0x3c,0xd0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04d03ce0 <unknown>

fmul    z0.d, p7/m, z0.d, z31.d
// CHECK-INST: fmul	z0.d, p7/m, z0.d, z31.d
// CHECK-ENCODING: [0xe0,0x9f,0xc2,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 65c29fe0 <unknown>

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bce0 <unknown>

fmul    z0.d, p7/m, z0.d, z31.d
// CHECK-INST: fmul	z0.d, p7/m, z0.d, z31.d
// CHECK-ENCODING: [0xe0,0x9f,0xc2,0x65]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 65c29fe0 <unknown>
