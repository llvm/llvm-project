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

clastb   w0, p7, w0, z31.b
// CHECK-INST: clastb	w0, p7, w0, z31.b
// CHECK-ENCODING: [0xe0,0xbf,0x31,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0531bfe0 <unknown>

clastb   w0, p7, w0, z31.h
// CHECK-INST: clastb	w0, p7, w0, z31.h
// CHECK-ENCODING: [0xe0,0xbf,0x71,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0571bfe0 <unknown>

clastb   w0, p7, w0, z31.s
// CHECK-INST: clastb	w0, p7, w0, z31.s
// CHECK-ENCODING: [0xe0,0xbf,0xb1,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05b1bfe0 <unknown>

clastb   x0, p7, x0, z31.d
// CHECK-INST: clastb	x0, p7, x0, z31.d
// CHECK-ENCODING: [0xe0,0xbf,0xf1,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05f1bfe0 <unknown>

clastb   b0, p7, b0, z31.b
// CHECK-INST: clastb	b0, p7, b0, z31.b
// CHECK-ENCODING: [0xe0,0x9f,0x2b,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 052b9fe0 <unknown>

clastb   h0, p7, h0, z31.h
// CHECK-INST: clastb	h0, p7, h0, z31.h
// CHECK-ENCODING: [0xe0,0x9f,0x6b,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 056b9fe0 <unknown>

clastb   s0, p7, s0, z31.s
// CHECK-INST: clastb	s0, p7, s0, z31.s
// CHECK-ENCODING: [0xe0,0x9f,0xab,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05ab9fe0 <unknown>

clastb   d0, p7, d0, z31.d
// CHECK-INST: clastb	d0, p7, d0, z31.d
// CHECK-ENCODING: [0xe0,0x9f,0xeb,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05eb9fe0 <unknown>

clastb   z0.b, p7, z0.b, z31.b
// CHECK-INST: clastb	z0.b, p7, z0.b, z31.b
// CHECK-ENCODING: [0xe0,0x9f,0x29,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05299fe0 <unknown>

clastb   z0.h, p7, z0.h, z31.h
// CHECK-INST: clastb	z0.h, p7, z0.h, z31.h
// CHECK-ENCODING: [0xe0,0x9f,0x69,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05699fe0 <unknown>

clastb   z0.s, p7, z0.s, z31.s
// CHECK-INST: clastb	z0.s, p7, z0.s, z31.s
// CHECK-ENCODING: [0xe0,0x9f,0xa9,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05a99fe0 <unknown>

clastb   z0.d, p7, z0.d, z31.d
// CHECK-INST: clastb	z0.d, p7, z0.d, z31.d
// CHECK-ENCODING: [0xe0,0x9f,0xe9,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05e99fe0 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bce0 <unknown>

clastb   z0.d, p7, z0.d, z31.d
// CHECK-INST: clastb	z0.d, p7, z0.d, z31.d
// CHECK-ENCODING: [0xe0,0x9f,0xe9,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 05e99fe0 <unknown>
