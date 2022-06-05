// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

clasta   w0, p7, w0, z31.b
// CHECK-INST: clasta	w0, p7, w0, z31.b
// CHECK-ENCODING: [0xe0,0xbf,0x30,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 bf 30 05 <unknown>

clasta   w0, p7, w0, z31.h
// CHECK-INST: clasta	w0, p7, w0, z31.h
// CHECK-ENCODING: [0xe0,0xbf,0x70,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 bf 70 05 <unknown>

clasta   w0, p7, w0, z31.s
// CHECK-INST: clasta	w0, p7, w0, z31.s
// CHECK-ENCODING: [0xe0,0xbf,0xb0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 bf b0 05 <unknown>

clasta   x0, p7, x0, z31.d
// CHECK-INST: clasta	x0, p7, x0, z31.d
// CHECK-ENCODING: [0xe0,0xbf,0xf0,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 bf f0 05 <unknown>

clasta   b0, p7, b0, z31.b
// CHECK-INST: clasta	b0, p7, b0, z31.b
// CHECK-ENCODING: [0xe0,0x9f,0x2a,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 9f 2a 05 <unknown>

clasta   h0, p7, h0, z31.h
// CHECK-INST: clasta	h0, p7, h0, z31.h
// CHECK-ENCODING: [0xe0,0x9f,0x6a,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 9f 6a 05 <unknown>

clasta   s0, p7, s0, z31.s
// CHECK-INST: clasta	s0, p7, s0, z31.s
// CHECK-ENCODING: [0xe0,0x9f,0xaa,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 9f aa 05 <unknown>

clasta   d0, p7, d0, z31.d
// CHECK-INST: clasta	d0, p7, d0, z31.d
// CHECK-ENCODING: [0xe0,0x9f,0xea,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 9f ea 05 <unknown>

clasta   z0.b, p7, z0.b, z31.b
// CHECK-INST: clasta	z0.b, p7, z0.b, z31.b
// CHECK-ENCODING: [0xe0,0x9f,0x28,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 9f 28 05 <unknown>

clasta   z0.h, p7, z0.h, z31.h
// CHECK-INST: clasta	z0.h, p7, z0.h, z31.h
// CHECK-ENCODING: [0xe0,0x9f,0x68,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 9f 68 05 <unknown>

clasta   z0.s, p7, z0.s, z31.s
// CHECK-INST: clasta	z0.s, p7, z0.s, z31.s
// CHECK-ENCODING: [0xe0,0x9f,0xa8,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 9f a8 05 <unknown>

clasta   z0.d, p7, z0.d, z31.d
// CHECK-INST: clasta	z0.d, p7, z0.d, z31.d
// CHECK-ENCODING: [0xe0,0x9f,0xe8,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 9f e8 05 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 bc 20 04 <unknown>

clasta   z0.d, p7, z0.d, z31.d
// CHECK-INST: clasta	z0.d, p7, z0.d, z31.d
// CHECK-ENCODING: [0xe0,0x9f,0xe8,0x05]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 9f e8 05 <unknown>
