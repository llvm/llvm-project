// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN


umlslb z0.h, z1.b, z31.b
// CHECK-INST: umlslb	z0.h, z1.b, z31.b
// CHECK-ENCODING: [0x20,0x58,0x5f,0x44]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 20 58 5f 44 <unknown>

umlslb z0.s, z1.h, z31.h
// CHECK-INST: umlslb	z0.s, z1.h, z31.h
// CHECK-ENCODING: [0x20,0x58,0x9f,0x44]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 20 58 9f 44 <unknown>

umlslb z0.d, z1.s, z31.s
// CHECK-INST: umlslb	z0.d, z1.s, z31.s
// CHECK-ENCODING: [0x20,0x58,0xdf,0x44]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 20 58 df 44 <unknown>

umlslb z0.s, z1.h, z7.h[7]
// CHECK-INST: umlslb	z0.s, z1.h, z7.h[7]
// CHECK-ENCODING: [0x20,0xb8,0xbf,0x44]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 20 b8 bf 44 <unknown>

umlslb z0.d, z1.s, z15.s[1]
// CHECK-INST: umlslb	z0.d, z1.s, z15.s[1]
// CHECK-ENCODING: [0x20,0xb8,0xef,0x44]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 20 b8 ef 44 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z21, z28
// CHECK-INST: movprfx	z21, z28
// CHECK-ENCODING: [0x95,0xbf,0x20,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 95 bf 20 04 <unknown>

umlslb z21.d, z1.s, z31.s
// CHECK-INST: umlslb	z21.d, z1.s, z31.s
// CHECK-ENCODING: [0x35,0x58,0xdf,0x44]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 35 58 df 44 <unknown>

movprfx z21, z28
// CHECK-INST: movprfx	z21, z28
// CHECK-ENCODING: [0x95,0xbf,0x20,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 95 bf 20 04 <unknown>

umlslb   z21.d, z10.s, z5.s[1]
// CHECK-INST: umlslb   z21.d, z10.s, z5.s[1]
// CHECK-ENCODING: [0x55,0xb9,0xe5,0x44]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 55 b9 e5 44 <unknown>
