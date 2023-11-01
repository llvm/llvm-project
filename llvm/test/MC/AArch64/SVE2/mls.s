// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:   | llvm-objdump -d --mattr=-sve2 - | FileCheck %s --check-prefix=CHECK-UNKNOWN

mls z0.h, z1.h, z7.h[7]
// CHECK-INST: mls	z0.h, z1.h, z7.h[7]
// CHECK-ENCODING: [0x20,0x0c,0x7f,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 447f0c20 <unknown>

mls z0.s, z1.s, z7.s[3]
// CHECK-INST: mls	z0.s, z1.s, z7.s[3]
// CHECK-ENCODING: [0x20,0x0c,0xbf,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 44bf0c20 <unknown>

mls z0.d, z1.d, z7.d[1]
// CHECK-INST: mls	z0.d, z1.d, z7.d[1]
// CHECK-ENCODING: [0x20,0x0c,0xf7,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 44f70c20 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bce0 <unknown>

mls z0.d, z1.d, z7.d[1]
// CHECK-INST: mls	z0.d, z1.d, z7.d[1]
// CHECK-ENCODING: [0x20,0x0c,0xf7,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 44f70c20 <unknown>
