// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d -mattr=+sve2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

adclt z0.s, z1.s, z31.s
// CHECK-INST: adclt z0.s, z1.s, z31.s
// CHECK-ENCODING: [0x20,0xd4,0x1f,0x45]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 20 d4 1f 45 <unknown>

adclt z0.d, z1.d, z31.d
// CHECK-INST: adclt z0.d, z1.d, z31.d
// CHECK-ENCODING: [0x20,0xd4,0x5f,0x45]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 20 d4 5f 45 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 bc 20 04 <unknown>

adclt z0.d, z1.d, z31.d
// CHECK-INST: adclt z0.d, z1.d, z31.d
// CHECK-ENCODING: [0x20,0xd4,0x5f,0x45]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 20 d4 5f 45 <unknown>
