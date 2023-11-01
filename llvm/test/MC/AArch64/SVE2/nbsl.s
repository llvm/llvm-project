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

nbsl z0.d, z0.d, z1.d, z2.d
// CHECK-INST: nbsl z0.d, z0.d, z1.d, z2.d
// CHECK-ENCODING: [0x40,0x3c,0xe1,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 04e13c40 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z31, z7
// CHECK-INST: movprfx z31, z7
// CHECK-ENCODING: [0xff,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bcff <unknown>

nbsl z31.d, z31.d, z30.d, z29.d
// CHECK-INST: nbsl z31.d, z31.d, z30.d, z29.d
// CHECK-ENCODING: [0xbf,0x3f,0xfe,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 04fe3fbf <unknown>
