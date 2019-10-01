// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d -mattr=+sve2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

fminp z0.h, p0/m, z0.h, z1.h
// CHECK-INST: fminp z0.h, p0/m, z0.h, z1.h
// CHECK-ENCODING: [0x20,0x80,0x57,0x64]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 20 80 57 64 <unknown>

fminp z29.s, p3/m, z29.s, z30.s
// CHECK-INST: fminp z29.s, p3/m, z29.s, z30.s
// CHECK-ENCODING: [0xdd,0x8f,0x97,0x64]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: dd 8f 97 64 <unknown>

fminp z31.d, p7/m, z31.d, z30.d
// CHECK-INST: fminp z31.d, p7/m, z31.d, z30.d
// CHECK-ENCODING: [0xdf,0x9f,0xd7,0x64]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: df 9f d7 64 <unknown>

// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z31.d, p0/z, z6.d
// CHECK-INST: movprfx z31.d, p0/z, z6.d
// CHECK-ENCODING: [0xdf,0x20,0xd0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: df 20 d0 04 <unknown>

fminp z31.d, p0/m, z31.d, z30.d
// CHECK-INST: fminp z31.d, p0/m, z31.d, z30.d
// CHECK-ENCODING: [0xdf,0x83,0xd7,0x64]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: df 83 d7 64 <unknown>

movprfx z31, z6
// CHECK-INST: movprfx z31, z6
// CHECK-ENCODING: [0xdf,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: df bc 20 04 <unknown>

fminp z31.d, p7/m, z31.d, z30.d
// CHECK-INST: fminp z31.d, p7/m, z31.d, z30.d
// CHECK-ENCODING: [0xdf,0x9f,0xd7,0x64]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: df 9f d7 64 <unknown>
