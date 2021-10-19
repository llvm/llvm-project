// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+streaming-sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

uadalp z0.h, p0/m, z1.b
// CHECK-INST: uadalp z0.h, p0/m, z1.b
// CHECK-ENCODING: [0x20,0xa0,0x45,0x44]
// CHECK-ERROR: instruction requires: streaming-sve or sve2
// CHECK-UNKNOWN: 20 a0 45 44 <unknown>

uadalp z29.s, p0/m, z30.h
// CHECK-INST: uadalp z29.s, p0/m, z30.h
// CHECK-ENCODING: [0xdd,0xa3,0x85,0x44]
// CHECK-ERROR: instruction requires: streaming-sve or sve2
// CHECK-UNKNOWN: dd a3 85 44 <unknown>

uadalp z30.d, p7/m, z31.s
// CHECK-INST: uadalp z30.d, p7/m, z31.s
// CHECK-ENCODING: [0xfe,0xbf,0xc5,0x44]
// CHECK-ERROR: instruction requires: streaming-sve or sve2
// CHECK-UNKNOWN: fe bf c5 44 <unknown>

// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z31.d, p0/z, z6.d
// CHECK-INST: movprfx z31.d, p0/z, z6.d
// CHECK-ENCODING: [0xdf,0x20,0xd0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: df 20 d0 04 <unknown>

uadalp z31.d, p0/m, z30.s
// CHECK-INST: uadalp z31.d, p0/m, z30.s
// CHECK-ENCODING: [0xdf,0xa3,0xc5,0x44]
// CHECK-ERROR: instruction requires: streaming-sve or sve2
// CHECK-UNKNOWN: df a3 c5 44 <unknown>

movprfx z31, z6
// CHECK-INST: movprfx z31, z6
// CHECK-ENCODING: [0xdf,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: df bc 20 04 <unknown>

uadalp z31.d, p0/m, z30.s
// CHECK-INST: uadalp z31.d, p0/m, z30.s
// CHECK-ENCODING: [0xdf,0xa3,0xc5,0x44]
// CHECK-ERROR: instruction requires: streaming-sve or sve2
// CHECK-UNKNOWN: df a3 c5 44 <unknown>
