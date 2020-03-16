// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

ftmad z0.h, z0.h, z31.h, #7
// CHECK-INST: ftmad	z0.h, z0.h, z31.h, #7
// CHECK-ENCODING: [0xe0,0x83,0x57,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 83 57 65 <unknown>

ftmad z0.s, z0.s, z31.s, #7
// CHECK-INST: ftmad	z0.s, z0.s, z31.s, #7
// CHECK-ENCODING: [0xe0,0x83,0x97,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 83 97 65 <unknown>

ftmad z0.d, z0.d, z31.d, #7
// CHECK-INST: ftmad	z0.d, z0.d, z31.d, #7
// CHECK-ENCODING: [0xe0,0x83,0xd7,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 83 d7 65 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 bc 20 04 <unknown>

ftmad z0.d, z0.d, z31.d, #7
// CHECK-INST: ftmad	z0.d, z0.d, z31.d, #7
// CHECK-ENCODING: [0xe0,0x83,0xd7,0x65]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 83 d7 65 <unknown>
