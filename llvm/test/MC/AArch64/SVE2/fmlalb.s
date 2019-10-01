// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d -mattr=+sve2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN


fmlalb z29.s, z30.h, z31.h
// CHECK-INST: fmlalb z29.s, z30.h, z31.h
// CHECK-ENCODING: [0xdd,0x83,0xbf,0x64]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: dd 83 bf 64 <unknown>

fmlalb z0.s, z1.h, z7.h[0]
// CHECK-INST: fmlalb	z0.s, z1.h, z7.h[0]
// CHECK-ENCODING: [0x20,0x40,0xa7,0x64]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 20 40 a7 64 <unknown>

fmlalb z30.s, z31.h, z7.h[7]
// CHECK-INST: fmlalb z30.s, z31.h, z7.h[7]
// CHECK-ENCODING: [0xfe,0x4b,0xbf,0x64]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: fe 4b bf 64 <unknown>

// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z29, z28
// CHECK-INST: movprfx	z29, z28
// CHECK-ENCODING: [0x9d,0xbf,0x20,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 9d bf 20 04 <unknown>

fmlalb z29.s, z30.h, z31.h
// CHECK-INST: fmlalb z29.s, z30.h, z31.h
// CHECK-ENCODING: [0xdd,0x83,0xbf,0x64]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: dd 83 bf 64 <unknown>

movprfx z21, z28
// CHECK-INST: movprfx	z21, z28
// CHECK-ENCODING: [0x95,0xbf,0x20,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 95 bf 20 04 <unknown>

fmlalb z21.s, z1.h, z7.h[7]
// CHECK-INST: fmlalb	z21.s, z1.h, z7.h[7]
// CHECK-ENCODING: [0x35,0x48,0xbf,0x64]
// CHECK-ERROR: instruction requires: sve2
// CHECK-UNKNOWN: 35 48 bf 64 <unknown>
