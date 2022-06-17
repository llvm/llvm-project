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


fmlslb z29.s, z30.h, z31.h
// CHECK-INST: fmlslb z29.s, z30.h, z31.h
// CHECK-ENCODING: [0xdd,0xa3,0xbf,0x64]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: dd a3 bf 64 <unknown>

fmlslb z0.s, z1.h, z7.h[0]
// CHECK-INST: fmlslb	z0.s, z1.h, z7.h[0]
// CHECK-ENCODING: [0x20,0x60,0xa7,0x64]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 20 60 a7 64 <unknown>

fmlslb z30.s, z31.h, z7.h[7]
// CHECK-INST: fmlslb z30.s, z31.h, z7.h[7]
// CHECK-ENCODING: [0xfe,0x6b,0xbf,0x64]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: fe 6b bf 64 <unknown>

// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z29, z28
// CHECK-INST: movprfx	z29, z28
// CHECK-ENCODING: [0x9d,0xbf,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 9d bf 20 04 <unknown>

fmlslb z29.s, z30.h, z31.h
// CHECK-INST: fmlslb z29.s, z30.h, z31.h
// CHECK-ENCODING: [0xdd,0xa3,0xbf,0x64]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: dd a3 bf 64 <unknown>

movprfx z21, z28
// CHECK-INST: movprfx	z21, z28
// CHECK-ENCODING: [0x95,0xbf,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 95 bf 20 04 <unknown>

fmlslb z21.s, z1.h, z7.h[7]
// CHECK-INST: fmlslb	z21.s, z1.h, z7.h[7]
// CHECK-ENCODING: [0x35,0x68,0xbf,0x64]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 35 68 bf 64 <unknown>
