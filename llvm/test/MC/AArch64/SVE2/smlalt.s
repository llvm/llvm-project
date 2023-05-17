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


smlalt z0.h, z1.b, z31.b
// CHECK-INST: smlalt	z0.h, z1.b, z31.b
// CHECK-ENCODING: [0x20,0x44,0x5f,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 445f4420 <unknown>

smlalt z0.s, z1.h, z31.h
// CHECK-INST: smlalt	z0.s, z1.h, z31.h
// CHECK-ENCODING: [0x20,0x44,0x9f,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 449f4420 <unknown>

smlalt z0.d, z1.s, z31.s
// CHECK-INST: smlalt	z0.d, z1.s, z31.s
// CHECK-ENCODING: [0x20,0x44,0xdf,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 44df4420 <unknown>

smlalt z0.s, z1.h, z7.h[7]
// CHECK-INST: smlalt	z0.s, z1.h, z7.h[7]
// CHECK-ENCODING: [0x20,0x8c,0xbf,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 44bf8c20 <unknown>

smlalt z0.d, z1.s, z15.s[1]
// CHECK-INST: smlalt	z0.d, z1.s, z15.s[1]
// CHECK-ENCODING: [0x20,0x8c,0xef,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 44ef8c20 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z21, z28
// CHECK-INST: movprfx	z21, z28
// CHECK-ENCODING: [0x95,0xbf,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bf95 <unknown>

smlalt z21.d, z1.s, z31.s
// CHECK-INST: smlalt	z21.d, z1.s, z31.s
// CHECK-ENCODING: [0x35,0x44,0xdf,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 44df4435 <unknown>

movprfx z21, z28
// CHECK-INST: movprfx	z21, z28
// CHECK-ENCODING: [0x95,0xbf,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bf95 <unknown>

smlalt   z21.d, z10.s, z5.s[1]
// CHECK-INST: smlalt   z21.d, z10.s, z5.s[1]
// CHECK-ENCODING: [0x55,0x8d,0xe5,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 44e58d55 <unknown>
