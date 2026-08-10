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


sqdmlalt z0.h, z1.b, z31.b
// CHECK-INST: sqdmlalt	z0.h, z1.b, z31.b
// CHECK-ENCODING: [0x20,0x64,0x5f,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 445f6420 <unknown>

sqdmlalt z0.s, z1.h, z31.h
// CHECK-INST: sqdmlalt	z0.s, z1.h, z31.h
// CHECK-ENCODING: [0x20,0x64,0x9f,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 449f6420 <unknown>

sqdmlalt z0.d, z1.s, z31.s
// CHECK-INST: sqdmlalt	z0.d, z1.s, z31.s
// CHECK-ENCODING: [0x20,0x64,0xdf,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 44df6420 <unknown>

sqdmlalt z0.s, z1.h, z7.h[7]
// CHECK-INST: sqdmlalt	z0.s, z1.h, z7.h[7]
// CHECK-ENCODING: [0x20,0x2c,0xbf,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 44bf2c20 <unknown>

sqdmlalt z0.d, z1.s, z15.s[3]
// CHECK-INST: sqdmlalt	z0.d, z1.s, z15.s[3]
// CHECK-ENCODING: [0x20,0x2c,0xff,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 44ff2c20 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z21, z28
// CHECK-INST: movprfx	z21, z28
// CHECK-ENCODING: [0x95,0xbf,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bf95 <unknown>

sqdmlalt z21.d, z1.s, z31.s
// CHECK-INST: sqdmlalt	z21.d, z1.s, z31.s
// CHECK-ENCODING: [0x35,0x64,0xdf,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 44df6435 <unknown>

movprfx z21, z28
// CHECK-INST: movprfx	z21, z28
// CHECK-ENCODING: [0x95,0xbf,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bf95 <unknown>

sqdmlalt   z21.d, z10.s, z5.s[1]
// CHECK-INST: sqdmlalt   z21.d, z10.s, z5.s[1]
// CHECK-ENCODING: [0x55,0x2d,0xe5,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 44e52d55 <unknown>
