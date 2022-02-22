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


sqdmlslt z0.h, z1.b, z31.b
// CHECK-INST: sqdmlslt	z0.h, z1.b, z31.b
// CHECK-ENCODING: [0x20,0x6c,0x5f,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 20 6c 5f 44 <unknown>

sqdmlslt z0.s, z1.h, z31.h
// CHECK-INST: sqdmlslt	z0.s, z1.h, z31.h
// CHECK-ENCODING: [0x20,0x6c,0x9f,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 20 6c 9f 44 <unknown>

sqdmlslt z0.d, z1.s, z31.s
// CHECK-INST: sqdmlslt	z0.d, z1.s, z31.s
// CHECK-ENCODING: [0x20,0x6c,0xdf,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 20 6c df 44 <unknown>

sqdmlslt z0.s, z1.h, z7.h[7]
// CHECK-INST: sqdmlslt	z0.s, z1.h, z7.h[7]
// CHECK-ENCODING: [0x20,0x3c,0xbf,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 20 3c bf 44 <unknown>

sqdmlslt z0.d, z1.s, z15.s[3]
// CHECK-INST: sqdmlslt	z0.d, z1.s, z15.s[3]
// CHECK-ENCODING: [0x20,0x3c,0xff,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 20 3c ff 44 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z21, z28
// CHECK-INST: movprfx	z21, z28
// CHECK-ENCODING: [0x95,0xbf,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 95 bf 20 04 <unknown>

sqdmlslt z21.d, z1.s, z31.s
// CHECK-INST: sqdmlslt	z21.d, z1.s, z31.s
// CHECK-ENCODING: [0x35,0x6c,0xdf,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 35 6c df 44 <unknown>

movprfx z21, z28
// CHECK-INST: movprfx	z21, z28
// CHECK-ENCODING: [0x95,0xbf,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 95 bf 20 04 <unknown>

sqdmlslt   z21.d, z10.s, z5.s[1]
// CHECK-INST: sqdmlslt   z21.d, z10.s, z5.s[1]
// CHECK-ENCODING: [0x55,0x3d,0xe5,0x44]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 55 3d e5 44 <unknown>
