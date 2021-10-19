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


sqdmlalb z0.h, z1.b, z31.b
// CHECK-INST: sqdmlalb	z0.h, z1.b, z31.b
// CHECK-ENCODING: [0x20,0x60,0x5f,0x44]
// CHECK-ERROR: instruction requires: streaming-sve or sve2
// CHECK-UNKNOWN: 20 60 5f 44 <unknown>

sqdmlalb z0.s, z1.h, z31.h
// CHECK-INST: sqdmlalb	z0.s, z1.h, z31.h
// CHECK-ENCODING: [0x20,0x60,0x9f,0x44]
// CHECK-ERROR: instruction requires: streaming-sve or sve2
// CHECK-UNKNOWN: 20 60 9f 44 <unknown>

sqdmlalb z0.d, z1.s, z31.s
// CHECK-INST: sqdmlalb	z0.d, z1.s, z31.s
// CHECK-ENCODING: [0x20,0x60,0xdf,0x44]
// CHECK-ERROR: instruction requires: streaming-sve or sve2
// CHECK-UNKNOWN: 20 60 df 44 <unknown>

sqdmlalb z0.s, z1.h, z7.h[7]
// CHECK-INST: sqdmlalb	z0.s, z1.h, z7.h[7]
// CHECK-ENCODING: [0x20,0x28,0xbf,0x44]
// CHECK-ERROR: instruction requires: streaming-sve or sve2
// CHECK-UNKNOWN: 20 28 bf 44 <unknown>

sqdmlalb z0.d, z1.s, z15.s[3]
// CHECK-INST: sqdmlalb	z0.d, z1.s, z15.s[3]
// CHECK-ENCODING: [0x20,0x28,0xff,0x44]
// CHECK-ERROR: instruction requires: streaming-sve or sve2
// CHECK-UNKNOWN: 20 28 ff 44 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z21, z28
// CHECK-INST: movprfx	z21, z28
// CHECK-ENCODING: [0x95,0xbf,0x20,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 95 bf 20 04 <unknown>

sqdmlalb z21.d, z1.s, z31.s
// CHECK-INST: sqdmlalb	z21.d, z1.s, z31.s
// CHECK-ENCODING: [0x35,0x60,0xdf,0x44]
// CHECK-ERROR: instruction requires: streaming-sve or sve2
// CHECK-UNKNOWN: 35 60 df 44 <unknown>

movprfx z21, z28
// CHECK-INST: movprfx	z21, z28
// CHECK-ENCODING: [0x95,0xbf,0x20,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 95 bf 20 04 <unknown>

sqdmlalb   z21.d, z10.s, z5.s[1]
// CHECK-INST: sqdmlalb   z21.d, z10.s, z5.s[1]
// CHECK-ENCODING: [0x55,0x29,0xe5,0x44]
// CHECK-ERROR: instruction requires: streaming-sve or sve2
// CHECK-UNKNOWN: 55 29 e5 44 <unknown>
