// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d -mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

smax    z0.b, z0.b, #-128
// CHECK-INST: smax	z0.b, z0.b, #-128
// CHECK-ENCODING: [0x00,0xd0,0x28,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 d0 28 25 <unknown>

smax    z31.b, z31.b, #127
// CHECK-INST: smax	z31.b, z31.b, #127
// CHECK-ENCODING: [0xff,0xcf,0x28,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff cf 28 25 <unknown>

smax    z0.h, z0.h, #-128
// CHECK-INST: smax	z0.h, z0.h, #-128
// CHECK-ENCODING: [0x00,0xd0,0x68,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 d0 68 25 <unknown>

smax    z31.h, z31.h, #127
// CHECK-INST: smax	z31.h, z31.h, #127
// CHECK-ENCODING: [0xff,0xcf,0x68,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff cf 68 25 <unknown>

smax    z0.s, z0.s, #-128
// CHECK-INST: smax	z0.s, z0.s, #-128
// CHECK-ENCODING: [0x00,0xd0,0xa8,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 d0 a8 25 <unknown>

smax    z31.s, z31.s, #127
// CHECK-INST: smax	z31.s, z31.s, #127
// CHECK-ENCODING: [0xff,0xcf,0xa8,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff cf a8 25 <unknown>

smax    z0.d, z0.d, #-128
// CHECK-INST: smax	z0.d, z0.d, #-128
// CHECK-ENCODING: [0x00,0xd0,0xe8,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 d0 e8 25 <unknown>

smax    z31.d, z31.d, #127
// CHECK-INST: smax	z31.d, z31.d, #127
// CHECK-ENCODING: [0xff,0xcf,0xe8,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff cf e8 25 <unknown>

smax    z31.b, p7/m, z31.b, z31.b
// CHECK-INST: smax    z31.b, p7/m, z31.b, z31.b
// CHECK-ENCODING: [0xff,0x1f,0x08,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 1f 08 04 <unknown>

smax    z31.h, p7/m, z31.h, z31.h
// CHECK-INST: smax    z31.h, p7/m, z31.h, z31.h
// CHECK-ENCODING: [0xff,0x1f,0x48,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 1f 48 04 <unknown>

smax    z31.s, p7/m, z31.s, z31.s
// CHECK-INST: smax    z31.s, p7/m, z31.s, z31.s
// CHECK-ENCODING: [0xff,0x1f,0x88,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 1f 88 04 <unknown>

smax    z31.d, p7/m, z31.d, z31.d
// CHECK-INST: smax    z31.d, p7/m, z31.d, z31.d
// CHECK-ENCODING: [0xff,0x1f,0xc8,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 1f c8 04 <unknown>
