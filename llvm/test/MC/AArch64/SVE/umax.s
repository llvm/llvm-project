// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d -mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

umax    z0.b, z0.b, #0
// CHECK-INST: umax	z0.b, z0.b, #0
// CHECK-ENCODING: [0x00,0xc0,0x29,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 c0 29 25 <unknown>

umax    z31.b, z31.b, #255
// CHECK-INST: umax	z31.b, z31.b, #255
// CHECK-ENCODING: [0xff,0xdf,0x29,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff df 29 25 <unknown>

umax    z0.b, z0.b, #0
// CHECK-INST: umax	z0.b, z0.b, #0
// CHECK-ENCODING: [0x00,0xc0,0x29,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 c0 29 25 <unknown>

umax    z31.b, z31.b, #255
// CHECK-INST: umax	z31.b, z31.b, #255
// CHECK-ENCODING: [0xff,0xdf,0x29,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff df 29 25 <unknown>

umax    z0.b, z0.b, #0
// CHECK-INST: umax	z0.b, z0.b, #0
// CHECK-ENCODING: [0x00,0xc0,0x29,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 c0 29 25 <unknown>

umax    z31.b, z31.b, #255
// CHECK-INST: umax	z31.b, z31.b, #255
// CHECK-ENCODING: [0xff,0xdf,0x29,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff df 29 25 <unknown>

umax    z0.b, z0.b, #0
// CHECK-INST: umax	z0.b, z0.b, #0
// CHECK-ENCODING: [0x00,0xc0,0x29,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 c0 29 25 <unknown>

umax    z31.b, z31.b, #255
// CHECK-INST: umax	z31.b, z31.b, #255
// CHECK-ENCODING: [0xff,0xdf,0x29,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff df 29 25 <unknown>

umax    z31.b, p7/m, z31.b, z31.b
// CHECK-INST: umax    z31.b, p7/m, z31.b, z31.b
// CHECK-ENCODING: [0xff,0x1f,0x09,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 1f 09 04 <unknown>

umax    z31.h, p7/m, z31.h, z31.h
// CHECK-INST: umax    z31.h, p7/m, z31.h, z31.h
// CHECK-ENCODING: [0xff,0x1f,0x49,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 1f 49 04 <unknown>

umax    z31.s, p7/m, z31.s, z31.s
// CHECK-INST: umax    z31.s, p7/m, z31.s, z31.s
// CHECK-ENCODING: [0xff,0x1f,0x89,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 1f 89 04 <unknown>

umax    z31.d, p7/m, z31.d, z31.d
// CHECK-INST: umax    z31.d, p7/m, z31.d, z31.d
// CHECK-ENCODING: [0xff,0x1f,0xc9,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 1f c9 04 <unknown>
