// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d -mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN


cmphs   p0.b, p0/z, z0.b, z0.b
// CHECK-INST: cmphs p0.b, p0/z, z0.b, z0.b
// CHECK-ENCODING: [0x00,0x00,0x00,0x24]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 00 00 24 <unknown>

cmphs   p0.h, p0/z, z0.h, z0.h
// CHECK-INST: cmphs p0.h, p0/z, z0.h, z0.h
// CHECK-ENCODING: [0x00,0x00,0x40,0x24]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 00 40 24 <unknown>

cmphs   p0.s, p0/z, z0.s, z0.s
// CHECK-INST: cmphs p0.s, p0/z, z0.s, z0.s
// CHECK-ENCODING: [0x00,0x00,0x80,0x24]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 00 80 24 <unknown>

cmphs   p0.d, p0/z, z0.d, z0.d
// CHECK-INST: cmphs p0.d, p0/z, z0.d, z0.d
// CHECK-ENCODING: [0x00,0x00,0xc0,0x24]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 00 c0 24 <unknown>

cmphs   p0.b, p0/z, z0.b, z0.d
// CHECK-INST: cmphs p0.b, p0/z, z0.b, z0.d
// CHECK-ENCODING: [0x00,0xc0,0x00,0x24]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 c0 00 24 <unknown>

cmphs   p0.h, p0/z, z0.h, z0.d
// CHECK-INST: cmphs p0.h, p0/z, z0.h, z0.d
// CHECK-ENCODING: [0x00,0xc0,0x40,0x24]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 c0 40 24 <unknown>

cmphs   p0.s, p0/z, z0.s, z0.d
// CHECK-INST: cmphs p0.s, p0/z, z0.s, z0.d
// CHECK-ENCODING: [0x00,0xc0,0x80,0x24]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 c0 80 24 <unknown>
