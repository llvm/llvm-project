// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d -mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

cmpge   p0.b, p0/z, z0.b, z0.b
// CHECK-INST: cmpge   p0.b, p0/z, z0.b, z0.b
// CHECK-ENCODING: [0x00,0x80,0x00,0x24]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 80 00 24 <unknown>

cmpge   p0.h, p0/z, z0.h, z0.h
// CHECK-INST: cmpge   p0.h, p0/z, z0.h, z0.h
// CHECK-ENCODING: [0x00,0x80,0x40,0x24]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 80 40 24 <unknown>

cmpge   p0.s, p0/z, z0.s, z0.s
// CHECK-INST: cmpge   p0.s, p0/z, z0.s, z0.s
// CHECK-ENCODING: [0x00,0x80,0x80,0x24]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 80 80 24 <unknown>

cmpge   p0.d, p0/z, z0.d, z0.d
// CHECK-INST: cmpge   p0.d, p0/z, z0.d, z0.d
// CHECK-ENCODING: [0x00,0x80,0xc0,0x24]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 80 c0 24 <unknown>

cmpge   p0.b, p0/z, z0.b, z0.d
// CHECK-INST: cmpge   p0.b, p0/z, z0.b, z0.d
// CHECK-ENCODING: [0x00,0x40,0x00,0x24]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 40 00 24 <unknown>

cmpge   p0.h, p0/z, z0.h, z0.d
// CHECK-INST: cmpge   p0.h, p0/z, z0.h, z0.d
// CHECK-ENCODING: [0x00,0x40,0x40,0x24]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 40 40 24 <unknown>

cmpge   p0.s, p0/z, z0.s, z0.d
// CHECK-INST: cmpge   p0.s, p0/z, z0.s, z0.d
// CHECK-ENCODING: [0x00,0x40,0x80,0x24]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 40 80 24 <unknown>
