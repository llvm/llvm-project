// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:   | llvm-objdump -d --mattr=-sve - | FileCheck %s --check-prefix=CHECK-UNKNOWN


cmpeq   p0.b, p0/z, z0.b, z0.b
// CHECK-INST: cmpeq p0.b, p0/z, z0.b, z0.b
// CHECK-ENCODING: [0x00,0xa0,0x00,0x24]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 a0 00 24 <unknown>

cmpeq   p0.h, p0/z, z0.h, z0.h
// CHECK-INST: cmpeq p0.h, p0/z, z0.h, z0.h
// CHECK-ENCODING: [0x00,0xa0,0x40,0x24]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 a0 40 24 <unknown>

cmpeq   p0.s, p0/z, z0.s, z0.s
// CHECK-INST: cmpeq p0.s, p0/z, z0.s, z0.s
// CHECK-ENCODING: [0x00,0xa0,0x80,0x24]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 a0 80 24 <unknown>

cmpeq   p0.d, p0/z, z0.d, z0.d
// CHECK-INST: cmpeq p0.d, p0/z, z0.d, z0.d
// CHECK-ENCODING: [0x00,0xa0,0xc0,0x24]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 a0 c0 24 <unknown>

cmpeq   p0.b, p0/z, z0.b, z0.d
// CHECK-INST: cmpeq p0.b, p0/z, z0.b, z0.d
// CHECK-ENCODING: [0x00,0x20,0x00,0x24]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 20 00 24 <unknown>

cmpeq   p0.h, p0/z, z0.h, z0.d
// CHECK-INST: cmpeq p0.h, p0/z, z0.h, z0.d
// CHECK-ENCODING: [0x00,0x20,0x40,0x24]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 20 40 24 <unknown>

cmpeq   p0.s, p0/z, z0.s, z0.d
// CHECK-INST: cmpeq p0.s, p0/z, z0.s, z0.d
// CHECK-ENCODING: [0x00,0x20,0x80,0x24]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 20 80 24 <unknown>

cmpeq   p0.b, p0/z, z0.b, #-16
// CHECK-INST: cmpeq p0.b, p0/z, z0.b, #-16
// CHECK-ENCODING: [0x00,0x80,0x10,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 80 10 25 <unknown>

cmpeq   p0.h, p0/z, z0.h, #-16
// CHECK-INST: cmpeq p0.h, p0/z, z0.h, #-16
// CHECK-ENCODING: [0x00,0x80,0x50,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 80 50 25 <unknown>

cmpeq   p0.s, p0/z, z0.s, #-16
// CHECK-INST: cmpeq p0.s, p0/z, z0.s, #-16
// CHECK-ENCODING: [0x00,0x80,0x90,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 80 90 25 <unknown>

cmpeq   p0.d, p0/z, z0.d, #-16
// CHECK-INST: cmpeq p0.d, p0/z, z0.d, #-16
// CHECK-ENCODING: [0x00,0x80,0xd0,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 80 d0 25 <unknown>

cmpeq   p0.b, p0/z, z0.b, #15
// CHECK-INST: cmpeq p0.b, p0/z, z0.b, #15
// CHECK-ENCODING: [0x00,0x80,0x0f,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 80 0f 25 <unknown>

cmpeq   p0.h, p0/z, z0.h, #15
// CHECK-INST: cmpeq p0.h, p0/z, z0.h, #15
// CHECK-ENCODING: [0x00,0x80,0x4f,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 80 4f 25 <unknown>

cmpeq   p0.s, p0/z, z0.s, #15
// CHECK-INST: cmpeq p0.s, p0/z, z0.s, #15
// CHECK-ENCODING: [0x00,0x80,0x8f,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 80 8f 25 <unknown>

cmpeq   p0.d, p0/z, z0.d, #15
// CHECK-INST: cmpeq p0.d, p0/z, z0.d, #15
// CHECK-ENCODING: [0x00,0x80,0xcf,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 80 cf 25 <unknown>
