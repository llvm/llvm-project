// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump --no-print-imm-hex -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:   | llvm-objdump --no-print-imm-hex -d --mattr=-sve - | FileCheck %s --check-prefix=CHECK-UNKNOWN


cmpgt   p0.b, p0/z, z0.b, z0.b
// CHECK-INST: cmpgt p0.b, p0/z, z0.b, z0.b
// CHECK-ENCODING: [0x10,0x80,0x00,0x24]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 24008010 <unknown>

cmpgt   p0.h, p0/z, z0.h, z0.h
// CHECK-INST: cmpgt p0.h, p0/z, z0.h, z0.h
// CHECK-ENCODING: [0x10,0x80,0x40,0x24]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 24408010 <unknown>

cmpgt   p0.s, p0/z, z0.s, z0.s
// CHECK-INST: cmpgt p0.s, p0/z, z0.s, z0.s
// CHECK-ENCODING: [0x10,0x80,0x80,0x24]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 24808010 <unknown>

cmpgt   p0.d, p0/z, z0.d, z0.d
// CHECK-INST: cmpgt p0.d, p0/z, z0.d, z0.d
// CHECK-ENCODING: [0x10,0x80,0xc0,0x24]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 24c08010 <unknown>

cmpgt   p0.b, p0/z, z0.b, z0.d
// CHECK-INST: cmpgt p0.b, p0/z, z0.b, z0.d
// CHECK-ENCODING: [0x10,0x40,0x00,0x24]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 24004010 <unknown>

cmpgt   p0.h, p0/z, z0.h, z0.d
// CHECK-INST: cmpgt p0.h, p0/z, z0.h, z0.d
// CHECK-ENCODING: [0x10,0x40,0x40,0x24]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 24404010 <unknown>

cmpgt   p0.s, p0/z, z0.s, z0.d
// CHECK-INST: cmpgt p0.s, p0/z, z0.s, z0.d
// CHECK-ENCODING: [0x10,0x40,0x80,0x24]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 24804010 <unknown>

cmpgt   p0.b, p0/z, z0.b, #-16
// CHECK-INST: cmpgt p0.b, p0/z, z0.b, #-16
// CHECK-ENCODING: [0x10,0x00,0x10,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25100010 <unknown>

cmpgt   p0.h, p0/z, z0.h, #-16
// CHECK-INST: cmpgt p0.h, p0/z, z0.h, #-16
// CHECK-ENCODING: [0x10,0x00,0x50,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25500010 <unknown>

cmpgt   p0.s, p0/z, z0.s, #-16
// CHECK-INST: cmpgt p0.s, p0/z, z0.s, #-16
// CHECK-ENCODING: [0x10,0x00,0x90,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25900010 <unknown>

cmpgt   p0.d, p0/z, z0.d, #-16
// CHECK-INST: cmpgt p0.d, p0/z, z0.d, #-16
// CHECK-ENCODING: [0x10,0x00,0xd0,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25d00010 <unknown>

cmpgt   p0.b, p0/z, z0.b, #15
// CHECK-INST: cmpgt p0.b, p0/z, z0.b, #15
// CHECK-ENCODING: [0x10,0x00,0x0f,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 250f0010 <unknown>

cmpgt   p0.h, p0/z, z0.h, #15
// CHECK-INST: cmpgt p0.h, p0/z, z0.h, #15
// CHECK-ENCODING: [0x10,0x00,0x4f,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 254f0010 <unknown>

cmpgt   p0.s, p0/z, z0.s, #15
// CHECK-INST: cmpgt p0.s, p0/z, z0.s, #15
// CHECK-ENCODING: [0x10,0x00,0x8f,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 258f0010 <unknown>

cmpgt   p0.d, p0/z, z0.d, #15
// CHECK-INST: cmpgt p0.d, p0/z, z0.d, #15
// CHECK-ENCODING: [0x10,0x00,0xcf,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25cf0010 <unknown>
