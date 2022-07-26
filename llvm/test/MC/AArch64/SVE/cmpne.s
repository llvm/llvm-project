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


cmpne   p0.b, p0/z, z0.b, z0.b
// CHECK-INST: cmpne p0.b, p0/z, z0.b, z0.b
// CHECK-ENCODING: [0x10,0xa0,0x00,0x24]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2400a010 <unknown>

cmpne   p0.h, p0/z, z0.h, z0.h
// CHECK-INST: cmpne p0.h, p0/z, z0.h, z0.h
// CHECK-ENCODING: [0x10,0xa0,0x40,0x24]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2440a010 <unknown>

cmpne   p0.s, p0/z, z0.s, z0.s
// CHECK-INST: cmpne p0.s, p0/z, z0.s, z0.s
// CHECK-ENCODING: [0x10,0xa0,0x80,0x24]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2480a010 <unknown>

cmpne   p0.d, p0/z, z0.d, z0.d
// CHECK-INST: cmpne p0.d, p0/z, z0.d, z0.d
// CHECK-ENCODING: [0x10,0xa0,0xc0,0x24]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 24c0a010 <unknown>

cmpne   p0.b, p0/z, z0.b, z0.d
// CHECK-INST: cmpne p0.b, p0/z, z0.b, z0.d
// CHECK-ENCODING: [0x10,0x20,0x00,0x24]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 24002010 <unknown>

cmpne   p0.h, p0/z, z0.h, z0.d
// CHECK-INST: cmpne p0.h, p0/z, z0.h, z0.d
// CHECK-ENCODING: [0x10,0x20,0x40,0x24]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 24402010 <unknown>

cmpne   p0.s, p0/z, z0.s, z0.d
// CHECK-INST: cmpne p0.s, p0/z, z0.s, z0.d
// CHECK-ENCODING: [0x10,0x20,0x80,0x24]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 24802010 <unknown>

cmpne   p0.b, p0/z, z0.b, #-16
// CHECK-INST: cmpne p0.b, p0/z, z0.b, #-16
// CHECK-ENCODING: [0x10,0x80,0x10,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25108010 <unknown>

cmpne   p0.h, p0/z, z0.h, #-16
// CHECK-INST: cmpne p0.h, p0/z, z0.h, #-16
// CHECK-ENCODING: [0x10,0x80,0x50,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25508010 <unknown>

cmpne   p0.s, p0/z, z0.s, #-16
// CHECK-INST: cmpne p0.s, p0/z, z0.s, #-16
// CHECK-ENCODING: [0x10,0x80,0x90,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25908010 <unknown>

cmpne   p0.d, p0/z, z0.d, #-16
// CHECK-INST: cmpne p0.d, p0/z, z0.d, #-16
// CHECK-ENCODING: [0x10,0x80,0xd0,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25d08010 <unknown>

cmpne   p0.b, p0/z, z0.b, #15
// CHECK-INST: cmpne p0.b, p0/z, z0.b, #15
// CHECK-ENCODING: [0x10,0x80,0x0f,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 250f8010 <unknown>

cmpne   p0.h, p0/z, z0.h, #15
// CHECK-INST: cmpne p0.h, p0/z, z0.h, #15
// CHECK-ENCODING: [0x10,0x80,0x4f,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 254f8010 <unknown>

cmpne   p0.s, p0/z, z0.s, #15
// CHECK-INST: cmpne p0.s, p0/z, z0.s, #15
// CHECK-ENCODING: [0x10,0x80,0x8f,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 258f8010 <unknown>

cmpne   p0.d, p0/z, z0.d, #15
// CHECK-INST: cmpne p0.d, p0/z, z0.d, #15
// CHECK-ENCODING: [0x10,0x80,0xcf,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 25cf8010 <unknown>
