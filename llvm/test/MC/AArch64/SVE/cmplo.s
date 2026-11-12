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

cmplo   p0.b, p0/z, z0.b, z1.b
// CHECK-INST: cmphi	p0.b, p0/z, z1.b, z0.b
// CHECK-ENCODING: [0x30,0x00,0x00,0x24]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 24000030 <unknown>

cmplo   p0.h, p0/z, z0.h, z1.h
// CHECK-INST: cmphi	p0.h, p0/z, z1.h, z0.h
// CHECK-ENCODING: [0x30,0x00,0x40,0x24]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 24400030 <unknown>

cmplo   p0.s, p0/z, z0.s, z1.s
// CHECK-INST: cmphi	p0.s, p0/z, z1.s, z0.s
// CHECK-ENCODING: [0x30,0x00,0x80,0x24]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 24800030 <unknown>

cmplo   p0.d, p0/z, z0.d, z1.d
// CHECK-INST: cmphi	p0.d, p0/z, z1.d, z0.d
// CHECK-ENCODING: [0x30,0x00,0xc0,0x24]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 24c00030 <unknown>

cmplo   p0.b, p0/z, z0.b, z0.d
// CHECK-INST: cmplo p0.b, p0/z, z0.b, z0.d
// CHECK-ENCODING: [0x00,0xe0,0x00,0x24]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2400e000 <unknown>

cmplo   p0.h, p0/z, z0.h, z0.d
// CHECK-INST: cmplo p0.h, p0/z, z0.h, z0.d
// CHECK-ENCODING: [0x00,0xe0,0x40,0x24]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2440e000 <unknown>

cmplo   p0.s, p0/z, z0.s, z0.d
// CHECK-INST: cmplo p0.s, p0/z, z0.s, z0.d
// CHECK-ENCODING: [0x00,0xe0,0x80,0x24]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2480e000 <unknown>

cmplo   p0.b, p0/z, z0.b, #0
// CHECK-INST: cmplo p0.b, p0/z, z0.b, #0
// CHECK-ENCODING: [0x00,0x20,0x20,0x24]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 24202000 <unknown>

cmplo   p0.h, p0/z, z0.h, #0
// CHECK-INST: cmplo p0.h, p0/z, z0.h, #0
// CHECK-ENCODING: [0x00,0x20,0x60,0x24]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 24602000 <unknown>

cmplo   p0.s, p0/z, z0.s, #0
// CHECK-INST: cmplo p0.s, p0/z, z0.s, #0
// CHECK-ENCODING: [0x00,0x20,0xa0,0x24]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 24a02000 <unknown>

cmplo   p0.d, p0/z, z0.d, #0
// CHECK-INST: cmplo p0.d, p0/z, z0.d, #0
// CHECK-ENCODING: [0x00,0x20,0xe0,0x24]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 24e02000 <unknown>

cmplo   p0.b, p0/z, z0.b, #127
// CHECK-INST: cmplo p0.b, p0/z, z0.b, #127
// CHECK-ENCODING: [0x00,0xe0,0x3f,0x24]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 243fe000 <unknown>

cmplo   p0.h, p0/z, z0.h, #127
// CHECK-INST: cmplo p0.h, p0/z, z0.h, #127
// CHECK-ENCODING: [0x00,0xe0,0x7f,0x24]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 247fe000 <unknown>

cmplo   p0.s, p0/z, z0.s, #127
// CHECK-INST: cmplo p0.s, p0/z, z0.s, #127
// CHECK-ENCODING: [0x00,0xe0,0xbf,0x24]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 24bfe000 <unknown>

cmplo   p0.d, p0/z, z0.d, #127
// CHECK-INST: cmplo p0.d, p0/z, z0.d, #127
// CHECK-ENCODING: [0x00,0xe0,0xff,0x24]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 24ffe000 <unknown>
