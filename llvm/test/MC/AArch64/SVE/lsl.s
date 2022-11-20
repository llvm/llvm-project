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

lsl     z0.b, z0.b, #0
// CHECK-INST: lsl	z0.b, z0.b, #0
// CHECK-ENCODING: [0x00,0x9c,0x28,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04289c00 <unknown>

lsl     z31.b, z31.b, #7
// CHECK-INST: lsl	z31.b, z31.b, #7
// CHECK-ENCODING: [0xff,0x9f,0x2f,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 042f9fff <unknown>

lsl     z0.h, z0.h, #0
// CHECK-INST: lsl	z0.h, z0.h, #0
// CHECK-ENCODING: [0x00,0x9c,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04309c00 <unknown>

lsl     z31.h, z31.h, #15
// CHECK-INST: lsl	z31.h, z31.h, #15
// CHECK-ENCODING: [0xff,0x9f,0x3f,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 043f9fff <unknown>

lsl     z0.s, z0.s, #0
// CHECK-INST: lsl	z0.s, z0.s, #0
// CHECK-ENCODING: [0x00,0x9c,0x60,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04609c00 <unknown>

lsl     z31.s, z31.s, #31
// CHECK-INST: lsl	z31.s, z31.s, #31
// CHECK-ENCODING: [0xff,0x9f,0x7f,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 047f9fff <unknown>

lsl     z0.d, z0.d, #0
// CHECK-INST: lsl	z0.d, z0.d, #0
// CHECK-ENCODING: [0x00,0x9c,0xa0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04a09c00 <unknown>

lsl     z31.d, z31.d, #63
// CHECK-INST: lsl	z31.d, z31.d, #63
// CHECK-ENCODING: [0xff,0x9f,0xff,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04ff9fff <unknown>

lsl     z0.b, p0/m, z0.b, #0
// CHECK-INST: lsl	z0.b, p0/m, z0.b, #0
// CHECK-ENCODING: [0x00,0x81,0x03,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04038100 <unknown>

lsl     z31.b, p0/m, z31.b, #7
// CHECK-INST: lsl	z31.b, p0/m, z31.b, #7
// CHECK-ENCODING: [0xff,0x81,0x03,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 040381ff <unknown>

lsl     z0.h, p0/m, z0.h, #0
// CHECK-INST: lsl	z0.h, p0/m, z0.h, #0
// CHECK-ENCODING: [0x00,0x82,0x03,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04038200 <unknown>

lsl     z31.h, p0/m, z31.h, #15
// CHECK-INST: lsl	z31.h, p0/m, z31.h, #15
// CHECK-ENCODING: [0xff,0x83,0x03,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 040383ff <unknown>

lsl     z0.s, p0/m, z0.s, #0
// CHECK-INST: lsl	z0.s, p0/m, z0.s, #0
// CHECK-ENCODING: [0x00,0x80,0x43,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04438000 <unknown>

lsl     z31.s, p0/m, z31.s, #31
// CHECK-INST: lsl	z31.s, p0/m, z31.s, #31
// CHECK-ENCODING: [0xff,0x83,0x43,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 044383ff <unknown>

lsl     z0.d, p0/m, z0.d, #0
// CHECK-INST: lsl	z0.d, p0/m, z0.d, #0
// CHECK-ENCODING: [0x00,0x80,0x83,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04838000 <unknown>

lsl     z31.d, p0/m, z31.d, #63
// CHECK-INST: lsl	z31.d, p0/m, z31.d, #63
// CHECK-ENCODING: [0xff,0x83,0xc3,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04c383ff <unknown>

lsl     z0.b, p0/m, z0.b, z0.b
// CHECK-INST: lsl	z0.b, p0/m, z0.b, z0.b
// CHECK-ENCODING: [0x00,0x80,0x13,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04138000 <unknown>

lsl     z0.h, p0/m, z0.h, z0.h
// CHECK-INST: lsl	z0.h, p0/m, z0.h, z0.h
// CHECK-ENCODING: [0x00,0x80,0x53,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04538000 <unknown>

lsl     z0.s, p0/m, z0.s, z0.s
// CHECK-INST: lsl	z0.s, p0/m, z0.s, z0.s
// CHECK-ENCODING: [0x00,0x80,0x93,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04938000 <unknown>

lsl     z0.d, p0/m, z0.d, z0.d
// CHECK-INST: lsl	z0.d, p0/m, z0.d, z0.d
// CHECK-ENCODING: [0x00,0x80,0xd3,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04d38000 <unknown>

lsl     z0.b, p0/m, z0.b, z1.d
// CHECK-INST: lsl	z0.b, p0/m, z0.b, z1.d
// CHECK-ENCODING: [0x20,0x80,0x1b,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 041b8020 <unknown>

lsl     z0.h, p0/m, z0.h, z1.d
// CHECK-INST: lsl	z0.h, p0/m, z0.h, z1.d
// CHECK-ENCODING: [0x20,0x80,0x5b,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 045b8020 <unknown>

lsl     z0.s, p0/m, z0.s, z1.d
// CHECK-INST: lsl	z0.s, p0/m, z0.s, z1.d
// CHECK-ENCODING: [0x20,0x80,0x9b,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 049b8020 <unknown>

lsl     z0.b, z1.b, z2.d
// CHECK-INST: lsl	z0.b, z1.b, z2.d
// CHECK-ENCODING: [0x20,0x8c,0x22,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04228c20 <unknown>

lsl     z0.h, z1.h, z2.d
// CHECK-INST: lsl	z0.h, z1.h, z2.d
// CHECK-ENCODING: [0x20,0x8c,0x62,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04628c20 <unknown>

lsl     z0.s, z1.s, z2.d
// CHECK-INST: lsl	z0.s, z1.s, z2.d
// CHECK-ENCODING: [0x20,0x8c,0xa2,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04a28c20 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z31.d, p0/z, z6.d
// CHECK-INST: movprfx	z31.d, p0/z, z6.d
// CHECK-ENCODING: [0xdf,0x20,0xd0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04d020df <unknown>

lsl     z31.d, p0/m, z31.d, #63
// CHECK-INST: lsl	z31.d, p0/m, z31.d, #63
// CHECK-ENCODING: [0xff,0x83,0xc3,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04c383ff <unknown>

movprfx z31, z6
// CHECK-INST: movprfx	z31, z6
// CHECK-ENCODING: [0xdf,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bcdf <unknown>

lsl     z31.d, p0/m, z31.d, #63
// CHECK-INST: lsl	z31.d, p0/m, z31.d, #63
// CHECK-ENCODING: [0xff,0x83,0xc3,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04c383ff <unknown>

movprfx z0.s, p0/z, z7.s
// CHECK-INST: movprfx	z0.s, p0/z, z7.s
// CHECK-ENCODING: [0xe0,0x20,0x90,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 049020e0 <unknown>

lsl     z0.s, p0/m, z0.s, z1.d
// CHECK-INST: lsl	z0.s, p0/m, z0.s, z1.d
// CHECK-ENCODING: [0x20,0x80,0x9b,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 049b8020 <unknown>

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bce0 <unknown>

lsl     z0.s, p0/m, z0.s, z1.d
// CHECK-INST: lsl	z0.s, p0/m, z0.s, z1.d
// CHECK-ENCODING: [0x20,0x80,0x9b,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 049b8020 <unknown>
