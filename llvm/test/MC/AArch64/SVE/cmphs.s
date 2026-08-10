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


cmphs   p0.b, p0/z, z0.b, z0.b
// CHECK-INST: cmphs p0.b, p0/z, z0.b, z0.b
// CHECK-ENCODING: [0x00,0x00,0x00,0x24]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 24000000 <unknown>

cmphs   p0.h, p0/z, z0.h, z0.h
// CHECK-INST: cmphs p0.h, p0/z, z0.h, z0.h
// CHECK-ENCODING: [0x00,0x00,0x40,0x24]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 24400000 <unknown>

cmphs   p0.s, p0/z, z0.s, z0.s
// CHECK-INST: cmphs p0.s, p0/z, z0.s, z0.s
// CHECK-ENCODING: [0x00,0x00,0x80,0x24]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 24800000 <unknown>

cmphs   p0.d, p0/z, z0.d, z0.d
// CHECK-INST: cmphs p0.d, p0/z, z0.d, z0.d
// CHECK-ENCODING: [0x00,0x00,0xc0,0x24]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 24c00000 <unknown>

cmphs   p0.b, p0/z, z0.b, z0.d
// CHECK-INST: cmphs p0.b, p0/z, z0.b, z0.d
// CHECK-ENCODING: [0x00,0xc0,0x00,0x24]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2400c000 <unknown>

cmphs   p0.h, p0/z, z0.h, z0.d
// CHECK-INST: cmphs p0.h, p0/z, z0.h, z0.d
// CHECK-ENCODING: [0x00,0xc0,0x40,0x24]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2440c000 <unknown>

cmphs   p0.s, p0/z, z0.s, z0.d
// CHECK-INST: cmphs p0.s, p0/z, z0.s, z0.d
// CHECK-ENCODING: [0x00,0xc0,0x80,0x24]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 2480c000 <unknown>

cmphs   p0.b, p0/z, z0.b, #0
// CHECK-INST: cmphs p0.b, p0/z, z0.b, #0
// CHECK-ENCODING: [0x00,0x00,0x20,0x24]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 24200000 <unknown>

cmphs   p0.h, p0/z, z0.h, #0
// CHECK-INST: cmphs p0.h, p0/z, z0.h, #0
// CHECK-ENCODING: [0x00,0x00,0x60,0x24]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 24600000 <unknown>

cmphs   p0.s, p0/z, z0.s, #0
// CHECK-INST: cmphs p0.s, p0/z, z0.s, #0
// CHECK-ENCODING: [0x00,0x00,0xa0,0x24]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 24a00000 <unknown>

cmphs   p0.d, p0/z, z0.d, #0
// CHECK-INST: cmphs p0.d, p0/z, z0.d, #0
// CHECK-ENCODING: [0x00,0x00,0xe0,0x24]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 24e00000 <unknown>

cmphs   p0.b, p0/z, z0.b, #127
// CHECK-INST: cmphs p0.b, p0/z, z0.b, #127
// CHECK-ENCODING: [0x00,0xc0,0x3f,0x24]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 243fc000 <unknown>

cmphs   p0.h, p0/z, z0.h, #127
// CHECK-INST: cmphs p0.h, p0/z, z0.h, #127
// CHECK-ENCODING: [0x00,0xc0,0x7f,0x24]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 247fc000 <unknown>

cmphs   p0.s, p0/z, z0.s, #127
// CHECK-INST: cmphs p0.s, p0/z, z0.s, #127
// CHECK-ENCODING: [0x00,0xc0,0xbf,0x24]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 24bfc000 <unknown>

cmphs   p0.d, p0/z, z0.d, #127
// CHECK-INST: cmphs p0.d, p0/z, z0.d, #127
// CHECK-ENCODING: [0x00,0xc0,0xff,0x24]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 24ffc000 <unknown>
