// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:        | llvm-objdump --no-print-imm-hex -d --mattr=+sve2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2 < %s \
// RUN:   | llvm-objdump --no-print-imm-hex -d --mattr=-sve2 - | FileCheck %s --check-prefix=CHECK-UNKNOWN

sqshlu z0.b, p0/m, z0.b, #0
// CHECK-INST: sqshlu z0.b, p0/m, z0.b, #0
// CHECK-ENCODING: [0x00,0x81,0x0f,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 040f8100 <unknown>

sqshlu z31.b, p0/m, z31.b, #7
// CHECK-INST: sqshlu z31.b, p0/m, z31.b, #7
// CHECK-ENCODING: [0xff,0x81,0x0f,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 040f81ff <unknown>

sqshlu z0.h, p0/m, z0.h, #0
// CHECK-INST: sqshlu z0.h, p0/m, z0.h, #0
// CHECK-ENCODING: [0x00,0x82,0x0f,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 040f8200 <unknown>

sqshlu z31.h, p0/m, z31.h, #15
// CHECK-INST: sqshlu z31.h, p0/m, z31.h, #15
// CHECK-ENCODING: [0xff,0x83,0x0f,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 040f83ff <unknown>

sqshlu z0.s, p0/m, z0.s, #0
// CHECK-INST: sqshlu z0.s, p0/m, z0.s, #0
// CHECK-ENCODING: [0x00,0x80,0x4f,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 044f8000 <unknown>

sqshlu z31.s, p0/m, z31.s, #31
// CHECK-INST: sqshlu z31.s, p0/m, z31.s, #31
// CHECK-ENCODING: [0xff,0x83,0x4f,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 044f83ff <unknown>

sqshlu z0.d, p0/m, z0.d, #0
// CHECK-INST: sqshlu z0.d, p0/m, z0.d, #0
// CHECK-ENCODING: [0x00,0x80,0x8f,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 048f8000 <unknown>

sqshlu z31.d, p0/m, z31.d, #63
// CHECK-INST: sqshlu z31.d, p0/m, z31.d, #63
// CHECK-ENCODING: [0xff,0x83,0xcf,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 04cf83ff <unknown>

// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z31.d, p0/z, z6.d
// CHECK-INST: movprfx z31.d, p0/z, z6.d
// CHECK-ENCODING: [0xdf,0x20,0xd0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04d020df <unknown>

sqshlu z31.d, p0/m, z31.d, #63
// CHECK-INST: sqshlu z31.d, p0/m, z31.d, #63
// CHECK-ENCODING: [0xff,0x83,0xcf,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 04cf83ff <unknown>

movprfx z31, z6
// CHECK-INST: movprfx z31, z6
// CHECK-ENCODING: [0xdf,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420bcdf <unknown>

sqshlu z31.d, p0/m, z31.d, #63
// CHECK-INST: sqshlu z31.d, p0/m, z31.d, #63
// CHECK-ENCODING: [0xff,0x83,0xcf,0x04]
// CHECK-ERROR: instruction requires: sve2 or sme
// CHECK-UNKNOWN: 04cf83ff <unknown>
