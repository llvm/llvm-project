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

uqincp  x0, p0.b
// CHECK-INST: uqincp  x0, p0.b
// CHECK-ENCODING: [0x00,0x8c,0x29,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 8c 29 25 <unknown>

uqincp  x0, p0.h
// CHECK-INST: uqincp  x0, p0.h
// CHECK-ENCODING: [0x00,0x8c,0x69,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 8c 69 25 <unknown>

uqincp  x0, p0.s
// CHECK-INST: uqincp  x0, p0.s
// CHECK-ENCODING: [0x00,0x8c,0xa9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 8c a9 25 <unknown>

uqincp  x0, p0.d
// CHECK-INST: uqincp  x0, p0.d
// CHECK-ENCODING: [0x00,0x8c,0xe9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 8c e9 25 <unknown>

uqincp  wzr, p15.b
// CHECK-INST: uqincp  wzr, p15.b
// CHECK-ENCODING: [0xff,0x89,0x29,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 89 29 25 <unknown>

uqincp  wzr, p15.h
// CHECK-INST: uqincp  wzr, p15.h
// CHECK-ENCODING: [0xff,0x89,0x69,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 89 69 25 <unknown>

uqincp  wzr, p15.s
// CHECK-INST: uqincp  wzr, p15.s
// CHECK-ENCODING: [0xff,0x89,0xa9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 89 a9 25 <unknown>

uqincp  wzr, p15.d
// CHECK-INST: uqincp  wzr, p15.d
// CHECK-ENCODING: [0xff,0x89,0xe9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 89 e9 25 <unknown>

uqincp  z0.h, p0
// CHECK-INST: uqincp  z0.h, p0.h
// CHECK-ENCODING: [0x00,0x80,0x69,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 80 69 25 <unknown>

uqincp  z0.h, p0.h
// CHECK-INST: uqincp  z0.h, p0.h
// CHECK-ENCODING: [0x00,0x80,0x69,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 80 69 25 <unknown>

uqincp  z0.s, p0
// CHECK-INST: uqincp  z0.s, p0.s
// CHECK-ENCODING: [0x00,0x80,0xa9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 80 a9 25 <unknown>

uqincp  z0.s, p0.s
// CHECK-INST: uqincp  z0.s, p0.s
// CHECK-ENCODING: [0x00,0x80,0xa9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 80 a9 25 <unknown>

uqincp  z0.d, p0
// CHECK-INST: uqincp  z0.d, p0.d
// CHECK-ENCODING: [0x00,0x80,0xe9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 80 e9 25 <unknown>

uqincp  z0.d, p0.d
// CHECK-INST: uqincp  z0.d, p0.d
// CHECK-ENCODING: [0x00,0x80,0xe9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 80 e9 25 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 bc 20 04 <unknown>

uqincp  z0.d, p0.d
// CHECK-INST: uqincp	z0.d, p0.d
// CHECK-ENCODING: [0x00,0x80,0xe9,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 80 e9 25 <unknown>
