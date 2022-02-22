// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+streaming-sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

sqdecp  x0, p0.b
// CHECK-INST: sqdecp x0, p0.b
// CHECK-ENCODING: [0x00,0x8c,0x2a,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 8c 2a 25 <unknown>

sqdecp  x0, p0.h
// CHECK-INST: sqdecp x0, p0.h
// CHECK-ENCODING: [0x00,0x8c,0x6a,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 8c 6a 25 <unknown>

sqdecp  x0, p0.s
// CHECK-INST: sqdecp x0, p0.s
// CHECK-ENCODING: [0x00,0x8c,0xaa,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 8c aa 25 <unknown>

sqdecp  x0, p0.d
// CHECK-INST: sqdecp x0, p0.d
// CHECK-ENCODING: [0x00,0x8c,0xea,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 8c ea 25 <unknown>

sqdecp  xzr, p15.b, wzr
// CHECK-INST: sqdecp xzr, p15.b, wzr
// CHECK-ENCODING: [0xff,0x89,0x2a,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 89 2a 25 <unknown>

sqdecp  xzr, p15.h, wzr
// CHECK-INST: sqdecp xzr, p15.h, wzr
// CHECK-ENCODING: [0xff,0x89,0x6a,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 89 6a 25 <unknown>

sqdecp  xzr, p15.s, wzr
// CHECK-INST: sqdecp xzr, p15.s, wzr
// CHECK-ENCODING: [0xff,0x89,0xaa,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 89 aa 25 <unknown>

sqdecp  xzr, p15.d, wzr
// CHECK-INST: sqdecp xzr, p15.d, wzr
// CHECK-ENCODING: [0xff,0x89,0xea,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 89 ea 25 <unknown>

sqdecp  z0.h, p0
// CHECK-INST: sqdecp z0.h, p0.h
// CHECK-ENCODING: [0x00,0x80,0x6a,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 80 6a 25 <unknown>

sqdecp  z0.h, p0.h
// CHECK-INST: sqdecp z0.h, p0.h
// CHECK-ENCODING: [0x00,0x80,0x6a,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 80 6a 25 <unknown>

sqdecp  z0.s, p0
// CHECK-INST: sqdecp z0.s, p0.s
// CHECK-ENCODING: [0x00,0x80,0xaa,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 80 aa 25 <unknown>

sqdecp  z0.s, p0.s
// CHECK-INST: sqdecp z0.s, p0.s
// CHECK-ENCODING: [0x00,0x80,0xaa,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 80 aa 25 <unknown>

sqdecp  z0.d, p0
// CHECK-INST: sqdecp z0.d, p0.d
// CHECK-ENCODING: [0x00,0x80,0xea,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 80 ea 25 <unknown>

sqdecp  z0.d, p0.d
// CHECK-INST: sqdecp z0.d, p0.d
// CHECK-ENCODING: [0x00,0x80,0xea,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 80 ea 25 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e0 bc 20 04 <unknown>

sqdecp  z0.d, p0.d
// CHECK-INST: sqdecp	z0.d, p0.d
// CHECK-ENCODING: [0x00,0x80,0xea,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 80 ea 25 <unknown>
