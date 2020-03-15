// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

sqincp  x0, p0.b
// CHECK-INST: sqincp  x0, p0.b
// CHECK-ENCODING: [0x00,0x8c,0x28,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 8c 28 25 <unknown>

sqincp  x0, p0.h
// CHECK-INST: sqincp  x0, p0.h
// CHECK-ENCODING: [0x00,0x8c,0x68,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 8c 68 25 <unknown>

sqincp  x0, p0.s
// CHECK-INST: sqincp  x0, p0.s
// CHECK-ENCODING: [0x00,0x8c,0xa8,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 8c a8 25 <unknown>

sqincp  x0, p0.d
// CHECK-INST: sqincp  x0, p0.d
// CHECK-ENCODING: [0x00,0x8c,0xe8,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 8c e8 25 <unknown>

sqincp  xzr, p15.b, wzr
// CHECK-INST: sqincp  xzr, p15.b, wzr
// CHECK-ENCODING: [0xff,0x89,0x28,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 89 28 25 <unknown>

sqincp  xzr, p15.h, wzr
// CHECK-INST: sqincp  xzr, p15.h, wzr
// CHECK-ENCODING: [0xff,0x89,0x68,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 89 68 25 <unknown>

sqincp  xzr, p15.s, wzr
// CHECK-INST: sqincp  xzr, p15.s, wzr
// CHECK-ENCODING: [0xff,0x89,0xa8,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 89 a8 25 <unknown>

sqincp  xzr, p15.d, wzr
// CHECK-INST: sqincp  xzr, p15.d, wzr
// CHECK-ENCODING: [0xff,0x89,0xe8,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ff 89 e8 25 <unknown>

sqincp  z0.h, p0
// CHECK-INST: sqincp  z0.h, p0.h
// CHECK-ENCODING: [0x00,0x80,0x68,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 80 68 25 <unknown>

sqincp  z0.h, p0.h
// CHECK-INST: sqincp  z0.h, p0.h
// CHECK-ENCODING: [0x00,0x80,0x68,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 80 68 25 <unknown>

sqincp  z0.s, p0
// CHECK-INST: sqincp  z0.s, p0.s
// CHECK-ENCODING: [0x00,0x80,0xa8,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 80 a8 25 <unknown>

sqincp  z0.s, p0.s
// CHECK-INST: sqincp  z0.s, p0.s
// CHECK-ENCODING: [0x00,0x80,0xa8,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 80 a8 25 <unknown>

sqincp  z0.d, p0
// CHECK-INST: sqincp  z0.d, p0.d
// CHECK-ENCODING: [0x00,0x80,0xe8,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 80 e8 25 <unknown>

sqincp  z0.d, p0.d
// CHECK-INST: sqincp  z0.d, p0.d
// CHECK-ENCODING: [0x00,0x80,0xe8,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 80 e8 25 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z0, z7
// CHECK-INST: movprfx	z0, z7
// CHECK-ENCODING: [0xe0,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: e0 bc 20 04 <unknown>

sqincp  z0.d, p0.d
// CHECK-INST: sqincp	z0.d, p0.d
// CHECK-ENCODING: [0x00,0x80,0xe8,0x25]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 80 e8 25 <unknown>
