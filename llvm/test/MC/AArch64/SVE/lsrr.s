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

lsrr    z0.b, p0/m, z0.b, z0.b
// CHECK-INST: lsrr	z0.b, p0/m, z0.b, z0.b
// CHECK-ENCODING: [0x00,0x80,0x15,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 00 80 15 04 <unknown>

lsrr    z0.h, p0/m, z0.h, z0.h
// CHECK-INST: lsrr	z0.h, p0/m, z0.h, z0.h
// CHECK-ENCODING: [0x00,0x80,0x55,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 00 80 55 04 <unknown>

lsrr    z0.s, p0/m, z0.s, z0.s
// CHECK-INST: lsrr	z0.s, p0/m, z0.s, z0.s
// CHECK-ENCODING: [0x00,0x80,0x95,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 00 80 95 04 <unknown>

lsrr    z0.d, p0/m, z0.d, z0.d
// CHECK-INST: lsrr	z0.d, p0/m, z0.d, z0.d
// CHECK-ENCODING: [0x00,0x80,0xd5,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 00 80 d5 04 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z5.d, p0/z, z7.d
// CHECK-INST: movprfx	z5.d, p0/z, z7.d
// CHECK-ENCODING: [0xe5,0x20,0xd0,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e5 20 d0 04 <unknown>

lsrr    z5.d, p0/m, z5.d, z0.d
// CHECK-INST: lsrr	z5.d, p0/m, z5.d, z0.d
// CHECK-ENCODING: [0x05,0x80,0xd5,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 05 80 d5 04 <unknown>

movprfx z5, z7
// CHECK-INST: movprfx	z5, z7
// CHECK-ENCODING: [0xe5,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: e5 bc 20 04 <unknown>

lsrr    z5.d, p0/m, z5.d, z0.d
// CHECK-INST: lsrr	z5.d, p0/m, z5.d, z0.d
// CHECK-ENCODING: [0x05,0x80,0xd5,0x04]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: 05 80 d5 04 <unknown>
