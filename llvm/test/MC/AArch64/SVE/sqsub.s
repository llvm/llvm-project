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


sqsub     z0.b, z0.b, z0.b
// CHECK-INST: sqsub z0.b, z0.b, z0.b
// CHECK-ENCODING: [0x00,0x18,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 18 20 04 <unknown>

sqsub     z0.h, z0.h, z0.h
// CHECK-INST: sqsub z0.h, z0.h, z0.h
// CHECK-ENCODING: [0x00,0x18,0x60,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 18 60 04 <unknown>

sqsub     z0.s, z0.s, z0.s
// CHECK-INST: sqsub z0.s, z0.s, z0.s
// CHECK-ENCODING: [0x00,0x18,0xa0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 18 a0 04 <unknown>

sqsub     z0.d, z0.d, z0.d
// CHECK-INST: sqsub z0.d, z0.d, z0.d
// CHECK-ENCODING: [0x00,0x18,0xe0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 18 e0 04 <unknown>

sqsub     z0.b, z0.b, #0
// CHECK-INST: sqsub z0.b, z0.b, #0
// CHECK-ENCODING: [0x00,0xc0,0x26,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 c0 26 25 <unknown>

sqsub     z31.b, z31.b, #255
// CHECK-INST: sqsub z31.b, z31.b, #255
// CHECK-ENCODING: [0xff,0xdf,0x26,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff df 26 25 <unknown>

sqsub     z0.h, z0.h, #0
// CHECK-INST: sqsub z0.h, z0.h, #0
// CHECK-ENCODING: [0x00,0xc0,0x66,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 c0 66 25 <unknown>

sqsub     z0.h, z0.h, #0, lsl #8
// CHECK-INST: sqsub z0.h, z0.h, #0, lsl #8
// CHECK-ENCODING: [0x00,0xe0,0x66,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 e0 66 25 <unknown>

sqsub     z31.h, z31.h, #255, lsl #8
// CHECK-INST: sqsub z31.h, z31.h, #65280
// CHECK-ENCODING: [0xff,0xff,0x66,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff ff 66 25 <unknown>

sqsub     z31.h, z31.h, #65280
// CHECK-INST: sqsub z31.h, z31.h, #65280
// CHECK-ENCODING: [0xff,0xff,0x66,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff ff 66 25 <unknown>

sqsub     z0.s, z0.s, #0
// CHECK-INST: sqsub z0.s, z0.s, #0
// CHECK-ENCODING: [0x00,0xc0,0xa6,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 c0 a6 25 <unknown>

sqsub     z0.s, z0.s, #0, lsl #8
// CHECK-INST: sqsub z0.s, z0.s, #0, lsl #8
// CHECK-ENCODING: [0x00,0xe0,0xa6,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 e0 a6 25 <unknown>

sqsub     z31.s, z31.s, #255, lsl #8
// CHECK-INST: sqsub z31.s, z31.s, #65280
// CHECK-ENCODING: [0xff,0xff,0xa6,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff ff a6 25 <unknown>

sqsub     z31.s, z31.s, #65280
// CHECK-INST: sqsub z31.s, z31.s, #65280
// CHECK-ENCODING: [0xff,0xff,0xa6,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff ff a6 25 <unknown>

sqsub     z0.d, z0.d, #0
// CHECK-INST: sqsub z0.d, z0.d, #0
// CHECK-ENCODING: [0x00,0xc0,0xe6,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 c0 e6 25 <unknown>

sqsub     z0.d, z0.d, #0, lsl #8
// CHECK-INST: sqsub z0.d, z0.d, #0, lsl #8
// CHECK-ENCODING: [0x00,0xe0,0xe6,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 e0 e6 25 <unknown>

sqsub     z31.d, z31.d, #255, lsl #8
// CHECK-INST: sqsub z31.d, z31.d, #65280
// CHECK-ENCODING: [0xff,0xff,0xe6,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff ff e6 25 <unknown>

sqsub     z31.d, z31.d, #65280
// CHECK-INST: sqsub z31.d, z31.d, #65280
// CHECK-ENCODING: [0xff,0xff,0xe6,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff ff e6 25 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z31, z6
// CHECK-INST: movprfx	z31, z6
// CHECK-ENCODING: [0xdf,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: df bc 20 04 <unknown>

sqsub     z31.d, z31.d, #65280
// CHECK-INST: sqsub	z31.d, z31.d, #65280
// CHECK-ENCODING: [0xff,0xff,0xe6,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff ff e6 25 <unknown>
