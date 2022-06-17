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


uqsub     z0.b, z0.b, z0.b
// CHECK-INST: uqsub z0.b, z0.b, z0.b
// CHECK-ENCODING: [0x00,0x1c,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 1c 20 04 <unknown>

uqsub     z0.h, z0.h, z0.h
// CHECK-INST: uqsub z0.h, z0.h, z0.h
// CHECK-ENCODING: [0x00,0x1c,0x60,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 1c 60 04 <unknown>

uqsub     z0.s, z0.s, z0.s
// CHECK-INST: uqsub z0.s, z0.s, z0.s
// CHECK-ENCODING: [0x00,0x1c,0xa0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 1c a0 04 <unknown>

uqsub     z0.d, z0.d, z0.d
// CHECK-INST: uqsub z0.d, z0.d, z0.d
// CHECK-ENCODING: [0x00,0x1c,0xe0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 1c e0 04 <unknown>

uqsub     z0.b, z0.b, #0
// CHECK-INST: uqsub z0.b, z0.b, #0
// CHECK-ENCODING: [0x00,0xc0,0x27,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 c0 27 25 <unknown>

uqsub     z31.b, z31.b, #255
// CHECK-INST: uqsub z31.b, z31.b, #255
// CHECK-ENCODING: [0xff,0xdf,0x27,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff df 27 25 <unknown>

uqsub     z0.h, z0.h, #0
// CHECK-INST: uqsub z0.h, z0.h, #0
// CHECK-ENCODING: [0x00,0xc0,0x67,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 c0 67 25 <unknown>

uqsub     z0.h, z0.h, #0, lsl #8
// CHECK-INST: uqsub z0.h, z0.h, #0, lsl #8
// CHECK-ENCODING: [0x00,0xe0,0x67,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 e0 67 25 <unknown>

uqsub     z31.h, z31.h, #255, lsl #8
// CHECK-INST: uqsub z31.h, z31.h, #65280
// CHECK-ENCODING: [0xff,0xff,0x67,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff ff 67 25 <unknown>

uqsub     z31.h, z31.h, #65280
// CHECK-INST: uqsub z31.h, z31.h, #65280
// CHECK-ENCODING: [0xff,0xff,0x67,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff ff 67 25 <unknown>

uqsub     z0.s, z0.s, #0
// CHECK-INST: uqsub z0.s, z0.s, #0
// CHECK-ENCODING: [0x00,0xc0,0xa7,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 c0 a7 25 <unknown>

uqsub     z0.s, z0.s, #0, lsl #8
// CHECK-INST: uqsub z0.s, z0.s, #0, lsl #8
// CHECK-ENCODING: [0x00,0xe0,0xa7,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 e0 a7 25 <unknown>

uqsub     z31.s, z31.s, #255, lsl #8
// CHECK-INST: uqsub z31.s, z31.s, #65280
// CHECK-ENCODING: [0xff,0xff,0xa7,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff ff a7 25 <unknown>

uqsub     z31.s, z31.s, #65280
// CHECK-INST: uqsub z31.s, z31.s, #65280
// CHECK-ENCODING: [0xff,0xff,0xa7,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff ff a7 25 <unknown>

uqsub     z0.d, z0.d, #0
// CHECK-INST: uqsub z0.d, z0.d, #0
// CHECK-ENCODING: [0x00,0xc0,0xe7,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 c0 e7 25 <unknown>

uqsub     z0.d, z0.d, #0, lsl #8
// CHECK-INST: uqsub z0.d, z0.d, #0, lsl #8
// CHECK-ENCODING: [0x00,0xe0,0xe7,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 e0 e7 25 <unknown>

uqsub     z31.d, z31.d, #255, lsl #8
// CHECK-INST: uqsub z31.d, z31.d, #65280
// CHECK-ENCODING: [0xff,0xff,0xe7,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff ff e7 25 <unknown>

uqsub     z31.d, z31.d, #65280
// CHECK-INST: uqsub z31.d, z31.d, #65280
// CHECK-ENCODING: [0xff,0xff,0xe7,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff ff e7 25 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z31, z6
// CHECK-INST: movprfx	z31, z6
// CHECK-ENCODING: [0xdf,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: df bc 20 04 <unknown>

uqsub     z31.d, z31.d, #65280
// CHECK-INST: uqsub	z31.d, z31.d, #65280
// CHECK-ENCODING: [0xff,0xff,0xe7,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff ff e7 25 <unknown>
