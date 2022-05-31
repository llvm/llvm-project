// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN


uqadd     z0.b, z0.b, z0.b
// CHECK-INST: uqadd z0.b, z0.b, z0.b
// CHECK-ENCODING: [0x00,0x14,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 14 20 04 <unknown>

uqadd     z0.h, z0.h, z0.h
// CHECK-INST: uqadd z0.h, z0.h, z0.h
// CHECK-ENCODING: [0x00,0x14,0x60,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 14 60 04 <unknown>

uqadd     z0.s, z0.s, z0.s
// CHECK-INST: uqadd z0.s, z0.s, z0.s
// CHECK-ENCODING: [0x00,0x14,0xa0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 14 a0 04 <unknown>

uqadd     z0.d, z0.d, z0.d
// CHECK-INST: uqadd z0.d, z0.d, z0.d
// CHECK-ENCODING: [0x00,0x14,0xe0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 14 e0 04 <unknown>

uqadd     z0.b, z0.b, #0
// CHECK-INST: uqadd z0.b, z0.b, #0
// CHECK-ENCODING: [0x00,0xc0,0x25,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 c0 25 25 <unknown>

uqadd     z31.b, z31.b, #255
// CHECK-INST: uqadd z31.b, z31.b, #255
// CHECK-ENCODING: [0xff,0xdf,0x25,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff df 25 25 <unknown>

uqadd     z0.h, z0.h, #0
// CHECK-INST: uqadd z0.h, z0.h, #0
// CHECK-ENCODING: [0x00,0xc0,0x65,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 c0 65 25 <unknown>

uqadd     z0.h, z0.h, #0, lsl #8
// CHECK-INST: uqadd z0.h, z0.h, #0, lsl #8
// CHECK-ENCODING: [0x00,0xe0,0x65,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 e0 65 25 <unknown>

uqadd     z31.h, z31.h, #255, lsl #8
// CHECK-INST: uqadd z31.h, z31.h, #65280
// CHECK-ENCODING: [0xff,0xff,0x65,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff ff 65 25 <unknown>

uqadd     z31.h, z31.h, #65280
// CHECK-INST: uqadd z31.h, z31.h, #65280
// CHECK-ENCODING: [0xff,0xff,0x65,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff ff 65 25 <unknown>

uqadd     z0.s, z0.s, #0
// CHECK-INST: uqadd z0.s, z0.s, #0
// CHECK-ENCODING: [0x00,0xc0,0xa5,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 c0 a5 25 <unknown>

uqadd     z0.s, z0.s, #0, lsl #8
// CHECK-INST: uqadd z0.s, z0.s, #0, lsl #8
// CHECK-ENCODING: [0x00,0xe0,0xa5,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 e0 a5 25 <unknown>

uqadd     z31.s, z31.s, #255, lsl #8
// CHECK-INST: uqadd z31.s, z31.s, #65280
// CHECK-ENCODING: [0xff,0xff,0xa5,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff ff a5 25 <unknown>

uqadd     z31.s, z31.s, #65280
// CHECK-INST: uqadd z31.s, z31.s, #65280
// CHECK-ENCODING: [0xff,0xff,0xa5,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff ff a5 25 <unknown>

uqadd     z0.d, z0.d, #0
// CHECK-INST: uqadd z0.d, z0.d, #0
// CHECK-ENCODING: [0x00,0xc0,0xe5,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 c0 e5 25 <unknown>

uqadd     z0.d, z0.d, #0, lsl #8
// CHECK-INST: uqadd z0.d, z0.d, #0, lsl #8
// CHECK-ENCODING: [0x00,0xe0,0xe5,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 e0 e5 25 <unknown>

uqadd     z31.d, z31.d, #255, lsl #8
// CHECK-INST: uqadd z31.d, z31.d, #65280
// CHECK-ENCODING: [0xff,0xff,0xe5,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff ff e5 25 <unknown>

uqadd     z31.d, z31.d, #65280
// CHECK-INST: uqadd z31.d, z31.d, #65280
// CHECK-ENCODING: [0xff,0xff,0xe5,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff ff e5 25 <unknown>


// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z31, z6
// CHECK-INST: movprfx	z31, z6
// CHECK-ENCODING: [0xdf,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: df bc 20 04 <unknown>

uqadd     z31.d, z31.d, #65280
// CHECK-INST: uqadd	z31.d, z31.d, #65280
// CHECK-ENCODING: [0xff,0xff,0xe5,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff ff e5 25 <unknown>
