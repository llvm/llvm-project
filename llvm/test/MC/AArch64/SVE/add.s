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

add     z31.s, z31.s, z31.s
// CHECK-INST: add     z31.s, z31.s, z31.s
// CHECK-ENCODING: [0xff,0x03,0xbf,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 03 bf 04 <unknown>

add     z23.d, z13.d, z8.d
// CHECK-INST: add     z23.d, z13.d, z8.d
// CHECK-ENCODING: [0xb7,0x01,0xe8,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: b7 01 e8 04 <unknown>

add     z23.b, p3/m, z23.b, z13.b
// CHECK-INST: add     z23.b, p3/m, z23.b, z13.b
// CHECK-ENCODING: [0xb7,0x0d,0x00,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: b7 0d 00 04 <unknown>

add     z0.s, z0.s, z0.s
// CHECK-INST: add     z0.s, z0.s, z0.s
// CHECK-ENCODING: [0x00,0x00,0xa0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 00 a0 04 <unknown>

add     z31.d, z31.d, z31.d
// CHECK-INST: add     z31.d, z31.d, z31.d
// CHECK-ENCODING: [0xff,0x03,0xff,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 03 ff 04 <unknown>

add     z21.b, z10.b, z21.b
// CHECK-INST: add     z21.b, z10.b, z21.b
// CHECK-ENCODING: [0x55,0x01,0x35,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 55 01 35 04 <unknown>

add     z31.b, z31.b, z31.b
// CHECK-INST: add     z31.b, z31.b, z31.b
// CHECK-ENCODING: [0xff,0x03,0x3f,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 03 3f 04 <unknown>

add     z0.h, p0/m, z0.h, z0.h
// CHECK-INST: add     z0.h, p0/m, z0.h, z0.h
// CHECK-ENCODING: [0x00,0x00,0x40,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 00 40 04 <unknown>

add     z0.h, z0.h, z0.h
// CHECK-INST: add     z0.h, z0.h, z0.h
// CHECK-ENCODING: [0x00,0x00,0x60,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 00 60 04 <unknown>

add     z0.b, p0/m, z0.b, z0.b
// CHECK-INST: add     z0.b, p0/m, z0.b, z0.b
// CHECK-ENCODING: [0x00,0x00,0x00,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 00 00 04 <unknown>

add     z0.s, p0/m, z0.s, z0.s
// CHECK-INST: add     z0.s, p0/m, z0.s, z0.s
// CHECK-ENCODING: [0x00,0x00,0x80,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 00 80 04 <unknown>

add     z23.b, z13.b, z8.b
// CHECK-INST: add     z23.b, z13.b, z8.b
// CHECK-ENCODING: [0xb7,0x01,0x28,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: b7 01 28 04 <unknown>

add     z0.d, z0.d, z0.d
// CHECK-INST: add     z0.d, z0.d, z0.d
// CHECK-ENCODING: [0x00,0x00,0xe0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 00 e0 04 <unknown>

add     z0.d, p0/m, z0.d, z0.d
// CHECK-INST: add     z0.d, p0/m, z0.d, z0.d
// CHECK-ENCODING: [0x00,0x00,0xc0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 00 c0 04 <unknown>

add     z31.h, z31.h, z31.h
// CHECK-INST: add     z31.h, z31.h, z31.h
// CHECK-ENCODING: [0xff,0x03,0x7f,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 03 7f 04 <unknown>

add     z0.b, z0.b, z0.b
// CHECK-INST: add     z0.b, z0.b, z0.b
// CHECK-ENCODING: [0x00,0x00,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 00 20 04 <unknown>

add     z21.d, z10.d, z21.d
// CHECK-INST: add     z21.d, z10.d, z21.d
// CHECK-ENCODING: [0x55,0x01,0xf5,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 55 01 f5 04 <unknown>

add     z23.h, p3/m, z23.h, z13.h
// CHECK-INST: add     z23.h, p3/m, z23.h, z13.h
// CHECK-ENCODING: [0xb7,0x0d,0x40,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: b7 0d 40 04 <unknown>

add     z23.s, p3/m, z23.s, z13.s
// CHECK-INST: add     z23.s, p3/m, z23.s, z13.s
// CHECK-ENCODING: [0xb7,0x0d,0x80,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: b7 0d 80 04 <unknown>

add     z31.s, p7/m, z31.s, z31.s
// CHECK-INST: add     z31.s, p7/m, z31.s, z31.s
// CHECK-ENCODING: [0xff,0x1f,0x80,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 1f 80 04 <unknown>

add     z21.h, z10.h, z21.h
// CHECK-INST: add     z21.h, z10.h, z21.h
// CHECK-ENCODING: [0x55,0x01,0x75,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 55 01 75 04 <unknown>

add     z23.d, p3/m, z23.d, z13.d
// CHECK-INST: add     z23.d, p3/m, z23.d, z13.d
// CHECK-ENCODING: [0xb7,0x0d,0xc0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: b7 0d c0 04 <unknown>

add     z21.d, p5/m, z21.d, z10.d
// CHECK-INST: add     z21.d, p5/m, z21.d, z10.d
// CHECK-ENCODING: [0x55,0x15,0xc0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 55 15 c0 04 <unknown>

add     z21.b, p5/m, z21.b, z10.b
// CHECK-INST: add     z21.b, p5/m, z21.b, z10.b
// CHECK-ENCODING: [0x55,0x15,0x00,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 55 15 00 04 <unknown>

add     z21.s, z10.s, z21.s
// CHECK-INST: add     z21.s, z10.s, z21.s
// CHECK-ENCODING: [0x55,0x01,0xb5,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 55 01 b5 04 <unknown>

add     z21.h, p5/m, z21.h, z10.h
// CHECK-INST: add     z21.h, p5/m, z21.h, z10.h
// CHECK-ENCODING: [0x55,0x15,0x40,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 55 15 40 04 <unknown>

add     z31.h, p7/m, z31.h, z31.h
// CHECK-INST: add     z31.h, p7/m, z31.h, z31.h
// CHECK-ENCODING: [0xff,0x1f,0x40,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 1f 40 04 <unknown>

add     z23.h, z13.h, z8.h
// CHECK-INST: add     z23.h, z13.h, z8.h
// CHECK-ENCODING: [0xb7,0x01,0x68,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: b7 01 68 04 <unknown>

add     z31.d, p7/m, z31.d, z31.d
// CHECK-INST: add     z31.d, p7/m, z31.d, z31.d
// CHECK-ENCODING: [0xff,0x1f,0xc0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 1f c0 04 <unknown>

add     z21.s, p5/m, z21.s, z10.s
// CHECK-INST: add     z21.s, p5/m, z21.s, z10.s
// CHECK-ENCODING: [0x55,0x15,0x80,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 55 15 80 04 <unknown>

add     z31.b, p7/m, z31.b, z31.b
// CHECK-INST: add     z31.b, p7/m, z31.b, z31.b
// CHECK-ENCODING: [0xff,0x1f,0x00,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 1f 00 04 <unknown>

add     z23.s, z13.s, z8.s
// CHECK-INST: add     z23.s, z13.s, z8.s
// CHECK-ENCODING: [0xb7,0x01,0xa8,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: b7 01 a8 04 <unknown>

add     z0.b, z0.b, #0
// CHECK-INST: add     z0.b, z0.b, #0
// CHECK-ENCODING: [0x00,0xc0,0x20,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 c0 20 25 <unknown>

add     z31.b, z31.b, #255
// CHECK-INST: add     z31.b, z31.b, #255
// CHECK-ENCODING: [0xff,0xdf,0x20,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff df 20 25 <unknown>

add     z0.h, z0.h, #0
// CHECK-INST: add     z0.h, z0.h, #0
// CHECK-ENCODING: [0x00,0xc0,0x60,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 c0 60 25 <unknown>

add     z0.h, z0.h, #0, lsl #8
// CHECK-INST: add     z0.h, z0.h, #0, lsl #8
// CHECK-ENCODING: [0x00,0xe0,0x60,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 e0 60 25 <unknown>

add     z31.h, z31.h, #255, lsl #8
// CHECK-INST: add     z31.h, z31.h, #65280
// CHECK-ENCODING: [0xff,0xff,0x60,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff ff 60 25 <unknown>

add     z31.h, z31.h, #65280
// CHECK-INST: add     z31.h, z31.h, #65280
// CHECK-ENCODING: [0xff,0xff,0x60,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff ff 60 25 <unknown>

add     z0.s, z0.s, #0
// CHECK-INST: add     z0.s, z0.s, #0
// CHECK-ENCODING: [0x00,0xc0,0xa0,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 c0 a0 25 <unknown>

add     z0.s, z0.s, #0, lsl #8
// CHECK-INST: add     z0.s, z0.s, #0, lsl #8
// CHECK-ENCODING: [0x00,0xe0,0xa0,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 e0 a0 25 <unknown>

add     z31.s, z31.s, #255, lsl #8
// CHECK-INST: add     z31.s, z31.s, #65280
// CHECK-ENCODING: [0xff,0xff,0xa0,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff ff a0 25 <unknown>

add     z31.s, z31.s, #65280
// CHECK-INST: add     z31.s, z31.s, #65280
// CHECK-ENCODING: [0xff,0xff,0xa0,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff ff a0 25 <unknown>

add     z0.d, z0.d, #0
// CHECK-INST: add     z0.d, z0.d, #0
// CHECK-ENCODING: [0x00,0xc0,0xe0,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 c0 e0 25 <unknown>

add     z0.d, z0.d, #0, lsl #8
// CHECK-INST: add     z0.d, z0.d, #0, lsl #8
// CHECK-ENCODING: [0x00,0xe0,0xe0,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 e0 e0 25 <unknown>

add     z31.d, z31.d, #255, lsl #8
// CHECK-INST: add     z31.d, z31.d, #65280
// CHECK-ENCODING: [0xff,0xff,0xe0,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff ff e0 25 <unknown>

add     z31.d, z31.d, #65280
// CHECK-INST: add     z31.d, z31.d, #65280
// CHECK-ENCODING: [0xff,0xff,0xe0,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff ff e0 25 <unknown>



// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z4.b, p7/z, z6.b
// CHECK-INST: movprfx	z4.b, p7/z, z6.b
// CHECK-ENCODING: [0xc4,0x3c,0x10,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: c4 3c 10 04 <unknown>

add     z4.b, p7/m, z4.b, z31.b
// CHECK-INST: add	z4.b, p7/m, z4.b, z31.b
// CHECK-ENCODING: [0xe4,0x1f,0x00,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e4 1f 00 04 <unknown>

movprfx z4, z6
// CHECK-INST: movprfx	z4, z6
// CHECK-ENCODING: [0xc4,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: c4 bc 20 04 <unknown>

add     z4.b, p7/m, z4.b, z31.b
// CHECK-INST: add	z4.b, p7/m, z4.b, z31.b
// CHECK-ENCODING: [0xe4,0x1f,0x00,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: e4 1f 00 04 <unknown>

movprfx z31, z6
// CHECK-INST: movprfx	z31, z6
// CHECK-ENCODING: [0xdf,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: df bc 20 04 <unknown>

add     z31.d, z31.d, #65280
// CHECK-INST: add	z31.d, z31.d, #65280
// CHECK-ENCODING: [0xff,0xff,0xe0,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff ff e0 25 <unknown>
