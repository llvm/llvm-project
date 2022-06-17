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

sub     z0.h, z0.h, z0.h
// CHECK-INST: sub     z0.h, z0.h, z0.h
// CHECK-ENCODING: [0x00,0x04,0x60,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 04 60 04 <unknown>

sub     z21.b, z10.b, z21.b
// CHECK-INST: sub     z21.b, z10.b, z21.b
// CHECK-ENCODING: [0x55,0x05,0x35,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 55 05 35 04 <unknown>

sub     z31.d, p7/m, z31.d, z31.d
// CHECK-INST: sub     z31.d, p7/m, z31.d, z31.d
// CHECK-ENCODING: [0xff,0x1f,0xc1,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 1f c1 04 <unknown>

sub     z23.h, p3/m, z23.h, z13.h
// CHECK-INST: sub     z23.h, p3/m, z23.h, z13.h
// CHECK-ENCODING: [0xb7,0x0d,0x41,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: b7 0d 41 04 <unknown>

sub     z31.h, z31.h, z31.h
// CHECK-INST: sub     z31.h, z31.h, z31.h
// CHECK-ENCODING: [0xff,0x07,0x7f,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 07 7f 04 <unknown>

sub     z21.h, z10.h, z21.h
// CHECK-INST: sub     z21.h, z10.h, z21.h
// CHECK-ENCODING: [0x55,0x05,0x75,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 55 05 75 04 <unknown>

sub     z31.b, z31.b, z31.b
// CHECK-INST: sub     z31.b, z31.b, z31.b
// CHECK-ENCODING: [0xff,0x07,0x3f,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 07 3f 04 <unknown>

sub     z0.s, z0.s, z0.s
// CHECK-INST: sub     z0.s, z0.s, z0.s
// CHECK-ENCODING: [0x00,0x04,0xa0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 04 a0 04 <unknown>

sub     z23.s, p3/m, z23.s, z13.s
// CHECK-INST: sub     z23.s, p3/m, z23.s, z13.s
// CHECK-ENCODING: [0xb7,0x0d,0x81,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: b7 0d 81 04 <unknown>

sub     z23.b, z13.b, z8.b
// CHECK-INST: sub     z23.b, z13.b, z8.b
// CHECK-ENCODING: [0xb7,0x05,0x28,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: b7 05 28 04 <unknown>

sub     z21.d, z10.d, z21.d
// CHECK-INST: sub     z21.d, z10.d, z21.d
// CHECK-ENCODING: [0x55,0x05,0xf5,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 55 05 f5 04 <unknown>

sub     z21.s, z10.s, z21.s
// CHECK-INST: sub     z21.s, z10.s, z21.s
// CHECK-ENCODING: [0x55,0x05,0xb5,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 55 05 b5 04 <unknown>

sub     z21.s, p5/m, z21.s, z10.s
// CHECK-INST: sub     z21.s, p5/m, z21.s, z10.s
// CHECK-ENCODING: [0x55,0x15,0x81,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 55 15 81 04 <unknown>

sub     z31.s, p7/m, z31.s, z31.s
// CHECK-INST: sub     z31.s, p7/m, z31.s, z31.s
// CHECK-ENCODING: [0xff,0x1f,0x81,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 1f 81 04 <unknown>

sub     z0.d, p0/m, z0.d, z0.d
// CHECK-INST: sub     z0.d, p0/m, z0.d, z0.d
// CHECK-ENCODING: [0x00,0x00,0xc1,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 00 c1 04 <unknown>

sub     z0.b, z0.b, z0.b
// CHECK-INST: sub     z0.b, z0.b, z0.b
// CHECK-ENCODING: [0x00,0x04,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 04 20 04 <unknown>

sub     z23.d, z13.d, z8.d
// CHECK-INST: sub     z23.d, z13.d, z8.d
// CHECK-ENCODING: [0xb7,0x05,0xe8,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: b7 05 e8 04 <unknown>

sub     z23.d, p3/m, z23.d, z13.d
// CHECK-INST: sub     z23.d, p3/m, z23.d, z13.d
// CHECK-ENCODING: [0xb7,0x0d,0xc1,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: b7 0d c1 04 <unknown>

sub     z23.s, z13.s, z8.s
// CHECK-INST: sub     z23.s, z13.s, z8.s
// CHECK-ENCODING: [0xb7,0x05,0xa8,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: b7 05 a8 04 <unknown>

sub     z31.b, p7/m, z31.b, z31.b
// CHECK-INST: sub     z31.b, p7/m, z31.b, z31.b
// CHECK-ENCODING: [0xff,0x1f,0x01,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 1f 01 04 <unknown>

sub     z0.h, p0/m, z0.h, z0.h
// CHECK-INST: sub     z0.h, p0/m, z0.h, z0.h
// CHECK-ENCODING: [0x00,0x00,0x41,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 00 41 04 <unknown>

sub     z31.d, z31.d, z31.d
// CHECK-INST: sub     z31.d, z31.d, z31.d
// CHECK-ENCODING: [0xff,0x07,0xff,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 07 ff 04 <unknown>

sub     z31.h, p7/m, z31.h, z31.h
// CHECK-INST: sub     z31.h, p7/m, z31.h, z31.h
// CHECK-ENCODING: [0xff,0x1f,0x41,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 1f 41 04 <unknown>

sub     z23.h, z13.h, z8.h
// CHECK-INST: sub     z23.h, z13.h, z8.h
// CHECK-ENCODING: [0xb7,0x05,0x68,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: b7 05 68 04 <unknown>

sub     z21.b, p5/m, z21.b, z10.b
// CHECK-INST: sub     z21.b, p5/m, z21.b, z10.b
// CHECK-ENCODING: [0x55,0x15,0x01,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 55 15 01 04 <unknown>

sub     z21.d, p5/m, z21.d, z10.d
// CHECK-INST: sub     z21.d, p5/m, z21.d, z10.d
// CHECK-ENCODING: [0x55,0x15,0xc1,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 55 15 c1 04 <unknown>

sub     z0.d, z0.d, z0.d
// CHECK-INST: sub     z0.d, z0.d, z0.d
// CHECK-ENCODING: [0x00,0x04,0xe0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 04 e0 04 <unknown>

sub     z31.s, z31.s, z31.s
// CHECK-INST: sub     z31.s, z31.s, z31.s
// CHECK-ENCODING: [0xff,0x07,0xbf,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 07 bf 04 <unknown>

sub     z0.b, p0/m, z0.b, z0.b
// CHECK-INST: sub     z0.b, p0/m, z0.b, z0.b
// CHECK-ENCODING: [0x00,0x00,0x01,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 00 01 04 <unknown>

sub     z0.s, p0/m, z0.s, z0.s
// CHECK-INST: sub     z0.s, p0/m, z0.s, z0.s
// CHECK-ENCODING: [0x00,0x00,0x81,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 00 81 04 <unknown>

sub     z21.h, p5/m, z21.h, z10.h
// CHECK-INST: sub     z21.h, p5/m, z21.h, z10.h
// CHECK-ENCODING: [0x55,0x15,0x41,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 55 15 41 04 <unknown>

sub     z23.b, p3/m, z23.b, z13.b
// CHECK-INST: sub     z23.b, p3/m, z23.b, z13.b
// CHECK-ENCODING: [0xb7,0x0d,0x01,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: b7 0d 01 04 <unknown>

// -----------------------
//

sub     z0.b, z0.b, #0
// CHECK-INST: sub     z0.b, z0.b, #0
// CHECK-ENCODING: [0x00,0xc0,0x21,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 c0 21 25 <unknown>

sub     z31.b, z31.b, #255
// CHECK-INST: sub     z31.b, z31.b, #255
// CHECK-ENCODING: [0xff,0xdf,0x21,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff df 21 25 <unknown>

sub     z0.h, z0.h, #0
// CHECK-INST: sub     z0.h, z0.h, #0
// CHECK-ENCODING: [0x00,0xc0,0x61,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 c0 61 25 <unknown>

sub     z0.h, z0.h, #0, lsl #8
// CHECK-INST: sub     z0.h, z0.h, #0, lsl #8
// CHECK-ENCODING: [0x00,0xe0,0x61,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 e0 61 25 <unknown>

sub     z31.h, z31.h, #255, lsl #8
// CHECK-INST: sub     z31.h, z31.h, #65280
// CHECK-ENCODING: [0xff,0xff,0x61,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff ff 61 25 <unknown>

sub     z31.h, z31.h, #65280
// CHECK-INST: sub     z31.h, z31.h, #65280
// CHECK-ENCODING: [0xff,0xff,0x61,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff ff 61 25 <unknown>

sub     z0.s, z0.s, #0
// CHECK-INST: sub     z0.s, z0.s, #0
// CHECK-ENCODING: [0x00,0xc0,0xa1,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 c0 a1 25 <unknown>

sub     z0.s, z0.s, #0, lsl #8
// CHECK-INST: sub     z0.s, z0.s, #0, lsl #8
// CHECK-ENCODING: [0x00,0xe0,0xa1,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 e0 a1 25 <unknown>

sub     z31.s, z31.s, #255, lsl #8
// CHECK-INST: sub     z31.s, z31.s, #65280
// CHECK-ENCODING: [0xff,0xff,0xa1,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff ff a1 25 <unknown>

sub     z31.s, z31.s, #65280
// CHECK-INST: sub     z31.s, z31.s, #65280
// CHECK-ENCODING: [0xff,0xff,0xa1,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff ff a1 25 <unknown>

sub     z0.d, z0.d, #0
// CHECK-INST: sub     z0.d, z0.d, #0
// CHECK-ENCODING: [0x00,0xc0,0xe1,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 c0 e1 25 <unknown>

sub     z0.d, z0.d, #0, lsl #8
// CHECK-INST: sub     z0.d, z0.d, #0, lsl #8
// CHECK-ENCODING: [0x00,0xe0,0xe1,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 e0 e1 25 <unknown>

sub     z31.d, z31.d, #255, lsl #8
// CHECK-INST: sub     z31.d, z31.d, #65280
// CHECK-ENCODING: [0xff,0xff,0xe1,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff ff e1 25 <unknown>

sub     z31.d, z31.d, #65280
// CHECK-INST: sub     z31.d, z31.d, #65280
// CHECK-ENCODING: [0xff,0xff,0xe1,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff ff e1 25 <unknown>



// --------------------------------------------------------------------------//
// Test compatibility with MOVPRFX instruction.

movprfx z23.b, p3/z, z30.b
// CHECK-INST: movprfx	z23.b, p3/z, z30.b
// CHECK-ENCODING: [0xd7,0x2f,0x10,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: d7 2f 10 04 <unknown>

sub     z23.b, p3/m, z23.b, z13.b
// CHECK-INST: sub	z23.b, p3/m, z23.b, z13.b
// CHECK-ENCODING: [0xb7,0x0d,0x01,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: b7 0d 01 04 <unknown>

movprfx z23, z30
// CHECK-INST: movprfx	z23, z30
// CHECK-ENCODING: [0xd7,0xbf,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: d7 bf 20 04 <unknown>

sub     z23.b, p3/m, z23.b, z13.b
// CHECK-INST: sub	z23.b, p3/m, z23.b, z13.b
// CHECK-ENCODING: [0xb7,0x0d,0x01,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: b7 0d 01 04 <unknown>

movprfx z31, z6
// CHECK-INST: movprfx	z31, z6
// CHECK-ENCODING: [0xdf,0xbc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: df bc 20 04 <unknown>

sub     z31.d, z31.d, #65280
// CHECK-INST: sub	z31.d, z31.d, #65280
// CHECK-ENCODING: [0xff,0xff,0xe1,0x25]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff ff e1 25 <unknown>
