// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme-f8f16,+sme-f8f32 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme-f8f16,+sme-f8f32 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme-f8f16,+sme-f8f32 --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme-f8f16,+sme-f8f32 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme-f8f16,-sme-f8f32 --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme-f8f16,+sme-f8f32 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme-f8f16,+sme-f8f32 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST


// FDOT
// x2

fdot    za.h[w8, 0, vgx2], {z0.b-z1.b}, z0.b  // 11000001-00100000-00010000-00001000
// CHECK-INST: fdot    za.h[w8, 0, vgx2], { z0.b, z1.b }, z0.b
// CHECK-ENCODING: [0x08,0x10,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f16
// CHECK-UNKNOWN: c1201008 <unknown>

fdot    za.h[w8, 0], {z0.b-z1.b}, z0.b  // 11000001-00100000-00010000-00001000
// CHECK-INST: fdot    za.h[w8, 0, vgx2], { z0.b, z1.b }, z0.b
// CHECK-ENCODING: [0x08,0x10,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f16
// CHECK-UNKNOWN: c1201008 <unknown>

fdot    za.h[w11, 7], {z13.b-z14.b}, z8.b  // 11000001-00101000-01110001-10101111
// CHECK-INST: fdot    za.h[w11, 7, vgx2], { z13.b, z14.b }, z8.b
// CHECK-ENCODING: [0xaf,0x71,0x28,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f16
// CHECK-UNKNOWN: c12871af <unknown>

fdot    za.h[w11, 7, vgx2], {z31.b-z0.b}, z15.b  // 11000001-00101111-01110011-11101111
// CHECK-INST: fdot    za.h[w11, 7, vgx2], { z31.b, z0.b }, z15.b
// CHECK-ENCODING: [0xef,0x73,0x2f,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f16
// CHECK-UNKNOWN: c12f73ef <unknown>

fdot    za.s[w8, 0, vgx2], {z0.b-z1.b}, z0.b  // 11000001-00100000-00010000-00011000
// CHECK-INST: fdot    za.s[w8, 0, vgx2], { z0.b, z1.b }, z0.b
// CHECK-ENCODING: [0x18,0x10,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f32
// CHECK-UNKNOWN: c1201018 <unknown>

fdot    za.s[w8, 0], {z0.b-z1.b}, z0.b  // 11000001-00100000-00010000-00011000
// CHECK-INST: fdot    za.s[w8, 0, vgx2], { z0.b, z1.b }, z0.b
// CHECK-ENCODING: [0x18,0x10,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f32
// CHECK-UNKNOWN: c1201018 <unknown>

fdot    za.s[w11, 7, vgx2], {z31.b-z0.b}, z15.b  // 11000001-00101111-01110011-11111111
// CHECK-INST: fdot    za.s[w11, 7, vgx2], { z31.b, z0.b }, z15.b
// CHECK-ENCODING: [0xff,0x73,0x2f,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f32
// CHECK-UNKNOWN: c12f73ff <unknown>

fdot    za.s[w11, 7], {z31.b-z0.b}, z15.b  // 11000001-00101111-01110011-11111111
// CHECK-INST: fdot    za.s[w11, 7, vgx2], { z31.b, z0.b }, z15.b
// CHECK-ENCODING: [0xff,0x73,0x2f,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f32
// CHECK-UNKNOWN: c12f73ff <unknown>

fdot    za.h[w8, 0, vgx2], {z0.b-z1.b}, {z0.b-z1.b}  // 11000001-10100000-00010000-00100000
// CHECK-INST: fdot    za.h[w8, 0, vgx2], { z0.b, z1.b }, { z0.b, z1.b }
// CHECK-ENCODING: [0x20,0x10,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f16
// CHECK-UNKNOWN: c1a01020 <unknown>

fdot    za.h[w8, 0], {z0.b-z1.b}, {z0.b-z1.b}  // 11000001-10100000-00010000-00100000
// CHECK-INST: fdot    za.h[w8, 0, vgx2], { z0.b, z1.b }, { z0.b, z1.b }
// CHECK-ENCODING: [0x20,0x10,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f16
// CHECK-UNKNOWN: c1a01020 <unknown>

fdot    za.h[w11, 7, vgx2], {z30.b-z31.b}, {z30.b-z31.b}  // 11000001-10111110-01110011-11100111
// CHECK-INST: fdot    za.h[w11, 7, vgx2], { z30.b, z31.b }, { z30.b, z31.b }
// CHECK-ENCODING: [0xe7,0x73,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f16
// CHECK-UNKNOWN: c1be73e7 <unknown>

fdot    za.h[w11, 7], {z30.b-z31.b}, {z30.b-z31.b}  // 11000001-10111110-01110011-11100111
// CHECK-INST: fdot    za.h[w11, 7, vgx2], { z30.b, z31.b }, { z30.b, z31.b }
// CHECK-ENCODING: [0xe7,0x73,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f16
// CHECK-UNKNOWN: c1be73e7 <unknown>

fdot    za.s[w8, 0, vgx2], {z0.b-z1.b}, {z0.b-z1.b}  // 11000001-10100000-00010000-00110000
// CHECK-INST: fdot    za.s[w8, 0, vgx2], { z0.b, z1.b }, { z0.b, z1.b }
// CHECK-ENCODING: [0x30,0x10,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f32
// CHECK-UNKNOWN: c1a01030 <unknown>

fdot    za.s[w8, 0], {z0.b-z1.b}, {z0.b-z1.b}  // 11000001-10100000-00010000-00110000
// CHECK-INST: fdot    za.s[w8, 0, vgx2], { z0.b, z1.b }, { z0.b, z1.b }
// CHECK-ENCODING: [0x30,0x10,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f32
// CHECK-UNKNOWN: c1a01030 <unknown>

fdot    za.s[w11, 7, vgx2], {z30.b-z31.b}, {z30.b-z31.b}  // 11000001-10111110-01110011-11110111
// CHECK-INST: fdot    za.s[w11, 7, vgx2], { z30.b, z31.b }, { z30.b, z31.b }
// CHECK-ENCODING: [0xf7,0x73,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f32
// CHECK-UNKNOWN: c1be73f7 <unknown>

fdot    za.s[w11, 7], {z30.b-z31.b}, {z30.b-z31.b}  // 11000001-10111110-01110011-11110111
// CHECK-INST: fdot    za.s[w11, 7, vgx2], { z30.b, z31.b }, { z30.b, z31.b }
// CHECK-ENCODING: [0xf7,0x73,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f32
// CHECK-UNKNOWN: c1be73f7 <unknown>

fdot    za.h[w8, 0, vgx2], {z0.b-z1.b}, z0.b[0]  // 11000001-11010000-00000000-00100000
// CHECK-INST: fdot    za.h[w8, 0, vgx2], { z0.b, z1.b }, z0.b[0]
// CHECK-ENCODING: [0x20,0x00,0xd0,0xc1]
// CHECK-ERROR:  instruction requires: sme-f8f16
// CHECK-UNKNOWN: c1d00020 <unknown>

fdot    za.h[w8, 0], {z0.b-z1.b}, z0.b[0]  // 11000001-11010000-00000000-00100000
// CHECK-INST: fdot    za.h[w8, 0, vgx2], { z0.b, z1.b }, z0.b[0]
// CHECK-ENCODING: [0x20,0x00,0xd0,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f16
// CHECK-UNKNOWN: c1d00020 <unknown>

fdot    za.h[w11, 7, vgx2], {z30.b-z31.b}, z15.b[7]  // 11000001-11011111-01101111-11101111
// CHECK-INST: fdot    za.h[w11, 7, vgx2], { z30.b, z31.b }, z15.b[7]
// CHECK-ENCODING: [0xef,0x6f,0xdf,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f16
// CHECK-UNKNOWN: c1df6fef <unknown>

fdot    za.h[w11, 7], {z30.b-z31.b}, z15.b[7]  // 11000001-11011111-01101111-11101111
// CHECK-INST: fdot    za.h[w11, 7, vgx2], { z30.b, z31.b }, z15.b[7]
// CHECK-ENCODING: [0xef,0x6f,0xdf,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f16
// CHECK-UNKNOWN: c1df6fef <unknown>

fdot    za.s[w8, 0, vgx2], {z0.b-z1.b}, z0.b[0]  // 11000001-01010000-00000000-00111000
// CHECK-INST: fdot    za.s[w8, 0, vgx2], { z0.b, z1.b }, z0.b[0]
// CHECK-ENCODING: [0x38,0x00,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f32
// CHECK-UNKNOWN: c1500038 <unknown>

fdot    za.s[w8, 0], {z0.b-z1.b}, z0.b[0]  // 11000001-01010000-00000000-00111000
// CHECK-INST: fdot    za.s[w8, 0, vgx2], { z0.b, z1.b }, z0.b[0]
// CHECK-ENCODING: [0x38,0x00,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f32
// CHECK-UNKNOWN: c1500038 <unknown>

fdot    za.s[w11, 7, vgx2], {z30.b-z31.b}, z15.b[3]  // 11000001-01011111-01101111-11111111
// CHECK-INST: fdot    za.s[w11, 7, vgx2], { z30.b, z31.b }, z15.b[3]
// CHECK-ENCODING: [0xff,0x6f,0x5f,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f32
// CHECK-UNKNOWN: c15f6fff <unknown>

fdot    za.s[w11, 7], {z30.b-z31.b}, z15.b[3]  // 11000001-01011111-01101111-11111111
// CHECK-INST: fdot    za.s[w11, 7, vgx2], { z30.b, z31.b }, z15.b[3]
// CHECK-ENCODING: [0xff,0x6f,0x5f,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f32
// CHECK-UNKNOWN: c15f6fff <unknown>

// x4


fdot    za.h[w8, 0, vgx4], {z0.b-z3.b}, z0.b  // 11000001-00110000-00010000-00001000
// CHECK-INST: fdot    za.h[w8, 0, vgx4], { z0.b - z3.b }, z0.b
// CHECK-ENCODING: [0x08,0x10,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f16
// CHECK-UNKNOWN: c1301008 <unknown>

fdot    za.h[w8, 0], {z0.b-z3.b}, z0.b  // 11000001-00110000-00010000-00001000
// CHECK-INST: fdot    za.h[w8, 0, vgx4], { z0.b - z3.b }, z0.b
// CHECK-ENCODING: [0x08,0x10,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f16
// CHECK-UNKNOWN: c1301008 <unknown>

fdot    za.h[w11, 7, vgx4], {z31.b-z2.b}, z15.b  // 11000001-00111111-01110011-11101111
// CHECK-INST: fdot    za.h[w11, 7, vgx4], {  z31.b, z0.b, z1.b, z2.b  }, z15.b
// CHECK-ENCODING: [0xef,0x73,0x3f,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f16
// CHECK-UNKNOWN: c13f73ef <unknown>

fdot    za.h[w11, 7], {z31.b-z2.b}, z15.b  // 11000001-00111111-01110011-11101111
// CHECK-INST: fdot    za.h[w11, 7, vgx4], {  z31.b, z0.b, z1.b, z2.b  }, z15.b
// CHECK-ENCODING: [0xef,0x73,0x3f,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f16
// CHECK-UNKNOWN: c13f73ef <unknown>

fdot    za.s[w8, 0, vgx4], {z0.b-z3.b}, z0.b  // 11000001-00110000-00010000-00011000
// CHECK-INST: fdot    za.s[w8, 0, vgx4], { z0.b - z3.b }, z0.b
// CHECK-ENCODING: [0x18,0x10,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f32
// CHECK-UNKNOWN: c1301018 <unknown>

fdot    za.s[w8, 0], {z0.b-z3.b}, z0.b  // 11000001-00110000-00010000-00011000
// CHECK-INST: fdot    za.s[w8, 0, vgx4], { z0.b - z3.b }, z0.b
// CHECK-ENCODING: [0x18,0x10,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f32
// CHECK-UNKNOWN: c1301018 <unknown>

fdot    za.s[w11, 7, vgx4], {z31.b-z2.b}, z15.b  // 11000001-00111111-01110011-11111111
// CHECK-INST: fdot    za.s[w11, 7, vgx4], {  z31.b, z0.b, z1.b, z2.b  }, z15.b
// CHECK-ENCODING: [0xff,0x73,0x3f,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f32
// CHECK-UNKNOWN: c13f73ff <unknown>

fdot    za.s[w11, 7], {z31.b-z2.b}, z15.b  // 11000001-00111111-01110011-11111111
// CHECK-INST: fdot    za.s[w11, 7, vgx4], {  z31.b, z0.b, z1.b, z2.b  }, z15.b
// CHECK-ENCODING: [0xff,0x73,0x3f,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f32
// CHECK-UNKNOWN: c13f73ff <unknown>

fdot    za.h[w8, 0, vgx4], {z0.b-z3.b}, {z0.b-z3.b}  // 11000001-10100001-00010000-00100000
// CHECK-INST: fdot    za.h[w8, 0, vgx4], { z0.b - z3.b }, { z0.b - z3.b }
// CHECK-ENCODING: [0x20,0x10,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f16
// CHECK-UNKNOWN: c1a11020 <unknown>

fdot    za.h[w8, 0], {z0.b-z3.b}, {z0.b-z3.b}  // 11000001-10100001-00010000-00100000
// CHECK-INST: fdot    za.h[w8, 0, vgx4], { z0.b - z3.b }, { z0.b - z3.b }
// CHECK-ENCODING: [0x20,0x10,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f16
// CHECK-UNKNOWN: c1a11020 <unknown>

fdot    za.h[w11, 7, vgx4], {z28.b-z31.b}, {z28.b-z31.b}  // 11000001-10111101-01110011-10100111
// CHECK-INST: fdot    za.h[w11, 7, vgx4], { z28.b - z31.b }, { z28.b - z31.b }
// CHECK-ENCODING: [0xa7,0x73,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f16
// CHECK-UNKNOWN: c1bd73a7 <unknown>

fdot    za.h[w11, 7], {z28.b-z31.b}, {z28.b-z31.b}  // 11000001-10111101-01110011-10100111
// CHECK-INST: fdot    za.h[w11, 7, vgx4], { z28.b - z31.b }, { z28.b - z31.b }
// CHECK-ENCODING: [0xa7,0x73,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f16
// CHECK-UNKNOWN: c1bd73a7 <unknown>
fdot    za.h[w8, 0, vgx4], {z0.b-z3.b}, z0.b[0]  // 11000001-00010000-10010000-01000000
// CHECK-INST: fdot    za.h[w8, 0, vgx4], { z0.b - z3.b }, z0.b[0]
// CHECK-ENCODING: [0x40,0x90,0x10,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f16
// CHECK-UNKNOWN: c1109040 <unknown>

fdot    za.s[w8, 0, vgx4], {z0.b-z3.b}, {z0.b-z3.b}  // 11000001-10100001-00010000-00110000
// CHECK-INST: fdot    za.s[w8, 0, vgx4], { z0.b - z3.b }, { z0.b - z3.b }
// CHECK-ENCODING: [0x30,0x10,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f32
// CHECK-UNKNOWN: c1a11030 <unknown>

fdot    za.s[w8, 0], {z0.b-z3.b}, {z0.b-z3.b}  // 11000001-10100001-00010000-00110000
// CHECK-INST: fdot    za.s[w8, 0, vgx4], { z0.b - z3.b }, { z0.b - z3.b }
// CHECK-ENCODING: [0x30,0x10,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f32
// CHECK-UNKNOWN: c1a11030 <unknown>

fdot    za.s[w11, 7, vgx4], {z28.b-z31.b}, {z28.b-z31.b}  // 11000001-10111101-01110011-10110111
// CHECK-INST: fdot    za.s[w11, 7, vgx4], { z28.b - z31.b }, { z28.b - z31.b }
// CHECK-ENCODING: [0xb7,0x73,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f32
// CHECK-UNKNOWN: c1bd73b7 <unknown>

fdot    za.s[w11, 7], {z28.b-z31.b}, {z28.b-z31.b}  // 11000001-10111101-01110011-10110111
// CHECK-INST: fdot    za.s[w11, 7, vgx4], { z28.b - z31.b }, { z28.b - z31.b }
// CHECK-ENCODING: [0xb7,0x73,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f32
// CHECK-UNKNOWN: c1bd73b7 <unknown>

fdot    za.h[w8, 0], {z0.b-z3.b}, z0.b[0]  // 11000001-00010000-10010000-01000000
// CHECK-INST: fdot    za.h[w8, 0, vgx4], { z0.b - z3.b }, z0.b[0]
// CHECK-ENCODING: [0x40,0x90,0x10,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f16
// CHECK-UNKNOWN: c1109040 <unknown>

fdot    za.h[w11, 7, vgx4], {z28.b-z31.b}, z15.b[7]  // 11000001-00011111-11111111-11001111
// CHECK-INST: fdot    za.h[w11, 7, vgx4], { z28.b - z31.b }, z15.b[7]
// CHECK-ENCODING: [0xcf,0xff,0x1f,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f16
// CHECK-UNKNOWN: c11fffcf <unknown>

fdot    za.h[w11, 7], {z28.b-z31.b}, z15.b[7]  // 11000001-00011111-11111111-11001111
// CHECK-INST: fdot    za.h[w11, 7, vgx4], { z28.b - z31.b }, z15.b[7]
// CHECK-ENCODING: [0xcf,0xff,0x1f,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f16
// CHECK-UNKNOWN: c11fffcf <unknown>

fdot    za.s[w8, 0, vgx4], {z0.b-z3.b}, z0.b[0]  // 11000001-01010000-10000000-00001000
// CHECK-INST: fdot    za.s[w8, 0, vgx4], { z0.b - z3.b }, z0.b[0]
// CHECK-ENCODING: [0x08,0x80,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f32
// CHECK-UNKNOWN: c1508008 <unknown>

fdot    za.s[w8, 0], {z0.b-z3.b}, z0.b[0]  // 11000001-01010000-10000000-00001000
// CHECK-INST: fdot    za.s[w8, 0, vgx4], { z0.b - z3.b }, z0.b[0]
// CHECK-ENCODING: [0x08,0x80,0x50,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f32
// CHECK-UNKNOWN: c1508008 <unknown>

fdot    za.s[w11, 7, vgx4], {z28.b-z31.b}, z15.b[3]  // 11000001-01011111-11101111-10001111
// CHECK-INST: fdot    za.s[w11, 7, vgx4], { z28.b - z31.b }, z15.b[3]
// CHECK-ENCODING: [0x8f,0xef,0x5f,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f32
// CHECK-UNKNOWN: c15fef8f <unknown>

fdot    za.s[w11, 7], {z28.b-z31.b}, z15.b[3]  // 11000001-01011111-11101111-10001111
// CHECK-INST: fdot    za.s[w11, 7, vgx4], { z28.b - z31.b }, z15.b[3]
// CHECK-ENCODING: [0x8f,0xef,0x5f,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f32
// CHECK-UNKNOWN: c15fef8f <unknown>


// FVDOT

fvdot   za.h[w8, 0, vgx2], {z0.b-z1.b}, z0.b[0]  // 11000001-11010000-00010000-00100000
// CHECK-INST: fvdot   za.h[w8, 0, vgx2], { z0.b, z1.b }, z0.b[0]
// CHECK-ENCODING: [0x20,0x10,0xd0,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f16
// CHECK-UNKNOWN: c1d01020 <unknown>

fvdot   za.h[w8, 0], {z0.b-z1.b}, z0.b[0]  // 11000001-11010000-00010000-00100000
// CHECK-INST: fvdot   za.h[w8, 0, vgx2], { z0.b, z1.b }, z0.b[0]
// CHECK-ENCODING: [0x20,0x10,0xd0,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f16
// CHECK-UNKNOWN: c1d01020 <unknown>

fvdot   za.h[w11, 7, vgx2], {z30.b-z31.b}, z15.b[7]  // 11000001-11011111-01111111-11101111
// CHECK-INST: fvdot   za.h[w11, 7, vgx2], { z30.b, z31.b }, z15.b[7]
// CHECK-ENCODING: [0xef,0x7f,0xdf,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f16
// CHECK-UNKNOWN: c1df7fef <unknown>

fvdot   za.h[w11, 7], {z30.b-z31.b}, z15.b[7]  // 11000001-11011111-01111111-11101111
// CHECK-INST: fvdot   za.h[w11, 7, vgx2], { z30.b, z31.b }, z15.b[7]
// CHECK-ENCODING: [0xef,0x7f,0xdf,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f16
// CHECK-UNKNOWN: c1df7fef <unknown>

// FVDOTB

fvdotb  za.s[w8, 0, vgx4], {z0.b-z1.b}, z0.b[0]  // 11000001-11010000-00001000-00000000
// CHECK-INST: fvdotb  za.s[w8, 0, vgx4], { z0.b, z1.b }, z0.b[0]
// CHECK-ENCODING: [0x00,0x08,0xd0,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f32
// CHECK-UNKNOWN: c1d00800 <unknown>

fvdotb  za.s[w11, 7, vgx4], {z30.b-z31.b}, z15.b[3]  // 11000001-11011111-01101111-11001111
// CHECK-INST: fvdotb  za.s[w11, 7, vgx4], { z30.b, z31.b }, z15.b[3]
// CHECK-ENCODING: [0xcf,0x6f,0xdf,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f32
// CHECK-UNKNOWN: c1df6fcf <unknown>

// FVDOTT
fvdott  za.s[w8, 0, vgx4], {z0.b-z1.b}, z0.b[0]  // 11000001-11010000-00001000-00010000
// CHECK-INST: fvdott  za.s[w8, 0, vgx4], { z0.b, z1.b }, z0.b[0]
// CHECK-ENCODING: [0x10,0x08,0xd0,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f32
// CHECK-UNKNOWN: c1d00810 <unknown>

fvdott  za.s[w11, 7, vgx4], {z30.b-z31.b}, z15.b[3]  // 11000001-11011111-01101111-11011111
// CHECK-INST: fvdott  za.s[w11, 7, vgx4], { z30.b, z31.b }, z15.b[3]
// CHECK-ENCODING: [0xdf,0x6f,0xdf,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f32
// CHECK-UNKNOWN: c1df6fdf <unknown>
