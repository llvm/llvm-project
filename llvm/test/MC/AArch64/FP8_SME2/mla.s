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


fmlal   za.h[w8, 0:1], z0.b, z0.b  // 11000001-00110000-00001100-00000000
// CHECK-INST: fmlal   za.h[w8, 0:1], z0.b, z0.b
// CHECK-ENCODING: [0x00,0x0c,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f16
// CHECK-UNKNOWN: c1300c00 <unknown>

fmlal   za.h[w11, 14:15], z31.b, z15.b  // 11000001-00111111-01101111-11100111
// CHECK-INST: fmlal   za.h[w11, 14:15], z31.b, z15.b
// CHECK-ENCODING: [0xe7,0x6f,0x3f,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f16
// CHECK-UNKNOWN: c13f6fe7 <unknown>

fmlal   za.h[w8, 0:1], z0.b, z0.b[0]  // 11000001-11000000-00000000-00000000
// CHECK-INST: fmlal   za.h[w8, 0:1], z0.b, z0.b[0]
// CHECK-ENCODING: [0x00,0x00,0xc0,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f16
// CHECK-UNKNOWN: c1c00000 <unknown>

fmlal   za.h[w11, 14:15], z31.b, z15.b[15]  // 11000001-11001111-11101111-11101111
// CHECK-INST: fmlal   za.h[w11, 14:15], z31.b, z15.b[15]
// CHECK-ENCODING: [0xef,0xef,0xcf,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f16
// CHECK-UNKNOWN: c1cfefef <unknown>

// x2

fmlal   za.h[w8, 0:1, vgx2], {z0.b-z1.b}, z0.b  // 11000001-00100000-00001000-00000100
// CHECK-INST: fmlal   za.h[w8, 0:1, vgx2], { z0.b, z1.b }, z0.b
// CHECK-ENCODING: [0x04,0x08,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f16
// CHECK-UNKNOWN: c1200804 <unknown>

fmlal   za.h[w8, 0:1], {z0.b-z1.b}, z0.b  // 11000001-00100000-00001000-00000100
// CHECK-INST: fmlal   za.h[w8, 0:1, vgx2], { z0.b, z1.b }, z0.b
// CHECK-ENCODING: [0x04,0x08,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f16
// CHECK-UNKNOWN: c1200804 <unknown>

fmlal   za.h[w11, 6:7, vgx2], {z31.b-z0.b}, z15.b  // 11000001-00101111-01101011-11100111
// CHECK-INST: fmlal   za.h[w11, 6:7, vgx2], { z31.b, z0.b }, z15.b
// CHECK-ENCODING: [0xe7,0x6b,0x2f,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f16
// CHECK-UNKNOWN: c12f6be7 <unknown>

fmlal   za.h[w11, 6:7], {z31.b-z0.b}, z15.b  // 11000001-00101111-01101011-11100111
// CHECK-INST: fmlal   za.h[w11, 6:7, vgx2], { z31.b, z0.b }, z15.b
// CHECK-ENCODING: [0xe7,0x6b,0x2f,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f16
// CHECK-UNKNOWN: c12f6be7 <unknown>

fmlal   za.h[w8, 0:1, vgx2], {z0.b-z1.b}, {z0.b-z1.b}  // 11000001-10100000-00001000-00100000
// CHECK-INST: fmlal   za.h[w8, 0:1, vgx2], { z0.b, z1.b }, { z0.b, z1.b }
// CHECK-ENCODING: [0x20,0x08,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f16
// CHECK-UNKNOWN: c1a00820 <unknown>

fmlal   za.h[w8, 0:1], {z0.b-z1.b}, {z0.b-z1.b}  // 11000001-10100000-00001000-00100000
// CHECK-INST: fmlal   za.h[w8, 0:1, vgx2], { z0.b, z1.b }, { z0.b, z1.b }
// CHECK-ENCODING: [0x20,0x08,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f16
// CHECK-UNKNOWN: c1a00820 <unknown>

fmlal   za.h[w11, 6:7, vgx2], {z30.b-z31.b}, {z30.b-z31.b}  // 11000001-10111110-01101011-11100011
// CHECK-INST: fmlal   za.h[w11, 6:7, vgx2], { z30.b, z31.b }, { z30.b, z31.b }
// CHECK-ENCODING: [0xe3,0x6b,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f16
// CHECK-UNKNOWN: c1be6be3 <unknown>

fmlal   za.h[w11, 6:7], {z30.b-z31.b}, {z30.b-z31.b}  // 11000001-10111110-01101011-11100011
// CHECK-INST: fmlal   za.h[w11, 6:7, vgx2], { z30.b, z31.b }, { z30.b, z31.b }
// CHECK-ENCODING: [0xe3,0x6b,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f16
// CHECK-UNKNOWN: c1be6be3 <unknown>

fmlal   za.h[w8, 0:1, vgx2], {z0.b-z1.b}, z0.b[0]  // 11000001-10010000-00010000-00110000
// CHECK-INST: fmlal   za.h[w8, 0:1, vgx2], { z0.b, z1.b }, z0.b[0]
// CHECK-ENCODING: [0x30,0x10,0x90,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f16
// CHECK-UNKNOWN: c1901030 <unknown>

fmlal   za.h[w8, 0:1], {z0.b-z1.b}, z0.b[0]  // 11000001-10010000-00010000-00110000
// CHECK-INST: fmlal   za.h[w8, 0:1, vgx2], { z0.b, z1.b }, z0.b[0]
// CHECK-ENCODING: [0x30,0x10,0x90,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f16
// CHECK-UNKNOWN: c1901030 <unknown>

fmlal   za.h[w11, 6:7, vgx2], {z30.b-z31.b}, z15.b[15]  // 11000001-10011111-01111111-11111111
// CHECK-INST: fmlal   za.h[w11, 6:7, vgx2], { z30.b, z31.b }, z15.b[15]
// CHECK-ENCODING: [0xff,0x7f,0x9f,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f16
// CHECK-UNKNOWN: c19f7fff <unknown>

fmlal   za.h[w11, 6:7], {z30.b-z31.b}, z15.b[15]  // 11000001-10011111-01111111-11111111
// CHECK-INST: fmlal   za.h[w11, 6:7, vgx2], { z30.b, z31.b }, z15.b[15]
// CHECK-ENCODING: [0xff,0x7f,0x9f,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f16
// CHECK-UNKNOWN: c19f7fff <unknown>

// x4

fmlal   za.h[w8, 0:1, vgx4], {z0.b-z3.b}, z0.b  // 11000001-00110000-00001000-00000100
// CHECK-INST: fmlal   za.h[w8, 0:1, vgx4], { z0.b - z3.b }, z0.b
// CHECK-ENCODING: [0x04,0x08,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f16
// CHECK-UNKNOWN: c1300804 <unknown>

fmlal   za.h[w8, 0:1], {z0.b-z3.b}, z0.b  // 11000001-00110000-00001000-00000100
// CHECK-INST: fmlal   za.h[w8, 0:1, vgx4], { z0.b - z3.b }, z0.b
// CHECK-ENCODING: [0x04,0x08,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f16
// CHECK-UNKNOWN: c1300804 <unknown>

fmlal   za.h[w11, 6:7, vgx4], {z31.b-z2.b}, z15.b  // 11000001-00111111-01101011-11100111
// CHECK-INST: fmlal   za.h[w11, 6:7, vgx4], {  z31.b, z0.b, z1.b, z2.b  }, z15.b
// CHECK-ENCODING: [0xe7,0x6b,0x3f,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f16
// CHECK-UNKNOWN: c13f6be7 <unknown>

fmlal   za.h[w11, 6:7], {z31.b-z2.b}, z15.b  // 11000001-00111111-01101011-11100111
// CHECK-INST: fmlal   za.h[w11, 6:7, vgx4], {  z31.b, z0.b, z1.b, z2.b  }, z15.b
// CHECK-ENCODING: [0xe7,0x6b,0x3f,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f16
// CHECK-UNKNOWN: c13f6be7 <unknown>

fmlal   za.h[w8, 0:1, vgx4], {z0.b-z3.b}, {z0.b-z3.b}  // 11000001-10100001-00001000-00100000
// CHECK-INST: fmlal   za.h[w8, 0:1, vgx4], { z0.b - z3.b }, { z0.b - z3.b }
// CHECK-ENCODING: [0x20,0x08,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f16
// CHECK-UNKNOWN: c1a10820 <unknown>

fmlal   za.h[w8, 0:1], {z0.b-z3.b}, {z0.b-z3.b}  // 11000001-10100001-00001000-00100000
// CHECK-INST: fmlal   za.h[w8, 0:1, vgx4], { z0.b - z3.b }, { z0.b - z3.b }
// CHECK-ENCODING: [0x20,0x08,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f16
// CHECK-UNKNOWN: c1a10820 <unknown>

fmlal   za.h[w11, 6:7, vgx4], {z28.b-z31.b}, {z28.b-z31.b}  // 11000001-10111101-01101011-10100011
// CHECK-INST: fmlal   za.h[w11, 6:7, vgx4], { z28.b - z31.b }, { z28.b - z31.b }
// CHECK-ENCODING: [0xa3,0x6b,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f16
// CHECK-UNKNOWN: c1bd6ba3 <unknown>

fmlal   za.h[w11, 6:7], {z28.b-z31.b}, {z28.b-z31.b}  // 11000001-10111101-01101011-10100011
// CHECK-INST: fmlal   za.h[w11, 6:7, vgx4], { z28.b - z31.b }, { z28.b - z31.b }
// CHECK-ENCODING: [0xa3,0x6b,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f16
// CHECK-UNKNOWN: c1bd6ba3 <unknown>

fmlal   za.h[w8, 0:1, vgx4], {z0.b-z3.b}, z0.b[0]  // 11000001-10010000-10010000-00100000
// CHECK-INST: fmlal   za.h[w8, 0:1, vgx4], { z0.b - z3.b }, z0.b[0]
// CHECK-ENCODING: [0x20,0x90,0x90,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f16
// CHECK-UNKNOWN: c1909020 <unknown>

fmlal   za.h[w8, 0:1], {z0.b-z3.b}, z0.b[0]  // 11000001-10010000-10010000-00100000
// CHECK-INST: fmlal   za.h[w8, 0:1, vgx4], { z0.b - z3.b }, z0.b[0]
// CHECK-ENCODING: [0x20,0x90,0x90,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f16
// CHECK-UNKNOWN: c1909020 <unknown>

fmlal   za.h[w11, 6:7, vgx4], {z28.b-z31.b}, z15.b[15]  // 11000001-10011111-11111111-10101111
// CHECK-INST: fmlal   za.h[w11, 6:7, vgx4], { z28.b - z31.b }, z15.b[15]
// CHECK-ENCODING: [0xaf,0xff,0x9f,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f16
// CHECK-UNKNOWN: c19fffaf <unknown>

fmlal   za.h[w11, 6:7], {z28.b-z31.b}, z15.b[15]  // 11000001-10011111-11111111-10101111
// CHECK-INST: fmlal   za.h[w11, 6:7, vgx4], { z28.b - z31.b }, z15.b[15]
// CHECK-ENCODING: [0xaf,0xff,0x9f,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f16
// CHECK-UNKNOWN: c19fffaf <unknown>


//FMLALL

fmlall  za.s[w8, 0:3], z0.b, z0.b  // 11000001-00110000-00000100-00000000
// CHECK-INST: fmlall  za.s[w8, 0:3], z0.b, z0.b
// CHECK-ENCODING: [0x00,0x04,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f32
// CHECK-UNKNOWN: c1300400 <unknown>

fmlall  za.s[w11, 12:15], z31.b, z15.b  // 11000001-00111111-01100111-11100011
// CHECK-INST: fmlall  za.s[w11, 12:15], z31.b, z15.b
// CHECK-ENCODING: [0xe3,0x67,0x3f,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f32
// CHECK-UNKNOWN: c13f67e3 <unknown>

fmlall  za.s[w8, 0:3], z0.b, z0.b[0]  // 11000001-01000000-00000000-00000000
// CHECK-INST: fmlall  za.s[w8, 0:3], z0.b, z0.b[0]
// CHECK-ENCODING: [0x00,0x00,0x40,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f32
// CHECK-UNKNOWN: c1400000 <unknown>

fmlall  za.s[w11, 12:15], z31.b, z15.b[15]  // 11000001-01001111-11111111-11100011
// CHECK-INST: fmlall  za.s[w11, 12:15], z31.b, z15.b[15]
// CHECK-ENCODING: [0xe3,0xff,0x4f,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f32
// CHECK-UNKNOWN: c14fffe3 <unknown>

// x2

fmlall  za.s[w8, 0:3, vgx2], {z0.b-z1.b}, z0.b  // 11000001-00100000-00000000-00000010
// CHECK-INST: fmlall  za.s[w8, 0:3, vgx2], { z0.b, z1.b }, z0.b
// CHECK-ENCODING: [0x02,0x00,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f32
// CHECK-UNKNOWN: c1200002 <unknown>

fmlall  za.s[w8, 0:3], {z0.b-z1.b}, z0.b  // 11000001-00100000-00000000-00000010
// CHECK-INST: fmlall  za.s[w8, 0:3, vgx2], { z0.b, z1.b }, z0.b
// CHECK-ENCODING: [0x02,0x00,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f32
// CHECK-UNKNOWN: c1200002 <unknown>

fmlall  za.s[w11, 4:7, vgx2], {z31.b-z0.b}, z15.b  // 11000001-00101111-01100011-11100011
// CHECK-INST: fmlall  za.s[w11, 4:7, vgx2], { z31.b, z0.b }, z15.b
// CHECK-ENCODING: [0xe3,0x63,0x2f,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f32
// CHECK-UNKNOWN: c12f63e3 <unknown>

fmlall  za.s[w11, 4:7], {z31.b-z0.b}, z15.b  // 11000001-00101111-01100011-11100011
// CHECK-INST: fmlall  za.s[w11, 4:7, vgx2], { z31.b, z0.b }, z15.b
// CHECK-ENCODING: [0xe3,0x63,0x2f,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f32
// CHECK-UNKNOWN: c12f63e3 <unknown>

fmlall  za.s[w8, 0:3, vgx2], {z0.b-z1.b}, {z0.b-z1.b}  // 11000001-10100000-00000000-00100000
// CHECK-INST: fmlall  za.s[w8, 0:3, vgx2], { z0.b, z1.b }, { z0.b, z1.b }
// CHECK-ENCODING: [0x20,0x00,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f32
// CHECK-UNKNOWN: c1a00020 <unknown>

fmlall  za.s[w8, 0:3], {z0.b-z1.b}, {z0.b-z1.b}  // 11000001-10100000-00000000-00100000
// CHECK-INST: fmlall  za.s[w8, 0:3, vgx2], { z0.b, z1.b }, { z0.b, z1.b }
// CHECK-ENCODING: [0x20,0x00,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f32
// CHECK-UNKNOWN: c1a00020 <unknown>

fmlall  za.s[w11, 4:7, vgx2], {z30.b-z31.b}, {z30.b-z31.b}  // 11000001-10111110-01100011-11100001
// CHECK-INST: fmlall  za.s[w11, 4:7, vgx2], { z30.b, z31.b }, { z30.b, z31.b }
// CHECK-ENCODING: [0xe1,0x63,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f32
// CHECK-UNKNOWN: c1be63e1 <unknown>

fmlall  za.s[w11, 4:7], {z30.b-z31.b}, {z30.b-z31.b}  // 11000001-10111110-01100011-11100001
// CHECK-INST: fmlall  za.s[w11, 4:7, vgx2], { z30.b, z31.b }, { z30.b, z31.b }
// CHECK-ENCODING: [0xe1,0x63,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f32
// CHECK-UNKNOWN: c1be63e1 <unknown>

fmlall  za.s[w8, 0:3, vgx2], {z0.b-z1.b}, z0.b[0]  // 11000001-10010000-00000000-00100000
// CHECK-INST: fmlall  za.s[w8, 0:3, vgx2], { z0.b, z1.b }, z0.b[0]
// CHECK-ENCODING: [0x20,0x00,0x90,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f32
// CHECK-UNKNOWN: c1900020 <unknown>

fmlall  za.s[w8, 0:3], {z0.b-z1.b}, z0.b[0]  // 11000001-10010000-00000000-00100000
// CHECK-INST: fmlall  za.s[w8, 0:3, vgx2], { z0.b, z1.b }, z0.b[0]
// CHECK-ENCODING: [0x20,0x00,0x90,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f32
// CHECK-UNKNOWN: c1900020 <unknown>

fmlall  za.s[w11, 4:7, vgx2], {z30.b-z31.b}, z15.b[15]  // 11000001-10011111-01101111-11100111
// CHECK-INST: fmlall  za.s[w11, 4:7, vgx2], { z30.b, z31.b }, z15.b[15]
// CHECK-ENCODING: [0xe7,0x6f,0x9f,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f32
// CHECK-UNKNOWN: c19f6fe7 <unknown>

fmlall  za.s[w11, 4:7], {z30.b-z31.b}, z15.b[15]  // 11000001-10011111-01101111-11100111
// CHECK-INST: fmlall  za.s[w11, 4:7, vgx2], { z30.b, z31.b }, z15.b[15]
// CHECK-ENCODING: [0xe7,0x6f,0x9f,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f32
// CHECK-UNKNOWN: c19f6fe7 <unknown>

// x4

fmlall  za.s[w8, 0:3, vgx4], {z0.b-z3.b}, z0.b  // 11000001-00110000-00000000-00000010
// CHECK-INST: fmlall  za.s[w8, 0:3, vgx4], { z0.b - z3.b }, z0.b
// CHECK-ENCODING: [0x02,0x00,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f32
// CHECK-UNKNOWN: c1300002 <unknown>

fmlall  za.s[w8, 0:3], {z0.b-z3.b}, z0.b  // 11000001-00110000-00000000-00000010
// CHECK-INST: fmlall  za.s[w8, 0:3, vgx4], { z0.b - z3.b }, z0.b
// CHECK-ENCODING: [0x02,0x00,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f32
// CHECK-UNKNOWN: c1300002 <unknown>

fmlall  za.s[w11, 4:7, vgx4], {z31.b-z2.b}, z15.b  // 11000001-00111111-01100011-11100011
// CHECK-INST: fmlall  za.s[w11, 4:7, vgx4], {  z31.b, z0.b, z1.b, z2.b  }, z15.b
// CHECK-ENCODING: [0xe3,0x63,0x3f,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f32
// CHECK-UNKNOWN: c13f63e3 <unknown>

fmlall  za.s[w11, 4:7], {z31.b-z2.b}, z15.b  // 11000001-00111111-01100011-11100011
// CHECK-INST: fmlall  za.s[w11, 4:7, vgx4], {  z31.b, z0.b, z1.b, z2.b  }, z15.b
// CHECK-ENCODING: [0xe3,0x63,0x3f,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f32
// CHECK-UNKNOWN: c13f63e3 <unknown>

fmlall  za.s[w8, 0:3, vgx4], {z0.b-z3.b}, {z0.b-z3.b}  // 11000001-10100001-00000000-00100000
// CHECK-INST: fmlall  za.s[w8, 0:3, vgx4], { z0.b - z3.b }, { z0.b - z3.b }
// CHECK-ENCODING: [0x20,0x00,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f32
// CHECK-UNKNOWN: c1a10020 <unknown>

fmlall  za.s[w8, 0:3], {z0.b-z3.b}, {z0.b-z3.b}  // 11000001-10100001-00000000-00100000
// CHECK-INST: fmlall  za.s[w8, 0:3, vgx4], { z0.b - z3.b }, { z0.b - z3.b }
// CHECK-ENCODING: [0x20,0x00,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f32
// CHECK-UNKNOWN: c1a10020 <unknown>

fmlall  za.s[w11, 4:7, vgx4], {z28.b-z31.b}, {z28.b-z31.b}  // 11000001-10111101-01100011-10100001
// CHECK-INST: fmlall  za.s[w11, 4:7, vgx4], { z28.b - z31.b }, { z28.b - z31.b }
// CHECK-ENCODING: [0xa1,0x63,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f32
// CHECK-UNKNOWN: c1bd63a1 <unknown>

fmlall  za.s[w11, 4:7], {z28.b-z31.b}, {z28.b-z31.b}  // 11000001-10111101-01100011-10100001
// CHECK-INST: fmlall  za.s[w11, 4:7, vgx4], { z28.b - z31.b }, { z28.b - z31.b }
// CHECK-ENCODING: [0xa1,0x63,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f32
// CHECK-UNKNOWN: c1bd63a1 <unknown>

fmlall  za.s[w8, 0:3, vgx4], {z0.b-z3.b}, z0.b[0]  // 11000001-00010000-10000000-01000000
// CHECK-INST: fmlall  za.s[w8, 0:3, vgx4], { z0.b - z3.b }, z0.b[0]
// CHECK-ENCODING: [0x40,0x80,0x10,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f32
// CHECK-UNKNOWN: c1108040 <unknown>

fmlall  za.s[w8, 0:3], {z0.b-z3.b}, z0.b[0]  // 11000001-00010000-10000000-01000000
// CHECK-INST: fmlall  za.s[w8, 0:3, vgx4], { z0.b - z3.b }, z0.b[0]
// CHECK-ENCODING: [0x40,0x80,0x10,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f32
// CHECK-UNKNOWN: c1108040 <unknown>

fmlall  za.s[w11, 4:7, vgx4], {z28.b-z31.b}, z15.b[15]  // 11000001-00011111-11101111-11000111
// CHECK-INST: fmlall  za.s[w11, 4:7, vgx4], { z28.b - z31.b }, z15.b[15]
// CHECK-ENCODING: [0xc7,0xef,0x1f,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f32
// CHECK-UNKNOWN: c11fefc7 <unknown>

fmlall  za.s[w11, 4:7], {z28.b-z31.b}, z15.b[15]  // 11000001-00011111-11101111-11000111
// CHECK-INST: fmlall  za.s[w11, 4:7, vgx4], { z28.b - z31.b }, z15.b[15]
// CHECK-ENCODING: [0xc7,0xef,0x1f,0xc1]
// CHECK-ERROR: instruction requires: sme-f8f32
// CHECK-UNKNOWN: c11fefc7 <unknown>
