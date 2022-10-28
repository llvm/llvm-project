// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST


usmlall za.s[w8, 0:3], z0.b, z0.b  // 11000001-00100000-00000100-00000100
// CHECK-INST: usmlall za.s[w8, 0:3], z0.b, z0.b
// CHECK-ENCODING: [0x04,0x04,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1200404 <unknown>

usmlall za.s[w10, 4:7], z10.b, z5.b  // 11000001-00100101-01000101-01000101
// CHECK-INST: usmlall za.s[w10, 4:7], z10.b, z5.b
// CHECK-ENCODING: [0x45,0x45,0x25,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1254545 <unknown>

usmlall za.s[w11, 12:15], z13.b, z8.b  // 11000001-00101000-01100101-10100111
// CHECK-INST: usmlall za.s[w11, 12:15], z13.b, z8.b
// CHECK-ENCODING: [0xa7,0x65,0x28,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12865a7 <unknown>

usmlall za.s[w11, 12:15], z31.b, z15.b  // 11000001-00101111-01100111-11100111
// CHECK-INST: usmlall za.s[w11, 12:15], z31.b, z15.b
// CHECK-ENCODING: [0xe7,0x67,0x2f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12f67e7 <unknown>

usmlall za.s[w8, 4:7], z17.b, z0.b  // 11000001-00100000-00000110-00100101
// CHECK-INST: usmlall za.s[w8, 4:7], z17.b, z0.b
// CHECK-ENCODING: [0x25,0x06,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1200625 <unknown>

usmlall za.s[w8, 4:7], z1.b, z14.b  // 11000001-00101110-00000100-00100101
// CHECK-INST: usmlall za.s[w8, 4:7], z1.b, z14.b
// CHECK-ENCODING: [0x25,0x04,0x2e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12e0425 <unknown>

usmlall za.s[w10, 0:3], z19.b, z4.b  // 11000001-00100100-01000110-01100100
// CHECK-INST: usmlall za.s[w10, 0:3], z19.b, z4.b
// CHECK-ENCODING: [0x64,0x46,0x24,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1244664 <unknown>

usmlall za.s[w8, 0:3], z12.b, z2.b  // 11000001-00100010-00000101-10000100
// CHECK-INST: usmlall za.s[w8, 0:3], z12.b, z2.b
// CHECK-ENCODING: [0x84,0x05,0x22,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1220584 <unknown>

usmlall za.s[w10, 4:7], z1.b, z10.b  // 11000001-00101010-01000100-00100101
// CHECK-INST: usmlall za.s[w10, 4:7], z1.b, z10.b
// CHECK-ENCODING: [0x25,0x44,0x2a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12a4425 <unknown>

usmlall za.s[w8, 4:7], z22.b, z14.b  // 11000001-00101110-00000110-11000101
// CHECK-INST: usmlall za.s[w8, 4:7], z22.b, z14.b
// CHECK-ENCODING: [0xc5,0x06,0x2e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12e06c5 <unknown>

usmlall za.s[w11, 8:11], z9.b, z1.b  // 11000001-00100001-01100101-00100110
// CHECK-INST: usmlall za.s[w11, 8:11], z9.b, z1.b
// CHECK-ENCODING: [0x26,0x65,0x21,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1216526 <unknown>

usmlall za.s[w9, 12:15], z12.b, z11.b  // 11000001-00101011-00100101-10000111
// CHECK-INST: usmlall za.s[w9, 12:15], z12.b, z11.b
// CHECK-ENCODING: [0x87,0x25,0x2b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12b2587 <unknown>


usmlall za.s[w8, 0:3], z0.b, z0.b[0]  // 11000001-00000000-00000000-00000100
// CHECK-INST: usmlall za.s[w8, 0:3], z0.b, z0.b[0]
// CHECK-ENCODING: [0x04,0x00,0x00,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1000004 <unknown>

usmlall za.s[w10, 4:7], z10.b, z5.b[5]  // 11000001-00000101-01010101-01000101
// CHECK-INST: usmlall za.s[w10, 4:7], z10.b, z5.b[5]
// CHECK-ENCODING: [0x45,0x55,0x05,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1055545 <unknown>

usmlall za.s[w11, 12:15], z13.b, z8.b[11]  // 11000001-00001000-11101101-10100111
// CHECK-INST: usmlall za.s[w11, 12:15], z13.b, z8.b[11]
// CHECK-ENCODING: [0xa7,0xed,0x08,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c108eda7 <unknown>

usmlall za.s[w11, 12:15], z31.b, z15.b[15]  // 11000001-00001111-11111111-11100111
// CHECK-INST: usmlall za.s[w11, 12:15], z31.b, z15.b[15]
// CHECK-ENCODING: [0xe7,0xff,0x0f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c10fffe7 <unknown>

usmlall za.s[w8, 4:7], z17.b, z0.b[3]  // 11000001-00000000-00001110-00100101
// CHECK-INST: usmlall za.s[w8, 4:7], z17.b, z0.b[3]
// CHECK-ENCODING: [0x25,0x0e,0x00,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1000e25 <unknown>

usmlall za.s[w8, 4:7], z1.b, z14.b[9]  // 11000001-00001110-10000100-00100101
// CHECK-INST: usmlall za.s[w8, 4:7], z1.b, z14.b[9]
// CHECK-ENCODING: [0x25,0x84,0x0e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c10e8425 <unknown>

usmlall za.s[w10, 0:3], z19.b, z4.b[5]  // 11000001-00000100-01010110-01100100
// CHECK-INST: usmlall za.s[w10, 0:3], z19.b, z4.b[5]
// CHECK-ENCODING: [0x64,0x56,0x04,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1045664 <unknown>

usmlall za.s[w8, 0:3], z12.b, z2.b[6]  // 11000001-00000010-00011001-10000100
// CHECK-INST: usmlall za.s[w8, 0:3], z12.b, z2.b[6]
// CHECK-ENCODING: [0x84,0x19,0x02,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1021984 <unknown>

usmlall za.s[w10, 4:7], z1.b, z10.b[10]  // 11000001-00001010-11001000-00100101
// CHECK-INST: usmlall za.s[w10, 4:7], z1.b, z10.b[10]
// CHECK-ENCODING: [0x25,0xc8,0x0a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c10ac825 <unknown>

usmlall za.s[w8, 4:7], z22.b, z14.b[2]  // 11000001-00001110-00001010-11000101
// CHECK-INST: usmlall za.s[w8, 4:7], z22.b, z14.b[2]
// CHECK-ENCODING: [0xc5,0x0a,0x0e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c10e0ac5 <unknown>

usmlall za.s[w11, 8:11], z9.b, z1.b[13]  // 11000001-00000001-11110101-00100110
// CHECK-INST: usmlall za.s[w11, 8:11], z9.b, z1.b[13]
// CHECK-ENCODING: [0x26,0xf5,0x01,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c101f526 <unknown>

usmlall za.s[w9, 12:15], z12.b, z11.b[10]  // 11000001-00001011-10101001-10000111
// CHECK-INST: usmlall za.s[w9, 12:15], z12.b, z11.b[10]
// CHECK-ENCODING: [0x87,0xa9,0x0b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c10ba987 <unknown>


usmlall za.s[w8, 0:3, vgx2], {z0.b, z1.b}, z0.b  // 11000001, 00100000, 00000000, 00000100
// CHECK, INST: usmlall za.s[w8, 0:3, vgx2], { z0.b, z1.b }, z0.b
// CHECK-ENCODING: [0x04,0x00,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1200004 <unknown>

usmlall za.s[w8, 0:3], {z0.b - z1.b}, z0.b  // 11000001-00100000-00000000-00000100
// CHECK, INST: usmlall za.s[w8, 0:3, vgx2], { z0.b, z1.b }, z0.b
// CHECK-ENCODING: [0x04,0x00,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1200004 <unknown>

usmlall za.s[w10, 4:7, vgx2], {z10.b, z11.b}, z5.b  // 11000001, 00100101, 01000001, 01000101
// CHECK, INST: usmlall za.s[w10, 4:7, vgx2], { z10.b, z11.b }, z5.b
// CHECK-ENCODING: [0x45,0x41,0x25,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1254145 <unknown>

usmlall za.s[w10, 4:7], {z10.b - z11.b}, z5.b  // 11000001-00100101-01000001-01000101
// CHECK, INST: usmlall za.s[w10, 4:7, vgx2], { z10.b, z11.b }, z5.b
// CHECK-ENCODING: [0x45,0x41,0x25,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1254145 <unknown>

usmlall za.s[w11, 4:7, vgx2], {z13.b, z14.b}, z8.b  // 11000001, 00101000, 01100001, 10100101
// CHECK, INST: usmlall za.s[w11, 4:7, vgx2], { z13.b, z14.b }, z8.b
// CHECK-ENCODING: [0xa5,0x61,0x28,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12861a5 <unknown>

usmlall za.s[w11, 4:7], {z13.b - z14.b}, z8.b  // 11000001-00101000-01100001-10100101
// CHECK, INST: usmlall za.s[w11, 4:7, vgx2], { z13.b, z14.b }, z8.b
// CHECK-ENCODING: [0xa5,0x61,0x28,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12861a5 <unknown>

usmlall za.s[w11, 4:7, vgx2], {z31.b, z0.b}, z15.b  // 11000001, 00101111, 01100011, 11100101
// CHECK, INST: usmlall za.s[w11, 4:7, vgx2], { z31.b, z0.b }, z15.b
// CHECK-ENCODING: [0xe5,0x63,0x2f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12f63e5 <unknown>

usmlall za.s[w11, 4:7], {z31.b - z0.b}, z15.b  // 11000001-00101111-01100011-11100101
// CHECK, INST: usmlall za.s[w11, 4:7, vgx2], { z31.b, z0.b }, z15.b
// CHECK-ENCODING: [0xe5,0x63,0x2f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12f63e5 <unknown>

usmlall za.s[w8, 4:7, vgx2], {z17.b, z18.b}, z0.b  // 11000001, 00100000, 00000010, 00100101
// CHECK, INST: usmlall za.s[w8, 4:7, vgx2], { z17.b, z18.b }, z0.b
// CHECK-ENCODING: [0x25,0x02,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1200225 <unknown>

usmlall za.s[w8, 4:7], {z17.b - z18.b}, z0.b  // 11000001-00100000-00000010-00100101
// CHECK, INST: usmlall za.s[w8, 4:7, vgx2], { z17.b, z18.b }, z0.b
// CHECK-ENCODING: [0x25,0x02,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1200225 <unknown>

usmlall za.s[w8, 4:7, vgx2], {z1.b, z2.b}, z14.b  // 11000001, 00101110, 00000000, 00100101
// CHECK, INST: usmlall za.s[w8, 4:7, vgx2], { z1.b, z2.b }, z14.b
// CHECK-ENCODING: [0x25,0x00,0x2e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12e0025 <unknown>

usmlall za.s[w8, 4:7], {z1.b - z2.b}, z14.b  // 11000001-00101110-00000000-00100101
// CHECK, INST: usmlall za.s[w8, 4:7, vgx2], { z1.b, z2.b }, z14.b
// CHECK-ENCODING: [0x25,0x00,0x2e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12e0025 <unknown>

usmlall za.s[w10, 0:3, vgx2], {z19.b, z20.b}, z4.b  // 11000001, 00100100, 01000010, 01100100
// CHECK, INST: usmlall za.s[w10, 0:3, vgx2], { z19.b, z20.b }, z4.b
// CHECK-ENCODING: [0x64,0x42,0x24,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1244264 <unknown>

usmlall za.s[w10, 0:3], {z19.b - z20.b}, z4.b  // 11000001-00100100-01000010-01100100
// CHECK, INST: usmlall za.s[w10, 0:3, vgx2], { z19.b, z20.b }, z4.b
// CHECK-ENCODING: [0x64,0x42,0x24,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1244264 <unknown>

usmlall za.s[w8, 0:3, vgx2], {z12.b, z13.b}, z2.b  // 11000001, 00100010, 00000001, 10000100
// CHECK, INST: usmlall za.s[w8, 0:3, vgx2], { z12.b, z13.b }, z2.b
// CHECK-ENCODING: [0x84,0x01,0x22,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1220184 <unknown>

usmlall za.s[w8, 0:3], {z12.b - z13.b}, z2.b  // 11000001-00100010-00000001-10000100
// CHECK, INST: usmlall za.s[w8, 0:3, vgx2], { z12.b, z13.b }, z2.b
// CHECK-ENCODING: [0x84,0x01,0x22,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1220184 <unknown>

usmlall za.s[w10, 4:7, vgx2], {z1.b, z2.b}, z10.b  // 11000001, 00101010, 01000000, 00100101
// CHECK, INST: usmlall za.s[w10, 4:7, vgx2], { z1.b, z2.b }, z10.b
// CHECK-ENCODING: [0x25,0x40,0x2a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12a4025 <unknown>

usmlall za.s[w10, 4:7], {z1.b - z2.b}, z10.b  // 11000001-00101010-01000000-00100101
// CHECK, INST: usmlall za.s[w10, 4:7, vgx2], { z1.b, z2.b }, z10.b
// CHECK-ENCODING: [0x25,0x40,0x2a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12a4025 <unknown>

usmlall za.s[w8, 4:7, vgx2], {z22.b, z23.b}, z14.b  // 11000001, 00101110, 00000010, 11000101
// CHECK, INST: usmlall za.s[w8, 4:7, vgx2], { z22.b, z23.b }, z14.b
// CHECK-ENCODING: [0xc5,0x02,0x2e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12e02c5 <unknown>

usmlall za.s[w8, 4:7], {z22.b - z23.b}, z14.b  // 11000001-00101110-00000010-11000101
// CHECK, INST: usmlall za.s[w8, 4:7, vgx2], { z22.b, z23.b }, z14.b
// CHECK-ENCODING: [0xc5,0x02,0x2e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12e02c5 <unknown>

usmlall za.s[w11, 0:3, vgx2], {z9.b, z10.b}, z1.b  // 11000001, 00100001, 01100001, 00100100
// CHECK, INST: usmlall za.s[w11, 0:3, vgx2], { z9.b, z10.b }, z1.b
// CHECK-ENCODING: [0x24,0x61,0x21,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1216124 <unknown>

usmlall za.s[w11, 0:3], {z9.b - z10.b}, z1.b  // 11000001-00100001-01100001-00100100
// CHECK, INST: usmlall za.s[w11, 0:3, vgx2], { z9.b, z10.b }, z1.b
// CHECK-ENCODING: [0x24,0x61,0x21,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1216124 <unknown>

usmlall za.s[w9, 4:7, vgx2], {z12.b, z13.b}, z11.b  // 11000001, 00101011, 00100001, 10000101
// CHECK, INST: usmlall za.s[w9, 4:7, vgx2], { z12.b, z13.b }, z11.b
// CHECK-ENCODING: [0x85,0x21,0x2b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12b2185 <unknown>

usmlall za.s[w9, 4:7], {z12.b - z13.b}, z11.b  // 11000001-00101011-00100001-10000101
// CHECK, INST: usmlall za.s[w9, 4:7, vgx2], { z12.b, z13.b }, z11.b
// CHECK-ENCODING: [0x85,0x21,0x2b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12b2185 <unknown>


usmlall za.s[w8, 0:3, vgx2], {z0.b, z1.b}, z0.b[0]  // 11000001, 00010000, 00000000, 00100000
// CHECK, INST: usmlall za.s[w8, 0:3, vgx2], { z0.b, z1.b }, z0.b[0]
// CHECK-ENCODING: [0x20,0x00,0x10,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1100020 <unknown>

usmlall za.s[w8, 0:3], {z0.b - z1.b}, z0.b[0]  // 11000001-00010000-00000000-00100000
// CHECK, INST: usmlall za.s[w8, 0:3, vgx2], { z0.b, z1.b }, z0.b[0]
// CHECK-ENCODING: [0x20,0x00,0x10,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1100020 <unknown>

usmlall za.s[w10, 4:7, vgx2], {z10.b, z11.b}, z5.b[6]  // 11000001, 00010101, 01000101, 01100101
// CHECK, INST: usmlall za.s[w10, 4:7, vgx2], { z10.b, z11.b }, z5.b[6]
// CHECK-ENCODING: [0x65,0x45,0x15,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1154565 <unknown>

usmlall za.s[w10, 4:7], {z10.b - z11.b}, z5.b[6]  // 11000001-00010101-01000101-01100101
// CHECK, INST: usmlall za.s[w10, 4:7, vgx2], { z10.b, z11.b }, z5.b[6]
// CHECK-ENCODING: [0x65,0x45,0x15,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1154565 <unknown>

usmlall za.s[w11, 4:7, vgx2], {z12.b, z13.b}, z8.b[15]  // 11000001, 00011000, 01101101, 10100111
// CHECK, INST: usmlall za.s[w11, 4:7, vgx2], { z12.b, z13.b }, z8.b[15]
// CHECK-ENCODING: [0xa7,0x6d,0x18,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1186da7 <unknown>

usmlall za.s[w11, 4:7], {z12.b - z13.b}, z8.b[15]  // 11000001-00011000-01101101-10100111
// CHECK, INST: usmlall za.s[w11, 4:7, vgx2], { z12.b, z13.b }, z8.b[15]
// CHECK-ENCODING: [0xa7,0x6d,0x18,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1186da7 <unknown>

usmlall za.s[w11, 4:7, vgx2], {z30.b, z31.b}, z15.b[15]  // 11000001, 00011111, 01101111, 11100111
// CHECK, INST: usmlall za.s[w11, 4:7, vgx2], { z30.b, z31.b }, z15.b[15]
// CHECK-ENCODING: [0xe7,0x6f,0x1f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11f6fe7 <unknown>

usmlall za.s[w11, 4:7], {z30.b - z31.b}, z15.b[15]  // 11000001-00011111-01101111-11100111
// CHECK, INST: usmlall za.s[w11, 4:7, vgx2], { z30.b, z31.b }, z15.b[15]
// CHECK-ENCODING: [0xe7,0x6f,0x1f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11f6fe7 <unknown>

usmlall za.s[w8, 4:7, vgx2], {z16.b, z17.b}, z0.b[14]  // 11000001, 00010000, 00001110, 00100101
// CHECK, INST: usmlall za.s[w8, 4:7, vgx2], { z16.b, z17.b }, z0.b[14]
// CHECK-ENCODING: [0x25,0x0e,0x10,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1100e25 <unknown>

usmlall za.s[w8, 4:7], {z16.b - z17.b}, z0.b[14]  // 11000001-00010000-00001110-00100101
// CHECK, INST: usmlall za.s[w8, 4:7, vgx2], { z16.b, z17.b }, z0.b[14]
// CHECK-ENCODING: [0x25,0x0e,0x10,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1100e25 <unknown>

usmlall za.s[w8, 4:7, vgx2], {z0.b, z1.b}, z14.b[4]  // 11000001, 00011110, 00000100, 00100001
// CHECK, INST: usmlall za.s[w8, 4:7, vgx2], { z0.b, z1.b }, z14.b[4]
// CHECK-ENCODING: [0x21,0x04,0x1e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11e0421 <unknown>

usmlall za.s[w8, 4:7], {z0.b - z1.b}, z14.b[4]  // 11000001-00011110-00000100-00100001
// CHECK, INST: usmlall za.s[w8, 4:7, vgx2], { z0.b, z1.b }, z14.b[4]
// CHECK-ENCODING: [0x21,0x04,0x1e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11e0421 <unknown>

usmlall za.s[w10, 0:3, vgx2], {z18.b, z19.b}, z4.b[4]  // 11000001, 00010100, 01000110, 01100000
// CHECK, INST: usmlall za.s[w10, 0:3, vgx2], { z18.b, z19.b }, z4.b[4]
// CHECK-ENCODING: [0x60,0x46,0x14,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1144660 <unknown>

usmlall za.s[w10, 0:3], {z18.b - z19.b}, z4.b[4]  // 11000001-00010100-01000110-01100000
// CHECK, INST: usmlall za.s[w10, 0:3, vgx2], { z18.b, z19.b }, z4.b[4]
// CHECK-ENCODING: [0x60,0x46,0x14,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1144660 <unknown>

usmlall za.s[w8, 0:3, vgx2], {z12.b, z13.b}, z2.b[8]  // 11000001, 00010010, 00001001, 10100000
// CHECK, INST: usmlall za.s[w8, 0:3, vgx2], { z12.b, z13.b }, z2.b[8]
// CHECK-ENCODING: [0xa0,0x09,0x12,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11209a0 <unknown>

usmlall za.s[w8, 0:3], {z12.b - z13.b}, z2.b[8]  // 11000001-00010010-00001001-10100000
// CHECK, INST: usmlall za.s[w8, 0:3, vgx2], { z12.b, z13.b }, z2.b[8]
// CHECK-ENCODING: [0xa0,0x09,0x12,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11209a0 <unknown>

usmlall za.s[w10, 4:7, vgx2], {z0.b, z1.b}, z10.b[8]  // 11000001, 00011010, 01001000, 00100001
// CHECK, INST: usmlall za.s[w10, 4:7, vgx2], { z0.b, z1.b }, z10.b[8]
// CHECK-ENCODING: [0x21,0x48,0x1a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11a4821 <unknown>

usmlall za.s[w10, 4:7], {z0.b - z1.b}, z10.b[8]  // 11000001-00011010-01001000-00100001
// CHECK, INST: usmlall za.s[w10, 4:7, vgx2], { z0.b, z1.b }, z10.b[8]
// CHECK-ENCODING: [0x21,0x48,0x1a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11a4821 <unknown>

usmlall za.s[w8, 4:7, vgx2], {z22.b, z23.b}, z14.b[10]  // 11000001, 00011110, 00001010, 11100101
// CHECK, INST: usmlall za.s[w8, 4:7, vgx2], { z22.b, z23.b }, z14.b[10]
// CHECK-ENCODING: [0xe5,0x0a,0x1e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11e0ae5 <unknown>

usmlall za.s[w8, 4:7], {z22.b - z23.b}, z14.b[10]  // 11000001-00011110-00001010-11100101
// CHECK, INST: usmlall za.s[w8, 4:7, vgx2], { z22.b, z23.b }, z14.b[10]
// CHECK-ENCODING: [0xe5,0x0a,0x1e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11e0ae5 <unknown>

usmlall za.s[w11, 0:3, vgx2], {z8.b, z9.b}, z1.b[5]  // 11000001, 00010001, 01100101, 00100010
// CHECK, INST: usmlall za.s[w11, 0:3, vgx2], { z8.b, z9.b }, z1.b[5]
// CHECK-ENCODING: [0x22,0x65,0x11,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1116522 <unknown>

usmlall za.s[w11, 0:3], {z8.b - z9.b}, z1.b[5]  // 11000001-00010001-01100101-00100010
// CHECK, INST: usmlall za.s[w11, 0:3, vgx2], { z8.b, z9.b }, z1.b[5]
// CHECK-ENCODING: [0x22,0x65,0x11,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1116522 <unknown>

usmlall za.s[w9, 4:7, vgx2], {z12.b, z13.b}, z11.b[11]  // 11000001, 00011011, 00101001, 10100111
// CHECK, INST: usmlall za.s[w9, 4:7, vgx2], { z12.b, z13.b }, z11.b[11]
// CHECK-ENCODING: [0xa7,0x29,0x1b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11b29a7 <unknown>

usmlall za.s[w9, 4:7], {z12.b - z13.b}, z11.b[11]  // 11000001-00011011-00101001-10100111
// CHECK, INST: usmlall za.s[w9, 4:7, vgx2], { z12.b, z13.b }, z11.b[11]
// CHECK-ENCODING: [0xa7,0x29,0x1b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11b29a7 <unknown>


usmlall za.s[w8, 0:3, vgx2], {z0.b, z1.b}, {z0.b, z1.b}  // 11000001, 10100000, 00000000, 00000100
// CHECK, INST: usmlall za.s[w8, 0:3, vgx2], { z0.b, z1.b }, { z0.b, z1.b }
// CHECK-ENCODING: [0x04,0x00,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a00004 <unknown>

usmlall za.s[w8, 0:3], {z0.b - z1.b}, {z0.b - z1.b}  // 11000001-10100000-00000000-00000100
// CHECK, INST: usmlall za.s[w8, 0:3, vgx2], { z0.b, z1.b }, { z0.b, z1.b }
// CHECK-ENCODING: [0x04,0x00,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a00004 <unknown>

usmlall za.s[w10, 4:7, vgx2], {z10.b, z11.b}, {z20.b, z21.b}  // 11000001, 10110100, 01000001, 01000101
// CHECK, INST: usmlall za.s[w10, 4:7, vgx2], { z10.b, z11.b }, { z20.b, z21.b }
// CHECK-ENCODING: [0x45,0x41,0xb4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b44145 <unknown>

usmlall za.s[w10, 4:7], {z10.b - z11.b}, {z20.b - z21.b}  // 11000001-10110100-01000001-01000101
// CHECK, INST: usmlall za.s[w10, 4:7, vgx2], { z10.b, z11.b }, { z20.b, z21.b }
// CHECK-ENCODING: [0x45,0x41,0xb4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b44145 <unknown>

usmlall za.s[w11, 4:7, vgx2], {z12.b, z13.b}, {z8.b, z9.b}  // 11000001, 10101000, 01100001, 10000101
// CHECK, INST: usmlall za.s[w11, 4:7, vgx2], { z12.b, z13.b }, { z8.b, z9.b }
// CHECK-ENCODING: [0x85,0x61,0xa8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a86185 <unknown>

usmlall za.s[w11, 4:7], {z12.b - z13.b}, {z8.b - z9.b}  // 11000001-10101000-01100001-10000101
// CHECK, INST: usmlall za.s[w11, 4:7, vgx2], { z12.b, z13.b }, { z8.b, z9.b }
// CHECK-ENCODING: [0x85,0x61,0xa8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a86185 <unknown>

usmlall za.s[w11, 4:7, vgx2], {z30.b, z31.b}, {z30.b, z31.b}  // 11000001, 10111110, 01100011, 11000101
// CHECK, INST: usmlall za.s[w11, 4:7, vgx2], { z30.b, z31.b }, { z30.b, z31.b }
// CHECK-ENCODING: [0xc5,0x63,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1be63c5 <unknown>

usmlall za.s[w11, 4:7], {z30.b - z31.b}, {z30.b - z31.b}  // 11000001-10111110-01100011-11000101
// CHECK, INST: usmlall za.s[w11, 4:7, vgx2], { z30.b, z31.b }, { z30.b, z31.b }
// CHECK-ENCODING: [0xc5,0x63,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1be63c5 <unknown>

usmlall za.s[w8, 4:7, vgx2], {z16.b, z17.b}, {z16.b, z17.b}  // 11000001, 10110000, 00000010, 00000101
// CHECK, INST: usmlall za.s[w8, 4:7, vgx2], { z16.b, z17.b }, { z16.b, z17.b }
// CHECK-ENCODING: [0x05,0x02,0xb0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b00205 <unknown>

usmlall za.s[w8, 4:7], {z16.b - z17.b}, {z16.b - z17.b}  // 11000001-10110000-00000010-00000101
// CHECK, INST: usmlall za.s[w8, 4:7, vgx2], { z16.b, z17.b }, { z16.b, z17.b }
// CHECK-ENCODING: [0x05,0x02,0xb0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b00205 <unknown>

usmlall za.s[w8, 4:7, vgx2], {z0.b, z1.b}, {z30.b, z31.b}  // 11000001, 10111110, 00000000, 00000101
// CHECK, INST: usmlall za.s[w8, 4:7, vgx2], { z0.b, z1.b }, { z30.b, z31.b }
// CHECK-ENCODING: [0x05,0x00,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1be0005 <unknown>

usmlall za.s[w8, 4:7], {z0.b - z1.b}, {z30.b - z31.b}  // 11000001-10111110-00000000-00000101
// CHECK, INST: usmlall za.s[w8, 4:7, vgx2], { z0.b, z1.b }, { z30.b, z31.b }
// CHECK-ENCODING: [0x05,0x00,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1be0005 <unknown>

usmlall za.s[w10, 0:3, vgx2], {z18.b, z19.b}, {z20.b, z21.b}  // 11000001, 10110100, 01000010, 01000100
// CHECK, INST: usmlall za.s[w10, 0:3, vgx2], { z18.b, z19.b }, { z20.b, z21.b }
// CHECK-ENCODING: [0x44,0x42,0xb4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b44244 <unknown>

usmlall za.s[w10, 0:3], {z18.b - z19.b}, {z20.b - z21.b}  // 11000001-10110100-01000010-01000100
// CHECK, INST: usmlall za.s[w10, 0:3, vgx2], { z18.b, z19.b }, { z20.b, z21.b }
// CHECK-ENCODING: [0x44,0x42,0xb4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b44244 <unknown>

usmlall za.s[w8, 0:3, vgx2], {z12.b, z13.b}, {z2.b, z3.b}  // 11000001, 10100010, 00000001, 10000100
// CHECK, INST: usmlall za.s[w8, 0:3, vgx2], { z12.b, z13.b }, { z2.b, z3.b }
// CHECK-ENCODING: [0x84,0x01,0xa2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a20184 <unknown>

usmlall za.s[w8, 0:3], {z12.b - z13.b}, {z2.b - z3.b}  // 11000001-10100010-00000001-10000100
// CHECK, INST: usmlall za.s[w8, 0:3, vgx2], { z12.b, z13.b }, { z2.b, z3.b }
// CHECK-ENCODING: [0x84,0x01,0xa2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a20184 <unknown>

usmlall za.s[w10, 4:7, vgx2], {z0.b, z1.b}, {z26.b, z27.b}  // 11000001, 10111010, 01000000, 00000101
// CHECK, INST: usmlall za.s[w10, 4:7, vgx2], { z0.b, z1.b }, { z26.b, z27.b }
// CHECK-ENCODING: [0x05,0x40,0xba,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1ba4005 <unknown>

usmlall za.s[w10, 4:7], {z0.b - z1.b}, {z26.b - z27.b}  // 11000001-10111010-01000000-00000101
// CHECK, INST: usmlall za.s[w10, 4:7, vgx2], { z0.b, z1.b }, { z26.b, z27.b }
// CHECK-ENCODING: [0x05,0x40,0xba,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1ba4005 <unknown>

usmlall za.s[w8, 4:7, vgx2], {z22.b, z23.b}, {z30.b, z31.b}  // 11000001, 10111110, 00000010, 11000101
// CHECK, INST: usmlall za.s[w8, 4:7, vgx2], { z22.b, z23.b }, { z30.b, z31.b }
// CHECK-ENCODING: [0xc5,0x02,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1be02c5 <unknown>

usmlall za.s[w8, 4:7], {z22.b - z23.b}, {z30.b - z31.b}  // 11000001-10111110-00000010-11000101
// CHECK, INST: usmlall za.s[w8, 4:7, vgx2], { z22.b, z23.b }, { z30.b, z31.b }
// CHECK-ENCODING: [0xc5,0x02,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1be02c5 <unknown>

usmlall za.s[w11, 0:3, vgx2], {z8.b, z9.b}, {z0.b, z1.b}  // 11000001, 10100000, 01100001, 00000100
// CHECK, INST: usmlall za.s[w11, 0:3, vgx2], { z8.b, z9.b }, { z0.b, z1.b }
// CHECK-ENCODING: [0x04,0x61,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a06104 <unknown>

usmlall za.s[w11, 0:3], {z8.b - z9.b}, {z0.b - z1.b}  // 11000001-10100000-01100001-00000100
// CHECK, INST: usmlall za.s[w11, 0:3, vgx2], { z8.b, z9.b }, { z0.b, z1.b }
// CHECK-ENCODING: [0x04,0x61,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a06104 <unknown>

usmlall za.s[w9, 4:7, vgx2], {z12.b, z13.b}, {z10.b, z11.b}  // 11000001, 10101010, 00100001, 10000101
// CHECK, INST: usmlall za.s[w9, 4:7, vgx2], { z12.b, z13.b }, { z10.b, z11.b }
// CHECK-ENCODING: [0x85,0x21,0xaa,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1aa2185 <unknown>

usmlall za.s[w9, 4:7], {z12.b - z13.b}, {z10.b - z11.b}  // 11000001-10101010-00100001-10000101
// CHECK, INST: usmlall za.s[w9, 4:7, vgx2], { z12.b, z13.b }, { z10.b, z11.b }
// CHECK-ENCODING: [0x85,0x21,0xaa,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1aa2185 <unknown>


usmlall za.s[w8, 0:3, vgx4], {z0.b - z3.b}, z0.b  // 11000001-00110000-00000000-00000100
// CHECK-INST: usmlall za.s[w8, 0:3, vgx4], { z0.b - z3.b }, z0.b
// CHECK-ENCODING: [0x04,0x00,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1300004 <unknown>

usmlall za.s[w8, 0:3], {z0.b - z3.b}, z0.b  // 11000001-00110000-00000000-00000100
// CHECK-INST: usmlall za.s[w8, 0:3, vgx4], { z0.b - z3.b }, z0.b
// CHECK-ENCODING: [0x04,0x00,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1300004 <unknown>

usmlall za.s[w10, 4:7, vgx4], {z10.b - z13.b}, z5.b  // 11000001-00110101-01000001-01000101
// CHECK-INST: usmlall za.s[w10, 4:7, vgx4], { z10.b - z13.b }, z5.b
// CHECK-ENCODING: [0x45,0x41,0x35,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1354145 <unknown>

usmlall za.s[w10, 4:7], {z10.b - z13.b}, z5.b  // 11000001-00110101-01000001-01000101
// CHECK-INST: usmlall za.s[w10, 4:7, vgx4], { z10.b - z13.b }, z5.b
// CHECK-ENCODING: [0x45,0x41,0x35,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1354145 <unknown>

usmlall za.s[w11, 4:7, vgx4], {z13.b - z16.b}, z8.b  // 11000001-00111000-01100001-10100101
// CHECK-INST: usmlall za.s[w11, 4:7, vgx4], { z13.b - z16.b }, z8.b
// CHECK-ENCODING: [0xa5,0x61,0x38,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13861a5 <unknown>

usmlall za.s[w11, 4:7], {z13.b - z16.b}, z8.b  // 11000001-00111000-01100001-10100101
// CHECK-INST: usmlall za.s[w11, 4:7, vgx4], { z13.b - z16.b }, z8.b
// CHECK-ENCODING: [0xa5,0x61,0x38,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13861a5 <unknown>

usmlall za.s[w11, 4:7, vgx4], {z31.b - z2.b}, z15.b  // 11000001-00111111-01100011-11100101
// CHECK-INST: usmlall za.s[w11, 4:7, vgx4], { z31.b, z0.b, z1.b, z2.b }, z15.b
// CHECK-ENCODING: [0xe5,0x63,0x3f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13f63e5 <unknown>

usmlall za.s[w11, 4:7], {z31.b - z2.b}, z15.b  // 11000001-00111111-01100011-11100101
// CHECK-INST: usmlall za.s[w11, 4:7, vgx4], { z31.b, z0.b, z1.b, z2.b }, z15.b
// CHECK-ENCODING: [0xe5,0x63,0x3f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13f63e5 <unknown>

usmlall za.s[w8, 4:7, vgx4], {z17.b - z20.b}, z0.b  // 11000001-00110000-00000010-00100101
// CHECK-INST: usmlall za.s[w8, 4:7, vgx4], { z17.b - z20.b }, z0.b
// CHECK-ENCODING: [0x25,0x02,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1300225 <unknown>

usmlall za.s[w8, 4:7], {z17.b - z20.b}, z0.b  // 11000001-00110000-00000010-00100101
// CHECK-INST: usmlall za.s[w8, 4:7, vgx4], { z17.b - z20.b }, z0.b
// CHECK-ENCODING: [0x25,0x02,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1300225 <unknown>

usmlall za.s[w8, 4:7, vgx4], {z1.b - z4.b}, z14.b  // 11000001-00111110-00000000-00100101
// CHECK-INST: usmlall za.s[w8, 4:7, vgx4], { z1.b - z4.b }, z14.b
// CHECK-ENCODING: [0x25,0x00,0x3e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13e0025 <unknown>

usmlall za.s[w8, 4:7], {z1.b - z4.b}, z14.b  // 11000001-00111110-00000000-00100101
// CHECK-INST: usmlall za.s[w8, 4:7, vgx4], { z1.b - z4.b }, z14.b
// CHECK-ENCODING: [0x25,0x00,0x3e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13e0025 <unknown>

usmlall za.s[w10, 0:3, vgx4], {z19.b - z22.b}, z4.b  // 11000001-00110100-01000010-01100100
// CHECK-INST: usmlall za.s[w10, 0:3, vgx4], { z19.b - z22.b }, z4.b
// CHECK-ENCODING: [0x64,0x42,0x34,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1344264 <unknown>

usmlall za.s[w10, 0:3], {z19.b - z22.b}, z4.b  // 11000001-00110100-01000010-01100100
// CHECK-INST: usmlall za.s[w10, 0:3, vgx4], { z19.b - z22.b }, z4.b
// CHECK-ENCODING: [0x64,0x42,0x34,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1344264 <unknown>

usmlall za.s[w8, 0:3, vgx4], {z12.b - z15.b}, z2.b  // 11000001-00110010-00000001-10000100
// CHECK-INST: usmlall za.s[w8, 0:3, vgx4], { z12.b - z15.b }, z2.b
// CHECK-ENCODING: [0x84,0x01,0x32,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1320184 <unknown>

usmlall za.s[w8, 0:3], {z12.b - z15.b}, z2.b  // 11000001-00110010-00000001-10000100
// CHECK-INST: usmlall za.s[w8, 0:3, vgx4], { z12.b - z15.b }, z2.b
// CHECK-ENCODING: [0x84,0x01,0x32,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1320184 <unknown>

usmlall za.s[w10, 4:7, vgx4], {z1.b - z4.b}, z10.b  // 11000001-00111010-01000000-00100101
// CHECK-INST: usmlall za.s[w10, 4:7, vgx4], { z1.b - z4.b }, z10.b
// CHECK-ENCODING: [0x25,0x40,0x3a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13a4025 <unknown>

usmlall za.s[w10, 4:7], {z1.b - z4.b}, z10.b  // 11000001-00111010-01000000-00100101
// CHECK-INST: usmlall za.s[w10, 4:7, vgx4], { z1.b - z4.b }, z10.b
// CHECK-ENCODING: [0x25,0x40,0x3a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13a4025 <unknown>

usmlall za.s[w8, 4:7, vgx4], {z22.b - z25.b}, z14.b  // 11000001-00111110-00000010-11000101
// CHECK-INST: usmlall za.s[w8, 4:7, vgx4], { z22.b - z25.b }, z14.b
// CHECK-ENCODING: [0xc5,0x02,0x3e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13e02c5 <unknown>

usmlall za.s[w8, 4:7], {z22.b - z25.b}, z14.b  // 11000001-00111110-00000010-11000101
// CHECK-INST: usmlall za.s[w8, 4:7, vgx4], { z22.b - z25.b }, z14.b
// CHECK-ENCODING: [0xc5,0x02,0x3e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13e02c5 <unknown>

usmlall za.s[w11, 0:3, vgx4], {z9.b - z12.b}, z1.b  // 11000001-00110001-01100001-00100100
// CHECK-INST: usmlall za.s[w11, 0:3, vgx4], { z9.b - z12.b }, z1.b
// CHECK-ENCODING: [0x24,0x61,0x31,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1316124 <unknown>

usmlall za.s[w11, 0:3], {z9.b - z12.b}, z1.b  // 11000001-00110001-01100001-00100100
// CHECK-INST: usmlall za.s[w11, 0:3, vgx4], { z9.b - z12.b }, z1.b
// CHECK-ENCODING: [0x24,0x61,0x31,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1316124 <unknown>

usmlall za.s[w9, 4:7, vgx4], {z12.b - z15.b}, z11.b  // 11000001-00111011-00100001-10000101
// CHECK-INST: usmlall za.s[w9, 4:7, vgx4], { z12.b - z15.b }, z11.b
// CHECK-ENCODING: [0x85,0x21,0x3b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13b2185 <unknown>

usmlall za.s[w9, 4:7], {z12.b - z15.b}, z11.b  // 11000001-00111011-00100001-10000101
// CHECK-INST: usmlall za.s[w9, 4:7, vgx4], { z12.b - z15.b }, z11.b
// CHECK-ENCODING: [0x85,0x21,0x3b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13b2185 <unknown>


usmlall za.s[w8, 0:3, vgx4], {z0.b - z3.b}, z0.b[0]  // 11000001-00010000-10000000-00100000
// CHECK-INST: usmlall za.s[w8, 0:3, vgx4], { z0.b - z3.b }, z0.b[0]
// CHECK-ENCODING: [0x20,0x80,0x10,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1108020 <unknown>

usmlall za.s[w8, 0:3], {z0.b - z3.b}, z0.b[0]  // 11000001-00010000-10000000-00100000
// CHECK-INST: usmlall za.s[w8, 0:3, vgx4], { z0.b - z3.b }, z0.b[0]
// CHECK-ENCODING: [0x20,0x80,0x10,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1108020 <unknown>

usmlall za.s[w10, 4:7, vgx4], {z8.b - z11.b}, z5.b[6]  // 11000001-00010101-11000101-00100101
// CHECK-INST: usmlall za.s[w10, 4:7, vgx4], { z8.b - z11.b }, z5.b[6]
// CHECK-ENCODING: [0x25,0xc5,0x15,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c115c525 <unknown>

usmlall za.s[w10, 4:7], {z8.b - z11.b}, z5.b[6]  // 11000001-00010101-11000101-00100101
// CHECK-INST: usmlall za.s[w10, 4:7, vgx4], { z8.b - z11.b }, z5.b[6]
// CHECK-ENCODING: [0x25,0xc5,0x15,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c115c525 <unknown>

usmlall za.s[w11, 4:7, vgx4], {z12.b - z15.b}, z8.b[15]  // 11000001-00011000-11101101-10100111
// CHECK-INST: usmlall za.s[w11, 4:7, vgx4], { z12.b - z15.b }, z8.b[15]
// CHECK-ENCODING: [0xa7,0xed,0x18,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c118eda7 <unknown>

usmlall za.s[w11, 4:7], {z12.b - z15.b}, z8.b[15]  // 11000001-00011000-11101101-10100111
// CHECK-INST: usmlall za.s[w11, 4:7, vgx4], { z12.b - z15.b }, z8.b[15]
// CHECK-ENCODING: [0xa7,0xed,0x18,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c118eda7 <unknown>

usmlall za.s[w11, 4:7, vgx4], {z28.b - z31.b}, z15.b[15]  // 11000001-00011111-11101111-10100111
// CHECK-INST: usmlall za.s[w11, 4:7, vgx4], { z28.b - z31.b }, z15.b[15]
// CHECK-ENCODING: [0xa7,0xef,0x1f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11fefa7 <unknown>

usmlall za.s[w11, 4:7], {z28.b - z31.b}, z15.b[15]  // 11000001-00011111-11101111-10100111
// CHECK-INST: usmlall za.s[w11, 4:7, vgx4], { z28.b - z31.b }, z15.b[15]
// CHECK-ENCODING: [0xa7,0xef,0x1f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11fefa7 <unknown>

usmlall za.s[w8, 4:7, vgx4], {z16.b - z19.b}, z0.b[14]  // 11000001-00010000-10001110-00100101
// CHECK-INST: usmlall za.s[w8, 4:7, vgx4], { z16.b - z19.b }, z0.b[14]
// CHECK-ENCODING: [0x25,0x8e,0x10,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1108e25 <unknown>

usmlall za.s[w8, 4:7], {z16.b - z19.b}, z0.b[14]  // 11000001-00010000-10001110-00100101
// CHECK-INST: usmlall za.s[w8, 4:7, vgx4], { z16.b - z19.b }, z0.b[14]
// CHECK-ENCODING: [0x25,0x8e,0x10,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1108e25 <unknown>

usmlall za.s[w8, 4:7, vgx4], {z0.b - z3.b}, z14.b[4]  // 11000001-00011110-10000100-00100001
// CHECK-INST: usmlall za.s[w8, 4:7, vgx4], { z0.b - z3.b }, z14.b[4]
// CHECK-ENCODING: [0x21,0x84,0x1e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11e8421 <unknown>

usmlall za.s[w8, 4:7], {z0.b - z3.b}, z14.b[4]  // 11000001-00011110-10000100-00100001
// CHECK-INST: usmlall za.s[w8, 4:7, vgx4], { z0.b - z3.b }, z14.b[4]
// CHECK-ENCODING: [0x21,0x84,0x1e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11e8421 <unknown>

usmlall za.s[w10, 0:3, vgx4], {z16.b - z19.b}, z4.b[4]  // 11000001-00010100-11000110-00100000
// CHECK-INST: usmlall za.s[w10, 0:3, vgx4], { z16.b - z19.b }, z4.b[4]
// CHECK-ENCODING: [0x20,0xc6,0x14,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c114c620 <unknown>

usmlall za.s[w10, 0:3], {z16.b - z19.b}, z4.b[4]  // 11000001-00010100-11000110-00100000
// CHECK-INST: usmlall za.s[w10, 0:3, vgx4], { z16.b - z19.b }, z4.b[4]
// CHECK-ENCODING: [0x20,0xc6,0x14,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c114c620 <unknown>

usmlall za.s[w8, 0:3, vgx4], {z12.b - z15.b}, z2.b[8]  // 11000001-00010010-10001001-10100000
// CHECK-INST: usmlall za.s[w8, 0:3, vgx4], { z12.b - z15.b }, z2.b[8]
// CHECK-ENCODING: [0xa0,0x89,0x12,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11289a0 <unknown>

usmlall za.s[w8, 0:3], {z12.b - z15.b}, z2.b[8]  // 11000001-00010010-10001001-10100000
// CHECK-INST: usmlall za.s[w8, 0:3, vgx4], { z12.b - z15.b }, z2.b[8]
// CHECK-ENCODING: [0xa0,0x89,0x12,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11289a0 <unknown>

usmlall za.s[w10, 4:7, vgx4], {z0.b - z3.b}, z10.b[8]  // 11000001-00011010-11001000-00100001
// CHECK-INST: usmlall za.s[w10, 4:7, vgx4], { z0.b - z3.b }, z10.b[8]
// CHECK-ENCODING: [0x21,0xc8,0x1a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11ac821 <unknown>

usmlall za.s[w10, 4:7], {z0.b - z3.b}, z10.b[8]  // 11000001-00011010-11001000-00100001
// CHECK-INST: usmlall za.s[w10, 4:7, vgx4], { z0.b - z3.b }, z10.b[8]
// CHECK-ENCODING: [0x21,0xc8,0x1a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11ac821 <unknown>

usmlall za.s[w8, 4:7, vgx4], {z20.b - z23.b}, z14.b[10]  // 11000001-00011110-10001010-10100101
// CHECK-INST: usmlall za.s[w8, 4:7, vgx4], { z20.b - z23.b }, z14.b[10]
// CHECK-ENCODING: [0xa5,0x8a,0x1e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11e8aa5 <unknown>

usmlall za.s[w8, 4:7], {z20.b - z23.b}, z14.b[10]  // 11000001-00011110-10001010-10100101
// CHECK-INST: usmlall za.s[w8, 4:7, vgx4], { z20.b - z23.b }, z14.b[10]
// CHECK-ENCODING: [0xa5,0x8a,0x1e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11e8aa5 <unknown>

usmlall za.s[w11, 0:3, vgx4], {z8.b - z11.b}, z1.b[5]  // 11000001-00010001-11100101-00100010
// CHECK-INST: usmlall za.s[w11, 0:3, vgx4], { z8.b - z11.b }, z1.b[5]
// CHECK-ENCODING: [0x22,0xe5,0x11,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c111e522 <unknown>

usmlall za.s[w11, 0:3], {z8.b - z11.b}, z1.b[5]  // 11000001-00010001-11100101-00100010
// CHECK-INST: usmlall za.s[w11, 0:3, vgx4], { z8.b - z11.b }, z1.b[5]
// CHECK-ENCODING: [0x22,0xe5,0x11,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c111e522 <unknown>

usmlall za.s[w9, 4:7, vgx4], {z12.b - z15.b}, z11.b[11]  // 11000001-00011011-10101001-10100111
// CHECK-INST: usmlall za.s[w9, 4:7, vgx4], { z12.b - z15.b }, z11.b[11]
// CHECK-ENCODING: [0xa7,0xa9,0x1b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11ba9a7 <unknown>

usmlall za.s[w9, 4:7], {z12.b - z15.b}, z11.b[11]  // 11000001-00011011-10101001-10100111
// CHECK-INST: usmlall za.s[w9, 4:7, vgx4], { z12.b - z15.b }, z11.b[11]
// CHECK-ENCODING: [0xa7,0xa9,0x1b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11ba9a7 <unknown>


usmlall za.s[w8, 0:3, vgx4], {z0.b - z3.b}, {z0.b - z3.b}  // 11000001-10100001-00000000-00000100
// CHECK-INST: usmlall za.s[w8, 0:3, vgx4], { z0.b - z3.b }, { z0.b - z3.b }
// CHECK-ENCODING: [0x04,0x00,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a10004 <unknown>

usmlall za.s[w8, 0:3], {z0.b - z3.b}, {z0.b - z3.b}  // 11000001-10100001-00000000-00000100
// CHECK-INST: usmlall za.s[w8, 0:3, vgx4], { z0.b - z3.b }, { z0.b - z3.b }
// CHECK-ENCODING: [0x04,0x00,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a10004 <unknown>

usmlall za.s[w10, 4:7, vgx4], {z8.b - z11.b}, {z20.b - z23.b}  // 11000001-10110101-01000001-00000101
// CHECK-INST: usmlall za.s[w10, 4:7, vgx4], { z8.b - z11.b }, { z20.b - z23.b }
// CHECK-ENCODING: [0x05,0x41,0xb5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b54105 <unknown>

usmlall za.s[w10, 4:7], {z8.b - z11.b}, {z20.b - z23.b}  // 11000001-10110101-01000001-00000101
// CHECK-INST: usmlall za.s[w10, 4:7, vgx4], { z8.b - z11.b }, { z20.b - z23.b }
// CHECK-ENCODING: [0x05,0x41,0xb5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b54105 <unknown>

usmlall za.s[w11, 4:7, vgx4], {z12.b - z15.b}, {z8.b - z11.b}  // 11000001-10101001-01100001-10000101
// CHECK-INST: usmlall za.s[w11, 4:7, vgx4], { z12.b - z15.b }, { z8.b - z11.b }
// CHECK-ENCODING: [0x85,0x61,0xa9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a96185 <unknown>

usmlall za.s[w11, 4:7], {z12.b - z15.b}, {z8.b - z11.b}  // 11000001-10101001-01100001-10000101
// CHECK-INST: usmlall za.s[w11, 4:7, vgx4], { z12.b - z15.b }, { z8.b - z11.b }
// CHECK-ENCODING: [0x85,0x61,0xa9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a96185 <unknown>

usmlall za.s[w11, 4:7, vgx4], {z28.b - z31.b}, {z28.b - z31.b}  // 11000001-10111101-01100011-10000101
// CHECK-INST: usmlall za.s[w11, 4:7, vgx4], { z28.b - z31.b }, { z28.b - z31.b }
// CHECK-ENCODING: [0x85,0x63,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bd6385 <unknown>

usmlall za.s[w11, 4:7], {z28.b - z31.b}, {z28.b - z31.b}  // 11000001-10111101-01100011-10000101
// CHECK-INST: usmlall za.s[w11, 4:7, vgx4], { z28.b - z31.b }, { z28.b - z31.b }
// CHECK-ENCODING: [0x85,0x63,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bd6385 <unknown>

usmlall za.s[w8, 4:7, vgx4], {z16.b - z19.b}, {z16.b - z19.b}  // 11000001-10110001-00000010-00000101
// CHECK-INST: usmlall za.s[w8, 4:7, vgx4], { z16.b - z19.b }, { z16.b - z19.b }
// CHECK-ENCODING: [0x05,0x02,0xb1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b10205 <unknown>

usmlall za.s[w8, 4:7], {z16.b - z19.b}, {z16.b - z19.b}  // 11000001-10110001-00000010-00000101
// CHECK-INST: usmlall za.s[w8, 4:7, vgx4], { z16.b - z19.b }, { z16.b - z19.b }
// CHECK-ENCODING: [0x05,0x02,0xb1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b10205 <unknown>

usmlall za.s[w8, 4:7, vgx4], {z0.b - z3.b}, {z28.b - z31.b}  // 11000001-10111101-00000000-00000101
// CHECK-INST: usmlall za.s[w8, 4:7, vgx4], { z0.b - z3.b }, { z28.b - z31.b }
// CHECK-ENCODING: [0x05,0x00,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bd0005 <unknown>

usmlall za.s[w8, 4:7], {z0.b - z3.b}, {z28.b - z31.b}  // 11000001-10111101-00000000-00000101
// CHECK-INST: usmlall za.s[w8, 4:7, vgx4], { z0.b - z3.b }, { z28.b - z31.b }
// CHECK-ENCODING: [0x05,0x00,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bd0005 <unknown>

usmlall za.s[w10, 0:3, vgx4], {z16.b - z19.b}, {z20.b - z23.b}  // 11000001-10110101-01000010-00000100
// CHECK-INST: usmlall za.s[w10, 0:3, vgx4], { z16.b - z19.b }, { z20.b - z23.b }
// CHECK-ENCODING: [0x04,0x42,0xb5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b54204 <unknown>

usmlall za.s[w10, 0:3], {z16.b - z19.b}, {z20.b - z23.b}  // 11000001-10110101-01000010-00000100
// CHECK-INST: usmlall za.s[w10, 0:3, vgx4], { z16.b - z19.b }, { z20.b - z23.b }
// CHECK-ENCODING: [0x04,0x42,0xb5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b54204 <unknown>

usmlall za.s[w8, 0:3, vgx4], {z12.b - z15.b}, {z0.b - z3.b}  // 11000001-10100001-00000001-10000100
// CHECK-INST: usmlall za.s[w8, 0:3, vgx4], { z12.b - z15.b }, { z0.b - z3.b }
// CHECK-ENCODING: [0x84,0x01,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a10184 <unknown>

usmlall za.s[w8, 0:3], {z12.b - z15.b}, {z0.b - z3.b}  // 11000001-10100001-00000001-10000100
// CHECK-INST: usmlall za.s[w8, 0:3, vgx4], { z12.b - z15.b }, { z0.b - z3.b }
// CHECK-ENCODING: [0x84,0x01,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a10184 <unknown>

usmlall za.s[w10, 4:7, vgx4], {z0.b - z3.b}, {z24.b - z27.b}  // 11000001-10111001-01000000-00000101
// CHECK-INST: usmlall za.s[w10, 4:7, vgx4], { z0.b - z3.b }, { z24.b - z27.b }
// CHECK-ENCODING: [0x05,0x40,0xb9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b94005 <unknown>

usmlall za.s[w10, 4:7], {z0.b - z3.b}, {z24.b - z27.b}  // 11000001-10111001-01000000-00000101
// CHECK-INST: usmlall za.s[w10, 4:7, vgx4], { z0.b - z3.b }, { z24.b - z27.b }
// CHECK-ENCODING: [0x05,0x40,0xb9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b94005 <unknown>

usmlall za.s[w8, 4:7, vgx4], {z20.b - z23.b}, {z28.b - z31.b}  // 11000001-10111101-00000010-10000101
// CHECK-INST: usmlall za.s[w8, 4:7, vgx4], { z20.b - z23.b }, { z28.b - z31.b }
// CHECK-ENCODING: [0x85,0x02,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bd0285 <unknown>

usmlall za.s[w8, 4:7], {z20.b - z23.b}, {z28.b - z31.b}  // 11000001-10111101-00000010-10000101
// CHECK-INST: usmlall za.s[w8, 4:7, vgx4], { z20.b - z23.b }, { z28.b - z31.b }
// CHECK-ENCODING: [0x85,0x02,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bd0285 <unknown>

usmlall za.s[w11, 0:3, vgx4], {z8.b - z11.b}, {z0.b - z3.b}  // 11000001-10100001-01100001-00000100
// CHECK-INST: usmlall za.s[w11, 0:3, vgx4], { z8.b - z11.b }, { z0.b - z3.b }
// CHECK-ENCODING: [0x04,0x61,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a16104 <unknown>

usmlall za.s[w11, 0:3], {z8.b - z11.b}, {z0.b - z3.b}  // 11000001-10100001-01100001-00000100
// CHECK-INST: usmlall za.s[w11, 0:3, vgx4], { z8.b - z11.b }, { z0.b - z3.b }
// CHECK-ENCODING: [0x04,0x61,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a16104 <unknown>

usmlall za.s[w9, 4:7, vgx4], {z12.b - z15.b}, {z8.b - z11.b}  // 11000001-10101001-00100001-10000101
// CHECK-INST: usmlall za.s[w9, 4:7, vgx4], { z12.b - z15.b }, { z8.b - z11.b }
// CHECK-ENCODING: [0x85,0x21,0xa9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a92185 <unknown>

usmlall za.s[w9, 4:7], {z12.b - z15.b}, {z8.b - z11.b}  // 11000001-10101001-00100001-10000101
// CHECK-INST: usmlall za.s[w9, 4:7, vgx4], { z12.b - z15.b }, { z8.b - z11.b }
// CHECK-ENCODING: [0x85,0x21,0xa9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a92185 <unknown>

