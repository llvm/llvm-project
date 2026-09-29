// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2,+sme-i16i64 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2,+sme-i16i64 < %s \
// RUN:        | llvm-objdump --no-print-imm-hex -d --mattr=+sme2,+sme-i16i64 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2,+sme-i16i64 < %s \
// RUN:        | llvm-objdump --no-print-imm-hex -d --mattr=-sme2 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2,+sme-i16i64 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2,+sme-i16i64 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST


umlsll  za.s[w8, 0:3], z0.b, z0.b  // 11000001-00100000-00000100-00011000
// CHECK-INST: umlsll  za.s[w8, 0:3], z0.b, z0.b
// CHECK-ENCODING: [0x18,0x04,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1200418 <unknown>

umlsll  za.s[w10, 4:7], z10.b, z5.b  // 11000001-00100101-01000101-01011001
// CHECK-INST: umlsll  za.s[w10, 4:7], z10.b, z5.b
// CHECK-ENCODING: [0x59,0x45,0x25,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1254559 <unknown>

umlsll  za.s[w11, 12:15], z13.b, z8.b  // 11000001-00101000-01100101-10111011
// CHECK-INST: umlsll  za.s[w11, 12:15], z13.b, z8.b
// CHECK-ENCODING: [0xbb,0x65,0x28,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12865bb <unknown>

umlsll  za.s[w11, 12:15], z31.b, z15.b  // 11000001-00101111-01100111-11111011
// CHECK-INST: umlsll  za.s[w11, 12:15], z31.b, z15.b
// CHECK-ENCODING: [0xfb,0x67,0x2f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12f67fb <unknown>

umlsll  za.s[w8, 4:7], z17.b, z0.b  // 11000001-00100000-00000110-00111001
// CHECK-INST: umlsll  za.s[w8, 4:7], z17.b, z0.b
// CHECK-ENCODING: [0x39,0x06,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1200639 <unknown>

umlsll  za.s[w8, 4:7], z1.b, z14.b  // 11000001-00101110-00000100-00111001
// CHECK-INST: umlsll  za.s[w8, 4:7], z1.b, z14.b
// CHECK-ENCODING: [0x39,0x04,0x2e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12e0439 <unknown>

umlsll  za.s[w10, 0:3], z19.b, z4.b  // 11000001-00100100-01000110-01111000
// CHECK-INST: umlsll  za.s[w10, 0:3], z19.b, z4.b
// CHECK-ENCODING: [0x78,0x46,0x24,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1244678 <unknown>

umlsll  za.s[w8, 0:3], z12.b, z2.b  // 11000001-00100010-00000101-10011000
// CHECK-INST: umlsll  za.s[w8, 0:3], z12.b, z2.b
// CHECK-ENCODING: [0x98,0x05,0x22,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1220598 <unknown>

umlsll  za.s[w10, 4:7], z1.b, z10.b  // 11000001-00101010-01000100-00111001
// CHECK-INST: umlsll  za.s[w10, 4:7], z1.b, z10.b
// CHECK-ENCODING: [0x39,0x44,0x2a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12a4439 <unknown>

umlsll  za.s[w8, 4:7], z22.b, z14.b  // 11000001-00101110-00000110-11011001
// CHECK-INST: umlsll  za.s[w8, 4:7], z22.b, z14.b
// CHECK-ENCODING: [0xd9,0x06,0x2e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12e06d9 <unknown>

umlsll  za.s[w11, 8:11], z9.b, z1.b  // 11000001-00100001-01100101-00111010
// CHECK-INST: umlsll  za.s[w11, 8:11], z9.b, z1.b
// CHECK-ENCODING: [0x3a,0x65,0x21,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c121653a <unknown>

umlsll  za.s[w9, 12:15], z12.b, z11.b  // 11000001-00101011-00100101-10011011
// CHECK-INST: umlsll  za.s[w9, 12:15], z12.b, z11.b
// CHECK-ENCODING: [0x9b,0x25,0x2b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12b259b <unknown>


umlsll  za.s[w8, 0:3], z0.b, z0.b[0]  // 11000001-00000000-00000000-00011000
// CHECK-INST: umlsll  za.s[w8, 0:3], z0.b, z0.b[0]
// CHECK-ENCODING: [0x18,0x00,0x00,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1000018 <unknown>

umlsll  za.s[w10, 4:7], z10.b, z5.b[5]  // 11000001-00000101-01010101-01011001
// CHECK-INST: umlsll  za.s[w10, 4:7], z10.b, z5.b[5]
// CHECK-ENCODING: [0x59,0x55,0x05,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1055559 <unknown>

umlsll  za.s[w11, 12:15], z13.b, z8.b[11]  // 11000001-00001000-11101101-10111011
// CHECK-INST: umlsll  za.s[w11, 12:15], z13.b, z8.b[11]
// CHECK-ENCODING: [0xbb,0xed,0x08,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c108edbb <unknown>

umlsll  za.s[w11, 12:15], z31.b, z15.b[15]  // 11000001-00001111-11111111-11111011
// CHECK-INST: umlsll  za.s[w11, 12:15], z31.b, z15.b[15]
// CHECK-ENCODING: [0xfb,0xff,0x0f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c10ffffb <unknown>

umlsll  za.s[w8, 4:7], z17.b, z0.b[3]  // 11000001-00000000-00001110-00111001
// CHECK-INST: umlsll  za.s[w8, 4:7], z17.b, z0.b[3]
// CHECK-ENCODING: [0x39,0x0e,0x00,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1000e39 <unknown>

umlsll  za.s[w8, 4:7], z1.b, z14.b[9]  // 11000001-00001110-10000100-00111001
// CHECK-INST: umlsll  za.s[w8, 4:7], z1.b, z14.b[9]
// CHECK-ENCODING: [0x39,0x84,0x0e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c10e8439 <unknown>

umlsll  za.s[w10, 0:3], z19.b, z4.b[5]  // 11000001-00000100-01010110-01111000
// CHECK-INST: umlsll  za.s[w10, 0:3], z19.b, z4.b[5]
// CHECK-ENCODING: [0x78,0x56,0x04,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1045678 <unknown>

umlsll  za.s[w8, 0:3], z12.b, z2.b[6]  // 11000001-00000010-00011001-10011000
// CHECK-INST: umlsll  za.s[w8, 0:3], z12.b, z2.b[6]
// CHECK-ENCODING: [0x98,0x19,0x02,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1021998 <unknown>

umlsll  za.s[w10, 4:7], z1.b, z10.b[10]  // 11000001-00001010-11001000-00111001
// CHECK-INST: umlsll  za.s[w10, 4:7], z1.b, z10.b[10]
// CHECK-ENCODING: [0x39,0xc8,0x0a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c10ac839 <unknown>

umlsll  za.s[w8, 4:7], z22.b, z14.b[2]  // 11000001-00001110-00001010-11011001
// CHECK-INST: umlsll  za.s[w8, 4:7], z22.b, z14.b[2]
// CHECK-ENCODING: [0xd9,0x0a,0x0e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c10e0ad9 <unknown>

umlsll  za.s[w11, 8:11], z9.b, z1.b[13]  // 11000001-00000001-11110101-00111010
// CHECK-INST: umlsll  za.s[w11, 8:11], z9.b, z1.b[13]
// CHECK-ENCODING: [0x3a,0xf5,0x01,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c101f53a <unknown>

umlsll  za.s[w9, 12:15], z12.b, z11.b[10]  // 11000001-00001011-10101001-10011011
// CHECK-INST: umlsll  za.s[w9, 12:15], z12.b, z11.b[10]
// CHECK-ENCODING: [0x9b,0xa9,0x0b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c10ba99b <unknown>


umlsll  za.d[w8, 0:3], z0.h, z0.h  // 11000001-01100000-00000100-00011000
// CHECK-INST: umlsll  za.d[w8, 0:3], z0.h, z0.h
// CHECK-ENCODING: [0x18,0x04,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1600418 <unknown>

umlsll  za.d[w10, 4:7], z10.h, z5.h  // 11000001-01100101-01000101-01011001
// CHECK-INST: umlsll  za.d[w10, 4:7], z10.h, z5.h
// CHECK-ENCODING: [0x59,0x45,0x65,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1654559 <unknown>

umlsll  za.d[w11, 12:15], z13.h, z8.h  // 11000001-01101000-01100101-10111011
// CHECK-INST: umlsll  za.d[w11, 12:15], z13.h, z8.h
// CHECK-ENCODING: [0xbb,0x65,0x68,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16865bb <unknown>

umlsll  za.d[w11, 12:15], z31.h, z15.h  // 11000001-01101111-01100111-11111011
// CHECK-INST: umlsll  za.d[w11, 12:15], z31.h, z15.h
// CHECK-ENCODING: [0xfb,0x67,0x6f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16f67fb <unknown>

umlsll  za.d[w8, 4:7], z17.h, z0.h  // 11000001-01100000-00000110-00111001
// CHECK-INST: umlsll  za.d[w8, 4:7], z17.h, z0.h
// CHECK-ENCODING: [0x39,0x06,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1600639 <unknown>

umlsll  za.d[w8, 4:7], z1.h, z14.h  // 11000001-01101110-00000100-00111001
// CHECK-INST: umlsll  za.d[w8, 4:7], z1.h, z14.h
// CHECK-ENCODING: [0x39,0x04,0x6e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16e0439 <unknown>

umlsll  za.d[w10, 0:3], z19.h, z4.h  // 11000001-01100100-01000110-01111000
// CHECK-INST: umlsll  za.d[w10, 0:3], z19.h, z4.h
// CHECK-ENCODING: [0x78,0x46,0x64,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1644678 <unknown>

umlsll  za.d[w8, 0:3], z12.h, z2.h  // 11000001-01100010-00000101-10011000
// CHECK-INST: umlsll  za.d[w8, 0:3], z12.h, z2.h
// CHECK-ENCODING: [0x98,0x05,0x62,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1620598 <unknown>

umlsll  za.d[w10, 4:7], z1.h, z10.h  // 11000001-01101010-01000100-00111001
// CHECK-INST: umlsll  za.d[w10, 4:7], z1.h, z10.h
// CHECK-ENCODING: [0x39,0x44,0x6a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16a4439 <unknown>

umlsll  za.d[w8, 4:7], z22.h, z14.h  // 11000001-01101110-00000110-11011001
// CHECK-INST: umlsll  za.d[w8, 4:7], z22.h, z14.h
// CHECK-ENCODING: [0xd9,0x06,0x6e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16e06d9 <unknown>

umlsll  za.d[w11, 8:11], z9.h, z1.h  // 11000001-01100001-01100101-00111010
// CHECK-INST: umlsll  za.d[w11, 8:11], z9.h, z1.h
// CHECK-ENCODING: [0x3a,0x65,0x61,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c161653a <unknown>

umlsll  za.d[w9, 12:15], z12.h, z11.h  // 11000001-01101011-00100101-10011011
// CHECK-INST: umlsll  za.d[w9, 12:15], z12.h, z11.h
// CHECK-ENCODING: [0x9b,0x25,0x6b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16b259b <unknown>


umlsll  za.d[w8, 0:3], z0.h, z0.h[0]  // 11000001-10000000-00000000-00011000
// CHECK-INST: umlsll  za.d[w8, 0:3], z0.h, z0.h[0]
// CHECK-ENCODING: [0x18,0x00,0x80,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1800018 <unknown>

umlsll  za.d[w10, 4:7], z10.h, z5.h[1]  // 11000001-10000101-01000101-01011001
// CHECK-INST: umlsll  za.d[w10, 4:7], z10.h, z5.h[1]
// CHECK-ENCODING: [0x59,0x45,0x85,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1854559 <unknown>

umlsll  za.d[w11, 12:15], z13.h, z8.h[7]  // 11000001-10001000-11101101-10111011
// CHECK-INST: umlsll  za.d[w11, 12:15], z13.h, z8.h[7]
// CHECK-ENCODING: [0xbb,0xed,0x88,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c188edbb <unknown>

umlsll  za.d[w11, 12:15], z31.h, z15.h[7]  // 11000001-10001111-11101111-11111011
// CHECK-INST: umlsll  za.d[w11, 12:15], z31.h, z15.h[7]
// CHECK-ENCODING: [0xfb,0xef,0x8f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c18feffb <unknown>

umlsll  za.d[w8, 4:7], z17.h, z0.h[3]  // 11000001-10000000-00001110-00111001
// CHECK-INST: umlsll  za.d[w8, 4:7], z17.h, z0.h[3]
// CHECK-ENCODING: [0x39,0x0e,0x80,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1800e39 <unknown>

umlsll  za.d[w8, 4:7], z1.h, z14.h[5]  // 11000001-10001110-10000100-00111001
// CHECK-INST: umlsll  za.d[w8, 4:7], z1.h, z14.h[5]
// CHECK-ENCODING: [0x39,0x84,0x8e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c18e8439 <unknown>

umlsll  za.d[w10, 0:3], z19.h, z4.h[1]  // 11000001-10000100-01000110-01111000
// CHECK-INST: umlsll  za.d[w10, 0:3], z19.h, z4.h[1]
// CHECK-ENCODING: [0x78,0x46,0x84,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1844678 <unknown>

umlsll  za.d[w8, 0:3], z12.h, z2.h[2]  // 11000001-10000010-00001001-10011000
// CHECK-INST: umlsll  za.d[w8, 0:3], z12.h, z2.h[2]
// CHECK-ENCODING: [0x98,0x09,0x82,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1820998 <unknown>

umlsll  za.d[w10, 4:7], z1.h, z10.h[6]  // 11000001-10001010-11001000-00111001
// CHECK-INST: umlsll  za.d[w10, 4:7], z1.h, z10.h[6]
// CHECK-ENCODING: [0x39,0xc8,0x8a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c18ac839 <unknown>

umlsll  za.d[w8, 4:7], z22.h, z14.h[2]  // 11000001-10001110-00001010-11011001
// CHECK-INST: umlsll  za.d[w8, 4:7], z22.h, z14.h[2]
// CHECK-ENCODING: [0xd9,0x0a,0x8e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c18e0ad9 <unknown>

umlsll  za.d[w11, 8:11], z9.h, z1.h[5]  // 11000001-10000001-11100101-00111010
// CHECK-INST: umlsll  za.d[w11, 8:11], z9.h, z1.h[5]
// CHECK-ENCODING: [0x3a,0xe5,0x81,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c181e53a <unknown>

umlsll  za.d[w9, 12:15], z12.h, z11.h[6]  // 11000001-10001011-10101001-10011011
// CHECK-INST: umlsll  za.d[w9, 12:15], z12.h, z11.h[6]
// CHECK-ENCODING: [0x9b,0xa9,0x8b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c18ba99b <unknown>


umlsll  za.s[w8, 0:3, vgx2], {z0.b, z1.b}, z0.b  // 11000001-00100000-00000000-00011000
// CHECK-INST: umlsll  za.s[w8, 0:3, vgx2], { z0.b, z1.b }, z0.b
// CHECK-ENCODING: [0x18,0x00,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1200018 <unknown>

umlsll  za.s[w8, 0:3], {z0.b - z1.b}, z0.b  // 11000001-00100000-00000000-00011000
// CHECK-INST: umlsll  za.s[w8, 0:3, vgx2], { z0.b, z1.b }, z0.b
// CHECK-ENCODING: [0x18,0x00,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1200018 <unknown>

umlsll  za.s[w10, 4:7, vgx2], {z10.b, z11.b}, z5.b  // 11000001-00100101-01000001-01011001
// CHECK-INST: umlsll  za.s[w10, 4:7, vgx2], { z10.b, z11.b }, z5.b
// CHECK-ENCODING: [0x59,0x41,0x25,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1254159 <unknown>

umlsll  za.s[w10, 4:7], {z10.b - z11.b}, z5.b  // 11000001-00100101-01000001-01011001
// CHECK-INST: umlsll  za.s[w10, 4:7, vgx2], { z10.b, z11.b }, z5.b
// CHECK-ENCODING: [0x59,0x41,0x25,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1254159 <unknown>

umlsll  za.s[w11, 4:7, vgx2], {z13.b, z14.b}, z8.b  // 11000001-00101000-01100001-10111001
// CHECK-INST: umlsll  za.s[w11, 4:7, vgx2], { z13.b, z14.b }, z8.b
// CHECK-ENCODING: [0xb9,0x61,0x28,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12861b9 <unknown>

umlsll  za.s[w11, 4:7], {z13.b - z14.b}, z8.b  // 11000001-00101000-01100001-10111001
// CHECK-INST: umlsll  za.s[w11, 4:7, vgx2], { z13.b, z14.b }, z8.b
// CHECK-ENCODING: [0xb9,0x61,0x28,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12861b9 <unknown>

umlsll  za.s[w11, 4:7, vgx2], {z31.b, z0.b}, z15.b  // 11000001-00101111-01100011-11111001
// CHECK-INST: umlsll  za.s[w11, 4:7, vgx2], { z31.b, z0.b }, z15.b
// CHECK-ENCODING: [0xf9,0x63,0x2f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12f63f9 <unknown>

umlsll  za.s[w11, 4:7], {z31.b - z0.b}, z15.b  // 11000001-00101111-01100011-11111001
// CHECK-INST: umlsll  za.s[w11, 4:7, vgx2], { z31.b, z0.b }, z15.b
// CHECK-ENCODING: [0xf9,0x63,0x2f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12f63f9 <unknown>

umlsll  za.s[w8, 4:7, vgx2], {z17.b, z18.b}, z0.b  // 11000001-00100000-00000010-00111001
// CHECK-INST: umlsll  za.s[w8, 4:7, vgx2], { z17.b, z18.b }, z0.b
// CHECK-ENCODING: [0x39,0x02,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1200239 <unknown>

umlsll  za.s[w8, 4:7], {z17.b - z18.b}, z0.b  // 11000001-00100000-00000010-00111001
// CHECK-INST: umlsll  za.s[w8, 4:7, vgx2], { z17.b, z18.b }, z0.b
// CHECK-ENCODING: [0x39,0x02,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1200239 <unknown>

umlsll  za.s[w8, 4:7, vgx2], {z1.b, z2.b}, z14.b  // 11000001-00101110-00000000-00111001
// CHECK-INST: umlsll  za.s[w8, 4:7, vgx2], { z1.b, z2.b }, z14.b
// CHECK-ENCODING: [0x39,0x00,0x2e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12e0039 <unknown>

umlsll  za.s[w8, 4:7], {z1.b - z2.b}, z14.b  // 11000001-00101110-00000000-00111001
// CHECK-INST: umlsll  za.s[w8, 4:7, vgx2], { z1.b, z2.b }, z14.b
// CHECK-ENCODING: [0x39,0x00,0x2e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12e0039 <unknown>

umlsll  za.s[w10, 0:3, vgx2], {z19.b, z20.b}, z4.b  // 11000001-00100100-01000010-01111000
// CHECK-INST: umlsll  za.s[w10, 0:3, vgx2], { z19.b, z20.b }, z4.b
// CHECK-ENCODING: [0x78,0x42,0x24,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1244278 <unknown>

umlsll  za.s[w10, 0:3], {z19.b - z20.b}, z4.b  // 11000001-00100100-01000010-01111000
// CHECK-INST: umlsll  za.s[w10, 0:3, vgx2], { z19.b, z20.b }, z4.b
// CHECK-ENCODING: [0x78,0x42,0x24,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1244278 <unknown>

umlsll  za.s[w8, 0:3, vgx2], {z12.b, z13.b}, z2.b  // 11000001-00100010-00000001-10011000
// CHECK-INST: umlsll  za.s[w8, 0:3, vgx2], { z12.b, z13.b }, z2.b
// CHECK-ENCODING: [0x98,0x01,0x22,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1220198 <unknown>

umlsll  za.s[w8, 0:3], {z12.b - z13.b}, z2.b  // 11000001-00100010-00000001-10011000
// CHECK-INST: umlsll  za.s[w8, 0:3, vgx2], { z12.b, z13.b }, z2.b
// CHECK-ENCODING: [0x98,0x01,0x22,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1220198 <unknown>

umlsll  za.s[w10, 4:7, vgx2], {z1.b, z2.b}, z10.b  // 11000001-00101010-01000000-00111001
// CHECK-INST: umlsll  za.s[w10, 4:7, vgx2], { z1.b, z2.b }, z10.b
// CHECK-ENCODING: [0x39,0x40,0x2a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12a4039 <unknown>

umlsll  za.s[w10, 4:7], {z1.b - z2.b}, z10.b  // 11000001-00101010-01000000-00111001
// CHECK-INST: umlsll  za.s[w10, 4:7, vgx2], { z1.b, z2.b }, z10.b
// CHECK-ENCODING: [0x39,0x40,0x2a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12a4039 <unknown>

umlsll  za.s[w8, 4:7, vgx2], {z22.b, z23.b}, z14.b  // 11000001-00101110-00000010-11011001
// CHECK-INST: umlsll  za.s[w8, 4:7, vgx2], { z22.b, z23.b }, z14.b
// CHECK-ENCODING: [0xd9,0x02,0x2e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12e02d9 <unknown>

umlsll  za.s[w8, 4:7], {z22.b - z23.b}, z14.b  // 11000001-00101110-00000010-11011001
// CHECK-INST: umlsll  za.s[w8, 4:7, vgx2], { z22.b, z23.b }, z14.b
// CHECK-ENCODING: [0xd9,0x02,0x2e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12e02d9 <unknown>

umlsll  za.s[w11, 0:3, vgx2], {z9.b, z10.b}, z1.b  // 11000001-00100001-01100001-00111000
// CHECK-INST: umlsll  za.s[w11, 0:3, vgx2], { z9.b, z10.b }, z1.b
// CHECK-ENCODING: [0x38,0x61,0x21,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1216138 <unknown>

umlsll  za.s[w11, 0:3], {z9.b - z10.b}, z1.b  // 11000001-00100001-01100001-00111000
// CHECK-INST: umlsll  za.s[w11, 0:3, vgx2], { z9.b, z10.b }, z1.b
// CHECK-ENCODING: [0x38,0x61,0x21,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1216138 <unknown>

umlsll  za.s[w9, 4:7, vgx2], {z12.b, z13.b}, z11.b  // 11000001-00101011-00100001-10011001
// CHECK-INST: umlsll  za.s[w9, 4:7, vgx2], { z12.b, z13.b }, z11.b
// CHECK-ENCODING: [0x99,0x21,0x2b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12b2199 <unknown>

umlsll  za.s[w9, 4:7], {z12.b - z13.b}, z11.b  // 11000001-00101011-00100001-10011001
// CHECK-INST: umlsll  za.s[w9, 4:7, vgx2], { z12.b, z13.b }, z11.b
// CHECK-ENCODING: [0x99,0x21,0x2b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12b2199 <unknown>


umlsll  za.s[w8, 0:3, vgx2], {z0.b, z1.b}, z0.b[0]  // 11000001-00010000-00000000-00011000
// CHECK-INST: umlsll  za.s[w8, 0:3, vgx2], { z0.b, z1.b }, z0.b[0]
// CHECK-ENCODING: [0x18,0x00,0x10,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1100018 <unknown>

umlsll  za.s[w8, 0:3], {z0.b - z1.b}, z0.b[0]  // 11000001-00010000-00000000-00011000
// CHECK-INST: umlsll  za.s[w8, 0:3, vgx2], { z0.b, z1.b }, z0.b[0]
// CHECK-ENCODING: [0x18,0x00,0x10,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1100018 <unknown>

umlsll  za.s[w10, 4:7, vgx2], {z10.b, z11.b}, z5.b[6]  // 11000001-00010101-01000101-01011101
// CHECK-INST: umlsll  za.s[w10, 4:7, vgx2], { z10.b, z11.b }, z5.b[6]
// CHECK-ENCODING: [0x5d,0x45,0x15,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c115455d <unknown>

umlsll  za.s[w10, 4:7], {z10.b - z11.b}, z5.b[6]  // 11000001-00010101-01000101-01011101
// CHECK-INST: umlsll  za.s[w10, 4:7, vgx2], { z10.b, z11.b }, z5.b[6]
// CHECK-ENCODING: [0x5d,0x45,0x15,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c115455d <unknown>

umlsll  za.s[w11, 4:7, vgx2], {z12.b, z13.b}, z8.b[15]  // 11000001-00011000-01101101-10011111
// CHECK-INST: umlsll  za.s[w11, 4:7, vgx2], { z12.b, z13.b }, z8.b[15]
// CHECK-ENCODING: [0x9f,0x6d,0x18,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1186d9f <unknown>

umlsll  za.s[w11, 4:7], {z12.b - z13.b}, z8.b[15]  // 11000001-00011000-01101101-10011111
// CHECK-INST: umlsll  za.s[w11, 4:7, vgx2], { z12.b, z13.b }, z8.b[15]
// CHECK-ENCODING: [0x9f,0x6d,0x18,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1186d9f <unknown>

umlsll  za.s[w11, 4:7, vgx2], {z30.b, z31.b}, z15.b[15]  // 11000001-00011111-01101111-11011111
// CHECK-INST: umlsll  za.s[w11, 4:7, vgx2], { z30.b, z31.b }, z15.b[15]
// CHECK-ENCODING: [0xdf,0x6f,0x1f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11f6fdf <unknown>

umlsll  za.s[w11, 4:7], {z30.b - z31.b}, z15.b[15]  // 11000001-00011111-01101111-11011111
// CHECK-INST: umlsll  za.s[w11, 4:7, vgx2], { z30.b, z31.b }, z15.b[15]
// CHECK-ENCODING: [0xdf,0x6f,0x1f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11f6fdf <unknown>

umlsll  za.s[w8, 4:7, vgx2], {z16.b, z17.b}, z0.b[14]  // 11000001-00010000-00001110-00011101
// CHECK-INST: umlsll  za.s[w8, 4:7, vgx2], { z16.b, z17.b }, z0.b[14]
// CHECK-ENCODING: [0x1d,0x0e,0x10,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1100e1d <unknown>

umlsll  za.s[w8, 4:7], {z16.b - z17.b}, z0.b[14]  // 11000001-00010000-00001110-00011101
// CHECK-INST: umlsll  za.s[w8, 4:7, vgx2], { z16.b, z17.b }, z0.b[14]
// CHECK-ENCODING: [0x1d,0x0e,0x10,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1100e1d <unknown>

umlsll  za.s[w8, 4:7, vgx2], {z0.b, z1.b}, z14.b[4]  // 11000001-00011110-00000100-00011001
// CHECK-INST: umlsll  za.s[w8, 4:7, vgx2], { z0.b, z1.b }, z14.b[4]
// CHECK-ENCODING: [0x19,0x04,0x1e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11e0419 <unknown>

umlsll  za.s[w8, 4:7], {z0.b - z1.b}, z14.b[4]  // 11000001-00011110-00000100-00011001
// CHECK-INST: umlsll  za.s[w8, 4:7, vgx2], { z0.b, z1.b }, z14.b[4]
// CHECK-ENCODING: [0x19,0x04,0x1e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11e0419 <unknown>

umlsll  za.s[w10, 0:3, vgx2], {z18.b, z19.b}, z4.b[4]  // 11000001-00010100-01000110-01011000
// CHECK-INST: umlsll  za.s[w10, 0:3, vgx2], { z18.b, z19.b }, z4.b[4]
// CHECK-ENCODING: [0x58,0x46,0x14,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1144658 <unknown>

umlsll  za.s[w10, 0:3], {z18.b - z19.b}, z4.b[4]  // 11000001-00010100-01000110-01011000
// CHECK-INST: umlsll  za.s[w10, 0:3, vgx2], { z18.b, z19.b }, z4.b[4]
// CHECK-ENCODING: [0x58,0x46,0x14,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1144658 <unknown>

umlsll  za.s[w8, 0:3, vgx2], {z12.b, z13.b}, z2.b[8]  // 11000001-00010010-00001001-10011000
// CHECK-INST: umlsll  za.s[w8, 0:3, vgx2], { z12.b, z13.b }, z2.b[8]
// CHECK-ENCODING: [0x98,0x09,0x12,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1120998 <unknown>

umlsll  za.s[w8, 0:3], {z12.b - z13.b}, z2.b[8]  // 11000001-00010010-00001001-10011000
// CHECK-INST: umlsll  za.s[w8, 0:3, vgx2], { z12.b, z13.b }, z2.b[8]
// CHECK-ENCODING: [0x98,0x09,0x12,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1120998 <unknown>

umlsll  za.s[w10, 4:7, vgx2], {z0.b, z1.b}, z10.b[8]  // 11000001-00011010-01001000-00011001
// CHECK-INST: umlsll  za.s[w10, 4:7, vgx2], { z0.b, z1.b }, z10.b[8]
// CHECK-ENCODING: [0x19,0x48,0x1a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11a4819 <unknown>

umlsll  za.s[w10, 4:7], {z0.b - z1.b}, z10.b[8]  // 11000001-00011010-01001000-00011001
// CHECK-INST: umlsll  za.s[w10, 4:7, vgx2], { z0.b, z1.b }, z10.b[8]
// CHECK-ENCODING: [0x19,0x48,0x1a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11a4819 <unknown>

umlsll  za.s[w8, 4:7, vgx2], {z22.b, z23.b}, z14.b[10]  // 11000001-00011110-00001010-11011101
// CHECK-INST: umlsll  za.s[w8, 4:7, vgx2], { z22.b, z23.b }, z14.b[10]
// CHECK-ENCODING: [0xdd,0x0a,0x1e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11e0add <unknown>

umlsll  za.s[w8, 4:7], {z22.b - z23.b}, z14.b[10]  // 11000001-00011110-00001010-11011101
// CHECK-INST: umlsll  za.s[w8, 4:7, vgx2], { z22.b, z23.b }, z14.b[10]
// CHECK-ENCODING: [0xdd,0x0a,0x1e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11e0add <unknown>

umlsll  za.s[w11, 0:3, vgx2], {z8.b, z9.b}, z1.b[5]  // 11000001-00010001-01100101-00011010
// CHECK-INST: umlsll  za.s[w11, 0:3, vgx2], { z8.b, z9.b }, z1.b[5]
// CHECK-ENCODING: [0x1a,0x65,0x11,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c111651a <unknown>

umlsll  za.s[w11, 0:3], {z8.b - z9.b}, z1.b[5]  // 11000001-00010001-01100101-00011010
// CHECK-INST: umlsll  za.s[w11, 0:3, vgx2], { z8.b, z9.b }, z1.b[5]
// CHECK-ENCODING: [0x1a,0x65,0x11,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c111651a <unknown>

umlsll  za.s[w9, 4:7, vgx2], {z12.b, z13.b}, z11.b[11]  // 11000001-00011011-00101001-10011111
// CHECK-INST: umlsll  za.s[w9, 4:7, vgx2], { z12.b, z13.b }, z11.b[11]
// CHECK-ENCODING: [0x9f,0x29,0x1b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11b299f <unknown>

umlsll  za.s[w9, 4:7], {z12.b - z13.b}, z11.b[11]  // 11000001-00011011-00101001-10011111
// CHECK-INST: umlsll  za.s[w9, 4:7, vgx2], { z12.b, z13.b }, z11.b[11]
// CHECK-ENCODING: [0x9f,0x29,0x1b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11b299f <unknown>


umlsll  za.s[w8, 0:3, vgx2], {z0.b, z1.b}, {z0.b, z1.b}  // 11000001-10100000-00000000-00011000
// CHECK-INST: umlsll  za.s[w8, 0:3, vgx2], { z0.b, z1.b }, { z0.b, z1.b }
// CHECK-ENCODING: [0x18,0x00,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a00018 <unknown>

umlsll  za.s[w8, 0:3], {z0.b - z1.b}, {z0.b - z1.b}  // 11000001-10100000-00000000-00011000
// CHECK-INST: umlsll  za.s[w8, 0:3, vgx2], { z0.b, z1.b }, { z0.b, z1.b }
// CHECK-ENCODING: [0x18,0x00,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a00018 <unknown>

umlsll  za.s[w10, 4:7, vgx2], {z10.b, z11.b}, {z20.b, z21.b}  // 11000001-10110100-01000001-01011001
// CHECK-INST: umlsll  za.s[w10, 4:7, vgx2], { z10.b, z11.b }, { z20.b, z21.b }
// CHECK-ENCODING: [0x59,0x41,0xb4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b44159 <unknown>

umlsll  za.s[w10, 4:7], {z10.b - z11.b}, {z20.b - z21.b}  // 11000001-10110100-01000001-01011001
// CHECK-INST: umlsll  za.s[w10, 4:7, vgx2], { z10.b, z11.b }, { z20.b, z21.b }
// CHECK-ENCODING: [0x59,0x41,0xb4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b44159 <unknown>

umlsll  za.s[w11, 4:7, vgx2], {z12.b, z13.b}, {z8.b, z9.b}  // 11000001-10101000-01100001-10011001
// CHECK-INST: umlsll  za.s[w11, 4:7, vgx2], { z12.b, z13.b }, { z8.b, z9.b }
// CHECK-ENCODING: [0x99,0x61,0xa8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a86199 <unknown>

umlsll  za.s[w11, 4:7], {z12.b - z13.b}, {z8.b - z9.b}  // 11000001-10101000-01100001-10011001
// CHECK-INST: umlsll  za.s[w11, 4:7, vgx2], { z12.b, z13.b }, { z8.b, z9.b }
// CHECK-ENCODING: [0x99,0x61,0xa8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a86199 <unknown>

umlsll  za.s[w11, 4:7, vgx2], {z30.b, z31.b}, {z30.b, z31.b}  // 11000001-10111110-01100011-11011001
// CHECK-INST: umlsll  za.s[w11, 4:7, vgx2], { z30.b, z31.b }, { z30.b, z31.b }
// CHECK-ENCODING: [0xd9,0x63,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1be63d9 <unknown>

umlsll  za.s[w11, 4:7], {z30.b - z31.b}, {z30.b - z31.b}  // 11000001-10111110-01100011-11011001
// CHECK-INST: umlsll  za.s[w11, 4:7, vgx2], { z30.b, z31.b }, { z30.b, z31.b }
// CHECK-ENCODING: [0xd9,0x63,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1be63d9 <unknown>

umlsll  za.s[w8, 4:7, vgx2], {z16.b, z17.b}, {z16.b, z17.b}  // 11000001-10110000-00000010-00011001
// CHECK-INST: umlsll  za.s[w8, 4:7, vgx2], { z16.b, z17.b }, { z16.b, z17.b }
// CHECK-ENCODING: [0x19,0x02,0xb0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b00219 <unknown>

umlsll  za.s[w8, 4:7], {z16.b - z17.b}, {z16.b - z17.b}  // 11000001-10110000-00000010-00011001
// CHECK-INST: umlsll  za.s[w8, 4:7, vgx2], { z16.b, z17.b }, { z16.b, z17.b }
// CHECK-ENCODING: [0x19,0x02,0xb0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b00219 <unknown>

umlsll  za.s[w8, 4:7, vgx2], {z0.b, z1.b}, {z30.b, z31.b}  // 11000001-10111110-00000000-00011001
// CHECK-INST: umlsll  za.s[w8, 4:7, vgx2], { z0.b, z1.b }, { z30.b, z31.b }
// CHECK-ENCODING: [0x19,0x00,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1be0019 <unknown>

umlsll  za.s[w8, 4:7], {z0.b - z1.b}, {z30.b - z31.b}  // 11000001-10111110-00000000-00011001
// CHECK-INST: umlsll  za.s[w8, 4:7, vgx2], { z0.b, z1.b }, { z30.b, z31.b }
// CHECK-ENCODING: [0x19,0x00,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1be0019 <unknown>

umlsll  za.s[w10, 0:3, vgx2], {z18.b, z19.b}, {z20.b, z21.b}  // 11000001-10110100-01000010-01011000
// CHECK-INST: umlsll  za.s[w10, 0:3, vgx2], { z18.b, z19.b }, { z20.b, z21.b }
// CHECK-ENCODING: [0x58,0x42,0xb4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b44258 <unknown>

umlsll  za.s[w10, 0:3], {z18.b - z19.b}, {z20.b - z21.b}  // 11000001-10110100-01000010-01011000
// CHECK-INST: umlsll  za.s[w10, 0:3, vgx2], { z18.b, z19.b }, { z20.b, z21.b }
// CHECK-ENCODING: [0x58,0x42,0xb4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b44258 <unknown>

umlsll  za.s[w8, 0:3, vgx2], {z12.b, z13.b}, {z2.b, z3.b}  // 11000001-10100010-00000001-10011000
// CHECK-INST: umlsll  za.s[w8, 0:3, vgx2], { z12.b, z13.b }, { z2.b, z3.b }
// CHECK-ENCODING: [0x98,0x01,0xa2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a20198 <unknown>

umlsll  za.s[w8, 0:3], {z12.b - z13.b}, {z2.b - z3.b}  // 11000001-10100010-00000001-10011000
// CHECK-INST: umlsll  za.s[w8, 0:3, vgx2], { z12.b, z13.b }, { z2.b, z3.b }
// CHECK-ENCODING: [0x98,0x01,0xa2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a20198 <unknown>

umlsll  za.s[w10, 4:7, vgx2], {z0.b, z1.b}, {z26.b, z27.b}  // 11000001-10111010-01000000-00011001
// CHECK-INST: umlsll  za.s[w10, 4:7, vgx2], { z0.b, z1.b }, { z26.b, z27.b }
// CHECK-ENCODING: [0x19,0x40,0xba,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1ba4019 <unknown>

umlsll  za.s[w10, 4:7], {z0.b - z1.b}, {z26.b - z27.b}  // 11000001-10111010-01000000-00011001
// CHECK-INST: umlsll  za.s[w10, 4:7, vgx2], { z0.b, z1.b }, { z26.b, z27.b }
// CHECK-ENCODING: [0x19,0x40,0xba,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1ba4019 <unknown>

umlsll  za.s[w8, 4:7, vgx2], {z22.b, z23.b}, {z30.b, z31.b}  // 11000001-10111110-00000010-11011001
// CHECK-INST: umlsll  za.s[w8, 4:7, vgx2], { z22.b, z23.b }, { z30.b, z31.b }
// CHECK-ENCODING: [0xd9,0x02,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1be02d9 <unknown>

umlsll  za.s[w8, 4:7], {z22.b - z23.b}, {z30.b - z31.b}  // 11000001-10111110-00000010-11011001
// CHECK-INST: umlsll  za.s[w8, 4:7, vgx2], { z22.b, z23.b }, { z30.b, z31.b }
// CHECK-ENCODING: [0xd9,0x02,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1be02d9 <unknown>

umlsll  za.s[w11, 0:3, vgx2], {z8.b, z9.b}, {z0.b, z1.b}  // 11000001-10100000-01100001-00011000
// CHECK-INST: umlsll  za.s[w11, 0:3, vgx2], { z8.b, z9.b }, { z0.b, z1.b }
// CHECK-ENCODING: [0x18,0x61,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a06118 <unknown>

umlsll  za.s[w11, 0:3], {z8.b - z9.b}, {z0.b - z1.b}  // 11000001-10100000-01100001-00011000
// CHECK-INST: umlsll  za.s[w11, 0:3, vgx2], { z8.b, z9.b }, { z0.b, z1.b }
// CHECK-ENCODING: [0x18,0x61,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a06118 <unknown>

umlsll  za.s[w9, 4:7, vgx2], {z12.b, z13.b}, {z10.b, z11.b}  // 11000001-10101010-00100001-10011001
// CHECK-INST: umlsll  za.s[w9, 4:7, vgx2], { z12.b, z13.b }, { z10.b, z11.b }
// CHECK-ENCODING: [0x99,0x21,0xaa,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1aa2199 <unknown>

umlsll  za.s[w9, 4:7], {z12.b - z13.b}, {z10.b - z11.b}  // 11000001-10101010-00100001-10011001
// CHECK-INST: umlsll  za.s[w9, 4:7, vgx2], { z12.b, z13.b }, { z10.b, z11.b }
// CHECK-ENCODING: [0x99,0x21,0xaa,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1aa2199 <unknown>


umlsll  za.d[w8, 0:3, vgx2], {z0.h, z1.h}, z0.h  // 11000001-01100000-00000000-00011000
// CHECK-INST: umlsll  za.d[w8, 0:3, vgx2], { z0.h, z1.h }, z0.h
// CHECK-ENCODING: [0x18,0x00,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1600018 <unknown>

umlsll  za.d[w8, 0:3], {z0.h - z1.h}, z0.h  // 11000001-01100000-00000000-00011000
// CHECK-INST: umlsll  za.d[w8, 0:3, vgx2], { z0.h, z1.h }, z0.h
// CHECK-ENCODING: [0x18,0x00,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1600018 <unknown>

umlsll  za.d[w10, 4:7, vgx2], {z10.h, z11.h}, z5.h  // 11000001-01100101-01000001-01011001
// CHECK-INST: umlsll  za.d[w10, 4:7, vgx2], { z10.h, z11.h }, z5.h
// CHECK-ENCODING: [0x59,0x41,0x65,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1654159 <unknown>

umlsll  za.d[w10, 4:7], {z10.h - z11.h}, z5.h  // 11000001-01100101-01000001-01011001
// CHECK-INST: umlsll  za.d[w10, 4:7, vgx2], { z10.h, z11.h }, z5.h
// CHECK-ENCODING: [0x59,0x41,0x65,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1654159 <unknown>

umlsll  za.d[w11, 4:7, vgx2], {z13.h, z14.h}, z8.h  // 11000001-01101000-01100001-10111001
// CHECK-INST: umlsll  za.d[w11, 4:7, vgx2], { z13.h, z14.h }, z8.h
// CHECK-ENCODING: [0xb9,0x61,0x68,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16861b9 <unknown>

umlsll  za.d[w11, 4:7], {z13.h - z14.h}, z8.h  // 11000001-01101000-01100001-10111001
// CHECK-INST: umlsll  za.d[w11, 4:7, vgx2], { z13.h, z14.h }, z8.h
// CHECK-ENCODING: [0xb9,0x61,0x68,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16861b9 <unknown>

umlsll  za.d[w11, 4:7, vgx2], {z31.h, z0.h}, z15.h  // 11000001-01101111-01100011-11111001
// CHECK-INST: umlsll  za.d[w11, 4:7, vgx2], { z31.h, z0.h }, z15.h
// CHECK-ENCODING: [0xf9,0x63,0x6f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16f63f9 <unknown>

umlsll  za.d[w11, 4:7], {z31.h - z0.h}, z15.h  // 11000001-01101111-01100011-11111001
// CHECK-INST: umlsll  za.d[w11, 4:7, vgx2], { z31.h, z0.h }, z15.h
// CHECK-ENCODING: [0xf9,0x63,0x6f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16f63f9 <unknown>

umlsll  za.d[w8, 4:7, vgx2], {z17.h, z18.h}, z0.h  // 11000001-01100000-00000010-00111001
// CHECK-INST: umlsll  za.d[w8, 4:7, vgx2], { z17.h, z18.h }, z0.h
// CHECK-ENCODING: [0x39,0x02,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1600239 <unknown>

umlsll  za.d[w8, 4:7], {z17.h - z18.h}, z0.h  // 11000001-01100000-00000010-00111001
// CHECK-INST: umlsll  za.d[w8, 4:7, vgx2], { z17.h, z18.h }, z0.h
// CHECK-ENCODING: [0x39,0x02,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1600239 <unknown>

umlsll  za.d[w8, 4:7, vgx2], {z1.h, z2.h}, z14.h  // 11000001-01101110-00000000-00111001
// CHECK-INST: umlsll  za.d[w8, 4:7, vgx2], { z1.h, z2.h }, z14.h
// CHECK-ENCODING: [0x39,0x00,0x6e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16e0039 <unknown>

umlsll  za.d[w8, 4:7], {z1.h - z2.h}, z14.h  // 11000001-01101110-00000000-00111001
// CHECK-INST: umlsll  za.d[w8, 4:7, vgx2], { z1.h, z2.h }, z14.h
// CHECK-ENCODING: [0x39,0x00,0x6e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16e0039 <unknown>

umlsll  za.d[w10, 0:3, vgx2], {z19.h, z20.h}, z4.h  // 11000001-01100100-01000010-01111000
// CHECK-INST: umlsll  za.d[w10, 0:3, vgx2], { z19.h, z20.h }, z4.h
// CHECK-ENCODING: [0x78,0x42,0x64,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1644278 <unknown>

umlsll  za.d[w10, 0:3], {z19.h - z20.h}, z4.h  // 11000001-01100100-01000010-01111000
// CHECK-INST: umlsll  za.d[w10, 0:3, vgx2], { z19.h, z20.h }, z4.h
// CHECK-ENCODING: [0x78,0x42,0x64,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1644278 <unknown>

umlsll  za.d[w8, 0:3, vgx2], {z12.h, z13.h}, z2.h  // 11000001-01100010-00000001-10011000
// CHECK-INST: umlsll  za.d[w8, 0:3, vgx2], { z12.h, z13.h }, z2.h
// CHECK-ENCODING: [0x98,0x01,0x62,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1620198 <unknown>

umlsll  za.d[w8, 0:3], {z12.h - z13.h}, z2.h  // 11000001-01100010-00000001-10011000
// CHECK-INST: umlsll  za.d[w8, 0:3, vgx2], { z12.h, z13.h }, z2.h
// CHECK-ENCODING: [0x98,0x01,0x62,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1620198 <unknown>

umlsll  za.d[w10, 4:7, vgx2], {z1.h, z2.h}, z10.h  // 11000001-01101010-01000000-00111001
// CHECK-INST: umlsll  za.d[w10, 4:7, vgx2], { z1.h, z2.h }, z10.h
// CHECK-ENCODING: [0x39,0x40,0x6a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16a4039 <unknown>

umlsll  za.d[w10, 4:7], {z1.h - z2.h}, z10.h  // 11000001-01101010-01000000-00111001
// CHECK-INST: umlsll  za.d[w10, 4:7, vgx2], { z1.h, z2.h }, z10.h
// CHECK-ENCODING: [0x39,0x40,0x6a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16a4039 <unknown>

umlsll  za.d[w8, 4:7, vgx2], {z22.h, z23.h}, z14.h  // 11000001-01101110-00000010-11011001
// CHECK-INST: umlsll  za.d[w8, 4:7, vgx2], { z22.h, z23.h }, z14.h
// CHECK-ENCODING: [0xd9,0x02,0x6e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16e02d9 <unknown>

umlsll  za.d[w8, 4:7], {z22.h - z23.h}, z14.h  // 11000001-01101110-00000010-11011001
// CHECK-INST: umlsll  za.d[w8, 4:7, vgx2], { z22.h, z23.h }, z14.h
// CHECK-ENCODING: [0xd9,0x02,0x6e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16e02d9 <unknown>

umlsll  za.d[w11, 0:3, vgx2], {z9.h, z10.h}, z1.h  // 11000001-01100001-01100001-00111000
// CHECK-INST: umlsll  za.d[w11, 0:3, vgx2], { z9.h, z10.h }, z1.h
// CHECK-ENCODING: [0x38,0x61,0x61,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1616138 <unknown>

umlsll  za.d[w11, 0:3], {z9.h - z10.h}, z1.h  // 11000001-01100001-01100001-00111000
// CHECK-INST: umlsll  za.d[w11, 0:3, vgx2], { z9.h, z10.h }, z1.h
// CHECK-ENCODING: [0x38,0x61,0x61,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1616138 <unknown>

umlsll  za.d[w9, 4:7, vgx2], {z12.h, z13.h}, z11.h  // 11000001-01101011-00100001-10011001
// CHECK-INST: umlsll  za.d[w9, 4:7, vgx2], { z12.h, z13.h }, z11.h
// CHECK-ENCODING: [0x99,0x21,0x6b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16b2199 <unknown>

umlsll  za.d[w9, 4:7], {z12.h - z13.h}, z11.h  // 11000001-01101011-00100001-10011001
// CHECK-INST: umlsll  za.d[w9, 4:7, vgx2], { z12.h, z13.h }, z11.h
// CHECK-ENCODING: [0x99,0x21,0x6b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16b2199 <unknown>


umlsll  za.d[w8, 0:3, vgx2], {z0.h, z1.h}, z0.h[0]  // 11000001-10010000-00000000-00011000
// CHECK-INST: umlsll  za.d[w8, 0:3, vgx2], { z0.h, z1.h }, z0.h[0]
// CHECK-ENCODING: [0x18,0x00,0x90,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1900018 <unknown>

umlsll  za.d[w8, 0:3], {z0.h - z1.h}, z0.h[0]  // 11000001-10010000-00000000-00011000
// CHECK-INST: umlsll  za.d[w8, 0:3, vgx2], { z0.h, z1.h }, z0.h[0]
// CHECK-ENCODING: [0x18,0x00,0x90,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1900018 <unknown>

umlsll  za.d[w10, 4:7, vgx2], {z10.h, z11.h}, z5.h[6]  // 11000001-10010101-01000101-01011101
// CHECK-INST: umlsll  za.d[w10, 4:7, vgx2], { z10.h, z11.h }, z5.h[6]
// CHECK-ENCODING: [0x5d,0x45,0x95,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c195455d <unknown>

umlsll  za.d[w10, 4:7], {z10.h - z11.h}, z5.h[6]  // 11000001-10010101-01000101-01011101
// CHECK-INST: umlsll  za.d[w10, 4:7, vgx2], { z10.h, z11.h }, z5.h[6]
// CHECK-ENCODING: [0x5d,0x45,0x95,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c195455d <unknown>

umlsll  za.d[w11, 4:7, vgx2], {z12.h, z13.h}, z8.h[7]  // 11000001-10011000-01100101-10011111
// CHECK-INST: umlsll  za.d[w11, 4:7, vgx2], { z12.h, z13.h }, z8.h[7]
// CHECK-ENCODING: [0x9f,0x65,0x98,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c198659f <unknown>

umlsll  za.d[w11, 4:7], {z12.h - z13.h}, z8.h[7]  // 11000001-10011000-01100101-10011111
// CHECK-INST: umlsll  za.d[w11, 4:7, vgx2], { z12.h, z13.h }, z8.h[7]
// CHECK-ENCODING: [0x9f,0x65,0x98,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c198659f <unknown>

umlsll  za.d[w11, 4:7, vgx2], {z30.h, z31.h}, z15.h[7]  // 11000001-10011111-01100111-11011111
// CHECK-INST: umlsll  za.d[w11, 4:7, vgx2], { z30.h, z31.h }, z15.h[7]
// CHECK-ENCODING: [0xdf,0x67,0x9f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c19f67df <unknown>

umlsll  za.d[w11, 4:7], {z30.h - z31.h}, z15.h[7]  // 11000001-10011111-01100111-11011111
// CHECK-INST: umlsll  za.d[w11, 4:7, vgx2], { z30.h, z31.h }, z15.h[7]
// CHECK-ENCODING: [0xdf,0x67,0x9f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c19f67df <unknown>

umlsll  za.d[w8, 4:7, vgx2], {z16.h, z17.h}, z0.h[6]  // 11000001-10010000-00000110-00011101
// CHECK-INST: umlsll  za.d[w8, 4:7, vgx2], { z16.h, z17.h }, z0.h[6]
// CHECK-ENCODING: [0x1d,0x06,0x90,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c190061d <unknown>

umlsll  za.d[w8, 4:7], {z16.h - z17.h}, z0.h[6]  // 11000001-10010000-00000110-00011101
// CHECK-INST: umlsll  za.d[w8, 4:7, vgx2], { z16.h, z17.h }, z0.h[6]
// CHECK-ENCODING: [0x1d,0x06,0x90,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c190061d <unknown>

umlsll  za.d[w8, 4:7, vgx2], {z0.h, z1.h}, z14.h[4]  // 11000001-10011110-00000100-00011001
// CHECK-INST: umlsll  za.d[w8, 4:7, vgx2], { z0.h, z1.h }, z14.h[4]
// CHECK-ENCODING: [0x19,0x04,0x9e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c19e0419 <unknown>

umlsll  za.d[w8, 4:7], {z0.h - z1.h}, z14.h[4]  // 11000001-10011110-00000100-00011001
// CHECK-INST: umlsll  za.d[w8, 4:7, vgx2], { z0.h, z1.h }, z14.h[4]
// CHECK-ENCODING: [0x19,0x04,0x9e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c19e0419 <unknown>

umlsll  za.d[w10, 0:3, vgx2], {z18.h, z19.h}, z4.h[4]  // 11000001-10010100-01000110-01011000
// CHECK-INST: umlsll  za.d[w10, 0:3, vgx2], { z18.h, z19.h }, z4.h[4]
// CHECK-ENCODING: [0x58,0x46,0x94,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1944658 <unknown>

umlsll  za.d[w10, 0:3], {z18.h - z19.h}, z4.h[4]  // 11000001-10010100-01000110-01011000
// CHECK-INST: umlsll  za.d[w10, 0:3, vgx2], { z18.h, z19.h }, z4.h[4]
// CHECK-ENCODING: [0x58,0x46,0x94,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1944658 <unknown>

umlsll  za.d[w8, 0:3, vgx2], {z12.h, z13.h}, z2.h[0]  // 11000001-10010010-00000001-10011000
// CHECK-INST: umlsll  za.d[w8, 0:3, vgx2], { z12.h, z13.h }, z2.h[0]
// CHECK-ENCODING: [0x98,0x01,0x92,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1920198 <unknown>

umlsll  za.d[w8, 0:3], {z12.h - z13.h}, z2.h[0]  // 11000001-10010010-00000001-10011000
// CHECK-INST: umlsll  za.d[w8, 0:3, vgx2], { z12.h, z13.h }, z2.h[0]
// CHECK-ENCODING: [0x98,0x01,0x92,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1920198 <unknown>

umlsll  za.d[w10, 4:7, vgx2], {z0.h, z1.h}, z10.h[0]  // 11000001-10011010-01000000-00011001
// CHECK-INST: umlsll  za.d[w10, 4:7, vgx2], { z0.h, z1.h }, z10.h[0]
// CHECK-ENCODING: [0x19,0x40,0x9a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c19a4019 <unknown>

umlsll  za.d[w10, 4:7], {z0.h - z1.h}, z10.h[0]  // 11000001-10011010-01000000-00011001
// CHECK-INST: umlsll  za.d[w10, 4:7, vgx2], { z0.h, z1.h }, z10.h[0]
// CHECK-ENCODING: [0x19,0x40,0x9a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c19a4019 <unknown>

umlsll  za.d[w8, 4:7, vgx2], {z22.h, z23.h}, z14.h[2]  // 11000001-10011110-00000010-11011101
// CHECK-INST: umlsll  za.d[w8, 4:7, vgx2], { z22.h, z23.h }, z14.h[2]
// CHECK-ENCODING: [0xdd,0x02,0x9e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c19e02dd <unknown>

umlsll  za.d[w8, 4:7], {z22.h - z23.h}, z14.h[2]  // 11000001-10011110-00000010-11011101
// CHECK-INST: umlsll  za.d[w8, 4:7, vgx2], { z22.h, z23.h }, z14.h[2]
// CHECK-ENCODING: [0xdd,0x02,0x9e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c19e02dd <unknown>

umlsll  za.d[w11, 0:3, vgx2], {z8.h, z9.h}, z1.h[5]  // 11000001-10010001-01100101-00011010
// CHECK-INST: umlsll  za.d[w11, 0:3, vgx2], { z8.h, z9.h }, z1.h[5]
// CHECK-ENCODING: [0x1a,0x65,0x91,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c191651a <unknown>

umlsll  za.d[w11, 0:3], {z8.h - z9.h}, z1.h[5]  // 11000001-10010001-01100101-00011010
// CHECK-INST: umlsll  za.d[w11, 0:3, vgx2], { z8.h, z9.h }, z1.h[5]
// CHECK-ENCODING: [0x1a,0x65,0x91,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c191651a <unknown>

umlsll  za.d[w9, 4:7, vgx2], {z12.h, z13.h}, z11.h[3]  // 11000001-10011011-00100001-10011111
// CHECK-INST: umlsll  za.d[w9, 4:7, vgx2], { z12.h, z13.h }, z11.h[3]
// CHECK-ENCODING: [0x9f,0x21,0x9b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c19b219f <unknown>

umlsll  za.d[w9, 4:7], {z12.h - z13.h}, z11.h[3]  // 11000001-10011011-00100001-10011111
// CHECK-INST: umlsll  za.d[w9, 4:7, vgx2], { z12.h, z13.h }, z11.h[3]
// CHECK-ENCODING: [0x9f,0x21,0x9b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c19b219f <unknown>


umlsll  za.d[w8, 0:3, vgx2], {z0.h, z1.h}, {z0.h, z1.h}  // 11000001-11100000-00000000-00011000
// CHECK-INST: umlsll  za.d[w8, 0:3, vgx2], { z0.h, z1.h }, { z0.h, z1.h }
// CHECK-ENCODING: [0x18,0x00,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e00018 <unknown>

umlsll  za.d[w8, 0:3], {z0.h - z1.h}, {z0.h - z1.h}  // 11000001-11100000-00000000-00011000
// CHECK-INST: umlsll  za.d[w8, 0:3, vgx2], { z0.h, z1.h }, { z0.h, z1.h }
// CHECK-ENCODING: [0x18,0x00,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e00018 <unknown>

umlsll  za.d[w10, 4:7, vgx2], {z10.h, z11.h}, {z20.h, z21.h}  // 11000001-11110100-01000001-01011001
// CHECK-INST: umlsll  za.d[w10, 4:7, vgx2], { z10.h, z11.h }, { z20.h, z21.h }
// CHECK-ENCODING: [0x59,0x41,0xf4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f44159 <unknown>

umlsll  za.d[w10, 4:7], {z10.h - z11.h}, {z20.h - z21.h}  // 11000001-11110100-01000001-01011001
// CHECK-INST: umlsll  za.d[w10, 4:7, vgx2], { z10.h, z11.h }, { z20.h, z21.h }
// CHECK-ENCODING: [0x59,0x41,0xf4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f44159 <unknown>

umlsll  za.d[w11, 4:7, vgx2], {z12.h, z13.h}, {z8.h, z9.h}  // 11000001-11101000-01100001-10011001
// CHECK-INST: umlsll  za.d[w11, 4:7, vgx2], { z12.h, z13.h }, { z8.h, z9.h }
// CHECK-ENCODING: [0x99,0x61,0xe8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e86199 <unknown>

umlsll  za.d[w11, 4:7], {z12.h - z13.h}, {z8.h - z9.h}  // 11000001-11101000-01100001-10011001
// CHECK-INST: umlsll  za.d[w11, 4:7, vgx2], { z12.h, z13.h }, { z8.h, z9.h }
// CHECK-ENCODING: [0x99,0x61,0xe8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e86199 <unknown>

umlsll  za.d[w11, 4:7, vgx2], {z30.h, z31.h}, {z30.h, z31.h}  // 11000001-11111110-01100011-11011001
// CHECK-INST: umlsll  za.d[w11, 4:7, vgx2], { z30.h, z31.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xd9,0x63,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fe63d9 <unknown>

umlsll  za.d[w11, 4:7], {z30.h - z31.h}, {z30.h - z31.h}  // 11000001-11111110-01100011-11011001
// CHECK-INST: umlsll  za.d[w11, 4:7, vgx2], { z30.h, z31.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xd9,0x63,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fe63d9 <unknown>

umlsll  za.d[w8, 4:7, vgx2], {z16.h, z17.h}, {z16.h, z17.h}  // 11000001-11110000-00000010-00011001
// CHECK-INST: umlsll  za.d[w8, 4:7, vgx2], { z16.h, z17.h }, { z16.h, z17.h }
// CHECK-ENCODING: [0x19,0x02,0xf0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f00219 <unknown>

umlsll  za.d[w8, 4:7], {z16.h - z17.h}, {z16.h - z17.h}  // 11000001-11110000-00000010-00011001
// CHECK-INST: umlsll  za.d[w8, 4:7, vgx2], { z16.h, z17.h }, { z16.h, z17.h }
// CHECK-ENCODING: [0x19,0x02,0xf0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f00219 <unknown>

umlsll  za.d[w8, 4:7, vgx2], {z0.h, z1.h}, {z30.h, z31.h}  // 11000001-11111110-00000000-00011001
// CHECK-INST: umlsll  za.d[w8, 4:7, vgx2], { z0.h, z1.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0x19,0x00,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fe0019 <unknown>

umlsll  za.d[w8, 4:7], {z0.h - z1.h}, {z30.h - z31.h}  // 11000001-11111110-00000000-00011001
// CHECK-INST: umlsll  za.d[w8, 4:7, vgx2], { z0.h, z1.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0x19,0x00,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fe0019 <unknown>

umlsll  za.d[w10, 0:3, vgx2], {z18.h, z19.h}, {z20.h, z21.h}  // 11000001-11110100-01000010-01011000
// CHECK-INST: umlsll  za.d[w10, 0:3, vgx2], { z18.h, z19.h }, { z20.h, z21.h }
// CHECK-ENCODING: [0x58,0x42,0xf4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f44258 <unknown>

umlsll  za.d[w10, 0:3], {z18.h - z19.h}, {z20.h - z21.h}  // 11000001-11110100-01000010-01011000
// CHECK-INST: umlsll  za.d[w10, 0:3, vgx2], { z18.h, z19.h }, { z20.h, z21.h }
// CHECK-ENCODING: [0x58,0x42,0xf4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f44258 <unknown>

umlsll  za.d[w8, 0:3, vgx2], {z12.h, z13.h}, {z2.h, z3.h}  // 11000001-11100010-00000001-10011000
// CHECK-INST: umlsll  za.d[w8, 0:3, vgx2], { z12.h, z13.h }, { z2.h, z3.h }
// CHECK-ENCODING: [0x98,0x01,0xe2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e20198 <unknown>

umlsll  za.d[w8, 0:3], {z12.h - z13.h}, {z2.h - z3.h}  // 11000001-11100010-00000001-10011000
// CHECK-INST: umlsll  za.d[w8, 0:3, vgx2], { z12.h, z13.h }, { z2.h, z3.h }
// CHECK-ENCODING: [0x98,0x01,0xe2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e20198 <unknown>

umlsll  za.d[w10, 4:7, vgx2], {z0.h, z1.h}, {z26.h, z27.h}  // 11000001-11111010-01000000-00011001
// CHECK-INST: umlsll  za.d[w10, 4:7, vgx2], { z0.h, z1.h }, { z26.h, z27.h }
// CHECK-ENCODING: [0x19,0x40,0xfa,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fa4019 <unknown>

umlsll  za.d[w10, 4:7], {z0.h - z1.h}, {z26.h - z27.h}  // 11000001-11111010-01000000-00011001
// CHECK-INST: umlsll  za.d[w10, 4:7, vgx2], { z0.h, z1.h }, { z26.h, z27.h }
// CHECK-ENCODING: [0x19,0x40,0xfa,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fa4019 <unknown>

umlsll  za.d[w8, 4:7, vgx2], {z22.h, z23.h}, {z30.h, z31.h}  // 11000001-11111110-00000010-11011001
// CHECK-INST: umlsll  za.d[w8, 4:7, vgx2], { z22.h, z23.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xd9,0x02,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fe02d9 <unknown>

umlsll  za.d[w8, 4:7], {z22.h - z23.h}, {z30.h - z31.h}  // 11000001-11111110-00000010-11011001
// CHECK-INST: umlsll  za.d[w8, 4:7, vgx2], { z22.h, z23.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xd9,0x02,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fe02d9 <unknown>

umlsll  za.d[w11, 0:3, vgx2], {z8.h, z9.h}, {z0.h, z1.h}  // 11000001-11100000-01100001-00011000
// CHECK-INST: umlsll  za.d[w11, 0:3, vgx2], { z8.h, z9.h }, { z0.h, z1.h }
// CHECK-ENCODING: [0x18,0x61,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e06118 <unknown>

umlsll  za.d[w11, 0:3], {z8.h - z9.h}, {z0.h - z1.h}  // 11000001-11100000-01100001-00011000
// CHECK-INST: umlsll  za.d[w11, 0:3, vgx2], { z8.h, z9.h }, { z0.h, z1.h }
// CHECK-ENCODING: [0x18,0x61,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e06118 <unknown>

umlsll  za.d[w9, 4:7, vgx2], {z12.h, z13.h}, {z10.h, z11.h}  // 11000001-11101010-00100001-10011001
// CHECK-INST: umlsll  za.d[w9, 4:7, vgx2], { z12.h, z13.h }, { z10.h, z11.h }
// CHECK-ENCODING: [0x99,0x21,0xea,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1ea2199 <unknown>

umlsll  za.d[w9, 4:7], {z12.h - z13.h}, {z10.h - z11.h}  // 11000001-11101010-00100001-10011001
// CHECK-INST: umlsll  za.d[w9, 4:7, vgx2], { z12.h, z13.h }, { z10.h, z11.h }
// CHECK-ENCODING: [0x99,0x21,0xea,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1ea2199 <unknown>


umlsll  za.s[w8, 0:3, vgx4], {z0.b - z3.b}, z0.b  // 11000001-00110000-00000000-00011000
// CHECK-INST: umlsll  za.s[w8, 0:3, vgx4], { z0.b - z3.b }, z0.b
// CHECK-ENCODING: [0x18,0x00,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1300018 <unknown>

umlsll  za.s[w8, 0:3], {z0.b - z3.b}, z0.b  // 11000001-00110000-00000000-00011000
// CHECK-INST: umlsll  za.s[w8, 0:3, vgx4], { z0.b - z3.b }, z0.b
// CHECK-ENCODING: [0x18,0x00,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1300018 <unknown>

umlsll  za.s[w10, 4:7, vgx4], {z10.b - z13.b}, z5.b  // 11000001-00110101-01000001-01011001
// CHECK-INST: umlsll  za.s[w10, 4:7, vgx4], { z10.b - z13.b }, z5.b
// CHECK-ENCODING: [0x59,0x41,0x35,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1354159 <unknown>

umlsll  za.s[w10, 4:7], {z10.b - z13.b}, z5.b  // 11000001-00110101-01000001-01011001
// CHECK-INST: umlsll  za.s[w10, 4:7, vgx4], { z10.b - z13.b }, z5.b
// CHECK-ENCODING: [0x59,0x41,0x35,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1354159 <unknown>

umlsll  za.s[w11, 4:7, vgx4], {z13.b - z16.b}, z8.b  // 11000001-00111000-01100001-10111001
// CHECK-INST: umlsll  za.s[w11, 4:7, vgx4], { z13.b - z16.b }, z8.b
// CHECK-ENCODING: [0xb9,0x61,0x38,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13861b9 <unknown>

umlsll  za.s[w11, 4:7], {z13.b - z16.b}, z8.b  // 11000001-00111000-01100001-10111001
// CHECK-INST: umlsll  za.s[w11, 4:7, vgx4], { z13.b - z16.b }, z8.b
// CHECK-ENCODING: [0xb9,0x61,0x38,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13861b9 <unknown>

umlsll  za.s[w11, 4:7, vgx4], {z31.b - z2.b}, z15.b  // 11000001-00111111-01100011-11111001
// CHECK-INST: umlsll  za.s[w11, 4:7, vgx4], { z31.b, z0.b, z1.b, z2.b }, z15.b
// CHECK-ENCODING: [0xf9,0x63,0x3f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13f63f9 <unknown>

umlsll  za.s[w11, 4:7], {z31.b - z2.b}, z15.b  // 11000001-00111111-01100011-11111001
// CHECK-INST: umlsll  za.s[w11, 4:7, vgx4], { z31.b, z0.b, z1.b, z2.b }, z15.b
// CHECK-ENCODING: [0xf9,0x63,0x3f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13f63f9 <unknown>

umlsll  za.s[w8, 4:7, vgx4], {z17.b - z20.b}, z0.b  // 11000001-00110000-00000010-00111001
// CHECK-INST: umlsll  za.s[w8, 4:7, vgx4], { z17.b - z20.b }, z0.b
// CHECK-ENCODING: [0x39,0x02,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1300239 <unknown>

umlsll  za.s[w8, 4:7], {z17.b - z20.b}, z0.b  // 11000001-00110000-00000010-00111001
// CHECK-INST: umlsll  za.s[w8, 4:7, vgx4], { z17.b - z20.b }, z0.b
// CHECK-ENCODING: [0x39,0x02,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1300239 <unknown>

umlsll  za.s[w8, 4:7, vgx4], {z1.b - z4.b}, z14.b  // 11000001-00111110-00000000-00111001
// CHECK-INST: umlsll  za.s[w8, 4:7, vgx4], { z1.b - z4.b }, z14.b
// CHECK-ENCODING: [0x39,0x00,0x3e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13e0039 <unknown>

umlsll  za.s[w8, 4:7], {z1.b - z4.b}, z14.b  // 11000001-00111110-00000000-00111001
// CHECK-INST: umlsll  za.s[w8, 4:7, vgx4], { z1.b - z4.b }, z14.b
// CHECK-ENCODING: [0x39,0x00,0x3e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13e0039 <unknown>

umlsll  za.s[w10, 0:3, vgx4], {z19.b - z22.b}, z4.b  // 11000001-00110100-01000010-01111000
// CHECK-INST: umlsll  za.s[w10, 0:3, vgx4], { z19.b - z22.b }, z4.b
// CHECK-ENCODING: [0x78,0x42,0x34,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1344278 <unknown>

umlsll  za.s[w10, 0:3], {z19.b - z22.b}, z4.b  // 11000001-00110100-01000010-01111000
// CHECK-INST: umlsll  za.s[w10, 0:3, vgx4], { z19.b - z22.b }, z4.b
// CHECK-ENCODING: [0x78,0x42,0x34,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1344278 <unknown>

umlsll  za.s[w8, 0:3, vgx4], {z12.b - z15.b}, z2.b  // 11000001-00110010-00000001-10011000
// CHECK-INST: umlsll  za.s[w8, 0:3, vgx4], { z12.b - z15.b }, z2.b
// CHECK-ENCODING: [0x98,0x01,0x32,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1320198 <unknown>

umlsll  za.s[w8, 0:3], {z12.b - z15.b}, z2.b  // 11000001-00110010-00000001-10011000
// CHECK-INST: umlsll  za.s[w8, 0:3, vgx4], { z12.b - z15.b }, z2.b
// CHECK-ENCODING: [0x98,0x01,0x32,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1320198 <unknown>

umlsll  za.s[w10, 4:7, vgx4], {z1.b - z4.b}, z10.b  // 11000001-00111010-01000000-00111001
// CHECK-INST: umlsll  za.s[w10, 4:7, vgx4], { z1.b - z4.b }, z10.b
// CHECK-ENCODING: [0x39,0x40,0x3a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13a4039 <unknown>

umlsll  za.s[w10, 4:7], {z1.b - z4.b}, z10.b  // 11000001-00111010-01000000-00111001
// CHECK-INST: umlsll  za.s[w10, 4:7, vgx4], { z1.b - z4.b }, z10.b
// CHECK-ENCODING: [0x39,0x40,0x3a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13a4039 <unknown>

umlsll  za.s[w8, 4:7, vgx4], {z22.b - z25.b}, z14.b  // 11000001-00111110-00000010-11011001
// CHECK-INST: umlsll  za.s[w8, 4:7, vgx4], { z22.b - z25.b }, z14.b
// CHECK-ENCODING: [0xd9,0x02,0x3e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13e02d9 <unknown>

umlsll  za.s[w8, 4:7], {z22.b - z25.b}, z14.b  // 11000001-00111110-00000010-11011001
// CHECK-INST: umlsll  za.s[w8, 4:7, vgx4], { z22.b - z25.b }, z14.b
// CHECK-ENCODING: [0xd9,0x02,0x3e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13e02d9 <unknown>

umlsll  za.s[w11, 0:3, vgx4], {z9.b - z12.b}, z1.b  // 11000001-00110001-01100001-00111000
// CHECK-INST: umlsll  za.s[w11, 0:3, vgx4], { z9.b - z12.b }, z1.b
// CHECK-ENCODING: [0x38,0x61,0x31,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1316138 <unknown>

umlsll  za.s[w11, 0:3], {z9.b - z12.b}, z1.b  // 11000001-00110001-01100001-00111000
// CHECK-INST: umlsll  za.s[w11, 0:3, vgx4], { z9.b - z12.b }, z1.b
// CHECK-ENCODING: [0x38,0x61,0x31,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1316138 <unknown>

umlsll  za.s[w9, 4:7, vgx4], {z12.b - z15.b}, z11.b  // 11000001-00111011-00100001-10011001
// CHECK-INST: umlsll  za.s[w9, 4:7, vgx4], { z12.b - z15.b }, z11.b
// CHECK-ENCODING: [0x99,0x21,0x3b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13b2199 <unknown>

umlsll  za.s[w9, 4:7], {z12.b - z15.b}, z11.b  // 11000001-00111011-00100001-10011001
// CHECK-INST: umlsll  za.s[w9, 4:7, vgx4], { z12.b - z15.b }, z11.b
// CHECK-ENCODING: [0x99,0x21,0x3b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13b2199 <unknown>


umlsll  za.s[w8, 0:3, vgx4], {z0.b - z3.b}, z0.b[0]  // 11000001-00010000-10000000-00011000
// CHECK-INST: umlsll  za.s[w8, 0:3, vgx4], { z0.b - z3.b }, z0.b[0]
// CHECK-ENCODING: [0x18,0x80,0x10,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1108018 <unknown>

umlsll  za.s[w8, 0:3], {z0.b - z3.b}, z0.b[0]  // 11000001-00010000-10000000-00011000
// CHECK-INST: umlsll  za.s[w8, 0:3, vgx4], { z0.b - z3.b }, z0.b[0]
// CHECK-ENCODING: [0x18,0x80,0x10,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1108018 <unknown>

umlsll  za.s[w10, 4:7, vgx4], {z8.b - z11.b}, z5.b[6]  // 11000001-00010101-11000101-00011101
// CHECK-INST: umlsll  za.s[w10, 4:7, vgx4], { z8.b - z11.b }, z5.b[6]
// CHECK-ENCODING: [0x1d,0xc5,0x15,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c115c51d <unknown>

umlsll  za.s[w10, 4:7], {z8.b - z11.b}, z5.b[6]  // 11000001-00010101-11000101-00011101
// CHECK-INST: umlsll  za.s[w10, 4:7, vgx4], { z8.b - z11.b }, z5.b[6]
// CHECK-ENCODING: [0x1d,0xc5,0x15,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c115c51d <unknown>

umlsll  za.s[w11, 4:7, vgx4], {z12.b - z15.b}, z8.b[15]  // 11000001-00011000-11101101-10011111
// CHECK-INST: umlsll  za.s[w11, 4:7, vgx4], { z12.b - z15.b }, z8.b[15]
// CHECK-ENCODING: [0x9f,0xed,0x18,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c118ed9f <unknown>

umlsll  za.s[w11, 4:7], {z12.b - z15.b}, z8.b[15]  // 11000001-00011000-11101101-10011111
// CHECK-INST: umlsll  za.s[w11, 4:7, vgx4], { z12.b - z15.b }, z8.b[15]
// CHECK-ENCODING: [0x9f,0xed,0x18,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c118ed9f <unknown>

umlsll  za.s[w11, 4:7, vgx4], {z28.b - z31.b}, z15.b[15]  // 11000001-00011111-11101111-10011111
// CHECK-INST: umlsll  za.s[w11, 4:7, vgx4], { z28.b - z31.b }, z15.b[15]
// CHECK-ENCODING: [0x9f,0xef,0x1f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11fef9f <unknown>

umlsll  za.s[w11, 4:7], {z28.b - z31.b}, z15.b[15]  // 11000001-00011111-11101111-10011111
// CHECK-INST: umlsll  za.s[w11, 4:7, vgx4], { z28.b - z31.b }, z15.b[15]
// CHECK-ENCODING: [0x9f,0xef,0x1f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11fef9f <unknown>

umlsll  za.s[w8, 4:7, vgx4], {z16.b - z19.b}, z0.b[14]  // 11000001-00010000-10001110-00011101
// CHECK-INST: umlsll  za.s[w8, 4:7, vgx4], { z16.b - z19.b }, z0.b[14]
// CHECK-ENCODING: [0x1d,0x8e,0x10,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1108e1d <unknown>

umlsll  za.s[w8, 4:7], {z16.b - z19.b}, z0.b[14]  // 11000001-00010000-10001110-00011101
// CHECK-INST: umlsll  za.s[w8, 4:7, vgx4], { z16.b - z19.b }, z0.b[14]
// CHECK-ENCODING: [0x1d,0x8e,0x10,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1108e1d <unknown>

umlsll  za.s[w8, 4:7, vgx4], {z0.b - z3.b}, z14.b[4]  // 11000001-00011110-10000100-00011001
// CHECK-INST: umlsll  za.s[w8, 4:7, vgx4], { z0.b - z3.b }, z14.b[4]
// CHECK-ENCODING: [0x19,0x84,0x1e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11e8419 <unknown>

umlsll  za.s[w8, 4:7], {z0.b - z3.b}, z14.b[4]  // 11000001-00011110-10000100-00011001
// CHECK-INST: umlsll  za.s[w8, 4:7, vgx4], { z0.b - z3.b }, z14.b[4]
// CHECK-ENCODING: [0x19,0x84,0x1e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11e8419 <unknown>

umlsll  za.s[w10, 0:3, vgx4], {z16.b - z19.b}, z4.b[4]  // 11000001-00010100-11000110-00011000
// CHECK-INST: umlsll  za.s[w10, 0:3, vgx4], { z16.b - z19.b }, z4.b[4]
// CHECK-ENCODING: [0x18,0xc6,0x14,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c114c618 <unknown>

umlsll  za.s[w10, 0:3], {z16.b - z19.b}, z4.b[4]  // 11000001-00010100-11000110-00011000
// CHECK-INST: umlsll  za.s[w10, 0:3, vgx4], { z16.b - z19.b }, z4.b[4]
// CHECK-ENCODING: [0x18,0xc6,0x14,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c114c618 <unknown>

umlsll  za.s[w8, 0:3, vgx4], {z12.b - z15.b}, z2.b[8]  // 11000001-00010010-10001001-10011000
// CHECK-INST: umlsll  za.s[w8, 0:3, vgx4], { z12.b - z15.b }, z2.b[8]
// CHECK-ENCODING: [0x98,0x89,0x12,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1128998 <unknown>

umlsll  za.s[w8, 0:3], {z12.b - z15.b}, z2.b[8]  // 11000001-00010010-10001001-10011000
// CHECK-INST: umlsll  za.s[w8, 0:3, vgx4], { z12.b - z15.b }, z2.b[8]
// CHECK-ENCODING: [0x98,0x89,0x12,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1128998 <unknown>

umlsll  za.s[w10, 4:7, vgx4], {z0.b - z3.b}, z10.b[8]  // 11000001-00011010-11001000-00011001
// CHECK-INST: umlsll  za.s[w10, 4:7, vgx4], { z0.b - z3.b }, z10.b[8]
// CHECK-ENCODING: [0x19,0xc8,0x1a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11ac819 <unknown>

umlsll  za.s[w10, 4:7], {z0.b - z3.b}, z10.b[8]  // 11000001-00011010-11001000-00011001
// CHECK-INST: umlsll  za.s[w10, 4:7, vgx4], { z0.b - z3.b }, z10.b[8]
// CHECK-ENCODING: [0x19,0xc8,0x1a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11ac819 <unknown>

umlsll  za.s[w8, 4:7, vgx4], {z20.b - z23.b}, z14.b[10]  // 11000001-00011110-10001010-10011101
// CHECK-INST: umlsll  za.s[w8, 4:7, vgx4], { z20.b - z23.b }, z14.b[10]
// CHECK-ENCODING: [0x9d,0x8a,0x1e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11e8a9d <unknown>

umlsll  za.s[w8, 4:7], {z20.b - z23.b}, z14.b[10]  // 11000001-00011110-10001010-10011101
// CHECK-INST: umlsll  za.s[w8, 4:7, vgx4], { z20.b - z23.b }, z14.b[10]
// CHECK-ENCODING: [0x9d,0x8a,0x1e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11e8a9d <unknown>

umlsll  za.s[w11, 0:3, vgx4], {z8.b - z11.b}, z1.b[5]  // 11000001-00010001-11100101-00011010
// CHECK-INST: umlsll  za.s[w11, 0:3, vgx4], { z8.b - z11.b }, z1.b[5]
// CHECK-ENCODING: [0x1a,0xe5,0x11,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c111e51a <unknown>

umlsll  za.s[w11, 0:3], {z8.b - z11.b}, z1.b[5]  // 11000001-00010001-11100101-00011010
// CHECK-INST: umlsll  za.s[w11, 0:3, vgx4], { z8.b - z11.b }, z1.b[5]
// CHECK-ENCODING: [0x1a,0xe5,0x11,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c111e51a <unknown>

umlsll  za.s[w9, 4:7, vgx4], {z12.b - z15.b}, z11.b[11]  // 11000001-00011011-10101001-10011111
// CHECK-INST: umlsll  za.s[w9, 4:7, vgx4], { z12.b - z15.b }, z11.b[11]
// CHECK-ENCODING: [0x9f,0xa9,0x1b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11ba99f <unknown>

umlsll  za.s[w9, 4:7], {z12.b - z15.b}, z11.b[11]  // 11000001-00011011-10101001-10011111
// CHECK-INST: umlsll  za.s[w9, 4:7, vgx4], { z12.b - z15.b }, z11.b[11]
// CHECK-ENCODING: [0x9f,0xa9,0x1b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11ba99f <unknown>


umlsll  za.s[w8, 0:3, vgx4], {z0.b - z3.b}, {z0.b - z3.b}  // 11000001-10100001-00000000-00011000
// CHECK-INST: umlsll  za.s[w8, 0:3, vgx4], { z0.b - z3.b }, { z0.b - z3.b }
// CHECK-ENCODING: [0x18,0x00,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a10018 <unknown>

umlsll  za.s[w8, 0:3], {z0.b - z3.b}, {z0.b - z3.b}  // 11000001-10100001-00000000-00011000
// CHECK-INST: umlsll  za.s[w8, 0:3, vgx4], { z0.b - z3.b }, { z0.b - z3.b }
// CHECK-ENCODING: [0x18,0x00,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a10018 <unknown>

umlsll  za.s[w10, 4:7, vgx4], {z8.b - z11.b}, {z20.b - z23.b}  // 11000001-10110101-01000001-00011001
// CHECK-INST: umlsll  za.s[w10, 4:7, vgx4], { z8.b - z11.b }, { z20.b - z23.b }
// CHECK-ENCODING: [0x19,0x41,0xb5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b54119 <unknown>

umlsll  za.s[w10, 4:7], {z8.b - z11.b}, {z20.b - z23.b}  // 11000001-10110101-01000001-00011001
// CHECK-INST: umlsll  za.s[w10, 4:7, vgx4], { z8.b - z11.b }, { z20.b - z23.b }
// CHECK-ENCODING: [0x19,0x41,0xb5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b54119 <unknown>

umlsll  za.s[w11, 4:7, vgx4], {z12.b - z15.b}, {z8.b - z11.b}  // 11000001-10101001-01100001-10011001
// CHECK-INST: umlsll  za.s[w11, 4:7, vgx4], { z12.b - z15.b }, { z8.b - z11.b }
// CHECK-ENCODING: [0x99,0x61,0xa9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a96199 <unknown>

umlsll  za.s[w11, 4:7], {z12.b - z15.b}, {z8.b - z11.b}  // 11000001-10101001-01100001-10011001
// CHECK-INST: umlsll  za.s[w11, 4:7, vgx4], { z12.b - z15.b }, { z8.b - z11.b }
// CHECK-ENCODING: [0x99,0x61,0xa9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a96199 <unknown>

umlsll  za.s[w11, 4:7, vgx4], {z28.b - z31.b}, {z28.b - z31.b}  // 11000001-10111101-01100011-10011001
// CHECK-INST: umlsll  za.s[w11, 4:7, vgx4], { z28.b - z31.b }, { z28.b - z31.b }
// CHECK-ENCODING: [0x99,0x63,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bd6399 <unknown>

umlsll  za.s[w11, 4:7], {z28.b - z31.b}, {z28.b - z31.b}  // 11000001-10111101-01100011-10011001
// CHECK-INST: umlsll  za.s[w11, 4:7, vgx4], { z28.b - z31.b }, { z28.b - z31.b }
// CHECK-ENCODING: [0x99,0x63,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bd6399 <unknown>

umlsll  za.s[w8, 4:7, vgx4], {z16.b - z19.b}, {z16.b - z19.b}  // 11000001-10110001-00000010-00011001
// CHECK-INST: umlsll  za.s[w8, 4:7, vgx4], { z16.b - z19.b }, { z16.b - z19.b }
// CHECK-ENCODING: [0x19,0x02,0xb1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b10219 <unknown>

umlsll  za.s[w8, 4:7], {z16.b - z19.b}, {z16.b - z19.b}  // 11000001-10110001-00000010-00011001
// CHECK-INST: umlsll  za.s[w8, 4:7, vgx4], { z16.b - z19.b }, { z16.b - z19.b }
// CHECK-ENCODING: [0x19,0x02,0xb1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b10219 <unknown>

umlsll  za.s[w8, 4:7, vgx4], {z0.b - z3.b}, {z28.b - z31.b}  // 11000001-10111101-00000000-00011001
// CHECK-INST: umlsll  za.s[w8, 4:7, vgx4], { z0.b - z3.b }, { z28.b - z31.b }
// CHECK-ENCODING: [0x19,0x00,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bd0019 <unknown>

umlsll  za.s[w8, 4:7], {z0.b - z3.b}, {z28.b - z31.b}  // 11000001-10111101-00000000-00011001
// CHECK-INST: umlsll  za.s[w8, 4:7, vgx4], { z0.b - z3.b }, { z28.b - z31.b }
// CHECK-ENCODING: [0x19,0x00,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bd0019 <unknown>

umlsll  za.s[w10, 0:3, vgx4], {z16.b - z19.b}, {z20.b - z23.b}  // 11000001-10110101-01000010-00011000
// CHECK-INST: umlsll  za.s[w10, 0:3, vgx4], { z16.b - z19.b }, { z20.b - z23.b }
// CHECK-ENCODING: [0x18,0x42,0xb5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b54218 <unknown>

umlsll  za.s[w10, 0:3], {z16.b - z19.b}, {z20.b - z23.b}  // 11000001-10110101-01000010-00011000
// CHECK-INST: umlsll  za.s[w10, 0:3, vgx4], { z16.b - z19.b }, { z20.b - z23.b }
// CHECK-ENCODING: [0x18,0x42,0xb5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b54218 <unknown>

umlsll  za.s[w8, 0:3, vgx4], {z12.b - z15.b}, {z0.b - z3.b}  // 11000001-10100001-00000001-10011000
// CHECK-INST: umlsll  za.s[w8, 0:3, vgx4], { z12.b - z15.b }, { z0.b - z3.b }
// CHECK-ENCODING: [0x98,0x01,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a10198 <unknown>

umlsll  za.s[w8, 0:3], {z12.b - z15.b}, {z0.b - z3.b}  // 11000001-10100001-00000001-10011000
// CHECK-INST: umlsll  za.s[w8, 0:3, vgx4], { z12.b - z15.b }, { z0.b - z3.b }
// CHECK-ENCODING: [0x98,0x01,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a10198 <unknown>

umlsll  za.s[w10, 4:7, vgx4], {z0.b - z3.b}, {z24.b - z27.b}  // 11000001-10111001-01000000-00011001
// CHECK-INST: umlsll  za.s[w10, 4:7, vgx4], { z0.b - z3.b }, { z24.b - z27.b }
// CHECK-ENCODING: [0x19,0x40,0xb9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b94019 <unknown>

umlsll  za.s[w10, 4:7], {z0.b - z3.b}, {z24.b - z27.b}  // 11000001-10111001-01000000-00011001
// CHECK-INST: umlsll  za.s[w10, 4:7, vgx4], { z0.b - z3.b }, { z24.b - z27.b }
// CHECK-ENCODING: [0x19,0x40,0xb9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b94019 <unknown>

umlsll  za.s[w8, 4:7, vgx4], {z20.b - z23.b}, {z28.b - z31.b}  // 11000001-10111101-00000010-10011001
// CHECK-INST: umlsll  za.s[w8, 4:7, vgx4], { z20.b - z23.b }, { z28.b - z31.b }
// CHECK-ENCODING: [0x99,0x02,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bd0299 <unknown>

umlsll  za.s[w8, 4:7], {z20.b - z23.b}, {z28.b - z31.b}  // 11000001-10111101-00000010-10011001
// CHECK-INST: umlsll  za.s[w8, 4:7, vgx4], { z20.b - z23.b }, { z28.b - z31.b }
// CHECK-ENCODING: [0x99,0x02,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bd0299 <unknown>

umlsll  za.s[w11, 0:3, vgx4], {z8.b - z11.b}, {z0.b - z3.b}  // 11000001-10100001-01100001-00011000
// CHECK-INST: umlsll  za.s[w11, 0:3, vgx4], { z8.b - z11.b }, { z0.b - z3.b }
// CHECK-ENCODING: [0x18,0x61,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a16118 <unknown>

umlsll  za.s[w11, 0:3], {z8.b - z11.b}, {z0.b - z3.b}  // 11000001-10100001-01100001-00011000
// CHECK-INST: umlsll  za.s[w11, 0:3, vgx4], { z8.b - z11.b }, { z0.b - z3.b }
// CHECK-ENCODING: [0x18,0x61,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a16118 <unknown>

umlsll  za.s[w9, 4:7, vgx4], {z12.b - z15.b}, {z8.b - z11.b}  // 11000001-10101001-00100001-10011001
// CHECK-INST: umlsll  za.s[w9, 4:7, vgx4], { z12.b - z15.b }, { z8.b - z11.b }
// CHECK-ENCODING: [0x99,0x21,0xa9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a92199 <unknown>

umlsll  za.s[w9, 4:7], {z12.b - z15.b}, {z8.b - z11.b}  // 11000001-10101001-00100001-10011001
// CHECK-INST: umlsll  za.s[w9, 4:7, vgx4], { z12.b - z15.b }, { z8.b - z11.b }
// CHECK-ENCODING: [0x99,0x21,0xa9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a92199 <unknown>


umlsll  za.d[w8, 0:3, vgx4], {z0.h - z3.h}, z0.h  // 11000001-01110000-00000000-00011000
// CHECK-INST: umlsll  za.d[w8, 0:3, vgx4], { z0.h - z3.h }, z0.h
// CHECK-ENCODING: [0x18,0x00,0x70,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1700018 <unknown>

umlsll  za.d[w8, 0:3], {z0.h - z3.h}, z0.h  // 11000001-01110000-00000000-00011000
// CHECK-INST: umlsll  za.d[w8, 0:3, vgx4], { z0.h - z3.h }, z0.h
// CHECK-ENCODING: [0x18,0x00,0x70,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1700018 <unknown>

umlsll  za.d[w10, 4:7, vgx4], {z10.h - z13.h}, z5.h  // 11000001-01110101-01000001-01011001
// CHECK-INST: umlsll  za.d[w10, 4:7, vgx4], { z10.h - z13.h }, z5.h
// CHECK-ENCODING: [0x59,0x41,0x75,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1754159 <unknown>

umlsll  za.d[w10, 4:7], {z10.h - z13.h}, z5.h  // 11000001-01110101-01000001-01011001
// CHECK-INST: umlsll  za.d[w10, 4:7, vgx4], { z10.h - z13.h }, z5.h
// CHECK-ENCODING: [0x59,0x41,0x75,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1754159 <unknown>

umlsll  za.d[w11, 4:7, vgx4], {z13.h - z16.h}, z8.h  // 11000001-01111000-01100001-10111001
// CHECK-INST: umlsll  za.d[w11, 4:7, vgx4], { z13.h - z16.h }, z8.h
// CHECK-ENCODING: [0xb9,0x61,0x78,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17861b9 <unknown>

umlsll  za.d[w11, 4:7], {z13.h - z16.h}, z8.h  // 11000001-01111000-01100001-10111001
// CHECK-INST: umlsll  za.d[w11, 4:7, vgx4], { z13.h - z16.h }, z8.h
// CHECK-ENCODING: [0xb9,0x61,0x78,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17861b9 <unknown>

umlsll  za.d[w11, 4:7, vgx4], {z31.h - z2.h}, z15.h  // 11000001-01111111-01100011-11111001
// CHECK-INST: umlsll  za.d[w11, 4:7, vgx4], { z31.h, z0.h, z1.h, z2.h }, z15.h
// CHECK-ENCODING: [0xf9,0x63,0x7f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17f63f9 <unknown>

umlsll  za.d[w11, 4:7], {z31.h - z2.h}, z15.h  // 11000001-01111111-01100011-11111001
// CHECK-INST: umlsll  za.d[w11, 4:7, vgx4], { z31.h, z0.h, z1.h, z2.h }, z15.h
// CHECK-ENCODING: [0xf9,0x63,0x7f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17f63f9 <unknown>

umlsll  za.d[w8, 4:7, vgx4], {z17.h - z20.h}, z0.h  // 11000001-01110000-00000010-00111001
// CHECK-INST: umlsll  za.d[w8, 4:7, vgx4], { z17.h - z20.h }, z0.h
// CHECK-ENCODING: [0x39,0x02,0x70,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1700239 <unknown>

umlsll  za.d[w8, 4:7], {z17.h - z20.h}, z0.h  // 11000001-01110000-00000010-00111001
// CHECK-INST: umlsll  za.d[w8, 4:7, vgx4], { z17.h - z20.h }, z0.h
// CHECK-ENCODING: [0x39,0x02,0x70,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1700239 <unknown>

umlsll  za.d[w8, 4:7, vgx4], {z1.h - z4.h}, z14.h  // 11000001-01111110-00000000-00111001
// CHECK-INST: umlsll  za.d[w8, 4:7, vgx4], { z1.h - z4.h }, z14.h
// CHECK-ENCODING: [0x39,0x00,0x7e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17e0039 <unknown>

umlsll  za.d[w8, 4:7], {z1.h - z4.h}, z14.h  // 11000001-01111110-00000000-00111001
// CHECK-INST: umlsll  za.d[w8, 4:7, vgx4], { z1.h - z4.h }, z14.h
// CHECK-ENCODING: [0x39,0x00,0x7e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17e0039 <unknown>

umlsll  za.d[w10, 0:3, vgx4], {z19.h - z22.h}, z4.h  // 11000001-01110100-01000010-01111000
// CHECK-INST: umlsll  za.d[w10, 0:3, vgx4], { z19.h - z22.h }, z4.h
// CHECK-ENCODING: [0x78,0x42,0x74,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1744278 <unknown>

umlsll  za.d[w10, 0:3], {z19.h - z22.h}, z4.h  // 11000001-01110100-01000010-01111000
// CHECK-INST: umlsll  za.d[w10, 0:3, vgx4], { z19.h - z22.h }, z4.h
// CHECK-ENCODING: [0x78,0x42,0x74,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1744278 <unknown>

umlsll  za.d[w8, 0:3, vgx4], {z12.h - z15.h}, z2.h  // 11000001-01110010-00000001-10011000
// CHECK-INST: umlsll  za.d[w8, 0:3, vgx4], { z12.h - z15.h }, z2.h
// CHECK-ENCODING: [0x98,0x01,0x72,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1720198 <unknown>

umlsll  za.d[w8, 0:3], {z12.h - z15.h}, z2.h  // 11000001-01110010-00000001-10011000
// CHECK-INST: umlsll  za.d[w8, 0:3, vgx4], { z12.h - z15.h }, z2.h
// CHECK-ENCODING: [0x98,0x01,0x72,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1720198 <unknown>

umlsll  za.d[w10, 4:7, vgx4], {z1.h - z4.h}, z10.h  // 11000001-01111010-01000000-00111001
// CHECK-INST: umlsll  za.d[w10, 4:7, vgx4], { z1.h - z4.h }, z10.h
// CHECK-ENCODING: [0x39,0x40,0x7a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17a4039 <unknown>

umlsll  za.d[w10, 4:7], {z1.h - z4.h}, z10.h  // 11000001-01111010-01000000-00111001
// CHECK-INST: umlsll  za.d[w10, 4:7, vgx4], { z1.h - z4.h }, z10.h
// CHECK-ENCODING: [0x39,0x40,0x7a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17a4039 <unknown>

umlsll  za.d[w8, 4:7, vgx4], {z22.h - z25.h}, z14.h  // 11000001-01111110-00000010-11011001
// CHECK-INST: umlsll  za.d[w8, 4:7, vgx4], { z22.h - z25.h }, z14.h
// CHECK-ENCODING: [0xd9,0x02,0x7e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17e02d9 <unknown>

umlsll  za.d[w8, 4:7], {z22.h - z25.h}, z14.h  // 11000001-01111110-00000010-11011001
// CHECK-INST: umlsll  za.d[w8, 4:7, vgx4], { z22.h - z25.h }, z14.h
// CHECK-ENCODING: [0xd9,0x02,0x7e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17e02d9 <unknown>

umlsll  za.d[w11, 0:3, vgx4], {z9.h - z12.h}, z1.h  // 11000001-01110001-01100001-00111000
// CHECK-INST: umlsll  za.d[w11, 0:3, vgx4], { z9.h - z12.h }, z1.h
// CHECK-ENCODING: [0x38,0x61,0x71,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1716138 <unknown>

umlsll  za.d[w11, 0:3], {z9.h - z12.h}, z1.h  // 11000001-01110001-01100001-00111000
// CHECK-INST: umlsll  za.d[w11, 0:3, vgx4], { z9.h - z12.h }, z1.h
// CHECK-ENCODING: [0x38,0x61,0x71,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1716138 <unknown>

umlsll  za.d[w9, 4:7, vgx4], {z12.h - z15.h}, z11.h  // 11000001-01111011-00100001-10011001
// CHECK-INST: umlsll  za.d[w9, 4:7, vgx4], { z12.h - z15.h }, z11.h
// CHECK-ENCODING: [0x99,0x21,0x7b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17b2199 <unknown>

umlsll  za.d[w9, 4:7], {z12.h - z15.h}, z11.h  // 11000001-01111011-00100001-10011001
// CHECK-INST: umlsll  za.d[w9, 4:7, vgx4], { z12.h - z15.h }, z11.h
// CHECK-ENCODING: [0x99,0x21,0x7b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17b2199 <unknown>


umlsll  za.d[w8, 0:3, vgx4], {z0.h - z3.h}, z0.h[0]  // 11000001-10010000-10000000-00011000
// CHECK-INST: umlsll  za.d[w8, 0:3, vgx4], { z0.h - z3.h }, z0.h[0]
// CHECK-ENCODING: [0x18,0x80,0x90,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1908018 <unknown>

umlsll  za.d[w8, 0:3], {z0.h - z3.h}, z0.h[0]  // 11000001-10010000-10000000-00011000
// CHECK-INST: umlsll  za.d[w8, 0:3, vgx4], { z0.h - z3.h }, z0.h[0]
// CHECK-ENCODING: [0x18,0x80,0x90,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1908018 <unknown>

umlsll  za.d[w10, 4:7, vgx4], {z8.h - z11.h}, z5.h[6]  // 11000001-10010101-11000101-00011101
// CHECK-INST: umlsll  za.d[w10, 4:7, vgx4], { z8.h - z11.h }, z5.h[6]
// CHECK-ENCODING: [0x1d,0xc5,0x95,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c195c51d <unknown>

umlsll  za.d[w10, 4:7], {z8.h - z11.h}, z5.h[6]  // 11000001-10010101-11000101-00011101
// CHECK-INST: umlsll  za.d[w10, 4:7, vgx4], { z8.h - z11.h }, z5.h[6]
// CHECK-ENCODING: [0x1d,0xc5,0x95,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c195c51d <unknown>

umlsll  za.d[w11, 4:7, vgx4], {z12.h - z15.h}, z8.h[7]  // 11000001-10011000-11100101-10011111
// CHECK-INST: umlsll  za.d[w11, 4:7, vgx4], { z12.h - z15.h }, z8.h[7]
// CHECK-ENCODING: [0x9f,0xe5,0x98,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c198e59f <unknown>

umlsll  za.d[w11, 4:7], {z12.h - z15.h}, z8.h[7]  // 11000001-10011000-11100101-10011111
// CHECK-INST: umlsll  za.d[w11, 4:7, vgx4], { z12.h - z15.h }, z8.h[7]
// CHECK-ENCODING: [0x9f,0xe5,0x98,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c198e59f <unknown>

umlsll  za.d[w11, 4:7, vgx4], {z28.h - z31.h}, z15.h[7]  // 11000001-10011111-11100111-10011111
// CHECK-INST: umlsll  za.d[w11, 4:7, vgx4], { z28.h - z31.h }, z15.h[7]
// CHECK-ENCODING: [0x9f,0xe7,0x9f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c19fe79f <unknown>

umlsll  za.d[w11, 4:7], {z28.h - z31.h}, z15.h[7]  // 11000001-10011111-11100111-10011111
// CHECK-INST: umlsll  za.d[w11, 4:7, vgx4], { z28.h - z31.h }, z15.h[7]
// CHECK-ENCODING: [0x9f,0xe7,0x9f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c19fe79f <unknown>

umlsll  za.d[w8, 4:7, vgx4], {z16.h - z19.h}, z0.h[6]  // 11000001-10010000-10000110-00011101
// CHECK-INST: umlsll  za.d[w8, 4:7, vgx4], { z16.h - z19.h }, z0.h[6]
// CHECK-ENCODING: [0x1d,0x86,0x90,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c190861d <unknown>

umlsll  za.d[w8, 4:7], {z16.h - z19.h}, z0.h[6]  // 11000001-10010000-10000110-00011101
// CHECK-INST: umlsll  za.d[w8, 4:7, vgx4], { z16.h - z19.h }, z0.h[6]
// CHECK-ENCODING: [0x1d,0x86,0x90,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c190861d <unknown>

umlsll  za.d[w8, 4:7, vgx4], {z0.h - z3.h}, z14.h[4]  // 11000001-10011110-10000100-00011001
// CHECK-INST: umlsll  za.d[w8, 4:7, vgx4], { z0.h - z3.h }, z14.h[4]
// CHECK-ENCODING: [0x19,0x84,0x9e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c19e8419 <unknown>

umlsll  za.d[w8, 4:7], {z0.h - z3.h}, z14.h[4]  // 11000001-10011110-10000100-00011001
// CHECK-INST: umlsll  za.d[w8, 4:7, vgx4], { z0.h - z3.h }, z14.h[4]
// CHECK-ENCODING: [0x19,0x84,0x9e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c19e8419 <unknown>

umlsll  za.d[w10, 0:3, vgx4], {z16.h - z19.h}, z4.h[4]  // 11000001-10010100-11000110-00011000
// CHECK-INST: umlsll  za.d[w10, 0:3, vgx4], { z16.h - z19.h }, z4.h[4]
// CHECK-ENCODING: [0x18,0xc6,0x94,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c194c618 <unknown>

umlsll  za.d[w10, 0:3], {z16.h - z19.h}, z4.h[4]  // 11000001-10010100-11000110-00011000
// CHECK-INST: umlsll  za.d[w10, 0:3, vgx4], { z16.h - z19.h }, z4.h[4]
// CHECK-ENCODING: [0x18,0xc6,0x94,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c194c618 <unknown>

umlsll  za.d[w8, 0:3, vgx4], {z12.h - z15.h}, z2.h[0]  // 11000001-10010010-10000001-10011000
// CHECK-INST: umlsll  za.d[w8, 0:3, vgx4], { z12.h - z15.h }, z2.h[0]
// CHECK-ENCODING: [0x98,0x81,0x92,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1928198 <unknown>

umlsll  za.d[w8, 0:3], {z12.h - z15.h}, z2.h[0]  // 11000001-10010010-10000001-10011000
// CHECK-INST: umlsll  za.d[w8, 0:3, vgx4], { z12.h - z15.h }, z2.h[0]
// CHECK-ENCODING: [0x98,0x81,0x92,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1928198 <unknown>

umlsll  za.d[w10, 4:7, vgx4], {z0.h - z3.h}, z10.h[0]  // 11000001-10011010-11000000-00011001
// CHECK-INST: umlsll  za.d[w10, 4:7, vgx4], { z0.h - z3.h }, z10.h[0]
// CHECK-ENCODING: [0x19,0xc0,0x9a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c19ac019 <unknown>

umlsll  za.d[w10, 4:7], {z0.h - z3.h}, z10.h[0]  // 11000001-10011010-11000000-00011001
// CHECK-INST: umlsll  za.d[w10, 4:7, vgx4], { z0.h - z3.h }, z10.h[0]
// CHECK-ENCODING: [0x19,0xc0,0x9a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c19ac019 <unknown>

umlsll  za.d[w8, 4:7, vgx4], {z20.h - z23.h}, z14.h[2]  // 11000001-10011110-10000010-10011101
// CHECK-INST: umlsll  za.d[w8, 4:7, vgx4], { z20.h - z23.h }, z14.h[2]
// CHECK-ENCODING: [0x9d,0x82,0x9e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c19e829d <unknown>

umlsll  za.d[w8, 4:7], {z20.h - z23.h}, z14.h[2]  // 11000001-10011110-10000010-10011101
// CHECK-INST: umlsll  za.d[w8, 4:7, vgx4], { z20.h - z23.h }, z14.h[2]
// CHECK-ENCODING: [0x9d,0x82,0x9e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c19e829d <unknown>

umlsll  za.d[w11, 0:3, vgx4], {z8.h - z11.h}, z1.h[5]  // 11000001-10010001-11100101-00011010
// CHECK-INST: umlsll  za.d[w11, 0:3, vgx4], { z8.h - z11.h }, z1.h[5]
// CHECK-ENCODING: [0x1a,0xe5,0x91,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c191e51a <unknown>

umlsll  za.d[w11, 0:3], {z8.h - z11.h}, z1.h[5]  // 11000001-10010001-11100101-00011010
// CHECK-INST: umlsll  za.d[w11, 0:3, vgx4], { z8.h - z11.h }, z1.h[5]
// CHECK-ENCODING: [0x1a,0xe5,0x91,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c191e51a <unknown>

umlsll  za.d[w9, 4:7, vgx4], {z12.h - z15.h}, z11.h[3]  // 11000001-10011011-10100001-10011111
// CHECK-INST: umlsll  za.d[w9, 4:7, vgx4], { z12.h - z15.h }, z11.h[3]
// CHECK-ENCODING: [0x9f,0xa1,0x9b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c19ba19f <unknown>

umlsll  za.d[w9, 4:7], {z12.h - z15.h}, z11.h[3]  // 11000001-10011011-10100001-10011111
// CHECK-INST: umlsll  za.d[w9, 4:7, vgx4], { z12.h - z15.h }, z11.h[3]
// CHECK-ENCODING: [0x9f,0xa1,0x9b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c19ba19f <unknown>


umlsll  za.d[w8, 0:3, vgx4], {z0.h - z3.h}, {z0.h - z3.h}  // 11000001-11100001-00000000-00011000
// CHECK-INST: umlsll  za.d[w8, 0:3, vgx4], { z0.h - z3.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x18,0x00,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e10018 <unknown>

umlsll  za.d[w8, 0:3], {z0.h - z3.h}, {z0.h - z3.h}  // 11000001-11100001-00000000-00011000
// CHECK-INST: umlsll  za.d[w8, 0:3, vgx4], { z0.h - z3.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x18,0x00,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e10018 <unknown>

umlsll  za.d[w10, 4:7, vgx4], {z8.h - z11.h}, {z20.h - z23.h}  // 11000001-11110101-01000001-00011001
// CHECK-INST: umlsll  za.d[w10, 4:7, vgx4], { z8.h - z11.h }, { z20.h - z23.h }
// CHECK-ENCODING: [0x19,0x41,0xf5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f54119 <unknown>

umlsll  za.d[w10, 4:7], {z8.h - z11.h}, {z20.h - z23.h}  // 11000001-11110101-01000001-00011001
// CHECK-INST: umlsll  za.d[w10, 4:7, vgx4], { z8.h - z11.h }, { z20.h - z23.h }
// CHECK-ENCODING: [0x19,0x41,0xf5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f54119 <unknown>

umlsll  za.d[w11, 4:7, vgx4], {z12.h - z15.h}, {z8.h - z11.h}  // 11000001-11101001-01100001-10011001
// CHECK-INST: umlsll  za.d[w11, 4:7, vgx4], { z12.h - z15.h }, { z8.h - z11.h }
// CHECK-ENCODING: [0x99,0x61,0xe9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e96199 <unknown>

umlsll  za.d[w11, 4:7], {z12.h - z15.h}, {z8.h - z11.h}  // 11000001-11101001-01100001-10011001
// CHECK-INST: umlsll  za.d[w11, 4:7, vgx4], { z12.h - z15.h }, { z8.h - z11.h }
// CHECK-ENCODING: [0x99,0x61,0xe9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e96199 <unknown>

umlsll  za.d[w11, 4:7, vgx4], {z28.h - z31.h}, {z28.h - z31.h}  // 11000001-11111101-01100011-10011001
// CHECK-INST: umlsll  za.d[w11, 4:7, vgx4], { z28.h - z31.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x99,0x63,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd6399 <unknown>

umlsll  za.d[w11, 4:7], {z28.h - z31.h}, {z28.h - z31.h}  // 11000001-11111101-01100011-10011001
// CHECK-INST: umlsll  za.d[w11, 4:7, vgx4], { z28.h - z31.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x99,0x63,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd6399 <unknown>

umlsll  za.d[w8, 4:7, vgx4], {z16.h - z19.h}, {z16.h - z19.h}  // 11000001-11110001-00000010-00011001
// CHECK-INST: umlsll  za.d[w8, 4:7, vgx4], { z16.h - z19.h }, { z16.h - z19.h }
// CHECK-ENCODING: [0x19,0x02,0xf1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f10219 <unknown>

umlsll  za.d[w8, 4:7], {z16.h - z19.h}, {z16.h - z19.h}  // 11000001-11110001-00000010-00011001
// CHECK-INST: umlsll  za.d[w8, 4:7, vgx4], { z16.h - z19.h }, { z16.h - z19.h }
// CHECK-ENCODING: [0x19,0x02,0xf1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f10219 <unknown>

umlsll  za.d[w8, 4:7, vgx4], {z0.h - z3.h}, {z28.h - z31.h}  // 11000001-11111101-00000000-00011001
// CHECK-INST: umlsll  za.d[w8, 4:7, vgx4], { z0.h - z3.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x19,0x00,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd0019 <unknown>

umlsll  za.d[w8, 4:7], {z0.h - z3.h}, {z28.h - z31.h}  // 11000001-11111101-00000000-00011001
// CHECK-INST: umlsll  za.d[w8, 4:7, vgx4], { z0.h - z3.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x19,0x00,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd0019 <unknown>

umlsll  za.d[w10, 0:3, vgx4], {z16.h - z19.h}, {z20.h - z23.h}  // 11000001-11110101-01000010-00011000
// CHECK-INST: umlsll  za.d[w10, 0:3, vgx4], { z16.h - z19.h }, { z20.h - z23.h }
// CHECK-ENCODING: [0x18,0x42,0xf5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f54218 <unknown>

umlsll  za.d[w10, 0:3], {z16.h - z19.h}, {z20.h - z23.h}  // 11000001-11110101-01000010-00011000
// CHECK-INST: umlsll  za.d[w10, 0:3, vgx4], { z16.h - z19.h }, { z20.h - z23.h }
// CHECK-ENCODING: [0x18,0x42,0xf5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f54218 <unknown>

umlsll  za.d[w8, 0:3, vgx4], {z12.h - z15.h}, {z0.h - z3.h}  // 11000001-11100001-00000001-10011000
// CHECK-INST: umlsll  za.d[w8, 0:3, vgx4], { z12.h - z15.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x98,0x01,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e10198 <unknown>

umlsll  za.d[w8, 0:3], {z12.h - z15.h}, {z0.h - z3.h}  // 11000001-11100001-00000001-10011000
// CHECK-INST: umlsll  za.d[w8, 0:3, vgx4], { z12.h - z15.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x98,0x01,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e10198 <unknown>

umlsll  za.d[w10, 4:7, vgx4], {z0.h - z3.h}, {z24.h - z27.h}  // 11000001-11111001-01000000-00011001
// CHECK-INST: umlsll  za.d[w10, 4:7, vgx4], { z0.h - z3.h }, { z24.h - z27.h }
// CHECK-ENCODING: [0x19,0x40,0xf9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f94019 <unknown>

umlsll  za.d[w10, 4:7], {z0.h - z3.h}, {z24.h - z27.h}  // 11000001-11111001-01000000-00011001
// CHECK-INST: umlsll  za.d[w10, 4:7, vgx4], { z0.h - z3.h }, { z24.h - z27.h }
// CHECK-ENCODING: [0x19,0x40,0xf9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f94019 <unknown>

umlsll  za.d[w8, 4:7, vgx4], {z20.h - z23.h}, {z28.h - z31.h}  // 11000001-11111101-00000010-10011001
// CHECK-INST: umlsll  za.d[w8, 4:7, vgx4], { z20.h - z23.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x99,0x02,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd0299 <unknown>

umlsll  za.d[w8, 4:7], {z20.h - z23.h}, {z28.h - z31.h}  // 11000001-11111101-00000010-10011001
// CHECK-INST: umlsll  za.d[w8, 4:7, vgx4], { z20.h - z23.h }, { z28.h - z31.h }
// CHECK-ENCODING: [0x99,0x02,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd0299 <unknown>

umlsll  za.d[w11, 0:3, vgx4], {z8.h - z11.h}, {z0.h - z3.h}  // 11000001-11100001-01100001-00011000
// CHECK-INST: umlsll  za.d[w11, 0:3, vgx4], { z8.h - z11.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x18,0x61,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e16118 <unknown>

umlsll  za.d[w11, 0:3], {z8.h - z11.h}, {z0.h - z3.h}  // 11000001-11100001-01100001-00011000
// CHECK-INST: umlsll  za.d[w11, 0:3, vgx4], { z8.h - z11.h }, { z0.h - z3.h }
// CHECK-ENCODING: [0x18,0x61,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e16118 <unknown>

umlsll  za.d[w9, 4:7, vgx4], {z12.h - z15.h}, {z8.h - z11.h}  // 11000001-11101001-00100001-10011001
// CHECK-INST: umlsll  za.d[w9, 4:7, vgx4], { z12.h - z15.h }, { z8.h - z11.h }
// CHECK-ENCODING: [0x99,0x21,0xe9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e92199 <unknown>

umlsll  za.d[w9, 4:7], {z12.h - z15.h}, {z8.h - z11.h}  // 11000001-11101001-00100001-10011001
// CHECK-INST: umlsll  za.d[w9, 4:7, vgx4], { z12.h - z15.h }, { z8.h - z11.h }
// CHECK-ENCODING: [0x99,0x21,0xe9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e92199 <unknown>

