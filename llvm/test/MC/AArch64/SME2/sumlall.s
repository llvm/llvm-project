// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump --no-print-imm-hex -d --mattr=+sme2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump --no-print-imm-hex -d --mattr=-sme2 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST


sumlall za.s[w8, 0:3], z0.b, z0.b[0]  // 11000001-00000000-00000000-00010100
// CHECK-INST: sumlall za.s[w8, 0:3], z0.b, z0.b[0]
// CHECK-ENCODING: [0x14,0x00,0x00,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1000014 <unknown>

sumlall za.s[w10, 4:7], z10.b, z5.b[5]  // 11000001-00000101-01010101-01010101
// CHECK-INST: sumlall za.s[w10, 4:7], z10.b, z5.b[5]
// CHECK-ENCODING: [0x55,0x55,0x05,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1055555 <unknown>

sumlall za.s[w11, 12:15], z13.b, z8.b[11]  // 11000001-00001000-11101101-10110111
// CHECK-INST: sumlall za.s[w11, 12:15], z13.b, z8.b[11]
// CHECK-ENCODING: [0xb7,0xed,0x08,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c108edb7 <unknown>

sumlall za.s[w11, 12:15], z31.b, z15.b[15]  // 11000001-00001111-11111111-11110111
// CHECK-INST: sumlall za.s[w11, 12:15], z31.b, z15.b[15]
// CHECK-ENCODING: [0xf7,0xff,0x0f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c10ffff7 <unknown>

sumlall za.s[w8, 4:7], z17.b, z0.b[3]  // 11000001-00000000-00001110-00110101
// CHECK-INST: sumlall za.s[w8, 4:7], z17.b, z0.b[3]
// CHECK-ENCODING: [0x35,0x0e,0x00,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1000e35 <unknown>

sumlall za.s[w8, 4:7], z1.b, z14.b[9]  // 11000001-00001110-10000100-00110101
// CHECK-INST: sumlall za.s[w8, 4:7], z1.b, z14.b[9]
// CHECK-ENCODING: [0x35,0x84,0x0e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c10e8435 <unknown>

sumlall za.s[w10, 0:3], z19.b, z4.b[5]  // 11000001-00000100-01010110-01110100
// CHECK-INST: sumlall za.s[w10, 0:3], z19.b, z4.b[5]
// CHECK-ENCODING: [0x74,0x56,0x04,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1045674 <unknown>

sumlall za.s[w8, 0:3], z12.b, z2.b[6]  // 11000001-00000010-00011001-10010100
// CHECK-INST: sumlall za.s[w8, 0:3], z12.b, z2.b[6]
// CHECK-ENCODING: [0x94,0x19,0x02,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1021994 <unknown>

sumlall za.s[w10, 4:7], z1.b, z10.b[10]  // 11000001-00001010-11001000-00110101
// CHECK-INST: sumlall za.s[w10, 4:7], z1.b, z10.b[10]
// CHECK-ENCODING: [0x35,0xc8,0x0a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c10ac835 <unknown>

sumlall za.s[w8, 4:7], z22.b, z14.b[2]  // 11000001-00001110-00001010-11010101
// CHECK-INST: sumlall za.s[w8, 4:7], z22.b, z14.b[2]
// CHECK-ENCODING: [0xd5,0x0a,0x0e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c10e0ad5 <unknown>

sumlall za.s[w11, 8:11], z9.b, z1.b[13]  // 11000001-00000001-11110101-00110110
// CHECK-INST: sumlall za.s[w11, 8:11], z9.b, z1.b[13]
// CHECK-ENCODING: [0x36,0xf5,0x01,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c101f536 <unknown>

sumlall za.s[w9, 12:15], z12.b, z11.b[10]  // 11000001-00001011-10101001-10010111
// CHECK-INST: sumlall za.s[w9, 12:15], z12.b, z11.b[10]
// CHECK-ENCODING: [0x97,0xa9,0x0b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c10ba997 <unknown>


sumlall za.s[w8, 0:3, vgx2], {z0.b, z1.b}, z0.b  // 11000001-00100000-00000000-00010100
// CHECK-INST: sumlall za.s[w8, 0:3, vgx2], { z0.b, z1.b }, z0.b
// CHECK-ENCODING: [0x14,0x00,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1200014 <unknown>

sumlall za.s[w8, 0:3], {z0.b, z1.b}, z0.b  // 11000001-00100000-00000000-00010100
// CHECK-INST: sumlall za.s[w8, 0:3, vgx2], { z0.b, z1.b }, z0.b
// CHECK-ENCODING: [0x14,0x00,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1200014 <unknown>

sumlall za.s[w10, 4:7, vgx2], {z10.b, z11.b}, z5.b  // 11000001-00100101-01000001-01010101
// CHECK-INST: sumlall za.s[w10, 4:7, vgx2], { z10.b, z11.b }, z5.b
// CHECK-ENCODING: [0x55,0x41,0x25,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1254155 <unknown>

sumlall za.s[w10, 4:7], {z10.b, z11.b}, z5.b  // 11000001-00100101-01000001-01010101
// CHECK-INST: sumlall za.s[w10, 4:7, vgx2], { z10.b, z11.b }, z5.b
// CHECK-ENCODING: [0x55,0x41,0x25,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1254155 <unknown>

sumlall za.s[w11, 4:7, vgx2], {z13.b - z14.b}, z8.b  // 11000001-00101000-01100001-10110101
// CHECK-INST: sumlall za.s[w11, 4:7, vgx2], { z13.b, z14.b }, z8.b
// CHECK-ENCODING: [0xb5,0x61,0x28,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12861b5 <unknown>

sumlall za.s[w11, 4:7], {z13.b - z14.b}, z8.b  // 11000001-00101000-01100001-10110101
// CHECK-INST: sumlall za.s[w11, 4:7, vgx2], { z13.b, z14.b }, z8.b
// CHECK-ENCODING: [0xb5,0x61,0x28,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12861b5 <unknown>

sumlall za.s[w11, 4:7, vgx2], {z31.b, z0.b}, z15.b  // 11000001-00101111-01100011-11110101
// CHECK-INST: sumlall za.s[w11, 4:7, vgx2], { z31.b, z0.b }, z15.b
// CHECK-ENCODING: [0xf5,0x63,0x2f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12f63f5 <unknown>

sumlall za.s[w11, 4:7], {z31.b, z0.b}, z15.b  // 11000001-00101111-01100011-11110101
// CHECK-INST: sumlall za.s[w11, 4:7, vgx2], { z31.b, z0.b }, z15.b
// CHECK-ENCODING: [0xf5,0x63,0x2f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12f63f5 <unknown>

sumlall za.s[w8, 4:7, vgx2], {z17.b, z18.b}, z0.b  // 11000001-00100000-00000010-00110101
// CHECK-INST: sumlall za.s[w8, 4:7, vgx2], { z17.b, z18.b }, z0.b
// CHECK-ENCODING: [0x35,0x02,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1200235 <unknown>

sumlall za.s[w8, 4:7], {z17.b, z18.b}, z0.b  // 11000001-00100000-00000010-00110101
// CHECK-INST: sumlall za.s[w8, 4:7, vgx2], { z17.b, z18.b }, z0.b
// CHECK-ENCODING: [0x35,0x02,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1200235 <unknown>

sumlall za.s[w8, 4:7, vgx2], {z1.b, z2.b}, z14.b  // 11000001-00101110-00000000-00110101
// CHECK-INST: sumlall za.s[w8, 4:7, vgx2], { z1.b, z2.b }, z14.b
// CHECK-ENCODING: [0x35,0x00,0x2e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12e0035 <unknown>

sumlall za.s[w8, 4:7], {z1.b, z2.b}, z14.b  // 11000001-00101110-00000000-00110101
// CHECK-INST: sumlall za.s[w8, 4:7, vgx2], { z1.b, z2.b }, z14.b
// CHECK-ENCODING: [0x35,0x00,0x2e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12e0035 <unknown>

sumlall za.s[w10, 0:3, vgx2], {z19.b, z20.b}, z4.b  // 11000001-00100100-01000010-01110100
// CHECK-INST: sumlall za.s[w10, 0:3, vgx2], { z19.b, z20.b }, z4.b
// CHECK-ENCODING: [0x74,0x42,0x24,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1244274 <unknown>

sumlall za.s[w10, 0:3], {z19.b, z20.b}, z4.b  // 11000001-00100100-01000010-01110100
// CHECK-INST: sumlall za.s[w10, 0:3, vgx2], { z19.b, z20.b }, z4.b
// CHECK-ENCODING: [0x74,0x42,0x24,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1244274 <unknown>

sumlall za.s[w8, 0:3, vgx2], {z12.b, z13.b}, z2.b  // 11000001-00100010-00000001-10010100
// CHECK-INST: sumlall za.s[w8, 0:3, vgx2], { z12.b, z13.b }, z2.b
// CHECK-ENCODING: [0x94,0x01,0x22,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1220194 <unknown>

sumlall za.s[w8, 0:3], {z12.b, z13.b}, z2.b  // 11000001-00100010-00000001-10010100
// CHECK-INST: sumlall za.s[w8, 0:3, vgx2], { z12.b, z13.b }, z2.b
// CHECK-ENCODING: [0x94,0x01,0x22,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1220194 <unknown>

sumlall za.s[w10, 4:7, vgx2], {z1.b, z2.b}, z10.b  // 11000001-00101010-01000000-00110101
// CHECK-INST: sumlall za.s[w10, 4:7, vgx2], { z1.b, z2.b }, z10.b
// CHECK-ENCODING: [0x35,0x40,0x2a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12a4035 <unknown>

sumlall za.s[w10, 4:7], {z1.b, z2.b}, z10.b  // 11000001-00101010-01000000-00110101
// CHECK-INST: sumlall za.s[w10, 4:7, vgx2], { z1.b, z2.b }, z10.b
// CHECK-ENCODING: [0x35,0x40,0x2a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12a4035 <unknown>

sumlall za.s[w8, 4:7, vgx2], {z22.b, z23.b}, z14.b  // 11000001-00101110-00000010-11010101
// CHECK-INST: sumlall za.s[w8, 4:7, vgx2], { z22.b, z23.b }, z14.b
// CHECK-ENCODING: [0xd5,0x02,0x2e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12e02d5 <unknown>

sumlall za.s[w8, 4:7], {z22.b, z23.b}, z14.b  // 11000001-00101110-00000010-11010101
// CHECK-INST: sumlall za.s[w8, 4:7, vgx2], { z22.b, z23.b }, z14.b
// CHECK-ENCODING: [0xd5,0x02,0x2e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12e02d5 <unknown>

sumlall za.s[w11, 0:3, vgx2], {z9.b, z10.b}, z1.b  // 11000001-00100001-01100001-00110100
// CHECK-INST: sumlall za.s[w11, 0:3, vgx2], { z9.b, z10.b }, z1.b
// CHECK-ENCODING: [0x34,0x61,0x21,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1216134 <unknown>

sumlall za.s[w11, 0:3], {z9.b, z10.b}, z1.b  // 11000001-00100001-01100001-00110100
// CHECK-INST: sumlall za.s[w11, 0:3, vgx2], { z9.b, z10.b }, z1.b
// CHECK-ENCODING: [0x34,0x61,0x21,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1216134 <unknown>

sumlall za.s[w9, 4:7, vgx2], {z12.b, z13.b}, z11.b  // 11000001-00101011-00100001-10010101
// CHECK-INST: sumlall za.s[w9, 4:7, vgx2], { z12.b, z13.b }, z11.b
// CHECK-ENCODING: [0x95,0x21,0x2b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12b2195 <unknown>

sumlall za.s[w9, 4:7], {z12.b, z13.b}, z11.b  // 11000001-00101011-00100001-10010101
// CHECK-INST: sumlall za.s[w9, 4:7, vgx2], { z12.b, z13.b }, z11.b
// CHECK-ENCODING: [0x95,0x21,0x2b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12b2195 <unknown>


sumlall za.s[w8, 0:3, vgx2], {z0.b, z1.b}, z0.b[0]  // 11000001-00010000-00000000-00110000
// CHECK-INST: sumlall za.s[w8, 0:3, vgx2], { z0.b, z1.b }, z0.b[0]
// CHECK-ENCODING: [0x30,0x00,0x10,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1100030 <unknown>

sumlall za.s[w8, 0:3], {z0.b, z1.b}, z0.b[0]  // 11000001-00010000-00000000-00110000
// CHECK-INST: sumlall za.s[w8, 0:3, vgx2], { z0.b, z1.b }, z0.b[0]
// CHECK-ENCODING: [0x30,0x00,0x10,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1100030 <unknown>

sumlall za.s[w10, 4:7, vgx2], {z10.b, z11.b}, z5.b[6]  // 11000001-00010101-01000101-01110101
// CHECK-INST: sumlall za.s[w10, 4:7, vgx2], { z10.b, z11.b }, z5.b[6]
// CHECK-ENCODING: [0x75,0x45,0x15,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1154575 <unknown>

sumlall za.s[w10, 4:7], {z10.b, z11.b}, z5.b[6]  // 11000001-00010101-01000101-01110101
// CHECK-INST: sumlall za.s[w10, 4:7, vgx2], { z10.b, z11.b }, z5.b[6]
// CHECK-ENCODING: [0x75,0x45,0x15,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1154575 <unknown>

sumlall za.s[w11, 4:7, vgx2], {z12.b, z13.b}, z8.b[15]  // 11000001-00011000-01101101-10110111
// CHECK-INST: sumlall za.s[w11, 4:7, vgx2], { z12.b, z13.b }, z8.b[15]
// CHECK-ENCODING: [0xb7,0x6d,0x18,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1186db7 <unknown>

sumlall za.s[w11, 4:7], {z12.b, z13.b}, z8.b[15]  // 11000001-00011000-01101101-10110111
// CHECK-INST: sumlall za.s[w11, 4:7, vgx2], { z12.b, z13.b }, z8.b[15]
// CHECK-ENCODING: [0xb7,0x6d,0x18,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1186db7 <unknown>

sumlall za.s[w11, 4:7, vgx2], {z30.b, z31.b}, z15.b[15]  // 11000001-00011111-01101111-11110111
// CHECK-INST: sumlall za.s[w11, 4:7, vgx2], { z30.b, z31.b }, z15.b[15]
// CHECK-ENCODING: [0xf7,0x6f,0x1f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11f6ff7 <unknown>

sumlall za.s[w11, 4:7], {z30.b, z31.b}, z15.b[15]  // 11000001-00011111-01101111-11110111
// CHECK-INST: sumlall za.s[w11, 4:7, vgx2], { z30.b, z31.b }, z15.b[15]
// CHECK-ENCODING: [0xf7,0x6f,0x1f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11f6ff7 <unknown>

sumlall za.s[w8, 4:7, vgx2], {z16.b, z17.b}, z0.b[14]  // 11000001-00010000-00001110-00110101
// CHECK-INST: sumlall za.s[w8, 4:7, vgx2], { z16.b, z17.b }, z0.b[14]
// CHECK-ENCODING: [0x35,0x0e,0x10,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1100e35 <unknown>

sumlall za.s[w8, 4:7], {z16.b, z17.b}, z0.b[14]  // 11000001-00010000-00001110-00110101
// CHECK-INST: sumlall za.s[w8, 4:7, vgx2], { z16.b, z17.b }, z0.b[14]
// CHECK-ENCODING: [0x35,0x0e,0x10,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1100e35 <unknown>

sumlall za.s[w8, 4:7, vgx2], {z0.b, z1.b}, z14.b[4]  // 11000001-00011110-00000100-00110001
// CHECK-INST: sumlall za.s[w8, 4:7, vgx2], { z0.b, z1.b }, z14.b[4]
// CHECK-ENCODING: [0x31,0x04,0x1e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11e0431 <unknown>

sumlall za.s[w8, 4:7], {z0.b, z1.b}, z14.b[4]  // 11000001-00011110-00000100-00110001
// CHECK-INST: sumlall za.s[w8, 4:7, vgx2], { z0.b, z1.b }, z14.b[4]
// CHECK-ENCODING: [0x31,0x04,0x1e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11e0431 <unknown>

sumlall za.s[w10, 0:3, vgx2], {z18.b, z19.b}, z4.b[4]  // 11000001-00010100-01000110-01110000
// CHECK-INST: sumlall za.s[w10, 0:3, vgx2], { z18.b, z19.b }, z4.b[4]
// CHECK-ENCODING: [0x70,0x46,0x14,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1144670 <unknown>

sumlall za.s[w10, 0:3], {z18.b, z19.b}, z4.b[4]  // 11000001-00010100-01000110-01110000
// CHECK-INST: sumlall za.s[w10, 0:3, vgx2], { z18.b, z19.b }, z4.b[4]
// CHECK-ENCODING: [0x70,0x46,0x14,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1144670 <unknown>

sumlall za.s[w8, 0:3, vgx2], {z12.b, z13.b}, z2.b[8]  // 11000001-00010010-00001001-10110000
// CHECK-INST: sumlall za.s[w8, 0:3, vgx2], { z12.b, z13.b }, z2.b[8]
// CHECK-ENCODING: [0xb0,0x09,0x12,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11209b0 <unknown>

sumlall za.s[w8, 0:3], {z12.b, z13.b}, z2.b[8]  // 11000001-00010010-00001001-10110000
// CHECK-INST: sumlall za.s[w8, 0:3, vgx2], { z12.b, z13.b }, z2.b[8]
// CHECK-ENCODING: [0xb0,0x09,0x12,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11209b0 <unknown>

sumlall za.s[w10, 4:7, vgx2], {z0.b, z1.b}, z10.b[8]  // 11000001-00011010-01001000-00110001
// CHECK-INST: sumlall za.s[w10, 4:7, vgx2], { z0.b, z1.b }, z10.b[8]
// CHECK-ENCODING: [0x31,0x48,0x1a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11a4831 <unknown>

sumlall za.s[w10, 4:7], {z0.b, z1.b}, z10.b[8]  // 11000001-00011010-01001000-00110001
// CHECK-INST: sumlall za.s[w10, 4:7, vgx2], { z0.b, z1.b }, z10.b[8]
// CHECK-ENCODING: [0x31,0x48,0x1a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11a4831 <unknown>

sumlall za.s[w8, 4:7, vgx2], {z22.b, z23.b}, z14.b[10]  // 11000001-00011110-00001010-11110101
// CHECK-INST: sumlall za.s[w8, 4:7, vgx2], { z22.b, z23.b }, z14.b[10]
// CHECK-ENCODING: [0xf5,0x0a,0x1e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11e0af5 <unknown>

sumlall za.s[w8, 4:7], {z22.b, z23.b}, z14.b[10]  // 11000001-00011110-00001010-11110101
// CHECK-INST: sumlall za.s[w8, 4:7, vgx2], { z22.b, z23.b }, z14.b[10]
// CHECK-ENCODING: [0xf5,0x0a,0x1e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11e0af5 <unknown>

sumlall za.s[w11, 0:3, vgx2], {z8.b, z9.b}, z1.b[5]  // 11000001-00010001-01100101-00110010
// CHECK-INST: sumlall za.s[w11, 0:3, vgx2], { z8.b, z9.b }, z1.b[5]
// CHECK-ENCODING: [0x32,0x65,0x11,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1116532 <unknown>

sumlall za.s[w11, 0:3], {z8.b, z9.b}, z1.b[5]  // 11000001-00010001-01100101-00110010
// CHECK-INST: sumlall za.s[w11, 0:3, vgx2], { z8.b, z9.b }, z1.b[5]
// CHECK-ENCODING: [0x32,0x65,0x11,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1116532 <unknown>

sumlall za.s[w9, 4:7, vgx2], {z12.b, z13.b}, z11.b[11]  // 11000001-00011011-00101001-10110111
// CHECK-INST: sumlall za.s[w9, 4:7, vgx2], { z12.b, z13.b }, z11.b[11]
// CHECK-ENCODING: [0xb7,0x29,0x1b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11b29b7 <unknown>

sumlall za.s[w9, 4:7], {z12.b, z13.b}, z11.b[11]  // 11000001-00011011-00101001-10110111
// CHECK-INST: sumlall za.s[w9, 4:7, vgx2], { z12.b, z13.b }, z11.b[11]
// CHECK-ENCODING: [0xb7,0x29,0x1b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11b29b7 <unknown>


sumlall za.s[w8, 0:3, vgx4], {z0.b - z3.b}, z0.b  // 11000001-00110000-00000000-00010100
// CHECK-INST: sumlall za.s[w8, 0:3, vgx4], { z0.b - z3.b }, z0.b
// CHECK-ENCODING: [0x14,0x00,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1300014 <unknown>

sumlall za.s[w8, 0:3], {z0.b - z3.b}, z0.b  // 11000001-00110000-00000000-00010100
// CHECK-INST: sumlall za.s[w8, 0:3, vgx4], { z0.b - z3.b }, z0.b
// CHECK-ENCODING: [0x14,0x00,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1300014 <unknown>

sumlall za.s[w10, 4:7, vgx4], {z10.b - z13.b}, z5.b  // 11000001-00110101-01000001-01010101
// CHECK-INST: sumlall za.s[w10, 4:7, vgx4], { z10.b - z13.b }, z5.b
// CHECK-ENCODING: [0x55,0x41,0x35,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1354155 <unknown>

sumlall za.s[w10, 4:7], {z10.b - z13.b}, z5.b  // 11000001-00110101-01000001-01010101
// CHECK-INST: sumlall za.s[w10, 4:7, vgx4], { z10.b - z13.b }, z5.b
// CHECK-ENCODING: [0x55,0x41,0x35,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1354155 <unknown>

sumlall za.s[w11, 4:7, vgx4], {z13.b - z16.b}, z8.b  // 11000001-00111000-01100001-10110101
// CHECK-INST: sumlall za.s[w11, 4:7, vgx4], { z13.b - z16.b }, z8.b
// CHECK-ENCODING: [0xb5,0x61,0x38,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13861b5 <unknown>

sumlall za.s[w11, 4:7], {z13.b - z16.b}, z8.b  // 11000001-00111000-01100001-10110101
// CHECK-INST: sumlall za.s[w11, 4:7, vgx4], { z13.b - z16.b }, z8.b
// CHECK-ENCODING: [0xb5,0x61,0x38,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13861b5 <unknown>

sumlall za.s[w11, 4:7, vgx4], {z31.b - z2.b}, z15.b  // 11000001-00111111-01100011-11110101
// CHECK-INST: sumlall za.s[w11, 4:7, vgx4], {  z31.b, z0.b, z1.b, z2.b }, z15.b
// CHECK-ENCODING: [0xf5,0x63,0x3f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13f63f5 <unknown>

sumlall za.s[w11, 4:7], {z31.b - z2.b}, z15.b  // 11000001-00111111-01100011-11110101
// CHECK-INST: sumlall za.s[w11, 4:7, vgx4], {  z31.b, z0.b, z1.b, z2.b }, z15.b
// CHECK-ENCODING: [0xf5,0x63,0x3f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13f63f5 <unknown>

sumlall za.s[w8, 4:7, vgx4], {z17.b - z20.b}, z0.b  // 11000001-00110000-00000010-00110101
// CHECK-INST: sumlall za.s[w8, 4:7, vgx4], { z17.b - z20.b }, z0.b
// CHECK-ENCODING: [0x35,0x02,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1300235 <unknown>

sumlall za.s[w8, 4:7], {z17.b - z20.b}, z0.b  // 11000001-00110000-00000010-00110101
// CHECK-INST: sumlall za.s[w8, 4:7, vgx4], { z17.b - z20.b }, z0.b
// CHECK-ENCODING: [0x35,0x02,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1300235 <unknown>

sumlall za.s[w8, 4:7, vgx4], {z1.b - z4.b}, z14.b  // 11000001-00111110-00000000-00110101
// CHECK-INST: sumlall za.s[w8, 4:7, vgx4], { z1.b - z4.b }, z14.b
// CHECK-ENCODING: [0x35,0x00,0x3e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13e0035 <unknown>

sumlall za.s[w8, 4:7], {z1.b - z4.b}, z14.b  // 11000001-00111110-00000000-00110101
// CHECK-INST: sumlall za.s[w8, 4:7, vgx4], { z1.b - z4.b }, z14.b
// CHECK-ENCODING: [0x35,0x00,0x3e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13e0035 <unknown>

sumlall za.s[w10, 0:3, vgx4], {z19.b - z22.b}, z4.b  // 11000001-00110100-01000010-01110100
// CHECK-INST: sumlall za.s[w10, 0:3, vgx4], { z19.b - z22.b }, z4.b
// CHECK-ENCODING: [0x74,0x42,0x34,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1344274 <unknown>

sumlall za.s[w10, 0:3], {z19.b - z22.b}, z4.b  // 11000001-00110100-01000010-01110100
// CHECK-INST: sumlall za.s[w10, 0:3, vgx4], { z19.b - z22.b }, z4.b
// CHECK-ENCODING: [0x74,0x42,0x34,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1344274 <unknown>

sumlall za.s[w8, 0:3, vgx4], {z12.b - z15.b}, z2.b  // 11000001-00110010-00000001-10010100
// CHECK-INST: sumlall za.s[w8, 0:3, vgx4], { z12.b - z15.b }, z2.b
// CHECK-ENCODING: [0x94,0x01,0x32,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1320194 <unknown>

sumlall za.s[w8, 0:3], {z12.b - z15.b}, z2.b  // 11000001-00110010-00000001-10010100
// CHECK-INST: sumlall za.s[w8, 0:3, vgx4], { z12.b - z15.b }, z2.b
// CHECK-ENCODING: [0x94,0x01,0x32,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1320194 <unknown>

sumlall za.s[w10, 4:7, vgx4], {z1.b - z4.b}, z10.b  // 11000001-00111010-01000000-00110101
// CHECK-INST: sumlall za.s[w10, 4:7, vgx4], { z1.b - z4.b }, z10.b
// CHECK-ENCODING: [0x35,0x40,0x3a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13a4035 <unknown>

sumlall za.s[w10, 4:7], {z1.b - z4.b}, z10.b  // 11000001-00111010-01000000-00110101
// CHECK-INST: sumlall za.s[w10, 4:7, vgx4], { z1.b - z4.b }, z10.b
// CHECK-ENCODING: [0x35,0x40,0x3a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13a4035 <unknown>

sumlall za.s[w8, 4:7, vgx4], {z22.b - z25.b}, z14.b  // 11000001-00111110-00000010-11010101
// CHECK-INST: sumlall za.s[w8, 4:7, vgx4], { z22.b - z25.b }, z14.b
// CHECK-ENCODING: [0xd5,0x02,0x3e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13e02d5 <unknown>

sumlall za.s[w8, 4:7], {z22.b - z25.b}, z14.b  // 11000001-00111110-00000010-11010101
// CHECK-INST: sumlall za.s[w8, 4:7, vgx4], { z22.b - z25.b }, z14.b
// CHECK-ENCODING: [0xd5,0x02,0x3e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13e02d5 <unknown>

sumlall za.s[w11, 0:3, vgx4], {z9.b - z12.b}, z1.b  // 11000001-00110001-01100001-00110100
// CHECK-INST: sumlall za.s[w11, 0:3, vgx4], { z9.b - z12.b }, z1.b
// CHECK-ENCODING: [0x34,0x61,0x31,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1316134 <unknown>

sumlall za.s[w11, 0:3], {z9.b - z12.b}, z1.b  // 11000001-00110001-01100001-00110100
// CHECK-INST: sumlall za.s[w11, 0:3, vgx4], { z9.b - z12.b }, z1.b
// CHECK-ENCODING: [0x34,0x61,0x31,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1316134 <unknown>

sumlall za.s[w9, 4:7, vgx4], {z12.b - z15.b}, z11.b  // 11000001-00111011-00100001-10010101
// CHECK-INST: sumlall za.s[w9, 4:7, vgx4], { z12.b - z15.b }, z11.b
// CHECK-ENCODING: [0x95,0x21,0x3b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13b2195 <unknown>

sumlall za.s[w9, 4:7], {z12.b - z15.b}, z11.b  // 11000001-00111011-00100001-10010101
// CHECK-INST: sumlall za.s[w9, 4:7, vgx4], { z12.b - z15.b }, z11.b
// CHECK-ENCODING: [0x95,0x21,0x3b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13b2195 <unknown>


sumlall za.s[w8, 0:3, vgx4], {z0.b - z3.b}, z0.b[0]  // 11000001-00010000-10000000-00110000
// CHECK-INST: sumlall za.s[w8, 0:3, vgx4], { z0.b - z3.b }, z0.b[0]
// CHECK-ENCODING: [0x30,0x80,0x10,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1108030 <unknown>

sumlall za.s[w8, 0:3], {z0.b - z3.b}, z0.b[0]  // 11000001-00010000-10000000-00110000
// CHECK-INST: sumlall za.s[w8, 0:3, vgx4], { z0.b - z3.b }, z0.b[0]
// CHECK-ENCODING: [0x30,0x80,0x10,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1108030 <unknown>

sumlall za.s[w10, 4:7, vgx4], {z8.b - z11.b}, z5.b[6]  // 11000001-00010101-11000101-00110101
// CHECK-INST: sumlall za.s[w10, 4:7, vgx4], { z8.b - z11.b }, z5.b[6]
// CHECK-ENCODING: [0x35,0xc5,0x15,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c115c535 <unknown>

sumlall za.s[w10, 4:7], {z8.b - z11.b}, z5.b[6]  // 11000001-00010101-11000101-00110101
// CHECK-INST: sumlall za.s[w10, 4:7, vgx4], { z8.b - z11.b }, z5.b[6]
// CHECK-ENCODING: [0x35,0xc5,0x15,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c115c535 <unknown>

sumlall za.s[w11, 4:7, vgx4], {z12.b - z15.b}, z8.b[15]  // 11000001-00011000-11101101-10110111
// CHECK-INST: sumlall za.s[w11, 4:7, vgx4], { z12.b - z15.b }, z8.b[15]
// CHECK-ENCODING: [0xb7,0xed,0x18,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c118edb7 <unknown>

sumlall za.s[w11, 4:7], {z12.b - z15.b}, z8.b[15]  // 11000001-00011000-11101101-10110111
// CHECK-INST: sumlall za.s[w11, 4:7, vgx4], { z12.b - z15.b }, z8.b[15]
// CHECK-ENCODING: [0xb7,0xed,0x18,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c118edb7 <unknown>

sumlall za.s[w11, 4:7, vgx4], {z28.b - z31.b}, z15.b[15]  // 11000001-00011111-11101111-10110111
// CHECK-INST: sumlall za.s[w11, 4:7, vgx4], { z28.b - z31.b }, z15.b[15]
// CHECK-ENCODING: [0xb7,0xef,0x1f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11fefb7 <unknown>

sumlall za.s[w11, 4:7], {z28.b - z31.b}, z15.b[15]  // 11000001-00011111-11101111-10110111
// CHECK-INST: sumlall za.s[w11, 4:7, vgx4], { z28.b - z31.b }, z15.b[15]
// CHECK-ENCODING: [0xb7,0xef,0x1f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11fefb7 <unknown>

sumlall za.s[w8, 4:7, vgx4], {z16.b - z19.b}, z0.b[14]  // 11000001-00010000-10001110-00110101
// CHECK-INST: sumlall za.s[w8, 4:7, vgx4], { z16.b - z19.b }, z0.b[14]
// CHECK-ENCODING: [0x35,0x8e,0x10,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1108e35 <unknown>

sumlall za.s[w8, 4:7], {z16.b - z19.b}, z0.b[14]  // 11000001-00010000-10001110-00110101
// CHECK-INST: sumlall za.s[w8, 4:7, vgx4], { z16.b - z19.b }, z0.b[14]
// CHECK-ENCODING: [0x35,0x8e,0x10,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1108e35 <unknown>

sumlall za.s[w8, 4:7, vgx4], {z0.b - z3.b}, z14.b[4]  // 11000001-00011110-10000100-00110001
// CHECK-INST: sumlall za.s[w8, 4:7, vgx4], { z0.b - z3.b }, z14.b[4]
// CHECK-ENCODING: [0x31,0x84,0x1e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11e8431 <unknown>

sumlall za.s[w8, 4:7], {z0.b - z3.b}, z14.b[4]  // 11000001-00011110-10000100-00110001
// CHECK-INST: sumlall za.s[w8, 4:7, vgx4], { z0.b - z3.b }, z14.b[4]
// CHECK-ENCODING: [0x31,0x84,0x1e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11e8431 <unknown>

sumlall za.s[w10, 0:3, vgx4], {z16.b - z19.b}, z4.b[4]  // 11000001-00010100-11000110-00110000
// CHECK-INST: sumlall za.s[w10, 0:3, vgx4], { z16.b - z19.b }, z4.b[4]
// CHECK-ENCODING: [0x30,0xc6,0x14,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c114c630 <unknown>

sumlall za.s[w10, 0:3], {z16.b - z19.b}, z4.b[4]  // 11000001-00010100-11000110-00110000
// CHECK-INST: sumlall za.s[w10, 0:3, vgx4], { z16.b - z19.b }, z4.b[4]
// CHECK-ENCODING: [0x30,0xc6,0x14,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c114c630 <unknown>

sumlall za.s[w8, 0:3, vgx4], {z12.b - z15.b}, z2.b[8]  // 11000001-00010010-10001001-10110000
// CHECK-INST: sumlall za.s[w8, 0:3, vgx4], { z12.b - z15.b }, z2.b[8]
// CHECK-ENCODING: [0xb0,0x89,0x12,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11289b0 <unknown>

sumlall za.s[w8, 0:3], {z12.b - z15.b}, z2.b[8]  // 11000001-00010010-10001001-10110000
// CHECK-INST: sumlall za.s[w8, 0:3, vgx4], { z12.b - z15.b }, z2.b[8]
// CHECK-ENCODING: [0xb0,0x89,0x12,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11289b0 <unknown>

sumlall za.s[w10, 4:7, vgx4], {z0.b - z3.b}, z10.b[8]  // 11000001-00011010-11001000-00110001
// CHECK-INST: sumlall za.s[w10, 4:7, vgx4], { z0.b - z3.b }, z10.b[8]
// CHECK-ENCODING: [0x31,0xc8,0x1a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11ac831 <unknown>

sumlall za.s[w10, 4:7], {z0.b - z3.b}, z10.b[8]  // 11000001-00011010-11001000-00110001
// CHECK-INST: sumlall za.s[w10, 4:7, vgx4], { z0.b - z3.b }, z10.b[8]
// CHECK-ENCODING: [0x31,0xc8,0x1a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11ac831 <unknown>

sumlall za.s[w8, 4:7, vgx4], {z20.b - z23.b}, z14.b[10]  // 11000001-00011110-10001010-10110101
// CHECK-INST: sumlall za.s[w8, 4:7, vgx4], { z20.b - z23.b }, z14.b[10]
// CHECK-ENCODING: [0xb5,0x8a,0x1e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11e8ab5 <unknown>

sumlall za.s[w8, 4:7], {z20.b - z23.b}, z14.b[10]  // 11000001-00011110-10001010-10110101
// CHECK-INST: sumlall za.s[w8, 4:7, vgx4], { z20.b - z23.b }, z14.b[10]
// CHECK-ENCODING: [0xb5,0x8a,0x1e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11e8ab5 <unknown>

sumlall za.s[w11, 0:3, vgx4], {z8.b - z11.b}, z1.b[5]  // 11000001-00010001-11100101-00110010
// CHECK-INST: sumlall za.s[w11, 0:3, vgx4], { z8.b - z11.b }, z1.b[5]
// CHECK-ENCODING: [0x32,0xe5,0x11,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c111e532 <unknown>

sumlall za.s[w11, 0:3], {z8.b - z11.b}, z1.b[5]  // 11000001-00010001-11100101-00110010
// CHECK-INST: sumlall za.s[w11, 0:3, vgx4], { z8.b - z11.b }, z1.b[5]
// CHECK-ENCODING: [0x32,0xe5,0x11,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c111e532 <unknown>

sumlall za.s[w9, 4:7, vgx4], {z12.b - z15.b}, z11.b[11]  // 11000001-00011011-10101001-10110111
// CHECK-INST: sumlall za.s[w9, 4:7, vgx4], { z12.b - z15.b }, z11.b[11]
// CHECK-ENCODING: [0xb7,0xa9,0x1b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11ba9b7 <unknown>

sumlall za.s[w9, 4:7], {z12.b - z15.b}, z11.b[11]  // 11000001-00011011-10101001-10110111
// CHECK-INST: sumlall za.s[w9, 4:7, vgx4], { z12.b - z15.b }, z11.b[11]
// CHECK-ENCODING: [0xb7,0xa9,0x1b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c11ba9b7 <unknown>

