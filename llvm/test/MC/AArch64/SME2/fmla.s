// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2,+sme-f64f64 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2,+sme-f64f64 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme2,+sme-f64f64 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2,+sme-f64f64 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2,+sme-f64f64 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2,+sme-f64f64 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST


fmla    za.d[w8, 0, vgx2], {z0.d, z1.d}, z0.d  // 11000001, 01100000, 00011000, 00000000
// CHECK-INST: fmla    za.d[w8, 0, vgx2], { z0.d, z1.d }, z0.d
// CHECK-ENCODING: [0x00,0x18,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1601800 <unknown>

fmla    za.d[w8, 0], {z0.d - z1.d}, z0.d  // 11000001-01100000-00011000-00000000
// CHECK-INST: fmla    za.d[w8, 0, vgx2], { z0.d, z1.d }, z0.d
// CHECK-ENCODING: [0x00,0x18,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1601800 <unknown>

fmla    za.d[w10, 5, vgx2], {z10.d, z11.d}, z5.d  // 11000001, 01100101, 01011001, 01000101
// CHECK-INST: fmla    za.d[w10, 5, vgx2], { z10.d, z11.d }, z5.d
// CHECK-ENCODING: [0x45,0x59,0x65,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1655945 <unknown>

fmla    za.d[w10, 5], {z10.d - z11.d}, z5.d  // 11000001-01100101-01011001-01000101
// CHECK-INST: fmla    za.d[w10, 5, vgx2], { z10.d, z11.d }, z5.d
// CHECK-ENCODING: [0x45,0x59,0x65,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1655945 <unknown>

fmla    za.d[w11, 7, vgx2], {z13.d, z14.d}, z8.d  // 11000001, 01101000, 01111001, 10100111
// CHECK-INST: fmla    za.d[w11, 7, vgx2], { z13.d, z14.d }, z8.d
// CHECK-ENCODING: [0xa7,0x79,0x68,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16879a7 <unknown>

fmla    za.d[w11, 7], {z13.d - z14.d}, z8.d  // 11000001-01101000-01111001-10100111
// CHECK-INST: fmla    za.d[w11, 7, vgx2], { z13.d, z14.d }, z8.d
// CHECK-ENCODING: [0xa7,0x79,0x68,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16879a7 <unknown>

fmla    za.d[w11, 7, vgx2], {z31.d, z0.d}, z15.d  // 11000001, 01101111, 01111011, 11100111
// CHECK-INST: fmla    za.d[w11, 7, vgx2], { z31.d, z0.d }, z15.d
// CHECK-ENCODING: [0xe7,0x7b,0x6f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16f7be7 <unknown>

fmla    za.d[w11, 7], {z31.d - z0.d}, z15.d  // 11000001-01101111-01111011-11100111
// CHECK-INST: fmla    za.d[w11, 7, vgx2], { z31.d, z0.d }, z15.d
// CHECK-ENCODING: [0xe7,0x7b,0x6f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16f7be7 <unknown>

fmla    za.d[w8, 5, vgx2], {z17.d, z18.d}, z0.d  // 11000001, 01100000, 00011010, 00100101
// CHECK-INST: fmla    za.d[w8, 5, vgx2], { z17.d, z18.d }, z0.d
// CHECK-ENCODING: [0x25,0x1a,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1601a25 <unknown>

fmla    za.d[w8, 5], {z17.d - z18.d}, z0.d  // 11000001-01100000-00011010-00100101
// CHECK-INST: fmla    za.d[w8, 5, vgx2], { z17.d, z18.d }, z0.d
// CHECK-ENCODING: [0x25,0x1a,0x60,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1601a25 <unknown>

fmla    za.d[w8, 1, vgx2], {z1.d, z2.d}, z14.d  // 11000001, 01101110, 00011000, 00100001
// CHECK-INST: fmla    za.d[w8, 1, vgx2], { z1.d, z2.d }, z14.d
// CHECK-ENCODING: [0x21,0x18,0x6e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16e1821 <unknown>

fmla    za.d[w8, 1], {z1.d - z2.d}, z14.d  // 11000001-01101110-00011000-00100001
// CHECK-INST: fmla    za.d[w8, 1, vgx2], { z1.d, z2.d }, z14.d
// CHECK-ENCODING: [0x21,0x18,0x6e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16e1821 <unknown>

fmla    za.d[w10, 0, vgx2], {z19.d, z20.d}, z4.d  // 11000001, 01100100, 01011010, 01100000
// CHECK-INST: fmla    za.d[w10, 0, vgx2], { z19.d, z20.d }, z4.d
// CHECK-ENCODING: [0x60,0x5a,0x64,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1645a60 <unknown>

fmla    za.d[w10, 0], {z19.d - z20.d}, z4.d  // 11000001-01100100-01011010-01100000
// CHECK-INST: fmla    za.d[w10, 0, vgx2], { z19.d, z20.d }, z4.d
// CHECK-ENCODING: [0x60,0x5a,0x64,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1645a60 <unknown>

fmla    za.d[w8, 0, vgx2], {z12.d, z13.d}, z2.d  // 11000001, 01100010, 00011001, 10000000
// CHECK-INST: fmla    za.d[w8, 0, vgx2], { z12.d, z13.d }, z2.d
// CHECK-ENCODING: [0x80,0x19,0x62,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1621980 <unknown>

fmla    za.d[w8, 0], {z12.d - z13.d}, z2.d  // 11000001-01100010-00011001-10000000
// CHECK-INST: fmla    za.d[w8, 0, vgx2], { z12.d, z13.d }, z2.d
// CHECK-ENCODING: [0x80,0x19,0x62,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1621980 <unknown>

fmla    za.d[w10, 1, vgx2], {z1.d, z2.d}, z10.d  // 11000001, 01101010, 01011000, 00100001
// CHECK-INST: fmla    za.d[w10, 1, vgx2], { z1.d, z2.d }, z10.d
// CHECK-ENCODING: [0x21,0x58,0x6a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16a5821 <unknown>

fmla    za.d[w10, 1], {z1.d - z2.d}, z10.d  // 11000001-01101010-01011000-00100001
// CHECK-INST: fmla    za.d[w10, 1, vgx2], { z1.d, z2.d }, z10.d
// CHECK-ENCODING: [0x21,0x58,0x6a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16a5821 <unknown>

fmla    za.d[w8, 5, vgx2], {z22.d, z23.d}, z14.d  // 11000001, 01101110, 00011010, 11000101
// CHECK-INST: fmla    za.d[w8, 5, vgx2], { z22.d, z23.d }, z14.d
// CHECK-ENCODING: [0xc5,0x1a,0x6e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16e1ac5 <unknown>

fmla    za.d[w8, 5], {z22.d - z23.d}, z14.d  // 11000001-01101110-00011010-11000101
// CHECK-INST: fmla    za.d[w8, 5, vgx2], { z22.d, z23.d }, z14.d
// CHECK-ENCODING: [0xc5,0x1a,0x6e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16e1ac5 <unknown>

fmla    za.d[w11, 2, vgx2], {z9.d, z10.d}, z1.d  // 11000001, 01100001, 01111001, 00100010
// CHECK-INST: fmla    za.d[w11, 2, vgx2], { z9.d, z10.d }, z1.d
// CHECK-ENCODING: [0x22,0x79,0x61,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1617922 <unknown>

fmla    za.d[w11, 2], {z9.d - z10.d}, z1.d  // 11000001-01100001-01111001-00100010
// CHECK-INST: fmla    za.d[w11, 2, vgx2], { z9.d, z10.d }, z1.d
// CHECK-ENCODING: [0x22,0x79,0x61,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1617922 <unknown>

fmla    za.d[w9, 7, vgx2], {z12.d, z13.d}, z11.d  // 11000001, 01101011, 00111001, 10000111
// CHECK-INST: fmla    za.d[w9, 7, vgx2], { z12.d, z13.d }, z11.d
// CHECK-ENCODING: [0x87,0x39,0x6b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16b3987 <unknown>

fmla    za.d[w9, 7], {z12.d - z13.d}, z11.d  // 11000001-01101011-00111001-10000111
// CHECK-INST: fmla    za.d[w9, 7, vgx2], { z12.d, z13.d }, z11.d
// CHECK-ENCODING: [0x87,0x39,0x6b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c16b3987 <unknown>


fmla    za.d[w8, 0, vgx2], {z0.d, z1.d}, {z0.d, z1.d}  // 11000001, 11100000, 00011000, 00000000
// CHECK-INST: fmla    za.d[w8, 0, vgx2], { z0.d, z1.d }, { z0.d, z1.d }
// CHECK-ENCODING: [0x00,0x18,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e01800 <unknown>

fmla    za.d[w8, 0], {z0.d - z1.d}, {z0.d - z1.d}  // 11000001-11100000-00011000-00000000
// CHECK-INST: fmla    za.d[w8, 0, vgx2], { z0.d, z1.d }, { z0.d, z1.d }
// CHECK-ENCODING: [0x00,0x18,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e01800 <unknown>

fmla    za.d[w10, 5, vgx2], {z10.d, z11.d}, {z20.d, z21.d}  // 11000001, 11110100, 01011001, 01000101
// CHECK-INST: fmla    za.d[w10, 5, vgx2], { z10.d, z11.d }, { z20.d, z21.d }
// CHECK-ENCODING: [0x45,0x59,0xf4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f45945 <unknown>

fmla    za.d[w10, 5], {z10.d - z11.d}, {z20.d - z21.d}  // 11000001-11110100-01011001-01000101
// CHECK-INST: fmla    za.d[w10, 5, vgx2], { z10.d, z11.d }, { z20.d, z21.d }
// CHECK-ENCODING: [0x45,0x59,0xf4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f45945 <unknown>

fmla    za.d[w11, 7, vgx2], {z12.d, z13.d}, {z8.d, z9.d}  // 11000001, 11101000, 01111001, 10000111
// CHECK-INST: fmla    za.d[w11, 7, vgx2], { z12.d, z13.d }, { z8.d, z9.d }
// CHECK-ENCODING: [0x87,0x79,0xe8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e87987 <unknown>

fmla    za.d[w11, 7], {z12.d - z13.d}, {z8.d - z9.d}  // 11000001-11101000-01111001-10000111
// CHECK-INST: fmla    za.d[w11, 7, vgx2], { z12.d, z13.d }, { z8.d, z9.d }
// CHECK-ENCODING: [0x87,0x79,0xe8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e87987 <unknown>

fmla    za.d[w11, 7, vgx2], {z30.d, z31.d}, {z30.d, z31.d}  // 11000001, 11111110, 01111011, 11000111
// CHECK-INST: fmla    za.d[w11, 7, vgx2], { z30.d, z31.d }, { z30.d, z31.d }
// CHECK-ENCODING: [0xc7,0x7b,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fe7bc7 <unknown>

fmla    za.d[w11, 7], {z30.d - z31.d}, {z30.d - z31.d}  // 11000001-11111110-01111011-11000111
// CHECK-INST: fmla    za.d[w11, 7, vgx2], { z30.d, z31.d }, { z30.d, z31.d }
// CHECK-ENCODING: [0xc7,0x7b,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fe7bc7 <unknown>

fmla    za.d[w8, 5, vgx2], {z16.d, z17.d}, {z16.d, z17.d}  // 11000001, 11110000, 00011010, 00000101
// CHECK-INST: fmla    za.d[w8, 5, vgx2], { z16.d, z17.d }, { z16.d, z17.d }
// CHECK-ENCODING: [0x05,0x1a,0xf0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f01a05 <unknown>

fmla    za.d[w8, 5], {z16.d - z17.d}, {z16.d - z17.d}  // 11000001-11110000-00011010-00000101
// CHECK-INST: fmla    za.d[w8, 5, vgx2], { z16.d, z17.d }, { z16.d, z17.d }
// CHECK-ENCODING: [0x05,0x1a,0xf0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f01a05 <unknown>

fmla    za.d[w8, 1, vgx2], {z0.d, z1.d}, {z30.d, z31.d}  // 11000001, 11111110, 00011000, 00000001
// CHECK-INST: fmla    za.d[w8, 1, vgx2], { z0.d, z1.d }, { z30.d, z31.d }
// CHECK-ENCODING: [0x01,0x18,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fe1801 <unknown>

fmla    za.d[w8, 1], {z0.d - z1.d}, {z30.d - z31.d}  // 11000001-11111110-00011000-00000001
// CHECK-INST: fmla    za.d[w8, 1, vgx2], { z0.d, z1.d }, { z30.d, z31.d }
// CHECK-ENCODING: [0x01,0x18,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fe1801 <unknown>

fmla    za.d[w10, 0, vgx2], {z18.d, z19.d}, {z20.d, z21.d}  // 11000001, 11110100, 01011010, 01000000
// CHECK-INST: fmla    za.d[w10, 0, vgx2], { z18.d, z19.d }, { z20.d, z21.d }
// CHECK-ENCODING: [0x40,0x5a,0xf4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f45a40 <unknown>

fmla    za.d[w10, 0], {z18.d - z19.d}, {z20.d - z21.d}  // 11000001-11110100-01011010-01000000
// CHECK-INST: fmla    za.d[w10, 0, vgx2], { z18.d, z19.d }, { z20.d, z21.d }
// CHECK-ENCODING: [0x40,0x5a,0xf4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f45a40 <unknown>

fmla    za.d[w8, 0, vgx2], {z12.d, z13.d}, {z2.d, z3.d}  // 11000001, 11100010, 00011001, 10000000
// CHECK-INST: fmla    za.d[w8, 0, vgx2], { z12.d, z13.d }, { z2.d, z3.d }
// CHECK-ENCODING: [0x80,0x19,0xe2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e21980 <unknown>

fmla    za.d[w8, 0], {z12.d - z13.d}, {z2.d - z3.d}  // 11000001-11100010-00011001-10000000
// CHECK-INST: fmla    za.d[w8, 0, vgx2], { z12.d, z13.d }, { z2.d, z3.d }
// CHECK-ENCODING: [0x80,0x19,0xe2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e21980 <unknown>

fmla    za.d[w10, 1, vgx2], {z0.d, z1.d}, {z26.d, z27.d}  // 11000001, 11111010, 01011000, 00000001
// CHECK-INST: fmla    za.d[w10, 1, vgx2], { z0.d, z1.d }, { z26.d, z27.d }
// CHECK-ENCODING: [0x01,0x58,0xfa,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fa5801 <unknown>

fmla    za.d[w10, 1], {z0.d - z1.d}, {z26.d - z27.d}  // 11000001-11111010-01011000-00000001
// CHECK-INST: fmla    za.d[w10, 1, vgx2], { z0.d, z1.d }, { z26.d, z27.d }
// CHECK-ENCODING: [0x01,0x58,0xfa,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fa5801 <unknown>

fmla    za.d[w8, 5, vgx2], {z22.d, z23.d}, {z30.d, z31.d}  // 11000001, 11111110, 00011010, 11000101
// CHECK-INST: fmla    za.d[w8, 5, vgx2], { z22.d, z23.d }, { z30.d, z31.d }
// CHECK-ENCODING: [0xc5,0x1a,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fe1ac5 <unknown>

fmla    za.d[w8, 5], {z22.d - z23.d}, {z30.d - z31.d}  // 11000001-11111110-00011010-11000101
// CHECK-INST: fmla    za.d[w8, 5, vgx2], { z22.d, z23.d }, { z30.d, z31.d }
// CHECK-ENCODING: [0xc5,0x1a,0xfe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fe1ac5 <unknown>

fmla    za.d[w11, 2, vgx2], {z8.d, z9.d}, {z0.d, z1.d}  // 11000001, 11100000, 01111001, 00000010
// CHECK-INST: fmla    za.d[w11, 2, vgx2], { z8.d, z9.d }, { z0.d, z1.d }
// CHECK-ENCODING: [0x02,0x79,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e07902 <unknown>

fmla    za.d[w11, 2], {z8.d - z9.d}, {z0.d - z1.d}  // 11000001-11100000-01111001-00000010
// CHECK-INST: fmla    za.d[w11, 2, vgx2], { z8.d, z9.d }, { z0.d, z1.d }
// CHECK-ENCODING: [0x02,0x79,0xe0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e07902 <unknown>

fmla    za.d[w9, 7, vgx2], {z12.d, z13.d}, {z10.d, z11.d}  // 11000001, 11101010, 00111001, 10000111
// CHECK-INST: fmla    za.d[w9, 7, vgx2], { z12.d, z13.d }, { z10.d, z11.d }
// CHECK-ENCODING: [0x87,0x39,0xea,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1ea3987 <unknown>

fmla    za.d[w9, 7], {z12.d - z13.d}, {z10.d - z11.d}  // 11000001-11101010-00111001-10000111
// CHECK-INST: fmla    za.d[w9, 7, vgx2], { z12.d, z13.d }, { z10.d, z11.d }
// CHECK-ENCODING: [0x87,0x39,0xea,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1ea3987 <unknown>


fmla    za.s[w8, 0, vgx2], {z0.s, z1.s}, z0.s  // 11000001, 00100000, 00011000, 00000000
// CHECK-INST: fmla    za.s[w8, 0, vgx2], { z0.s, z1.s }, z0.s
// CHECK-ENCODING: [0x00,0x18,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1201800 <unknown>

fmla    za.s[w8, 0], {z0.s - z1.s}, z0.s  // 11000001-00100000-00011000-00000000
// CHECK-INST: fmla    za.s[w8, 0, vgx2], { z0.s, z1.s }, z0.s
// CHECK-ENCODING: [0x00,0x18,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1201800 <unknown>

fmla    za.s[w10, 5, vgx2], {z10.s, z11.s}, z5.s  // 11000001, 00100101, 01011001, 01000101
// CHECK-INST: fmla    za.s[w10, 5, vgx2], { z10.s, z11.s }, z5.s
// CHECK-ENCODING: [0x45,0x59,0x25,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1255945 <unknown>

fmla    za.s[w10, 5], {z10.s - z11.s}, z5.s  // 11000001-00100101-01011001-01000101
// CHECK-INST: fmla    za.s[w10, 5, vgx2], { z10.s, z11.s }, z5.s
// CHECK-ENCODING: [0x45,0x59,0x25,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1255945 <unknown>

fmla    za.s[w11, 7, vgx2], {z13.s, z14.s}, z8.s  // 11000001, 00101000, 01111001, 10100111
// CHECK-INST: fmla    za.s[w11, 7, vgx2], { z13.s, z14.s }, z8.s
// CHECK-ENCODING: [0xa7,0x79,0x28,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12879a7 <unknown>

fmla    za.s[w11, 7], {z13.s - z14.s}, z8.s  // 11000001-00101000-01111001-10100111
// CHECK-INST: fmla    za.s[w11, 7, vgx2], { z13.s, z14.s }, z8.s
// CHECK-ENCODING: [0xa7,0x79,0x28,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12879a7 <unknown>

fmla    za.s[w11, 7, vgx2], {z31.s, z0.s}, z15.s  // 11000001, 00101111, 01111011, 11100111
// CHECK-INST: fmla    za.s[w11, 7, vgx2], { z31.s, z0.s }, z15.s
// CHECK-ENCODING: [0xe7,0x7b,0x2f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12f7be7 <unknown>

fmla    za.s[w11, 7], {z31.s - z0.s}, z15.s  // 11000001-00101111-01111011-11100111
// CHECK-INST: fmla    za.s[w11, 7, vgx2], { z31.s, z0.s }, z15.s
// CHECK-ENCODING: [0xe7,0x7b,0x2f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12f7be7 <unknown>

fmla    za.s[w8, 5, vgx2], {z17.s, z18.s}, z0.s  // 11000001, 00100000, 00011010, 00100101
// CHECK-INST: fmla    za.s[w8, 5, vgx2], { z17.s, z18.s }, z0.s
// CHECK-ENCODING: [0x25,0x1a,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1201a25 <unknown>

fmla    za.s[w8, 5], {z17.s - z18.s}, z0.s  // 11000001-00100000-00011010-00100101
// CHECK-INST: fmla    za.s[w8, 5, vgx2], { z17.s, z18.s }, z0.s
// CHECK-ENCODING: [0x25,0x1a,0x20,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1201a25 <unknown>

fmla    za.s[w8, 1, vgx2], {z1.s, z2.s}, z14.s  // 11000001, 00101110, 00011000, 00100001
// CHECK-INST: fmla    za.s[w8, 1, vgx2], { z1.s, z2.s }, z14.s
// CHECK-ENCODING: [0x21,0x18,0x2e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12e1821 <unknown>

fmla    za.s[w8, 1], {z1.s - z2.s}, z14.s  // 11000001-00101110-00011000-00100001
// CHECK-INST: fmla    za.s[w8, 1, vgx2], { z1.s, z2.s }, z14.s
// CHECK-ENCODING: [0x21,0x18,0x2e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12e1821 <unknown>

fmla    za.s[w10, 0, vgx2], {z19.s, z20.s}, z4.s  // 11000001, 00100100, 01011010, 01100000
// CHECK-INST: fmla    za.s[w10, 0, vgx2], { z19.s, z20.s }, z4.s
// CHECK-ENCODING: [0x60,0x5a,0x24,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1245a60 <unknown>

fmla    za.s[w10, 0], {z19.s - z20.s}, z4.s  // 11000001-00100100-01011010-01100000
// CHECK-INST: fmla    za.s[w10, 0, vgx2], { z19.s, z20.s }, z4.s
// CHECK-ENCODING: [0x60,0x5a,0x24,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1245a60 <unknown>

fmla    za.s[w8, 0, vgx2], {z12.s, z13.s}, z2.s  // 11000001, 00100010, 00011001, 10000000
// CHECK-INST: fmla    za.s[w8, 0, vgx2], { z12.s, z13.s }, z2.s
// CHECK-ENCODING: [0x80,0x19,0x22,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1221980 <unknown>

fmla    za.s[w8, 0], {z12.s - z13.s}, z2.s  // 11000001-00100010-00011001-10000000
// CHECK-INST: fmla    za.s[w8, 0, vgx2], { z12.s, z13.s }, z2.s
// CHECK-ENCODING: [0x80,0x19,0x22,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1221980 <unknown>

fmla    za.s[w10, 1, vgx2], {z1.s, z2.s}, z10.s  // 11000001, 00101010, 01011000, 00100001
// CHECK-INST: fmla    za.s[w10, 1, vgx2], { z1.s, z2.s }, z10.s
// CHECK-ENCODING: [0x21,0x58,0x2a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12a5821 <unknown>

fmla    za.s[w10, 1], {z1.s - z2.s}, z10.s  // 11000001-00101010-01011000-00100001
// CHECK-INST: fmla    za.s[w10, 1, vgx2], { z1.s, z2.s }, z10.s
// CHECK-ENCODING: [0x21,0x58,0x2a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12a5821 <unknown>

fmla    za.s[w8, 5, vgx2], {z22.s, z23.s}, z14.s  // 11000001, 00101110, 00011010, 11000101
// CHECK-INST: fmla    za.s[w8, 5, vgx2], { z22.s, z23.s }, z14.s
// CHECK-ENCODING: [0xc5,0x1a,0x2e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12e1ac5 <unknown>

fmla    za.s[w8, 5], {z22.s - z23.s}, z14.s  // 11000001-00101110-00011010-11000101
// CHECK-INST: fmla    za.s[w8, 5, vgx2], { z22.s, z23.s }, z14.s
// CHECK-ENCODING: [0xc5,0x1a,0x2e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12e1ac5 <unknown>

fmla    za.s[w11, 2, vgx2], {z9.s, z10.s}, z1.s  // 11000001, 00100001, 01111001, 00100010
// CHECK-INST: fmla    za.s[w11, 2, vgx2], { z9.s, z10.s }, z1.s
// CHECK-ENCODING: [0x22,0x79,0x21,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1217922 <unknown>

fmla    za.s[w11, 2], {z9.s - z10.s}, z1.s  // 11000001-00100001-01111001-00100010
// CHECK-INST: fmla    za.s[w11, 2, vgx2], { z9.s, z10.s }, z1.s
// CHECK-ENCODING: [0x22,0x79,0x21,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1217922 <unknown>

fmla    za.s[w9, 7, vgx2], {z12.s, z13.s}, z11.s  // 11000001, 00101011, 00111001, 10000111
// CHECK-INST: fmla    za.s[w9, 7, vgx2], { z12.s, z13.s }, z11.s
// CHECK-ENCODING: [0x87,0x39,0x2b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12b3987 <unknown>

fmla    za.s[w9, 7], {z12.s - z13.s}, z11.s  // 11000001-00101011-00111001-10000111
// CHECK-INST: fmla    za.s[w9, 7, vgx2], { z12.s, z13.s }, z11.s
// CHECK-ENCODING: [0x87,0x39,0x2b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c12b3987 <unknown>


fmla    za.s[w8, 0, vgx2], {z0.s, z1.s}, {z0.s, z1.s}  // 11000001, 10100000, 00011000, 00000000
// CHECK-INST: fmla    za.s[w8, 0, vgx2], { z0.s, z1.s }, { z0.s, z1.s }
// CHECK-ENCODING: [0x00,0x18,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a01800 <unknown>

fmla    za.s[w8, 0], {z0.s - z1.s}, {z0.s - z1.s}  // 11000001-10100000-00011000-00000000
// CHECK-INST: fmla    za.s[w8, 0, vgx2], { z0.s, z1.s }, { z0.s, z1.s }
// CHECK-ENCODING: [0x00,0x18,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a01800 <unknown>

fmla    za.s[w10, 5, vgx2], {z10.s, z11.s}, {z20.s, z21.s}  // 11000001, 10110100, 01011001, 01000101
// CHECK-INST: fmla    za.s[w10, 5, vgx2], { z10.s, z11.s }, { z20.s, z21.s }
// CHECK-ENCODING: [0x45,0x59,0xb4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b45945 <unknown>

fmla    za.s[w10, 5], {z10.s - z11.s}, {z20.s - z21.s}  // 11000001-10110100-01011001-01000101
// CHECK-INST: fmla    za.s[w10, 5, vgx2], { z10.s, z11.s }, { z20.s, z21.s }
// CHECK-ENCODING: [0x45,0x59,0xb4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b45945 <unknown>

fmla    za.s[w11, 7, vgx2], {z12.s, z13.s}, {z8.s, z9.s}  // 11000001, 10101000, 01111001, 10000111
// CHECK-INST: fmla    za.s[w11, 7, vgx2], { z12.s, z13.s }, { z8.s, z9.s }
// CHECK-ENCODING: [0x87,0x79,0xa8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a87987 <unknown>

fmla    za.s[w11, 7], {z12.s - z13.s}, {z8.s - z9.s}  // 11000001-10101000-01111001-10000111
// CHECK-INST: fmla    za.s[w11, 7, vgx2], { z12.s, z13.s }, { z8.s, z9.s }
// CHECK-ENCODING: [0x87,0x79,0xa8,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a87987 <unknown>

fmla    za.s[w11, 7, vgx2], {z30.s, z31.s}, {z30.s, z31.s}  // 11000001, 10111110, 01111011, 11000111
// CHECK-INST: fmla    za.s[w11, 7, vgx2], { z30.s, z31.s }, { z30.s, z31.s }
// CHECK-ENCODING: [0xc7,0x7b,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1be7bc7 <unknown>

fmla    za.s[w11, 7], {z30.s - z31.s}, {z30.s - z31.s}  // 11000001-10111110-01111011-11000111
// CHECK-INST: fmla    za.s[w11, 7, vgx2], { z30.s, z31.s }, { z30.s, z31.s }
// CHECK-ENCODING: [0xc7,0x7b,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1be7bc7 <unknown>

fmla    za.s[w8, 5, vgx2], {z16.s, z17.s}, {z16.s, z17.s}  // 11000001, 10110000, 00011010, 00000101
// CHECK-INST: fmla    za.s[w8, 5, vgx2], { z16.s, z17.s }, { z16.s, z17.s }
// CHECK-ENCODING: [0x05,0x1a,0xb0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b01a05 <unknown>

fmla    za.s[w8, 5], {z16.s - z17.s}, {z16.s - z17.s}  // 11000001-10110000-00011010-00000101
// CHECK-INST: fmla    za.s[w8, 5, vgx2], { z16.s, z17.s }, { z16.s, z17.s }
// CHECK-ENCODING: [0x05,0x1a,0xb0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b01a05 <unknown>

fmla    za.s[w8, 1, vgx2], {z0.s, z1.s}, {z30.s, z31.s}  // 11000001, 10111110, 00011000, 00000001
// CHECK-INST: fmla    za.s[w8, 1, vgx2], { z0.s, z1.s }, { z30.s, z31.s }
// CHECK-ENCODING: [0x01,0x18,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1be1801 <unknown>

fmla    za.s[w8, 1], {z0.s - z1.s}, {z30.s - z31.s}  // 11000001-10111110-00011000-00000001
// CHECK-INST: fmla    za.s[w8, 1, vgx2], { z0.s, z1.s }, { z30.s, z31.s }
// CHECK-ENCODING: [0x01,0x18,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1be1801 <unknown>

fmla    za.s[w10, 0, vgx2], {z18.s, z19.s}, {z20.s, z21.s}  // 11000001, 10110100, 01011010, 01000000
// CHECK-INST: fmla    za.s[w10, 0, vgx2], { z18.s, z19.s }, { z20.s, z21.s }
// CHECK-ENCODING: [0x40,0x5a,0xb4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b45a40 <unknown>

fmla    za.s[w10, 0], {z18.s - z19.s}, {z20.s - z21.s}  // 11000001-10110100-01011010-01000000
// CHECK-INST: fmla    za.s[w10, 0, vgx2], { z18.s, z19.s }, { z20.s, z21.s }
// CHECK-ENCODING: [0x40,0x5a,0xb4,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b45a40 <unknown>

fmla    za.s[w8, 0, vgx2], {z12.s, z13.s}, {z2.s, z3.s}  // 11000001, 10100010, 00011001, 10000000
// CHECK-INST: fmla    za.s[w8, 0, vgx2], { z12.s, z13.s }, { z2.s, z3.s }
// CHECK-ENCODING: [0x80,0x19,0xa2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a21980 <unknown>

fmla    za.s[w8, 0], {z12.s - z13.s}, {z2.s - z3.s}  // 11000001-10100010-00011001-10000000
// CHECK-INST: fmla    za.s[w8, 0, vgx2], { z12.s, z13.s }, { z2.s, z3.s }
// CHECK-ENCODING: [0x80,0x19,0xa2,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a21980 <unknown>

fmla    za.s[w10, 1, vgx2], {z0.s, z1.s}, {z26.s, z27.s}  // 11000001, 10111010, 01011000, 00000001
// CHECK-INST: fmla    za.s[w10, 1, vgx2], { z0.s, z1.s }, { z26.s, z27.s }
// CHECK-ENCODING: [0x01,0x58,0xba,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1ba5801 <unknown>

fmla    za.s[w10, 1], {z0.s - z1.s}, {z26.s - z27.s}  // 11000001-10111010-01011000-00000001
// CHECK-INST: fmla    za.s[w10, 1, vgx2], { z0.s, z1.s }, { z26.s, z27.s }
// CHECK-ENCODING: [0x01,0x58,0xba,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1ba5801 <unknown>

fmla    za.s[w8, 5, vgx2], {z22.s, z23.s}, {z30.s, z31.s}  // 11000001, 10111110, 00011010, 11000101
// CHECK-INST: fmla    za.s[w8, 5, vgx2], { z22.s, z23.s }, { z30.s, z31.s }
// CHECK-ENCODING: [0xc5,0x1a,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1be1ac5 <unknown>

fmla    za.s[w8, 5], {z22.s - z23.s}, {z30.s - z31.s}  // 11000001-10111110-00011010-11000101
// CHECK-INST: fmla    za.s[w8, 5, vgx2], { z22.s, z23.s }, { z30.s, z31.s }
// CHECK-ENCODING: [0xc5,0x1a,0xbe,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1be1ac5 <unknown>

fmla    za.s[w11, 2, vgx2], {z8.s, z9.s}, {z0.s, z1.s}  // 11000001, 10100000, 01111001, 00000010
// CHECK-INST: fmla    za.s[w11, 2, vgx2], { z8.s, z9.s }, { z0.s, z1.s }
// CHECK-ENCODING: [0x02,0x79,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a07902 <unknown>

fmla    za.s[w11, 2], {z8.s - z9.s}, {z0.s - z1.s}  // 11000001-10100000-01111001-00000010
// CHECK-INST: fmla    za.s[w11, 2, vgx2], { z8.s, z9.s }, { z0.s, z1.s }
// CHECK-ENCODING: [0x02,0x79,0xa0,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a07902 <unknown>

fmla    za.s[w9, 7, vgx2], {z12.s, z13.s}, {z10.s, z11.s}  // 11000001, 10101010, 00111001, 10000111
// CHECK-INST: fmla    za.s[w9, 7, vgx2], { z12.s, z13.s }, { z10.s, z11.s }
// CHECK-ENCODING: [0x87,0x39,0xaa,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1aa3987 <unknown>

fmla    za.s[w9, 7], {z12.s - z13.s}, {z10.s - z11.s}  // 11000001-10101010-00111001-10000111
// CHECK-INST: fmla    za.s[w9, 7, vgx2], { z12.s, z13.s }, { z10.s, z11.s }
// CHECK-ENCODING: [0x87,0x39,0xaa,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1aa3987 <unknown>


fmla    za.d[w8, 0, vgx4], {z0.d - z3.d}, z0.d  // 11000001-01110000-00011000-00000000
// CHECK-INST: fmla    za.d[w8, 0, vgx4], { z0.d - z3.d }, z0.d
// CHECK-ENCODING: [0x00,0x18,0x70,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1701800 <unknown>

fmla    za.d[w8, 0], {z0.d - z3.d}, z0.d  // 11000001-01110000-00011000-00000000
// CHECK-INST: fmla    za.d[w8, 0, vgx4], { z0.d - z3.d }, z0.d
// CHECK-ENCODING: [0x00,0x18,0x70,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1701800 <unknown>

fmla    za.d[w10, 5, vgx4], {z10.d - z13.d}, z5.d  // 11000001-01110101-01011001-01000101
// CHECK-INST: fmla    za.d[w10, 5, vgx4], { z10.d - z13.d }, z5.d
// CHECK-ENCODING: [0x45,0x59,0x75,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1755945 <unknown>

fmla    za.d[w10, 5], {z10.d - z13.d}, z5.d  // 11000001-01110101-01011001-01000101
// CHECK-INST: fmla    za.d[w10, 5, vgx4], { z10.d - z13.d }, z5.d
// CHECK-ENCODING: [0x45,0x59,0x75,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1755945 <unknown>

fmla    za.d[w11, 7, vgx4], {z13.d - z16.d}, z8.d  // 11000001-01111000-01111001-10100111
// CHECK-INST: fmla    za.d[w11, 7, vgx4], { z13.d - z16.d }, z8.d
// CHECK-ENCODING: [0xa7,0x79,0x78,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17879a7 <unknown>

fmla    za.d[w11, 7], {z13.d - z16.d}, z8.d  // 11000001-01111000-01111001-10100111
// CHECK-INST: fmla    za.d[w11, 7, vgx4], { z13.d - z16.d }, z8.d
// CHECK-ENCODING: [0xa7,0x79,0x78,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17879a7 <unknown>

fmla    za.d[w11, 7, vgx4], {z31.d - z2.d}, z15.d  // 11000001-01111111-01111011-11100111
// CHECK-INST: fmla    za.d[w11, 7, vgx4], { z31.d, z0.d, z1.d, z2.d }, z15.d
// CHECK-ENCODING: [0xe7,0x7b,0x7f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17f7be7 <unknown>

fmla    za.d[w11, 7], {z31.d - z2.d}, z15.d  // 11000001-01111111-01111011-11100111
// CHECK-INST: fmla    za.d[w11, 7, vgx4], { z31.d, z0.d, z1.d, z2.d }, z15.d
// CHECK-ENCODING: [0xe7,0x7b,0x7f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17f7be7 <unknown>

fmla    za.d[w8, 5, vgx4], {z17.d - z20.d}, z0.d  // 11000001-01110000-00011010-00100101
// CHECK-INST: fmla    za.d[w8, 5, vgx4], { z17.d - z20.d }, z0.d
// CHECK-ENCODING: [0x25,0x1a,0x70,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1701a25 <unknown>

fmla    za.d[w8, 5], {z17.d - z20.d}, z0.d  // 11000001-01110000-00011010-00100101
// CHECK-INST: fmla    za.d[w8, 5, vgx4], { z17.d - z20.d }, z0.d
// CHECK-ENCODING: [0x25,0x1a,0x70,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1701a25 <unknown>

fmla    za.d[w8, 1, vgx4], {z1.d - z4.d}, z14.d  // 11000001-01111110-00011000-00100001
// CHECK-INST: fmla    za.d[w8, 1, vgx4], { z1.d - z4.d }, z14.d
// CHECK-ENCODING: [0x21,0x18,0x7e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17e1821 <unknown>

fmla    za.d[w8, 1], {z1.d - z4.d}, z14.d  // 11000001-01111110-00011000-00100001
// CHECK-INST: fmla    za.d[w8, 1, vgx4], { z1.d - z4.d }, z14.d
// CHECK-ENCODING: [0x21,0x18,0x7e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17e1821 <unknown>

fmla    za.d[w10, 0, vgx4], {z19.d - z22.d}, z4.d  // 11000001-01110100-01011010-01100000
// CHECK-INST: fmla    za.d[w10, 0, vgx4], { z19.d - z22.d }, z4.d
// CHECK-ENCODING: [0x60,0x5a,0x74,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1745a60 <unknown>

fmla    za.d[w10, 0], {z19.d - z22.d}, z4.d  // 11000001-01110100-01011010-01100000
// CHECK-INST: fmla    za.d[w10, 0, vgx4], { z19.d - z22.d }, z4.d
// CHECK-ENCODING: [0x60,0x5a,0x74,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1745a60 <unknown>

fmla    za.d[w8, 0, vgx4], {z12.d - z15.d}, z2.d  // 11000001-01110010-00011001-10000000
// CHECK-INST: fmla    za.d[w8, 0, vgx4], { z12.d - z15.d }, z2.d
// CHECK-ENCODING: [0x80,0x19,0x72,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1721980 <unknown>

fmla    za.d[w8, 0], {z12.d - z15.d}, z2.d  // 11000001-01110010-00011001-10000000
// CHECK-INST: fmla    za.d[w8, 0, vgx4], { z12.d - z15.d }, z2.d
// CHECK-ENCODING: [0x80,0x19,0x72,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1721980 <unknown>

fmla    za.d[w10, 1, vgx4], {z1.d - z4.d}, z10.d  // 11000001-01111010-01011000-00100001
// CHECK-INST: fmla    za.d[w10, 1, vgx4], { z1.d - z4.d }, z10.d
// CHECK-ENCODING: [0x21,0x58,0x7a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17a5821 <unknown>

fmla    za.d[w10, 1], {z1.d - z4.d}, z10.d  // 11000001-01111010-01011000-00100001
// CHECK-INST: fmla    za.d[w10, 1, vgx4], { z1.d - z4.d }, z10.d
// CHECK-ENCODING: [0x21,0x58,0x7a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17a5821 <unknown>

fmla    za.d[w8, 5, vgx4], {z22.d - z25.d}, z14.d  // 11000001-01111110-00011010-11000101
// CHECK-INST: fmla    za.d[w8, 5, vgx4], { z22.d - z25.d }, z14.d
// CHECK-ENCODING: [0xc5,0x1a,0x7e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17e1ac5 <unknown>

fmla    za.d[w8, 5], {z22.d - z25.d}, z14.d  // 11000001-01111110-00011010-11000101
// CHECK-INST: fmla    za.d[w8, 5, vgx4], { z22.d - z25.d }, z14.d
// CHECK-ENCODING: [0xc5,0x1a,0x7e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17e1ac5 <unknown>

fmla    za.d[w11, 2, vgx4], {z9.d - z12.d}, z1.d  // 11000001-01110001-01111001-00100010
// CHECK-INST: fmla    za.d[w11, 2, vgx4], { z9.d - z12.d }, z1.d
// CHECK-ENCODING: [0x22,0x79,0x71,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1717922 <unknown>

fmla    za.d[w11, 2], {z9.d - z12.d}, z1.d  // 11000001-01110001-01111001-00100010
// CHECK-INST: fmla    za.d[w11, 2, vgx4], { z9.d - z12.d }, z1.d
// CHECK-ENCODING: [0x22,0x79,0x71,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1717922 <unknown>

fmla    za.d[w9, 7, vgx4], {z12.d - z15.d}, z11.d  // 11000001-01111011-00111001-10000111
// CHECK-INST: fmla    za.d[w9, 7, vgx4], { z12.d - z15.d }, z11.d
// CHECK-ENCODING: [0x87,0x39,0x7b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17b3987 <unknown>

fmla    za.d[w9, 7], {z12.d - z15.d}, z11.d  // 11000001-01111011-00111001-10000111
// CHECK-INST: fmla    za.d[w9, 7, vgx4], { z12.d - z15.d }, z11.d
// CHECK-ENCODING: [0x87,0x39,0x7b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c17b3987 <unknown>


fmla    za.d[w8, 0, vgx4], {z0.d - z3.d}, {z0.d - z3.d}  // 11000001-11100001-00011000-00000000
// CHECK-INST: fmla    za.d[w8, 0, vgx4], { z0.d - z3.d }, { z0.d - z3.d }
// CHECK-ENCODING: [0x00,0x18,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e11800 <unknown>

fmla    za.d[w8, 0], {z0.d - z3.d}, {z0.d - z3.d}  // 11000001-11100001-00011000-00000000
// CHECK-INST: fmla    za.d[w8, 0, vgx4], { z0.d - z3.d }, { z0.d - z3.d }
// CHECK-ENCODING: [0x00,0x18,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e11800 <unknown>

fmla    za.d[w10, 5, vgx4], {z8.d - z11.d}, {z20.d - z23.d}  // 11000001-11110101-01011001-00000101
// CHECK-INST: fmla    za.d[w10, 5, vgx4], { z8.d - z11.d }, { z20.d - z23.d }
// CHECK-ENCODING: [0x05,0x59,0xf5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f55905 <unknown>

fmla    za.d[w10, 5], {z8.d - z11.d}, {z20.d - z23.d}  // 11000001-11110101-01011001-00000101
// CHECK-INST: fmla    za.d[w10, 5, vgx4], { z8.d - z11.d }, { z20.d - z23.d }
// CHECK-ENCODING: [0x05,0x59,0xf5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f55905 <unknown>

fmla    za.d[w11, 7, vgx4], {z12.d - z15.d}, {z8.d - z11.d}  // 11000001-11101001-01111001-10000111
// CHECK-INST: fmla    za.d[w11, 7, vgx4], { z12.d - z15.d }, { z8.d - z11.d }
// CHECK-ENCODING: [0x87,0x79,0xe9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e97987 <unknown>

fmla    za.d[w11, 7], {z12.d - z15.d}, {z8.d - z11.d}  // 11000001-11101001-01111001-10000111
// CHECK-INST: fmla    za.d[w11, 7, vgx4], { z12.d - z15.d }, { z8.d - z11.d }
// CHECK-ENCODING: [0x87,0x79,0xe9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e97987 <unknown>

fmla    za.d[w11, 7, vgx4], {z28.d - z31.d}, {z28.d - z31.d}  // 11000001-11111101-01111011-10000111
// CHECK-INST: fmla    za.d[w11, 7, vgx4], { z28.d - z31.d }, { z28.d - z31.d }
// CHECK-ENCODING: [0x87,0x7b,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd7b87 <unknown>

fmla    za.d[w11, 7], {z28.d - z31.d}, {z28.d - z31.d}  // 11000001-11111101-01111011-10000111
// CHECK-INST: fmla    za.d[w11, 7, vgx4], { z28.d - z31.d }, { z28.d - z31.d }
// CHECK-ENCODING: [0x87,0x7b,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd7b87 <unknown>

fmla    za.d[w8, 5, vgx4], {z16.d - z19.d}, {z16.d - z19.d}  // 11000001-11110001-00011010-00000101
// CHECK-INST: fmla    za.d[w8, 5, vgx4], { z16.d - z19.d }, { z16.d - z19.d }
// CHECK-ENCODING: [0x05,0x1a,0xf1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f11a05 <unknown>

fmla    za.d[w8, 5], {z16.d - z19.d}, {z16.d - z19.d}  // 11000001-11110001-00011010-00000101
// CHECK-INST: fmla    za.d[w8, 5, vgx4], { z16.d - z19.d }, { z16.d - z19.d }
// CHECK-ENCODING: [0x05,0x1a,0xf1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f11a05 <unknown>

fmla    za.d[w8, 1, vgx4], {z0.d - z3.d}, {z28.d - z31.d}  // 11000001-11111101-00011000-00000001
// CHECK-INST: fmla    za.d[w8, 1, vgx4], { z0.d - z3.d }, { z28.d - z31.d }
// CHECK-ENCODING: [0x01,0x18,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd1801 <unknown>

fmla    za.d[w8, 1], {z0.d - z3.d}, {z28.d - z31.d}  // 11000001-11111101-00011000-00000001
// CHECK-INST: fmla    za.d[w8, 1, vgx4], { z0.d - z3.d }, { z28.d - z31.d }
// CHECK-ENCODING: [0x01,0x18,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd1801 <unknown>

fmla    za.d[w10, 0, vgx4], {z16.d - z19.d}, {z20.d - z23.d}  // 11000001-11110101-01011010-00000000
// CHECK-INST: fmla    za.d[w10, 0, vgx4], { z16.d - z19.d }, { z20.d - z23.d }
// CHECK-ENCODING: [0x00,0x5a,0xf5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f55a00 <unknown>

fmla    za.d[w10, 0], {z16.d - z19.d}, {z20.d - z23.d}  // 11000001-11110101-01011010-00000000
// CHECK-INST: fmla    za.d[w10, 0, vgx4], { z16.d - z19.d }, { z20.d - z23.d }
// CHECK-ENCODING: [0x00,0x5a,0xf5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f55a00 <unknown>

fmla    za.d[w8, 0, vgx4], {z12.d - z15.d}, {z0.d - z3.d}  // 11000001-11100001-00011001-10000000
// CHECK-INST: fmla    za.d[w8, 0, vgx4], { z12.d - z15.d }, { z0.d - z3.d }
// CHECK-ENCODING: [0x80,0x19,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e11980 <unknown>

fmla    za.d[w8, 0], {z12.d - z15.d}, {z0.d - z3.d}  // 11000001-11100001-00011001-10000000
// CHECK-INST: fmla    za.d[w8, 0, vgx4], { z12.d - z15.d }, { z0.d - z3.d }
// CHECK-ENCODING: [0x80,0x19,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e11980 <unknown>

fmla    za.d[w10, 1, vgx4], {z0.d - z3.d}, {z24.d - z27.d}  // 11000001-11111001-01011000-00000001
// CHECK-INST: fmla    za.d[w10, 1, vgx4], { z0.d - z3.d }, { z24.d - z27.d }
// CHECK-ENCODING: [0x01,0x58,0xf9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f95801 <unknown>

fmla    za.d[w10, 1], {z0.d - z3.d}, {z24.d - z27.d}  // 11000001-11111001-01011000-00000001
// CHECK-INST: fmla    za.d[w10, 1, vgx4], { z0.d - z3.d }, { z24.d - z27.d }
// CHECK-ENCODING: [0x01,0x58,0xf9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1f95801 <unknown>

fmla    za.d[w8, 5, vgx4], {z20.d - z23.d}, {z28.d - z31.d}  // 11000001-11111101-00011010-10000101
// CHECK-INST: fmla    za.d[w8, 5, vgx4], { z20.d - z23.d }, { z28.d - z31.d }
// CHECK-ENCODING: [0x85,0x1a,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd1a85 <unknown>

fmla    za.d[w8, 5], {z20.d - z23.d}, {z28.d - z31.d}  // 11000001-11111101-00011010-10000101
// CHECK-INST: fmla    za.d[w8, 5, vgx4], { z20.d - z23.d }, { z28.d - z31.d }
// CHECK-ENCODING: [0x85,0x1a,0xfd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1fd1a85 <unknown>

fmla    za.d[w11, 2, vgx4], {z8.d - z11.d}, {z0.d - z3.d}  // 11000001-11100001-01111001-00000010
// CHECK-INST: fmla    za.d[w11, 2, vgx4], { z8.d - z11.d }, { z0.d - z3.d }
// CHECK-ENCODING: [0x02,0x79,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e17902 <unknown>

fmla    za.d[w11, 2], {z8.d - z11.d}, {z0.d - z3.d}  // 11000001-11100001-01111001-00000010
// CHECK-INST: fmla    za.d[w11, 2, vgx4], { z8.d - z11.d }, { z0.d - z3.d }
// CHECK-ENCODING: [0x02,0x79,0xe1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e17902 <unknown>

fmla    za.d[w9, 7, vgx4], {z12.d - z15.d}, {z8.d - z11.d}  // 11000001-11101001-00111001-10000111
// CHECK-INST: fmla    za.d[w9, 7, vgx4], { z12.d - z15.d }, { z8.d - z11.d }
// CHECK-ENCODING: [0x87,0x39,0xe9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e93987 <unknown>

fmla    za.d[w9, 7], {z12.d - z15.d}, {z8.d - z11.d}  // 11000001-11101001-00111001-10000111
// CHECK-INST: fmla    za.d[w9, 7, vgx4], { z12.d - z15.d }, { z8.d - z11.d }
// CHECK-ENCODING: [0x87,0x39,0xe9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1e93987 <unknown>


fmla    za.s[w8, 0, vgx4], {z0.s - z3.s}, z0.s  // 11000001-00110000-00011000-00000000
// CHECK-INST: fmla    za.s[w8, 0, vgx4], { z0.s - z3.s }, z0.s
// CHECK-ENCODING: [0x00,0x18,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1301800 <unknown>

fmla    za.s[w8, 0], {z0.s - z3.s}, z0.s  // 11000001-00110000-00011000-00000000
// CHECK-INST: fmla    za.s[w8, 0, vgx4], { z0.s - z3.s }, z0.s
// CHECK-ENCODING: [0x00,0x18,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1301800 <unknown>

fmla    za.s[w10, 5, vgx4], {z10.s - z13.s}, z5.s  // 11000001-00110101-01011001-01000101
// CHECK-INST: fmla    za.s[w10, 5, vgx4], { z10.s - z13.s }, z5.s
// CHECK-ENCODING: [0x45,0x59,0x35,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1355945 <unknown>

fmla    za.s[w10, 5], {z10.s - z13.s}, z5.s  // 11000001-00110101-01011001-01000101
// CHECK-INST: fmla    za.s[w10, 5, vgx4], { z10.s - z13.s }, z5.s
// CHECK-ENCODING: [0x45,0x59,0x35,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1355945 <unknown>

fmla    za.s[w11, 7, vgx4], {z13.s - z16.s}, z8.s  // 11000001-00111000-01111001-10100111
// CHECK-INST: fmla    za.s[w11, 7, vgx4], { z13.s - z16.s }, z8.s
// CHECK-ENCODING: [0xa7,0x79,0x38,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13879a7 <unknown>

fmla    za.s[w11, 7], {z13.s - z16.s}, z8.s  // 11000001-00111000-01111001-10100111
// CHECK-INST: fmla    za.s[w11, 7, vgx4], { z13.s - z16.s }, z8.s
// CHECK-ENCODING: [0xa7,0x79,0x38,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13879a7 <unknown>

fmla    za.s[w11, 7, vgx4], {z31.s - z2.s}, z15.s  // 11000001-00111111-01111011-11100111
// CHECK-INST: fmla    za.s[w11, 7, vgx4], { z31.s, z0.s, z1.s, z2.s }, z15.s
// CHECK-ENCODING: [0xe7,0x7b,0x3f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13f7be7 <unknown>

fmla    za.s[w11, 7], {z31.s - z2.s}, z15.s  // 11000001-00111111-01111011-11100111
// CHECK-INST: fmla    za.s[w11, 7, vgx4], { z31.s, z0.s, z1.s, z2.s }, z15.s
// CHECK-ENCODING: [0xe7,0x7b,0x3f,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13f7be7 <unknown>

fmla    za.s[w8, 5, vgx4], {z17.s - z20.s}, z0.s  // 11000001-00110000-00011010-00100101
// CHECK-INST: fmla    za.s[w8, 5, vgx4], { z17.s - z20.s }, z0.s
// CHECK-ENCODING: [0x25,0x1a,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1301a25 <unknown>

fmla    za.s[w8, 5], {z17.s - z20.s}, z0.s  // 11000001-00110000-00011010-00100101
// CHECK-INST: fmla    za.s[w8, 5, vgx4], { z17.s - z20.s }, z0.s
// CHECK-ENCODING: [0x25,0x1a,0x30,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1301a25 <unknown>

fmla    za.s[w8, 1, vgx4], {z1.s - z4.s}, z14.s  // 11000001-00111110-00011000-00100001
// CHECK-INST: fmla    za.s[w8, 1, vgx4], { z1.s - z4.s }, z14.s
// CHECK-ENCODING: [0x21,0x18,0x3e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13e1821 <unknown>

fmla    za.s[w8, 1], {z1.s - z4.s}, z14.s  // 11000001-00111110-00011000-00100001
// CHECK-INST: fmla    za.s[w8, 1, vgx4], { z1.s - z4.s }, z14.s
// CHECK-ENCODING: [0x21,0x18,0x3e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13e1821 <unknown>

fmla    za.s[w10, 0, vgx4], {z19.s - z22.s}, z4.s  // 11000001-00110100-01011010-01100000
// CHECK-INST: fmla    za.s[w10, 0, vgx4], { z19.s - z22.s }, z4.s
// CHECK-ENCODING: [0x60,0x5a,0x34,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1345a60 <unknown>

fmla    za.s[w10, 0], {z19.s - z22.s}, z4.s  // 11000001-00110100-01011010-01100000
// CHECK-INST: fmla    za.s[w10, 0, vgx4], { z19.s - z22.s }, z4.s
// CHECK-ENCODING: [0x60,0x5a,0x34,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1345a60 <unknown>

fmla    za.s[w8, 0, vgx4], {z12.s - z15.s}, z2.s  // 11000001-00110010-00011001-10000000
// CHECK-INST: fmla    za.s[w8, 0, vgx4], { z12.s - z15.s }, z2.s
// CHECK-ENCODING: [0x80,0x19,0x32,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1321980 <unknown>

fmla    za.s[w8, 0], {z12.s - z15.s}, z2.s  // 11000001-00110010-00011001-10000000
// CHECK-INST: fmla    za.s[w8, 0, vgx4], { z12.s - z15.s }, z2.s
// CHECK-ENCODING: [0x80,0x19,0x32,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1321980 <unknown>

fmla    za.s[w10, 1, vgx4], {z1.s - z4.s}, z10.s  // 11000001-00111010-01011000-00100001
// CHECK-INST: fmla    za.s[w10, 1, vgx4], { z1.s - z4.s }, z10.s
// CHECK-ENCODING: [0x21,0x58,0x3a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13a5821 <unknown>

fmla    za.s[w10, 1], {z1.s - z4.s}, z10.s  // 11000001-00111010-01011000-00100001
// CHECK-INST: fmla    za.s[w10, 1, vgx4], { z1.s - z4.s }, z10.s
// CHECK-ENCODING: [0x21,0x58,0x3a,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13a5821 <unknown>

fmla    za.s[w8, 5, vgx4], {z22.s - z25.s}, z14.s  // 11000001-00111110-00011010-11000101
// CHECK-INST: fmla    za.s[w8, 5, vgx4], { z22.s - z25.s }, z14.s
// CHECK-ENCODING: [0xc5,0x1a,0x3e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13e1ac5 <unknown>

fmla    za.s[w8, 5], {z22.s - z25.s}, z14.s  // 11000001-00111110-00011010-11000101
// CHECK-INST: fmla    za.s[w8, 5, vgx4], { z22.s - z25.s }, z14.s
// CHECK-ENCODING: [0xc5,0x1a,0x3e,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13e1ac5 <unknown>

fmla    za.s[w11, 2, vgx4], {z9.s - z12.s}, z1.s  // 11000001-00110001-01111001-00100010
// CHECK-INST: fmla    za.s[w11, 2, vgx4], { z9.s - z12.s }, z1.s
// CHECK-ENCODING: [0x22,0x79,0x31,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1317922 <unknown>

fmla    za.s[w11, 2], {z9.s - z12.s}, z1.s  // 11000001-00110001-01111001-00100010
// CHECK-INST: fmla    za.s[w11, 2, vgx4], { z9.s - z12.s }, z1.s
// CHECK-ENCODING: [0x22,0x79,0x31,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1317922 <unknown>

fmla    za.s[w9, 7, vgx4], {z12.s - z15.s}, z11.s  // 11000001-00111011-00111001-10000111
// CHECK-INST: fmla    za.s[w9, 7, vgx4], { z12.s - z15.s }, z11.s
// CHECK-ENCODING: [0x87,0x39,0x3b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13b3987 <unknown>

fmla    za.s[w9, 7], {z12.s - z15.s}, z11.s  // 11000001-00111011-00111001-10000111
// CHECK-INST: fmla    za.s[w9, 7, vgx4], { z12.s - z15.s }, z11.s
// CHECK-ENCODING: [0x87,0x39,0x3b,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c13b3987 <unknown>



fmla    za.s[w8, 0, vgx4], {z0.s - z3.s}, {z0.s - z3.s}  // 11000001-10100001-00011000-00000000
// CHECK-INST: fmla    za.s[w8, 0, vgx4], { z0.s - z3.s }, { z0.s - z3.s }
// CHECK-ENCODING: [0x00,0x18,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a11800 <unknown>

fmla    za.s[w8, 0], {z0.s - z3.s}, {z0.s - z3.s}  // 11000001-10100001-00011000-00000000
// CHECK-INST: fmla    za.s[w8, 0, vgx4], { z0.s - z3.s }, { z0.s - z3.s }
// CHECK-ENCODING: [0x00,0x18,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a11800 <unknown>

fmla    za.s[w10, 5, vgx4], {z8.s - z11.s}, {z20.s - z23.s}  // 11000001-10110101-01011001-00000101
// CHECK-INST: fmla    za.s[w10, 5, vgx4], { z8.s - z11.s }, { z20.s - z23.s }
// CHECK-ENCODING: [0x05,0x59,0xb5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b55905 <unknown>

fmla    za.s[w10, 5], {z8.s - z11.s}, {z20.s - z23.s}  // 11000001-10110101-01011001-00000101
// CHECK-INST: fmla    za.s[w10, 5, vgx4], { z8.s - z11.s }, { z20.s - z23.s }
// CHECK-ENCODING: [0x05,0x59,0xb5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b55905 <unknown>

fmla    za.s[w11, 7, vgx4], {z12.s - z15.s}, {z8.s - z11.s}  // 11000001-10101001-01111001-10000111
// CHECK-INST: fmla    za.s[w11, 7, vgx4], { z12.s - z15.s }, { z8.s - z11.s }
// CHECK-ENCODING: [0x87,0x79,0xa9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a97987 <unknown>

fmla    za.s[w11, 7], {z12.s - z15.s}, {z8.s - z11.s}  // 11000001-10101001-01111001-10000111
// CHECK-INST: fmla    za.s[w11, 7, vgx4], { z12.s - z15.s }, { z8.s - z11.s }
// CHECK-ENCODING: [0x87,0x79,0xa9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a97987 <unknown>

fmla    za.s[w11, 7, vgx4], {z28.s - z31.s}, {z28.s - z31.s}  // 11000001-10111101-01111011-10000111
// CHECK-INST: fmla    za.s[w11, 7, vgx4], { z28.s - z31.s }, { z28.s - z31.s }
// CHECK-ENCODING: [0x87,0x7b,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bd7b87 <unknown>

fmla    za.s[w11, 7], {z28.s - z31.s}, {z28.s - z31.s}  // 11000001-10111101-01111011-10000111
// CHECK-INST: fmla    za.s[w11, 7, vgx4], { z28.s - z31.s }, { z28.s - z31.s }
// CHECK-ENCODING: [0x87,0x7b,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bd7b87 <unknown>

fmla    za.s[w8, 5, vgx4], {z16.s - z19.s}, {z16.s - z19.s}  // 11000001-10110001-00011010-00000101
// CHECK-INST: fmla    za.s[w8, 5, vgx4], { z16.s - z19.s }, { z16.s - z19.s }
// CHECK-ENCODING: [0x05,0x1a,0xb1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b11a05 <unknown>

fmla    za.s[w8, 5], {z16.s - z19.s}, {z16.s - z19.s}  // 11000001-10110001-00011010-00000101
// CHECK-INST: fmla    za.s[w8, 5, vgx4], { z16.s - z19.s }, { z16.s - z19.s }
// CHECK-ENCODING: [0x05,0x1a,0xb1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b11a05 <unknown>

fmla    za.s[w8, 1, vgx4], {z0.s - z3.s}, {z28.s - z31.s}  // 11000001-10111101-00011000-00000001
// CHECK-INST: fmla    za.s[w8, 1, vgx4], { z0.s - z3.s }, { z28.s - z31.s }
// CHECK-ENCODING: [0x01,0x18,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bd1801 <unknown>

fmla    za.s[w8, 1], {z0.s - z3.s}, {z28.s - z31.s}  // 11000001-10111101-00011000-00000001
// CHECK-INST: fmla    za.s[w8, 1, vgx4], { z0.s - z3.s }, { z28.s - z31.s }
// CHECK-ENCODING: [0x01,0x18,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bd1801 <unknown>

fmla    za.s[w10, 0, vgx4], {z16.s - z19.s}, {z20.s - z23.s}  // 11000001-10110101-01011010-00000000
// CHECK-INST: fmla    za.s[w10, 0, vgx4], { z16.s - z19.s }, { z20.s - z23.s }
// CHECK-ENCODING: [0x00,0x5a,0xb5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b55a00 <unknown>

fmla    za.s[w10, 0], {z16.s - z19.s}, {z20.s - z23.s}  // 11000001-10110101-01011010-00000000
// CHECK-INST: fmla    za.s[w10, 0, vgx4], { z16.s - z19.s }, { z20.s - z23.s }
// CHECK-ENCODING: [0x00,0x5a,0xb5,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b55a00 <unknown>

fmla    za.s[w8, 0, vgx4], {z12.s - z15.s}, {z0.s - z3.s}  // 11000001-10100001-00011001-10000000
// CHECK-INST: fmla    za.s[w8, 0, vgx4], { z12.s - z15.s }, { z0.s - z3.s }
// CHECK-ENCODING: [0x80,0x19,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a11980 <unknown>

fmla    za.s[w8, 0], {z12.s - z15.s}, {z0.s - z3.s}  // 11000001-10100001-00011001-10000000
// CHECK-INST: fmla    za.s[w8, 0, vgx4], { z12.s - z15.s }, { z0.s - z3.s }
// CHECK-ENCODING: [0x80,0x19,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a11980 <unknown>

fmla    za.s[w10, 1, vgx4], {z0.s - z3.s}, {z24.s - z27.s}  // 11000001-10111001-01011000-00000001
// CHECK-INST: fmla    za.s[w10, 1, vgx4], { z0.s - z3.s }, { z24.s - z27.s }
// CHECK-ENCODING: [0x01,0x58,0xb9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b95801 <unknown>

fmla    za.s[w10, 1], {z0.s - z3.s}, {z24.s - z27.s}  // 11000001-10111001-01011000-00000001
// CHECK-INST: fmla    za.s[w10, 1, vgx4], { z0.s - z3.s }, { z24.s - z27.s }
// CHECK-ENCODING: [0x01,0x58,0xb9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1b95801 <unknown>

fmla    za.s[w8, 5, vgx4], {z20.s - z23.s}, {z28.s - z31.s}  // 11000001-10111101-00011010-10000101
// CHECK-INST: fmla    za.s[w8, 5, vgx4], { z20.s - z23.s }, { z28.s - z31.s }
// CHECK-ENCODING: [0x85,0x1a,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bd1a85 <unknown>

fmla    za.s[w8, 5], {z20.s - z23.s}, {z28.s - z31.s}  // 11000001-10111101-00011010-10000101
// CHECK-INST: fmla    za.s[w8, 5, vgx4], { z20.s - z23.s }, { z28.s - z31.s }
// CHECK-ENCODING: [0x85,0x1a,0xbd,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1bd1a85 <unknown>

fmla    za.s[w11, 2, vgx4], {z8.s - z11.s}, {z0.s - z3.s}  // 11000001-10100001-01111001-00000010
// CHECK-INST: fmla    za.s[w11, 2, vgx4], { z8.s - z11.s }, { z0.s - z3.s }
// CHECK-ENCODING: [0x02,0x79,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a17902 <unknown>

fmla    za.s[w11, 2], {z8.s - z11.s}, {z0.s - z3.s}  // 11000001-10100001-01111001-00000010
// CHECK-INST: fmla    za.s[w11, 2, vgx4], { z8.s - z11.s }, { z0.s - z3.s }
// CHECK-ENCODING: [0x02,0x79,0xa1,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a17902 <unknown>

fmla    za.s[w9, 7, vgx4], {z12.s - z15.s}, {z8.s - z11.s}  // 11000001-10101001-00111001-10000111
// CHECK-INST: fmla    za.s[w9, 7, vgx4], { z12.s - z15.s }, { z8.s - z11.s }
// CHECK-ENCODING: [0x87,0x39,0xa9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a93987 <unknown>

fmla    za.s[w9, 7], {z12.s - z15.s}, {z8.s - z11.s}  // 11000001-10101001-00111001-10000111
// CHECK-INST: fmla    za.s[w9, 7, vgx4], { z12.s - z15.s }, { z8.s - z11.s }
// CHECK-ENCODING: [0x87,0x39,0xa9,0xc1]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c1a93987 <unknown>

