// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p1 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2,-sve2p1 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

whilelt pn8.h, x0, x0, vlx2  // 00100101-01100000-01000100-00010000
// CHECK-INST: whilelt pn8.h, x0, x0, vlx2
// CHECK-ENCODING: [0x10,0x44,0x60,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25604410 <unknown>

whilelt pn13.h, x10, x21, vlx2  // 00100101-01110101-01000101-01010101
// CHECK-INST: whilelt pn13.h, x10, x21, vlx2
// CHECK-ENCODING: [0x55,0x45,0x75,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25754555 <unknown>

whilelt pn15.h, x13, x8, vlx4  // 00100101-01101000-01100101-10110111
// CHECK-INST: whilelt pn15.h, x13, x8, vlx4
// CHECK-ENCODING: [0xb7,0x65,0x68,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 256865b7 <unknown>

whilelt pn15.h, xzr, xzr, vlx4  // 00100101-01111111-01100111-11110111
// CHECK-INST: whilelt pn15.h, xzr, xzr, vlx4
// CHECK-ENCODING: [0xf7,0x67,0x7f,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 257f67f7 <unknown>

whilelt {p0.h, p1.h}, x0, x0  // 00100101-01100000-01010100-00010000
// CHECK-INST: whilelt { p0.h, p1.h }, x0, x0
// CHECK-ENCODING: [0x10,0x54,0x60,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25605410 <unknown>

whilelt {p4.h, p5.h}, x10, x21  // 00100101-01110101-01010101-01010100
// CHECK-INST: whilelt { p4.h, p5.h }, x10, x21
// CHECK-ENCODING: [0x54,0x55,0x75,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25755554 <unknown>

whilelt {p6.h, p7.h}, x13, x8  // 00100101-01101000-01010101-10110110
// CHECK-INST: whilelt { p6.h, p7.h }, x13, x8
// CHECK-ENCODING: [0xb6,0x55,0x68,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 256855b6 <unknown>

whilelt {p14.h, p15.h}, xzr, xzr  // 00100101-01111111-01010111-11111110
// CHECK-INST: whilelt { p14.h, p15.h }, xzr, xzr
// CHECK-ENCODING: [0xfe,0x57,0x7f,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 257f57fe <unknown>

whilelt pn8.s, x0, x0, vlx2  // 00100101-10100000-01000100-00010000
// CHECK-INST: whilelt pn8.s, x0, x0, vlx2
// CHECK-ENCODING: [0x10,0x44,0xa0,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25a04410 <unknown>

whilelt pn13.s, x10, x21, vlx2  // 00100101-10110101-01000101-01010101
// CHECK-INST: whilelt pn13.s, x10, x21, vlx2
// CHECK-ENCODING: [0x55,0x45,0xb5,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25b54555 <unknown>

whilelt pn15.s, x13, x8, vlx4  // 00100101-10101000-01100101-10110111
// CHECK-INST: whilelt pn15.s, x13, x8, vlx4
// CHECK-ENCODING: [0xb7,0x65,0xa8,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25a865b7 <unknown>

whilelt pn15.s, xzr, xzr, vlx4  // 00100101-10111111-01100111-11110111
// CHECK-INST: whilelt pn15.s, xzr, xzr, vlx4
// CHECK-ENCODING: [0xf7,0x67,0xbf,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25bf67f7 <unknown>

whilelt {p0.s, p1.s}, x0, x0  // 00100101-10100000-01010100-00010000
// CHECK-INST: whilelt { p0.s, p1.s }, x0, x0
// CHECK-ENCODING: [0x10,0x54,0xa0,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25a05410 <unknown>

whilelt {p4.s, p5.s}, x10, x21  // 00100101-10110101-01010101-01010100
// CHECK-INST: whilelt { p4.s, p5.s }, x10, x21
// CHECK-ENCODING: [0x54,0x55,0xb5,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25b55554 <unknown>

whilelt {p6.s, p7.s}, x13, x8  // 00100101-10101000-01010101-10110110
// CHECK-INST: whilelt { p6.s, p7.s }, x13, x8
// CHECK-ENCODING: [0xb6,0x55,0xa8,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25a855b6 <unknown>

whilelt {p14.s, p15.s}, xzr, xzr  // 00100101-10111111-01010111-11111110
// CHECK-INST: whilelt { p14.s, p15.s }, xzr, xzr
// CHECK-ENCODING: [0xfe,0x57,0xbf,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25bf57fe <unknown>

whilelt pn8.d, x0, x0, vlx2  // 00100101-11100000-01000100-00010000
// CHECK-INST: whilelt pn8.d, x0, x0, vlx2
// CHECK-ENCODING: [0x10,0x44,0xe0,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25e04410 <unknown>

whilelt pn13.d, x10, x21, vlx2  // 00100101-11110101-01000101-01010101
// CHECK-INST: whilelt pn13.d, x10, x21, vlx2
// CHECK-ENCODING: [0x55,0x45,0xf5,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25f54555 <unknown>

whilelt pn15.d, x13, x8, vlx4  // 00100101-11101000-01100101-10110111
// CHECK-INST: whilelt pn15.d, x13, x8, vlx4
// CHECK-ENCODING: [0xb7,0x65,0xe8,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25e865b7 <unknown>

whilelt pn15.d, xzr, xzr, vlx4  // 00100101-11111111-01100111-11110111
// CHECK-INST: whilelt pn15.d, xzr, xzr, vlx4
// CHECK-ENCODING: [0xf7,0x67,0xff,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25ff67f7 <unknown>

whilelt {p0.d, p1.d}, x0, x0  // 00100101-11100000-01010100-00010000
// CHECK-INST: whilelt { p0.d, p1.d }, x0, x0
// CHECK-ENCODING: [0x10,0x54,0xe0,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25e05410 <unknown>

whilelt {p4.d, p5.d}, x10, x21  // 00100101-11110101-01010101-01010100
// CHECK-INST: whilelt { p4.d, p5.d }, x10, x21
// CHECK-ENCODING: [0x54,0x55,0xf5,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25f55554 <unknown>

whilelt {p6.d, p7.d}, x13, x8  // 00100101-11101000-01010101-10110110
// CHECK-INST: whilelt { p6.d, p7.d }, x13, x8
// CHECK-ENCODING: [0xb6,0x55,0xe8,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25e855b6 <unknown>

whilelt {p14.d, p15.d}, xzr, xzr  // 00100101-11111111-01010111-11111110
// CHECK-INST: whilelt { p14.d, p15.d }, xzr, xzr
// CHECK-ENCODING: [0xfe,0x57,0xff,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25ff57fe <unknown>

whilelt pn8.b, x0, x0, vlx2  // 00100101-00100000-01000100-00010000
// CHECK-INST: whilelt pn8.b, x0, x0, vlx2
// CHECK-ENCODING: [0x10,0x44,0x20,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25204410 <unknown>

whilelt pn13.b, x10, x21, vlx2  // 00100101-00110101-01000101-01010101
// CHECK-INST: whilelt pn13.b, x10, x21, vlx2
// CHECK-ENCODING: [0x55,0x45,0x35,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25354555 <unknown>

whilelt pn15.b, x13, x8, vlx4  // 00100101-00101000-01100101-10110111
// CHECK-INST: whilelt pn15.b, x13, x8, vlx4
// CHECK-ENCODING: [0xb7,0x65,0x28,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 252865b7 <unknown>

whilelt pn15.b, xzr, xzr, vlx4  // 00100101-00111111-01100111-11110111
// CHECK-INST: whilelt pn15.b, xzr, xzr, vlx4
// CHECK-ENCODING: [0xf7,0x67,0x3f,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 253f67f7 <unknown>

whilelt {p0.b, p1.b}, x0, x0  // 00100101-00100000-01010100-00010000
// CHECK-INST: whilelt { p0.b, p1.b }, x0, x0
// CHECK-ENCODING: [0x10,0x54,0x20,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25205410 <unknown>

whilelt {p4.b, p5.b}, x10, x21  // 00100101-00110101-01010101-01010100
// CHECK-INST: whilelt { p4.b, p5.b }, x10, x21
// CHECK-ENCODING: [0x54,0x55,0x35,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25355554 <unknown>

whilelt {p6.b, p7.b}, x13, x8  // 00100101-00101000-01010101-10110110
// CHECK-INST: whilelt { p6.b, p7.b }, x13, x8
// CHECK-ENCODING: [0xb6,0x55,0x28,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 252855b6 <unknown>

whilelt {p14.b, p15.b}, xzr, xzr  // 00100101-00111111-01010111-11111110
// CHECK-INST: whilelt { p14.b, p15.b }, xzr, xzr
// CHECK-ENCODING: [0xfe,0x57,0x3f,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 253f57fe <unknown>
