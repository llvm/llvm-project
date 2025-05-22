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

whilehs pn8.h, x0, x0, vlx2  // 00100101-01100000-01001000-00010000
// CHECK-INST: whilehs pn8.h, x0, x0, vlx2
// CHECK-ENCODING: [0x10,0x48,0x60,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25604810 <unknown>

whilehs pn13.h, x10, x21, vlx2  // 00100101-01110101-01001001-01010101
// CHECK-INST: whilehs pn13.h, x10, x21, vlx2
// CHECK-ENCODING: [0x55,0x49,0x75,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25754955 <unknown>

whilehs pn15.h, x13, x8, vlx4  // 00100101-01101000-01101001-10110111
// CHECK-INST: whilehs pn15.h, x13, x8, vlx4
// CHECK-ENCODING: [0xb7,0x69,0x68,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 256869b7 <unknown>

whilehs pn15.h, xzr, xzr, vlx4  // 00100101-01111111-01101011-11110111
// CHECK-INST: whilehs pn15.h, xzr, xzr, vlx4
// CHECK-ENCODING: [0xf7,0x6b,0x7f,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 257f6bf7 <unknown>

whilehs {p0.h, p1.h}, x0, x0  // 00100101-01100000-01011000-00010000
// CHECK-INST: whilehs { p0.h, p1.h }, x0, x0
// CHECK-ENCODING: [0x10,0x58,0x60,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25605810 <unknown>

whilehs {p4.h, p5.h}, x10, x21  // 00100101-01110101-01011001-01010100
// CHECK-INST: whilehs { p4.h, p5.h }, x10, x21
// CHECK-ENCODING: [0x54,0x59,0x75,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25755954 <unknown>

whilehs {p6.h, p7.h}, x13, x8  // 00100101-01101000-01011001-10110110
// CHECK-INST: whilehs { p6.h, p7.h }, x13, x8
// CHECK-ENCODING: [0xb6,0x59,0x68,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 256859b6 <unknown>

whilehs {p14.h, p15.h}, xzr, xzr  // 00100101-01111111-01011011-11111110
// CHECK-INST: whilehs { p14.h, p15.h }, xzr, xzr
// CHECK-ENCODING: [0xfe,0x5b,0x7f,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 257f5bfe <unknown>

whilehs pn8.s, x0, x0, vlx2  // 00100101-10100000-01001000-00010000
// CHECK-INST: whilehs pn8.s, x0, x0, vlx2
// CHECK-ENCODING: [0x10,0x48,0xa0,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25a04810 <unknown>

whilehs pn13.s, x10, x21, vlx2  // 00100101-10110101-01001001-01010101
// CHECK-INST: whilehs pn13.s, x10, x21, vlx2
// CHECK-ENCODING: [0x55,0x49,0xb5,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25b54955 <unknown>

whilehs pn15.s, x13, x8, vlx4  // 00100101-10101000-01101001-10110111
// CHECK-INST: whilehs pn15.s, x13, x8, vlx4
// CHECK-ENCODING: [0xb7,0x69,0xa8,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25a869b7 <unknown>

whilehs pn15.s, xzr, xzr, vlx4  // 00100101-10111111-01101011-11110111
// CHECK-INST: whilehs pn15.s, xzr, xzr, vlx4
// CHECK-ENCODING: [0xf7,0x6b,0xbf,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25bf6bf7 <unknown>

whilehs {p0.s, p1.s}, x0, x0  // 00100101-10100000-01011000-00010000
// CHECK-INST: whilehs { p0.s, p1.s }, x0, x0
// CHECK-ENCODING: [0x10,0x58,0xa0,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25a05810 <unknown>

whilehs {p4.s, p5.s}, x10, x21  // 00100101-10110101-01011001-01010100
// CHECK-INST: whilehs { p4.s, p5.s }, x10, x21
// CHECK-ENCODING: [0x54,0x59,0xb5,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25b55954 <unknown>

whilehs {p6.s, p7.s}, x13, x8  // 00100101-10101000-01011001-10110110
// CHECK-INST: whilehs { p6.s, p7.s }, x13, x8
// CHECK-ENCODING: [0xb6,0x59,0xa8,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25a859b6 <unknown>

whilehs {p14.s, p15.s}, xzr, xzr  // 00100101-10111111-01011011-11111110
// CHECK-INST: whilehs { p14.s, p15.s }, xzr, xzr
// CHECK-ENCODING: [0xfe,0x5b,0xbf,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25bf5bfe <unknown>

whilehs pn8.d, x0, x0, vlx2  // 00100101-11100000-01001000-00010000
// CHECK-INST: whilehs pn8.d, x0, x0, vlx2
// CHECK-ENCODING: [0x10,0x48,0xe0,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25e04810 <unknown>

whilehs pn13.d, x10, x21, vlx2  // 00100101-11110101-01001001-01010101
// CHECK-INST: whilehs pn13.d, x10, x21, vlx2
// CHECK-ENCODING: [0x55,0x49,0xf5,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25f54955 <unknown>

whilehs pn15.d, x13, x8, vlx4  // 00100101-11101000-01101001-10110111
// CHECK-INST: whilehs pn15.d, x13, x8, vlx4
// CHECK-ENCODING: [0xb7,0x69,0xe8,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25e869b7 <unknown>

whilehs pn15.d, xzr, xzr, vlx4  // 00100101-11111111-01101011-11110111
// CHECK-INST: whilehs pn15.d, xzr, xzr, vlx4
// CHECK-ENCODING: [0xf7,0x6b,0xff,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25ff6bf7 <unknown>

whilehs {p0.d, p1.d}, x0, x0  // 00100101-11100000-01011000-00010000
// CHECK-INST: whilehs { p0.d, p1.d }, x0, x0
// CHECK-ENCODING: [0x10,0x58,0xe0,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25e05810 <unknown>

whilehs {p4.d, p5.d}, x10, x21  // 00100101-11110101-01011001-01010100
// CHECK-INST: whilehs { p4.d, p5.d }, x10, x21
// CHECK-ENCODING: [0x54,0x59,0xf5,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25f55954 <unknown>

whilehs {p6.d, p7.d}, x13, x8  // 00100101-11101000-01011001-10110110
// CHECK-INST: whilehs { p6.d, p7.d }, x13, x8
// CHECK-ENCODING: [0xb6,0x59,0xe8,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25e859b6 <unknown>

whilehs {p14.d, p15.d}, xzr, xzr  // 00100101-11111111-01011011-11111110
// CHECK-INST: whilehs { p14.d, p15.d }, xzr, xzr
// CHECK-ENCODING: [0xfe,0x5b,0xff,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25ff5bfe <unknown>

whilehs pn8.b, x0, x0, vlx2  // 00100101-00100000-01001000-00010000
// CHECK-INST: whilehs pn8.b, x0, x0, vlx2
// CHECK-ENCODING: [0x10,0x48,0x20,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25204810 <unknown>

whilehs pn13.b, x10, x21, vlx2  // 00100101-00110101-01001001-01010101
// CHECK-INST: whilehs pn13.b, x10, x21, vlx2
// CHECK-ENCODING: [0x55,0x49,0x35,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25354955 <unknown>

whilehs pn15.b, x13, x8, vlx4  // 00100101-00101000-01101001-10110111
// CHECK-INST: whilehs pn15.b, x13, x8, vlx4
// CHECK-ENCODING: [0xb7,0x69,0x28,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 252869b7 <unknown>

whilehs pn15.b, xzr, xzr, vlx4  // 00100101-00111111-01101011-11110111
// CHECK-INST: whilehs pn15.b, xzr, xzr, vlx4
// CHECK-ENCODING: [0xf7,0x6b,0x3f,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 253f6bf7 <unknown>

whilehs {p0.b, p1.b}, x0, x0  // 00100101-00100000-01011000-00010000
// CHECK-INST: whilehs { p0.b, p1.b }, x0, x0
// CHECK-ENCODING: [0x10,0x58,0x20,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25205810 <unknown>

whilehs {p4.b, p5.b}, x10, x21  // 00100101-00110101-01011001-01010100
// CHECK-INST: whilehs { p4.b, p5.b }, x10, x21
// CHECK-ENCODING: [0x54,0x59,0x35,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25355954 <unknown>

whilehs {p6.b, p7.b}, x13, x8  // 00100101-00101000-01011001-10110110
// CHECK-INST: whilehs { p6.b, p7.b }, x13, x8
// CHECK-ENCODING: [0xb6,0x59,0x28,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 252859b6 <unknown>

whilehs {p14.b, p15.b}, xzr, xzr  // 00100101-00111111-01011011-11111110
// CHECK-INST: whilehs { p14.b, p15.b }, xzr, xzr
// CHECK-ENCODING: [0xfe,0x5b,0x3f,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 253f5bfe <unknown>
