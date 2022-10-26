// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p1 < %s \
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

whilehi pn8.h, x0, x0, vlx2  // 00100101-01100000-01001000-00011000
// CHECK-INST: whilehi pn8.h, x0, x0, vlx2
// CHECK-ENCODING: [0x18,0x48,0x60,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25604818 <unknown>

whilehi pn13.h, x10, x21, vlx2  // 00100101-01110101-01001001-01011101
// CHECK-INST: whilehi pn13.h, x10, x21, vlx2
// CHECK-ENCODING: [0x5d,0x49,0x75,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 2575495d <unknown>

whilehi pn15.h, x13, x8, vlx4  // 00100101-01101000-01101001-10111111
// CHECK-INST: whilehi pn15.h, x13, x8, vlx4
// CHECK-ENCODING: [0xbf,0x69,0x68,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 256869bf <unknown>

whilehi pn15.h, xzr, xzr, vlx4  // 00100101-01111111-01101011-11111111
// CHECK-INST: whilehi pn15.h, xzr, xzr, vlx4
// CHECK-ENCODING: [0xff,0x6b,0x7f,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 257f6bff <unknown>

whilehi {p0.h, p1.h}, x0, x0  // 00100101-01100000-01011000-00010001
// CHECK-INST: whilehi { p0.h, p1.h }, x0, x0
// CHECK-ENCODING: [0x11,0x58,0x60,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25605811 <unknown>

whilehi {p4.h, p5.h}, x10, x21  // 00100101-01110101-01011001-01010101
// CHECK-INST: whilehi { p4.h, p5.h }, x10, x21
// CHECK-ENCODING: [0x55,0x59,0x75,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25755955 <unknown>

whilehi {p6.h, p7.h}, x13, x8  // 00100101-01101000-01011001-10110111
// CHECK-INST: whilehi { p6.h, p7.h }, x13, x8
// CHECK-ENCODING: [0xb7,0x59,0x68,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 256859b7 <unknown>

whilehi {p14.h, p15.h}, xzr, xzr  // 00100101-01111111-01011011-11111111
// CHECK-INST: whilehi { p14.h, p15.h }, xzr, xzr
// CHECK-ENCODING: [0xff,0x5b,0x7f,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 257f5bff <unknown>

whilehi pn8.s, x0, x0, vlx2  // 00100101-10100000-01001000-00011000
// CHECK-INST: whilehi pn8.s, x0, x0, vlx2
// CHECK-ENCODING: [0x18,0x48,0xa0,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25a04818 <unknown>

whilehi pn13.s, x10, x21, vlx2  // 00100101-10110101-01001001-01011101
// CHECK-INST: whilehi pn13.s, x10, x21, vlx2
// CHECK-ENCODING: [0x5d,0x49,0xb5,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25b5495d <unknown>

whilehi pn15.s, x13, x8, vlx4  // 00100101-10101000-01101001-10111111
// CHECK-INST: whilehi pn15.s, x13, x8, vlx4
// CHECK-ENCODING: [0xbf,0x69,0xa8,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25a869bf <unknown>

whilehi pn15.s, xzr, xzr, vlx4  // 00100101-10111111-01101011-11111111
// CHECK-INST: whilehi pn15.s, xzr, xzr, vlx4
// CHECK-ENCODING: [0xff,0x6b,0xbf,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25bf6bff <unknown>

whilehi {p0.s, p1.s}, x0, x0  // 00100101-10100000-01011000-00010001
// CHECK-INST: whilehi { p0.s, p1.s }, x0, x0
// CHECK-ENCODING: [0x11,0x58,0xa0,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25a05811 <unknown>

whilehi {p4.s, p5.s}, x10, x21  // 00100101-10110101-01011001-01010101
// CHECK-INST: whilehi { p4.s, p5.s }, x10, x21
// CHECK-ENCODING: [0x55,0x59,0xb5,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25b55955 <unknown>

whilehi {p6.s, p7.s}, x13, x8  // 00100101-10101000-01011001-10110111
// CHECK-INST: whilehi { p6.s, p7.s }, x13, x8
// CHECK-ENCODING: [0xb7,0x59,0xa8,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25a859b7 <unknown>

whilehi {p14.s, p15.s}, xzr, xzr  // 00100101-10111111-01011011-11111111
// CHECK-INST: whilehi { p14.s, p15.s }, xzr, xzr
// CHECK-ENCODING: [0xff,0x5b,0xbf,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25bf5bff <unknown>

whilehi pn8.d, x0, x0, vlx2  // 00100101-11100000-01001000-00011000
// CHECK-INST: whilehi pn8.d, x0, x0, vlx2
// CHECK-ENCODING: [0x18,0x48,0xe0,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25e04818 <unknown>

whilehi pn13.d, x10, x21, vlx2  // 00100101-11110101-01001001-01011101
// CHECK-INST: whilehi pn13.d, x10, x21, vlx2
// CHECK-ENCODING: [0x5d,0x49,0xf5,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25f5495d <unknown>

whilehi pn15.d, x13, x8, vlx4  // 00100101-11101000-01101001-10111111
// CHECK-INST: whilehi pn15.d, x13, x8, vlx4
// CHECK-ENCODING: [0xbf,0x69,0xe8,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25e869bf <unknown>

whilehi pn15.d, xzr, xzr, vlx4  // 00100101-11111111-01101011-11111111
// CHECK-INST: whilehi pn15.d, xzr, xzr, vlx4
// CHECK-ENCODING: [0xff,0x6b,0xff,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25ff6bff <unknown>

whilehi {p0.d, p1.d}, x0, x0  // 00100101-11100000-01011000-00010001
// CHECK-INST: whilehi { p0.d, p1.d }, x0, x0
// CHECK-ENCODING: [0x11,0x58,0xe0,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25e05811 <unknown>

whilehi {p4.d, p5.d}, x10, x21  // 00100101-11110101-01011001-01010101
// CHECK-INST: whilehi { p4.d, p5.d }, x10, x21
// CHECK-ENCODING: [0x55,0x59,0xf5,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25f55955 <unknown>

whilehi {p6.d, p7.d}, x13, x8  // 00100101-11101000-01011001-10110111
// CHECK-INST: whilehi { p6.d, p7.d }, x13, x8
// CHECK-ENCODING: [0xb7,0x59,0xe8,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25e859b7 <unknown>

whilehi {p14.d, p15.d}, xzr, xzr  // 00100101-11111111-01011011-11111111
// CHECK-INST: whilehi { p14.d, p15.d }, xzr, xzr
// CHECK-ENCODING: [0xff,0x5b,0xff,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25ff5bff <unknown>

whilehi pn8.b, x0, x0, vlx2  // 00100101-00100000-01001000-00011000
// CHECK-INST: whilehi pn8.b, x0, x0, vlx2
// CHECK-ENCODING: [0x18,0x48,0x20,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25204818 <unknown>

whilehi pn13.b, x10, x21, vlx2  // 00100101-00110101-01001001-01011101
// CHECK-INST: whilehi pn13.b, x10, x21, vlx2
// CHECK-ENCODING: [0x5d,0x49,0x35,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 2535495d <unknown>

whilehi pn15.b, x13, x8, vlx4  // 00100101-00101000-01101001-10111111
// CHECK-INST: whilehi pn15.b, x13, x8, vlx4
// CHECK-ENCODING: [0xbf,0x69,0x28,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 252869bf <unknown>

whilehi pn15.b, xzr, xzr, vlx4  // 00100101-00111111-01101011-11111111
// CHECK-INST: whilehi pn15.b, xzr, xzr, vlx4
// CHECK-ENCODING: [0xff,0x6b,0x3f,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 253f6bff <unknown>

whilehi {p0.b, p1.b}, x0, x0  // 00100101-00100000-01011000-00010001
// CHECK-INST: whilehi { p0.b, p1.b }, x0, x0
// CHECK-ENCODING: [0x11,0x58,0x20,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25205811 <unknown>

whilehi {p4.b, p5.b}, x10, x21  // 00100101-00110101-01011001-01010101
// CHECK-INST: whilehi { p4.b, p5.b }, x10, x21
// CHECK-ENCODING: [0x55,0x59,0x35,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25355955 <unknown>

whilehi {p6.b, p7.b}, x13, x8  // 00100101-00101000-01011001-10110111
// CHECK-INST: whilehi { p6.b, p7.b }, x13, x8
// CHECK-ENCODING: [0xb7,0x59,0x28,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 252859b7 <unknown>

whilehi {p14.b, p15.b}, xzr, xzr  // 00100101-00111111-01011011-11111111
// CHECK-INST: whilehi { p14.b, p15.b }, xzr, xzr
// CHECK-ENCODING: [0xff,0x5b,0x3f,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 253f5bff <unknown>
