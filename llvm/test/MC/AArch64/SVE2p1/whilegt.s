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

whilegt pn8.h, x0, x0, vlx2  // 00100101-01100000-01000000-00011000
// CHECK-INST: whilegt pn8.h, x0, x0, vlx2
// CHECK-ENCODING: [0x18,0x40,0x60,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25604018 <unknown>

whilegt pn13.h, x10, x21, vlx2  // 00100101-01110101-01000001-01011101
// CHECK-INST: whilegt pn13.h, x10, x21, vlx2
// CHECK-ENCODING: [0x5d,0x41,0x75,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 2575415d <unknown>

whilegt pn15.h, x13, x8, vlx4  // 00100101-01101000-01100001-10111111
// CHECK-INST: whilegt pn15.h, x13, x8, vlx4
// CHECK-ENCODING: [0xbf,0x61,0x68,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 256861bf <unknown>

whilegt pn15.h, xzr, xzr, vlx4  // 00100101-01111111-01100011-11111111
// CHECK-INST: whilegt pn15.h, xzr, xzr, vlx4
// CHECK-ENCODING: [0xff,0x63,0x7f,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 257f63ff <unknown>

whilegt {p0.h, p1.h}, x0, x0  // 00100101-01100000-01010000-00010001
// CHECK-INST: whilegt { p0.h, p1.h }, x0, x0
// CHECK-ENCODING: [0x11,0x50,0x60,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25605011 <unknown>

whilegt {p4.h, p5.h}, x10, x21  // 00100101-01110101-01010001-01010101
// CHECK-INST: whilegt { p4.h, p5.h }, x10, x21
// CHECK-ENCODING: [0x55,0x51,0x75,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25755155 <unknown>

whilegt {p6.h, p7.h}, x13, x8  // 00100101-01101000-01010001-10110111
// CHECK-INST: whilegt { p6.h, p7.h }, x13, x8
// CHECK-ENCODING: [0xb7,0x51,0x68,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 256851b7 <unknown>

whilegt {p14.h, p15.h}, xzr, xzr  // 00100101-01111111-01010011-11111111
// CHECK-INST: whilegt { p14.h, p15.h }, xzr, xzr
// CHECK-ENCODING: [0xff,0x53,0x7f,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 257f53ff <unknown>

whilegt pn8.s, x0, x0, vlx2  // 00100101-10100000-01000000-00011000
// CHECK-INST: whilegt pn8.s, x0, x0, vlx2
// CHECK-ENCODING: [0x18,0x40,0xa0,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25a04018 <unknown>

whilegt pn13.s, x10, x21, vlx2  // 00100101-10110101-01000001-01011101
// CHECK-INST: whilegt pn13.s, x10, x21, vlx2
// CHECK-ENCODING: [0x5d,0x41,0xb5,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25b5415d <unknown>

whilegt pn15.s, x13, x8, vlx4  // 00100101-10101000-01100001-10111111
// CHECK-INST: whilegt pn15.s, x13, x8, vlx4
// CHECK-ENCODING: [0xbf,0x61,0xa8,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25a861bf <unknown>

whilegt pn15.s, xzr, xzr, vlx4  // 00100101-10111111-01100011-11111111
// CHECK-INST: whilegt pn15.s, xzr, xzr, vlx4
// CHECK-ENCODING: [0xff,0x63,0xbf,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25bf63ff <unknown>

whilegt {p0.s, p1.s}, x0, x0  // 00100101-10100000-01010000-00010001
// CHECK-INST: whilegt { p0.s, p1.s }, x0, x0
// CHECK-ENCODING: [0x11,0x50,0xa0,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25a05011 <unknown>

whilegt {p4.s, p5.s}, x10, x21  // 00100101-10110101-01010001-01010101
// CHECK-INST: whilegt { p4.s, p5.s }, x10, x21
// CHECK-ENCODING: [0x55,0x51,0xb5,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25b55155 <unknown>

whilegt {p6.s, p7.s}, x13, x8  // 00100101-10101000-01010001-10110111
// CHECK-INST: whilegt { p6.s, p7.s }, x13, x8
// CHECK-ENCODING: [0xb7,0x51,0xa8,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25a851b7 <unknown>

whilegt {p14.s, p15.s}, xzr, xzr  // 00100101-10111111-01010011-11111111
// CHECK-INST: whilegt { p14.s, p15.s }, xzr, xzr
// CHECK-ENCODING: [0xff,0x53,0xbf,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25bf53ff <unknown>

whilegt pn8.d, x0, x0, vlx2  // 00100101-11100000-01000000-00011000
// CHECK-INST: whilegt pn8.d, x0, x0, vlx2
// CHECK-ENCODING: [0x18,0x40,0xe0,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25e04018 <unknown>

whilegt pn13.d, x10, x21, vlx2  // 00100101-11110101-01000001-01011101
// CHECK-INST: whilegt pn13.d, x10, x21, vlx2
// CHECK-ENCODING: [0x5d,0x41,0xf5,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25f5415d <unknown>

whilegt pn15.d, x13, x8, vlx4  // 00100101-11101000-01100001-10111111
// CHECK-INST: whilegt pn15.d, x13, x8, vlx4
// CHECK-ENCODING: [0xbf,0x61,0xe8,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25e861bf <unknown>

whilegt pn15.d, xzr, xzr, vlx4  // 00100101-11111111-01100011-11111111
// CHECK-INST: whilegt pn15.d, xzr, xzr, vlx4
// CHECK-ENCODING: [0xff,0x63,0xff,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25ff63ff <unknown>

whilegt {p0.d, p1.d}, x0, x0  // 00100101-11100000-01010000-00010001
// CHECK-INST: whilegt { p0.d, p1.d }, x0, x0
// CHECK-ENCODING: [0x11,0x50,0xe0,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25e05011 <unknown>

whilegt {p4.d, p5.d}, x10, x21  // 00100101-11110101-01010001-01010101
// CHECK-INST: whilegt { p4.d, p5.d }, x10, x21
// CHECK-ENCODING: [0x55,0x51,0xf5,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25f55155 <unknown>

whilegt {p6.d, p7.d}, x13, x8  // 00100101-11101000-01010001-10110111
// CHECK-INST: whilegt { p6.d, p7.d }, x13, x8
// CHECK-ENCODING: [0xb7,0x51,0xe8,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25e851b7 <unknown>

whilegt {p14.d, p15.d}, xzr, xzr  // 00100101-11111111-01010011-11111111
// CHECK-INST: whilegt { p14.d, p15.d }, xzr, xzr
// CHECK-ENCODING: [0xff,0x53,0xff,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25ff53ff <unknown>

whilegt pn8.b, x0, x0, vlx2  // 00100101-00100000-01000000-00011000
// CHECK-INST: whilegt pn8.b, x0, x0, vlx2
// CHECK-ENCODING: [0x18,0x40,0x20,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25204018 <unknown>

whilegt pn13.b, x10, x21, vlx2  // 00100101-00110101-01000001-01011101
// CHECK-INST: whilegt pn13.b, x10, x21, vlx2
// CHECK-ENCODING: [0x5d,0x41,0x35,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 2535415d <unknown>

whilegt pn15.b, x13, x8, vlx4  // 00100101-00101000-01100001-10111111
// CHECK-INST: whilegt pn15.b, x13, x8, vlx4
// CHECK-ENCODING: [0xbf,0x61,0x28,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 252861bf <unknown>

whilegt pn15.b, xzr, xzr, vlx4  // 00100101-00111111-01100011-11111111
// CHECK-INST: whilegt pn15.b, xzr, xzr, vlx4
// CHECK-ENCODING: [0xff,0x63,0x3f,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 253f63ff <unknown>

whilegt {p0.b, p1.b}, x0, x0  // 00100101-00100000-01010000-00010001
// CHECK-INST: whilegt { p0.b, p1.b }, x0, x0
// CHECK-ENCODING: [0x11,0x50,0x20,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25205011 <unknown>

whilegt {p4.b, p5.b}, x10, x21  // 00100101-00110101-01010001-01010101
// CHECK-INST: whilegt { p4.b, p5.b }, x10, x21
// CHECK-ENCODING: [0x55,0x51,0x35,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25355155 <unknown>

whilegt {p6.b, p7.b}, x13, x8  // 00100101-00101000-01010001-10110111
// CHECK-INST: whilegt { p6.b, p7.b }, x13, x8
// CHECK-ENCODING: [0xb7,0x51,0x28,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 252851b7 <unknown>

whilegt {p14.b, p15.b}, xzr, xzr  // 00100101-00111111-01010011-11111111
// CHECK-INST: whilegt { p14.b, p15.b }, xzr, xzr
// CHECK-ENCODING: [0xff,0x53,0x3f,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 253f53ff <unknown>
