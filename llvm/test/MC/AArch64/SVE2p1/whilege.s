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

whilege pn8.h, x0, x0, vlx2  // 00100101-01100000-01000000-00010000
// CHECK-INST: whilege pn8.h, x0, x0, vlx2
// CHECK-ENCODING: [0x10,0x40,0x60,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25604010 <unknown>

whilege pn13.h, x10, x21, vlx2  // 00100101-01110101-01000001-01010101
// CHECK-INST: whilege pn13.h, x10, x21, vlx2
// CHECK-ENCODING: [0x55,0x41,0x75,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25754155 <unknown>

whilege pn15.h, x13, x8, vlx4  // 00100101-01101000-01100001-10110111
// CHECK-INST: whilege pn15.h, x13, x8, vlx4
// CHECK-ENCODING: [0xb7,0x61,0x68,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 256861b7 <unknown>

whilege pn15.h, xzr, xzr, vlx4  // 00100101-01111111-01100011-11110111
// CHECK-INST: whilege pn15.h, xzr, xzr, vlx4
// CHECK-ENCODING: [0xf7,0x63,0x7f,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 257f63f7 <unknown>

whilege pn8.s, x0, x0, vlx2  // 00100101-10100000-01000000-00010000
// CHECK-INST: whilege pn8.s, x0, x0, vlx2
// CHECK-ENCODING: [0x10,0x40,0xa0,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25a04010 <unknown>

whilege pn13.s, x10, x21, vlx2  // 00100101-10110101-01000001-01010101
// CHECK-INST: whilege pn13.s, x10, x21, vlx2
// CHECK-ENCODING: [0x55,0x41,0xb5,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25b54155 <unknown>

whilege pn15.s, x13, x8, vlx4  // 00100101-10101000-01100001-10110111
// CHECK-INST: whilege pn15.s, x13, x8, vlx4
// CHECK-ENCODING: [0xb7,0x61,0xa8,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25a861b7 <unknown>

whilege pn15.s, xzr, xzr, vlx4  // 00100101-10111111-01100011-11110111
// CHECK-INST: whilege pn15.s, xzr, xzr, vlx4
// CHECK-ENCODING: [0xf7,0x63,0xbf,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25bf63f7 <unknown>

whilege pn8.d, x0, x0, vlx2  // 00100101-11100000-01000000-00010000
// CHECK-INST: whilege pn8.d, x0, x0, vlx2
// CHECK-ENCODING: [0x10,0x40,0xe0,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25e04010 <unknown>

whilege pn13.d, x10, x21, vlx2  // 00100101-11110101-01000001-01010101
// CHECK-INST: whilege pn13.d, x10, x21, vlx2
// CHECK-ENCODING: [0x55,0x41,0xf5,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25f54155 <unknown>

whilege pn15.d, x13, x8, vlx4  // 00100101-11101000-01100001-10110111
// CHECK-INST: whilege pn15.d, x13, x8, vlx4
// CHECK-ENCODING: [0xb7,0x61,0xe8,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25e861b7 <unknown>

whilege pn15.d, xzr, xzr, vlx4  // 00100101-11111111-01100011-11110111
// CHECK-INST: whilege pn15.d, xzr, xzr, vlx4
// CHECK-ENCODING: [0xf7,0x63,0xff,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25ff63f7 <unknown>

whilege pn8.b, x0, x0, vlx2  // 00100101-00100000-01000000-00010000
// CHECK-INST: whilege pn8.b, x0, x0, vlx2
// CHECK-ENCODING: [0x10,0x40,0x20,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25204010 <unknown>

whilege pn13.b, x10, x21, vlx2  // 00100101-00110101-01000001-01010101
// CHECK-INST: whilege pn13.b, x10, x21, vlx2
// CHECK-ENCODING: [0x55,0x41,0x35,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25354155 <unknown>

whilege pn15.b, x13, x8, vlx4  // 00100101-00101000-01100001-10110111
// CHECK-INST: whilege pn15.b, x13, x8, vlx4
// CHECK-ENCODING: [0xb7,0x61,0x28,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 252861b7 <unknown>

whilege pn15.b, xzr, xzr, vlx4  // 00100101-00111111-01100011-11110111
// CHECK-INST: whilege pn15.b, xzr, xzr, vlx4
// CHECK-ENCODING: [0xf7,0x63,0x3f,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 253f63f7 <unknown>
