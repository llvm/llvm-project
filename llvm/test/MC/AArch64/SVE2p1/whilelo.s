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

whilelo pn8.h, x0, x0, vlx2  // 00100101-01100000-01001100-00010000
// CHECK-INST: whilelo pn8.h, x0, x0, vlx2
// CHECK-ENCODING: [0x10,0x4c,0x60,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25604c10 <unknown>

whilelo pn13.h, x10, x21, vlx2  // 00100101-01110101-01001101-01010101
// CHECK-INST: whilelo pn13.h, x10, x21, vlx2
// CHECK-ENCODING: [0x55,0x4d,0x75,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25754d55 <unknown>

whilelo pn15.h, x13, x8, vlx4  // 00100101-01101000-01101101-10110111
// CHECK-INST: whilelo pn15.h, x13, x8, vlx4
// CHECK-ENCODING: [0xb7,0x6d,0x68,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25686db7 <unknown>

whilelo pn15.h, xzr, xzr, vlx4  // 00100101-01111111-01101111-11110111
// CHECK-INST: whilelo pn15.h, xzr, xzr, vlx4
// CHECK-ENCODING: [0xf7,0x6f,0x7f,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 257f6ff7 <unknown>

whilelo pn8.s, x0, x0, vlx2  // 00100101-10100000-01001100-00010000
// CHECK-INST: whilelo pn8.s, x0, x0, vlx2
// CHECK-ENCODING: [0x10,0x4c,0xa0,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25a04c10 <unknown>

whilelo pn13.s, x10, x21, vlx2  // 00100101-10110101-01001101-01010101
// CHECK-INST: whilelo pn13.s, x10, x21, vlx2
// CHECK-ENCODING: [0x55,0x4d,0xb5,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25b54d55 <unknown>

whilelo pn15.s, x13, x8, vlx4  // 00100101-10101000-01101101-10110111
// CHECK-INST: whilelo pn15.s, x13, x8, vlx4
// CHECK-ENCODING: [0xb7,0x6d,0xa8,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25a86db7 <unknown>

whilelo pn15.s, xzr, xzr, vlx4  // 00100101-10111111-01101111-11110111
// CHECK-INST: whilelo pn15.s, xzr, xzr, vlx4
// CHECK-ENCODING: [0xf7,0x6f,0xbf,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25bf6ff7 <unknown>

whilelo pn8.d, x0, x0, vlx2  // 00100101-11100000-01001100-00010000
// CHECK-INST: whilelo pn8.d, x0, x0, vlx2
// CHECK-ENCODING: [0x10,0x4c,0xe0,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25e04c10 <unknown>

whilelo pn13.d, x10, x21, vlx2  // 00100101-11110101-01001101-01010101
// CHECK-INST: whilelo pn13.d, x10, x21, vlx2
// CHECK-ENCODING: [0x55,0x4d,0xf5,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25f54d55 <unknown>

whilelo pn15.d, x13, x8, vlx4  // 00100101-11101000-01101101-10110111
// CHECK-INST: whilelo pn15.d, x13, x8, vlx4
// CHECK-ENCODING: [0xb7,0x6d,0xe8,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25e86db7 <unknown>

whilelo pn15.d, xzr, xzr, vlx4  // 00100101-11111111-01101111-11110111
// CHECK-INST: whilelo pn15.d, xzr, xzr, vlx4
// CHECK-ENCODING: [0xf7,0x6f,0xff,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25ff6ff7 <unknown>

whilelo pn8.b, x0, x0, vlx2  // 00100101-00100000-01001100-00010000
// CHECK-INST: whilelo pn8.b, x0, x0, vlx2
// CHECK-ENCODING: [0x10,0x4c,0x20,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25204c10 <unknown>

whilelo pn13.b, x10, x21, vlx2  // 00100101-00110101-01001101-01010101
// CHECK-INST: whilelo pn13.b, x10, x21, vlx2
// CHECK-ENCODING: [0x55,0x4d,0x35,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25354d55 <unknown>

whilelo pn15.b, x13, x8, vlx4  // 00100101-00101000-01101101-10110111
// CHECK-INST: whilelo pn15.b, x13, x8, vlx4
// CHECK-ENCODING: [0xb7,0x6d,0x28,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25286db7 <unknown>

whilelo pn15.b, xzr, xzr, vlx4  // 00100101-00111111-01101111-11110111
// CHECK-INST: whilelo pn15.b, xzr, xzr, vlx4
// CHECK-ENCODING: [0xf7,0x6f,0x3f,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 253f6ff7 <unknown>
