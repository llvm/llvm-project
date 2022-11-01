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

whilele pn8.h, x0, x0, vlx2  // 00100101-01100000-01000100-00011000
// CHECK-INST: whilele pn8.h, x0, x0, vlx2
// CHECK-ENCODING: [0x18,0x44,0x60,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25604418 <unknown>

whilele pn13.h, x10, x21, vlx2  // 00100101-01110101-01000101-01011101
// CHECK-INST: whilele pn13.h, x10, x21, vlx2
// CHECK-ENCODING: [0x5d,0x45,0x75,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 2575455d <unknown>

whilele pn15.h, x13, x8, vlx4  // 00100101-01101000-01100101-10111111
// CHECK-INST: whilele pn15.h, x13, x8, vlx4
// CHECK-ENCODING: [0xbf,0x65,0x68,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 256865bf <unknown>

whilele pn15.h, xzr, xzr, vlx4  // 00100101-01111111-01100111-11111111
// CHECK-INST: whilele pn15.h, xzr, xzr, vlx4
// CHECK-ENCODING: [0xff,0x67,0x7f,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 257f67ff <unknown>

whilele pn8.s, x0, x0, vlx2  // 00100101-10100000-01000100-00011000
// CHECK-INST: whilele pn8.s, x0, x0, vlx2
// CHECK-ENCODING: [0x18,0x44,0xa0,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25a04418 <unknown>

whilele pn13.s, x10, x21, vlx2  // 00100101-10110101-01000101-01011101
// CHECK-INST: whilele pn13.s, x10, x21, vlx2
// CHECK-ENCODING: [0x5d,0x45,0xb5,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25b5455d <unknown>

whilele pn15.s, x13, x8, vlx4  // 00100101-10101000-01100101-10111111
// CHECK-INST: whilele pn15.s, x13, x8, vlx4
// CHECK-ENCODING: [0xbf,0x65,0xa8,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25a865bf <unknown>

whilele pn15.s, xzr, xzr, vlx4  // 00100101-10111111-01100111-11111111
// CHECK-INST: whilele pn15.s, xzr, xzr, vlx4
// CHECK-ENCODING: [0xff,0x67,0xbf,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25bf67ff <unknown>

whilele pn8.d, x0, x0, vlx2  // 00100101-11100000-01000100-00011000
// CHECK-INST: whilele pn8.d, x0, x0, vlx2
// CHECK-ENCODING: [0x18,0x44,0xe0,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25e04418 <unknown>

whilele pn13.d, x10, x21, vlx2  // 00100101-11110101-01000101-01011101
// CHECK-INST: whilele pn13.d, x10, x21, vlx2
// CHECK-ENCODING: [0x5d,0x45,0xf5,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25f5455d <unknown>

whilele pn15.d, x13, x8, vlx4  // 00100101-11101000-01100101-10111111
// CHECK-INST: whilele pn15.d, x13, x8, vlx4
// CHECK-ENCODING: [0xbf,0x65,0xe8,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25e865bf <unknown>

whilele pn15.d, xzr, xzr, vlx4  // 00100101-11111111-01100111-11111111
// CHECK-INST: whilele pn15.d, xzr, xzr, vlx4
// CHECK-ENCODING: [0xff,0x67,0xff,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25ff67ff <unknown>

whilele pn8.b, x0, x0, vlx2  // 00100101-00100000-01000100-00011000
// CHECK-INST: whilele pn8.b, x0, x0, vlx2
// CHECK-ENCODING: [0x18,0x44,0x20,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25204418 <unknown>

whilele pn13.b, x10, x21, vlx2  // 00100101-00110101-01000101-01011101
// CHECK-INST: whilele pn13.b, x10, x21, vlx2
// CHECK-ENCODING: [0x5d,0x45,0x35,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 2535455d <unknown>

whilele pn15.b, x13, x8, vlx4  // 00100101-00101000-01100101-10111111
// CHECK-INST: whilele pn15.b, x13, x8, vlx4
// CHECK-ENCODING: [0xbf,0x65,0x28,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 252865bf <unknown>

whilele pn15.b, xzr, xzr, vlx4  // 00100101-00111111-01100111-11111111
// CHECK-INST: whilele pn15.b, xzr, xzr, vlx4
// CHECK-ENCODING: [0xff,0x67,0x3f,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 253f67ff <unknown>
