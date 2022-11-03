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

cntp    x0, pn0.h, vlx2  // 00100101-01100000-10000010-00000000
// CHECK-INST: cntp    x0, pn0.h, vlx2
// CHECK-ENCODING: [0x00,0x82,0x60,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25608200 <unknown>

cntp    x21, pn10.h, vlx4  // 00100101-01100000-10000111-01010101
// CHECK-INST: cntp    x21, pn10.h, vlx4
// CHECK-ENCODING: [0x55,0x87,0x60,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25608755 <unknown>

cntp    x23, pn13.h, vlx4  // 00100101-01100000-10000111-10110111
// CHECK-INST: cntp    x23, pn13.h, vlx4
// CHECK-ENCODING: [0xb7,0x87,0x60,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 256087b7 <unknown>

cntp    xzr, pn15.h, vlx4  // 00100101-01100000-10000111-11111111
// CHECK-INST: cntp    xzr, pn15.h, vlx4
// CHECK-ENCODING: [0xff,0x87,0x60,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 256087ff <unknown>

cntp    x0, pn0.s, vlx2  // 00100101-10100000-10000010-00000000
// CHECK-INST: cntp    x0, pn0.s, vlx2
// CHECK-ENCODING: [0x00,0x82,0xa0,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25a08200 <unknown>

cntp    x21, pn10.s, vlx4  // 00100101-10100000-10000111-01010101
// CHECK-INST: cntp    x21, pn10.s, vlx4
// CHECK-ENCODING: [0x55,0x87,0xa0,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25a08755 <unknown>

cntp    x23, pn13.s, vlx4  // 00100101-10100000-10000111-10110111
// CHECK-INST: cntp    x23, pn13.s, vlx4
// CHECK-ENCODING: [0xb7,0x87,0xa0,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25a087b7 <unknown>

cntp    xzr, pn15.s, vlx4  // 00100101-10100000-10000111-11111111
// CHECK-INST: cntp    xzr, pn15.s, vlx4
// CHECK-ENCODING: [0xff,0x87,0xa0,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25a087ff <unknown>

cntp    x0, pn0.d, vlx2  // 00100101-11100000-10000010-00000000
// CHECK-INST: cntp    x0, pn0.d, vlx2
// CHECK-ENCODING: [0x00,0x82,0xe0,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25e08200 <unknown>

cntp    x21, pn10.d, vlx4  // 00100101-11100000-10000111-01010101
// CHECK-INST: cntp    x21, pn10.d, vlx4
// CHECK-ENCODING: [0x55,0x87,0xe0,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25e08755 <unknown>

cntp    x23, pn13.d, vlx4  // 00100101-11100000-10000111-10110111
// CHECK-INST: cntp    x23, pn13.d, vlx4
// CHECK-ENCODING: [0xb7,0x87,0xe0,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25e087b7 <unknown>

cntp    xzr, pn15.d, vlx4  // 00100101-11100000-10000111-11111111
// CHECK-INST: cntp    xzr, pn15.d, vlx4
// CHECK-ENCODING: [0xff,0x87,0xe0,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25e087ff <unknown>

cntp    x0, pn0.b, vlx2  // 00100101-00100000-10000010-00000000
// CHECK-INST: cntp    x0, pn0.b, vlx2
// CHECK-ENCODING: [0x00,0x82,0x20,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25208200 <unknown>

cntp    x21, pn10.b, vlx4  // 00100101-00100000-10000111-01010101
// CHECK-INST: cntp    x21, pn10.b, vlx4
// CHECK-ENCODING: [0x55,0x87,0x20,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25208755 <unknown>

cntp    x23, pn13.b, vlx4  // 00100101-00100000-10000111-10110111
// CHECK-INST: cntp    x23, pn13.b, vlx4
// CHECK-ENCODING: [0xb7,0x87,0x20,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 252087b7 <unknown>

cntp    xzr, pn15.b, vlx4  // 00100101-00100000-10000111-11111111
// CHECK-INST: cntp    xzr, pn15.b, vlx4
// CHECK-ENCODING: [0xff,0x87,0x20,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 252087ff <unknown>
