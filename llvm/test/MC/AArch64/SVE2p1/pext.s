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

pext    p0.h, pn8[0]  // 00100101-01100000-01110000-00010000
// CHECK-INST: pext    p0.h, pn8[0]
// CHECK-ENCODING: [0x10,0x70,0x60,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25607010 <unknown>

pext    p5.h, pn10[1]  // 00100101-01100000-01110001-01010101
// CHECK-INST: pext    p5.h, pn10[1]
// CHECK-ENCODING: [0x55,0x71,0x60,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25607155 <unknown>

pext    p7.h, pn13[1]  // 00100101-01100000-01110001-10110111
// CHECK-INST: pext    p7.h, pn13[1]
// CHECK-ENCODING: [0xb7,0x71,0x60,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 256071b7 <unknown>

pext    p15.h, pn15[3]  // 00100101-01100000-01110011-11111111
// CHECK-INST: pext    p15.h, pn15[3]
// CHECK-ENCODING: [0xff,0x73,0x60,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 256073ff <unknown>

pext    p0.s, pn8[0]  // 00100101-10100000-01110000-00010000
// CHECK-INST: pext    p0.s, pn8[0]
// CHECK-ENCODING: [0x10,0x70,0xa0,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25a07010 <unknown>

pext    p5.s, pn10[1]  // 00100101-10100000-01110001-01010101
// CHECK-INST: pext    p5.s, pn10[1]
// CHECK-ENCODING: [0x55,0x71,0xa0,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25a07155 <unknown>

pext    p7.s, pn13[1]  // 00100101-10100000-01110001-10110111
// CHECK-INST: pext    p7.s, pn13[1]
// CHECK-ENCODING: [0xb7,0x71,0xa0,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25a071b7 <unknown>

pext    p15.s, pn15[3]  // 00100101-10100000-01110011-11111111
// CHECK-INST: pext    p15.s, pn15[3]
// CHECK-ENCODING: [0xff,0x73,0xa0,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25a073ff <unknown>

pext    p0.d, pn8[0]  // 00100101-11100000-01110000-00010000
// CHECK-INST: pext    p0.d, pn8[0]
// CHECK-ENCODING: [0x10,0x70,0xe0,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25e07010 <unknown>

pext    p5.d, pn10[1]  // 00100101-11100000-01110001-01010101
// CHECK-INST: pext    p5.d, pn10[1]
// CHECK-ENCODING: [0x55,0x71,0xe0,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25e07155 <unknown>

pext    p7.d, pn13[1]  // 00100101-11100000-01110001-10110111
// CHECK-INST: pext    p7.d, pn13[1]
// CHECK-ENCODING: [0xb7,0x71,0xe0,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25e071b7 <unknown>

pext    p15.d, pn15[3]  // 00100101-11100000-01110011-11111111
// CHECK-INST: pext    p15.d, pn15[3]
// CHECK-ENCODING: [0xff,0x73,0xe0,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25e073ff <unknown>

pext    p0.b, pn8[0]  // 00100101-00100000-01110000-00010000
// CHECK-INST: pext    p0.b, pn8[0]
// CHECK-ENCODING: [0x10,0x70,0x20,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25207010 <unknown>

pext    p5.b, pn10[1]  // 00100101-00100000-01110001-01010101
// CHECK-INST: pext    p5.b, pn10[1]
// CHECK-ENCODING: [0x55,0x71,0x20,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 25207155 <unknown>

pext    p7.b, pn13[1]  // 00100101-00100000-01110001-10110111
// CHECK-INST: pext    p7.b, pn13[1]
// CHECK-ENCODING: [0xb7,0x71,0x20,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 252071b7 <unknown>

pext    p15.b, pn15[3]  // 00100101-00100000-01110011-11111111
// CHECK-INST: pext    p15.b, pn15[3]
// CHECK-ENCODING: [0xff,0x73,0x20,0x25]
// CHECK-ERROR: instruction requires: sme2 or sve2p1
// CHECK-UNKNOWN: 252073ff <unknown>
