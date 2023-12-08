// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p1 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p1 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2p1 < %s \
// RUN:        | llvm-objdump -d --no-print-imm-hex --mattr=+sme2p1 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2p1 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2p1,-sve2p1 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p1 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2p1 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST


tbxq    z0.h, z0.h, z0.h  // 00000101-01100000-00110100-00000000
// CHECK-INST: tbxq    z0.h, z0.h, z0.h
// CHECK-ENCODING: [0x00,0x34,0x60,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 05603400 <unknown>

tbxq    z21.h, z10.h, z21.h  // 00000101-01110101-00110101-01010101
// CHECK-INST: tbxq    z21.h, z10.h, z21.h
// CHECK-ENCODING: [0x55,0x35,0x75,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 05753555 <unknown>

tbxq    z23.h, z13.h, z8.h  // 00000101-01101000-00110101-10110111
// CHECK-INST: tbxq    z23.h, z13.h, z8.h
// CHECK-ENCODING: [0xb7,0x35,0x68,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 056835b7 <unknown>

tbxq    z31.h, z31.h, z31.h  // 00000101-01111111-00110111-11111111
// CHECK-INST: tbxq    z31.h, z31.h, z31.h
// CHECK-ENCODING: [0xff,0x37,0x7f,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 057f37ff <unknown>


tbxq    z0.s, z0.s, z0.s  // 00000101-10100000-00110100-00000000
// CHECK-INST: tbxq    z0.s, z0.s, z0.s
// CHECK-ENCODING: [0x00,0x34,0xa0,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 05a03400 <unknown>

tbxq    z21.s, z10.s, z21.s  // 00000101-10110101-00110101-01010101
// CHECK-INST: tbxq    z21.s, z10.s, z21.s
// CHECK-ENCODING: [0x55,0x35,0xb5,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 05b53555 <unknown>

tbxq    z23.s, z13.s, z8.s  // 00000101-10101000-00110101-10110111
// CHECK-INST: tbxq    z23.s, z13.s, z8.s
// CHECK-ENCODING: [0xb7,0x35,0xa8,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 05a835b7 <unknown>

tbxq    z31.s, z31.s, z31.s  // 00000101-10111111-00110111-11111111
// CHECK-INST: tbxq    z31.s, z31.s, z31.s
// CHECK-ENCODING: [0xff,0x37,0xbf,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 05bf37ff <unknown>


tbxq    z0.d, z0.d, z0.d  // 00000101-11100000-00110100-00000000
// CHECK-INST: tbxq    z0.d, z0.d, z0.d
// CHECK-ENCODING: [0x00,0x34,0xe0,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 05e03400 <unknown>

tbxq    z21.d, z10.d, z21.d  // 00000101-11110101-00110101-01010101
// CHECK-INST: tbxq    z21.d, z10.d, z21.d
// CHECK-ENCODING: [0x55,0x35,0xf5,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 05f53555 <unknown>

tbxq    z23.d, z13.d, z8.d  // 00000101-11101000-00110101-10110111
// CHECK-INST: tbxq    z23.d, z13.d, z8.d
// CHECK-ENCODING: [0xb7,0x35,0xe8,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 05e835b7 <unknown>

tbxq    z31.d, z31.d, z31.d  // 00000101-11111111-00110111-11111111
// CHECK-INST: tbxq    z31.d, z31.d, z31.d
// CHECK-ENCODING: [0xff,0x37,0xff,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 05ff37ff <unknown>


tbxq    z0.b, z0.b, z0.b  // 00000101-00100000-00110100-00000000
// CHECK-INST: tbxq    z0.b, z0.b, z0.b
// CHECK-ENCODING: [0x00,0x34,0x20,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 05203400 <unknown>

tbxq    z21.b, z10.b, z21.b  // 00000101-00110101-00110101-01010101
// CHECK-INST: tbxq    z21.b, z10.b, z21.b
// CHECK-ENCODING: [0x55,0x35,0x35,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 05353555 <unknown>

tbxq    z23.b, z13.b, z8.b  // 00000101-00101000-00110101-10110111
// CHECK-INST: tbxq    z23.b, z13.b, z8.b
// CHECK-ENCODING: [0xb7,0x35,0x28,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 052835b7 <unknown>

tbxq    z31.b, z31.b, z31.b  // 00000101-00111111-00110111-11111111
// CHECK-INST: tbxq    z31.b, z31.b, z31.b
// CHECK-ENCODING: [0xff,0x37,0x3f,0x05]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 053f37ff <unknown>

