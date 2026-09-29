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


fminnmqv v0.2d, p0, z0.d  // 01100100-11010101-10100000-00000000
// CHECK-INST: fminnmqv v0.2d, p0, z0.d
// CHECK-ENCODING: [0x00,0xa0,0xd5,0x64]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 64d5a000 <unknown>

fminnmqv v21.2d, p5, z10.d  // 01100100-11010101-10110101-01010101
// CHECK-INST: fminnmqv v21.2d, p5, z10.d
// CHECK-ENCODING: [0x55,0xb5,0xd5,0x64]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 64d5b555 <unknown>

fminnmqv v23.2d, p3, z13.d  // 01100100-11010101-10101101-10110111
// CHECK-INST: fminnmqv v23.2d, p3, z13.d
// CHECK-ENCODING: [0xb7,0xad,0xd5,0x64]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 64d5adb7 <unknown>

fminnmqv v31.2d, p7, z31.d  // 01100100-11010101-10111111-11111111
// CHECK-INST: fminnmqv v31.2d, p7, z31.d
// CHECK-ENCODING: [0xff,0xbf,0xd5,0x64]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 64d5bfff <unknown>


fminnmqv v0.8h, p0, z0.h  // 01100100-01010101-10100000-00000000
// CHECK-INST: fminnmqv v0.8h, p0, z0.h
// CHECK-ENCODING: [0x00,0xa0,0x55,0x64]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 6455a000 <unknown>

fminnmqv v21.8h, p5, z10.h  // 01100100-01010101-10110101-01010101
// CHECK-INST: fminnmqv v21.8h, p5, z10.h
// CHECK-ENCODING: [0x55,0xb5,0x55,0x64]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 6455b555 <unknown>

fminnmqv v23.8h, p3, z13.h  // 01100100-01010101-10101101-10110111
// CHECK-INST: fminnmqv v23.8h, p3, z13.h
// CHECK-ENCODING: [0xb7,0xad,0x55,0x64]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 6455adb7 <unknown>

fminnmqv v31.8h, p7, z31.h  // 01100100-01010101-10111111-11111111
// CHECK-INST: fminnmqv v31.8h, p7, z31.h
// CHECK-ENCODING: [0xff,0xbf,0x55,0x64]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 6455bfff <unknown>


fminnmqv v0.4s, p0, z0.s  // 01100100-10010101-10100000-00000000
// CHECK-INST: fminnmqv v0.4s, p0, z0.s
// CHECK-ENCODING: [0x00,0xa0,0x95,0x64]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 6495a000 <unknown>

fminnmqv v21.4s, p5, z10.s  // 01100100-10010101-10110101-01010101
// CHECK-INST: fminnmqv v21.4s, p5, z10.s
// CHECK-ENCODING: [0x55,0xb5,0x95,0x64]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 6495b555 <unknown>

fminnmqv v23.4s, p3, z13.s  // 01100100-10010101-10101101-10110111
// CHECK-INST: fminnmqv v23.4s, p3, z13.s
// CHECK-ENCODING: [0xb7,0xad,0x95,0x64]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 6495adb7 <unknown>

fminnmqv v31.4s, p7, z31.s  // 01100100-10010101-10111111-11111111
// CHECK-INST: fminnmqv v31.4s, p7, z31.s
// CHECK-ENCODING: [0xff,0xbf,0x95,0x64]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 6495bfff <unknown>

