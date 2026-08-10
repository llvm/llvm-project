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


faddqv  v0.2d, p0, z0.d  // 01100100-11010000-10100000-00000000
// CHECK-INST: faddqv  v0.2d, p0, z0.d
// CHECK-ENCODING: [0x00,0xa0,0xd0,0x64]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 64d0a000 <unknown>

faddqv  v21.2d, p5, z10.d  // 01100100-11010000-10110101-01010101
// CHECK-INST: faddqv  v21.2d, p5, z10.d
// CHECK-ENCODING: [0x55,0xb5,0xd0,0x64]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 64d0b555 <unknown>

faddqv  v23.2d, p3, z13.d  // 01100100-11010000-10101101-10110111
// CHECK-INST: faddqv  v23.2d, p3, z13.d
// CHECK-ENCODING: [0xb7,0xad,0xd0,0x64]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 64d0adb7 <unknown>

faddqv  v31.2d, p7, z31.d  // 01100100-11010000-10111111-11111111
// CHECK-INST: faddqv  v31.2d, p7, z31.d
// CHECK-ENCODING: [0xff,0xbf,0xd0,0x64]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 64d0bfff <unknown>


faddqv  v0.8h, p0, z0.h  // 01100100-01010000-10100000-00000000
// CHECK-INST: faddqv  v0.8h, p0, z0.h
// CHECK-ENCODING: [0x00,0xa0,0x50,0x64]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 6450a000 <unknown>

faddqv  v21.8h, p5, z10.h  // 01100100-01010000-10110101-01010101
// CHECK-INST: faddqv  v21.8h, p5, z10.h
// CHECK-ENCODING: [0x55,0xb5,0x50,0x64]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 6450b555 <unknown>

faddqv  v23.8h, p3, z13.h  // 01100100-01010000-10101101-10110111
// CHECK-INST: faddqv  v23.8h, p3, z13.h
// CHECK-ENCODING: [0xb7,0xad,0x50,0x64]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 6450adb7 <unknown>

faddqv  v31.8h, p7, z31.h  // 01100100-01010000-10111111-11111111
// CHECK-INST: faddqv  v31.8h, p7, z31.h
// CHECK-ENCODING: [0xff,0xbf,0x50,0x64]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 6450bfff <unknown>


faddqv  v0.4s, p0, z0.s  // 01100100-10010000-10100000-00000000
// CHECK-INST: faddqv  v0.4s, p0, z0.s
// CHECK-ENCODING: [0x00,0xa0,0x90,0x64]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 6490a000 <unknown>

faddqv  v21.4s, p5, z10.s  // 01100100-10010000-10110101-01010101
// CHECK-INST: faddqv  v21.4s, p5, z10.s
// CHECK-ENCODING: [0x55,0xb5,0x90,0x64]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 6490b555 <unknown>

faddqv  v23.4s, p3, z13.s  // 01100100-10010000-10101101-10110111
// CHECK-INST: faddqv  v23.4s, p3, z13.s
// CHECK-ENCODING: [0xb7,0xad,0x90,0x64]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 6490adb7 <unknown>

faddqv  v31.4s, p7, z31.s  // 01100100-10010000-10111111-11111111
// CHECK-INST: faddqv  v31.4s, p7, z31.s
// CHECK-ENCODING: [0xff,0xbf,0x90,0x64]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 6490bfff <unknown>

