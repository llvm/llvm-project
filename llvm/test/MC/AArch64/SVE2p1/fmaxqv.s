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


fmaxqv  v0.2d, p0, z0.d  // 01100100-11010110-10100000-00000000
// CHECK-INST: fmaxqv  v0.2d, p0, z0.d
// CHECK-ENCODING: [0x00,0xa0,0xd6,0x64]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 64d6a000 <unknown>

fmaxqv  v21.2d, p5, z10.d  // 01100100-11010110-10110101-01010101
// CHECK-INST: fmaxqv  v21.2d, p5, z10.d
// CHECK-ENCODING: [0x55,0xb5,0xd6,0x64]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 64d6b555 <unknown>

fmaxqv  v23.2d, p3, z13.d  // 01100100-11010110-10101101-10110111
// CHECK-INST: fmaxqv  v23.2d, p3, z13.d
// CHECK-ENCODING: [0xb7,0xad,0xd6,0x64]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 64d6adb7 <unknown>

fmaxqv  v31.2d, p7, z31.d  // 01100100-11010110-10111111-11111111
// CHECK-INST: fmaxqv  v31.2d, p7, z31.d
// CHECK-ENCODING: [0xff,0xbf,0xd6,0x64]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 64d6bfff <unknown>


fmaxqv  v0.8h, p0, z0.h  // 01100100-01010110-10100000-00000000
// CHECK-INST: fmaxqv  v0.8h, p0, z0.h
// CHECK-ENCODING: [0x00,0xa0,0x56,0x64]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 6456a000 <unknown>

fmaxqv  v21.8h, p5, z10.h  // 01100100-01010110-10110101-01010101
// CHECK-INST: fmaxqv  v21.8h, p5, z10.h
// CHECK-ENCODING: [0x55,0xb5,0x56,0x64]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 6456b555 <unknown>

fmaxqv  v23.8h, p3, z13.h  // 01100100-01010110-10101101-10110111
// CHECK-INST: fmaxqv  v23.8h, p3, z13.h
// CHECK-ENCODING: [0xb7,0xad,0x56,0x64]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 6456adb7 <unknown>

fmaxqv  v31.8h, p7, z31.h  // 01100100-01010110-10111111-11111111
// CHECK-INST: fmaxqv  v31.8h, p7, z31.h
// CHECK-ENCODING: [0xff,0xbf,0x56,0x64]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 6456bfff <unknown>


fmaxqv  v0.4s, p0, z0.s  // 01100100-10010110-10100000-00000000
// CHECK-INST: fmaxqv  v0.4s, p0, z0.s
// CHECK-ENCODING: [0x00,0xa0,0x96,0x64]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 6496a000 <unknown>

fmaxqv  v21.4s, p5, z10.s  // 01100100-10010110-10110101-01010101
// CHECK-INST: fmaxqv  v21.4s, p5, z10.s
// CHECK-ENCODING: [0x55,0xb5,0x96,0x64]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 6496b555 <unknown>

fmaxqv  v23.4s, p3, z13.s  // 01100100-10010110-10101101-10110111
// CHECK-INST: fmaxqv  v23.4s, p3, z13.s
// CHECK-ENCODING: [0xb7,0xad,0x96,0x64]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 6496adb7 <unknown>

fmaxqv  v31.4s, p7, z31.s  // 01100100-10010110-10111111-11111111
// CHECK-INST: fmaxqv  v31.4s, p7, z31.s
// CHECK-ENCODING: [0xff,0xbf,0x96,0x64]
// CHECK-ERROR: instruction requires: sme2p1 or sve2p1
// CHECK-UNKNOWN: 6496bfff <unknown>

