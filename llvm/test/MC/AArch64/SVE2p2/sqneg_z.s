// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2p2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2p2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2p2 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p2 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sve2p2 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

sqneg   z0.b, p0/z, z0.b  // 01000100-00001011-10100000-00000000
// CHECK-INST: sqneg   z0.b, p0/z, z0.b
// CHECK-ENCODING: [0x00,0xa0,0x0b,0x44]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 440ba000 <unknown>

sqneg   z23.h, p3/z, z13.h  // 01000100-01001011-10101101-10110111
// CHECK-INST: sqneg   z23.h, p3/z, z13.h
// CHECK-ENCODING: [0xb7,0xad,0x4b,0x44]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 444badb7 <unknown>

sqneg   z21.s, p5/z, z10.s  // 01000100-10001011-10110101-01010101
// CHECK-INST: sqneg   z21.s, p5/z, z10.s
// CHECK-ENCODING: [0x55,0xb5,0x8b,0x44]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 448bb555 <unknown>

sqneg   z31.d, p7/z, z31.d  // 01000100-11001011-10111111-11111111
// CHECK-INST: sqneg   z31.d, p7/z, z31.d
// CHECK-ENCODING: [0xff,0xbf,0xcb,0x44]
// CHECK-ERROR: instruction requires: sme2p2 or sve2p2
// CHECK-UNKNOWN: 44cbbfff <unknown>